"""Trade-Flow Anomaly Radar - Pillar 5
Real-time detection of wash trades, spoofing, and suspicious patterns
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from neo4j import GraphDatabase
from kafka import KafkaConsumer, KafkaProducer
import redis
import json
import logging
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


@dataclass
class AnomalyConfig:
    wash_trade_time_window: int = 120  # seconds
    wash_trade_price_threshold: float = 0.001  # 0.1% price difference
    spoof_cancel_ratio: float = 0.8  # 80% cancellation rate
    layer_depth_threshold: int = 5  # Order book layers
    motif_scan_interval: int = 60  # seconds
    alert_confidence_threshold: float = 0.7
    kafka_topic: str = 'trade-flow'
    redis_ttl: int = 3600  # 1 hour


class TradeAnomalyGAT(nn.Module):
    """Graph Attention Network for trade anomaly classification"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)  # Binary: normal/anomalous
        )
        
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        x = torch.relu(x)
        return self.classifier(x)


class TradeFlowAnomalyRadar:
    """Real-time trade anomaly detection system"""
    
    def __init__(self, neo4j_uri: str, auth: tuple, 
                 kafka_bootstrap: str = 'localhost:9092',
                 redis_host: str = 'localhost',
                 config: AnomalyConfig = None):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=auth)
        self.redis_client = redis.Redis(host=redis_host, decode_responses=True)
        self.config = config or AnomalyConfig()
        
        # Kafka setup
        self.consumer = KafkaConsumer(
            self.config.kafka_topic,
            bootstrap_servers=kafka_bootstrap,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Model
        self.model = TradeAnomalyGAT(input_dim=12)  # 12 trade features
        self.model.eval()
        
        # Pattern cache
        self.pattern_cache = defaultdict(list)
        self.alert_history = []
        
    def detect_wash_trades(self, time_window: datetime) -> List[Dict]:
        """Detect circular trading patterns (wash trades)"""
        with self.driver.session() as session:
            # Cypher query for wash trade pattern
            results = session.run("""
                // Find circular trading patterns
                MATCH path = (t1:Trade)-[:SAME_TRADER]->(trader:Trader)-[:SAME_TRADER]->(t2:Trade)
                WHERE t1.timestamp > $start_time 
                AND t2.timestamp > $start_time
                AND t1.side <> t2.side  // Buy followed by Sell
                AND abs(t1.timestamp - t2.timestamp) < $time_window
                AND abs(t1.price - t2.price) / t1.price < $price_threshold
                AND t1.symbol = t2.symbol
                WITH path, t1, t2, trader,
                     abs(t1.quantity - t2.quantity) / t1.quantity AS qty_diff
                WHERE qty_diff < 0.1  // Similar quantities
                
                // Check for intermediate trades
                OPTIONAL MATCH (trader)-[:EXECUTED]->(t3:Trade)
                WHERE t3.timestamp > t1.timestamp 
                AND t3.timestamp < t2.timestamp
                AND t3.symbol = t1.symbol
                
                WITH path, t1, t2, trader, count(t3) AS intermediate_trades
                WHERE intermediate_trades < 3  // Few or no intermediate trades
                
                RETURN trader.id AS trader_id,
                       t1.trade_id AS buy_trade,
                       t2.trade_id AS sell_trade,
                       t1.symbol AS symbol,
                       t1.quantity AS quantity,
                       t1.price AS buy_price,
                       t2.price AS sell_price,
                       t2.timestamp - t1.timestamp AS round_trip_time,
                       intermediate_trades
                ORDER BY round_trip_time
                LIMIT 100
            """, 
            start_time=(time_window - timedelta(seconds=self.config.wash_trade_time_window)).isoformat(),
            time_window=self.config.wash_trade_time_window,
            price_threshold=self.config.wash_trade_price_threshold)
            
            wash_trades = []
            for record in results:
                wash_trades.append({
                    'type': 'wash_trade',
                    'trader_id': record['trader_id'],
                    'symbol': record['symbol'],
                    'buy_trade': record['buy_trade'],
                    'sell_trade': record['sell_trade'],
                    'quantity': record['quantity'],
                    'price_impact': abs(record['sell_price'] - record['buy_price']) / record['buy_price'],
                    'round_trip_time': record['round_trip_time'],
                    'confidence': 0.9 if record['intermediate_trades'] == 0 else 0.7,
                    'detected_at': datetime.now().isoformat()
                })
                
        return wash_trades
        
    def detect_spoofing(self, time_window: datetime) -> List[Dict]:
        """Detect spoofing patterns (large orders placed and quickly cancelled)"""
        with self.driver.session() as session:
            results = session.run("""
                // Find traders with high cancellation rates
                MATCH (trader:Trader)-[:PLACED]->(o:Order)
                WHERE o.timestamp > $start_time
                WITH trader, 
                     count(CASE WHEN o.status = 'CANCELLED' THEN 1 END) AS cancelled,
                     count(o) AS total,
                     avg(o.quantity) AS avg_quantity,
                     avg(CASE WHEN o.status = 'CANCELLED' THEN o.quantity END) AS avg_cancelled_qty
                WHERE total > 10 AND cancelled * 1.0 / total > $cancel_ratio
                AND avg_cancelled_qty > avg_quantity * 1.5  // Cancelled orders are larger
                
                // Get pattern details
                MATCH (trader)-[:PLACED]->(o:Order)
                WHERE o.timestamp > $start_time AND o.status = 'CANCELLED'
                WITH trader, o, cancelled * 1.0 / total AS cancel_rate
                ORDER BY o.quantity DESC
                LIMIT 5
                
                RETURN trader.id AS trader_id,
                       collect({
                           order_id: o.order_id,
                           symbol: o.symbol,
                           side: o.side,
                           quantity: o.quantity,
                           price: o.price,
                           time_to_cancel: o.cancel_time - o.timestamp
                       }) AS suspicious_orders,
                       cancel_rate
            """,
            start_time=time_window.isoformat(),
            cancel_ratio=self.config.spoof_cancel_ratio)
            
            spoofing_patterns = []
            for record in results:
                spoofing_patterns.append({
                    'type': 'spoofing',
                    'trader_id': record['trader_id'],
                    'cancel_rate': record['cancel_rate'],
                    'suspicious_orders': record['suspicious_orders'],
                    'confidence': min(0.95, 0.5 + record['cancel_rate'] / 2),
                    'detected_at': datetime.now().isoformat()
                })
                
        return spoofing_patterns
        
    def detect_layering(self, symbol: str, time_window: datetime) -> List[Dict]:
        """Detect layering (multiple orders at different price levels)"""
        with self.driver.session() as session:
            results = session.run("""
                // Find traders with orders across multiple price levels
                MATCH (trader:Trader)-[:PLACED]->(o:Order {symbol: $symbol})
                WHERE o.timestamp > $start_time AND o.status = 'ACTIVE'
                WITH trader, o.side AS side, count(DISTINCT o.price) AS price_levels,
                     collect(o) AS orders
                WHERE price_levels >= $depth_threshold
                
                // Analyze the pattern
                UNWIND orders AS order
                WITH trader, side, price_levels,
                     avg(order.quantity) AS avg_qty,
                     stdev(order.quantity) AS qty_stdev,
                     max(order.price) - min(order.price) AS price_spread
                     
                RETURN trader.id AS trader_id,
                       side,
                       price_levels,
                       avg_qty,
                       qty_stdev / avg_qty AS qty_variation,
                       price_spread
            """,
            symbol=symbol,
            start_time=time_window.isoformat(),
            depth_threshold=self.config.layer_depth_threshold)
            
            layering_patterns = []
            for record in results:
                layering_patterns.append({
                    'type': 'layering',
                    'trader_id': record['trader_id'],
                    'symbol': symbol,
                    'side': record['side'],
                    'price_levels': record['price_levels'],
                    'avg_quantity': record['avg_qty'],
                    'quantity_variation': record['qty_variation'],
                    'price_spread': record['price_spread'],
                    'confidence': min(0.9, 0.5 + record['price_levels'] / 20),
                    'detected_at': datetime.now().isoformat()
                })
                
        return layering_patterns
        
    def run_gat_anomaly_detection(self, trades: List[Dict]) -> List[Dict]:
        """Run Graph Attention Network for anomaly detection"""
        if len(trades) < 10:
            return []
            
        # Build trade graph
        nodes = []
        edges = []
        node_map = {}
        
        # Create nodes for trades
        for i, trade in enumerate(trades):
            node_map[trade['trade_id']] = i
            
            # Feature vector
            features = [
                trade.get('price', 0),
                trade.get('quantity', 0),
                trade.get('timestamp', 0),
                1 if trade.get('side') == 'BUY' else 0,
                trade.get('trader_risk_score', 0.5),
                trade.get('symbol_volatility', 0.2),
                trade.get('market_impact', 0),
                trade.get('order_book_imbalance', 0),
                trade.get('trade_velocity', 0),
                trade.get('trader_concentration', 0),
                trade.get('time_since_last_trade', 0),
                trade.get('price_deviation', 0)
            ]
            nodes.append(features)
            
        # Create edges based on relationships
        for i, t1 in enumerate(trades):
            for j, t2 in enumerate(trades):
                if i != j:
                    # Same trader
                    if t1.get('trader_id') == t2.get('trader_id'):
                        edges.append([i, j])
                    # Same symbol within time window
                    elif (t1.get('symbol') == t2.get('symbol') and 
                          abs(t1.get('timestamp', 0) - t2.get('timestamp', 0)) < 300):
                        edges.append([i, j])
                        
        # Convert to PyTorch geometric data
        x = torch.tensor(nodes, dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Run model
        with torch.no_grad():
            logits = self.model(x, edge_index)
            probs = torch.softmax(logits, dim=1)
            anomaly_scores = probs[:, 1].numpy()
            
        # Generate alerts
        alerts = []
        for i, (trade, score) in enumerate(zip(trades, anomaly_scores)):
            if score > self.config.alert_confidence_threshold:
                alerts.append({
                    'type': 'gat_anomaly',
                    'trade_id': trade['trade_id'],
                    'trader_id': trade.get('trader_id'),
                    'symbol': trade.get('symbol'),
                    'anomaly_score': float(score),
                    'confidence': float(score),
                    'detected_at': datetime.now().isoformat()
                })
                
        return alerts
        
    def process_trade_stream(self):
        """Process real-time trade stream from Kafka"""
        trade_buffer = []
        last_scan = datetime.now()
        
        for message in self.consumer:
            trade = message.value
            trade_buffer.append(trade)
            
            # Periodic pattern scanning
            if (datetime.now() - last_scan).seconds > self.config.motif_scan_interval:
                # Run different detection algorithms
                current_time = datetime.now()
                
                # Wash trade detection
                wash_trades = self.detect_wash_trades(current_time)
                for alert in wash_trades:
                    self._send_alert(alert)
                    
                # Spoofing detection
                spoofing = self.detect_spoofing(current_time)
                for alert in spoofing:
                    self._send_alert(alert)
                    
                # GAT-based detection on recent trades
                if len(trade_buffer) > 100:
                    gat_alerts = self.run_gat_anomaly_detection(trade_buffer[-500:])
                    for alert in gat_alerts:
                        self._send_alert(alert)
                        
                # Clear old trades from buffer
                trade_buffer = trade_buffer[-1000:]
                last_scan = datetime.now()
                
    def _send_alert(self, alert: Dict):
        """Send alert to monitoring systems"""
        # Store in Redis
        alert_key = f"anomaly_alert:{alert['type']}:{datetime.now().timestamp()}"
        self.redis_client.setex(alert_key, self.config.redis_ttl, json.dumps(alert))
        
        # Send to Kafka alerts topic
        self.producer.send('anomaly-alerts', value=alert)
        
        # Store in Neo4j
        with self.driver.session() as session:
            session.run("""
                CREATE (a:AnomalyAlert {
                    type: $type,
                    trader_id: $trader_id,
                    symbol: $symbol,
                    confidence: $confidence,
                    details: $details,
                    detected_at: datetime($detected_at),
                    created_at: datetime()
                })
            """, 
            type=alert['type'],
            trader_id=alert.get('trader_id'),
            symbol=alert.get('symbol'),
            confidence=alert['confidence'],
            details=json.dumps(alert),
            detected_at=alert['detected_at'])
            
        # Log
        logger.warning(f"Anomaly detected: {alert['type']} - "
                      f"Trader: {alert.get('trader_id')} - "
                      f"Confidence: {alert['confidence']:.2%}")
                      
    def get_trader_risk_score(self, trader_id: str) -> float:
        """Calculate risk score for a trader based on alert history"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:AnomalyAlert {trader_id: $trader_id})
                WHERE a.detected_at > datetime() - duration('P7D')  // Last 7 days
                WITH count(a) AS alert_count,
                     avg(a.confidence) AS avg_confidence,
                     collect(DISTINCT a.type) AS alert_types
                     
                MATCH (t:Trader {id: $trader_id})-[:EXECUTED]->(trade:Trade)
                WHERE trade.timestamp > datetime() - duration('P7D')
                WITH alert_count, avg_confidence, alert_types,
                     count(trade) AS trade_count,
                     sum(trade.quantity * trade.price) AS volume
                     
                RETURN alert_count, avg_confidence, size(alert_types) AS unique_alerts,
                       trade_count, volume
            """, trader_id=trader_id)
            
            record = result.single()
            if not record:
                return 0.5  # Default medium risk
                
            # Risk score calculation
            alert_score = min(1.0, record['alert_count'] / 10)
            confidence_score = record['avg_confidence'] or 0
            diversity_score = min(1.0, record['unique_alerts'] / 3)
            
            # Weighted average
            risk_score = (alert_score * 0.4 + 
                         confidence_score * 0.4 + 
                         diversity_score * 0.2)
                         
            return risk_score
            
    def generate_surveillance_report(self, date: datetime) -> Dict:
        """Generate daily surveillance report"""
        with self.driver.session() as session:
            # Aggregate statistics
            stats = session.run("""
                MATCH (a:AnomalyAlert)
                WHERE date(a.detected_at) = date($date)
                WITH a.type AS alert_type, count(a) AS count,
                     avg(a.confidence) AS avg_confidence,
                     collect(DISTINCT a.trader_id) AS traders
                RETURN alert_type, count, avg_confidence, size(traders) AS unique_traders
                ORDER BY count DESC
            """, date=date.isoformat())
            
            alert_summary = []
            for record in stats:
                alert_summary.append({
                    'type': record['alert_type'],
                    'count': record['count'],
                    'avg_confidence': record['avg_confidence'],
                    'unique_traders': record['unique_traders']
                })
                
            # High risk traders
            high_risk = session.run("""
                MATCH (t:Trader)
                WHERE t.risk_score > 0.8
                RETURN t.id AS trader_id, t.risk_score AS score,
                       t.total_alerts AS alerts, t.last_alert_date AS last_alert
                ORDER BY score DESC
                LIMIT 10
            """)
            
            high_risk_traders = [dict(record) for record in high_risk]
            
        return {
            'date': date.isoformat(),
            'alert_summary': alert_summary,
            'total_alerts': sum(a['count'] for a in alert_summary),
            'high_risk_traders': high_risk_traders,
            'generated_at': datetime.now().isoformat()
        }
        
    def close(self):
        """Clean up resources"""
        self.consumer.close()
        self.producer.close()
        self.redis_client.close()
        self.driver.close()