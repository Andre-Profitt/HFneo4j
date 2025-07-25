"""GNN Alpha Engine (GraphTrader) - Pillar 2
Temporal Graph Neural Networks for short-horizon return prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import redis
from neo4j import GraphDatabase
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass 
class GNNConfig:
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    prediction_horizon: int = 1  # days
    feature_window: int = 20  # days of historical features
    edge_threshold: float = 0.3  # min correlation for edges


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for sequential graph snapshots"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        output = self.out_linear(context)
        
        return output, attn_weights


class DishFTGNN(nn.Module):
    """Distillation-based Future-aware GNN for stock prediction"""
    
    def __init__(self, config: GNNConfig, num_features: int, num_classes: int = 3):
        super().__init__()
        self.config = config
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(num_features, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        for i in range(config.num_layers):
            in_dim = config.hidden_dim if i > 0 else config.hidden_dim
            self.gat_layers.append(
                GATConv(in_dim, config.hidden_dim, heads=config.num_heads, 
                       dropout=config.dropout, concat=False)
            )
            
        # Temporal attention
        self.temporal_attn = TemporalAttention(config.hidden_dim, config.num_heads)
        
        # Memory module (simplified TGN-style)
        self.memory_dim = config.hidden_dim
        self.memory_updater = nn.GRUCell(config.hidden_dim, self.memory_dim)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, num_classes)
        )
        
        # Teacher network components (for distillation)
        self.teacher_encoder = nn.Sequential(
            nn.Linear(num_features + 5, config.hidden_dim),  # +5 for future features
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None,
                memory: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the network"""
        
        # Encode node features
        h = self.node_encoder(x)
        
        # Graph convolutions
        for i, gat in enumerate(self.gat_layers):
            h = gat(h, edge_index)
            if i < len(self.gat_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.config.dropout, training=self.training)
                
        # Update memory if provided
        if memory is not None:
            h = self.memory_updater(h, memory)
            
        # Global pooling for graph-level prediction
        if batch is not None:
            h = global_mean_pool(h, batch)
            
        # Classification
        logits = self.classifier(h)
        
        return {
            'logits': logits,
            'embeddings': h,
            'predictions': F.softmax(logits, dim=-1)
        }
        
    def teacher_forward(self, x: torch.Tensor, future_features: torch.Tensor) -> torch.Tensor:
        """Teacher network with access to future information"""
        combined = torch.cat([x, future_features], dim=-1)
        return self.teacher_encoder(combined)


class GNNAlphaEngine:
    """Main engine for GNN-based alpha generation"""
    
    def __init__(self, neo4j_uri: str, auth: tuple, redis_host: str = 'localhost',
                 config: GNNConfig = None):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=auth)
        self.redis_client = redis.Redis(host=redis_host, decode_responses=True)
        self.config = config or GNNConfig()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def extract_graph_snapshot(self, date: datetime) -> Data:
        """Extract graph snapshot from Neo4j for given date"""
        with self.driver.session() as session:
            # Get nodes with features
            nodes_result = session.run("""
                MATCH (s:Security)
                WHERE s.last_updated <= $date
                RETURN s.symbol AS symbol,
                       s.returns_1d AS r1d,
                       s.returns_5d AS r5d,
                       s.returns_20d AS r20d,
                       s.volume_zscore AS vol_z,
                       s.implied_vol AS iv,
                       s.market_cap_log AS mcap,
                       s.price_momentum AS momentum,
                       s.sector_encoded AS sector
                ORDER BY s.symbol
            """, date=date.isoformat())
            
            nodes = []
            node_features = []
            symbol_to_idx = {}
            
            for i, record in enumerate(nodes_result):
                symbol_to_idx[record['symbol']] = i
                nodes.append(record['symbol'])
                
                # Construct feature vector
                features = [
                    record['r1d'] or 0,
                    record['r5d'] or 0,
                    record['r20d'] or 0,
                    record['vol_z'] or 0,
                    record['iv'] or 0.3,
                    record['mcap'] or 0,
                    record['momentum'] or 0,
                    record['sector'] or 0
                ]
                node_features.append(features)
                
            # Get edges (correlations or other relationships)
            edges_result = session.run("""
                MATCH (s1:Security)-[r:CORRELATES_WITH {date: $date}]->(s2:Security)
                WHERE abs(r.correlation) > $threshold
                RETURN s1.symbol AS source, s2.symbol AS target,
                       r.correlation AS weight
            """, date=date.isoformat(), threshold=self.config.edge_threshold)
            
            edge_list = []
            edge_weights = []
            
            for record in edges_result:
                if record['source'] in symbol_to_idx and record['target'] in symbol_to_idx:
                    src_idx = symbol_to_idx[record['source']]
                    tgt_idx = symbol_to_idx[record['target']]
                    edge_list.append([src_idx, tgt_idx])
                    edge_list.append([tgt_idx, src_idx])  # Undirected
                    edge_weights.append(abs(record['weight']))
                    edge_weights.append(abs(record['weight']))
                    
        # Convert to PyTorch geometric data
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                   symbols=nodes, date=date)
                   
    def prepare_training_data(self, start_date: datetime, end_date: datetime,
                            prediction_horizon: int = 1) -> List[Data]:
        """Prepare training data with labels"""
        data_list = []
        current_date = start_date
        
        while current_date <= end_date - timedelta(days=prediction_horizon):
            # Get graph snapshot
            graph_data = self.extract_graph_snapshot(current_date)
            
            # Get labels (future returns)
            with self.driver.session() as session:
                labels_result = session.run("""
                    MATCH (s:Security)
                    WHERE s.last_updated = $future_date
                    RETURN s.symbol AS symbol,
                           s.returns_1d AS future_return
                    ORDER BY s.symbol
                """, future_date=(current_date + timedelta(days=prediction_horizon)).isoformat())
                
                labels = []
                for record in labels_result:
                    if record['future_return'] is None:
                        label = 1  # flat
                    elif record['future_return'] > 0.005:  # 0.5% threshold
                        label = 2  # up
                    elif record['future_return'] < -0.005:
                        label = 0  # down
                    else:
                        label = 1  # flat
                    labels.append(label)
                    
            graph_data.y = torch.tensor(labels, dtype=torch.long)
            data_list.append(graph_data)
            
            current_date += timedelta(days=1)
            
        return data_list
        
    def train(self, train_data: List[Data], val_data: List[Data], epochs: int = 100):
        """Train the GNN model"""
        # Initialize model
        num_features = train_data[0].x.shape[1]
        self.model = DishFTGNN(self.config, num_features).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.config.batch_size)
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out['logits'], batch.y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = out['logits'].argmax(dim=1)
                train_correct += (pred == batch.y).sum().item()
                train_total += batch.y.size(0)
                
            # Validation
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                    pred = out['logits'].argmax(dim=1)
                    val_correct += (pred == batch.y).sum().item()
                    val_total += batch.y.size(0)
                    
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_gnn_model.pt')
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                          f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
                          
    def predict(self, date: datetime) -> Dict[str, Dict]:
        """Generate predictions for given date"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
            
        self.model.eval()
        graph_data = self.extract_graph_snapshot(date)
        graph_data = graph_data.to(self.device)
        
        with torch.no_grad():
            out = self.model(graph_data.x, graph_data.edge_index)
            
        predictions = {}
        probs = out['predictions'].cpu().numpy()
        
        for i, symbol in enumerate(graph_data.symbols):
            predictions[symbol] = {
                'p_down': float(probs[i, 0]),
                'p_flat': float(probs[i, 1]),
                'p_up': float(probs[i, 2]),
                'signal': int(probs[i].argmax()),
                'confidence': float(probs[i].max()),
                'embedding': out['embeddings'][i].cpu().numpy().tolist()
            }
            
        # Store predictions in Redis
        self._store_predictions(date, predictions)
        
        # Store in Neo4j
        self._store_predictions_neo4j(date, predictions)
        
        return predictions
        
    def _store_predictions(self, date: datetime, predictions: Dict[str, Dict]):
        """Store predictions in Redis for fast access"""
        key = f"gnn_predictions:{date.isoformat()}"
        self.redis_client.setex(key, 86400, json.dumps(predictions))  # 24h TTL
        
    def _store_predictions_neo4j(self, date: datetime, predictions: Dict[str, Dict]):
        """Store predictions in Neo4j"""
        with self.driver.session() as session:
            session.run("""
                UNWIND $predictions AS pred
                MATCH (s:Security {symbol: pred.symbol})
                CREATE (p:Prediction {
                    date: $date,
                    model: 'DishFTGNN',
                    p_up: pred.p_up,
                    p_flat: pred.p_flat,
                    p_down: pred.p_down,
                    signal: pred.signal,
                    confidence: pred.confidence,
                    created_at: datetime()
                })
                CREATE (s)-[:HAS_PREDICTION]->(p)
            """, date=date.isoformat(), 
                predictions=[{'symbol': k, **v} for k, v in predictions.items()])
                
    def close(self):
        """Clean up resources"""
        self.driver.close()
        self.redis_client.close()