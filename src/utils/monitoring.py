"""Monitoring and Dashboard System
Real-time monitoring for the hedge fund graph stack
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from neo4j import GraphDatabase
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
from flask import Flask
import redis
import json
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Prometheus metrics
pipeline_processed = Counter('hedge_fund_pipeline_processed_total', 
                           'Total records processed', ['pipeline', 'status'])
pipeline_duration = Histogram('hedge_fund_pipeline_duration_seconds',
                            'Pipeline processing duration', ['pipeline'])
active_strategies = Gauge('hedge_fund_active_strategies', 
                        'Number of active trading strategies')
portfolio_value = Gauge('hedge_fund_portfolio_value_usd', 
                       'Current portfolio value in USD')
risk_metrics = Gauge('hedge_fund_risk_metrics',
                    'Risk metrics', ['metric'])
graph_metrics = Gauge('hedge_fund_graph_metrics',
                     'Graph database metrics', ['metric'])
anomaly_alerts = Counter('hedge_fund_anomaly_alerts_total',
                       'Anomaly alerts triggered', ['type', 'severity'])


@dataclass
class MonitoringConfig:
    prometheus_port: int = 8000
    dash_port: int = 8050
    update_interval: int = 30  # seconds
    alert_thresholds: Dict = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'max_drawdown': -0.1,  # -10%
                'daily_var': -0.05,    # -5%
                'pipeline_error_rate': 0.05,  # 5%
                'latency_ms': 1000     # 1 second
            }


class SystemMonitor:
    """Core monitoring system for all components"""
    
    def __init__(self, neo4j_uri: str, auth: tuple, redis_host: str = 'localhost',
                 config: MonitoringConfig = None):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=auth)
        self.redis_client = redis.Redis(host=redis_host, decode_responses=True)
        self.config = config or MonitoringConfig()
        
        # Start Prometheus metrics server
        start_http_server(self.config.prometheus_port)
        logger.info(f"Prometheus metrics available at http://localhost:{self.config.prometheus_port}")
        
    def collect_graph_metrics(self) -> Dict:
        """Collect Neo4j database metrics"""
        with self.driver.session() as session:
            # Node counts
            node_counts = session.run("""
                MATCH (s:Security) WITH count(s) AS securities
                MATCH (e:Entity) WITH securities, count(e) AS entities
                MATCH (t:Trade) WITH securities, entities, count(t) AS trades
                MATCH (a:AnomalyAlert) WITH securities, entities, trades, count(a) AS alerts
                RETURN securities, entities, trades, alerts
            """).single()
            
            # Relationship counts
            rel_counts = session.run("""
                MATCH ()-[c:CORRELATES_WITH]->() WITH count(c) AS correlations
                MATCH ()-[e:EXPOSURE]->() WITH correlations, count(e) AS exposures
                MATCH ()-[p:HAS_PRICE]->() WITH correlations, exposures, count(p) AS prices
                RETURN correlations, exposures, prices
            """).single()
            
            # Database size
            db_stats = session.run("""
                CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Store file sizes')
                YIELD attributes
                RETURN attributes.TotalStoreSize.value AS total_size
            """).single()
            
        metrics = {
            'nodes': {
                'securities': node_counts['securities'] if node_counts else 0,
                'entities': node_counts['entities'] if node_counts else 0,
                'trades': node_counts['trades'] if node_counts else 0,
                'alerts': node_counts['alerts'] if node_counts else 0
            },
            'relationships': {
                'correlations': rel_counts['correlations'] if rel_counts else 0,
                'exposures': rel_counts['exposures'] if rel_counts else 0,
                'prices': rel_counts['prices'] if rel_counts else 0
            },
            'database_size_mb': db_stats['total_size'] / 1024 / 1024 if db_stats else 0
        }
        
        # Update Prometheus metrics
        for metric_name, value in metrics['nodes'].items():
            graph_metrics.labels(metric=f'nodes_{metric_name}').set(value)
        for metric_name, value in metrics['relationships'].items():
            graph_metrics.labels(metric=f'relationships_{metric_name}').set(value)
        graph_metrics.labels(metric='database_size_mb').set(metrics['database_size_mb'])
        
        return metrics
        
    def collect_strategy_metrics(self) -> Dict:
        """Collect trading strategy performance metrics"""
        metrics = {}
        
        # Get latest backtest results
        with self.driver.session() as session:
            results = session.run("""
                MATCH (b:BacktestResult)
                WHERE b.run_date > datetime() - duration('P1D')
                RETURN b.strategy AS strategy, b.metrics AS metrics
                ORDER BY b.run_date DESC
            """)
            
            for record in results:
                strategy = record['strategy']
                try:
                    strategy_metrics = json.loads(record['metrics'])
                    metrics[strategy] = {
                        'sharpe_ratio': strategy_metrics.get('sharpe_ratio', 0),
                        'total_return': strategy_metrics.get('total_return', 0),
                        'max_drawdown': strategy_metrics.get('max_drawdown', 0),
                        'volatility': strategy_metrics.get('volatility', 0)
                    }
                except:
                    logger.error(f"Failed to parse metrics for {strategy}")
                    
        # Get active positions
        active_count = self.redis_client.get('active_strategies_count')
        if active_count:
            active_strategies.set(int(active_count))
            
        return metrics
        
    def collect_risk_metrics(self) -> Dict:
        """Collect portfolio risk metrics"""
        metrics = {}
        
        # Get from Redis cache (updated by risk engine)
        risk_keys = ['portfolio_var', 'portfolio_cvar', 'leverage_ratio', 
                    'concentration_risk', 'correlation_risk']
        
        for key in risk_keys:
            value = self.redis_client.get(f'risk_metric:{key}')
            if value:
                metrics[key] = float(value)
                risk_metrics.labels(metric=key).set(float(value))
                
        # Check contagion metrics
        with self.driver.session() as session:
            contagion = session.run("""
                MATCH (e:Entity {is_sifi: true})
                WITH count(e) AS sifi_count, avg(e.systemic_score) AS avg_score
                RETURN sifi_count, avg_score
            """).single()
            
            if contagion:
                metrics['sifi_count'] = contagion['sifi_count']
                metrics['avg_systemic_score'] = contagion['avg_score'] or 0
                
        return metrics
        
    def collect_system_health(self) -> Dict:
        """Collect overall system health metrics"""
        health = {
            'neo4j': 'healthy',
            'redis': 'healthy',
            'pipelines': {},
            'alerts': []
        }
        
        # Check Neo4j
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
        except Exception as e:
            health['neo4j'] = 'unhealthy'
            health['alerts'].append(f"Neo4j connection failed: {e}")
            
        # Check Redis
        try:
            self.redis_client.ping()
        except Exception as e:
            health['redis'] = 'unhealthy'
            health['alerts'].append(f"Redis connection failed: {e}")
            
        # Check pipeline status
        pipeline_status = self.redis_client.hgetall('pipeline_status')
        for pipeline, status in pipeline_status.items():
            try:
                status_dict = json.loads(status)
                health['pipelines'][pipeline] = {
                    'status': status_dict.get('status', 'unknown'),
                    'last_run': status_dict.get('last_run'),
                    'error_rate': status_dict.get('error_rate', 0)
                }
                
                # Check error threshold
                if status_dict.get('error_rate', 0) > self.config.alert_thresholds['pipeline_error_rate']:
                    health['alerts'].append(
                        f"Pipeline {pipeline} error rate: {status_dict['error_rate']:.2%}"
                    )
            except:
                health['pipelines'][pipeline] = {'status': 'error'}
                
        return health
        
    def check_alerts(self) -> List[Dict]:
        """Check for system alerts based on thresholds"""
        alerts = []
        
        # Check risk metrics
        risk_metrics = self.collect_risk_metrics()
        
        # VaR breach
        if risk_metrics.get('portfolio_var', 0) < self.config.alert_thresholds['daily_var']:
            alerts.append({
                'type': 'risk',
                'severity': 'high',
                'message': f"Daily VaR breach: {risk_metrics['portfolio_var']:.2%}",
                'timestamp': datetime.now()
            })
            anomaly_alerts.labels(type='risk', severity='high').inc()
            
        # Check strategy performance
        strategy_metrics = self.collect_strategy_metrics()
        for strategy, metrics in strategy_metrics.items():
            if metrics['max_drawdown'] < self.config.alert_thresholds['max_drawdown']:
                alerts.append({
                    'type': 'performance',
                    'severity': 'medium',
                    'message': f"{strategy} max drawdown: {metrics['max_drawdown']:.2%}",
                    'timestamp': datetime.now()
                })
                anomaly_alerts.labels(type='performance', severity='medium').inc()
                
        # Check recent anomalies
        with self.driver.session() as session:
            recent_anomalies = session.run("""
                MATCH (a:AnomalyAlert)
                WHERE a.detected_at > datetime() - duration('PT1H')
                AND a.confidence > 0.8
                RETURN a.type AS type, count(a) AS count,
                       avg(a.confidence) AS avg_confidence
                HAVING count > 5
            """)
            
            for record in recent_anomalies:
                alerts.append({
                    'type': 'anomaly',
                    'severity': 'high',
                    'message': f"High frequency {record['type']} detected: {record['count']} alerts",
                    'timestamp': datetime.now()
                })
                anomaly_alerts.labels(type='anomaly', severity='high').inc()
                
        return alerts
        
    async def continuous_monitoring(self):
        """Run continuous monitoring loop"""
        while True:
            try:
                # Collect all metrics
                graph_metrics = self.collect_graph_metrics()
                strategy_metrics = self.collect_strategy_metrics()
                risk_metrics = self.collect_risk_metrics()
                system_health = self.collect_system_health()
                alerts = self.check_alerts()
                
                # Store snapshot in Redis
                snapshot = {
                    'timestamp': datetime.now().isoformat(),
                    'graph_metrics': graph_metrics,
                    'strategy_metrics': strategy_metrics,
                    'risk_metrics': risk_metrics,
                    'system_health': system_health,
                    'alerts': [a for a in alerts if isinstance(a, dict)]
                }
                
                self.redis_client.setex(
                    'monitoring_snapshot',
                    300,  # 5 minute TTL
                    json.dumps(snapshot, default=str)
                )
                
                # Log critical alerts
                for alert in alerts:
                    if isinstance(alert, dict) and alert.get('severity') == 'high':
                        logger.warning(f"ALERT: {alert['message']}")
                        
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                
            await asyncio.sleep(self.config.update_interval)


class DashboardApp:
    """Interactive dashboard using Dash"""
    
    def __init__(self, redis_host: str = 'localhost', port: int = 8050):
        self.redis_client = redis.Redis(host=redis_host, decode_responses=True)
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Create dashboard layout"""
        self.app.layout = html.Div([
            html.H1('Hedge Fund Graph Stack Dashboard', 
                   style={'textAlign': 'center', 'marginBottom': 30}),
                   
            # System health row
            html.Div([
                html.Div([
                    html.H3('System Health'),
                    html.Div(id='health-status', style={'fontSize': 24})
                ], className='four columns'),
                
                html.Div([
                    html.H3('Active Strategies'),
                    html.Div(id='active-strategies', style={'fontSize': 24})
                ], className='four columns'),
                
                html.Div([
                    html.H3('Portfolio Value'),
                    html.Div(id='portfolio-value', style={'fontSize': 24})
                ], className='four columns'),
            ], className='row', style={'marginBottom': 30}),
            
            # Charts row
            html.Div([
                html.Div([
                    dcc.Graph(id='strategy-performance')
                ], className='six columns'),
                
                html.Div([
                    dcc.Graph(id='risk-metrics')
                ], className='six columns'),
            ], className='row', style={'marginBottom': 30}),
            
            # Graph metrics row
            html.Div([
                html.Div([
                    dcc.Graph(id='graph-metrics')
                ], className='six columns'),
                
                html.Div([
                    dcc.Graph(id='anomaly-timeline')
                ], className='six columns'),
            ], className='row', style={'marginBottom': 30}),
            
            # Alerts table
            html.Div([
                html.H3('Recent Alerts'),
                html.Div(id='alerts-table')
            ], style={'marginBottom': 30}),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # 30 seconds
                n_intervals=0
            )
        ])
        
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('health-status', 'children'),
             Output('health-status', 'style'),
             Output('active-strategies', 'children'),
             Output('portfolio-value', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_status(n):
            """Update status indicators"""
            try:
                snapshot = self.redis_client.get('monitoring_snapshot')
                if snapshot:
                    data = json.loads(snapshot)
                    
                    # System health
                    health = data.get('system_health', {})
                    if health.get('neo4j') == 'healthy' and health.get('redis') == 'healthy':
                        health_text = '✅ Healthy'
                        health_style = {'fontSize': 24, 'color': 'green'}
                    else:
                        health_text = '⚠️ Degraded'
                        health_style = {'fontSize': 24, 'color': 'orange'}
                        
                    # Active strategies
                    strategies = len(data.get('strategy_metrics', {}))
                    
                    # Portfolio value (mock for demo)
                    portfolio_value = '$127.3M'
                    
                    return health_text, health_style, f"{strategies} Active", portfolio_value
                    
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                
            return '❌ Error', {'fontSize': 24, 'color': 'red'}, 'N/A', 'N/A'
            
        @self.app.callback(
            Output('strategy-performance', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_strategy_chart(n):
            """Update strategy performance chart"""
            try:
                snapshot = self.redis_client.get('monitoring_snapshot')
                if snapshot:
                    data = json.loads(snapshot)
                    strategies = data.get('strategy_metrics', {})
                    
                    if strategies:
                        df = pd.DataFrame(strategies).T
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=df.index,
                            y=df['sharpe_ratio'],
                            name='Sharpe Ratio',
                            marker_color='lightblue'
                        ))
                        
                        fig.update_layout(
                            title='Strategy Performance (Sharpe Ratio)',
                            xaxis_title='Strategy',
                            yaxis_title='Sharpe Ratio',
                            height=400
                        )
                        
                        return fig
                        
            except Exception as e:
                logger.error(f"Strategy chart error: {e}")
                
            return go.Figure()
            
        @self.app.callback(
            Output('risk-metrics', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_risk_chart(n):
            """Update risk metrics chart"""
            try:
                snapshot = self.redis_client.get('monitoring_snapshot')
                if snapshot:
                    data = json.loads(snapshot)
                    risk = data.get('risk_metrics', {})
                    
                    if risk:
                        fig = go.Figure()
                        
                        # Create gauge charts for key risk metrics
                        fig.add_trace(go.Indicator(
                            mode = "gauge+number+delta",
                            value = abs(risk.get('portfolio_var', 0)) * 100,
                            title = {'text': "Daily VaR (%)"},
                            delta = {'reference': 5},
                            gauge = {
                                'axis': {'range': [None, 10]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 5], 'color': "lightgray"},
                                    {'range': [5, 10], 'color': "gray"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 5
                                }
                            }
                        ))
                        
                        fig.update_layout(
                            title='Portfolio Risk Metrics',
                            height=400
                        )
                        
                        return fig
                        
            except Exception as e:
                logger.error(f"Risk chart error: {e}")
                
            return go.Figure()
            
        @self.app.callback(
            Output('graph-metrics', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_graph_chart(n):
            """Update graph database metrics"""
            try:
                snapshot = self.redis_client.get('monitoring_snapshot')
                if snapshot:
                    data = json.loads(snapshot)
                    graph = data.get('graph_metrics', {})
                    
                    if graph:
                        nodes = graph.get('nodes', {})
                        relationships = graph.get('relationships', {})
                        
                        fig = go.Figure()
                        
                        # Nodes bar chart
                        fig.add_trace(go.Bar(
                            x=list(nodes.keys()),
                            y=list(nodes.values()),
                            name='Nodes',
                            marker_color='lightgreen'
                        ))
                        
                        fig.update_layout(
                            title='Graph Database Statistics',
                            xaxis_title='Entity Type',
                            yaxis_title='Count',
                            height=400
                        )
                        
                        return fig
                        
            except Exception as e:
                logger.error(f"Graph chart error: {e}")
                
            return go.Figure()
            
        @self.app.callback(
            Output('alerts-table', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_alerts_table(n):
            """Update alerts table"""
            try:
                snapshot = self.redis_client.get('monitoring_snapshot')
                if snapshot:
                    data = json.loads(snapshot)
                    alerts = data.get('alerts', [])
                    
                    if alerts:
                        df = pd.DataFrame(alerts)
                        
                        # Color code by severity
                        def severity_style(severity):
                            if severity == 'high':
                                return {'color': 'red', 'fontWeight': 'bold'}
                            elif severity == 'medium':
                                return {'color': 'orange'}
                            else:
                                return {'color': 'black'}
                                
                        return dash_table.DataTable(
                            data=df.to_dict('records'),
                            columns=[{"name": i, "id": i} for i in df.columns],
                            style_data_conditional=[
                                {
                                    'if': {'row_index': i},
                                    'backgroundColor': 'rgba(255, 0, 0, 0.1)' 
                                        if df.iloc[i]['severity'] == 'high' else 'white'
                                }
                                for i in range(len(df))
                            ],
                            style_cell={'textAlign': 'left'},
                            page_size=10
                        )
                        
            except Exception as e:
                logger.error(f"Alerts table error: {e}")
                
            return html.Div("No recent alerts")
            
    def run(self, debug: bool = False):
        """Run the dashboard"""
        self.app.run_server(debug=debug, port=8050, host='0.0.0.0')


def create_grafana_dashboards():
    """Create Grafana dashboard JSON configs"""
    
    dashboard = {
        "dashboard": {
            "title": "Hedge Fund Graph Stack",
            "panels": [
                {
                    "title": "Strategy Performance",
                    "targets": [
                        {"expr": "hedge_fund_strategy_sharpe_ratio"}
                    ],
                    "type": "graph"
                },
                {
                    "title": "Portfolio Risk Metrics",
                    "targets": [
                        {"expr": "hedge_fund_risk_metrics"}
                    ],
                    "type": "gauge"
                },
                {
                    "title": "Graph Database Size",
                    "targets": [
                        {"expr": "hedge_fund_graph_metrics{metric='database_size_mb'}"}
                    ],
                    "type": "stat"
                },
                {
                    "title": "Anomaly Alerts",
                    "targets": [
                        {"expr": "rate(hedge_fund_anomaly_alerts_total[5m])"}
                    ],
                    "type": "graph"
                }
            ]
        }
    }
    
    return dashboard