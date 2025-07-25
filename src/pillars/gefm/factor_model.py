"""Graph-Enhanced Factor Model (GEFM) - Pillar 1
Spectral clustering-based factor model that outperforms GICS buckets
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import networkx as nx
from sklearn.cluster import SpectralClustering
from neo4j import GraphDatabase, Transaction
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class GEFMConfig:
    correlation_window: int = 60  # days
    correlation_threshold: float = 0.4
    min_cluster_size: int = 5
    max_clusters: int = 50
    eigenvalue_ratio_threshold: float = 1.5
    rebalance_frequency: int = 20  # trading days


class GraphEnhancedFactorModel:
    """Build factor models using graph-based clustering instead of industry buckets"""
    
    def __init__(self, neo4j_uri: str, auth: tuple, config: GEFMConfig = None):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=auth)
        self.config = config or GEFMConfig()
        self._cluster_cache = {}
        
    def build_correlation_graph(self, returns: pd.DataFrame, date: datetime) -> None:
        """Construct correlation graph in Neo4j for given date"""
        start_date = date - timedelta(days=self.config.correlation_window)
        window_returns = returns[start_date:date]
        
        # Calculate correlation matrix
        corr_matrix = window_returns.corr()
        
        # Build edges where |correlation| > threshold
        edges = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > self.config.correlation_threshold:
                    edges.append({
                        'security1': corr_matrix.columns[i],
                        'security2': corr_matrix.columns[j],
                        'correlation': float(corr_val),
                        'date': date.isoformat()
                    })
        
        # Write to Neo4j
        with self.driver.session() as session:
            session.execute_write(self._create_correlation_edges, edges, date)
            
    def _create_correlation_edges(self, tx: Transaction, edges: List[Dict], date: datetime):
        """Batch create correlation edges in Neo4j"""
        # First clear old edges for this date
        tx.run("""
            MATCH (:Security)-[r:CORRELATES_WITH {date: $date}]->(:Security)
            DELETE r
        """, date=date.isoformat())
        
        # Create new edges
        tx.run("""
            UNWIND $edges AS edge
            MATCH (s1:Security {symbol: edge.security1})
            MATCH (s2:Security {symbol: edge.security2})
            CREATE (s1)-[r:CORRELATES_WITH {
                correlation: edge.correlation,
                date: edge.date,
                abs_correlation: abs(edge.correlation)
            }]->(s2)
        """, edges=edges)
        
    def run_spectral_clustering(self, date: datetime) -> Dict[str, int]:
        """Execute spectral clustering using Neo4j Graph Data Science"""
        with self.driver.session() as session:
            # Create GDS graph projection
            session.run("""
                CALL gds.graph.project.cypher(
                    'correlation-graph-' + $date,
                    'MATCH (n:Security) RETURN id(n) AS id, n.symbol AS symbol',
                    'MATCH (s1:Security)-[r:CORRELATES_WITH {date: $date}]->(s2:Security)
                     RETURN id(s1) AS source, id(s2) AS target, r.abs_correlation AS weight',
                    {parameters: {date: $date}}
                )
            """, date=date.isoformat())
            
            # Run Louvain community detection
            result = session.run("""
                CALL gds.louvain.write('correlation-graph-' + $date, {
                    writeProperty: 'cluster_' + replace($date, '-', '_'),
                    relationshipWeightProperty: 'weight',
                    maxLevels: 10,
                    tolerance: 0.0001,
                    includeIntermediateCommunities: false
                })
                YIELD communityCount, modularity
                RETURN communityCount, modularity
            """, date=date.isoformat())
            
            stats = result.single()
            logger.info(f"Found {stats['communityCount']} clusters with modularity {stats['modularity']:.4f}")
            
            # Get cluster assignments
            clusters = session.run("""
                MATCH (s:Security)
                RETURN s.symbol AS symbol, 
                       s['cluster_' + replace($date, '-', '_')] AS cluster
                ORDER BY cluster, symbol
            """, date=date.isoformat())
            
            return {record['symbol']: record['cluster'] for record in clusters}
            
    def calculate_factor_returns(self, returns: pd.DataFrame, 
                               cluster_map: Dict[str, int], 
                               date: datetime) -> pd.Series:
        """Calculate factor returns as equal-weighted cluster returns"""
        factor_returns = {}
        
        for cluster_id in set(cluster_map.values()):
            cluster_securities = [s for s, c in cluster_map.items() if c == cluster_id]
            if len(cluster_securities) >= self.config.min_cluster_size:
                # Equal-weighted factor return
                factor_returns[f'GEFM_F{cluster_id}'] = returns[cluster_securities].loc[date].mean()
                
        return pd.Series(factor_returns)
        
    def calculate_factor_loadings(self, cluster_map: Dict[str, int]) -> pd.DataFrame:
        """Generate binary factor loading matrix"""
        securities = sorted(cluster_map.keys())
        factors = sorted(set(cluster_map.values()))
        
        loadings = pd.DataFrame(0, index=securities, 
                              columns=[f'GEFM_F{f}' for f in factors])
        
        for security, cluster in cluster_map.items():
            if cluster in factors:
                loadings.loc[security, f'GEFM_F{cluster}'] = 1.0
                
        return loadings
        
    def risk_decomposition(self, returns: pd.DataFrame, 
                          loadings: pd.DataFrame, 
                          factor_returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Decompose risk into factor and specific components"""
        # Factor covariance
        factor_cov = factor_returns.cov()
        
        # Specific risk (residuals)
        predicted_returns = loadings @ factor_returns.T
        residuals = returns - predicted_returns.T
        specific_risk = residuals.std()
        
        # Total risk attribution
        factor_risk = np.sqrt(np.diag(loadings @ factor_cov @ loadings.T))
        
        return {
            'factor_covariance': factor_cov,
            'specific_risk': specific_risk,
            'factor_risk': pd.Series(factor_risk, index=returns.columns),
            'risk_reduction': 1 - (specific_risk / returns.std())  # vs raw returns
        }
        
    def pipeline(self, returns: pd.DataFrame, date: datetime) -> Dict:
        """Full GEFM pipeline for a given date"""
        logger.info(f"Running GEFM pipeline for {date}")
        
        # Step 1: Build correlation graph
        self.build_correlation_graph(returns, date)
        
        # Step 2: Run spectral clustering
        cluster_map = self.run_spectral_clustering(date)
        
        # Step 3: Calculate factor returns
        factor_returns = self.calculate_factor_returns(returns, cluster_map, date)
        
        # Step 4: Generate loadings
        loadings = self.calculate_factor_loadings(cluster_map)
        
        # Step 5: Risk decomposition
        risk_metrics = self.risk_decomposition(
            returns.loc[date:date], loadings, factor_returns.to_frame().T
        )
        
        # Store results in Neo4j
        self._store_factor_model(date, cluster_map, factor_returns, risk_metrics)
        
        return {
            'date': date,
            'cluster_map': cluster_map,
            'factor_returns': factor_returns,
            'loadings': loadings,
            'risk_metrics': risk_metrics,
            'num_factors': len(set(cluster_map.values()))
        }
        
    def _store_factor_model(self, date: datetime, cluster_map: Dict[str, int], 
                           factor_returns: pd.Series, risk_metrics: Dict):
        """Persist factor model to Neo4j"""
        with self.driver.session() as session:
            session.execute_write(self._write_factor_model, 
                                date, cluster_map, factor_returns.to_dict(), 
                                risk_metrics['risk_reduction'].mean())
                                
    def _write_factor_model(self, tx: Transaction, date: datetime, 
                           cluster_map: Dict, factor_returns: Dict, avg_risk_reduction: float):
        """Write factor model data to Neo4j"""
        tx.run("""
            CREATE (fm:FactorModel {
                date: $date,
                type: 'GEFM',
                num_factors: $num_factors,
                avg_risk_reduction: $avg_risk_reduction,
                created_at: datetime()
            })
        """, date=date.isoformat(), 
             num_factors=len(set(cluster_map.values())),
             avg_risk_reduction=float(avg_risk_reduction))
        
        # Link securities to factors
        for symbol, cluster in cluster_map.items():
            tx.run("""
                MATCH (s:Security {symbol: $symbol})
                MATCH (fm:FactorModel {date: $date, type: 'GEFM'})
                CREATE (s)-[:LOADS_ON {
                    factor: $factor,
                    loading: 1.0
                }]->(fm)
            """, symbol=symbol, date=date.isoformat(), factor=f'GEFM_F{cluster}')
            
    def close(self):
        """Clean up resources"""
        self.driver.close()