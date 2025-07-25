"""Network Contagion & Stress Testing Lab - Pillar 3
DebtRank and systemic risk analysis for prime broker exposures
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from neo4j import GraphDatabase
import logging
from datetime import datetime
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

logger = logging.getLogger(__name__)


@dataclass
class ContagionConfig:
    convergence_threshold: float = 1e-6
    max_iterations: int = 100
    default_loss_given_default: float = 0.6
    stress_shock_size: float = 0.15
    centrality_measure: str = 'debtrank'  # or 'katz', 'eigenvector'


class DebtRankEngine:
    """Calculate DebtRank and other systemic risk measures"""
    
    def __init__(self, neo4j_uri: str, auth: tuple, config: ContagionConfig = None):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=auth)
        self.config = config or ContagionConfig()
        self._exposure_cache = {}
        
    def load_exposure_network(self, date: datetime) -> Tuple[np.ndarray, List[str], Dict]:
        """Load exposure matrix from Neo4j"""
        with self.driver.session() as session:
            # Get all nodes (funds, prime brokers, CCPs)
            nodes_result = session.run("""
                MATCH (n:Entity)
                WHERE n.active = true
                RETURN n.entity_id AS id, n.name AS name, n.type AS type,
                       n.total_assets AS assets
                ORDER BY id
            """)
            
            nodes = []
            node_map = {}
            for i, record in enumerate(nodes_result):
                nodes.append(record['id'])
                node_map[record['id']] = {
                    'index': i,
                    'name': record['name'],
                    'type': record['type'],
                    'assets': record['assets'] or 0
                }
                
            n = len(nodes)
            exposure_matrix = np.zeros((n, n))
            
            # Get exposures
            exposures = session.run("""
                MATCH (a:Entity)-[e:EXPOSURE {date: $date}]->(b:Entity)
                WHERE a.active = true AND b.active = true
                RETURN a.entity_id AS source, b.entity_id AS target,
                       e.amount AS amount, e.exposure_type AS type
            """, date=date.isoformat())
            
            for record in exposures:
                i = node_map[record['source']]['index']
                j = node_map[record['target']]['index']
                exposure_matrix[i, j] = record['amount']
                
        return exposure_matrix, nodes, node_map
        
    def calculate_debtrank(self, exposure_matrix: np.ndarray, 
                          initial_shock: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        DebtRank algorithm implementation
        h_i^(k+1) = min(1, h_i^(k) + sum_j E_ij * h_j^(k))
        """
        n = exposure_matrix.shape[0]
        h = initial_shock.copy()
        h_prev = np.zeros(n)
        
        # Normalize exposures by total assets (row-wise)
        row_sums = exposure_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        E_normalized = exposure_matrix / row_sums[:, np.newaxis]
        
        iterations = 0
        convergence_history = []
        
        while np.max(np.abs(h - h_prev)) > self.config.convergence_threshold:
            if iterations >= self.config.max_iterations:
                logger.warning(f"DebtRank did not converge after {iterations} iterations")
                break
                
            h_prev = h.copy()
            h_new = h_prev + E_normalized.T @ h_prev
            h = np.minimum(h_new, 1.0)  # Cap at 1 (total default)
            
            convergence_history.append(np.max(np.abs(h - h_prev)))
            iterations += 1
            
        # Calculate impact metrics
        total_impact = np.sum(h)
        systemic_risk = total_impact - np.sum(initial_shock)
        
        return h, {
            'iterations': iterations,
            'total_impact': total_impact,
            'systemic_risk': systemic_risk,
            'convergence_history': convergence_history,
            'amplification_factor': total_impact / np.sum(initial_shock) if np.sum(initial_shock) > 0 else 0
        }
        
    def calculate_katz_centrality(self, exposure_matrix: np.ndarray, 
                                 alpha: float = None) -> np.ndarray:
        """Katz centrality for systemic importance"""
        n = exposure_matrix.shape[0]
        
        # Normalize and transpose for centrality calculation
        A = exposure_matrix.T
        row_sums = A.sum(axis=1)
        row_sums[row_sums == 0] = 1
        A_normalized = A / row_sums[:, np.newaxis]
        
        # Compute largest eigenvalue for alpha selection
        if alpha is None:
            eigenvalues = np.linalg.eigvals(A_normalized)
            alpha = 0.9 / np.max(np.abs(eigenvalues))
            
        # Katz centrality: (I - alpha*A)^(-1) * 1
        I = np.eye(n)
        try:
            katz = np.linalg.solve(I - alpha * A_normalized, np.ones(n))
            katz = katz / np.sum(katz)  # Normalize
        except np.linalg.LinAlgError:
            logger.error("Katz centrality computation failed")
            katz = np.ones(n) / n
            
        return katz
        
    def run_stress_scenarios(self, exposure_matrix: np.ndarray, 
                           nodes: List[str], 
                           node_map: Dict) -> pd.DataFrame:
        """Run stress tests for each entity default"""
        results = []
        
        for i, node_id in enumerate(nodes):
            # Single entity shock
            initial_shock = np.zeros(len(nodes))
            initial_shock[i] = self.config.stress_shock_size
            
            # Calculate contagion
            h, metrics = self.calculate_debtrank(exposure_matrix, initial_shock)
            
            # Find most affected nodes
            affected_indices = np.argsort(h)[-5:][::-1]  # Top 5
            affected_nodes = [(nodes[j], h[j]) for j in affected_indices if j != i]
            
            results.append({
                'shocked_entity': node_id,
                'entity_name': node_map[node_id]['name'],
                'entity_type': node_map[node_id]['type'],
                'total_impact': metrics['total_impact'],
                'systemic_risk': metrics['systemic_risk'],
                'amplification': metrics['amplification_factor'],
                'most_affected': affected_nodes[:3]
            })
            
        return pd.DataFrame(results).sort_values('systemic_risk', ascending=False)
        
    def identify_systemically_important(self, exposure_matrix: np.ndarray,
                                      nodes: List[str],
                                      node_map: Dict) -> pd.DataFrame:
        """Identify systemically important financial institutions (SIFIs)"""
        # DebtRank importance
        uniform_shock = np.ones(len(nodes)) * 0.01
        h_debtrank, _ = self.calculate_debtrank(exposure_matrix, uniform_shock)
        
        # Katz centrality
        katz = self.calculate_katz_centrality(exposure_matrix)
        
        # Eigenvector centrality
        G = nx.from_numpy_array(exposure_matrix, create_using=nx.DiGraph)
        try:
            eigen_cent = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
            eigen_values = np.array([eigen_cent.get(i, 0) for i in range(len(nodes))])
        except:
            eigen_values = np.ones(len(nodes)) / len(nodes)
            
        # Combine metrics
        results = []
        for i, node_id in enumerate(nodes):
            results.append({
                'entity_id': node_id,
                'name': node_map[node_id]['name'],
                'type': node_map[node_id]['type'],
                'debtrank_score': h_debtrank[i],
                'katz_centrality': katz[i],
                'eigenvector_centrality': eigen_values[i],
                'combined_score': (h_debtrank[i] + katz[i] + eigen_values[i]) / 3
            })
            
        df = pd.DataFrame(results).sort_values('combined_score', ascending=False)
        
        # Flag as SIFI if in top 10% by combined score
        threshold = df['combined_score'].quantile(0.9)
        df['is_sifi'] = df['combined_score'] >= threshold
        
        return df
        
    def calculate_network_metrics(self, exposure_matrix: np.ndarray) -> Dict:
        """Calculate overall network health metrics"""
        G = nx.from_numpy_array(exposure_matrix, create_using=nx.DiGraph)
        
        # Basic metrics
        density = nx.density(G)
        
        # Clustering
        try:
            clustering = nx.average_clustering(G, weight='weight')
        except:
            clustering = 0
            
        # Connectivity
        is_connected = nx.is_weakly_connected(G)
        num_components = nx.number_weakly_connected_components(G)
        
        # Centralization
        in_degrees = dict(G.in_degree(weight='weight'))
        out_degrees = dict(G.out_degree(weight='weight'))
        
        max_in = max(in_degrees.values()) if in_degrees else 0
        max_out = max(out_degrees.values()) if out_degrees else 0
        
        return {
            'density': density,
            'clustering_coefficient': clustering,
            'is_connected': is_connected,
            'num_components': num_components,
            'max_in_degree': max_in,
            'max_out_degree': max_out,
            'reciprocity': nx.reciprocity(G) if G.number_of_edges() > 0 else 0
        }
        
    def store_results(self, date: datetime, stress_results: pd.DataFrame,
                     sifi_results: pd.DataFrame, network_metrics: Dict):
        """Store analysis results in Neo4j"""
        with self.driver.session() as session:
            # Store stress test results
            session.run("""
                CREATE (st:StressTest {
                    date: $date,
                    type: 'DebtRank',
                    run_time: datetime(),
                    network_metrics: $metrics
                })
            """, date=date.isoformat(), metrics=network_metrics)
            
            # Mark SIFIs
            for _, row in sifi_results[sifi_results['is_sifi']].iterrows():
                session.run("""
                    MATCH (e:Entity {entity_id: $entity_id})
                    SET e.is_sifi = true,
                        e.systemic_score = $score,
                        e.sifi_updated = $date
                """, entity_id=row['entity_id'], 
                     score=row['combined_score'],
                     date=date.isoformat())
                     
    def pipeline(self, date: datetime) -> Dict:
        """Full contagion analysis pipeline"""
        logger.info(f"Running contagion analysis for {date}")
        
        # Load network
        exposure_matrix, nodes, node_map = self.load_exposure_network(date)
        
        # Run stress scenarios
        stress_results = self.run_stress_scenarios(exposure_matrix, nodes, node_map)
        
        # Identify SIFIs
        sifi_results = self.identify_systemically_important(exposure_matrix, nodes, node_map)
        
        # Calculate network metrics
        network_metrics = self.calculate_network_metrics(exposure_matrix)
        
        # Store results
        self.store_results(date, stress_results, sifi_results, network_metrics)
        
        return {
            'date': date,
            'stress_results': stress_results,
            'sifi_results': sifi_results,
            'network_metrics': network_metrics,
            'top_systemic_entities': sifi_results.head(10)
        }
        
    def close(self):
        """Clean up resources"""
        self.driver.close()