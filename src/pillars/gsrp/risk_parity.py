"""Graph-Signal Risk Parity (GSRP) - Pillar 4
Portfolio optimization with graph regularization for smoother P&L
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import cvxpy as cp
from neo4j import GraphDatabase
import logging
from datetime import datetime, timedelta
import scipy.sparse as sp
from scipy.linalg import sqrtm

logger = logging.getLogger(__name__)


@dataclass
class GSRPConfig:
    min_weight: float = 0.0
    max_weight: float = 0.1
    lambda_graph: float = 0.5  # Graph regularization weight
    lambda_l1: float = 0.01   # L1 penalty for sparsity
    risk_parity_weight: float = 0.3  # Blend with standard MVO
    rebalance_threshold: float = 0.02  # 2% deviation triggers rebalance
    graph_type: str = 'correlation'  # or 'sector', 'supply_chain'
    solver: str = 'ECOS'  # or 'OSQP', 'SCS'


class GraphSignalRiskParity:
    """Optimize portfolio weights to minimize graph total variation"""
    
    def __init__(self, neo4j_uri: str, auth: tuple, config: GSRPConfig = None):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=auth)
        self.config = config or GSRPConfig()
        self._laplacian_cache = {}
        
    def construct_laplacian(self, securities: List[str], date: datetime) -> np.ndarray:
        """Construct graph Laplacian matrix from Neo4j relationships"""
        n = len(securities)
        symbol_to_idx = {sym: i for i, sym in enumerate(securities)}
        
        # Initialize adjacency matrix
        W = np.zeros((n, n))
        
        with self.driver.session() as session:
            # Get edges based on graph type
            if self.config.graph_type == 'correlation':
                edges = session.run("""
                    MATCH (s1:Security)-[r:CORRELATES_WITH {date: $date}]->(s2:Security)
                    WHERE s1.symbol IN $symbols AND s2.symbol IN $symbols
                    AND abs(r.correlation) > 0.3
                    RETURN s1.symbol AS source, s2.symbol AS target,
                           abs(r.correlation) AS weight
                """, symbols=securities, date=date.isoformat())
                
            elif self.config.graph_type == 'sector':
                edges = session.run("""
                    MATCH (s1:Security)-[:IN_SECTOR]->(sec:Sector)<-[:IN_SECTOR]-(s2:Security)
                    WHERE s1.symbol IN $symbols AND s2.symbol IN $symbols
                    AND s1.symbol <> s2.symbol
                    RETURN s1.symbol AS source, s2.symbol AS target,
                           1.0 AS weight
                """, symbols=securities)
                
            # Build adjacency matrix
            for record in edges:
                if record['source'] in symbol_to_idx and record['target'] in symbol_to_idx:
                    i = symbol_to_idx[record['source']]
                    j = symbol_to_idx[record['target']]
                    W[i, j] = record['weight']
                    W[j, i] = record['weight']  # Ensure symmetry
                    
        # Compute Laplacian L = D - W
        D = np.diag(W.sum(axis=1))
        L = D - W
        
        # Normalize Laplacian
        D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(D.diagonal(), 1e-8)))
        L_norm = D_sqrt_inv @ L @ D_sqrt_inv
        
        return L_norm
        
    def graph_total_variation(self, weights: np.ndarray, laplacian: np.ndarray) -> float:
        """Calculate graph total variation: w^T L w"""
        return weights.T @ laplacian @ weights
        
    def risk_parity_objective(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Risk parity objective: sum of squared differences in risk contributions"""
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights / portfolio_vol
        contrib = weights * marginal_contrib
        
        # Equal risk contribution target
        target_contrib = portfolio_vol / len(weights)
        
        return np.sum((contrib - target_contrib) ** 2)
        
    def optimize(self, returns: pd.DataFrame, 
                expected_returns: Optional[pd.Series] = None,
                target_return: Optional[float] = None,
                date: Optional[datetime] = None) -> Dict:
        """
        Solve the graph-regularized portfolio optimization problem:
        min_w { w^T Σ w + λ w^T L w } subject to constraints
        """
        securities = returns.columns.tolist()
        n = len(securities)
        
        # Calculate covariance matrix
        cov_matrix = returns.cov().values
        
        # Get expected returns (use historical mean if not provided)
        if expected_returns is None:
            expected_returns = returns.mean()
        mu = expected_returns.values
        
        # Construct Laplacian
        if date is None:
            date = datetime.now()
        L = self.construct_laplacian(securities, date)
        
        # Define optimization variables
        w = cp.Variable(n)
        
        # Objective function
        portfolio_variance = cp.quad_form(w, cov_matrix)
        graph_smoothness = cp.quad_form(w, L)
        l1_penalty = cp.norm(w, 1)
        
        objective = portfolio_variance + self.config.lambda_graph * graph_smoothness
        
        if self.config.lambda_l1 > 0:
            objective += self.config.lambda_l1 * l1_penalty
            
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
            w >= self.config.min_weight,  # Long only / min position
            w <= self.config.max_weight,  # Concentration limit
        ]
        
        # Add return constraint if target specified
        if target_return is not None:
            constraints.append(mu.T @ w >= target_return)
            
        # Solve the problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        
        try:
            problem.solve(solver=getattr(cp, self.config.solver), verbose=False)
            
            if problem.status != cp.OPTIMAL:
                logger.warning(f"Optimization status: {problem.status}")
                
            optimal_weights = w.value
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Fallback to equal weights
            optimal_weights = np.ones(n) / n
            
        # Calculate portfolio metrics
        portfolio_return = optimal_weights @ mu
        portfolio_vol = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)
        sharpe_ratio = portfolio_return / portfolio_vol * np.sqrt(252)
        
        # Graph smoothness metric
        gtv = self.graph_total_variation(optimal_weights, L)
        
        # Risk contributions
        marginal_contrib = cov_matrix @ optimal_weights / portfolio_vol
        risk_contrib = optimal_weights * marginal_contrib
        
        # Compare with standard MVO
        mvo_weights = self._solve_standard_mvo(cov_matrix, mu, target_return)
        mvo_vol = np.sqrt(mvo_weights @ cov_matrix @ mvo_weights)
        mvo_gtv = self.graph_total_variation(mvo_weights, L)
        
        results = {
            'weights': pd.Series(optimal_weights, index=securities),
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'graph_total_variation': gtv,
            'risk_contributions': pd.Series(risk_contrib, index=securities),
            'optimization_status': problem.status if 'problem' in locals() else 'FAILED',
            'improvement_vs_mvo': {
                'volatility_reduction': (mvo_vol - portfolio_vol) / mvo_vol,
                'gtv_reduction': (mvo_gtv - gtv) / mvo_gtv if mvo_gtv > 0 else 0
            }
        }
        
        # Store results in Neo4j
        self._store_optimization_results(date, results)
        
        return results
        
    def _solve_standard_mvo(self, cov_matrix: np.ndarray, 
                           expected_returns: np.ndarray,
                           target_return: Optional[float] = None) -> np.ndarray:
        """Solve standard mean-variance optimization for comparison"""
        n = len(expected_returns)
        w = cp.Variable(n)
        
        objective = cp.quad_form(w, cov_matrix)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= self.config.max_weight
        ]
        
        if target_return is not None:
            constraints.append(expected_returns.T @ w >= target_return)
            
        problem = cp.Problem(cp.Minimize(objective), constraints)
        
        try:
            problem.solve(solver=getattr(cp, self.config.solver), verbose=False)
            return w.value
        except:
            return np.ones(n) / n
            
    def multi_period_optimization(self, returns_history: pd.DataFrame,
                                lookback_days: int = 252,
                                rebalance_frequency: int = 20) -> pd.DataFrame:
        """Run rolling window optimization with rebalancing"""
        results = []
        
        dates = returns_history.index[lookback_days:]
        
        for i, date in enumerate(dates):
            if i % rebalance_frequency != 0 and i > 0:
                # No rebalancing - use previous weights
                results.append(results[-1])
                continue
                
            # Get historical window
            window_end = date
            window_start = window_end - timedelta(days=lookback_days)
            returns_window = returns_history[window_start:window_end]
            
            # Optimize
            opt_result = self.optimize(returns_window, date=date)
            
            results.append({
                'date': date,
                'weights': opt_result['weights'],
                'expected_return': opt_result['expected_return'],
                'volatility': opt_result['volatility'],
                'sharpe_ratio': opt_result['sharpe_ratio'],
                'gtv': opt_result['graph_total_variation']
            })
            
            logger.info(f"Optimized for {date}: Sharpe={opt_result['sharpe_ratio']:.3f}, "
                       f"GTV={opt_result['graph_total_variation']:.6f}")
                       
        return pd.DataFrame(results).set_index('date')
        
    def calculate_turnover(self, weights_history: pd.DataFrame) -> pd.Series:
        """Calculate portfolio turnover from weight history"""
        turnover = weights_history.diff().abs().sum(axis=1)
        return turnover
        
    def backtest(self, returns: pd.DataFrame, weights_history: pd.DataFrame) -> Dict:
        """Backtest the strategy with transaction costs"""
        # Align dates
        common_dates = returns.index.intersection(weights_history.index)
        returns_aligned = returns.loc[common_dates]
        weights_aligned = weights_history.loc[common_dates]
        
        # Calculate portfolio returns
        portfolio_returns = (returns_aligned * weights_aligned).sum(axis=1)
        
        # Transaction costs (5 bps)
        turnover = self.calculate_turnover(weights_aligned)
        transaction_costs = turnover * 0.0005
        
        # Net returns
        net_returns = portfolio_returns - transaction_costs
        
        # Performance metrics
        total_return = (1 + net_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(net_returns)) - 1
        volatility = net_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility
        max_drawdown = (net_returns.cumsum() - net_returns.cumsum().cummax()).min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_turnover': turnover.mean(),
            'portfolio_returns': net_returns
        }
        
    def _store_optimization_results(self, date: datetime, results: Dict):
        """Store optimization results in Neo4j"""
        with self.driver.session() as session:
            # Create optimization record
            session.run("""
                CREATE (opt:Optimization {
                    date: $date,
                    type: 'GSRP',
                    expected_return: $expected_return,
                    volatility: $volatility,
                    sharpe_ratio: $sharpe_ratio,
                    graph_total_variation: $gtv,
                    lambda_graph: $lambda_graph,
                    created_at: datetime()
                })
            """, date=date.isoformat(),
                expected_return=float(results['expected_return']),
                volatility=float(results['volatility']),
                sharpe_ratio=float(results['sharpe_ratio']),
                gtv=float(results['graph_total_variation']),
                lambda_graph=self.config.lambda_graph)
            
            # Store weights
            for symbol, weight in results['weights'].items():
                if weight > 1e-6:  # Only store non-zero weights
                    session.run("""
                        MATCH (s:Security {symbol: $symbol})
                        MATCH (opt:Optimization {date: $date, type: 'GSRP'})
                        CREATE (s)-[:HAS_WEIGHT {
                            weight: $weight,
                            risk_contribution: $risk_contrib
                        }]->(opt)
                    """, symbol=symbol, 
                        date=date.isoformat(),
                        weight=float(weight),
                        risk_contrib=float(results['risk_contributions'][symbol]))
                        
    def close(self):
        """Clean up resources"""
        self.driver.close()