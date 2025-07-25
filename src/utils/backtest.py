"""Backtesting Framework
Comprehensive backtesting for all graph-based strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from neo4j import GraphDatabase
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000_000  # $10M
    position_limits: Dict[str, float] = None  # Max position per security
    transaction_cost: float = 0.0005  # 5 bps
    slippage_model: str = 'linear'  # 'linear', 'square_root', 'none'
    rebalance_frequency: int = 20  # Trading days
    risk_free_rate: float = 0.02  # Annual
    benchmark: str = 'SPY'
    
    def __post_init__(self):
        if self.position_limits is None:
            self.position_limits = {'default': 0.1, 'max_sector': 0.3}


class PerformanceMetrics:
    """Calculate comprehensive performance metrics"""
    
    @staticmethod
    def calculate_returns_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict:
        """Calculate return-based metrics"""
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Risk-adjusted metrics
        sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        calmar = annual_return / abs(PerformanceMetrics.max_drawdown(returns)) if PerformanceMetrics.max_drawdown(returns) != 0 else 0
        
        # Distribution metrics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': PerformanceMetrics.max_drawdown(returns),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': np.percentile(returns, 5),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean()
        }
    
    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - running_max) / running_max
        return drawdowns.min()
        
    @staticmethod
    def rolling_metrics(returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        
        rolling_returns = returns.rolling(window=window).apply(
            lambda x: (1 + x).prod() - 1
        )
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = rolling_returns / rolling_vol
        
        return pd.DataFrame({
            'rolling_return': rolling_returns,
            'rolling_volatility': rolling_vol,
            'rolling_sharpe': rolling_sharpe
        })


class GraphBacktester:
    """Backtest graph-based trading strategies"""
    
    def __init__(self, neo4j_uri: str, auth: tuple, config: BacktestConfig = None):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=auth)
        self.config = config or BacktestConfig()
        self.results_cache = {}
        
    def load_historical_data(self, start_date: datetime, end_date: datetime,
                           symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Load historical price data from Neo4j"""
        
        with self.driver.session() as session:
            query = """
                MATCH (s:Security)-[:HAS_PRICE]->(p:PricePoint)
                WHERE p.date >= $start_date AND p.date <= $end_date
            """
            
            if symbols:
                query += " AND s.symbol IN $symbols"
                
            query += """
                RETURN s.symbol AS symbol, p.date AS date,
                       p.close AS close, p.returns AS returns,
                       p.volume AS volume
                ORDER BY date, symbol
            """
            
            result = session.run(query, 
                               start_date=start_date.isoformat(),
                               end_date=end_date.isoformat(),
                               symbols=symbols)
            
            records = list(result)
            if not records:
                raise ValueError("No data found for the specified period")
                
            df = pd.DataFrame(records)
            
            # Pivot to wide format
            prices = df.pivot(index='date', columns='symbol', values='close')
            returns = df.pivot(index='date', columns='symbol', values='returns')
            volumes = df.pivot(index='date', columns='symbol', values='volume')
            
            return {
                'prices': prices,
                'returns': returns,
                'volumes': volumes
            }
            
    def backtest_gefm_strategy(self, start_date: datetime, end_date: datetime) -> Dict:
        """Backtest Graph-Enhanced Factor Model strategy"""
        logger.info("Backtesting GEFM strategy")
        
        # Load factor model results from Neo4j
        with self.driver.session() as session:
            factors = session.run("""
                MATCH (fm:FactorModel {type: 'GEFM'})
                WHERE fm.date >= $start_date AND fm.date <= $end_date
                MATCH (s:Security)-[l:LOADS_ON]->(fm)
                RETURN fm.date AS date, s.symbol AS symbol,
                       l.factor AS factor, l.loading AS loading
                ORDER BY date, symbol
            """, start_date=start_date.isoformat(), end_date=end_date.isoformat())
            
            factor_df = pd.DataFrame(list(factors))
            
        if factor_df.empty:
            raise ValueError("No GEFM factors found for backtesting period")
            
        # Load returns
        data = self.load_historical_data(start_date, end_date)
        returns = data['returns']
        
        # Build factor portfolios
        portfolio_returns = []
        dates = factor_df['date'].unique()
        
        for date in dates:
            if date not in returns.index:
                continue
                
            # Get factor loadings for this date
            date_factors = factor_df[factor_df['date'] == date]
            
            # Equal weight within each factor
            weights = {}
            for factor in date_factors['factor'].unique():
                factor_stocks = date_factors[date_factors['factor'] == factor]['symbol'].tolist()
                for stock in factor_stocks:
                    if stock in returns.columns:
                        weights[stock] = 1 / len(factor_stocks) / len(date_factors['factor'].unique())
                        
            # Calculate portfolio return
            if weights:
                date_return = sum(weights.get(stock, 0) * returns.loc[date, stock] 
                                for stock in returns.columns)
                portfolio_returns.append(date_return)
                
        strategy_returns = pd.Series(portfolio_returns, index=dates[dates.isin(returns.index)])
        
        # Calculate metrics
        metrics = PerformanceMetrics.calculate_returns_metrics(strategy_returns)
        
        # Compare with benchmark
        benchmark_returns = returns[self.config.benchmark] if self.config.benchmark in returns.columns else returns.mean(axis=1)
        benchmark_metrics = PerformanceMetrics.calculate_returns_metrics(benchmark_returns)
        
        return {
            'strategy_metrics': metrics,
            'benchmark_metrics': benchmark_metrics,
            'excess_return': metrics['annual_return'] - benchmark_metrics['annual_return'],
            'information_ratio': (metrics['annual_return'] - benchmark_metrics['annual_return']) / returns.std() if returns.std() > 0 else 0,
            'strategy_returns': strategy_returns,
            'benchmark_returns': benchmark_returns
        }
        
    def backtest_gnn_signals(self, start_date: datetime, end_date: datetime,
                           confidence_threshold: float = 0.6) -> Dict:
        """Backtest GNN alpha signals"""
        logger.info("Backtesting GNN signals")
        
        # Load predictions
        with self.driver.session() as session:
            predictions = session.run("""
                MATCH (s:Security)-[:HAS_PREDICTION]->(p:Prediction)
                WHERE p.date >= $start_date AND p.date <= $end_date
                AND p.model = 'DishFTGNN'
                RETURN p.date AS date, s.symbol AS symbol,
                       p.signal AS signal, p.confidence AS confidence,
                       p.p_up AS p_up, p.p_down AS p_down
                ORDER BY date, symbol
            """, start_date=start_date.isoformat(), end_date=end_date.isoformat())
            
            pred_df = pd.DataFrame(list(predictions))
            
        if pred_df.empty:
            raise ValueError("No GNN predictions found for backtesting period")
            
        # Load returns
        data = self.load_historical_data(start_date, end_date)
        returns = data['returns']
        
        # Generate trading signals
        portfolio_returns = []
        positions = {}
        
        for date in pred_df['date'].unique():
            if date not in returns.index:
                continue
                
            date_preds = pred_df[pred_df['date'] == date]
            
            # Filter by confidence
            high_conf = date_preds[date_preds['confidence'] > confidence_threshold]
            
            # Long/short positions
            longs = high_conf[high_conf['signal'] == 2]['symbol'].tolist()  # Up signal
            shorts = high_conf[high_conf['signal'] == 0]['symbol'].tolist()  # Down signal
            
            # Calculate weights (equal weight long/short)
            new_positions = {}
            if longs:
                for symbol in longs:
                    if symbol in returns.columns:
                        new_positions[symbol] = 1 / len(longs) * 0.5  # 50% long
                        
            if shorts:
                for symbol in shorts:
                    if symbol in returns.columns:
                        new_positions[symbol] = -1 / len(shorts) * 0.5  # 50% short
                        
            # Calculate returns including transaction costs
            date_return = 0
            turnover = 0
            
            for symbol in returns.columns:
                old_weight = positions.get(symbol, 0)
                new_weight = new_positions.get(symbol, 0)
                
                if symbol in returns.columns and not pd.isna(returns.loc[date, symbol]):
                    date_return += new_weight * returns.loc[date, symbol]
                    turnover += abs(new_weight - old_weight)
                    
            # Apply transaction costs
            date_return -= turnover * self.config.transaction_cost
            portfolio_returns.append(date_return)
            positions = new_positions
            
        strategy_returns = pd.Series(
            portfolio_returns, 
            index=pred_df['date'].unique()[pred_df['date'].unique().isin(returns.index)]
        )
        
        # Calculate metrics
        metrics = PerformanceMetrics.calculate_returns_metrics(strategy_returns)
        
        # Signal analysis
        signal_analysis = self._analyze_signal_quality(pred_df, returns)
        
        return {
            'strategy_metrics': metrics,
            'signal_analysis': signal_analysis,
            'strategy_returns': strategy_returns,
            'avg_turnover': turnover / len(portfolio_returns) if portfolio_returns else 0
        }
        
    def backtest_graph_risk_parity(self, start_date: datetime, end_date: datetime) -> Dict:
        """Backtest Graph-Signal Risk Parity strategy"""
        logger.info("Backtesting GSRP strategy")
        
        # Load optimized weights
        with self.driver.session() as session:
            weights = session.run("""
                MATCH (opt:Optimization {type: 'GSRP'})
                WHERE opt.date >= $start_date AND opt.date <= $end_date
                MATCH (s:Security)-[w:HAS_WEIGHT]->(opt)
                RETURN opt.date AS date, s.symbol AS symbol,
                       w.weight AS weight, opt.graph_total_variation AS gtv
                ORDER BY date, symbol
            """, start_date=start_date.isoformat(), end_date=end_date.isoformat())
            
            weights_df = pd.DataFrame(list(weights))
            
        if weights_df.empty:
            raise ValueError("No GSRP weights found for backtesting period")
            
        # Load returns
        data = self.load_historical_data(start_date, end_date)
        returns = data['returns']
        
        # Calculate portfolio returns
        portfolio_returns = []
        rebalance_dates = []
        
        current_weights = {}
        for date in returns.index:
            # Check if we need to rebalance
            date_weights = weights_df[weights_df['date'] == date]
            
            if not date_weights.empty:
                # Rebalance
                new_weights = {}
                for _, row in date_weights.iterrows():
                    if row['symbol'] in returns.columns:
                        new_weights[row['symbol']] = row['weight']
                        
                # Calculate turnover
                turnover = sum(abs(new_weights.get(s, 0) - current_weights.get(s, 0)) 
                             for s in set(new_weights) | set(current_weights))
                             
                current_weights = new_weights
                rebalance_dates.append(date)
            else:
                turnover = 0
                
            # Calculate return
            date_return = sum(current_weights.get(symbol, 0) * returns.loc[date, symbol] 
                            for symbol in returns.columns if not pd.isna(returns.loc[date, symbol]))
                            
            # Apply costs on rebalance days
            if turnover > 0:
                date_return -= turnover * self.config.transaction_cost
                
            portfolio_returns.append(date_return)
            
        strategy_returns = pd.Series(portfolio_returns, index=returns.index)
        
        # Calculate metrics
        metrics = PerformanceMetrics.calculate_returns_metrics(strategy_returns)
        
        # Compare with standard MVO
        benchmark_returns = returns.mean(axis=1)  # Equal weight benchmark
        benchmark_metrics = PerformanceMetrics.calculate_returns_metrics(benchmark_returns)
        
        # Graph smoothness analysis
        gtv_values = weights_df.groupby('date')['gtv'].first()
        
        return {
            'strategy_metrics': metrics,
            'benchmark_metrics': benchmark_metrics,
            'improvement': {
                'return': metrics['annual_return'] - benchmark_metrics['annual_return'],
                'sharpe': metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio'],
                'max_dd': metrics['max_drawdown'] - benchmark_metrics['max_drawdown']
            },
            'strategy_returns': strategy_returns,
            'benchmark_returns': benchmark_returns,
            'rebalance_count': len(rebalance_dates),
            'avg_gtv': gtv_values.mean() if not gtv_values.empty else 0
        }
        
    def _analyze_signal_quality(self, predictions: pd.DataFrame, returns: pd.DataFrame) -> Dict:
        """Analyze the quality of trading signals"""
        
        # Merge predictions with next-day returns
        analysis = []
        
        for _, pred in predictions.iterrows():
            date = pred['date']
            symbol = pred['symbol']
            
            # Find next trading day
            future_dates = returns.index[returns.index > date]
            if len(future_dates) > 0 and symbol in returns.columns:
                next_date = future_dates[0]
                actual_return = returns.loc[next_date, symbol]
                
                if not pd.isna(actual_return):
                    analysis.append({
                        'predicted_signal': pred['signal'],
                        'confidence': pred['confidence'],
                        'actual_return': actual_return,
                        'correct': (pred['signal'] == 2 and actual_return > 0) or 
                                  (pred['signal'] == 0 and actual_return < 0) or
                                  (pred['signal'] == 1 and abs(actual_return) < 0.001)
                    })
                    
        if not analysis:
            return {}
            
        analysis_df = pd.DataFrame(analysis)
        
        # Calculate accuracy by confidence level
        confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        accuracy_by_confidence = {}
        
        for i in range(len(confidence_bins) - 1):
            mask = (analysis_df['confidence'] >= confidence_bins[i]) & \
                   (analysis_df['confidence'] < confidence_bins[i + 1])
            subset = analysis_df[mask]
            
            if len(subset) > 0:
                accuracy_by_confidence[f"{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}"] = {
                    'accuracy': subset['correct'].mean(),
                    'count': len(subset),
                    'avg_return': subset['actual_return'].mean()
                }
                
        return {
            'overall_accuracy': analysis_df['correct'].mean(),
            'accuracy_by_confidence': accuracy_by_confidence,
            'signal_distribution': analysis_df['predicted_signal'].value_counts().to_dict()
        }
        
    def create_tearsheet(self, backtest_results: Dict, strategy_name: str):
        """Create a comprehensive performance tearsheet"""
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'{strategy_name} Backtest Results', fontsize=16)
        
        strategy_returns = backtest_results['strategy_returns']
        
        # 1. Cumulative returns
        ax = axes[0, 0]
        cum_returns = (1 + strategy_returns).cumprod()
        cum_returns.plot(ax=ax, label='Strategy')
        
        if 'benchmark_returns' in backtest_results:
            bench_cum = (1 + backtest_results['benchmark_returns']).cumprod()
            bench_cum.plot(ax=ax, label='Benchmark', alpha=0.7)
            
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        
        # 2. Drawdowns
        ax = axes[0, 1]
        running_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - running_max) / running_max
        drawdowns.plot(ax=ax, color='red')
        ax.set_title('Drawdowns')
        ax.set_ylabel('Drawdown %')
        ax.fill_between(drawdowns.index, drawdowns, 0, color='red', alpha=0.3)
        
        # 3. Returns distribution
        ax = axes[1, 0]
        strategy_returns.hist(bins=50, ax=ax, alpha=0.7, density=True)
        ax.set_title('Returns Distribution')
        ax.set_xlabel('Daily Returns')
        ax.set_ylabel('Frequency')
        
        # Add normal distribution overlay
        mu, std = strategy_returns.mean(), strategy_returns.std()
        x = np.linspace(strategy_returns.min(), strategy_returns.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2, label='Normal')
        ax.legend()
        
        # 4. Rolling Sharpe
        ax = axes[1, 1]
        rolling_sharpe = PerformanceMetrics.rolling_metrics(strategy_returns, 252)['rolling_sharpe']
        rolling_sharpe.plot(ax=ax)
        ax.set_title('Rolling Sharpe Ratio (252 days)')
        ax.set_ylabel('Sharpe Ratio')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 5. Monthly returns heatmap
        ax = axes[2, 0]
        monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_pivot = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        }).pivot(index='Year', columns='Month', values='Return')
        
        sns.heatmap(monthly_pivot, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=ax)
        ax.set_title('Monthly Returns Heatmap')
        
        # 6. Performance metrics table
        ax = axes[2, 1]
        ax.axis('off')
        
        metrics = backtest_results['strategy_metrics']
        table_data = [
            ['Total Return', f"{metrics['total_return']:.2%}"],
            ['Annual Return', f"{metrics['annual_return']:.2%}"],
            ['Volatility', f"{metrics['volatility']:.2%}"],
            ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.3f}"],
            ['Max Drawdown', f"{metrics['max_drawdown']:.2%}"],
            ['Win Rate', f"{metrics['win_rate']:.2%}"],
            ['Profit Factor', f"{metrics['profit_factor']:.2f}"]
        ]
        
        table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                        colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title('Performance Metrics', pad=20)
        
        plt.tight_layout()
        return fig
        
    def compare_strategies(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Compare all strategies side by side"""
        
        results = {}
        
        # Run all backtests
        try:
            gefm_results = self.backtest_gefm_strategy(start_date, end_date)
            results['GEFM'] = gefm_results['strategy_metrics']
        except Exception as e:
            logger.error(f"GEFM backtest failed: {e}")
            
        try:
            gnn_results = self.backtest_gnn_signals(start_date, end_date)
            results['GNN'] = gnn_results['strategy_metrics']
        except Exception as e:
            logger.error(f"GNN backtest failed: {e}")
            
        try:
            gsrp_results = self.backtest_graph_risk_parity(start_date, end_date)
            results['GSRP'] = gsrp_results['strategy_metrics']
        except Exception as e:
            logger.error(f"GSRP backtest failed: {e}")
            
        # Create comparison DataFrame
        comparison = pd.DataFrame(results).T
        
        # Add rankings
        for metric in ['annual_return', 'sharpe_ratio', 'sortino_ratio']:
            if metric in comparison.columns:
                comparison[f'{metric}_rank'] = comparison[metric].rank(ascending=False)
                
        return comparison
        
    def store_backtest_results(self, results: Dict, strategy_name: str):
        """Store backtest results in Neo4j"""
        
        with self.driver.session() as session:
            session.run("""
                CREATE (b:BacktestResult {
                    strategy: $strategy,
                    run_date: datetime(),
                    start_date: $start_date,
                    end_date: $end_date,
                    metrics: $metrics,
                    config: $config
                })
            """,
            strategy=strategy_name,
            start_date=results.get('start_date', '').isoformat() if isinstance(results.get('start_date'), datetime) else '',
            end_date=results.get('end_date', '').isoformat() if isinstance(results.get('end_date'), datetime) else '',
            metrics=json.dumps(results.get('strategy_metrics', {})),
            config=json.dumps(self.config.__dict__))
            
    def close(self):
        """Clean up resources"""
        self.driver.close()