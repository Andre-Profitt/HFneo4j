#!/usr/bin/env python3
"""Load sample S&P 500 data for testing GEFM and other pillars"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from neo4j import GraphDatabase
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pillars.gefm.factor_model import GraphEnhancedFactorModel
from src.pillars.contagion.debtrank import DebtRankEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "hedgefund123!")


def fetch_sp500_data(symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical price data from Yahoo Finance"""
    logger.info(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
    
    # Calculate returns
    returns = pd.DataFrame()
    for symbol in symbols:
        try:
            if len(symbols) == 1:
                prices = data['Adj Close']
            else:
                prices = data[symbol]['Adj Close']
            returns[symbol] = prices.pct_change()
        except:
            logger.warning(f"Failed to process {symbol}")
            
    returns = returns.dropna()
    logger.info(f"Fetched {len(returns)} days of return data")
    return returns


def create_synthetic_exposures(driver, date: datetime):
    """Create synthetic exposure network for contagion testing"""
    with driver.session() as session:
        # Create exposures between hedge funds and prime brokers
        exposures = [
            # Hedge funds to prime brokers
            {'from': 'HF001', 'to': 'PB001', 'amount': 1500000000, 'type': 'margin'},
            {'from': 'HF001', 'to': 'PB002', 'amount': 800000000, 'type': 'margin'},
            {'from': 'HF002', 'to': 'PB001', 'amount': 2000000000, 'type': 'margin'},
            {'from': 'HF002', 'to': 'PB002', 'amount': 1500000000, 'type': 'margin'},
            {'from': 'HF003', 'to': 'PB002', 'amount': 1000000000, 'type': 'margin'},
            
            # Prime brokers to CCP
            {'from': 'PB001', 'to': 'CCP001', 'amount': 5000000000, 'type': 'clearing'},
            {'from': 'PB002', 'to': 'CCP001', 'amount': 4500000000, 'type': 'clearing'},
            
            # Some inter-hedge fund exposures (e.g., via swaps)
            {'from': 'HF001', 'to': 'HF002', 'amount': 200000000, 'type': 'swap'},
            {'from': 'HF002', 'to': 'HF003', 'amount': 150000000, 'type': 'swap'},
        ]
        
        session.run("""
            UNWIND $exposures AS exp
            MATCH (a:Entity {entity_id: exp.from})
            MATCH (b:Entity {entity_id: exp.to})
            MERGE (a)-[r:EXPOSURE {date: $date}]->(b)
            SET r.amount = exp.amount,
                r.exposure_type = exp.type,
                r.created_at = datetime()
        """, exposures=exposures, date=date.isoformat())
        
        logger.info(f"Created {len(exposures)} synthetic exposures")


def run_gefm_example(returns: pd.DataFrame, date: datetime):
    """Run GEFM factor model example"""
    logger.info("Running GEFM example...")
    
    gefm = GraphEnhancedFactorModel(
        NEO4J_URI, 
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    
    try:
        # Run the pipeline
        results = gefm.pipeline(returns, date)
        
        logger.info(f"GEFM Results:")
        logger.info(f"  - Found {results['num_factors']} factors")
        logger.info(f"  - Average risk reduction: {results['risk_metrics']['risk_reduction'].mean():.2%}")
        logger.info(f"  - Factor returns range: [{results['factor_returns'].min():.4f}, {results['factor_returns'].max():.4f}]")
        
        # Show top clusters
        cluster_sizes = pd.Series(results['cluster_map']).value_counts()
        logger.info(f"  - Cluster sizes: {dict(cluster_sizes.head())}")
        
        return results
        
    finally:
        gefm.close()


def run_debtrank_example(date: datetime):
    """Run DebtRank contagion analysis example"""
    logger.info("Running DebtRank example...")
    
    debtrank = DebtRankEngine(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    
    try:
        # Run the pipeline
        results = debtrank.pipeline(date)
        
        logger.info(f"DebtRank Results:")
        logger.info(f"  - Network density: {results['network_metrics']['density']:.4f}")
        logger.info(f"  - Clustering coefficient: {results['network_metrics']['clustering_coefficient']:.4f}")
        
        # Show top systemic entities
        logger.info("\nTop 5 Systemically Important Entities:")
        for _, row in results['top_systemic_entities'].head(5).iterrows():
            logger.info(f"  - {row['name']} ({row['type']}): score={row['combined_score']:.4f}")
            
        # Show highest impact scenarios
        logger.info("\nHighest Impact Stress Scenarios:")
        for _, row in results['stress_results'].head(3).iterrows():
            logger.info(f"  - {row['entity_name']} default: systemic_risk={row['systemic_risk']:.4f}, amplification={row['amplification']:.2f}x")
            
        return results
        
    finally:
        debtrank.close()


def main():
    """Main example runner"""
    # Define test parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months of data
    test_date = end_date - timedelta(days=1)    # Yesterday
    
    # Test symbols (subset of S&P 500)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'JNJ', 'UNH', 'XOM']
    
    # Initialize driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        # Fetch market data
        returns = fetch_sp500_data(symbols, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Create synthetic exposures
        create_synthetic_exposures(driver, test_date)
        
        # Run GEFM
        gefm_results = run_gefm_example(returns, test_date)
        
        # Run DebtRank
        debtrank_results = run_debtrank_example(test_date)
        
        logger.info("\nExample completed successfully!")
        logger.info("Check Neo4j browser at http://localhost:7474 to explore the graph")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise
    finally:
        driver.close()


if __name__ == "__main__":
    # Install yfinance if needed
    try:
        import yfinance
    except ImportError:
        logger.info("Installing yfinance...")
        os.system("pip install yfinance")
        import yfinance
        
    main()