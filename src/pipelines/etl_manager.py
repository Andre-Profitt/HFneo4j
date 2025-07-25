"""ETL Pipeline Manager
Orchestrates data ingestion, transformation, and loading into Neo4j
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import asyncio
import aiofiles
from neo4j import GraphDatabase, AsyncGraphDatabase
from kafka import KafkaProducer
import pyarrow.parquet as pq
import pyarrow as pa
from dataclasses import dataclass
import schedule
import time
import boto3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import ray

logger = logging.getLogger(__name__)


@dataclass
class ETLConfig:
    batch_size: int = 10000
    parallel_workers: int = 8
    s3_bucket: str = 'hedge-fund-data'
    checkpoint_interval: int = 1000
    error_threshold: float = 0.05  # 5% error rate triggers alert
    data_sources: List[str] = None
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = ['bloomberg', 'refinitiv', 'quandl', 'internal']


class DataValidator:
    """Validate and clean incoming data"""
    
    @staticmethod
    def validate_price_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Validate price/return data"""
        errors = []
        
        # Check for required columns
        required_cols = ['symbol', 'date', 'close', 'volume']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
            
        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates(['symbol', 'date'])
        if len(df) < original_len:
            errors.append(f"Removed {original_len - len(df)} duplicates")
            
        # Validate price ranges
        invalid_prices = df[(df['close'] <= 0) | (df['close'] > 1e6)]
        if len(invalid_prices) > 0:
            errors.append(f"Found {len(invalid_prices)} invalid prices")
            df = df[~df.index.isin(invalid_prices.index)]
            
        # Handle missing values
        df['volume'] = df['volume'].fillna(0)
        
        # Calculate returns
        df = df.sort_values(['symbol', 'date'])
        df['returns'] = df.groupby('symbol')['close'].pct_change()
        
        # Add data quality metrics
        quality_metrics = {
            'total_records': original_len,
            'valid_records': len(df),
            'error_rate': 1 - len(df) / original_len if original_len > 0 else 0,
            'symbols': df['symbol'].nunique(),
            'date_range': (df['date'].min(), df['date'].max()),
            'errors': errors
        }
        
        return df, quality_metrics
        
    @staticmethod
    def validate_exposure_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Validate exposure/position data"""
        errors = []
        
        # Required columns
        required_cols = ['source_entity', 'target_entity', 'exposure_amount', 'date']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
            return pd.DataFrame(), {'error': 'Missing required columns'}
            
        # Validate amounts
        invalid_amounts = df[df['exposure_amount'] < 0]
        if len(invalid_amounts) > 0:
            errors.append(f"Found {len(invalid_amounts)} negative exposures")
            df = df[df['exposure_amount'] >= 0]
            
        # Check entity existence
        all_entities = set(df['source_entity'].unique()) | set(df['target_entity'].unique())
        
        quality_metrics = {
            'total_records': len(df),
            'unique_entities': len(all_entities),
            'total_exposure': df['exposure_amount'].sum(),
            'errors': errors
        }
        
        return df, quality_metrics


class ETLPipeline:
    """Main ETL pipeline orchestrator"""
    
    def __init__(self, neo4j_uri: str, auth: tuple, config: ETLConfig = None):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=auth)
        self.config = config or ETLConfig()
        self.validator = DataValidator()
        
        # Initialize Ray for distributed processing
        if not ray.is_initialized():
            ray.init(num_cpus=self.config.parallel_workers)
            
        # S3 client
        self.s3_client = boto3.client('s3')
        
        # Kafka producer for streaming
        self.kafka_producer = KafkaProducer(
            bootstrap_servers='localhost:9092',
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Metrics
        self.pipeline_metrics = {
            'processed_records': 0,
            'failed_records': 0,
            'last_run': None,
            'average_processing_time': 0
        }
        
    @ray.remote
    def process_price_batch(self, batch_df: pd.DataFrame) -> Dict:
        """Process a batch of price data in parallel"""
        start_time = time.time()
        
        # Validate
        clean_df, metrics = DataValidator.validate_price_data(batch_df)
        
        if metrics['error_rate'] > 0.1:
            logger.error(f"High error rate in batch: {metrics['error_rate']:.2%}")
            
        # Calculate additional features
        for window in [5, 20, 60]:
            clean_df[f'returns_{window}d'] = (
                clean_df.groupby('symbol')['returns']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
        # Volume metrics
        clean_df['volume_zscore'] = (
            clean_df.groupby('symbol')['volume']
            .transform(lambda x: (x - x.mean()) / x.std())
        )
        
        # Market cap (if available)
        if 'market_cap' in clean_df.columns:
            clean_df['market_cap_log'] = np.log(clean_df['market_cap'] + 1)
            
        processing_time = time.time() - start_time
        
        return {
            'data': clean_df,
            'metrics': metrics,
            'processing_time': processing_time
        }
        
    def load_price_data_to_neo4j(self, df: pd.DataFrame, batch_size: int = 5000):
        """Load price data into Neo4j"""
        with self.driver.session() as session:
            # Create/update securities
            securities = df[['symbol']].drop_duplicates()
            
            session.run("""
                UNWIND $securities AS sec
                MERGE (s:Security {symbol: sec.symbol})
                SET s.last_updated = datetime()
            """, securities=securities.to_dict('records'))
            
            # Load price data in batches
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                session.run("""
                    UNWIND $prices AS price
                    MATCH (s:Security {symbol: price.symbol})
                    CREATE (p:PricePoint {
                        date: date(price.date),
                        close: price.close,
                        volume: price.volume,
                        returns: price.returns,
                        returns_5d: price.returns_5d,
                        returns_20d: price.returns_20d,
                        returns_60d: price.returns_60d,
                        volume_zscore: price.volume_zscore
                    })
                    CREATE (s)-[:HAS_PRICE {date: p.date}]->(p)
                """, prices=batch.to_dict('records'))
                
                logger.info(f"Loaded {min(i+batch_size, len(df))}/{len(df)} price records")
                
    def build_correlation_graphs(self, date: datetime, lookback_days: int = 60):
        """Build correlation graphs for GEFM"""
        logger.info(f"Building correlation graph for {date}")
        
        with self.driver.session() as session:
            # Get returns data
            result = session.run("""
                MATCH (s:Security)-[:HAS_PRICE]->(p:PricePoint)
                WHERE p.date >= $start_date AND p.date <= $end_date
                RETURN s.symbol AS symbol, p.date AS date, p.returns AS returns
                ORDER BY symbol, date
            """, 
            start_date=(date - timedelta(days=lookback_days)).isoformat(),
            end_date=date.isoformat())
            
            # Convert to DataFrame
            records = list(result)
            if not records:
                logger.warning(f"No data found for correlation calculation on {date}")
                return
                
            df = pd.DataFrame(records)
            returns_pivot = df.pivot(index='date', columns='symbol', values='returns')
            
            # Calculate correlations
            corr_matrix = returns_pivot.corr()
            
            # Create correlation edges
            edges = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.3 and not np.isnan(corr_val):
                        edges.append({
                            'symbol1': corr_matrix.columns[i],
                            'symbol2': corr_matrix.columns[j],
                            'correlation': float(corr_val),
                            'date': date.isoformat()
                        })
                        
            # Load to Neo4j
            if edges:
                session.run("""
                    UNWIND $edges AS edge
                    MATCH (s1:Security {symbol: edge.symbol1})
                    MATCH (s2:Security {symbol: edge.symbol2})
                    MERGE (s1)-[r:CORRELATES_WITH {date: edge.date}]->(s2)
                    SET r.correlation = edge.correlation,
                        r.abs_correlation = abs(edge.correlation)
                """, edges=edges)
                
                logger.info(f"Created {len(edges)} correlation edges for {date}")
                
    def process_s3_file(self, bucket: str, key: str, file_type: str = 'parquet'):
        """Process a file from S3"""
        logger.info(f"Processing s3://{bucket}/{key}")
        
        try:
            # Download file
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            
            if file_type == 'parquet':
                df = pd.read_parquet(obj['Body'])
            elif file_type == 'csv':
                df = pd.read_csv(obj['Body'])
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            # Determine data type and process
            if 'close' in df.columns and 'symbol' in df.columns:
                # Price data
                futures = []
                for i in range(0, len(df), self.config.batch_size):
                    batch = df.iloc[i:i+self.config.batch_size]
                    future = self.process_price_batch.remote(batch)
                    futures.append(future)
                    
                # Gather results
                results = ray.get(futures)
                
                # Load to Neo4j
                for result in results:
                    if result['metrics']['error_rate'] < self.config.error_threshold:
                        self.load_price_data_to_neo4j(result['data'])
                        
            elif 'exposure_amount' in df.columns:
                # Exposure data
                clean_df, metrics = self.validator.validate_exposure_data(df)
                if metrics.get('error_rate', 1) < self.config.error_threshold:
                    self.load_exposure_data_to_neo4j(clean_df)
                    
            # Update metrics
            self.pipeline_metrics['processed_records'] += len(df)
            self.pipeline_metrics['last_run'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to process {key}: {e}")
            self.pipeline_metrics['failed_records'] += 1
            
    def load_exposure_data_to_neo4j(self, df: pd.DataFrame):
        """Load exposure data into Neo4j"""
        with self.driver.session() as session:
            # Ensure entities exist
            all_entities = (
                df[['source_entity']].rename(columns={'source_entity': 'entity_id'})
                .append(df[['target_entity']].rename(columns={'target_entity': 'entity_id'}))
                .drop_duplicates()
            )
            
            session.run("""
                UNWIND $entities AS ent
                MERGE (e:Entity {entity_id: ent.entity_id})
                SET e.last_updated = datetime()
            """, entities=all_entities.to_dict('records'))
            
            # Create exposures
            session.run("""
                UNWIND $exposures AS exp
                MATCH (source:Entity {entity_id: exp.source_entity})
                MATCH (target:Entity {entity_id: exp.target_entity})
                MERGE (source)-[r:EXPOSURE {date: exp.date}]->(target)
                SET r.amount = exp.exposure_amount,
                    r.exposure_type = exp.exposure_type,
                    r.updated_at = datetime()
            """, exposures=df.to_dict('records'))
            
    def run_daily_pipeline(self):
        """Run complete daily ETL pipeline"""
        logger.info("Starting daily ETL pipeline")
        start_time = time.time()
        
        try:
            # 1. Process new files from S3
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.s3_bucket,
                Prefix=f"data/{datetime.now().strftime('%Y%m%d')}/"
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    self.process_s3_file(self.config.s3_bucket, obj['Key'])
                    
            # 2. Build correlation graphs
            self.build_correlation_graphs(datetime.now())
            
            # 3. Run graph algorithms
            with self.driver.session() as session:
                # Update graph projections
                session.run("CALL gds.graph.drop('security-correlation', false)")
                session.run("""
                    CALL gds.graph.project.cypher(
                        'security-correlation',
                        'MATCH (n:Security) RETURN id(n) AS id',
                        'MATCH (s1:Security)-[r:CORRELATES_WITH]->(s2:Security)
                         WHERE r.date = date()
                         RETURN id(s1) AS source, id(s2) AS target, 
                                abs(r.correlation) AS weight'
                    )
                """)
                
            # 4. Send completion notification
            self.kafka_producer.send('etl-events', {
                'event': 'pipeline_completed',
                'timestamp': datetime.now().isoformat(),
                'metrics': self.pipeline_metrics
            })
            
            processing_time = time.time() - start_time
            logger.info(f"Daily pipeline completed in {processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.kafka_producer.send('etl-events', {
                'event': 'pipeline_failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
    def schedule_pipelines(self):
        """Schedule recurring pipeline runs"""
        # Daily price data pipeline
        schedule.every().day.at("06:00").do(self.run_daily_pipeline)
        
        # Hourly correlation updates
        schedule.every().hour.do(
            lambda: self.build_correlation_graphs(datetime.now())
        )
        
        # Real-time trade processing
        schedule.every(5).minutes.do(self.process_trade_queue)
        
        logger.info("ETL pipelines scheduled")
        
        while True:
            schedule.run_pending()
            time.sleep(60)
            
    def process_trade_queue(self):
        """Process queued trades from Kafka"""
        # This would connect to the trade flow topic
        # and process trades in near real-time
        pass
        
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status and metrics"""
        with self.driver.session() as session:
            db_stats = session.run("""
                MATCH (s:Security) WITH count(s) AS securities
                MATCH (p:PricePoint) WITH securities, count(p) AS prices
                MATCH ()-[c:CORRELATES_WITH]->() WITH securities, prices, count(c) AS correlations
                MATCH (e:Entity) WITH securities, prices, correlations, count(e) AS entities
                MATCH ()-[x:EXPOSURE]->() 
                RETURN securities, prices, correlations, entities, count(x) AS exposures
            """).single()
            
        return {
            'pipeline_metrics': self.pipeline_metrics,
            'database_stats': dict(db_stats) if db_stats else {},
            'last_update': datetime.now().isoformat(),
            'status': 'healthy' if self.pipeline_metrics.get('error_rate', 0) < 0.05 else 'degraded'
        }
        
    def close(self):
        """Clean up resources"""
        self.driver.close()
        self.kafka_producer.close()
        ray.shutdown()