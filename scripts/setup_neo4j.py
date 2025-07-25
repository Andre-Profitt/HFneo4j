#!/usr/bin/env python3
"""Initialize Neo4j database with schema and constraints"""
import os
import sys
import time
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "hedgefund123!")


def wait_for_neo4j(driver, max_retries=30):
    """Wait for Neo4j to be ready"""
    for i in range(max_retries):
        try:
            with driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j is ready!")
            return True
        except Exception as e:
            logger.info(f"Waiting for Neo4j... ({i+1}/{max_retries})")
            time.sleep(2)
    return False


def create_constraints_and_indexes(driver):
    """Create necessary constraints and indexes"""
    constraints = [
        # Security constraints
        "CREATE CONSTRAINT security_symbol IF NOT EXISTS FOR (s:Security) REQUIRE s.symbol IS UNIQUE",
        "CREATE CONSTRAINT security_cusip IF NOT EXISTS FOR (s:Security) REQUIRE s.cusip IS UNIQUE",
        
        # Entity constraints (for contagion analysis)
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
        
        # Trade constraints
        "CREATE CONSTRAINT trade_id IF NOT EXISTS FOR (t:Trade) REQUIRE t.trade_id IS UNIQUE",
        
        # Factor model constraints
        "CREATE CONSTRAINT factor_model_date IF NOT EXISTS FOR (fm:FactorModel) REQUIRE (fm.date, fm.type) IS UNIQUE",
    ]
    
    indexes = [
        # Time-based indexes
        "CREATE INDEX security_date IF NOT EXISTS FOR (s:Security) ON (s.last_updated)",
        "CREATE INDEX correlation_date IF NOT EXISTS FOR ()-[r:CORRELATES_WITH]-() ON (r.date)",
        "CREATE INDEX exposure_date IF NOT EXISTS FOR ()-[r:EXPOSURE]-() ON (r.date)",
        
        # Performance indexes
        "CREATE INDEX security_sector IF NOT EXISTS FOR (s:Security) ON (s.sector)",
        "CREATE INDEX security_market_cap IF NOT EXISTS FOR (s:Security) ON (s.market_cap)",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        
        # Trade indexes
        "CREATE INDEX trade_time IF NOT EXISTS FOR (t:Trade) ON (t.timestamp)",
        "CREATE INDEX trade_symbol IF NOT EXISTS FOR (t:Trade) ON (t.symbol)",
        
        # Text search
        "CREATE FULLTEXT INDEX security_search IF NOT EXISTS FOR (s:Security) ON EACH [s.name, s.symbol, s.description]",
    ]
    
    with driver.session() as session:
        # Create constraints
        for constraint in constraints:
            try:
                session.run(constraint)
                logger.info(f"Created constraint: {constraint.split('CONSTRAINT')[1].split('IF')[0].strip()}")
            except Exception as e:
                logger.warning(f"Constraint might already exist: {e}")
                
        # Create indexes
        for index in indexes:
            try:
                session.run(index)
                logger.info(f"Created index: {index.split('INDEX')[1].split('IF')[0].strip()}")
            except Exception as e:
                logger.warning(f"Index might already exist: {e}")


def create_graph_projections(driver):
    """Create common graph projections for GDS"""
    projections = [
        {
            'name': 'security-correlation',
            'node_query': 'MATCH (n:Security) RETURN id(n) AS id',
            'relationship_query': '''
                MATCH (s1:Security)-[r:CORRELATES_WITH]->(s2:Security)
                RETURN id(s1) AS source, id(s2) AS target, 
                       r.correlation AS correlation,
                       abs(r.correlation) AS abs_correlation
            '''
        },
        {
            'name': 'entity-exposure',
            'node_query': '''
                MATCH (n:Entity) 
                RETURN id(n) AS id, n.total_assets AS assets
            ''',
            'relationship_query': '''
                MATCH (e1:Entity)-[r:EXPOSURE]->(e2:Entity)
                RETURN id(e1) AS source, id(e2) AS target,
                       r.amount AS weight
            '''
        }
    ]
    
    with driver.session() as session:
        for proj in projections:
            try:
                # Drop if exists
                session.run(f"CALL gds.graph.drop('{proj['name']}', false)")
            except:
                pass
                
            # Create projection
            session.run(f"""
                CALL gds.graph.project.cypher(
                    '{proj['name']}',
                    "{proj['node_query']}",
                    "{proj['relationship_query']}"
                )
            """)
            logger.info(f"Created graph projection: {proj['name']}")


def initialize_sample_data(driver):
    """Load initial sample securities and entities"""
    with driver.session() as session:
        # Create sample securities (S&P 500 leaders)
        securities = [
            {'symbol': 'AAPL', 'name': 'Apple Inc.', 'sector': 'Technology', 'market_cap': 3000000000000},
            {'symbol': 'MSFT', 'name': 'Microsoft Corp.', 'sector': 'Technology', 'market_cap': 2800000000000},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'sector': 'Technology', 'market_cap': 1800000000000},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'sector': 'Consumer Discretionary', 'market_cap': 1700000000000},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corp.', 'sector': 'Technology', 'market_cap': 1200000000000},
            {'symbol': 'META', 'name': 'Meta Platforms', 'sector': 'Technology', 'market_cap': 900000000000},
            {'symbol': 'BRK.B', 'name': 'Berkshire Hathaway', 'sector': 'Financials', 'market_cap': 800000000000},
            {'symbol': 'JPM', 'name': 'JPMorgan Chase', 'sector': 'Financials', 'market_cap': 500000000000},
            {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'sector': 'Healthcare', 'market_cap': 450000000000},
            {'symbol': 'UNH', 'name': 'UnitedHealth Group', 'sector': 'Healthcare', 'market_cap': 500000000000},
        ]
        
        session.run("""
            UNWIND $securities AS sec
            MERGE (s:Security {symbol: sec.symbol})
            SET s.name = sec.name,
                s.sector = sec.sector,
                s.market_cap = sec.market_cap,
                s.last_updated = datetime()
        """, securities=securities)
        logger.info(f"Created {len(securities)} sample securities")
        
        # Create sample entities (funds, prime brokers)
        entities = [
            {'entity_id': 'HF001', 'name': 'Alpha Capital', 'type': 'HedgeFund', 'assets': 5000000000},
            {'entity_id': 'HF002', 'name': 'Quantum Partners', 'type': 'HedgeFund', 'assets': 8000000000},
            {'entity_id': 'HF003', 'name': 'Sigma Strategies', 'type': 'HedgeFund', 'assets': 3000000000},
            {'entity_id': 'PB001', 'name': 'Goldman Sachs PB', 'type': 'PrimeBroker', 'assets': 50000000000},
            {'entity_id': 'PB002', 'name': 'Morgan Stanley PB', 'type': 'PrimeBroker', 'assets': 45000000000},
            {'entity_id': 'CCP001', 'name': 'CME Clearing', 'type': 'CCP', 'assets': 100000000000},
        ]
        
        session.run("""
            UNWIND $entities AS ent
            MERGE (e:Entity {entity_id: ent.entity_id})
            SET e.name = ent.name,
                e.type = ent.type,
                e.total_assets = ent.assets,
                e.active = true,
                e.created_at = datetime()
        """, entities=entities)
        logger.info(f"Created {len(entities)} sample entities")


def main():
    """Main setup function"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        # Wait for Neo4j
        if not wait_for_neo4j(driver):
            logger.error("Neo4j is not available!")
            sys.exit(1)
            
        # Create schema
        logger.info("Creating constraints and indexes...")
        create_constraints_and_indexes(driver)
        
        # Initialize sample data
        logger.info("Loading sample data...")
        initialize_sample_data(driver)
        
        # Create graph projections
        logger.info("Creating graph projections...")
        create_graph_projections(driver)
        
        logger.info("Neo4j setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)
    finally:
        driver.close()


if __name__ == "__main__":
    main()