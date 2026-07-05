"""
Neo4j driver singleton for knowledge graph operations.
All Cypher queries use parameterized inputs to prevent injection.
"""
from neo4j import GraphDatabase
from app.core.config import get_settings

_driver = None


def get_neo4j_driver():
    global _driver
    if _driver is None:
        settings = get_settings()
        _driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    return _driver


def close_neo4j_driver():
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def run_cypher(query: str, params: dict = None):
    """Execute a read Cypher query and return records."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]


def write_cypher(query: str, params: dict = None):
    """Execute a write Cypher query."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run(query, params or {})


def init_neo4j_schema():
    """Create indexes and constraints on first startup."""
    queries = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
        "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.course_id)",
        "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.course_id)",
        "CREATE INDEX IF NOT EXISTS FOR (p:PYQ) ON (p.course_id)",
    ]
    driver = get_neo4j_driver()
    with driver.session() as session:
        for q in queries:
            try:
                session.run(q)
            except Exception:
                pass  # Index may already exist
