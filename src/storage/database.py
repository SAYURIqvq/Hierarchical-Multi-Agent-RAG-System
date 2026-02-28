"""
Database connection and session management.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from src.models.database_models import Base


class DatabaseManager:
    """Manage database connections and sessions."""
    
    def __init__(self, database_url: str = None):
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection string
                         Default: SQLite for development
        """
        if database_url is None:
            # Default: SQLite in data directory
            db_path = "data/agentic_rag.db"
            os.makedirs("data", exist_ok=True)
            database_url = f"sqlite:///{db_path}"
        
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            echo=False,  # Set True for SQL logging
            pool_pre_ping=True  # Check connections
        )
        
        # Session factory
        session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(session_factory)
        
        print(f"💾 Database initialized: {database_url}")
    
    def create_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        print("✅ Database tables created")
    
    def drop_tables(self):
        """Drop all tables (careful!)."""
        Base.metadata.drop_all(self.engine)
        print("⚠️  Database tables dropped")
    
    def get_session(self):
        """Get a new database session."""
        return self.Session()
    
    def close_session(self, session):
        """Close a database session."""
        session.close()
    
    def cleanup(self):
        """Cleanup all sessions."""
        self.Session.remove()


# Global database instance (singleton pattern)
_db_manager = None


def get_db_manager(database_url: str = None) -> DatabaseManager:
    """
    Get database manager singleton.
    
    Args:
        database_url: Database connection string
        
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager(database_url)
        _db_manager.create_tables()
    
    return _db_manager