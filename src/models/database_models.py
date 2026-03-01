"""
Database models for metadata storage.
Uses SQLAlchemy ORM for database operations.
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, 
    ForeignKey, Boolean, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Document(Base):
    """Document metadata table."""
    
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(512), nullable=False)
    file_type = Column(String(10), nullable=False)  # PDF, DOCX, TXT
    file_size = Column(Integer)  # bytes
    page_count = Column(Integer)
    
    # Chunking info
    chunking_mode = Column(String(20), default='flat')  # flat, hierarchical
    total_chunks = Column(Integer, default=0)
    total_parents = Column(Integer, default=0)
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', chunks={self.total_chunks})>"


class Chunk(Base):
    """Chunk metadata table (both parent and child)."""
    
    __tablename__ = 'chunks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(100), unique=True, nullable=False)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    
    # Chunk content
    text = Column(Text, nullable=False)
    token_count = Column(Integer)
    start_idx = Column(Integer)
    end_idx = Column(Integer)
    
    # Hierarchical info
    chunk_type = Column(String(10), nullable=False)  # parent, child
    parent_id = Column(Integer, ForeignKey('chunks.id'), nullable=True)
    
    # Vector info (reference to ChromaDB)
    vector_id = Column(String(100), unique=True)  # ID in ChromaDB
    embedding_model = Column(String(50), default='voyage-large-2')
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    parent = relationship("Chunk", remote_side=[id], backref="children")
    
    def __repr__(self):
        return f"<Chunk(id={self.id}, chunk_id='{self.chunk_id}', type='{self.chunk_type}')>"


class QueryLog(Base):
    """Query log for analytics and learning."""
    
    __tablename__ = 'query_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_text = Column(Text, nullable=False)
    
    # Retrieval info
    chunks_retrieved = Column(Integer)
    avg_relevance_score = Column(Float)
    retrieval_mode = Column(String(20))  # flat, hierarchical
    
    # Generation info
    answer_text = Column(Text)
    answer_length = Column(Integer)
    citations_count = Column(Integer)
    
    # Performance
    retrieval_time_ms = Column(Float)
    generation_time_ms = Column(Float)
    total_time_ms = Column(Float)
    
    # Feedback (for future learning)
    user_feedback = Column(String(20))  # thumbs_up, thumbs_down
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<QueryLog(id={self.id}, query='{self.query_text[:50]}...')>"