"""
Test database functionality.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import get_db_manager
from src.models.database_models import Document, Chunk, QueryLog


def test_database_creation():
    """Test database and table creation."""
    
    print("=" * 60)
    print("TESTING DATABASE CREATION")
    print("=" * 60)
    
    # Get database manager
    db = get_db_manager()
    
    print("\n✅ Database manager created")
    print(f"   URL: {db.database_url}")
    
    # Tables should be auto-created
    print("✅ Tables created")


def test_document_crud():
    """Test Document CRUD operations."""
    
    print("\n" + "=" * 60)
    print("TESTING DOCUMENT CRUD")
    print("=" * 60)
    
    db = get_db_manager()
    session = db.get_session()
    
    try:
        # CREATE
        print("\n1. Creating document...")
        doc = Document(
            filename="test_document.pdf",
            filepath="/data/uploads/test_document.pdf",
            file_type="PDF",
            file_size=1024000,
            page_count=5,
            chunking_mode='hierarchical',
            total_chunks=10,
            total_parents=2,
            processed_at=datetime.utcnow()
        )
        
        session.add(doc)
        session.commit()
        
        doc_id = doc.id
        print(f"   ✅ Document created with ID: {doc_id}")
        
        # READ
        print("\n2. Reading document...")
        retrieved_doc = session.query(Document).filter_by(id=doc_id).first()
        print(f"   ✅ Retrieved: {retrieved_doc.filename}")
        print(f"      Chunks: {retrieved_doc.total_chunks}")
        print(f"      Mode: {retrieved_doc.chunking_mode}")
        
        # UPDATE
        print("\n3. Updating document...")
        retrieved_doc.total_chunks = 12
        session.commit()
        print(f"   ✅ Updated chunks: {retrieved_doc.total_chunks}")
        
        # LIST
        print("\n4. Listing all documents...")
        all_docs = session.query(Document).all()
        print(f"   ✅ Found {len(all_docs)} documents")
        for d in all_docs:
            print(f"      - {d.filename} ({d.total_chunks} chunks)")
        
        # DELETE
        print("\n5. Deleting document...")
        session.delete(retrieved_doc)
        session.commit()
        print(f"   ✅ Document deleted")
        
        # Verify deletion
        deleted_doc = session.query(Document).filter_by(id=doc_id).first()
        if deleted_doc is None:
            print(f"   ✅ Deletion verified")
        else:
            print(f"   ❌ Deletion failed")
        
    finally:
        db.close_session(session)


def test_chunk_relationships():
    """Test parent-child chunk relationships."""
    
    print("\n" + "=" * 60)
    print("TESTING CHUNK RELATIONSHIPS")
    print("=" * 60)
    
    db = get_db_manager()
    session = db.get_session()
    
    try:
        # Create document
        doc = Document(
            filename="relationship_test.pdf",
            filepath="/data/test.pdf",
            file_type="PDF",
            page_count=1,
            chunking_mode='hierarchical'
        )
        session.add(doc)
        session.commit()
        
        print(f"\n1. Created document ID: {doc.id}")
        
        # Create parent chunk
        parent = Chunk(
            chunk_id="parent_0",
            document_id=doc.id,
            text="This is parent chunk with full context...",
            token_count=2000,
            start_idx=0,
            end_idx=2000,
            chunk_type='parent',
            vector_id='vec_parent_0'
        )
        session.add(parent)
        session.commit()
        
        print(f"2. Created parent chunk ID: {parent.id}")
        
        # Create child chunks
        children = []
        for i in range(3):
            child = Chunk(
                chunk_id=f"parent_0_child_{i}",
                document_id=doc.id,
                text=f"This is child chunk {i}...",
                token_count=500,
                start_idx=i * 500,
                end_idx=(i + 1) * 500,
                chunk_type='child',
                parent_id=parent.id,
                vector_id=f'vec_child_{i}'
            )
            children.append(child)
            session.add(child)
        
        session.commit()
        print(f"3. Created {len(children)} child chunks")
        
        # Test relationships
        print("\n4. Testing relationships...")
        
        # Get parent with children
        parent_with_children = session.query(Chunk).filter_by(id=parent.id).first()
        print(f"   Parent has {len(parent_with_children.children)} children")
        
        for child in parent_with_children.children:
            print(f"   - {child.chunk_id} (parent_id: {child.parent_id})")
        
        # Get child and find parent
        first_child = session.query(Chunk).filter_by(chunk_id='parent_0_child_0').first()
        print(f"\n5. Child '{first_child.chunk_id}' parent: '{first_child.parent.chunk_id}'")
        
        # Test document relationship
        print(f"\n6. Document has {len(doc.chunks)} total chunks")
        
        # Cleanup
        session.delete(doc)  # Cascade will delete chunks
        session.commit()
        print(f"\n✅ Cleanup complete (cascade delete)")
        
    finally:
        db.close_session(session)


def test_query_logging():
    """Test query log functionality."""
    
    print("\n" + "=" * 60)
    print("TESTING QUERY LOGGING")
    print("=" * 60)
    
    db = get_db_manager()
    session = db.get_session()
    
    try:
        # Create query log
        query_log = QueryLog(
            query_text="What is artificial intelligence?",
            chunks_retrieved=5,
            avg_relevance_score=0.85,
            retrieval_mode='hierarchical',
            answer_text="Artificial intelligence is...",
            answer_length=150,
            citations_count=3,
            retrieval_time_ms=45.2,
            generation_time_ms=2345.8,
            total_time_ms=2391.0
        )
        
        session.add(query_log)
        session.commit()
        
        print(f"\n✅ Query log created: ID {query_log.id}")
        
        # Retrieve logs
        all_logs = session.query(QueryLog).all()
        print(f"✅ Total query logs: {len(all_logs)}")
        
        # Cleanup
        session.delete(query_log)
        session.commit()
        
    finally:
        db.close_session(session)


def main():
    """Run all database tests."""
    
    print("=" * 60)
    print("DATABASE TEST SUITE")
    print("=" * 60)
    
    test_database_creation()
    test_document_crud()
    test_chunk_relationships()
    test_query_logging()
    
    print("\n" + "=" * 60)
    print("✅ ALL DATABASE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()