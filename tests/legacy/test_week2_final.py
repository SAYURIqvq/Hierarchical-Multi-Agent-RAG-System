"""
Week 2 Final System Test
Comprehensive validation of all Week 2 features.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_poc import DocumentLoader, Embedder, AnswerGenerator
from src.ingestion.hierarchical_chunker import HierarchicalChunker
from src.storage.chroma_store import ChromaVectorStore
from src.storage.database import get_db_manager
from src.evaluation.simple_evaluator import SimpleEvaluator


def test_complete_system():
    """Test complete Week 2 system."""
    
    print("=" * 60)
    print("WEEK 2 FINAL SYSTEM TEST")
    print("=" * 60)
    
    results = {}
    
    # 1. Test Hierarchical Chunking
    print("\n1️⃣ Testing Hierarchical Chunking...")
    try:
        chunker = HierarchicalChunker(parent_size=1000, child_size=300)
        text = "Artificial intelligence " * 100
        parents, children = chunker.chunk_text(text)
        
        assert len(parents) > 0, "No parent chunks created"
        assert len(children) > 0, "No child chunks created"
        assert all(c.parent_id for c in children), "Children missing parent_id"
        
        results['hierarchical_chunking'] = '✅ PASS'
        print(f"   ✅ Parents: {len(parents)}, Children: {len(children)}")
    except Exception as e:
        results['hierarchical_chunking'] = f'❌ FAIL: {e}'
        print(f"   ❌ Error: {e}")
    
    # 2. Test ChromaDB Persistence
    print("\n2️⃣ Testing ChromaDB Persistence...")
    try:
        store = ChromaVectorStore(persist_directory="data/test_week2_final")
        embedder = Embedder()
        
        # Add test data
        from src.ingestion.hierarchical_chunker import Chunk
        parent = Chunk(
            chunk_id="test_parent",
            text="Test parent chunk",
            tokens=[],
            token_count=100,
            start_idx=0,
            end_idx=100,
            chunk_type='parent'
        )
        parent.embedding = embedder.embed_query(parent.text)
        
        child = Chunk(
            chunk_id="test_child",
            text="Test child chunk",
            tokens=[],
            token_count=50,
            start_idx=0,
            end_idx=50,
            chunk_type='child',
            parent_id="test_parent"
        )
        child.embedding = embedder.embed_query(child.text)
        
        store.add_chunks([parent], [child])
        
        # Test persistence
        store2 = ChromaVectorStore(persist_directory="data/test_week2_final")
        stats = store2.get_stats()
        
        assert stats['total_vectors'] == 2, "Persistence failed"
        
        # Cleanup
        store2.clear_all()
        
        results['chromadb_persistence'] = '✅ PASS'
        print(f"   ✅ Persistence verified: {stats['total_vectors']} vectors")
    except Exception as e:
        results['chromadb_persistence'] = f'❌ FAIL: {e}'
        print(f"   ❌ Error: {e}")
    
    # 3. Test SQLite Database
    print("\n3️⃣ Testing SQLite Database...")
    try:
        from src.models.database_models import Document, Chunk as DBChunk
        
        db = get_db_manager()
        session = db.get_session()
        
        # Create test document
        doc = Document(
            filename="test.pdf",
            filepath="/test/test.pdf",
            file_type="PDF",
            page_count=1,
            chunking_mode='hierarchical',
            total_chunks=2,
            total_parents=1
        )
        session.add(doc)
        session.commit()
        
        doc_id = doc.id
        
        # Verify
        retrieved = session.query(Document).filter_by(id=doc_id).first()
        assert retrieved is not None, "Document not found"
        assert retrieved.chunking_mode == 'hierarchical', "Mode incorrect"
        
        # Cleanup
        session.delete(retrieved)
        session.commit()
        db.close_session(session)
        
        results['sqlite_database'] = '✅ PASS'
        print(f"   ✅ Database CRUD working")
    except Exception as e:
        results['sqlite_database'] = f'❌ FAIL: {e}'
        print(f"   ❌ Error: {e}")
    
    # 4. Test Evaluation Framework
    print("\n4️⃣ Testing Evaluation Framework...")
    try:
        evaluator = SimpleEvaluator()
        
        scores = evaluator.evaluate_single(
            question="What is AI?",
            answer="AI is artificial intelligence, the simulation of human intelligence by machines.",
            contexts=["Artificial intelligence is intelligence demonstrated by machines."],
            ground_truth="AI is the simulation of human intelligence."
        )
        
        assert 'relevancy' in scores, "Missing relevancy score"
        assert 'faithfulness' in scores, "Missing faithfulness score"
        assert 'completeness' in scores, "Missing completeness score"
        assert 'overall' in scores, "Missing overall score"
        assert all(0 <= s <= 1 for s in scores.values()), "Invalid score range"
        
        results['evaluation_framework'] = '✅ PASS'
        print(f"   ✅ All metrics working: {scores['overall']:.3f} overall")
    except Exception as e:
        results['evaluation_framework'] = f'❌ FAIL: {e}'
        print(f"   ❌ Error: {e}")
    
    # 5. Performance Benchmark
    print("\n5️⃣ Performance Benchmark...")
    try:
        store = ChromaVectorStore(persist_directory="data/test_perf")
        embedder = Embedder()
        
        # Add test vectors
        from src.ingestion.hierarchical_chunker import Chunk
        chunks = []
        for i in range(10):
            chunk = Chunk(
                chunk_id=f"chunk_{i}",
                text=f"Test chunk {i} with some content",
                tokens=[],
                token_count=10,
                start_idx=0,
                end_idx=10,
                chunk_type='child'
            )
            chunk.embedding = embedder.embed_query(chunk.text)
            chunks.append(chunk)
        
        store.add_chunks([], chunks)
        
        # Benchmark search
        query_emb = embedder.embed_query("test query")
        
        times = []
        for _ in range(10):
            start = time.time()
            store.search(query_emb, top_k=5)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times) * 1000  # ms
        
        # Cleanup
        store.clear_all()
        
        assert avg_time < 100, f"Search too slow: {avg_time:.1f}ms"
        
        results['performance'] = f'✅ PASS ({avg_time:.1f}ms)'
        print(f"   ✅ Search speed: {avg_time:.1f}ms average")
    except Exception as e:
        results['performance'] = f'❌ FAIL: {e}'
        print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test, result in results.items():
        print(f"{test.replace('_', ' ').title():30} {result}")
    
    passed = sum(1 for r in results.values() if '✅' in r)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"OVERALL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print(f"{'='*60}")
    
    if passed == total:
        print("\n🎉 ALL WEEK 2 FEATURES WORKING!")
        print("✅ System is production-ready")
    else:
        print("\n⚠️  Some tests failed - review above")
    
    return passed == total


if __name__ == "__main__":
    success = test_complete_system()
    exit(0 if success else 1)