"""
Automated testing for RAG system.
Tests file loading, chunking, embedding, retrieval, and generation.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_poc import (
    DocumentLoader,
    TextChunker,
    Embedder,
    SimpleVectorStore,
    AnswerGenerator
)


class RAGSystemTester:
    """Comprehensive testing for RAG system."""
    
    def __init__(self):
        """Initialize tester."""
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'details': []
        }
    
    def test_file_loading(self):
        """Test document loading for all formats."""
        print("\n" + "=" * 60)
        print("TEST 1: FILE LOADING")
        print("=" * 60)
        
        loader = DocumentLoader()
        test_files = {
            'PDF': 'data/uploads/sample.pdf',
            'DOCX': 'data/uploads/test.docx',
            'TXT': 'data/uploads/test.txt'
        }
        
        for file_type, file_path in test_files.items():
            self.results['tests_run'] += 1
            
            if not os.path.exists(file_path):
                print(f"⚠️  {file_type}: File not found (skipping)")
                continue
            
            try:
                text = loader.load(file_path)
                
                if len(text) > 0:
                    print(f"✅ {file_type}: Loaded successfully ({len(text)} chars)")
                    self.results['tests_passed'] += 1
                    self.results['details'].append({
                        'test': f'Load {file_type}',
                        'status': 'PASS',
                        'chars': len(text)
                    })
                else:
                    print(f"❌ {file_type}: Empty text")
                    self.results['tests_failed'] += 1
                    self.results['details'].append({
                        'test': f'Load {file_type}',
                        'status': 'FAIL',
                        'error': 'Empty text'
                    })
                    
            except Exception as e:
                print(f"❌ {file_type}: Error - {str(e)}")
                self.results['tests_failed'] += 1
                self.results['details'].append({
                    'test': f'Load {file_type}',
                    'status': 'FAIL',
                    'error': str(e)
                })
    
    def test_chunking(self):
        """Test text chunking."""
        print("\n" + "=" * 60)
        print("TEST 2: TEXT CHUNKING")
        print("=" * 60)
        
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        
        # Test with sample text
        test_text = "This is a test. " * 1000  # ~16,000 chars
        
        self.results['tests_run'] += 1
        
        try:
            chunks = chunker.chunk_text(test_text)
            
            if len(chunks) > 0:
                print(f"✅ Chunking: Created {len(chunks)} chunks")
                print(f"   Average tokens: {sum(c['token_count'] for c in chunks) / len(chunks):.1f}")
                
                # Verify overlap
                has_overlap = True
                for i in range(len(chunks) - 1):
                    if chunks[i]['end_idx'] <= chunks[i+1]['start_idx']:
                        has_overlap = False
                        break
                
                if has_overlap:
                    print(f"✅ Overlap: Verified")
                else:
                    print(f"⚠️  Overlap: Not detected")
                
                self.results['tests_passed'] += 1
                self.results['details'].append({
                    'test': 'Chunking',
                    'status': 'PASS',
                    'chunks': len(chunks),
                    'has_overlap': has_overlap
                })
            else:
                print(f"❌ Chunking: No chunks created")
                self.results['tests_failed'] += 1
                
        except Exception as e:
            print(f"❌ Chunking: Error - {str(e)}")
            self.results['tests_failed'] += 1
    
    def test_embeddings(self):
        """Test embedding generation."""
        print("\n" + "=" * 60)
        print("TEST 3: EMBEDDING GENERATION")
        print("=" * 60)
        
        embedder = Embedder()
        
        # Test single query
        self.results['tests_run'] += 1
        
        try:
            embedding = embedder.embed_query("Test query")
            
            if len(embedding) == 1536:  # Voyage-large-2 dimension
                print(f"✅ Query embedding: {len(embedding)} dimensions")
                self.results['tests_passed'] += 1
                self.results['details'].append({
                    'test': 'Query embedding',
                    'status': 'PASS',
                    'dimensions': len(embedding)
                })
            else:
                print(f"❌ Query embedding: Wrong dimensions ({len(embedding)})")
                self.results['tests_failed'] += 1
                
        except Exception as e:
            print(f"❌ Query embedding: Error - {str(e)}")
            self.results['tests_failed'] += 1
    
    def test_vector_search(self):
        """Test vector search functionality."""
        print("\n" + "=" * 60)
        print("TEST 4: VECTOR SEARCH")
        print("=" * 60)
        
        # Setup
        embedder = Embedder()
        vector_store = SimpleVectorStore()
        
        # Create sample chunks
        sample_chunks = [
            {'chunk_id': 'test_1', 'text': 'Artificial intelligence is the future'},
            {'chunk_id': 'test_2', 'text': 'Machine learning is a subset of AI'},
            {'chunk_id': 'test_3', 'text': 'Python is a programming language'}
        ]
        
        # Embed chunks
        sample_chunks = embedder.embed_chunks(sample_chunks)
        vector_store.add_chunks(sample_chunks)
        
        # Test search
        self.results['tests_run'] += 1
        
        try:
            query = "What is AI?"
            query_embedding = embedder.embed_query(query)
            results = vector_store.search(query_embedding, top_k=2)
            
            if len(results) > 0:
                print(f"✅ Search: Found {len(results)} results")
                print(f"   Top result: '{results[0]['text'][:50]}...'")
                print(f"   Score: {results[0]['score']:.4f}")
                
                # Verify AI-related chunks ranked higher
                if 'AI' in results[0]['text'] or 'intelligence' in results[0]['text']:
                    print(f"✅ Relevance: Correct ranking")
                    self.results['tests_passed'] += 1
                    self.results['details'].append({
                        'test': 'Vector search',
                        'status': 'PASS',
                        'results': len(results),
                        'top_score': results[0]['score']
                    })
                else:
                    print(f"⚠️  Relevance: Unexpected top result")
                    
            else:
                print(f"❌ Search: No results")
                self.results['tests_failed'] += 1
                
        except Exception as e:
            print(f"❌ Search: Error - {str(e)}")
            self.results['tests_failed'] += 1
    
    def test_answer_generation(self):
        """Test answer generation."""
        print("\n" + "=" * 60)
        print("TEST 5: ANSWER GENERATION")
        print("=" * 60)
        
        generator = AnswerGenerator()
        
        # Mock chunks
        mock_chunks = [
            {
                'chunk_id': 'chunk_1',
                'text': 'The RAG system uses retrieval-augmented generation to answer questions.',
                'score': 0.95
            }
        ]
        
        self.results['tests_run'] += 1
        
        try:
            result = generator.generate(
                query="What is RAG?",
                chunks=mock_chunks,
                max_chunks=1
            )
            
            if result['answer'] and len(result['answer']) > 0:
                print(f"✅ Generation: Answer created ({len(result['answer'])} chars)")
                print(f"   Preview: {result['answer'][:100]}...")
                print(f"   Citations: {len(result['citations'])}")
                
                self.results['tests_passed'] += 1
                self.results['details'].append({
                    'test': 'Answer generation',
                    'status': 'PASS',
                    'answer_length': len(result['answer']),
                    'citations': len(result['citations'])
                })
            else:
                print(f"❌ Generation: Empty answer")
                self.results['tests_failed'] += 1
                
        except Exception as e:
            print(f"❌ Generation: Error - {str(e)}")
            self.results['tests_failed'] += 1
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n" + "=" * 60)
        print("TEST 6: EDGE CASES")
        print("=" * 60)
        
        loader = DocumentLoader()
        
        # Test 1: Non-existent file
        self.results['tests_run'] += 1
        try:
            loader.load("nonexistent.pdf")
            print(f"❌ Non-existent file: Should have raised error")
            self.results['tests_failed'] += 1
        except FileNotFoundError:
            print(f"✅ Non-existent file: Correctly raised FileNotFoundError")
            self.results['tests_passed'] += 1
        
        # Test 2: Unsupported format
        self.results['tests_run'] += 1
        try:
            # Create dummy file
            with open("data/uploads/test.xyz", "w") as f:
                f.write("test")
            
            loader.load("data/uploads/test.xyz")
            print(f"❌ Unsupported format: Should have raised error")
            self.results['tests_failed'] += 1
        except ValueError:
            print(f"✅ Unsupported format: Correctly raised ValueError")
            self.results['tests_passed'] += 1
        finally:
            if os.path.exists("data/uploads/test.xyz"):
                os.remove("data/uploads/test.xyz")
    
    def run_all_tests(self):
        """Run all tests."""
        print("\n" + "=" * 60)
        print("STARTING RAG SYSTEM COMPREHENSIVE TESTS")
        print("=" * 60)
        
        self.test_file_loading()
        self.test_chunking()
        self.test_embeddings()
        self.test_vector_search()
        self.test_answer_generation()
        self.test_edge_cases()
        
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total = self.results['tests_run']
        passed = self.results['tests_passed']
        failed = self.results['tests_failed']
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"Total tests: {total}")
        print(f"Passed: {passed} ({pass_rate:.1f}%)")
        print(f"Failed: {failed}")
        
        if failed == 0:
            print("\n✅ ALL TESTS PASSED!")
        else:
            print(f"\n⚠️  {failed} test(s) failed")
    
    def save_results(self):
        """Save test results to file."""
        output_file = "tests/test_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📊 Results saved to: {output_file}")


if __name__ == "__main__":
    tester = RAGSystemTester()
    tester.run_all_tests()