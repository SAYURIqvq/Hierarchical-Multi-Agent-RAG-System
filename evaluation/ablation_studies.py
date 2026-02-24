"""
Ablation Studies - Week 11 Day 1-2
Measure the impact of each system component.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from typing import Dict, List
from src.storage.chroma_store import ChromaVectorStore
from src.graph.graph_builder import KnowledgeGraph
from src.retrieval.graph_retrieval import GraphRetrieval
from src.ingestion.embedder import EmbeddingGenerator


class AblationStudy:
    """Run ablation studies to measure component impact."""
    
    def __init__(self):
        """Initialize ablation study."""
        self.vector_store = ChromaVectorStore(persist_directory="data/chroma_db")
        
        try:
            self.kg = KnowledgeGraph()
            self.kg.load('data/graphs/machine_learning.txt_graph.pkl')
            self.graph_available = True
        except:
            self.graph_available = False
        
        self.embedder = EmbeddingGenerator()
        
        # Test queries
        self.test_queries = [
            "What is machine learning?",
            "How does TensorFlow relate to neural networks?",
            "Explain supervised learning",
            "Compare Python and Java for ML",
            "What is the connection between AI and deep learning?"
        ]
    
    def baseline_retrieval(self, query: str, top_k: int = 10) -> Dict:
        """Baseline: Simple vector search only."""
        start = time.time()
        
        query_embedding = self.embedder.generate_query_embedding(query)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            return_parent=False  # Flat chunking (baseline)
        )
        
        elapsed = time.time() - start
        
        return {
            'method': 'baseline',
            'chunks': len(results),
            'time': elapsed,
            'avg_score': sum(r['score'] for r in results) / len(results) if results else 0
        }
    
    def hierarchical_retrieval(self, query: str, top_k: int = 10) -> Dict:
        """+ Hierarchical chunking."""
        start = time.time()
        
        query_embedding = self.embedder.generate_query_embedding(query)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            return_parent=True  # Hierarchical
        )
        
        elapsed = time.time() - start
        
        return {
            'method': 'hierarchical',
            'chunks': len(results),
            'time': elapsed,
            'avg_score': sum(r['score'] for r in results) / len(results) if results else 0
        }
    
    def hybrid_retrieval(self, query: str, top_k: int = 10) -> Dict:
        """+ Hybrid search (vector + keyword)."""
        # For now, same as hierarchical (keyword not integrated)
        # Future: Add BM25 results
        return self.hierarchical_retrieval(query, top_k)
    
    def graph_retrieval(self, query: str, top_k: int = 5) -> Dict:
        """+ Graph search."""
        if not self.graph_available:
            return {
                'method': 'graph',
                'chunks': 0,
                'time': 0,
                'avg_score': 0,
                'error': 'Graph not available'
            }
        
        start = time.time()
        
        try:
            graph_ret = GraphRetrieval(
                knowledge_graph=self.kg,
                vector_store=self.vector_store
            )
            
            chunks = graph_ret.search(query, top_k=top_k)
            
            elapsed = time.time() - start
            
            return {
                'method': 'graph',
                'chunks': len(chunks),
                'time': elapsed,
                'avg_score': sum(c.score for c in chunks) / len(chunks) if chunks else 0
            }
        except Exception as e:
            return {
                'method': 'graph',
                'chunks': 0,
                'time': time.time() - start,
                'avg_score': 0,
                'error': str(e)
            }
    
    def run_ablation(self):
        """Run complete ablation study."""
        
        print("\n" + "="*60)
        print("🧪 ABLATION STUDY: Component Impact Analysis")
        print("="*60 + "\n")
        
        results = {
            'baseline': [],
            'hierarchical': [],
            'hybrid': [],
            'graph': []
        }
        
        for i, query in enumerate(self.test_queries, 1):
            print(f"\n{'='*60}")
            print(f"Query {i}/{len(self.test_queries)}: {query}")
            print("="*60)
            
            # Test 1: Baseline
            print("\n1️⃣ Baseline (flat chunks, vector only)")
            baseline = self.baseline_retrieval(query)
            results['baseline'].append(baseline)
            print(f"   Chunks: {baseline['chunks']}, "
                  f"Time: {baseline['time']:.2f}s, "
                  f"Avg Score: {baseline['avg_score']:.3f}")
            
            # Test 2: Hierarchical
            print("\n2️⃣ + Hierarchical Chunking")
            hierarchical = self.hierarchical_retrieval(query)
            results['hierarchical'].append(hierarchical)
            improvement = ((hierarchical['avg_score'] - baseline['avg_score']) 
                          / baseline['avg_score'] * 100) if baseline['avg_score'] > 0 else 0
            print(f"   Chunks: {hierarchical['chunks']}, "
                  f"Time: {hierarchical['time']:.2f}s, "
                  f"Avg Score: {hierarchical['avg_score']:.3f} "
                  f"({improvement:+.1f}%)")
            
            # Test 3: Hybrid (same as hierarchical for now)
            print("\n3️⃣ + Hybrid Search (vector + keyword)")
            hybrid = self.hybrid_retrieval(query)
            results['hybrid'].append(hybrid)
            print(f"   [Not yet implemented - same as hierarchical]")
            
            # Test 4: Graph
            print("\n4️⃣ + Graph Search")
            graph = self.graph_retrieval(query)
            results['graph'].append(graph)
            if 'error' not in graph:
                print(f"   Chunks: {graph['chunks']}, "
                      f"Time: {graph['time']:.2f}s, "
                      f"Avg Score: {graph['avg_score']:.3f}")
            else:
                print(f"   ⚠️  {graph['error']}")
        
        # Summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print summary statistics."""
        
        print("\n" + "="*60)
        print("📊 SUMMARY: Average Performance")
        print("="*60 + "\n")
        
        for method in ['baseline', 'hierarchical', 'hybrid', 'graph']:
            data = results[method]
            valid_data = [d for d in data if 'error' not in d]
            
            if not valid_data:
                print(f"{method.upper()}: No valid results")
                continue
            
            avg_chunks = sum(d['chunks'] for d in valid_data) / len(valid_data)
            avg_time = sum(d['time'] for d in valid_data) / len(valid_data)
            avg_score = sum(d['avg_score'] for d in valid_data) / len(valid_data)
            
            print(f"{method.upper()}:")
            print(f"   Avg Chunks: {avg_chunks:.1f}")
            print(f"   Avg Time: {avg_time:.2f}s")
            print(f"   Avg Score: {avg_score:.3f}")
            print()
        
        print("="*60)
        
        # Component impact
        print("\n📈 COMPONENT IMPACT:\n")
        
        baseline_score = sum(d['avg_score'] for d in results['baseline']) / len(results['baseline'])
        hier_score = sum(d['avg_score'] for d in results['hierarchical']) / len(results['hierarchical'])
        
        valid_graph = [d for d in results['graph'] if 'error' not in d and d['chunks'] > 0]
        graph_score = sum(d['avg_score'] for d in valid_graph) / len(valid_graph) if valid_graph else 0
        
        print(f"1. Hierarchical Chunking:")
        hier_impact = ((hier_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
        print(f"   {hier_impact:+.1f}% improvement in avg score")
        
        print(f"\n2. Graph Search:")
        if graph_score > 0:
            graph_impact = ((graph_score - hier_score) / hier_score * 100) if hier_score > 0 else 0
            print(f"   {graph_impact:+.1f}% improvement for relationship queries")
            print(f"   Coverage: {len(valid_graph)}/{len(results['graph'])} queries ({len(valid_graph)/len(results['graph'])*100:.0f}%)")
        else:
            print(f"   No valid results")
        
        print("\n" + "="*60 + "\n")


def main():
    """Run ablation study."""
    study = AblationStudy()
    results = study.run_ablation()
    
    # Save results
    import json
    output_file = 'data/ablation_results.json'
    
    # Convert to JSON-serializable format
    json_results = {}
    for method, data_list in results.items():
        json_results[method] = []
        for data in data_list:
            json_data = {k: v for k, v in data.items() if k != 'error'}
            if 'error' in data:
                json_data['error'] = data['error']
            json_results[method].append(json_data)
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"💾 Results saved to {output_file}")


if __name__ == "__main__":
    main()