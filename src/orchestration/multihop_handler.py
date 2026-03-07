"""
Multi-hop Query Handler - Week 5 Day 5
Processes decomposed queries and aggregates results.
"""

from typing import List
from src.models.agent_state import AgentState, Chunk
from src.agents.synthesis import SynthesisAgent


class MultiHopHandler:
    """Handle multi-hop query processing."""
    
    def __init__(self):
        self.synthesis = SynthesisAgent()
    
    def process_sub_queries(
        self,
        sub_queries: List[str],
        vector_store,
        embedder,
        top_k: int = 5
    ) -> List[Chunk]:
        """
        Process each sub-query and aggregate results.
        
        Args:
            sub_queries: List of decomposed sub-questions
            vector_store: Vector store instance
            embedder: Embedder instance
            top_k: Chunks per sub-query
        
        Returns:
            Aggregated and deduplicated chunks
        """
        all_chunks = []
        
        print(f"\n🔀 Processing {len(sub_queries)} sub-queries...")
        
        for i, sub_q in enumerate(sub_queries, 1):
            print(f"   {i}. {sub_q[:50]}...")
            
            # Generate embedding
            embedding = embedder.generate_query_embedding(sub_q)
            
            # Search
            results = vector_store.search(
                query_embedding=embedding,
                top_k=top_k,
                return_parent=True
            )
            
            # Convert to Chunks
            for result in results:
                chunk = Chunk(
                    text=result['text'],
                    doc_id='unknown',
                    chunk_id=result['chunk_id'],
                    score=result['score'],
                    metadata={
                        'filename': result.get('metadata', {}).get('filename', 'uploaded_document'),
                        'sub_query_index': i,
                        'sub_query': sub_q,
                        **result.get('metadata', {})
                    }
                )
                all_chunks.append(chunk)
        
        print(f"✅ Retrieved {len(all_chunks)} total chunks")
        
        # Deduplicate and rank
        state = AgentState(query="aggregate", chunks=all_chunks)
        state = self.synthesis.run(state)
        
        print(f"✅ After deduplication: {len(state.chunks)} unique chunks")
        
        return state.chunks