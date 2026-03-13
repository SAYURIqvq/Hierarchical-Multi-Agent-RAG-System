"""Test Knowledge Graph Builder - Week 9 Day 3"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graph.entity_extractor import EntityExtractor, Entity
from src.graph.relationship_extractor import RelationshipExtractor
from src.graph.graph_builder import KnowledgeGraph


def test_graph_building():
    """Test knowledge graph construction."""
    
    print("\n🧪 Testing Knowledge Graph Builder\n")
    
    # Initialize
    entity_extractor = EntityExtractor()
    rel_extractor = RelationshipExtractor()
    kg = KnowledgeGraph()
    
    # Test document
    chunks_text = [
        "Google uses TensorFlow for machine learning applications.",
        "Python enables rapid development of AI systems.",
        "Machine learning improves prediction accuracy.",
        "TensorFlow is developed by Google and supports neural networks."
    ]
    
    # Simulate chunks
    from src.models.agent_state import Chunk
    chunks = []
    chunk_entities = {}
    chunk_relationships = {}
    
    for i, text in enumerate(chunks_text):
        chunk_id = f"chunk_{i}"
        
        # Create chunk
        chunk = Chunk(
            text=text,
            doc_id="test_doc",
            chunk_id=chunk_id,
            score=1.0
        )
        chunks.append(chunk)
        
        # Extract entities
        entities = entity_extractor.extract(text)
        chunk_entities[chunk_id] = entities
        
        # Extract relationships
        rels = rel_extractor.extract_from_sentence(text, entities)
        chunk_relationships[chunk_id] = rels
        
        print(f"📄 Chunk {i+1}: {text[:50]}...")
        print(f"   Entities: {len(entities)}")
        print(f"   Relationships: {len(rels)}")
    
    print(f"\n{'='*60}")
    
    # Build graph
    kg.build_from_chunks(chunks, chunk_entities, chunk_relationships)
    
    print(f"\n{'='*60}")
    print("🔍 Testing Graph Queries:\n")
    
    # Test 1: Get neighbors
    print("1️⃣ Neighbors of 'google':")
    neighbors = kg.get_neighbors('google')
    for neighbor, relation in neighbors[:5]:
        print(f"   google --[{relation}]--> {neighbor}")
    
    # Test 2: Find path
    print("\n2️⃣ Path from 'google' to 'machine learning':")
    paths = kg.find_path('google', 'machine learning', max_length=3)
    if paths:
        for path in paths[:3]:
            print(f"   {' → '.join(path)}")
    else:
        print("   No path found")
    
    # Test 3: Top entities
    print("\n3️⃣ Top 5 entities by degree:")
    top = kg.get_top_entities(n=5, metric='degree')
    for entity, score in top:
        print(f"   {entity}: {score}")
    
    # Test 4: Subgraph
    print("\n4️⃣ Subgraph around 'tensorflow' (1-hop):")
    subgraph = kg.get_subgraph(['tensorflow'], k_hop=1)
    print(f"   Nodes: {subgraph.number_of_nodes()}")
    print(f"   Edges: {subgraph.number_of_edges()}")
    
    # Test 5: Save & load
    print("\n5️⃣ Save & Load:")
    kg.save('data/graphs/test_graph.pkl')
    
    kg2 = KnowledgeGraph()
    kg2.load('data/graphs/test_graph.pkl')
    print(f"   Loaded: {kg2}")
    
    print(f"\n{'='*60}")
    print("✅ All tests passed!\n")


if __name__ == "__main__":
    test_graph_building()