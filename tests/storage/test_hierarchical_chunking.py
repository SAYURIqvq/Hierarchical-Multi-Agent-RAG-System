"""
Test hierarchical chunking functionality.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # ← ADD THIS

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.hierarchical_chunker import HierarchicalChunker


def test_hierarchical_chunking():
    """Test hierarchical chunking with sample text."""
    
    print("=" * 60)
    print("TESTING HIERARCHICAL CHUNKING")
    print("=" * 60)
    
    # Sample text (repeat to get enough tokens)
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    
    Machine learning is a subset of artificial intelligence that focuses on 
    the development of algorithms that can learn from and make predictions on data.
    Deep learning is a subset of machine learning that uses neural networks with 
    multiple layers.
    
    Natural language processing (NLP) is a branch of AI that helps computers 
    understand, interpret and manipulate human language. NLP draws from many 
    disciplines, including computer science and computational linguistics.
    """ * 20  # Repeat to get ~4000+ tokens
    
    # Create chunker
    chunker = HierarchicalChunker(
        parent_size=2000,
        child_size=500,
        child_overlap=50
    )
    
    # Count tokens
    token_count = chunker.count_tokens(sample_text)
    print(f"\nSample text: {token_count:,} tokens")
    
    # Create chunks
    parent_chunks, child_chunks = chunker.chunk_text(sample_text)
    
    # Display results
    print(f"\n📊 RESULTS:")
    print(f"   Parents: {len(parent_chunks)}")
    print(f"   Children: {len(child_chunks)}")
    print(f"   Ratio: {len(child_chunks)/len(parent_chunks):.1f} children/parent")
    
    # Show first parent and its children
    if parent_chunks:
        parent = parent_chunks[0]
        print(f"\n📦 First Parent Chunk:")
        print(f"   ID: {parent.chunk_id}")
        print(f"   Tokens: {parent.token_count}")
        print(f"   Children: {len(parent.children_ids)}")
        print(f"   Text preview: {parent.text[:100]}...")
        
        print(f"\n👶 Its Children:")
        for child_id in parent.children_ids[:3]:  # Show first 3
            child = next(c for c in child_chunks if c.chunk_id == child_id)
            print(f"   • {child.chunk_id}: {child.token_count} tokens")
            print(f"     Preview: {child.text[:80]}...")
    
    # Test parent retrieval
    print(f"\n🔍 Testing Parent Retrieval:")
    if child_chunks:
        test_child = child_chunks[2]  # Pick a middle child
        parent = chunker.get_parent_context(test_child, parent_chunks)
        
        print(f"   Child: {test_child.chunk_id}")
        print(f"   Parent: {parent.chunk_id if parent else 'Not found'}")
        
        if parent:
            print(f"   ✅ Parent retrieval working!")
        else:
            print(f"   ❌ Parent not found")
    
    # Verify relationships
    print(f"\n🔗 Verifying Relationships:")
    all_linked = True
    for child in child_chunks:
        if child.parent_id:
            parent_exists = any(p.chunk_id == child.parent_id for p in parent_chunks)
            if not parent_exists:
                print(f"   ❌ Child {child.chunk_id} has invalid parent_id")
                all_linked = False
    
    if all_linked:
        print(f"   ✅ All children properly linked to parents")
    
    print("\n" + "=" * 60)
    print("✅ HIERARCHICAL CHUNKING TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_hierarchical_chunking()