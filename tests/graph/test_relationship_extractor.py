"""Test Hybrid Relationship Extractor - Week 9 Day 2"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graph.entity_extractor import EntityExtractor
from src.graph.relationship_extractor import RelationshipExtractor


def test_hybrid_extraction():
    """Test hybrid relationship extraction."""
    
    print("\n🧪 Testing HYBRID Relationship Extractor\n")
    
    # Initialize
    entity_extractor = EntityExtractor()
    rel_extractor = RelationshipExtractor()
    
    # Test sentences
    sentences = [
        "Google uses TensorFlow for machine learning applications.",
        "Python enables rapid development of AI systems.",
        "Machine learning improves prediction accuracy.",
        "Facebook implements neural networks in their platform."
    ]
    
    total_rels = 0
    
    for i, sent in enumerate(sentences, 1):
        print(f"{'='*60}")
        print(f"📝 Sentence {i}: {sent}")
        
        # Extract entities
        entities = entity_extractor.extract(sent)
        print(f"   Entities ({len(entities)}): {[e.text for e in entities]}")
        
        if len(entities) < 2:
            print(f"   ⚠️  Need 2+ entities for relationships")
            print()
            continue
        
        # Extract relationships
        rels = rel_extractor.extract_from_sentence(sent, entities)
        
        if rels:
            print(f"   Relationships ({len(rels)}):")
            for rel in rels:
                confidence_emoji = "🟢" if rel.confidence > 0.7 else "🟡" if rel.confidence > 0.5 else "⚪"
                print(f"      {confidence_emoji} {rel} (confidence: {rel.confidence:.1f})")
            total_rels += len(rels)
        else:
            print(f"   ❌ No relationships found")
        
        print()
    
    # Summary
    print(f"{'='*60}")
    print(f"📊 SUMMARY:")
    print(f"   Total sentences: {len(sentences)}")
    print(f"   Total relationships: {total_rels}")
    print(f"   Average per sentence: {total_rels/len(sentences):.1f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_hybrid_extraction()