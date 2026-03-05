"""
Relationship Extractor - Week 9 Day 2
Extract relationships between entities using dependency parsing.
"""

import spacy
from typing import List, Tuple, Dict, Set
from src.graph.entity_extractor import Entity
import re

class Relationship:
    """Represents a relationship between two entities."""
    
    def __init__(
        self,
        source: str,
        relation: str,
        target: str,
        source_label: str = None,
        target_label: str = None,
        confidence: float = 1.0
    ):
        self.source = source.lower().strip()
        self.relation = relation.lower().strip()
        self.target = target.lower().strip()
        self.source_label = source_label
        self.target_label = target_label
        self.confidence = confidence
    
    def __repr__(self):
        return f"({self.source}) --[{self.relation}]--> ({self.target})"
    
    def to_tuple(self) -> Tuple[str, str, str]:
        """Convert to (source, relation, target) tuple."""
        return (self.source, self.relation, self.target)


class RelationshipExtractor:
    """
    Extract relationships between entities using dependency parsing.
    
    Extraction methods:
    1. Verb-based: subject-verb-object patterns
    2. Preposition-based: entity-prep-entity patterns
    3. Pattern-based: predefined relationship patterns
    """
    
    def __init__(self, model: str = "en_core_web_md"):
        """Initialize with spaCy model."""
        self.nlp = spacy.load(model)
        
        # Verb patterns that indicate relationships
        self.relationship_verbs = {
            'use', 'enable', 'implement', 'support', 'provide',
            'include', 'contain', 'require', 'depend', 'cause',
            'improve', 'enhance', 'reduce', 'increase', 'affect',
            'integrate', 'combine', 'connect', 'link', 'relate'
        }
        
        # Prepositions that indicate relationships
        self.relationship_preps = {
            'for', 'with', 'in', 'on', 'by', 'through',
            'via', 'using', 'from', 'to', 'into'
        }
    
    def extract_from_sentence(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Extract relationships using hybrid approach (3 methods).
        
        Args:
            text: Sentence text
            entities: Entities in the sentence
        
        Returns:
            List of Relationship objects
        """
        if len(entities) < 2:
            return []  # Need at least 2 entities
        
        doc = self.nlp(text)
        relationships = []
        
        # Create entity lookup
        entity_texts = {e.normalized: e for e in entities}
        
        # ========== METHOD 1: CO-OCCURRENCE (BASELINE) ==========
        # Always works, provides baseline connectivity
        cooccur_rels = self.extract_cooccurrence(entities)
        relationships.extend(cooccur_rels)
        
        # ========== METHOD 2: PATTERN-BASED (SPECIFIC) ==========
        # High precision for known patterns
        pattern_rels = self.extract_patterns(text, entity_texts)
        relationships.extend(pattern_rels)
        
        # ========== METHOD 3: DEPENDENCY PARSING (FLEXIBLE) ==========
        # Catches various grammatical patterns
        svo_rels = self._extract_svo_patterns(doc, entity_texts)
        relationships.extend(svo_rels)
        
        prep_rels = self._extract_prep_patterns(doc, entity_texts)
        relationships.extend(prep_rels)
        
        # Deduplicate (keep highest confidence)
        return self._deduplicate_by_confidence(relationships)


    def _deduplicate_by_confidence(
        self,
        relationships: List[Relationship]
    ) -> List[Relationship]:
        """
        Deduplicate relationships, keeping highest confidence.
        """
        # Group by (source, target) pair
        from collections import defaultdict
        grouped = defaultdict(list)
        
        for rel in relationships:
            key = (rel.source, rel.target)
            grouped[key].append(rel)
        
        # Keep best relationship for each pair
        unique = []
        for rels in grouped.values():
            # Sort by confidence (highest first)
            best = sorted(rels, key=lambda r: r.confidence, reverse=True)[0]
            unique.append(best)
        
        return unique

    def extract_cooccurrence(self, entities: List[Entity]) -> List[Relationship]:
        """
        Method 1: Co-occurrence based relationships.
        Connect all entities that appear together.
        """
        relationships = []
        
        # Connect every pair of entities
        for i, ent1 in enumerate(entities):
            for ent2 in entities[i+1:]:
                rel = Relationship(
                    source=ent1.normalized,
                    relation='related_to',
                    target=ent2.normalized,
                    source_label=ent1.label,
                    target_label=ent2.label,
                    confidence=0.5  # Low confidence (generic)
                )
                relationships.append(rel)
        
        return relationships


    def extract_patterns(self, text: str, entity_texts: Dict[str, Entity]) -> List[Relationship]:
        """
        Method 2: Pattern-based extraction using regex.
        Matches specific relationship patterns.
        """
        import re
        relationships = []
        
        # Define patterns for common relationships
        patterns = {
            'uses': r'(\w+(?:\s+\w+)?)\s+(?:use|uses|using)\s+(\w+(?:\s+\w+)?)',
            'enables': r'(\w+(?:\s+\w+)?)\s+(?:enable|enables|enabling)\s+(\w+(?:\s+\w+)?)',
            'improves': r'(\w+(?:\s+\w+)?)\s+(?:improve|improves|improving)\s+(\w+(?:\s+\w+)?)',
            'implements': r'(\w+(?:\s+\w+)?)\s+(?:implement|implements|implementing)\s+(\w+(?:\s+\w+)?)',
            'requires': r'(\w+(?:\s+\w+)?)\s+(?:require|requires|requiring)\s+(\w+(?:\s+\w+)?)',
            'causes': r'(\w+(?:\s+\w+)?)\s+(?:cause|causes|causing)\s+(\w+(?:\s+\w+)?)',
            'provides': r'(\w+(?:\s+\w+)?)\s+(?:provide|provides|providing)\s+(\w+(?:\s+\w+)?)',
        }
        
        # Match each pattern
        for rel_type, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                source_text = match.group(1).lower().strip()
                target_text = match.group(2).lower().strip()
                
                # Check if both are known entities
                if source_text in entity_texts and target_text in entity_texts:
                    source_entity = entity_texts[source_text]
                    target_entity = entity_texts[target_text]
                    
                    rel = Relationship(
                        source=source_entity.normalized,
                        relation=rel_type,
                        target=target_entity.normalized,
                        source_label=source_entity.label,
                        target_label=target_entity.label,
                        confidence=0.8  # High confidence (explicit pattern)
                    )
                    relationships.append(rel)
        
        return relationships    

    def _extract_svo_patterns(
        self,
        doc,
        entity_texts: Dict[str, Entity]
    ) -> List[Relationship]:
        """Extract Subject-Verb-Object relationships."""
        relationships = []
        
        for token in doc:
            # Find verbs that indicate relationships
            if token.pos_ == 'VERB' and token.lemma_ in self.relationship_verbs:
                # Find subject
                subject = None
                for child in token.children:
                    if child.dep_ in ('nsubj', 'nsubjpass'):
                        subject = self._get_entity_span(child, entity_texts)
                        break
                
                # Find object
                obj = None
                for child in token.children:
                    if child.dep_ in ('dobj', 'pobj', 'attr'):
                        obj = self._get_entity_span(child, entity_texts)
                        break
                
                # Create relationship if both found
                if subject and obj:
                    rel = Relationship(
                        source=subject.text,
                        relation=token.lemma_,
                        target=obj.text,
                        source_label=subject.label,
                        target_label=obj.label
                    )
                    relationships.append(rel)
        
        return relationships
    
    def _extract_prep_patterns(
        self,
        doc,
        entity_texts: Dict[str, Entity]
    ) -> List[Relationship]:
        """Extract preposition-based relationships."""
        relationships = []
        
        for token in doc:
            # Find prepositions
            if token.dep_ == 'prep' and token.text.lower() in self.relationship_preps:
                # Get head entity
                head_entity = self._get_entity_span(token.head, entity_texts)
                
                # Get object of preposition
                for child in token.children:
                    if child.dep_ == 'pobj':
                        obj_entity = self._get_entity_span(child, entity_texts)
                        
                        if head_entity and obj_entity:
                            rel = Relationship(
                                source=head_entity.text,
                                relation=token.text,
                                target=obj_entity.text,
                                source_label=head_entity.label,
                                target_label=obj_entity.label
                            )
                            relationships.append(rel)
        
        return relationships
    
    def _get_entity_span(self, token, entity_texts: Dict[str, Entity]) -> Entity:
        """Get entity that contains this token."""
        # Check if token itself is an entity
        token_text = token.text.lower().strip()
        if token_text in entity_texts:
            return entity_texts[token_text]
        
        # Check compound spans
        # Get full noun phrase
        if token.pos_ in ('NOUN', 'PROPN'):
            # Collect compound
            span_tokens = [token]
            
            # Get left compounds
            for left in token.lefts:
                if left.dep_ in ('compound', 'amod'):
                    span_tokens.insert(0, left)
            
            # Get right compounds
            for right in token.rights:
                if right.dep_ in ('compound', 'amod'):
                    span_tokens.append(right)
            
            # Build span text
            span_text = ' '.join([t.text for t in span_tokens]).lower().strip()
            
            if span_text in entity_texts:
                return entity_texts[span_text]
        
        return None
    
    def _extract_pattern_based(
        self,
        text: str,
        entity_texts: Dict[str, Entity]
    ) -> List[Relationship]:
        """Extract relationships using regex patterns."""
        relationships = []
        
        # Pattern: X uses Y
        pattern_uses = r'(\w+(?:\s+\w+)?)\s+(?:use|uses|using)\s+(\w+(?:\s+\w+)?)'
        for match in re.finditer(pattern_uses, text, re.IGNORECASE):
            source, target = match.group(1).lower(), match.group(2).lower()
            if source in entity_texts and target in entity_texts:
                rel = Relationship(source, 'uses', target)
                relationships.append(rel)
        
        # Pattern: X enables Y
        pattern_enables = r'(\w+(?:\s+\w+)?)\s+(?:enable|enables|enabling)\s+(\w+(?:\s+\w+)?)'
        for match in re.finditer(pattern_enables, text, re.IGNORECASE):
            source, target = match.group(1).lower(), match.group(2).lower()
            if source in entity_texts and target in entity_texts:
                rel = Relationship(source, 'enables', target)
                relationships.append(rel)
        
        # Pattern: X improves Y
        pattern_improves = r'(\w+(?:\s+\w+)?)\s+(?:improve|improves|improving)\s+(\w+(?:\s+\w+)?)'
        for match in re.finditer(pattern_improves, text, re.IGNORECASE):
            source, target = match.group(1).lower(), match.group(2).lower()
            if source in entity_texts and target in entity_texts:
                rel = Relationship(source, 'improves', target)
                relationships.append(rel)
        
        # Pattern: X for Y (simple co-occurrence)
        pattern_for = r'(\w+(?:\s+\w+)?)\s+for\s+(\w+(?:\s+\w+)?)'
        for match in re.finditer(pattern_for, text, re.IGNORECASE):
            source, target = match.group(1).lower(), match.group(2).lower()
            if source in entity_texts and target in entity_texts:
                rel = Relationship(source, 'for', target)
                relationships.append(rel)
        
        return relationships
    
    def extract_from_chunks(
        self,
        chunks: List[any],
        chunk_entities: Dict[str, List[Entity]]
    ) -> List[Relationship]:
        """
        Extract relationships from multiple chunks.
        
        Args:
            chunks: List of Chunk objects
            chunk_entities: Dict mapping chunk_id to entities
        
        Returns:
            List of all relationships
        """
        all_relationships = []
        
        for chunk in chunks:
            entities = chunk_entities.get(chunk.chunk_id, [])
            
            if len(entities) < 2:
                continue  # Need at least 2 entities for a relationship
            
            # Split into sentences
            doc = self.nlp(chunk.text)
            for sent in doc.sents:
                # Filter entities in this sentence
                sent_entities = [
                    e for e in entities
                    if e.start >= sent.start_char and e.end <= sent.end_char
                ]
                
                if len(sent_entities) >= 2:
                    rels = self.extract_from_sentence(sent.text, sent_entities)
                    all_relationships.extend(rels)
        
        return all_relationships
    
    def deduplicate_relationships(
        self,
        relationships: List[Relationship]
    ) -> List[Relationship]:
        """Remove duplicate relationships."""
        seen = set()
        unique = []
        
        for rel in relationships:
            key = rel.to_tuple()
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        
        return unique
    