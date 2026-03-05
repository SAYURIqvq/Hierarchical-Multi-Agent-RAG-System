"""
Entity Extractor - Week 9 Day 1
Extract named entities from text using spaCy NER.
"""

import spacy
from typing import List, Dict, Set, Tuple
from collections import defaultdict


class Entity:
    """Represents an extracted entity."""
    
    def __init__(self, text: str, label: str, start: int, end: int):
        self.text = text
        self.label = label
        self.start = start
        self.end = end
        self.normalized = self._normalize(text)
    
    def _normalize(self, text: str) -> str:
        """Normalize entity text (lowercase, strip)."""
        return text.strip().lower()
    
    def __repr__(self):
        return f"Entity({self.text}, {self.label})"


class EntityExtractor:
    """
    Extract entities from text using spaCy.
    
    Extracts:
    - PERSON: People, authors
    - ORG: Organizations, companies
    - PRODUCT: Products, technologies
    - CONCEPT: Key concepts, methods
    - GPE: Countries, cities
    """
    
    def __init__(self, model: str = "en_core_web_md"):
        """Initialize with spaCy model."""
        self.nlp = spacy.load(model)
        
        # Entity types to extract
        self.target_labels = {
            'PERSON', 'ORG', 'PRODUCT', 'GPE',
            'WORK_OF_ART', 'LAW', 'LANGUAGE',
            'NORP', 'FAC', 'EVENT'
        }
        
        # ========== ADD CUSTOM TECH PATTERNS ==========
        # Known tech terms often missed by NER
        self.tech_terms = {
            # Languages
            'python', 'java', 'javascript', 'c++', 'rust', 'go',
            
            # Frameworks
            'tensorflow', 'pytorch', 'keras', 'react', 'angular', 'vue',
            'django', 'flask', 'fastapi', 'spring', 'express',
            
            # Cloud/Infra
            'docker', 'kubernetes', 'aws', 'gcp', 'azure',
            
            # ML/AI Concepts
            'machine learning', 'deep learning', 'ai', 'artificial intelligence',
            'neural network', 'neural networks', 'nlp', 'computer vision',
            'reinforcement learning', 'supervised learning', 'unsupervised learning',
            
            # General Tech Concepts  
            'development', 'programming', 'software', 'application', 'system',
            'platform', 'framework', 'library', 'algorithm', 'model',
            'accuracy', 'performance', 'scalability', 'deployment',
            'api', 'database', 'architecture', 'microservices'
        }
    def extract(self, text: str) -> List[Entity]:
        """Extract entities with custom tech term detection."""
        
        doc = self.nlp(text)
        
        entities = []
        
        # Standard NER entities
        for ent in doc.ents:
            if ent.label_ in self.target_labels:
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char
                )
                entities.append(entity)
        
        # ========== ADD CUSTOM TECH TERMS ==========
        # Find tech terms manually
        text_lower = text.lower()
        for term in self.tech_terms:
            if term in text_lower:
                # Find position
                start = text_lower.find(term)
                if start != -1:
                    entity = Entity(
                        text=term.title(),
                        label='TECH',  # Custom label
                        start=start,
                        end=start + len(term)
                    )
                    entities.append(entity)
        # ========== END CUSTOM ==========
        
        return entities
    
    def extract_from_chunks(
        self,
        chunks: List[any]
    ) -> Dict[str, List[Entity]]:
        """
        Extract entities from multiple chunks.
        
        Args:
            chunks: List of Chunk objects
        
        Returns:
            Dict mapping chunk_id to entities
        """
        chunk_entities = {}
        
        for chunk in chunks:
            entities = self.extract(chunk.text)
            chunk_entities[chunk.chunk_id] = entities
        
        return chunk_entities
    
    def deduplicate_entities(
        self,
        entities: List[Entity]
    ) -> Dict[str, Set[str]]:
        """
        Group and deduplicate entities by type.
        
        Returns:
            Dict mapping label to set of unique entity texts
        """
        grouped = defaultdict(set)
        
        for entity in entities:
            grouped[entity.label].add(entity.normalized)
        
        return dict(grouped)
    
    def get_entity_frequency(
        self,
        entities: List[Entity]
    ) -> Dict[Tuple[str, str], int]:
        """
        Count entity occurrences.
        
        Returns:
            Dict mapping (text, label) to frequency
        """
        freq = defaultdict(int)
        
        for entity in entities:
            key = (entity.normalized, entity.label)
            freq[key] += 1
        
        return dict(freq)