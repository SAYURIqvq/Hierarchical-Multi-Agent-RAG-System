"""
Embedding Generator - Generate embeddings using Voyage AI.

Uses Voyage AI API for high-quality semantic embeddings.
Supports batch processing for efficiency.
"""

from typing import List, Dict, Any
import time

import voyageai

from src.config import get_settings
from src.utils.logger import setup_logger
from src.utils.exceptions import AgenticRAGException


class EmbeddingError(AgenticRAGException):
    """Error during embedding generation."""
    pass


class EmbeddingGenerator:
    """
    Generate embeddings using Voyage AI.
    
    Features:
    - Batch processing for efficiency
    - Automatic retry on failure
    - Rate limiting handling
    - Caching support
    
    Example:
        >>> generator = EmbeddingGenerator()
        >>> texts = ["Hello world", "Another text"]
        >>> embeddings = generator.generate(texts)
        >>> print(len(embeddings))  # 2
        >>> print(len(embeddings[0]))  # 1536 (embedding dimension)
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        batch_size: int = None
    ):
        """
        Initialize embedding generator.
        
        Args:
            api_key: Voyage AI API key (default from config)
            model: Model name (default from config)
            batch_size: Batch size for processing (default from config)
        """
        self.logger = setup_logger("embedder")
        settings = get_settings()
        
        self.api_key = api_key or settings.voyage_api_key
        self.model = model or settings.embedding_model
        self.batch_size = batch_size or settings.batch_size
        
        # Initialize Voyage AI client
        try:
            self.client = voyageai.Client(api_key=self.api_key)
            self.logger.info(f"Initialized Voyage AI client with model: {self.model}")
        except Exception as e:
            raise EmbeddingError(
                message=f"Failed to initialize Voyage AI client: {str(e)}",
                details={"error": str(e)}
            )
        
        # Statistics
        self.total_embeddings = 0
        self.total_tokens = 0
    
    def generate(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Processes in batches for efficiency.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors (each is List[float])
        
        Raises:
            EmbeddingError: If embedding generation fails
        
        Example:
            >>> texts = ["Text 1", "Text 2", "Text 3"]
            >>> embeddings = generator.generate(texts)
            >>> len(embeddings) == len(texts)  # True
        """
        if not texts:
            return []
        
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            
            self.logger.debug(
                f"Processing batch {batch_num}/{total_batches} "
                f"({len(batch)} texts)"
            )
            
            try:
                batch_embeddings = self._generate_batch(batch)
                all_embeddings.extend(batch_embeddings)
                
                self.total_embeddings += len(batch)
                
            except Exception as e:
                self.logger.error(f"Batch {batch_num} failed: {str(e)}")
                raise EmbeddingError(
                    message=f"Failed to generate embeddings for batch {batch_num}: {str(e)}",
                    details={
                        "batch_num": batch_num,
                        "batch_size": len(batch),
                        "error": str(e)
                    }
                ) from e
        
        self.logger.info(
            f"Generated {len(all_embeddings)} embeddings "
            f"(total: {self.total_embeddings})"
        )
        
        return all_embeddings
    
    def _generate_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: Batch of text strings
        
        Returns:
            List of embedding vectors
        """
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Call Voyage AI API
                result = self.client.embed(
                    texts=texts,
                    model=self.model,
                    input_type="document"  # For indexing/retrieval
                )
                
                # Extract embeddings from result
                embeddings = result.embeddings
                
                # Validate embeddings
                if len(embeddings) != len(texts):
                    raise EmbeddingError(
                        message=f"Expected {len(texts)} embeddings, got {len(embeddings)}",
                        details={"expected": len(texts), "got": len(embeddings)}
                    )
                
                return embeddings
                
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {retry_delay}s: {str(e)}"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Uses 'query' input type for optimal retrieval performance.
        
        Args:
            query: Search query string
        
        Returns:
            Query embedding vector
        
        Example:
            >>> query = "What is machine learning?"
            >>> embedding = generator.generate_query_embedding(query)
            >>> len(embedding)  # 1536
        """
        try:
            result = self.client.embed(
                texts=[query],
                model=self.model,
                input_type="query"  # For search queries
            )
            
            return result.embeddings[0]
            
        except Exception as e:
            raise EmbeddingError(
                message=f"Failed to generate query embedding: {str(e)}",
                details={"query": query[:100], "error": str(e)}
            ) from e
    
    def get_embedding_dimension(self) -> int:
        """
        Get embedding dimension for the current model.
        
        Returns:
            Embedding dimension (e.g., 1536 for voyage-large-2)
        
        Example:
            >>> dim = generator.get_embedding_dimension()
            >>> print(dim)  # 1536
        """
        # Model dimension mapping
        model_dimensions = {
            "voyage-large-2": 1536,
            "voyage-2": 1024,
            "voyage-code-2": 1536,
            "voyage-lite-02-instruct": 1024
        }
        
        return model_dimensions.get(self.model, 1536)  # Default 1536
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get embedding generation statistics.
        
        Returns:
            Dictionary with statistics
        
        Example:
            >>> stats = generator.get_stats()
            >>> print(stats['total_embeddings'])
        """
        return {
            "total_embeddings": self.total_embeddings,
            "model": self.model,
            "batch_size": self.batch_size,
            "embedding_dimension": self.get_embedding_dimension()
        }
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.total_embeddings = 0
        self.total_tokens = 0
        self.logger.info("Statistics reset")


class CachedEmbeddingGenerator(EmbeddingGenerator):
    """
    Embedding generator with in-memory caching.
    
    Caches embeddings to avoid redundant API calls for same texts.
    
    Example:
        >>> generator = CachedEmbeddingGenerator()
        >>> emb1 = generator.generate(["same text"])
        >>> emb2 = generator.generate(["same text"])  # Retrieved from cache
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with cache."""
        super().__init__(*args, **kwargs)
        self.cache: Dict[str, List[float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def generate(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings with caching.
        
        Checks cache first, only generates for uncached texts.
        """
        embeddings = []
        texts_to_generate = []
        indices_to_generate = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if text in self.cache:
                embeddings.append(self.cache[text])
                self.cache_hits += 1
            else:
                embeddings.append(None)  # Placeholder
                texts_to_generate.append(text)
                indices_to_generate.append(i)
                self.cache_misses += 1
        
        # Generate embeddings for uncached texts
        if texts_to_generate:
            new_embeddings = super().generate(texts_to_generate)
            
            # Update cache and results
            for idx, text, emb in zip(indices_to_generate, texts_to_generate, new_embeddings):
                self.cache[text] = emb
                embeddings[idx] = emb
        
        if texts_to_generate:
            self.logger.debug(
                f"Cache stats: {self.cache_hits} hits, "
                f"{self.cache_misses} misses "
                f"(hit rate: {self.cache_hits/(self.cache_hits + self.cache_misses)*100:.1f}%)"
            )
        
        return embeddings
    
    def clear_cache(self):
        """Clear embedding cache."""
        cache_size = len(self.cache)
        self.cache.clear()
        self.logger.info(f"Cleared cache ({cache_size} entries)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics including cache stats."""
        stats = super().get_stats()
        stats.update({
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0.0
            )
        })
        return stats