"""
Performance Tracker - Week 5 Day 6
Track latency, cache hits, and agent execution times.
"""

import time
from typing import Dict, Any, List
from datetime import datetime
import json
from pathlib import Path


class PerformanceTracker:
    """Track system performance metrics."""
    
    def __init__(self):
        self.metrics = []
        self.session_start = datetime.now()
    
    def track_query(
        self,
        query: str,
        latency: float,
        chunks_retrieved: int,
        strategy: str,
        iterations: int,
        cache_hit: bool = False
    ) -> None:
        """Record query performance."""
        
        metric = {
            'timestamp': datetime.now().isoformat(),
            'query_length': len(query),
            'latency_ms': latency * 1000,
            'chunks_retrieved': chunks_retrieved,
            'strategy': strategy,
            'iterations': iterations,
            'cache_hit': cache_hit
        }
        
        self.metrics.append(metric)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        
        if not self.metrics:
            return {}
        
        latencies = [m['latency_ms'] for m in self.metrics]
        cache_hits = sum(1 for m in self.metrics if m.get('cache_hit'))
        
        return {
            'total_queries': len(self.metrics),
            'avg_latency_ms': sum(latencies) / len(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'cache_hit_rate': cache_hits / len(self.metrics) if self.metrics else 0,
            'avg_chunks': sum(m['chunks_retrieved'] for m in self.metrics) / len(self.metrics),
            'session_duration_min': (datetime.now() - self.session_start).total_seconds() / 60
        }
    
    def save_metrics(self, filepath: str = "data/metrics.json") -> None:
        """Save metrics to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)