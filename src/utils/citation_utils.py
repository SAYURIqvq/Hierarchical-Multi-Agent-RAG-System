"""
Citation Utilities - Helper functions for citation handling.

Provides utilities for:
- Citation extraction
- Citation validation
- Source formatting
- Reference generation
"""

import re
from typing import List, Dict, Tuple, Set


class CitationUtils:
    """
    Utilities for citation handling in generated answers.
    
    Example:
        >>> utils = CitationUtils()
        >>> citations = utils.extract_citations("Answer [1] with [2] sources")
        >>> print(citations)  # [1, 2]
    """
    
    @staticmethod
    def extract_citations(text: str) -> List[int]:
        """
        Extract all citation numbers from text.
        
        Args:
            text: Text with citations like [1], [2], [3]
        
        Returns:
            List of unique citation numbers (sorted)
        
        Example:
            >>> text = "Python [1] is great [2]. It's used [1] widely."
            >>> CitationUtils.extract_citations(text)
            [1, 2]
        """
        citations = re.findall(r'\[(\d+)\]', text)
        unique = sorted(set(int(c) for c in citations))
        return unique
    
    @staticmethod
    def validate_citations(text: str, max_citation: int) -> Dict[str, any]:
        """
        Validate that citations are valid.
        
        Checks:
        - All citations are within range [1, max_citation]
        - No missing citations (if [1], [3] exist, [2] should too)
        
        Args:
            text: Text with citations
            max_citation: Maximum valid citation number
        
        Returns:
            Dictionary with validation results
        
        Example:
            >>> text = "Answer [1] and [3]."
            >>> CitationUtils.validate_citations(text, max_citation=3)
            {'valid': False, 'errors': ['Missing citation [2]'], ...}
        """
        citations = CitationUtils.extract_citations(text)
        
        errors = []
        warnings = []
        
        # Check for out-of-range citations
        for citation in citations:
            if citation > max_citation:
                errors.append(
                    f"Citation [{citation}] exceeds maximum [{max_citation}]"
                )
            if citation < 1:
                errors.append(f"Invalid citation [{citation}] (must be >= 1)")
        
        # Check for gaps (optional warning)
        if citations:
            expected = set(range(1, max(citations) + 1))
            actual = set(citations)
            missing = expected - actual
            
            if missing:
                warnings.append(
                    f"Missing citations: {sorted(missing)}"
                )
        
        # Check if any citations exist
        if not citations:
            warnings.append("No citations found in answer")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'citation_count': len(citations),
            'citations': citations
        }
    
    @staticmethod
    def format_source_list(
        chunks: List,
        citations_used: List[int] = None
    ) -> str:
        """
        Format source list for citations.
        
        Args:
            chunks: List of chunk objects with metadata
            citations_used: List of citation numbers used (optional)
        
        Returns:
            Formatted source list string
        
        Example:
            >>> chunks = [chunk1, chunk2, chunk3]
            >>> sources = CitationUtils.format_source_list(chunks, [1, 3])
            >>> print(sources)
            [1] document1.pdf
            [3] document2.pdf
        """
        if citations_used is None:
            citations_used = list(range(1, len(chunks) + 1))
        
        source_lines = []
        
        for citation_num in sorted(citations_used):
            if citation_num <= len(chunks):
                chunk = chunks[citation_num - 1]
                
                # Extract source info
                filename = chunk.metadata.get('filename', 'Unknown')
                doc_id = chunk.metadata.get('doc_id', '')
                
                # Format source line
                source_line = f"[{citation_num}] {filename}"
                
                source_lines.append(source_line)
        
        return "\n".join(source_lines)
    
    @staticmethod
    def count_citations_per_source(text: str) -> Dict[int, int]:
        """
        Count how many times each citation is used.
        
        Args:
            text: Text with citations
        
        Returns:
            Dictionary mapping citation number to count
        
        Example:
            >>> text = "Python [1] is great [2]. Python [1] rocks!"
            >>> CitationUtils.count_citations_per_source(text)
            {1: 2, 2: 1}
        """
        citations = re.findall(r'\[(\d+)\]', text)
        
        counts = {}
        for citation in citations:
            num = int(citation)
            counts[num] = counts.get(num, 0) + 1
        
        return counts
    
    @staticmethod
    def has_sufficient_citations(
        text: str,
        min_citations: int = 1
    ) -> Tuple[bool, str]:
        """
        Check if answer has sufficient citations.
        
        Args:
            text: Answer text
            min_citations: Minimum required citations
        
        Returns:
            Tuple of (is_sufficient, message)
        
        Example:
            >>> text = "Answer with [1] citation."
            >>> CitationUtils.has_sufficient_citations(text, min_citations=2)
            (False, 'Only 1 citation found, need at least 2')
        """
        citations = CitationUtils.extract_citations(text)
        count = len(citations)
        
        if count >= min_citations:
            return True, f"Sufficient citations: {count}"
        else:
            return False, f"Only {count} citation(s) found, need at least {min_citations}"
    
    @staticmethod
    def remove_duplicate_citations(text: str) -> str:
        """
        Remove duplicate consecutive citations.
        
        Example: [1][1] -> [1]
                 [1][2][1] -> [1][2][1] (not consecutive)
        
        Args:
            text: Text with possible duplicate citations
        
        Returns:
            Text with duplicates removed
        
        Example:
            >>> text = "Answer [1][1] with [2][2][3]."
            >>> CitationUtils.remove_duplicate_citations(text)
            'Answer [1] with [2][3].'
        """
        # Replace consecutive duplicates
        pattern = r'\[(\d+)\](?:\[\1\])+'
        
        def replace_duplicates(match):
            return f"[{match.group(1)}]"
        
        return re.sub(pattern, replace_duplicates, text)