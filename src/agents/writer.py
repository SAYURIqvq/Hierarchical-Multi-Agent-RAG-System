"""
Writer Agent - Tactical Level 2 Agent.

Generates formatted answers from retrieved chunks with citations.
Uses LLM to synthesize information into coherent responses.
"""

from typing import List, Dict, Any, Optional
import re

from langchain_anthropic import ChatAnthropic

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.config import get_settings
from src.utils.logger import setup_logger
from src.utils.exceptions import AgenticRAGException


class WriterError(AgenticRAGException):
    """Error during answer generation."""
    pass


class WriterAgent(BaseAgent):
    """
    Writer Agent - Answer generation with citations.
    
    Takes retrieved chunks and user query, generates:
    - Coherent answer synthesizing information
    - Inline citations [1], [2], [3]
    - Source list with references
    
    Features:
    - LLM-based answer generation
    - Citation extraction and formatting
    - Answer quality checks
    - Source attribution
    
    Attributes:
        llm: Language model for generation
        max_tokens: Maximum tokens for answer
        temperature: LLM temperature
        include_sources: Whether to include source list
        
    Example:
        >>> agent = WriterAgent(llm=llm)
        >>> state = AgentState(query="What is Python?", chunks=[...])
        >>> result = agent.run(state)
        >>> print(result.answer)
        Python is a programming language [1] created by Guido van Rossum [2]...
    """
    
    def __init__(
        self,
        llm: Optional[ChatAnthropic] = None,
        max_tokens: int = None,
        temperature: float = None,
        include_sources: bool = True
    ):
        """
        Initialize Writer Agent.
        
        Args:
            llm: ChatAnthropic instance (creates if None)
            max_tokens: Max tokens for answer (default from config)
            temperature: LLM temperature (default from config)
            include_sources: Include source list in answer
        
        Example:
            >>> from langchain_anthropic import ChatAnthropic
            >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
            >>> agent = WriterAgent(llm=llm)
        """
        super().__init__(name="writer", version="1.0.0")
        
        settings = get_settings()
        
        # Initialize LLM
        if llm is None:
            self.llm = ChatAnthropic(
                model=settings.llm_model,
                temperature=temperature or settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
                api_key=settings.anthropic_api_key
            )
        else:
            self.llm = llm
        
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.temperature = temperature or settings.llm_temperature
        self.include_sources = include_sources
        
        self.log(
            f"Initialized with model={settings.llm_model}, "
            f"max_tokens={self.max_tokens}, "
            f"temperature={self.temperature}",
            level="info"
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute answer generation.
        
        Args:
            state: Current state with query and chunks
        
        Returns:
            Updated state with generated answer
        
        Raises:
            WriterError: If generation fails
        
        Example:
            >>> state = AgentState(query="What is ML?", chunks=[...])
            >>> result = agent.execute(state)
            >>> print(result.answer)
        """
        try:
            query = state.query
            chunks = state.chunks
            
            if not chunks:
                self.log("No chunks provided for answer generation", level="warning")
                state.answer = "I don't have enough information to answer this question."
                return state
            
            self.log(
                f"Generating answer for: {query[:50]}... "
                f"(using {len(chunks)} chunks)",
                level="info"
            )
            
            # Generate answer with citations
            answer = self._generate_answer(query, chunks)
            
            # Extract and format citations
            formatted_answer = self._format_answer(answer, chunks)
            
            # Update state
            state.answer = formatted_answer
            
            # Add metadata
            state.metadata["writer"] = {
                "chunks_used": len(chunks),
                "answer_length": len(formatted_answer),
                "citations_count": self._count_citations(formatted_answer)
            }
            
            self.log(
                f"Answer generated: {len(formatted_answer)} chars, "
                f"{self._count_citations(formatted_answer)} citations",
                level="info"
            )
            
            return state
            
        except Exception as e:
            self.log(f"Answer generation failed: {str(e)}", level="error")
            raise WriterError(
                message=f"Failed to generate answer: {str(e)}",
                details={"query": state.query}
            ) from e
    
    def _generate_answer(self, query: str, chunks: List[Chunk]) -> str:
        """
        Generate answer using LLM.
        
        Args:
            query: User query
            chunks: Retrieved chunks
        
        Returns:
            Generated answer with citations
        """
        # Prepare context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get('filename', 'unknown')
            score = chunk.score if chunk.score else 0.0
            context_parts.append(
                f"[{i}] (Source: {source}, Relevance: {score:.2f})\n{chunk.text}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Create prompt with strict citation rules
        prompt = f"""You are a precise and faithful assistant. Your ONLY job is to answer questions using the provided context. You must NEVER add information that is not explicitly stated in the context.

User Question: {query}

Context (with source references):
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROUNDING RULES (most important):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. EVERY claim in your answer MUST come directly from the context above
2. Do NOT infer, assume, or add information beyond what the context states
3. Do NOT paraphrase in a way that changes the meaning
4. If the context does not contain the answer, say:
   "The provided documents do not contain information about [topic]."
5. Do NOT use your general knowledge — ONLY the context matters

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CITATION RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Cite ONLY the specific chunk(s) that directly support EACH statement
2. Use inline citations: [1], [2], [3] — NOT grouped like [1][2][3][4]
3. Each sentence should cite ONLY the chunks it actually uses
4. If a statement uses ONLY chunk 2, cite [2] alone
5. If combining info from chunks 2 and 5, cite [2][5]
6. Different paragraphs will naturally cite DIFFERENT chunks
7. DO NOT cite all chunks in every paragraph

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ WRONG — adds info not in context (hallucination):
"Machine learning was invented in 1950 by Alan Turing [1]."
(If context does not explicitly say this, do NOT write it)

❌ WRONG — groups citations:
"ML is AI subset [1][2]. Uses algorithms [1][2]."

❌ WRONG — uses general knowledge:
"As is commonly known, neural networks have multiple layers."
(Only write this if the context actually states it)

✅ CORRECT — grounded + proper citation:
"Machine learning is a subset of artificial intelligence [1]. 
It uses algorithms to learn patterns from data [2]."

✅ CORRECT — honest when info is missing:
"The provided documents do not contain information about 
the history of machine learning."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Answer using ONLY information from context
2. Cite specific chunks per statement
3. Be comprehensive but concise
4. If context lacks info, state clearly using the template above
5. Write naturally but stay strictly grounded
6. DO NOT add "Sources:" section — ONLY inline citations
7. End your answer immediately after the last sentence

Answer (inline citations only, no Sources section):"""
        
        # Generate
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
            
            return answer
            
        except Exception as e:
            raise WriterError(
                message=f"LLM generation failed: {str(e)}",
                details={"query": query}
            ) from e
    
    def _format_answer(self, answer: str, chunks: List[Chunk]) -> str:
        """
        Format answer with source list.
        
        Args:
            answer: Generated answer with citations
            chunks: Source chunks
        
        Returns:
            Formatted answer with source list
        """
        if not self.include_sources:
            return answer
        
        # Extract unique citation numbers
        citations = re.findall(r'\[(\d+)\]', answer)
        unique_citations = sorted(set(int(c) for c in citations))
        
        if not unique_citations:
            return answer
        
        # Build source list
        sources_section = "\n\n---\n\n**Sources:**\n"
        
        for citation_num in unique_citations:
            # Get corresponding chunk (1-indexed)
            if citation_num <= len(chunks):
                chunk = chunks[citation_num - 1]
                source = chunk.metadata.get('filename', 'Unknown source')
                
                # Add source entry
                sources_section += f"\n[{citation_num}] {source}"
        
        return answer + sources_section
    
    def _count_citations(self, answer: str) -> int:
        """
        Count number of citations in answer.
        
        Args:
            answer: Answer text
        
        Returns:
            Number of unique citations
        """
        citations = re.findall(r'\[(\d+)\]', answer)
        return len(set(citations))
    
    def generate_with_feedback(
        self,
        query: str,
        chunks: List[Chunk],
        feedback: str
    ) -> str:
        """
        Regenerate answer with feedback from Critic.
        
        Args:
            query: User query
            chunks: Retrieved chunks
            feedback: Feedback from Critic agent
        
        Returns:
            Improved answer
        
        Example:
            >>> answer = agent.generate_with_feedback(
            ...     query="What is ML?",
            ...     chunks=chunks,
            ...     feedback="Add more examples"
            ... )
        """
        # Prepare context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get('filename', 'unknown')
            context_parts.append(f"[{i}] (Source: {source})\n{chunk.text}\n")
        
        context = "\n".join(context_parts)
        
        # Create improvement prompt
        prompt = f"""You are improving an answer based on feedback.

Original Question: {query}

Context:
{context}

Feedback for improvement:
{feedback}

Instructions:
1. Generate an IMPROVED answer addressing the feedback
2. Use inline citations [1], [2], [3]
3. Maintain accuracy and source attribution
4. Address all points in the feedback

Improved Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
            
            return self._format_answer(answer, chunks)
            
        except Exception as e:
            self.log(f"Answer regeneration failed: {str(e)}", level="error")
            raise WriterError(
                message=f"Failed to regenerate answer: {str(e)}",
                details={"feedback": feedback}
            ) from e