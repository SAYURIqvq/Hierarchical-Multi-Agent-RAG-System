"""
Example usage of LangGraph Workflow.

Demonstrates how to use the complete Agentic RAG workflow
with all agents coordinated through LangGraph.
"""

from langchain_anthropic import ChatAnthropic

from src.orchestration.langgraph_workflow import AgenticRAGWorkflow
from src.agents.planner import PlannerAgent
from src.agents.validator import ValidatorAgent
from src.agents.retrieval_coordinator import RetrievalCoordinator
from src.agents.retrieval.vector_agent import VectorSearchAgent
from src.agents.retrieval.keyword_agent import KeywordSearchAgent
from src.agents.retrieval.graph_agent import GraphSearchAgent
from src.config import get_settings


def main():
    """Demo complete workflow usage."""
    
    print("=" * 70)
    print("Agentic RAG Workflow Demo")
    print("=" * 70)
    print()
    
    # Load settings
    settings = get_settings()
    
    print("Step 1: Initializing components...")
    print("-" * 70)
    
    # Initialize LLM
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.anthropic_api_key
    )
    print(f"✓ LLM: {settings.llm_model}")
    
    # Initialize agents
    planner = PlannerAgent(llm=llm)
    print("✓ Planner Agent initialized")
    
    validator = ValidatorAgent(llm=llm)
    print(f"✓ Validator Agent initialized (threshold={validator.threshold})")
    
    # Initialize retrieval agents (mock mode)
    vector_agent = VectorSearchAgent(top_k=5, mock_mode=True)
    keyword_agent = KeywordSearchAgent(top_k=5, mock_mode=True)
    graph_agent = GraphSearchAgent(top_k=5, mock_mode=True)
    print("✓ Retrieval Agents initialized (MOCK mode)")
    
    # Initialize coordinator
    coordinator = RetrievalCoordinator(
        vector_agent=vector_agent,
        keyword_agent=keyword_agent,
        graph_agent=graph_agent,
        top_k=10
    )
    print(f"✓ Retrieval Coordinator initialized (top_k={coordinator.top_k})")
    
    # Build workflow
    workflow = AgenticRAGWorkflow(
        planner=planner,
        coordinator=coordinator,
        validator=validator
    )
    print("✓ LangGraph Workflow built")
    print()
    
    # Display workflow info
    print("Workflow Structure:")
    print("-" * 70)
    info = workflow.get_workflow_info()
    print(f"Nodes: {', '.join(info['nodes'])}")
    print(f"Fixed Edges:")
    for edge in info['edges']['fixed']:
        print(f"  • {edge}")
    print(f"Conditional Edges:")
    for edge in info['edges']['conditional']:
        print(f"  • {edge}")
    print()
    
    # Test queries
    test_queries = [
        "What is Python?",
        "Compare Python and Java in terms of performance",
        "How does machine learning work?",
    ]
    
    print("Step 2: Running queries through workflow...")
    print("=" * 70)
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 70)
        
        # Run workflow
        result = workflow.run(query)
        
        # Display results
        print(f"Strategy: {result.strategy.value.upper()}")
        print(f"Complexity: {result.complexity:.3f}")
        print(f"Chunks Retrieved: {len(result.chunks)}")
        print(f"Retrieval Rounds: {result.retrieval_round}")
        print(f"Validation Score: {result.validation_score:.3f}")
        print(f"Validation Status: {result.validation_status}")
        
        # Show top chunks
        if result.chunks:
            print(f"\nTop 3 Chunks:")
            for j, chunk in enumerate(result.chunks[:3], 1):
                print(f"  {j}. Score: {chunk.score:.2f} | {chunk.text[:60]}...")
        
        print()
    
    # Demo with trace
    print("Step 3: Running query with execution trace...")
    print("=" * 70)
    print()
    
    trace_query = "What is the relationship between Python and data science?"
    print(f"Query: {trace_query}")
    print("-" * 70)
    
    trace = workflow.run_with_trace(trace_query)
    
    print(f"Execution Path: {' → '.join(trace['execution_path'])}")
    print(f"Total Nodes Executed: {trace['total_nodes_executed']}")
    print()
    
    print("Node Outputs:")
    print(f"  Planner:")
    print(f"    - Complexity: {trace['node_outputs']['planner']['complexity']:.3f}")
    print(f"    - Strategy: {trace['node_outputs']['planner']['strategy']}")
    
    print(f"  Retrieval:")
    for idx, attempt in enumerate(trace['node_outputs']['retrieval']):
        print(f"    Round {attempt['round']}: {attempt['chunk_count']} chunks")
    
    print(f"  Validator:")
    print(f"    - Score: {trace['node_outputs']['validator']['score']:.3f}")
    print(f"    - Status: {trace['node_outputs']['validator']['status']}")
    
    print()
    print("=" * 70)
    print("Workflow Demo Complete!")
    print("=" * 70)


def demo_retry_scenario():
    """Demo workflow with retry scenario."""
    
    print("\n" + "=" * 70)
    print("Retry Scenario Demo")
    print("=" * 70)
    print()
    
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.anthropic_api_key
    )
    
    # Create workflow with stricter validator (higher threshold)
    planner = PlannerAgent(llm=llm)
    
    # Stricter validator (higher threshold = more likely to retry)
    validator = ValidatorAgent(llm=llm, threshold=0.8, max_retries=2)
    
    vector_agent = VectorSearchAgent(top_k=3, mock_mode=True)
    coordinator = RetrievalCoordinator(
        vector_agent=vector_agent,
        top_k=5
    )
    
    workflow = AgenticRAGWorkflow(planner, coordinator, validator)
    
    print(f"Using STRICT validator (threshold={validator.threshold})")
    print("This may trigger retrieval retries...")
    print()
    
    query = "Explain quantum computing"
    print(f"Query: {query}")
    print("-" * 70)
    
    trace = workflow.run_with_trace(query)
    
    retrieval_attempts = len(trace['node_outputs']['retrieval'])
    print(f"Retrieval Attempts: {retrieval_attempts}")
    
    for idx, attempt in enumerate(trace['node_outputs']['retrieval'], 1):
        print(f"  Attempt {idx}: {attempt['chunk_count']} chunks")
    
    print(f"\nFinal Status: {trace['node_outputs']['validator']['status']}")
    print(f"Final Score: {trace['node_outputs']['validator']['score']:.3f}")
    print()


def demo_different_strategies():
    """Demo workflow with different query complexities."""
    
    print("\n" + "=" * 70)
    print("Strategy Selection Demo")
    print("=" * 70)
    print()
    
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.anthropic_api_key
    )
    
    planner = PlannerAgent(llm=llm)
    validator = ValidatorAgent(llm=llm)
    
    vector_agent = VectorSearchAgent(top_k=5, mock_mode=True)
    keyword_agent = KeywordSearchAgent(top_k=5, mock_mode=True)
    graph_agent = GraphSearchAgent(top_k=5, mock_mode=True)
    
    coordinator = RetrievalCoordinator(
        vector_agent=vector_agent,
        keyword_agent=keyword_agent,
        graph_agent=graph_agent,
        top_k=10
    )
    
    workflow = AgenticRAGWorkflow(planner, coordinator, validator)
    
    # Different query types
    queries = [
        ("Simple", "What is X?"),
        ("Moderate", "How does Y work and what are its benefits?"),
        ("Complex", "Compare A and B in terms of X, Y, and Z, considering their relationship to C"),
    ]
    
    print("Testing different query complexities:")
    print()
    
    for label, query in queries:
        print(f"{label} Query: {query}")
        result = workflow.run(query)
        
        print(f"  → Complexity: {result.complexity:.3f}")
        print(f"  → Strategy: {result.strategy.value.upper()}")
        print(f"  → Chunks: {len(result.chunks)}")
        print()


if __name__ == "__main__":
    # Run main demo
    main()
    
    # Run additional demos
    demo_retry_scenario()
    demo_different_strategies()