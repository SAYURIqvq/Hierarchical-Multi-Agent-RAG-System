"""
Example usage of Planner Agent.

Demonstrates how to use the Planner Agent to analyze queries
and select execution strategies.
"""

from langchain_anthropic import ChatAnthropic
from src.agents.planner import PlannerAgent
from src.models.agent_state import AgentState
from src.config import get_settings


def main():
    """Demo Planner Agent usage."""
    
    print("=" * 60)
    print("Planner Agent Demo")
    print("=" * 60)
    print()
    
    # Load settings
    settings = get_settings()
    
    # Initialize LLM
    print("Initializing Claude...")
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.anthropic_api_key
    )
    
    # Create Planner Agent
    planner = PlannerAgent(llm=llm)
    
    print(f"Planner initialized with thresholds:")
    print(f"  Simple: < {planner.simple_threshold}")
    print(f"  Multihop: {planner.simple_threshold} - {planner.multihop_threshold}")
    print(f"  Graph: > {planner.multihop_threshold}")
    print()
    
    # Test queries
    test_queries = [
        "What is Python?",
        "How do I install Python on Windows?",
        "Compare Python and Java in terms of performance and ease of use",
        "Explain the relationship between Python's GIL and multi-threaded performance across different CPU architectures",
        "What are the implications of using async/await versus threading in Python for I/O-bound versus CPU-bound tasks?",
    ]
    
    print("Analyzing queries:")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        
        # Create state
        state = AgentState(query=query)
        
        # Run planner
        result = planner.run(state)
        
        # Display results
        print(f"   Complexity: {result.complexity:.3f}")
        print(f"   Strategy: {result.strategy.value.upper()}")
        
        # Get detailed analysis
        details = planner.analyze_query_details(query)
        print(f"   Features:")
        print(f"     - Length: {details['features']['length_score']:.2f}")
        print(f"     - Questions: {details['features']['question_score']:.2f}")
        print(f"     - Entities: {details['features']['entity_score']:.2f}")
        print(f"     - Relationships: {details['features']['relationship_score']:.2f}")
        print(f"   Scores:")
        print(f"     - Heuristic: {details['heuristic_score']:.3f}")
        print(f"     - Semantic: {details['semantic_score']:.3f}")
    
    print()
    print("-" * 60)
    
    # Show metrics
    metrics = planner.get_metrics()
    print(f"\nPlanner Metrics:")
    print(f"  Total calls: {metrics['total_calls']}")
    print(f"  Success rate: {metrics['success_rate']}%")
    print(f"  Average time: {metrics['average_time_seconds']:.3f}s")
    
    print()
    print("=" * 60)


def demo_custom_thresholds():
    """Demo with custom thresholds."""
    
    print("\n" + "=" * 60)
    print("Custom Thresholds Demo")
    print("=" * 60)
    print()
    
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.anthropic_api_key
    )
    
    # Create planner with stricter thresholds
    planner = PlannerAgent(
        llm=llm,
        simple_threshold=0.2,  # Stricter (lower)
        multihop_threshold=0.5  # Stricter (lower)
    )
    
    print("Using stricter thresholds: simple=0.2, multihop=0.5")
    print()
    
    query = "Compare Python and Java"
    state = AgentState(query=query)
    result = planner.run(state)
    
    print(f"Query: {query}")
    print(f"Complexity: {result.complexity:.3f}")
    print(f"Strategy: {result.strategy.value.upper()}")
    print()
    
    print("With stricter thresholds, more queries will use complex strategies.")
    print("=" * 60)


def demo_query_comparison():
    """Compare multiple variations of similar queries."""
    
    print("\n" + "=" * 60)
    print("Query Variation Comparison")
    print("=" * 60)
    print()
    
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.anthropic_api_key
    )
    
    planner = PlannerAgent(llm=llm)
    
    # Similar queries with increasing complexity
    variations = [
        "Python",
        "What is Python?",
        "What is Python and how is it used?",
        "What is Python and how does it compare to Java?",
        "Compare Python and Java in terms of syntax, performance, and ecosystem",
    ]
    
    print("Analyzing query complexity progression:")
    print()
    
    for query in variations:
        state = AgentState(query=query)
        result = planner.run(state)
        
        print(f"{result.complexity:.3f} | {result.strategy.value:10s} | {query}")
    
    print()
    print("Notice how complexity increases with query detail.")
    print("=" * 60)


if __name__ == "__main__":
    # Run main demo
    main()
    
    # Additional demos
    demo_custom_thresholds()
    demo_query_comparison()