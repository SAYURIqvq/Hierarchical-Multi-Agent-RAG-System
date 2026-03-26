# Workflow Architecture - LangGraph

## Overview

The Agentic RAG system uses LangGraph for orchestrating multi-agent workflows with conditional routing and retry logic.

## Workflow Structure
```
START
  ↓
┌─────────────┐
│  PLANNER    │ Analyze complexity, select strategy
└──────┬──────┘
       ↓
┌─────────────┐
│  RETRIEVAL  │ Spawn swarm, retrieve chunks
└──────┬──────┘
       ↓
┌─────────────┐
│  VALIDATOR  │ Check quality
└──────┬──────┘
       ↓
   Decision?
       ├─ PROCEED → END
       └─ RETRIEVE_MORE → (back to RETRIEVAL)
```

## Nodes

### 1. Planner Node
- **Agent:** PlannerAgent
- **Input:** Query
- **Output:** Complexity, Strategy
- **Next:** Always → Retrieval

### 2. Retrieval Node
- **Agent:** RetrievalCoordinator
- **Input:** Query, Strategy
- **Output:** Chunks
- **Next:** Always → Validator

### 3. Validator Node
- **Agent:** ValidatorAgent
- **Input:** Query, Chunks
- **Output:** Validation Status, Score
- **Next:** Conditional (PROCEED or RETRIEVE_MORE)

## Conditional Routing

**Validator Decision:**
```python
if validation_status == "PROCEED":
    → END (workflow complete)
elif validation_status == "RETRIEVE_MORE":
    → RETRIEVAL (retry)
```

**Retry Limits:**
- Max retries controlled by `validator.max_retries` (default: 2)
- After max retries, validator forces PROCEED

## State Management

**AgentState Fields:**
```python
{
    "query": str,
    "complexity": float,
    "strategy": Strategy,
    "chunks": List[Chunk],
    "retrieval_round": int,
    "validation_status": str,
    "validation_score": float,
    "metadata": dict
}
```

State is passed through all nodes and updated by each agent.

## Usage

### Basic Usage
```python
from src.orchestration.langgraph_workflow import AgenticRAGWorkflow

# Initialize agents
planner = PlannerAgent(llm=llm)
coordinator = RetrievalCoordinator(...)
validator = ValidatorAgent(llm=llm)

# Build workflow
workflow = AgenticRAGWorkflow(planner, coordinator, validator)

# Run query
result = workflow.run("What is machine learning?")

print(result.chunks)
print(result.validation_status)
```

### With Execution Trace
```python
trace = workflow.run_with_trace("Complex query")

print(trace["execution_path"])
# ['planner', 'retrieval', 'validator']

print(trace["node_outputs"])
# Detailed outputs from each node
```

## Error Handling

Each node wraps execution in try-catch and raises `OrchestrationError` on failure:
```python
try:
    result = workflow.run(query)
except OrchestrationError as e:
    print(f"Node {e.node_name} failed: {e.message}")
```

## Performance

**Typical Execution:**
- Simple query: ~2-3 seconds (no retry)
- Complex query: ~4-6 seconds (with potential retry)
- Max execution: ~10 seconds (max retries reached)

## Extending the Workflow

### Add New Node
```python
def _my_custom_node(self, state: AgentState) -> AgentState:
    """Custom processing."""
    # Your logic here
    return state

# In _build_workflow():
graph.add_node("custom", self._my_custom_node)
graph.add_edge("validator", "custom")
```

### Modify Routing
```python
def _custom_routing(self, state: AgentState) -> str:
    """Custom decision logic."""
    if state.complexity > 0.8:
        return "complex_path"
    else:
        return "simple_path"

graph.add_conditional_edges(
    "planner",
    self._custom_routing,
    {
        "simple_path": "retrieval",
        "complex_path": "advanced_retrieval"
    }
)
```