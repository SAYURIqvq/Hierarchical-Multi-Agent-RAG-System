# Planner Agent Documentation

## Overview

The Planner Agent is the **Strategic Level 1** agent that analyzes query complexity and selects the appropriate execution strategy for the system.

## Responsibilities

1. **Analyze Query Complexity** - Score queries from 0.0 (simple) to 1.0 (complex)
2. **Select Strategy** - Map complexity to SIMPLE, MULTIHOP, or GRAPH strategy
3. **Feature Extraction** - Identify query characteristics
4. **Semantic Analysis** - Use LLM to assess reasoning depth

## Complexity Factors

### 1. Length Score (30% weight)
- Based on word count
- 0-5 words: Low (0.0-0.2)
- 6-15 words: Medium (0.3-0.6)
- 16+ words: High (0.7-1.0)

### 2. Question Score (20% weight)
- Question marks count
- Question words (what, why, how)
- Multiple sub-questions

### 3. Entity Score (20% weight)
- Comparisons (compare, contrast, difference)
- Entity indicators (between, among, types of)
- List requests

### 4. Relationship Score (30% weight)
- Causal indicators (causes, leads to, results in)
- Impact indicators (effect of, impact of)
- Dependency indicators (depends on, related to)

### 5. Semantic Complexity (40% of final score)
- LLM-based analysis
- Depth of reasoning required
- Multi-step thinking needs
- Fallback to heuristics if LLM unavailable

## Strategy Mapping

**SIMPLE** (complexity < 0.3):
- Factual questions
- Single-concept queries
- Direct lookups
- Fast path processing

**MULTIHOP** (0.3 ≤ complexity < 0.7):
- Comparisons
- Multi-step reasoning
- Requires synthesis
- Standard agent pipeline

**GRAPH** (complexity ≥ 0.7):
- Relationship queries
- Complex analysis
- Multi-entity reasoning
- GraphRAG pathway

## Usage

### Basic Usage
```python
from langchain_anthropic import ChatAnthropic
from src.agents.planner import PlannerAgent
from src.models.agent_state import AgentState
from src.config import get_settings

# Initialize
settings = get_settings()
llm = ChatAnthropic(api_key=settings.anthropic_api_key)
planner = PlannerAgent(llm=llm)

# Analyze query
state = AgentState(query="What is Python?")
result = planner.run(state)

print(result.complexity)  # 0.234
print(result.strategy)    # Strategy.SIMPLE
```

### Custom Thresholds
```python
planner = PlannerAgent(
    llm=llm,
    simple_threshold=0.4,    # Custom
    multihop_threshold=0.8   # Custom
)
```

### Detailed Analysis
```python
details = planner.analyze_query_details("Compare X and Y")

print(details["features"])
# {
#   'length_score': 0.24,
#   'question_score': 0.0,
#   'entity_score': 0.4,
#   'relationship_score': 0.0
# }

print(details["heuristic_score"])   # 0.312
print(details["semantic_score"])     # 0.650
print(details["final_complexity"])   # 0.447
```

## Configuration

Settings from `src/config.py`:
```python
planner_complexity_threshold_simple: float = 0.3
planner_complexity_threshold_multihop: float = 0.7
```

Override via environment variables:
```bash
PLANNER_COMPLEXITY_THRESHOLD_SIMPLE=0.4
PLANNER_COMPLEXITY_THRESHOLD_MULTIHOP=0.8
```

## Performance

- **Average execution time**: 0.5-1.5s (includes LLM call)
- **Without LLM (fallback)**: 0.01-0.05s
- **Metrics tracked**: Calls, success rate, timing

## Error Handling

The planner handles errors gracefully:
- LLM failures → Use fallback scoring
- Invalid responses → Parse with regex, fallback if needed
- All errors wrapped in `AgentExecutionError`

## Testing

Run tests:
```bash
pytest tests/test_planner.py -v
```

Coverage: >90%

## Examples

See `examples/test_planner_usage.py` for complete examples.