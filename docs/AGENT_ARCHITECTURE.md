# Agent Architecture - Design Patterns

## Overview

The Agentic RAG system uses a **Hierarchical Multi-Agent Architecture** where agents are organized in three levels and communicate through shared state.

---

## Agent Hierarchy
```
Level 1 (Strategic)
    └─ Planner Agent
        │
Level 2 (Tactical)
    ├─ Query Decomposer
    ├─ Retrieval Coordinator
    ├─ Validator Agent
    ├─ Synthesis Agent
    ├─ Writer Agent
    └─ Critic Agent
        │
Level 3 (Operational - Swarm)
    ├─ Vector Search Agent
    ├─ Keyword Search Agent
    └─ Graph Search Agent
```

---

## BaseAgent Pattern

All agents inherit from `BaseAgent` and follow this pattern:

### **Interface:**
```python
class MyAgent(BaseAgent):
    def execute(self, state: AgentState) -> AgentState:
        # 1. Read current state
        query = state.query
        
        # 2. Process
        result = self.process(query)
        
        # 3. Update state
        state.my_field = result
        
        # 4. Return updated state
        return state
```

### **Key Principles:**

1. **Single Responsibility**: Each agent does ONE thing well
2. **Immutable Input**: Don't modify state in-place (create new objects)
3. **Explicit Output**: Always return updated state
4. **Self-Contained**: Agent should work independently
5. **Observable**: Use logging and metrics

---

## State Flow
```
User Query
    ↓
AgentState(query="...")
    ↓
Planner.execute(state)
    → state.complexity = 0.7
    → state.strategy = "multihop"
    ↓
Coordinator.execute(state)
    → state.chunks = [...]
    ↓
Validator.execute(state)
    → state.validation_status = "PROCEED"
    ↓
Writer.execute(state)
    → state.answer = "..."
    ↓
Final Answer
```

---

## Agent Communication

Agents communicate **only through AgentState**:

### ✅ **Correct:**
```python
class Agent1(BaseAgent):
    def execute(self, state: AgentState) -> AgentState:
        state.field1 = "value"
        return state

class Agent2(BaseAgent):
    def execute(self, state: AgentState) -> AgentState:
        # Read from state
        value = state.field1
        return state
```

### ❌ **Incorrect:**
```python
class Agent1(BaseAgent):
    def execute(self, state: AgentState) -> AgentState:
        # Don't store in instance variable
        self.result = "value"  # ❌
        return state

class Agent2(BaseAgent):
    def __init__(self, agent1):
        # Don't depend on other agents directly
        self.agent1 = agent1  # ❌
```

---

## Error Handling Pattern

### **In Agent:**
```python
def execute(self, state: AgentState) -> AgentState:
    try:
        result = risky_operation()
        state.result = result
        return state
    except Exception as e:
        self.log(f"Error: {e}", level="error")
        raise AgentExecutionError(
            agent_name=self.name,
            message=str(e)
        )
```

### **In Orchestrator:**
```python
try:
    state = agent.run(state)
except AgentExecutionError as e:
    # Handle gracefully
    logger.error(f"Agent {e.agent_name} failed: {e}")
    # Retry or skip
```

---

## Metrics Pattern

Every agent automatically tracks:
- Total calls
- Success/failure rate
- Execution time
- Average time

### **Usage:**
```python
agent = MyAgent(name="test")

# Execute
agent.run(state)

# Check metrics
metrics = agent.get_metrics()
print(f"Success rate: {metrics['success_rate']}%")
print(f"Avg time: {metrics['average_time_seconds']}s")
```

---

## Logging Pattern

### **Levels:**
- `DEBUG`: Detailed diagnostic info
- `INFO`: General progress updates
- `WARNING`: Unexpected but handled
- `ERROR`: Execution failed

### **Usage:**
```python
def execute(self, state: AgentState) -> AgentState:
    self.log("Starting analysis", level="debug")
    self.log("Processing 10 chunks", level="info")
    
    if quality < threshold:
        self.log("Low quality detected", level="warning")
    
    return state
```

---

## Testing Pattern

### **Test Structure:**
```python
class TestMyAgent:
    def test_basic_execution(self):
        agent = MyAgent(name="test")
        state = AgentState(query="test")
        
        result = agent.run(state)
        
        assert result.my_field == expected_value
    
    def test_error_handling(self):
        agent = MyAgent(name="test")
        # Test failure scenarios
    
    def test_metrics_tracking(self):
        agent = MyAgent(name="test")
        # Verify metrics updated
```

---

## Best Practices

### **DO:**
- ✅ Use `run()` instead of `execute()` directly
- ✅ Log important decisions
- ✅ Return updated state
- ✅ Handle errors gracefully
- ✅ Write comprehensive tests

### **DON'T:**
- ❌ Share state between agents via instance variables
- ❌ Call other agents directly
- ❌ Modify state in-place without returning
- ❌ Catch exceptions without logging
- ❌ Skip error handling

---

## Performance Considerations

### **Timing:**
```python
# Automatic timing via run()
result = agent.run(state)  # Timed automatically

# Manual timing if needed
import time
start = time.time()
result = agent.execute(state)
elapsed = time.time() - start
```

### **Memory:**
- Don't store large data in agent instance
- Use state for data passing
- Clean up after execution

### **Concurrency:**
- BaseAgent is NOT thread-safe
- Create separate instances for parallel execution
- Use Swarm pattern for parallelism

---

## Extension Guide

### **Creating New Agent:**

1. **Inherit from BaseAgent:**
```python
from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState

class MyNewAgent(BaseAgent):
    def __init__(self, param1: str):
        super().__init__(name="my_new_agent", version="1.0.0")
        self.param1 = param1
```

2. **Implement execute():**
```python
    def execute(self, state: AgentState) -> AgentState:
        self.log("Starting execution")
        
        # Your logic here
        result = self._my_logic(state.query)
        
        # Update state
        state.my_result = result
        
        return state
```

3. **Add helper methods:**
```python
    def _my_logic(self, query: str) -> str:
        # Private helper method
        return processed_result
```

4. **Write tests:**
```python
def test_my_new_agent():
    agent = MyNewAgent(param1="test")
    state = AgentState(query="test")
    
    result = agent.run(state)
    
    assert result.my_result is not None
```

---

## Common Patterns

### **1. Retry Pattern:**
```python
def execute(self, state: AgentState) -> AgentState:
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            result = self._try_operation()
            state.result = result
            return state
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            self.log(f"Retry {attempt + 1}/{max_retries}", level="warning")
    
    return state
```

### **2. Conditional Execution:**
```python
def execute(self, state: AgentState) -> AgentState:
    if state.complexity < 0.3:
        # Simple path
        state.result = self._simple_process()
    else:
        # Complex path
        state.result = self._complex_process()
    
    return state
```

### **3. Multi-Step Processing:**
```python
def execute(self, state: AgentState) -> AgentState:
    # Step 1
    data = self._step1(state.query)
    self.log("Step 1 complete")
    
    # Step 2
    processed = self._step2(data)
    self.log("Step 2 complete")
    
    # Step 3
    state.result = self._step3(processed)
    self.log("All steps complete")
    
    return state
```

---

## Version History

- **v1.0.0** (Dec 2024): Initial BaseAgent implementation
  - Abstract execute() method
  - Automatic metrics tracking
  - Integrated logging
  - Error handling wrapper

---

## References

- `src/agents/base_agent.py` - Implementation
- `tests/test_base_agent.py` - Test examples
- `src/models/agent_state.py` - State definition