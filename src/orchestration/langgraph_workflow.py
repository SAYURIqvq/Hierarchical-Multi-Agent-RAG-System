"""
LangGraph Workflow - Agent Orchestration.

Defines the complete multi-agent workflow using LangGraph state machine.
Coordinates Planner, Retrieval Coordinator, and Validator agents.

Workflow:
    START → PLANNER → RETRIEVAL → VALIDATOR → END
                          ↑           ↓
                          └─ (retry) ─┘
"""

from typing import Dict, Any, Literal, TypedDict
from langgraph.graph import StateGraph, END

from src.models.agent_state import AgentState
from src.agents.planner import PlannerAgent
from src.agents.retrieval_coordinator import RetrievalCoordinator
from src.agents.validator import ValidatorAgent
from src.utils.logger import setup_logger
from src.utils.exceptions import OrchestrationError


class AgenticRAGWorkflow:
    """
    LangGraph workflow for Agentic RAG system.
    
    Orchestrates multi-agent pipeline with conditional routing
    and retry logic using LangGraph state machine.
    
    Attributes:
        planner: Planner agent instance
        coordinator: Retrieval coordinator instance
        validator: Validator agent instance
        workflow: Compiled LangGraph workflow
        logger: Logger instance
        
    Example:
        >>> from langchain_anthropic import ChatAnthropic
        >>> 
        >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> planner = PlannerAgent(llm=llm)
        >>> coordinator = RetrievalCoordinator(...)
        >>> validator = ValidatorAgent(llm=llm)
        >>> 
        >>> workflow = AgenticRAGWorkflow(planner, coordinator, validator)
        >>> result = workflow.run("What is Python?")
        >>> 
        >>> print(result.answer)  # Final answer
        >>> print(result.chunks)  # Retrieved chunks
    """
    
    def __init__(
        self,
        planner: PlannerAgent,
        coordinator: RetrievalCoordinator,
        validator: ValidatorAgent
    ):
        """
        Initialize LangGraph workflow.
        
        Args:
            planner: Planner agent for complexity analysis
            coordinator: Retrieval coordinator for chunk retrieval
            validator: Validator agent for quality control
        
        Example:
            >>> workflow = AgenticRAGWorkflow(
            ...     planner=planner_agent,
            ...     coordinator=retrieval_coordinator,
            ...     validator=validator_agent
            ... )
        """
        self.planner = planner
        self.coordinator = coordinator
        self.validator = validator
        
        self.logger = setup_logger("workflow", level="INFO")
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
        self.logger.info("AgenticRAG workflow initialized")
    
    def _build_workflow(self) -> StateGraph:
        """
        Build LangGraph workflow with nodes and edges.
        
        Creates state machine with:
        - Nodes: planner, retrieval, validator
        - Edges: Fixed and conditional transitions
        - Retry logic: validator → retrieval (if needed)
        
        Returns:
            Compiled StateGraph workflow
        """
        self.logger.info("Building LangGraph workflow...")
        
        # Define state schema for LangGraph
        class WorkflowState(TypedDict):
            """State schema for LangGraph."""
            agent_state: AgentState
        
        # Create state graph
        graph = StateGraph(WorkflowState)
        
        # Add nodes (with wrappers for LangGraph compatibility)
        graph.add_node("planner", self._planner_node_wrapper)
        graph.add_node("retrieval", self._retrieval_node_wrapper)
        graph.add_node("validator", self._validator_node_wrapper)
        
        # Define edges
        # START → planner
        graph.set_entry_point("planner")
        
        # planner → retrieval (always)
        graph.add_edge("planner", "retrieval")
        
        # retrieval → validator (always)
        graph.add_edge("retrieval", "validator")
        
        # validator → END or retrieval (conditional)
        graph.add_conditional_edges(
            "validator",
            self._should_continue_wrapper,
            {
                "continue": "retrieval",  # Retry retrieval
                "end": END               # Finish workflow
            }
        )
        
        # Compile workflow
        compiled = graph.compile()
        
        self.logger.info("Workflow built successfully")
        
        return compiled
    
    def _planner_node(self, state: AgentState) -> AgentState:
        """
        Execute planner agent node.
        
        Analyzes query complexity and selects execution strategy.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with complexity and strategy
        """
        self.logger.info("Executing PLANNER node")
        
        try:
            result = self.planner.run(state)
            
            self.logger.info(
                f"Planner: complexity={result.complexity:.2f}, "
                f"strategy={result.strategy.value}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Planner node failed: {str(e)}")
            raise OrchestrationError(
                node_name="planner",
                message=f"Planner execution failed: {str(e)}",
                details={"query": state.query}
            ) from e
    
    def _retrieval_node(self, state: AgentState) -> AgentState:
        """
        Execute retrieval coordinator node.
        
        Spawns retrieval swarm and aggregates results.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with retrieved chunks
        """
        self.logger.info(
            f"Executing RETRIEVAL node (round {state.retrieval_round})"
        )
        
        try:
            result = self.coordinator.run(state)
            
            self.logger.info(
                f"Retrieval: {len(result.chunks)} chunks retrieved "
                f"(round {result.retrieval_round})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Retrieval node failed: {str(e)}")
            raise OrchestrationError(
                node_name="retrieval",
                message=f"Retrieval execution failed: {str(e)}",
                details={
                    "query": state.query,
                    "round": state.retrieval_round
                }
            ) from e
    
    def _validator_node(self, state: AgentState) -> AgentState:
        """
        Execute validator agent node.
        
        Validates chunk quality and decides to proceed or retry.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with validation results
        """
        self.logger.info("Executing VALIDATOR node")
        
        try:
            result = self.validator.run(state)
            
            self.logger.info(
                f"Validator: score={result.validation_score:.2f}, "
                f"decision={result.validation_status}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Validator node failed: {str(e)}")
            raise OrchestrationError(
                node_name="validator",
                message=f"Validator execution failed: {str(e)}",
                details={
                    "query": state.query,
                    "chunk_count": len(state.chunks)
                }
            ) from e
    
    def _should_continue(
        self,
        state: AgentState
    ) -> Literal["continue", "end"]:
        """
        Conditional edge: Determine if workflow should continue or end.
        
        Decision logic:
        - If validation_status == "PROCEED" → end
        - If validation_status == "RETRIEVE_MORE" → continue (retry)
        
        Args:
            state: Current workflow state
        
        Returns:
            "continue" to retry retrieval, "end" to finish
        """
        decision = state.validation_status
        
        if decision == "PROCEED":
            self.logger.info("Validation passed → Ending workflow")
            return "end"
        elif decision == "RETRIEVE_MORE":
            self.logger.info(
                f"Validation failed → Retrying retrieval "
                f"(round {state.retrieval_round})"
            )
            return "continue"
        else:
            # Default to end if unknown status
            self.logger.warning(
                f"Unknown validation status '{decision}' → Ending workflow"
            )
            return "end"
    
    def _planner_node_wrapper(self, state: dict) -> dict:
        """
        LangGraph wrapper for planner node.
        
        Converts between LangGraph state format and AgentState.
        """
        agent_state = state["agent_state"]
        updated_state = self._planner_node(agent_state)
        return {"agent_state": updated_state}
    
    def _retrieval_node_wrapper(self, state: dict) -> dict:
        """
        LangGraph wrapper for retrieval node.
        
        Converts between LangGraph state format and AgentState.
        """
        agent_state = state["agent_state"]
        updated_state = self._retrieval_node(agent_state)
        return {"agent_state": updated_state}
    
    def _validator_node_wrapper(self, state: dict) -> dict:
        """
        LangGraph wrapper for validator node.
        
        Converts between LangGraph state format and AgentState.
        """
        agent_state = state["agent_state"]
        updated_state = self._validator_node(agent_state)
        return {"agent_state": updated_state}
    
    def _should_continue_wrapper(self, state: dict) -> Literal["continue", "end"]:
        """
        LangGraph wrapper for conditional routing.
        
        Converts between LangGraph state format and AgentState.
        """
        agent_state = state["agent_state"]
        return self._should_continue(agent_state)
    
    def run(self, query: str) -> AgentState:
        """
        Run complete workflow for a query.
        
        Executes full pipeline: Planner → Retrieval → Validator
        with automatic retry loop if needed.
        
        Args:
            query: User query string
        
        Returns:
            Final state with all results
        
        Raises:
            OrchestrationError: If workflow execution fails
        
        Example:
            >>> result = workflow.run("What is machine learning?")
            >>> print(result.complexity)
            >>> print(result.strategy)
            >>> print(len(result.chunks))
            >>> print(result.validation_status)
        """
        self.logger.info(f"Starting workflow for query: {query[:50]}...")
        
        try:
            # Create initial state
            initial_agent_state = AgentState(query=query)
            initial_state = {"agent_state": initial_agent_state}
            
            # Run workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Extract agent state from wrapper
            final_agent_state = final_state["agent_state"]
            
            self.logger.info(
                f"Workflow completed: "
                f"strategy={final_agent_state.strategy.value}, "
                f"chunks={len(final_agent_state.chunks)}, "
                f"rounds={final_agent_state.retrieval_round}, "
                f"validation={final_agent_state.validation_status}"
            )
            
            return final_agent_state
            
        except OrchestrationError:
            # Re-raise orchestration errors
            raise
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            raise OrchestrationError(
                node_name="workflow",
                message=f"Workflow execution failed: {str(e)}",
                details={"query": query}
            ) from e
    
    def run_with_trace(self, query: str) -> Dict[str, Any]:
        """
        Run workflow with detailed execution trace.
        
        Returns both final state and execution path for debugging.
        
        Args:
            query: User query string
        
        Returns:
            Dictionary with:
            - final_state: Final AgentState
            - execution_path: List of nodes executed
            - node_outputs: Intermediate states
        
        Example:
            >>> trace = workflow.run_with_trace("What is Python?")
            >>> print(trace["execution_path"])  # ['planner', 'retrieval', 'validator']
            >>> print(trace["final_state"].chunks)
        """
        self.logger.info(f"Starting traced workflow for: {query[:50]}...")
        
        execution_path = []
        node_outputs = {}
        
        # Create initial state
        state = AgentState(query=query)
        
        # Execute nodes manually for tracing
        try:
            # Node 1: Planner
            execution_path.append("planner")
            state = self._planner_node(state)
            node_outputs["planner"] = {
                "complexity": state.complexity,
                "strategy": state.strategy.value
            }
            
            # Node 2: Retrieval (with potential retries)
            retrieval_attempts = []
            max_iterations = 5  # Prevent infinite loop
            iteration = 0
            
            while iteration < max_iterations:
                execution_path.append("retrieval")
                state = self._retrieval_node(state)
                
                retrieval_attempts.append({
                    "round": state.retrieval_round,
                    "chunk_count": len(state.chunks)
                })
                
                # Node 3: Validator
                execution_path.append("validator")
                state = self._validator_node(state)
                
                # Check if should continue
                if self._should_continue(state) == "end":
                    break
                
                iteration += 1
            
            node_outputs["retrieval"] = retrieval_attempts
            node_outputs["validator"] = {
                "score": state.validation_score,
                "status": state.validation_status
            }
            
            return {
                "final_state": state,
                "execution_path": execution_path,
                "node_outputs": node_outputs,
                "total_nodes_executed": len(execution_path)
            }
            
        except Exception as e:
            self.logger.error(f"Traced workflow failed: {str(e)}")
            raise OrchestrationError(
                node_name="workflow_trace",
                message=f"Traced execution failed: {str(e)}",
                details={
                    "query": query,
                    "execution_path": execution_path
                }
            ) from e
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """
        Get information about the workflow structure.
        
        Returns:
            Dictionary with workflow metadata
        
        Example:
            >>> info = workflow.get_workflow_info()
            >>> print(info["nodes"])
            >>> print(info["edges"])
        """
        return {
            "nodes": ["planner", "retrieval", "validator"],
            "edges": {
                "fixed": [
                    "START → planner",
                    "planner → retrieval",
                    "retrieval → validator"
                ],
                "conditional": [
                    "validator → END (if PROCEED)",
                    "validator → retrieval (if RETRIEVE_MORE)"
                ]
            },
            "retry_logic": "validator can trigger retrieval retry",
            "max_retries": "controlled by validator.max_retries",
            "state_type": "AgentState"
        }