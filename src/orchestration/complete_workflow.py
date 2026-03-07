"""
Complete LangGraph Workflow - All 10 Agents Orchestration.

Implements full multi-agent pipeline with conditional routing:
- Planner → Decomposer → Retrieval → Validator (with retry)
- Synthesis → Writer ↔ Critic (self-reflection loop)

Supports all strategies: SIMPLE, MULTIHOP, GRAPH
"""

from typing import Dict, Any, Literal, TypedDict
from langgraph.graph import StateGraph, END

from src.models.agent_state import AgentState, Strategy
from src.agents.planner import PlannerAgent
from src.agents.query_decomposer import QueryDecomposer
from src.agents.retrieval_coordinator import RetrievalCoordinator
from src.agents.validator import ValidatorAgent
from src.agents.synthesis import SynthesisAgent
from src.agents.writer import WriterAgent
from src.agents.critic import CriticAgent, CriticDecision
from src.utils.logger import setup_logger
from src.utils.exceptions import OrchestrationError


class CompleteAgenticRAGWorkflow:
    """
    Complete LangGraph workflow for Agentic RAG system.
    
    Orchestrates all 10 agents in multi-stage pipeline:
    
    Stage 1 - Planning:
        Planner → Decomposer
    
    Stage 2 - Retrieval (with retry):
        Retrieval Coordinator → Validator
        (loops back if validation fails)
    
    Stage 3 - Synthesis:
        Synthesis (deduplicate + rank)
    
    Stage 4 - Generation (self-reflection):
        Writer ↔ Critic
        (loops back if critic requests regeneration)
    
    Attributes:
        planner: Planner agent
        decomposer: Query decomposer
        coordinator: Retrieval coordinator
        validator: Validator agent
        synthesis: Synthesis agent
        writer: Writer agent
        critic: Critic agent
        workflow: Compiled LangGraph workflow
        logger: Logger instance
        
    Example:
        >>> from langchain_anthropic import ChatAnthropic
        >>> 
        >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> 
        >>> workflow = CompleteAgenticRAGWorkflow(
        ...     planner=PlannerAgent(llm=llm),
        ...     decomposer=QueryDecomposer(),
        ...     coordinator=RetrievalCoordinator(...),
        ...     validator=ValidatorAgent(llm=llm),
        ...     synthesis=SynthesisAgent(),
        ...     writer=WriterAgent(llm=llm),
        ...     critic=CriticAgent(llm=llm)
        ... )
        >>> 
        >>> result = workflow.run("Compare Python and Java")
        >>> print(result.answer)
        >>> print(f"Quality: {result.critic_score:.2f}")
    """
    
    def __init__(
        self,
        planner: PlannerAgent,
        decomposer: QueryDecomposer,
        coordinator: RetrievalCoordinator,
        validator: ValidatorAgent,
        synthesis: SynthesisAgent,
        writer: WriterAgent,
        critic: CriticAgent
    ):
        """
        Initialize complete LangGraph workflow.
        
        Args:
            planner: Planner agent for complexity analysis
            decomposer: Query decomposer for sub-queries
            coordinator: Retrieval coordinator (manages swarm)
            validator: Validator agent for quality control
            synthesis: Synthesis agent for deduplication
            writer: Writer agent for answer generation
            critic: Critic agent for quality evaluation
        """
        self.planner = planner
        self.decomposer = decomposer
        self.coordinator = coordinator
        self.validator = validator
        self.synthesis = synthesis
        self.writer = writer
        self.critic = critic
        
        self.logger = setup_logger("complete_workflow", level="INFO")
        
        # Build complete workflow graph
        self.workflow = self._build_workflow()
        
        self.logger.info("Complete AgenticRAG workflow initialized (10 agents)")
    
    def _build_workflow(self) -> StateGraph:
        """
        Build complete LangGraph workflow with all agents.
        
        Creates state machine with:
        - 7 nodes (agents)
        - Fixed edges (sequential flow)
        - 2 conditional edges (retry loops)
        
        Returns:
            Compiled StateGraph workflow
        """
        self.logger.info("Building complete LangGraph workflow...")
        
        # Define state schema
        class WorkflowState(TypedDict):
            """State schema for LangGraph."""
            agent_state: AgentState
        
        # Create state graph
        graph = StateGraph(WorkflowState)
        
        # ========== ADD NODES ==========
        graph.add_node("planner", self._planner_node_wrapper)
        graph.add_node("decomposer", self._decomposer_node_wrapper)
        graph.add_node("retrieval", self._retrieval_node_wrapper)
        graph.add_node("validator", self._validator_node_wrapper)
        graph.add_node("synthesis", self._synthesis_node_wrapper)
        graph.add_node("writer", self._writer_node_wrapper)
        graph.add_node("critic", self._critic_node_wrapper)
        
        # ========== STAGE 1: PLANNING ==========
        # START → planner
        graph.set_entry_point("planner")
        
        # planner → decomposer
        graph.add_edge("planner", "decomposer")
        
        # ========== STAGE 2: RETRIEVAL WITH RETRY ==========
        # decomposer → retrieval
        graph.add_edge("decomposer", "retrieval")
        
        # retrieval → validator
        graph.add_edge("retrieval", "validator")
        
        # validator → retrieval (retry) OR synthesis (proceed)
        graph.add_conditional_edges(
            "validator",
            self._should_retry_retrieval_wrapper,
            {
                "retry": "retrieval",      # Retry if validation fails
                "proceed": "synthesis"     # Continue if validation passes
            }
        )
        
        # ========== STAGE 3: SYNTHESIS ==========
        # synthesis → writer (no conditional, always proceed)
        graph.add_edge("synthesis", "writer")
        
        # ========== STAGE 4: GENERATION WITH SELF-REFLECTION ==========
        # writer → critic
        graph.add_edge("writer", "critic")
        
        # critic → writer (regenerate) OR END (approved)
        graph.add_conditional_edges(
            "critic",
            self._should_regenerate_wrapper,
            {
                "regenerate": "writer",    # Regenerate if quality low
                "finish": END              # Finish if approved
            }
        )
        
        # Compile workflow
        compiled = graph.compile()
        
        self.logger.info("Complete workflow built successfully (7 nodes, 2 conditional edges)")
        
        return compiled
    
    # ========== NODE EXECUTION METHODS ==========
    
    def _planner_node(self, state: AgentState) -> AgentState:
        """Execute planner agent node."""
        self.logger.info("🧠 Executing PLANNER node")
        
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
    
    def _decomposer_node(self, state: AgentState) -> AgentState:
        """Execute decomposer agent node."""
        self.logger.info("✂️ Executing DECOMPOSER node")
        
        try:
            result = self.decomposer.run(state)
            
            self.logger.info(
                f"Decomposer: {len(result.sub_queries)} sub-queries "
                f"(strategy: {result.strategy.value})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Decomposer node failed: {str(e)}")
            raise OrchestrationError(
                node_name="decomposer",
                message=f"Decomposer execution failed: {str(e)}",
                details={"query": state.query, "strategy": state.strategy}
            ) from e
    
    def _retrieval_node(self, state: AgentState) -> AgentState:
        """Execute retrieval coordinator node."""
        self.logger.info(
            f"🔍 Executing RETRIEVAL node (round {state.retrieval_round})"
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
        """Execute validator agent node."""
        self.logger.info("✅ Executing VALIDATOR node")
        
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
    
    def _synthesis_node(self, state: AgentState) -> AgentState:
        """Execute synthesis agent node."""
        self.logger.info("🔄 Executing SYNTHESIS node")
        
        try:
            result = self.synthesis.run(state)
            
            metadata = result.metadata.get("synthesis", {})
            self.logger.info(
                f"Synthesis: {metadata.get('input_count', 0)} → "
                f"{metadata.get('final_count', 0)} chunks "
                f"(dedup: {metadata.get('deduplication_rate', 0):.1%})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Synthesis node failed: {str(e)}")
            raise OrchestrationError(
                node_name="synthesis",
                message=f"Synthesis execution failed: {str(e)}",
                details={
                    "query": state.query,
                    "chunk_count": len(state.chunks)
                }
            ) from e
    
    def _writer_node(self, state: AgentState) -> AgentState:
        """Execute writer agent node."""
        
        # Check if regeneration
        regeneration_count = state.metadata.get("regeneration_count", 0)
        
        if regeneration_count > 0:
            self.logger.info(f"✍️ Executing WRITER node (regeneration {regeneration_count})")
            
            try:
                # Use regeneration method with critic feedback
                improved_answer = self.writer.generate_with_feedback(
                    query=state.query,
                    chunks=state.chunks,
                    feedback=state.critic_feedback
                )
                
                state.answer = improved_answer
                state.metadata["regeneration_count"] = regeneration_count
                
                self.logger.info(
                    f"Writer: Answer regenerated (length: {len(improved_answer)} chars)"
                )
                
                return state
                
            except Exception as e:
                self.logger.error(f"Writer regeneration failed: {str(e)}")
                raise OrchestrationError(
                    node_name="writer_regenerate",
                    message=f"Writer regeneration failed: {str(e)}",
                    details={"query": state.query, "regeneration": regeneration_count}
                ) from e
        
        else:
            self.logger.info("✍️ Executing WRITER node (initial generation)")
            
            try:
                result = self.writer.run(state)
                
                self.logger.info(
                    f"Writer: Answer generated (length: {len(result.answer)} chars)"
                )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Writer node failed: {str(e)}")
                raise OrchestrationError(
                    node_name="writer",
                    message=f"Writer execution failed: {str(e)}",
                    details={"query": state.query}
                ) from e
    
    def _critic_node(self, state: AgentState) -> AgentState:
        """Execute critic agent node."""
        self.logger.info("🔎 Executing CRITIC node")
        
        try:
            result = self.critic.run(state)
            
            self.logger.info(
                f"Critic: score={result.critic_score:.2f}, "
                f"decision={result.critic_decision.value}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Critic node failed: {str(e)}")
            raise OrchestrationError(
                node_name="critic",
                message=f"Critic execution failed: {str(e)}",
                details={"query": state.query}
            ) from e
    
    # ========== CONDITIONAL ROUTING ==========
    
    def _should_retry_retrieval(self, state: AgentState) -> Literal["retry", "proceed"]:
        """
        Conditional edge: Determine if retrieval should retry.
        
        Decision logic:
        - validation_status == "PROCEED" → proceed
        - validation_status == "RETRIEVE_MORE" → retry
        
        Args:
            state: Current workflow state
        
        Returns:
            "retry" to retry retrieval, "proceed" to continue
        """
        decision = state.validation_status
        
        if decision == "PROCEED":
            self.logger.info("✅ Validation passed → Proceeding to synthesis")
            return "proceed"
        
        elif decision == "RETRIEVE_MORE":
            self.logger.info(
                f"⚠️ Validation failed → Retrying retrieval "
                f"(round {state.retrieval_round})"
            )
            return "retry"
        
        else:
            # Default to proceed if unknown
            self.logger.warning(
                f"Unknown validation status '{decision}' → Proceeding"
            )
            return "proceed"
    
    def _should_regenerate(self, state: AgentState) -> Literal["regenerate", "finish"]:
        """
        Conditional edge: Determine if writer should regenerate.
        
        Decision logic:
        - critic_decision == "APPROVED" → finish
        - critic_decision == "REGENERATE" AND iterations < max → regenerate
        - iterations >= max → finish (force)
        
        Args:
            state: Current workflow state
        
        Returns:
            "regenerate" to regenerate answer, "finish" to end
        """
        decision = state.critic_decision
        regeneration_count = state.metadata.get("regeneration_count", 0)
        max_iterations = self.critic.max_iterations
        
        if decision == CriticDecision.APPROVED:
            self.logger.info("✅ Critic approved → Finishing workflow")
            return "finish"
        
        elif decision == CriticDecision.REGENERATE:
            if regeneration_count < max_iterations:
                self.logger.info(
                    f"🔄 Critic requests regeneration "
                    f"(iteration {regeneration_count + 1}/{max_iterations})"
                )
                
                # Increment regeneration count
                state.metadata["regeneration_count"] = regeneration_count + 1
                
                return "regenerate"
            else:
                self.logger.warning(
                    f"⚠️ Max iterations reached ({max_iterations}) → Finishing"
                )
                return "finish"
        
        else:
            # INSUFFICIENT_INFO or unknown
            self.logger.info(f"ℹ️ Critic decision: {decision.value} → Finishing")
            return "finish"
    
    # ========== LANGGRAPH WRAPPERS ==========
    
    def _planner_node_wrapper(self, state: dict) -> dict:
        """LangGraph wrapper for planner node."""
        agent_state = state["agent_state"]
        updated_state = self._planner_node(agent_state)
        return {"agent_state": updated_state}
    
    def _decomposer_node_wrapper(self, state: dict) -> dict:
        """LangGraph wrapper for decomposer node."""
        agent_state = state["agent_state"]
        updated_state = self._decomposer_node(agent_state)
        return {"agent_state": updated_state}
    
    def _retrieval_node_wrapper(self, state: dict) -> dict:
        """LangGraph wrapper for retrieval node."""
        agent_state = state["agent_state"]
        updated_state = self._retrieval_node(agent_state)
        return {"agent_state": updated_state}
    
    def _validator_node_wrapper(self, state: dict) -> dict:
        """LangGraph wrapper for validator node."""
        agent_state = state["agent_state"]
        updated_state = self._validator_node(agent_state)
        return {"agent_state": updated_state}
    
    def _synthesis_node_wrapper(self, state: dict) -> dict:
        """LangGraph wrapper for synthesis node."""
        agent_state = state["agent_state"]
        updated_state = self._synthesis_node(agent_state)
        return {"agent_state": updated_state}
    
    def _writer_node_wrapper(self, state: dict) -> dict:
        """LangGraph wrapper for writer node."""
        agent_state = state["agent_state"]
        updated_state = self._writer_node(agent_state)
        return {"agent_state": updated_state}
    
    def _critic_node_wrapper(self, state: dict) -> dict:
        """LangGraph wrapper for critic node."""
        agent_state = state["agent_state"]
        updated_state = self._critic_node(agent_state)
        return {"agent_state": updated_state}
    
    def _should_retry_retrieval_wrapper(self, state: dict) -> Literal["retry", "proceed"]:
        """LangGraph wrapper for retrieval retry conditional."""
        agent_state = state["agent_state"]
        return self._should_retry_retrieval(agent_state)
    
    def _should_regenerate_wrapper(self, state: dict) -> Literal["regenerate", "finish"]:
        """LangGraph wrapper for regeneration conditional."""
        agent_state = state["agent_state"]
        return self._should_regenerate(agent_state)
    
    # ========== PUBLIC API ==========
    
    def run(self, query: str) -> AgentState:
        """
        Run complete workflow for a query.
        
        Executes full pipeline:
        1. Planner → Decomposer
        2. Retrieval → Validator (with retry)
        3. Synthesis
        4. Writer ↔ Critic (self-reflection)
        
        Args:
            query: User query string
        
        Returns:
            Final state with answer and all metadata
        
        Raises:
            OrchestrationError: If workflow execution fails
        
        Example:
            >>> result = workflow.run("What is machine learning?")
            >>> print(result.answer)
            >>> print(f"Complexity: {result.complexity:.2f}")
            >>> print(f"Strategy: {result.strategy}")
            >>> print(f"Chunks used: {len(result.chunks)}")
            >>> print(f"Quality score: {result.critic_score:.2f}")
        """
        self.logger.info(f"Starting complete workflow for query: {query[:50]}...")
        
        try:
            # Create initial state
            initial_agent_state = AgentState(query=query)
            initial_state = {"agent_state": initial_agent_state}
            
            # Run workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Extract agent state
            final_agent_state = final_state["agent_state"]
            
            self.logger.info(
                f"✅ Workflow completed: "
                f"strategy={final_agent_state.strategy.value}, "
                f"chunks={len(final_agent_state.chunks)}, "
                f"retrieval_rounds={final_agent_state.retrieval_round}, "
                f"validation={final_agent_state.validation_status}, "
                f"critic_score={final_agent_state.critic_score:.2f}, "
                f"regenerations={final_agent_state.metadata.get('regeneration_count', 0)}"
            )
            
            return final_agent_state
            
        except OrchestrationError:
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
        
        Returns final state plus execution path for debugging.
        
        Args:
            query: User query string
        
        Returns:
            Dictionary with:
            - final_state: Final AgentState
            - execution_path: List of nodes executed
            - node_timings: Time spent in each node
            - total_duration: Total execution time
        
        Example:
            >>> trace = workflow.run_with_trace("What is Python?")
            >>> print(trace["execution_path"])
            >>> print(f"Total time: {trace['total_duration']:.2f}s")
            >>> for node, time in trace["node_timings"].items():
            ...     print(f"{node}: {time:.2f}s")
        """
        import time
        
        self.logger.info(f"Starting traced workflow for: {query[:50]}...")
        
        execution_path = []
        node_timings = {}
        start_time = time.time()
        
        # Create initial state
        state = AgentState(query=query)
        
        try:
            # Stage 1: Planning
            node_start = time.time()
            execution_path.append("planner")
            state = self._planner_node(state)
            node_timings["planner"] = time.time() - node_start
            
            node_start = time.time()
            execution_path.append("decomposer")
            state = self._decomposer_node(state)
            node_timings["decomposer"] = time.time() - node_start
            
            # Stage 2: Retrieval with retry
            retrieval_attempts = 0
            max_retrieval_attempts = 5
            
            while retrieval_attempts < max_retrieval_attempts:
                node_start = time.time()
                execution_path.append("retrieval")
                state = self._retrieval_node(state)
                
                if "retrieval" not in node_timings:
                    node_timings["retrieval"] = []
                node_timings["retrieval"].append(time.time() - node_start)
                
                node_start = time.time()
                execution_path.append("validator")
                state = self._validator_node(state)
                
                if "validator" not in node_timings:
                    node_timings["validator"] = []
                node_timings["validator"].append(time.time() - node_start)
                
                # Check if should continue
                if self._should_retry_retrieval(state) == "proceed":
                    break
                
                retrieval_attempts += 1
            
            # Stage 3: Synthesis
            node_start = time.time()
            execution_path.append("synthesis")
            state = self._synthesis_node(state)
            node_timings["synthesis"] = time.time() - node_start
            
            # Stage 4: Generation with self-reflection
            regeneration_attempts = 0
            max_regenerations = self.critic.max_iterations
            
            while regeneration_attempts <= max_regenerations:
                node_start = time.time()
                execution_path.append("writer")
                state = self._writer_node(state)
                
                if "writer" not in node_timings:
                    node_timings["writer"] = []
                node_timings["writer"].append(time.time() - node_start)
                
                node_start = time.time()
                execution_path.append("critic")
                state = self._critic_node(state)
                
                if "critic" not in node_timings:
                    node_timings["critic"] = []
                node_timings["critic"].append(time.time() - node_start)
                
                # Check if should continue
                if self._should_regenerate(state) == "finish":
                    break
                
                regeneration_attempts += 1
            
            total_duration = time.time() - start_time
            
            return {
                "final_state": state,
                "execution_path": execution_path,
                "node_timings": node_timings,
                "total_duration": total_duration,
                "retrieval_attempts": retrieval_attempts + 1,
                "regeneration_attempts": regeneration_attempts
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
            >>> print(f"Nodes: {len(info['nodes'])}")
            >>> print(f"Retry loops: {len(info['conditional_edges'])}")
        """
        return {
            "nodes": [
                "planner",
                "decomposer",
                "retrieval",
                "validator",
                "synthesis",
                "writer",
                "critic"
            ],
            "edges": {
                "fixed": [
                    "START → planner",
                    "planner → decomposer",
                    "decomposer → retrieval",
                    "retrieval → validator",
                    "synthesis → writer",
                    "writer → critic"
                ],
                "conditional": [
                    "validator → retrieval (retry) | synthesis (proceed)",
                    "critic → writer (regenerate) | END (finish)"
                ]
            },
            "retry_mechanisms": {
                "retrieval": "validator triggers retry if validation fails",
                "generation": "critic triggers regeneration if quality low"
            },
            "max_retries": {
                "retrieval": self.validator.max_retries,
                "generation": self.critic.max_iterations
            },
            "state_type": "AgentState",
            "total_agents": 10,  # 7 in workflow + 3 in coordinator swarm
            "stages": [
                "Planning (Planner, Decomposer)",
                "Retrieval (Coordinator → Validator with retry)",
                "Synthesis (Deduplication + Ranking)",
                "Generation (Writer ↔ Critic self-reflection)"
            ]
        }