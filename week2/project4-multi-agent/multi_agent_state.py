# """
# Shared state for multi-agent system.
#
# State is passed between agents and tracks:
# - Input query
# - Research findings
# - Analysis results
# - Final report
# - Agent decisions
# """
#
# from typing import TypedDict, List, Annotated, Optional
# from langgraph.graph import add_messages
# from pydantic import BaseModel, Field
#
#
# # ============================================================================
# # STATE DEFINITIONS
# # ============================================================================
#
# class AgentMessage(BaseModel):
#     """Message from an agent."""
#     agent: str = Field(description="Agent name (research, analysis, report)")
#     content: str = Field(description="Message content")
#     metadata: Optional[dict] = Field(default=None, description="Additional metadata")
#
#
# class MultiAgentState(TypedDict):
#     """
#     Shared state passed between agents.
#
#     This is the "memory" that flows through the graph.
#     Each agent reads from and writes to this state.
#     """
#     # Input
#     query: str  # Original user query
#
#     # Workflow tracking
#     next_agent: str  # Which agent should run next
#     messages: Annotated[List[AgentMessage], add_messages]  # Agent conversation log
#
#     # Agent outputs (each agent writes to its section)
#     research_data: Optional[str]  # Research agent findings
#     analysis_results: Optional[str]  # Analysis agent results
#     final_report: Optional[str]  # Report agent output
#
#     # Metadata
#     step_count: int  # Track number of steps
#     workflow_complete: bool  # Flag when done
#
#
# # ============================================================================
# # HELPER FUNCTIONS
# # ============================================================================
#
# def create_initial_state(query: str) -> MultiAgentState:
#     """Create initial state for a new workflow."""
#     return MultiAgentState(
#         query=query,
#         next_agent="research",  # Start with research
#         messages=[],
#         research_data=None,
#         analysis_results=None,
#         final_report=None,
#         step_count=0,
#         workflow_complete=False
#     )
#
#
# def add_agent_message(state: MultiAgentState, agent: str, content: str, metadata: dict = None) -> MultiAgentState:
#     """Add a message to the conversation log."""
#     message = AgentMessage(agent=agent, content=content, metadata=metadata)
#     state["messages"].append(message)
#     state["step_count"] += 1
#     return state

"""
Shared state for multi-agent system.

State is passed between agents and tracks:
- Input query
- Research findings
- Analysis results
- Final report
- Agent decisions
"""

from typing import TypedDict, List, Dict, Optional
from pydantic import BaseModel, Field


# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class MultiAgentState(TypedDict):
    """
    Shared state passed between agents.
    
    This is the "memory" that flows through the graph.
    Each agent reads from and writes to this state.
    """
    # Input
    query: str  # Original user query
    
    # Workflow tracking
    next_agent: str  # Which agent should run next
    messages: List[Dict[str, str]]  # Agent conversation log (simple dicts, not Pydantic)
    
    # Agent outputs (each agent writes to its section)
    research_data: Optional[str]  # Research agent findings
    analysis_results: Optional[str]  # Analysis agent results
    final_report: Optional[str]  # Report agent output
    
    # Metadata
    step_count: int  # Track number of steps
    workflow_complete: bool  # Flag when done


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_initial_state(query: str) -> MultiAgentState:
    """Create initial state for a new workflow."""
    return MultiAgentState(
        query=query,
        next_agent="research",  # Start with research
        messages=[],
        research_data=None,
        analysis_results=None,
        final_report=None,
        step_count=0,
        workflow_complete=False
    )


def add_agent_message(state: MultiAgentState, agent: str, content: str, metadata: dict = None) -> MultiAgentState:
    """Add a message to the conversation log."""
    message = {
        "agent": agent,
        "content": content,
        "metadata": metadata or {}
    }
    state["messages"].append(message)
    state["step_count"] += 1
    return state