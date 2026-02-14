"""
LangGraph-based multi-agent orchestration.

Defines the workflow graph:
START → Research → Analysis → Report → END
"""

from langgraph.graph import StateGraph, END
import logging

from multi_agent_state import MultiAgentState, create_initial_state
from agents import ResearchAgent, AnalysisAgent, ReportAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# BUILD MULTI-AGENT GRAPH
# ============================================================================

def create_multi_agent_graph(model_name: str = "qwen2.5:7b"):
    """
    Create LangGraph workflow for multi-agent system.
    
    Graph structure:
    START → research → analysis → report → END
    
    Args:
        model_name: LLM model to use for all agents
        
    Returns:
        Compiled graph ready to execute
    """
    
    # Initialize agents
    research_agent = ResearchAgent(model_name=model_name)
    analysis_agent = AnalysisAgent(model_name=model_name)
    report_agent = ReportAgent(model_name=model_name)
    
    # Create graph
    workflow = StateGraph(MultiAgentState)
    
    # Add nodes (agents)
    workflow.add_node("research", research_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("report", report_agent)
    
    # Define edges (flow)
    workflow.set_entry_point("research")  # Start with research
    workflow.add_edge("research", "analysis")  # Research → Analysis
    workflow.add_edge("analysis", "report")  # Analysis → Report
    workflow.add_edge("report", END)  # Report → END
    
    # Compile graph
    app = workflow.compile()
    
    logger.info("✅ Multi-agent graph compiled successfully")
    logger.info("   Flow: START → Research → Analysis → Report → END")
    
    return app


# ============================================================================
# EXECUTION FUNCTION
# ============================================================================

def run_multi_agent_workflow(query: str, model_name: str = "qwen2.5:7b") -> MultiAgentState:
    """
    Run complete multi-agent workflow.
    
    Args:
        query: User question
        model_name: LLM model to use
        
    Returns:
        Final state with completed workflow
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 Starting Multi-Agent Workflow")
    logger.info(f"{'='*70}")
    logger.info(f"Query: {query}")
    logger.info(f"Model: {model_name}")
    logger.info(f"{'='*70}\n")
    
    # Create graph
    app = create_multi_agent_graph(model_name=model_name)
    
    # Create initial state
    initial_state = create_initial_state(query)
    
    # Execute workflow
    final_state = app.invoke(initial_state)
    
    return final_state