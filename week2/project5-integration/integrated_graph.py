"""
Integrated multi-agent workflow combining all Week 2 projects.
"""

from langgraph.graph import StateGraph, END
import logging

from multi_agent_state import MultiAgentState, create_initial_state
from integrated_agents import EnhancedResearchAgent, EnhancedAnalysisAgent, EnhancedReportAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_integrated_graph(model_name: str = "qwen2.5:7b"):
    """
    Create integrated research assistant workflow.
    
    Combines:
    - Multi-agent orchestration (Project 4)
    - Multi-tool support (Project 3)
    - Structured output (Project 1)
    - ReAct reasoning (Project 2)
    
    Returns:
        Compiled graph ready to execute
    """
    
    # Initialize enhanced agents
    research_agent = EnhancedResearchAgent(model_name=model_name)
    analysis_agent = EnhancedAnalysisAgent(model_name=model_name)
    report_agent = EnhancedReportAgent(model_name=model_name)
    
    # Create graph
    workflow = StateGraph(MultiAgentState)
    
    # Add nodes
    workflow.add_node("research", research_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("report", report_agent)
    
    # Define flow
    workflow.set_entry_point("research")
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "report")
    workflow.add_edge("report", END)
    
    # Compile
    app = workflow.compile()
    
    logger.info("✅ Integrated workflow compiled")
    logger.info("   Components:")
    logger.info("   - Multi-Agent Orchestration (LangGraph)")
    logger.info("   - Multi-Tool Support (8 tools)")
    logger.info("   - Structured Output (Pydantic)")
    logger.info("   - Automatic File Export")
    logger.info(f"   - Model: {model_name}")
    
    return app


def run_integrated_research(query: str, model_name: str = "qwen2.5:7b") -> MultiAgentState:
    """Run complete integrated research workflow."""
    logger.info(f"\n{'='*80}")
    logger.info("🚀 INTEGRATED RESEARCH ASSISTANT")
    logger.info(f"{'='*80}")
    logger.info(f"Query: {query}")
    logger.info(f"{'='*80}\n")
    
    app = create_integrated_graph(model_name=model_name)
    initial_state = create_initial_state(query)
    final_state = app.invoke(initial_state)
    
    return final_state