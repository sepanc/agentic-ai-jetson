"""
Specialized agents for multi-agent system.

Each agent has a specific role:
- Research Agent: Gathers information from available tools
- Analysis Agent: Processes and analyzes research data
- Report Agent: Formats findings into final report
"""

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Literal, Optional
import logging

from multi_agent_state import MultiAgentState, add_agent_message
from tools_extended import get_tool_by_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# AGENT ACTION SCHEMAS
# ============================================================================

class ResearchAction(BaseModel):
    """Research agent action schema."""
    thought: str = Field(description="Reasoning about what information to gather")
    tool: Literal["search_document", "database_query", "api_call", "web_search"] = Field(
        description="Tool to use for research"
    )
    tool_input: str = Field(description="Input for the selected tool")


class AnalysisAction(BaseModel):
    """Analysis agent action schema."""
    thought: str = Field(description="Reasoning about how to analyze the data")
    analysis_type: Literal["calculate", "summarize", "compare", "extract"] = Field(
        description="Type of analysis to perform"
    )
    analysis_input: str = Field(description="What to analyze")


class ReportAction(BaseModel):
    """Report agent action schema."""
    thought: str = Field(description="Reasoning about report structure")
    report_format: Literal["summary", "detailed", "bullet_points"] = Field(
        description="Format for the report"
    )
    include_metadata: bool = Field(default=False, description="Include workflow metadata")


# ============================================================================
# RESEARCH AGENT
# ============================================================================

class ResearchAgent:
    """
    Research Agent: Gathers information using available tools.
    
    Responsibilities:
    - Search documents for relevant information
    - Query databases for data
    - Call APIs for external information
    - Provide raw findings to analysis agent
    """
    
    def __init__(self, model_name: str = "qwen2.5:7b"):
        """Initialize research agent with LLM."""
        base_llm = ChatOllama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.1,
            num_predict=512,
            format="json"
        )
        self.llm = base_llm.with_structured_output(ResearchAction)
    
    def __call__(self, state: MultiAgentState) -> MultiAgentState:
        """
        Execute research agent.
        
        Reads: state["query"]
        Writes: state["research_data"], state["next_agent"]
        """
        logger.info("🔬 Research Agent starting...")
        
        query = state["query"]
        
        # Generate research action
        prompt = f"""You are a research agent. Your job is to gather information.

User Query: {query}

Available research tools:
- search_document: Search internal documents
- database_query: Query databases with SQL
- api_call: Call external APIs
- web_search: Search the web

What information do you need to gather? Choose ONE tool and provide the input.
Return JSON with: thought, tool, tool_input"""

        try:
            action = self.llm.invoke(prompt)
            logger.info(f"  Thought: {action.thought}")
            logger.info(f"  Tool: {action.tool}({action.tool_input})")
            
            # Execute tool
            tool = get_tool_by_name(action.tool)
            result = tool.execute(action.tool_input)
            
            logger.info(f"  Result: {result[:150]}...")
            
            # Update state
            state["research_data"] = result
            state["next_agent"] = "analysis"  # Move to analysis
            
            # Log message
            add_agent_message(
                state,
                agent="research",
                content=f"Gathered information using {action.tool}",
                metadata={"tool": action.tool, "result_length": len(result)}
            )
            
            logger.info("✅ Research complete, handing off to Analysis Agent")
            
        except Exception as e:
            logger.error(f"❌ Research failed: {e}")
            state["research_data"] = f"Error during research: {e}"
            state["next_agent"] = "report"  # Skip analysis, go to report
        
        return state


# ============================================================================
# ANALYSIS AGENT
# ============================================================================

class AnalysisAgent:
    """
    Analysis Agent: Processes research data.
    
    Responsibilities:
    - Calculate statistics from data
    - Summarize findings
    - Compare multiple data points
    - Extract key insights
    """
    
    def __init__(self, model_name: str = "qwen2.5:7b"):
        """Initialize analysis agent with LLM."""
        base_llm = ChatOllama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.1,
            num_predict=512,
            format="json"
        )
        self.llm = base_llm.with_structured_output(AnalysisAction)
    
    def __call__(self, state: MultiAgentState) -> MultiAgentState:
        """
        Execute analysis agent.
        
        Reads: state["query"], state["research_data"]
        Writes: state["analysis_results"], state["next_agent"]
        """
        logger.info("📊 Analysis Agent starting...")
        
        query = state["query"]
        research_data = state["research_data"]
        
        # If research failed, skip analysis
        if not research_data or "Error" in research_data:
            logger.warning("⚠️  No research data available, skipping analysis")
            state["analysis_results"] = "No analysis performed (no research data)"
            state["next_agent"] = "report"
            return state
        
        # Generate analysis action
        prompt = f"""You are an analysis agent. Your job is to analyze research findings.

User Query: {query}

Research Findings:
{research_data[:500]}

What analysis should you perform? Choose analysis type and what to analyze.
Types: calculate (do math), summarize (condense), compare (find differences), extract (pull key info)

Return JSON with: thought, analysis_type, analysis_input"""

        try:
            action = self.llm.invoke(prompt)
            logger.info(f"  Thought: {action.thought}")
            logger.info(f"  Analysis: {action.analysis_type} on {action.analysis_input[:50]}...")
            
            # Perform analysis based on type
            result = self._perform_analysis(
                action.analysis_type,
                action.analysis_input,
                research_data
            )
            
            logger.info(f"  Result: {result[:150]}...")
            
            # Update state
            state["analysis_results"] = result
            state["next_agent"] = "report"  # Move to report
            
            # Log message
            add_agent_message(
                state,
                agent="analysis",
                content=f"Completed {action.analysis_type} analysis",
                metadata={"analysis_type": action.analysis_type}
            )
            
            logger.info("✅ Analysis complete, handing off to Report Agent")
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            state["analysis_results"] = f"Error during analysis: {e}"
            state["next_agent"] = "report"
        
        return state
    
    def _perform_analysis(self, analysis_type: str, analysis_input: str, research_data: str) -> str:
        """Perform the actual analysis."""
        if analysis_type == "calculate":
            # Use calculator tool if needed
            try:
                from tools_extended import get_tool_by_name
                calc = get_tool_by_name("calculator")
                return calc.execute(analysis_input)
            except:
                return f"Calculation requested: {analysis_input}"
        
        elif analysis_type == "summarize":
            # For now, return first 200 chars (could use LLM for better summary)
            return f"Summary: {research_data[:200]}..."
        
        elif analysis_type == "extract":
            # Extract specific information
            return f"Extracted: {analysis_input} from research data"
        
        elif analysis_type == "compare":
            return f"Comparison: {analysis_input}"
        
        else:
            return f"Analysis ({analysis_type}): {analysis_input}"


# ============================================================================
# REPORT AGENT
# ============================================================================

class ReportAgent:
    """
    Report Agent: Formats final output.
    
    Responsibilities:
    - Compile research and analysis into coherent report
    - Format according to user preferences
    - Add metadata if requested
    - Produce final answer
    """
    
    def __init__(self, model_name: str = "qwen2.5:7b"):
        """Initialize report agent with LLM."""
        base_llm = ChatOllama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.2,  # Slightly higher for creative formatting
            num_predict=512,
            format="json"
        )
        self.llm = base_llm.with_structured_output(ReportAction)
    
    def __call__(self, state: MultiAgentState) -> MultiAgentState:
        """
        Execute report agent.
        
        Reads: state["query"], state["research_data"], state["analysis_results"]
        Writes: state["final_report"], state["workflow_complete"]
        """
        logger.info("📝 Report Agent starting...")
        
        query = state["query"]
        research_data = state["research_data"] or "No research data"
        analysis_results = state["analysis_results"] or "No analysis performed"
        
        # Generate report action
        prompt = f"""You are a report agent. Your job is to create a final report.

User Query: {query}

Research Data:
{research_data[:300]}

Analysis Results:
{analysis_results[:300]}

How should you format this report? Choose format: summary, detailed, or bullet_points

Return JSON with: thought, report_format, include_metadata"""

        try:
            action = self.llm.invoke(prompt)
            logger.info(f"  Thought: {action.thought}")
            logger.info(f"  Format: {action.report_format}")
            
            # Generate report
            report = self._generate_report(
                query,
                research_data,
                analysis_results,
                action.report_format,
                action.include_metadata,
                state["messages"]
            )
            
            # Update state
            state["final_report"] = report
            state["workflow_complete"] = True
            state["next_agent"] = "END"  # Signal completion
            
            # Log message
            add_agent_message(
                state,
                agent="report",
                content="Generated final report",
                metadata={"format": action.report_format, "length": len(report)}
            )
            
            logger.info("✅ Report complete, workflow finished")
            logger.info(f"\n{'='*70}\n📄 FINAL REPORT:\n{'='*70}\n{report}\n{'='*70}\n")
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            state["final_report"] = f"Error generating report: {e}"
            state["workflow_complete"] = True
            state["next_agent"] = "END"
        
        return state
    
    def _generate_report(
        self,
        query: str,
        research_data: str,
        analysis_results: str,
        format_type: str,
        include_metadata: bool,
        messages: list
    ) -> str:
        """Generate formatted report."""
        
        if format_type == "summary":
            report = f"""QUERY: {query}

FINDINGS: {analysis_results if analysis_results != "No analysis performed" else research_data[:200]}"""
        
        elif format_type == "bullet_points":
            report = f"""Query: {query}

Key Findings:
- Research: {research_data[:100]}...
- Analysis: {analysis_results[:100]}..."""
        
        else:  # detailed
            report = f"""=== DETAILED REPORT ===

Query: {query}

Research Findings:
{research_data}

Analysis Results:
{analysis_results}"""
        
        # Add metadata if requested
        if include_metadata:
            report += f"""

--- Workflow Metadata ---
Total Steps: {len(messages)}
Agents Involved: {', '.join(set(msg['agent'] for msg in messages))}"""
        
        return report