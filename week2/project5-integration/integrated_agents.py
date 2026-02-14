"""
Integrated agents with full multi-tool support.

Combines:
- Multi-agent coordination from Project 4
- Multi-tool usage from Project 3
- ReAct reasoning from Project 2
- Structured output from Project 1
"""

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
import logging
import json

from multi_agent_state import MultiAgentState, add_agent_message
from tools_extended import get_tool_by_name, AVAILABLE_TOOLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENHANCED ACTION SCHEMAS (with more tools)
# ============================================================================

class EnhancedResearchAction(BaseModel):
    """Enhanced research agent with all 8 tools."""
    thought: str = Field(description="Reasoning about what information to gather")
    tool: Literal[
        "search_document",
        "database_query",
        "api_call",
        "web_search",
        "get_datetime",
        "file_read"
    ] = Field(description="Tool to use for research")
    tool_input: str = Field(description="Input for the selected tool")


class EnhancedAnalysisAction(BaseModel):
    """Enhanced analysis with calculator support."""
    thought: str = Field(description="Reasoning about analysis approach")
    tool: Literal["calculator", "none"] = Field(
        description="Use calculator for math, or 'none' for text analysis"
    )
    tool_input: str = Field(description="Expression to calculate or data to analyze")
    summary: str = Field(description="Brief summary of analysis findings")


class EnhancedReportAction(BaseModel):
    """Enhanced report with file save option."""
    thought: str = Field(description="Reasoning about report structure")
    report_format: Literal["summary", "detailed", "bullet_points"] = Field(
        description="Format for the report"
    )
    save_to_file: bool = Field(default=False, description="Whether to save report to file")
    filename: Optional[str] = Field(default=None, description="Filename if saving")


# ============================================================================
# ENHANCED RESEARCH AGENT (Multi-Tool)
# ============================================================================

class EnhancedResearchAgent:
    """
    Research Agent with full multi-tool support.
    
    Can use any of 8 tools:
    - search_document: Internal knowledge
    - database_query: SQL queries
    - api_call: External APIs
    - web_search: Web results
    - get_datetime: Current date/time
    - file_read: Read saved files
    """
    
    def __init__(self, model_name: str = "qwen2.5:7b"):
        base_llm = ChatOllama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.1,
            num_predict=512,
            format="json"
        )
        self.llm = base_llm.with_structured_output(EnhancedResearchAction)
    
    def __call__(self, state: MultiAgentState) -> MultiAgentState:
        """Execute enhanced research with multi-tool support."""
        logger.info("🔬 Enhanced Research Agent starting...")
        
        query = state["query"]
        
        # List available tools
        available_tools = [t.name for t in AVAILABLE_TOOLS if t.category in ["information", "database", "api", "file_ops"]]
        logger.info(f"   Available tools: {', '.join(available_tools)}")
        
        prompt = f"""You are an enhanced research agent with access to multiple tools.

User Query: {query}

Available Tools (choose the MOST appropriate):
- search_document: Search internal documents (best for: exam info, project details)
- database_query: SQL queries (best for: customer data, orders, analytics)
- api_call: External APIs (best for: user profiles, system analytics)
- web_search: Web search (best for: current events, external info)
- get_datetime: Current date/time (best for: time-based queries)
- file_read: Read saved files (best for: previously saved reports)

What information do you need? Choose ONE tool and provide input.
Return JSON with: thought, tool, tool_input"""

        try:
            action = self.llm.invoke(prompt)
            logger.info(f"   💭 Thought: {action.thought}")
            logger.info(f"   🔧 Tool: {action.tool}")
            logger.info(f"   📥 Input: {action.tool_input[:50]}...")
            
            # Execute tool
            tool = get_tool_by_name(action.tool)
            result = tool.execute(action.tool_input)
            
            logger.info(f"   ✅ Result: {len(result)} characters retrieved")
            
            # Update state
            state["research_data"] = result
            state["next_agent"] = "analysis"
            
            add_agent_message(
                state,
                agent="enhanced_research",
                content=f"Used {action.tool} to gather information",
                metadata={
                    "tool": action.tool,
                    "tool_input": action.tool_input,
                    "result_length": len(result)
                }
            )
            
            logger.info("   ✅ Research complete → Analysis Agent")
            
        except Exception as e:
            logger.error(f"   ❌ Research failed: {e}")
            state["research_data"] = f"Error: {e}"
            state["next_agent"] = "report"
        
        return state


# ============================================================================
# ENHANCED ANALYSIS AGENT (with Calculator)
# ============================================================================

class EnhancedAnalysisAgent:
    """
    Analysis Agent with calculator support for numeric operations.
    """
    
    def __init__(self, model_name: str = "qwen2.5:7b"):
        base_llm = ChatOllama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.1,
            num_predict=512,
            format="json"
        )
        self.llm = base_llm.with_structured_output(EnhancedAnalysisAction)
    
    def __call__(self, state: MultiAgentState) -> MultiAgentState:
        """Execute enhanced analysis with calculator."""
        logger.info("📊 Enhanced Analysis Agent starting...")
        
        query = state["query"]
        research_data = state["research_data"]
        
        if not research_data or "Error" in research_data:
            logger.warning("   ⚠️  No research data, skipping analysis")
            state["analysis_results"] = "No analysis performed"
            state["next_agent"] = "report"
            return state
        
        prompt = f"""You are an analysis agent with calculator access.

User Query: {query}

Research Data:
{research_data[:500]}

Analyze this data. If you need to calculate something (averages, percentages, totals):
- Set tool='calculator' and tool_input='expression' (e.g., "100 * 0.75")
- Otherwise, set tool='none' for text analysis

Provide a summary of your findings.
Return JSON with: thought, tool, tool_input, summary"""

        try:
            action = self.llm.invoke(prompt)
            logger.info(f"   💭 Thought: {action.thought}")
            logger.info(f"   🔧 Tool: {action.tool}")
            
            # Perform calculation if needed
            calculation_result = None
            if action.tool == "calculator":
                try:
                    calc_tool = get_tool_by_name("calculator")
                    calculation_result = calc_tool.execute(action.tool_input)
                    logger.info(f"   🔢 Calculation: {action.tool_input} = {calculation_result}")
                except Exception as e:
                    logger.warning(f"   ⚠️  Calculation failed: {e}")
            
            # Build analysis results
            analysis_text = action.summary
            if calculation_result:
                analysis_text = f"{action.summary}\n\nCalculation: {action.tool_input} = {calculation_result}"
            
            state["analysis_results"] = analysis_text
            state["next_agent"] = "report"
            
            add_agent_message(
                state,
                agent="enhanced_analysis",
                content="Completed analysis" + (" with calculations" if calculation_result else ""),
                metadata={
                    "used_calculator": action.tool == "calculator",
                    "calculation": calculation_result
                }
            )
            
            logger.info("   ✅ Analysis complete → Report Agent")
            
        except Exception as e:
            logger.error(f"   ❌ Analysis failed: {e}")
            state["analysis_results"] = f"Error: {e}"
            state["next_agent"] = "report"
        
        return state


# ============================================================================
# ENHANCED REPORT AGENT (with File Save)
# ============================================================================

class EnhancedReportAgent:
    """
    Report Agent with automatic file save capability.
    """
    
    def __init__(self, model_name: str = "qwen2.5:7b"):
        base_llm = ChatOllama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.2,
            num_predict=512,
            format="json"
        )
        self.llm = base_llm.with_structured_output(EnhancedReportAction)
    
    def __call__(self, state: MultiAgentState) -> MultiAgentState:
        """Execute enhanced report with file save."""
        logger.info("📝 Enhanced Report Agent starting...")
        
        query = state["query"]
        research_data = state["research_data"] or "No research data"
        analysis_results = state["analysis_results"] or "No analysis performed"
        
        prompt = f"""You are a report agent that formats research findings.

User Query: {query}

Research Data:
{research_data[:300]}

Analysis Results:
{analysis_results[:300]}

Choose report format: summary, detailed, or bullet_points
Should this report be saved to a file? (save_to_file: true/false)
If yes, suggest a filename (e.g., "research_report.txt")

Return JSON with: thought, report_format, save_to_file, filename"""

        try:
            action = self.llm.invoke(prompt)
            logger.info(f"   💭 Thought: {action.thought}")
            logger.info(f"   📄 Format: {action.report_format}")
            logger.info(f"   💾 Save to file: {action.save_to_file}")
            
            # Generate report
            report = self._generate_report(
                query,
                research_data,
                analysis_results,
                action.report_format,
                state["step_count"]
            )
            
            # Save to file if requested
            file_status = None
            if action.save_to_file and action.filename:
                file_status = self._save_report_to_file(report, action.filename)
                logger.info(f"   💾 File save: {file_status}")
            
            state["final_report"] = report
            state["workflow_complete"] = True
            state["next_agent"] = "END"
            
            add_agent_message(
                state,
                agent="enhanced_report",
                content="Generated report" + (f" and saved to {action.filename}" if file_status else ""),
                metadata={
                    "format": action.report_format,
                    "length": len(report),
                    "saved_to_file": action.save_to_file,
                    "filename": action.filename,
                    "file_status": file_status
                }
            )
            
            logger.info("   ✅ Report complete, workflow finished")
            logger.info(f"\n{'='*70}")
            logger.info("📄 FINAL REPORT")
            logger.info(f"{'='*70}")
            logger.info(report)
            if file_status:
                logger.info(f"\n💾 {file_status}")
            logger.info(f"{'='*70}\n")
            
        except Exception as e:
            logger.error(f"   ❌ Report generation failed: {e}")
            state["final_report"] = f"Error: {e}"
            state["workflow_complete"] = True
            state["next_agent"] = "END"
        
        return state
    
    def _generate_report(
        self,
        query: str,
        research: str,
        analysis: str,
        format_type: str,
        steps: int
    ) -> str:
        """Generate formatted report."""
        
        if format_type == "summary":
            return f"""RESEARCH SUMMARY

Query: {query}

Key Findings:
{analysis if analysis != "No analysis performed" else research[:300]}

Completed in {steps} steps."""
        
        elif format_type == "bullet_points":
            return f"""RESEARCH REPORT

Query: {query}

- Research: {research[:150]}...
- Analysis: {analysis[:150]}...
- Status: Complete ({steps} steps)"""
        
        else:  # detailed
            return f"""=== DETAILED RESEARCH REPORT ===

Query: {query}

Research Findings:
{research}

Analysis Results:
{analysis}

Workflow Summary:
- Total Steps: {steps}
- Status: Complete
"""
    
    def _save_report_to_file(self, report: str, filename: str) -> str:
        """Save report using file_write tool."""
        try:
            from tools_extended import get_tool_by_name
            file_tool = get_tool_by_name("file_write")
            
            # Format: "filename|content"
            file_input = f"{filename}|{report}"
            result = file_tool.execute(file_input)
            
            return result
        except Exception as e:
            return f"File save failed: {e}"