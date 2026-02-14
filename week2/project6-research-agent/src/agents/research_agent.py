"""
Multi-source research agent using LangGraph for orchestration.
Analyzes queries, routes to appropriate tools, synthesizes findings.
"""

from typing import List, Literal
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
import asyncio
import logging
from pathlib import Path
import json

from ..models.schemas import (
    ResearchState, QueryAnalysis, QueryCategory,
    ResearchReport, Source, SourceType,
    VectorSource, ArxivSource, WebSource, FileSource  # Add these imports
)

from ..tools.vector_search import VectorSearchTool
from ..tools.arxiv_search import ArxivSearchTool
from ..tools.web_search import WebSearchTool
from ..tools.file_search import FileSearchTool
from ..config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchAgent:
    """Multi-source research agent with LLM-driven query routing"""
    
    def __init__(
        self,
        ollama_base_url: str = None,
        model: str = None,
        temperature: float = 0.1
    ):
        self.ollama_base_url = ollama_base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        
        # Initialize LLM
        self.llm = Ollama(
            base_url=self.ollama_base_url,
            model=self.model,
            temperature=temperature
        )
        
        # Initialize tools
        self.vector_tool = VectorSearchTool()
        self.arxiv_tool = ArxivSearchTool()
        self.web_tool = WebSearchTool()
        self.file_tool = FileSearchTool()
        
        # Build graph
        self.graph = self._build_graph()
        
        logger.info(f"ResearchAgent initialized with model: {self.model}")
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("plan_retrieval", self._plan_retrieval)
        workflow.add_node("execute_retrieval", self._execute_retrieval)
        workflow.add_node("synthesize", self._synthesize)
        workflow.add_node("format_report", self._format_report)
        
        # Define edges
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "plan_retrieval")
        workflow.add_edge("plan_retrieval", "execute_retrieval")
        workflow.add_edge("execute_retrieval", "synthesize")
        workflow.add_edge("synthesize", "format_report")
        workflow.add_edge("format_report", END)
        
        return workflow.compile()
    
    def _analyze_query(self, state: ResearchState) -> ResearchState:
        """
        Node 1: Analyze query to determine category and retrieval strategy
        """
        logger.info(f"Analyzing query: {state.query}")
        
        prompt = f"""Analyze this research query and classify it:

Query: {state.query}

Respond ONLY with valid JSON matching this exact structure:
{{
  "category": "academic" or "general" or "hybrid",
  "reasoning": "brief explanation of why",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "use_arxiv": true or false,
  "use_web": true or false,
  "use_vector": true,
  "use_files": true
}}

Classification guidelines:
- "academic": Requires research papers, citations, theoretical foundation → use_arxiv=true, use_web=false
- "general": Current trends, best practices, news, how-to → use_arxiv=false, use_web=true  
- "hybrid": Mix of both → use_arxiv=true, use_web=true

Always use vector (local knowledge) and files (internal data).
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Parse JSON response
            # LLM might include markdown code fences, strip them
            response_clean = response.strip()
            if response_clean.startswith("```"):
                response_clean = response_clean.split("```")[1]
                if response_clean.startswith("json"):
                    response_clean = response_clean[4:]
            response_clean = response_clean.strip()
            
            analysis_dict = json.loads(response_clean)
            analysis = QueryAnalysis(**analysis_dict)
            
            state.analysis = analysis
            logger.info(f"Query classified as: {analysis.category}")
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Fallback: treat as hybrid
            state.analysis = QueryAnalysis(
                category=QueryCategory.HYBRID,
                reasoning="Fallback due to parsing error",
                keywords=state.query.split()[:5],
                use_arxiv=True,
                use_web=True,
                use_vector=True,
                use_files=True
            )
            state.errors.append(f"Analysis parsing error: {e}")
        
        return state
    
    def _plan_retrieval(self, state: ResearchState) -> ResearchState:
        """
        Node 2: Create retrieval plan based on analysis
        """
        logger.info("Planning retrieval strategy")
        
        if not state.analysis:
            state.errors.append("No analysis available for planning")
            return state
        
        # Determine which tools to use based on analysis
        tools_to_use = []
        
        if state.analysis.use_vector:
            tools_to_use.append(SourceType.VECTOR)
        if state.analysis.use_arxiv:
            tools_to_use.append(SourceType.ARXIV)
        if state.analysis.use_web:
            tools_to_use.append(SourceType.WEB)
        if state.analysis.use_files:
            tools_to_use.append(SourceType.FILE)
        
        logger.info(f"Will query {len(tools_to_use)} sources: {[t.value for t in tools_to_use]}")
        
        return state
    
    def _execute_retrieval(self, state: ResearchState) -> ResearchState:
        """
        Node 3: Execute retrieval from all planned sources (parallel)
        """
        logger.info("Executing parallel retrieval")
        
        if not state.analysis:
            state.errors.append("No analysis available for retrieval")
            return state
        
        # Build search query from keywords
        search_query = " ".join(state.analysis.keywords[:5])
        
        # Execute retrievals in parallel
        tasks = []
        
        if state.analysis.use_vector:
            tasks.append(("vector", self._retrieve_vector(search_query)))
        if state.analysis.use_arxiv:
            tasks.append(("arxiv", self._retrieve_arxiv(search_query)))
        if state.analysis.use_web:
            tasks.append(("web", self._retrieve_web(search_query)))
        if state.analysis.use_files:
            tasks.append(("files", self._retrieve_files()))
        
        # Run all tasks
        for name, task in tasks:
            try:
                results = task  # Synchronous for now
                
                if name == "vector":
                    state.vector_results = results
                elif name == "arxiv":
                    state.arxiv_results = results
                elif name == "web":
                    state.web_results = results
                elif name == "files":
                    state.file_results = results
                    
                logger.info(f"{name.capitalize()} retrieval: {len(results)} results")
                
            except Exception as e:
                logger.error(f"{name.capitalize()} retrieval failed: {e}")
                state.errors.append(f"{name} retrieval error: {e}")
        
        # Combine all sources
        state.all_sources = (
    [Source(**s.model_dump()) for s in state.vector_results] +
    [Source(**s.model_dump()) for s in state.arxiv_results] +
    [Source(**s.model_dump()) for s in state.web_results] +
    [Source(**s.model_dump()) for s in state.file_results]
        )
        
        logger.info(f"Total sources retrieved: {len(state.all_sources)}")
        
        return state
    
    def _retrieve_vector(self, query: str) -> List[VectorSource]:
        """Retrieve from vector database"""
        try:
            results = self.vector_tool.search(query, top_k=3, min_relevance=0.3)
            return results
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def _retrieve_arxiv(self, query: str) -> List[ArxivSource]:
        """Retrieve from Arxiv"""
        try:
            results = self.arxiv_tool.search(query, max_results=3)
            return results
        except Exception as e:
            logger.error(f"Arxiv search error: {e}")
            return []
    
    def _retrieve_web(self, query: str) -> List[WebSource]:
        """Retrieve from web (Serper)"""
        try:
            results = self.web_tool.search(query, num_results=3)
            return results
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    def _retrieve_files(self) -> List[FileSource]:
        """Retrieve from local files"""
        try:
            results = []
            
            # Search CSV
            csv_result = self.file_tool.search_csv("edge_costs.csv", max_rows=10)
            results.append(csv_result)
            
            # Search JSON
            json_result = self.file_tool.search_json("education_constraints.json")
            results.append(json_result)
            
            return results
        except Exception as e:
            logger.error(f"File search error: {e}")
            return []
    
    def _synthesize(self, state: ResearchState) -> ResearchState:
        """
        Node 4: Synthesize findings from all sources using LLM
        """
        logger.info("Synthesizing findings from all sources")
        
        if not state.all_sources:
            state.errors.append("No sources available for synthesis")
            state.synthesis = "No sources found to answer the query."
            return state
        
        # Build context from all sources
        sources_context = ""
        for i, source in enumerate(state.all_sources, 1):
            sources_context += f"\n--- Source {i} ({source.type}) ---\n"
            sources_context += f"Title: {source.title}\n"
            sources_context += f"Relevance: {source.relevance_score:.2f}\n"
            sources_context += f"Content: {source.content[:500]}...\n"
        
        prompt = f"""You are a research assistant synthesizing information from multiple sources.

Original Query: {state.query}

Sources Retrieved:
{sources_context}

Task: Synthesize these sources into a comprehensive research summary.

Provide:
1. Executive summary (2-3 sentences)
2. Key findings (3-5 bullet points)
3. Recommendations (2-4 actionable items)

Focus on answering the original query directly using evidence from the sources.
"""
        
        try:
            synthesis = self.llm.invoke(prompt)
            state.synthesis = synthesis
            logger.info("Synthesis complete")
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            state.synthesis = "Synthesis failed due to LLM error."
            state.errors.append(f"Synthesis error: {e}")
        
        return state
    
    def _format_report(self, state: ResearchState) -> ResearchState:
        """
        Node 5: Format final report with Pydantic validation (retry logic)
        """
        logger.info("Formatting final research report")
        
        if not state.synthesis:
            state.errors.append("No synthesis available for report formatting")
            return state
        
        # Extract structured data from synthesis using LLM
        prompt = f"""Extract structured information from this research synthesis:

{state.synthesis}

Respond ONLY with valid JSON matching this exact structure:
{{
  "summary": "2-3 sentence executive summary",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "recommendations": ["recommendation 1", "recommendation 2"],
  "confidence": 0.85
}}

Ensure:
- summary: 50-500 characters
- key_findings: 3-7 items
- recommendations: 2-5 items
- confidence: 0.0-1.0 (how confident in findings)
"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                
                # Clean response
                response_clean = response.strip()
                if response_clean.startswith("```"):
                    response_clean = response_clean.split("```")[1]
                    if response_clean.startswith("json"):
                        response_clean = response_clean[4:]
                response_clean = response_clean.strip()
                
                # Parse JSON
                report_dict = json.loads(response_clean)
                
                # Build report
                sources_queried = []
                if state.vector_results:
                    sources_queried.append(SourceType.VECTOR)
                if state.arxiv_results:
                    sources_queried.append(SourceType.ARXIV)
                if state.web_results:
                    sources_queried.append(SourceType.WEB)
                if state.file_results:
                    sources_queried.append(SourceType.FILE)
                
                # Sort sources by relevance
                sorted_sources = sorted(
                    state.all_sources,
                    key=lambda x: x.relevance_score,
                    reverse=True
                )
                
                report = ResearchReport(
                    query=state.query,
                    category=state.analysis.category if state.analysis else QueryCategory.GENERAL,
                    summary=report_dict["summary"],
                    key_findings=report_dict["key_findings"],
                    recommendations=report_dict["recommendations"],
                    confidence=report_dict["confidence"],
                    sources=sorted_sources,
                    sources_queried=sources_queried
                )
                
                state.report = report
                logger.info("Report formatted successfully")
                return state
                
            except Exception as e:
                logger.warning(f"Report formatting attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    prompt += f"\n\nPrevious error: {e}\nPlease fix and provide valid JSON."
                else:
                    state.errors.append(f"Report formatting failed after {max_retries} attempts: {e}")
        
        return state
    
    def research(self, query: str) -> ResearchReport:
        """
        Main entry point: Execute full research pipeline
        
        Args:
            query: Research question from user
            
        Returns:
            ResearchReport with findings
        """
        logger.info(f"Starting research for: {query}")
        
        # Initialize state
        initial_state = ResearchState(query=query)
        
        # Execute graph
        final_state = ResearchState(**self.graph.invoke(initial_state)) # type: ignore
        
        # Check for errors
        if final_state.errors:
            logger.warning(f"Research completed with {len(final_state.errors)} errors:")
            for error in final_state.errors:
                logger.warning(f"  - {error}")
        
        if not final_state.report:
            logger.error("Research failed - no report generated")
            # Return minimal error report
            return ResearchReport(
                query=query,
                category=QueryCategory.GENERAL,
                summary="Research failed due to errors. See logs for details.",
                key_findings=["Error occurred during research"],
                recommendations=["Check system logs", "Retry with simpler query"],
                confidence=0.0,
                sources=[],
                sources_queried=[]
            )
        
        logger.info("Research complete")
        return final_state.report


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    agent = ResearchAgent()
    
    # Test queries
    queries = [
        "What are current best practices for deploying LLMs on edge devices in education?",
        "Explain the transformer attention mechanism from a research perspective",
        "Compare costs of Jetson vs cloud for 24/7 AI workloads"
    ]
    
    for query in queries[:1]:  # Test just first query
        print("\n" + "="*80)
        print(f"QUERY: {query}")
        print("="*80)
        
        report = agent.research(query)
        
        print("\n" + report.to_markdown())
        
        # Save to file
        output_file = Path("research_report.md")
        output_file.write_text(report.to_markdown())
        print(f"\n✓ Report saved to: {output_file}")