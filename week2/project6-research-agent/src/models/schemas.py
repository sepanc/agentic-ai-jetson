from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from datetime import datetime
from enum import Enum


# ============================================================================
# Source Type Enums
# ============================================================================

class SourceType(str, Enum):
    """Types of information sources"""
    VECTOR = "vector"
    ARXIV = "arxiv"
    WEB = "web"
    FILE = "file"


class QueryCategory(str, Enum):
    """Classification of user query intent"""
    ACADEMIC = "academic"  # Research papers, citations, theoretical
    GENERAL = "general"    # Current trends, news, best practices
    HYBRID = "hybrid"      # Mix of both


# ============================================================================
# Individual Source Models
# ============================================================================

class Source(BaseModel):
    """Single information source with content and metadata"""
    type: SourceType
    title: str
    content: str
    relevance_score: float = Field(ge=0, le=1, description="How relevant to query (0-1)")
    url: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class VectorSource(Source):
    """Source from vector database (ChromaDB)"""
    type: Literal[SourceType.VECTOR] = SourceType.VECTOR
    document_id: str
    chunk_id: Optional[str] = None
    distance: float = Field(description="Vector distance (lower = more similar)")


class ArxivSource(Source):
    """Source from Arxiv API"""
    type: Literal[SourceType.ARXIV] = SourceType.ARXIV
    arxiv_id: str
    authors: List[str] = Field(default_factory=list)
    published: Optional[str] = None
    categories: List[str] = Field(default_factory=list)


class WebSource(Source):
    """Source from Serper web search"""
    type: Literal[SourceType.WEB] = SourceType.WEB
    position: int = Field(description="Search result ranking")
    snippet: str = Field(description="Search result snippet")


class FileSource(Source):
    """Source from local file system (CSV/JSON)"""
    type: Literal[SourceType.FILE] = SourceType.FILE
    file_path: str
    file_type: Literal["csv", "json"]
    row_count: Optional[int] = None


# ============================================================================
# Query Analysis
# ============================================================================

class QueryAnalysis(BaseModel):
    """LLM analysis of user query to determine retrieval strategy"""
    category: QueryCategory
    reasoning: str = Field(description="Why this category was chosen")
    keywords: List[str] = Field(description="Key terms for search")
    use_arxiv: bool = Field(description="Should query Arxiv API")
    use_web: bool = Field(description="Should query Serper API")
    use_vector: bool = Field(default=True, description="Should query vector DB")
    use_files: bool = Field(default=True, description="Should query local files")
    
    class Config:
        use_enum_values = True


# ============================================================================
# Retrieval Plan
# ============================================================================

class RetrievalPlan(BaseModel):
    """Plan for which sources to query (generated from QueryAnalysis)"""
    query: str
    category: QueryCategory
    tools_to_use: List[SourceType] = Field(description="Which retrieval tools to invoke")
    search_terms: Dict[SourceType, str] = Field(
        description="Customized search terms per source type",
        default_factory=dict
    )
    
    class Config:
        use_enum_values = True


# ============================================================================
# Research Report (Final Output)
# ============================================================================

class ResearchReport(BaseModel):
    """Final structured research report delivered to user"""
    query: str
    category: QueryCategory
    
    # Executive Summary
    summary: str = Field(
        description="2-3 sentence high-level overview of findings",
        min_length=50,
        max_length=500
    )
    
    # Key Findings
    key_findings: List[str] = Field(
        description="3-5 bullet points of main insights",
        min_items=3,
        max_items=7
    )
    
    # Source Evidence
    sources: List[Source] = Field(
        description="All sources used, sorted by relevance",
        min_items=1
    )
    
    # Recommendations
    recommendations: List[str] = Field(
        description="2-4 actionable recommendations based on findings",
        min_items=2,
        max_items=5
    )
    
    # Metadata
    confidence: float = Field(
        ge=0,
        le=1,
        description="Overall confidence in findings (0-1)"
    )
    sources_queried: List[SourceType] = Field(
        description="Which source types were actually used"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    
    class Config:
        use_enum_values = True
    
    def to_markdown(self) -> str:
        """Format report as readable markdown"""
        md = f"# Research Report\n\n"
        md += f"**Query:** {self.query}\n"
        md += f"**Category:** {self.category}\n"
        md += f"**Confidence:** {self.confidence:.1%}\n"
        md += f"**Generated:** {self.timestamp}\n\n"
        
        md += f"## Summary\n{self.summary}\n\n"
        
        md += f"## Key Findings\n"
        for i, finding in enumerate(self.key_findings, 1):
            md += f"{i}. {finding}\n"
        md += "\n"
        
        md += f"## Recommendations\n"
        for i, rec in enumerate(self.recommendations, 1):
            md += f"{i}. {rec}\n"
        md += "\n"
        
        md += f"## Sources ({len(self.sources)})\n"
        for i, source in enumerate(self.sources, 1):
            md += f"\n### {i}. [{source.type.upper()}] {source.title}\n"
            if source.url:
                md += f"**URL:** {source.url}\n"
            md += f"**Relevance:** {source.relevance_score:.1%}\n"
            md += f"{source.content[:200]}...\n"
        
        return md


# ============================================================================
# LangGraph State Management
# ============================================================================

class ResearchState(BaseModel):
    """State passed between LangGraph nodes"""
    
    # Input
    query: str
    
    # Query Analysis Stage
    analysis: Optional[QueryAnalysis] = None
    plan: Optional[RetrievalPlan] = None
    
    # Retrieval Stage
    vector_results: List[VectorSource] = Field(default_factory=list)
    arxiv_results: List[ArxivSource] = Field(default_factory=list)
    web_results: List[WebSource] = Field(default_factory=list)
    file_results: List[FileSource] = Field(default_factory=list)
    
    # Synthesis Stage
    all_sources: List[Source] = Field(default_factory=list)
    synthesis: Optional[str] = None
    
    # Output Stage
    report: Optional[ResearchReport] = None
    
    # Metadata
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# Validation Test Examples
# ============================================================================

if __name__ == "__main__":
    # Test QueryAnalysis
    analysis = QueryAnalysis(
        category=QueryCategory.ACADEMIC,
        reasoning="User asking about research on transformer architecture",
        keywords=["transformer", "attention mechanism", "BERT"],
        use_arxiv=True,
        use_web=False
    )
    print("✓ QueryAnalysis validation passed")
    print(analysis.model_dump_json(indent=2))
    
    # Test ResearchReport
    report = ResearchReport(
        query="What are best practices for edge AI deployment?",
        category=QueryCategory.GENERAL,
        summary="Edge AI deployment requires balancing cost, latency, and compliance. Key factors include hardware selection, model optimization, and regulatory constraints.",
        key_findings=[
            "NVIDIA Jetson offers 60% cost savings vs cloud for 24/7 workloads",
            "FERPA compliance requires on-premises data processing in education",
            "Quantization to INT8 reduces model size by 75% with <2% accuracy loss"
        ],
        sources=[
            WebSource(
                title="Edge AI Computing Guide",
                content="Comprehensive guide to deploying AI at the edge...",
                relevance_score=0.95,
                url="https://example.com/edge-ai",
                position=1,
                snippet="Edge computing brings AI processing closer to data sources..."
            )
        ],
        recommendations=[
            "Start with Jetson Orin Nano for prototyping ($499)",
            "Implement model quantization before deployment",
            "Plan for offline-first operation in rural deployments"
        ],
        confidence=0.85,
        sources_queried=[SourceType.WEB, SourceType.VECTOR, SourceType.FILE]
    )
    print("\n✓ ResearchReport validation passed")
    print(report.to_markdown())