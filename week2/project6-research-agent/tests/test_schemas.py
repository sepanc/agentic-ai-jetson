import pytest
from src.models.schemas import (
    QueryAnalysis, QueryCategory, ResearchReport, 
    ResearchState, Source, SourceType, WebSource
)
from pydantic import ValidationError


def test_query_analysis_valid():
    """Test valid QueryAnalysis creation"""
    analysis = QueryAnalysis(
        category=QueryCategory.ACADEMIC,
        reasoning="Research-focused query about LLMs",
        keywords=["transformer", "attention"],
        use_arxiv=True,
        use_web=False
    )
    assert analysis.category == QueryCategory.ACADEMIC
    assert analysis.use_arxiv is True


def test_query_analysis_invalid_category():
    """Test that invalid category raises ValidationError"""
    with pytest.raises(ValidationError):
        QueryAnalysis(
            category="invalid_category",  # Should fail
            reasoning="Test",
            keywords=["test"],
            use_arxiv=True,
            use_web=False
        )


def test_research_report_confidence_bounds():
    """Test confidence score validation (0-1)"""
    # Valid confidence
    report = ResearchReport(
        query="test",
        category=QueryCategory.GENERAL,
        summary="Valid summary with enough characters to meet minimum",
        key_findings=["Finding 1", "Finding 2", "Finding 3"],
        sources=[
            WebSource(
                title="Test",
                content="Content",
                relevance_score=0.9,
                url="http://test.com",
                position=1,
                snippet="Snippet"
            )
        ],
        recommendations=["Rec 1", "Rec 2"],
        confidence=0.85,
        sources_queried=[SourceType.WEB]
    )
    assert report.confidence == 0.85
    
    # Invalid confidence (>1)
    with pytest.raises(ValidationError):
        ResearchReport(
            query="test",
            category=QueryCategory.GENERAL,
            summary="Valid summary",
            key_findings=["Finding 1", "Finding 2", "Finding 3"],
            sources=[
                WebSource(
                    title="Test",
                    content="Content",
                    relevance_score=0.9,
                    url="http://test.com",
                    position=1,
                    snippet="Snippet"
                )
            ],
            recommendations=["Rec 1", "Rec 2"],
            confidence=1.5,  # Should fail
            sources_queried=[SourceType.WEB]
        )


def test_research_state_initialization():
    """Test ResearchState can be initialized with just query"""
    state = ResearchState(query="What is edge AI?")
    assert state.query == "What is edge AI?"
    assert state.analysis is None
    assert len(state.vector_results) == 0


def test_markdown_formatting():
    """Test ResearchReport markdown output"""
    report = ResearchReport(
        query="Test query",
        category=QueryCategory.GENERAL,
        summary="This is a test summary that meets the minimum character requirement for validation.",
        key_findings=["Finding 1", "Finding 2", "Finding 3"],
        sources=[
            WebSource(
                title="Test Source",
                content="Test content for source",
                relevance_score=0.9,
                url="http://example.com",
                position=1,
                snippet="Test snippet"
            )
        ],
        recommendations=["Recommendation 1", "Recommendation 2"],
        confidence=0.85,
        sources_queried=[SourceType.WEB]
    )
    
    markdown = report.to_markdown()
    assert "# Research Report" in markdown
    assert "Test query" in markdown
    assert "Finding 1" in markdown
    assert "Recommendation 1" in markdown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])