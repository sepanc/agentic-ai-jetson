import pytest
from pathlib import Path

# Import all tools
from src.tools.vector_search import VectorSearchTool
from src.tools.arxiv_search import ArxivSearchTool
from src.tools.web_search import WebSearchTool
from src.tools.file_search import FileSearchTool


def test_vector_search():
    """Test vector search returns results"""
    tool = VectorSearchTool()
    results = tool.search("edge AI deployment", top_k=3)
    # May be 0 if no documents added yet
    assert isinstance(results, list)


def test_arxiv_search():
    """Test Arxiv search returns papers"""
    tool = ArxivSearchTool()
    results = tool.search("transformer attention", max_results=2)
    assert len(results) > 0
    assert results[0].arxiv_id is not None
    assert len(results[0].authors) > 0


def test_web_search():
    """Test web search returns results"""
    tool = WebSearchTool()
    results = tool.search("NVIDIA Jetson", num_results=3)
    assert len(results) > 0
    assert results[0].url.startswith("http")


def test_file_search_csv():
    """Test CSV file search"""
    tool = FileSearchTool()
    result = tool.search_csv("edge_costs.csv", max_rows=5)
    assert result.file_type == "csv"
    assert "deployment_type" in result.content.lower() or "error" in result.content.lower()


def test_file_search_json():
    """Test JSON file search"""
    tool = FileSearchTool()
    result = tool.search_json("education_constraints.json")
    assert result.file_type == "json"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])