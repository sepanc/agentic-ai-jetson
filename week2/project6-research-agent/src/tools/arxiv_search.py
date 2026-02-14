"""
Arxiv API search tool for academic papers.
"""

import arxiv
from typing import List
import logging

from ..models.schemas import ArxivSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArxivSearchTool:
    """Search Arxiv for academic papers"""
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.client = arxiv.Client()
    
    def search(
        self,
        query: str,
        max_results: int = None,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
    ) -> List[ArxivSource]:
        """
        Search Arxiv for papers matching query
        
        Args:
            query: Search query (e.g., "transformer attention mechanism")
            max_results: Override default max results
            sort_by: Sort criterion (Relevance, LastUpdatedDate, SubmittedDate)
            
        Returns:
            List of ArxivSource objects
        """
        max_results = max_results or self.max_results
        
        try:
            # Create search
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_by
            )
            
            # Execute search
            results = list(self.client.results(search))
            
            # Convert to ArxivSource objects
            sources = []
            for i, paper in enumerate(results):
                # Calculate relevance score (approximate - Arxiv doesn't provide this)
                # Use position in results as proxy (1st result = 1.0, decreasing)
                relevance_score = max(0.5, 1.0 - (i * 0.1))
                
                source = ArxivSource(
                    title=paper.title,
                    content=paper.summary,
                    relevance_score=relevance_score,
                    url=paper.entry_id,
                    arxiv_id=paper.get_short_id(),
                    authors=[author.name for author in paper.authors],
                    published=paper.published.isoformat() if paper.published else None,
                    categories=paper.categories,
                    metadata={
                        "pdf_url": paper.pdf_url,
                        "updated": paper.updated.isoformat() if paper.updated else None,
                        "comment": paper.comment,
                        "journal_ref": paper.journal_ref,
                        "doi": paper.doi,
                        "primary_category": paper.primary_category
                    }
                )
                sources.append(source)
            
            logger.info(f"Arxiv search returned {len(sources)} results for: {query[:50]}...")
            return sources
            
        except Exception as e:
            logger.error(f"Arxiv search failed: {e}")
            return []
    
    def get_paper_by_id(self, arxiv_id: str) -> ArxivSource:
        """
        Retrieve specific paper by Arxiv ID
        
        Args:
            arxiv_id: Arxiv paper ID (e.g., "1706.03762" for Transformer paper)
            
        Returns:
            ArxivSource object or None if not found
        """
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(self.client.results(search))
            
            source = ArxivSource(
                title=paper.title,
                content=paper.summary,
                relevance_score=1.0,  # Direct ID lookup = 100% relevant
                url=paper.entry_id,
                arxiv_id=paper.get_short_id(),
                authors=[author.name for author in paper.authors],
                published=paper.published.isoformat() if paper.published else None,
                categories=paper.categories,
                metadata={
                    "pdf_url": paper.pdf_url,
                    "updated": paper.updated.isoformat() if paper.updated else None
                }
            )
            
            logger.info(f"Retrieved Arxiv paper: {arxiv_id}")
            return source
            
        except Exception as e:
            logger.error(f"Failed to retrieve Arxiv paper {arxiv_id}: {e}")
            return None


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    tool = ArxivSearchTool()
    
    print("="*80)
    print("Testing Arxiv search...")
    print("="*80)
    
    # Test 1: Search query
    query = "transformer attention mechanism deep learning"
    results = tool.search(query, max_results=3)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} papers:\n")
    
    for i, paper in enumerate(results, 1):
        print(f"{i}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print(f"   Arxiv ID: {paper.arxiv_id}")
        print(f"   Published: {paper.published}")
        print(f"   Relevance: {paper.relevance_score:.2f}")
        print(f"   URL: {paper.url}")
        print(f"   Summary: {paper.content[:150]}...\n")
    
    # Test 2: Get specific paper by ID
    print("\n" + "="*80)
    print("Testing paper retrieval by ID...")
    print("="*80)
    
    transformer_paper = tool.get_paper_by_id("1706.03762")
    if transformer_paper:
        print(f"\nRetrieved: {transformer_paper.title}")
        print(f"Authors: {', '.join(transformer_paper.authors)}")
        print(f"PDF: {transformer_paper.metadata['pdf_url']}")