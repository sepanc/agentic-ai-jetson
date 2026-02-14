"""
Web search tool using Serper API (Google search results).
"""

import httpx
from typing import List
import logging

from ..models.schemas import WebSource
from ..config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSearchTool:
    """Search the web using Serper API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.serper_api_key
        self.base_url = "https://google.serper.dev/search"
    
    def search(
        self,
        query: str,
        num_results: int = 5,
        timeout: float = 10.0
    ) -> List[WebSource]:
        """
        Search web using Serper API
        
        Args:
            query: Search query string
            num_results: Number of results to return (max 10)
            timeout: Request timeout in seconds
            
        Returns:
            List of WebSource objects
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": min(num_results, 10)  # Serper max is 10
        }
        
        try:
            response = httpx.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            data = response.json()
            organic = data.get("organic", [])
            
            # Convert to WebSource objects
            sources = []
            for item in organic[:num_results]:
                # Calculate relevance based on position (1st = 1.0, decreasing)
                position = item.get("position", 0)
                relevance_score = max(0.5, 1.0 - (position * 0.05))
                
                source = WebSource(
                    title=item.get("title", "Untitled"),
                    content=item.get("snippet", ""),
                    relevance_score=relevance_score,
                    url=item.get("link", ""),
                    position=position,
                    snippet=item.get("snippet", ""),
                    metadata={
                        "date": item.get("date"),
                        "sitelinks": item.get("sitelinks", []),
                        "rich_snippet": item.get("richSnippet")
                    }
                )
                sources.append(source)
            
            logger.info(f"Web search returned {len(sources)} results for: {query[:50]}...")
            return sources
            
        except httpx.HTTPError as e:
            logger.error(f"Serper API HTTP error: {e}")
            return []
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    tool = WebSearchTool()
    
    print("="*80)
    print("Testing web search (Serper API)...")
    print("="*80)
    
    query = "NVIDIA Jetson edge AI deployment best practices"
    results = tool.search(query, num_results=5)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.title}")
        print(f"   Position: {result.position}")
        print(f"   Relevance: {result.relevance_score:.2f}")
        print(f"   URL: {result.url}")
        print(f"   Snippet: {result.snippet[:100]}...\n")