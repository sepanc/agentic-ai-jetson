import httpx
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class SerperSearch:
    """Direct Serper API integration via HTTP"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        self.base_url = "https://google.serper.dev/search"
        
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search using Serper API
        
        Args:
            query: Search query string
            num_results: Number of results to return (max 10)
            
        Returns:
            List of search results with title, link, snippet
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": num_results
        }
        
        try:
            response = httpx.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=10.0
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract organic results
            organic = data.get("organic", [])
            
            results = []
            for item in organic[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "position": item.get("position", 0)
                })
            
            return results
            
        except httpx.HTTPError as e:
            print(f"Serper API error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []


# Test function
if __name__ == "__main__":
    searcher = SerperSearch()
    results = searcher.search("NVIDIA Jetson edge AI deployment", num_results=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   {result['link']}")
        print(f"   {result['snippet'][:100]}...")