"""
File search tool for local CSV and JSON data sources.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

from ..models.schemas import FileSource
from ..config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileSearchTool:
    """Search local CSV and JSON files for relevant data"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or settings.metrics_dir
    
    def search_csv(
        self,
        filename: str,
        search_column: str = None,
        search_term: str = None,
        max_rows: int = 10
    ) -> FileSource:
        """
        Search CSV file and return relevant rows
        
        Args:
            filename: CSV filename (in data_dir)
            search_column: Column to search in (None = return all)
            search_term: Term to search for (None = return all)
            max_rows: Maximum rows to include in content
            
        Returns:
            FileSource object with CSV data
        """
        file_path = self.data_dir / filename
        
        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Filter rows if search criteria provided
            if search_column and search_term:
                filtered_rows = [
                    row for row in rows
                    if search_term.lower() in str(row.get(search_column, '')).lower()
                ]
            else:
                filtered_rows = rows
            
            # Limit rows
            filtered_rows = filtered_rows[:max_rows]
            
            # Format content as table
            if filtered_rows:
                headers = list(filtered_rows[0].keys())
                content = "| " + " | ".join(headers) + " |\n"
                content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                
                for row in filtered_rows:
                    content += "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n"
            else:
                content = "No matching rows found"
            
            # Calculate relevance (simple: 1.0 if exact match, 0.7 if partial)
            relevance = 1.0 if search_term and any(
                search_term.lower() == str(row.get(search_column, '')).lower()
                for row in filtered_rows
            ) else 0.7
            
            source = FileSource(
                title=f"CSV: {filename}",
                content=content,
                relevance_score=relevance,
                file_path=str(file_path),
                file_type="csv",
                row_count=len(filtered_rows),
                metadata={
                    "total_rows": len(rows),
                    "filtered_rows": len(filtered_rows),
                    "search_column": search_column,
                    "search_term": search_term,
                    "columns": list(rows[0].keys()) if rows else []
                }
            )
            
            logger.info(f"CSV search: {filename} returned {len(filtered_rows)} rows")
            return source
            
        except Exception as e:
            logger.error(f"CSV search failed for {filename}: {e}")
            return FileSource(
                title=f"CSV: {filename} (error)",
                content=f"Error reading file: {e}",
                relevance_score=0.0,
                file_path=str(file_path),
                file_type="csv"
            )
    
    def search_json(
        self,
        filename: str,
        json_path: List[str] = None,
        search_term: str = None
    ) -> FileSource:
        """
        Search JSON file and return relevant data
        
        Args:
            filename: JSON filename (in data_dir)
            json_path: Path to nested key (e.g., ["regulatory_compliance", "FERPA"])
            search_term: Term to search for in values (None = return all)
            
        Returns:
            FileSource object with JSON data
        """
        file_path = self.data_dir / filename
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Navigate to nested path if provided
            current = data
            if json_path:
                for key in json_path:
                    current = current.get(key, {})
            
            # Filter by search term if provided
            if search_term and isinstance(current, dict):
                filtered = self._filter_dict_by_term(current, search_term)
            else:
                filtered = current
            
            # Format content
            content = json.dumps(filtered, indent=2)
            
            # Calculate relevance
            relevance = 0.9 if json_path else 0.7
            
            source = FileSource(
                title=f"JSON: {filename}",
                content=content,
                relevance_score=relevance,
                file_path=str(file_path),
                file_type="json",
                metadata={
                    "json_path": json_path,
                    "search_term": search_term,
                    "data_type": type(filtered).__name__
                }
            )
            
            logger.info(f"JSON search: {filename} returned data")
            return source
            
        except Exception as e:
            logger.error(f"JSON search failed for {filename}: {e}")
            return FileSource(
                title=f"JSON: {filename} (error)",
                content=f"Error reading file: {e}",
                relevance_score=0.0,
                file_path=str(file_path),
                file_type="json"
            )
    
    def _filter_dict_by_term(self, data: Dict, term: str) -> Dict:
        """Recursively filter dict by search term"""
        filtered = {}
        term_lower = term.lower()
        
        for key, value in data.items():
            if isinstance(value, dict):
                nested_filtered = self._filter_dict_by_term(value, term)
                if nested_filtered:
                    filtered[key] = nested_filtered
            elif term_lower in str(value).lower() or term_lower in str(key).lower():
                filtered[key] = value
        
        return filtered
    
    def list_available_files(self) -> Dict[str, List[str]]:
        """List all CSV and JSON files in data directory"""
        csv_files = [f.name for f in self.data_dir.glob("*.csv")]
        json_files = [f.name for f in self.data_dir.glob("*.json")]
        
        return {
            "csv": csv_files,
            "json": json_files
        }


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    tool = FileSearchTool()
    
    print("="*80)
    print("Available files:")
    print("="*80)
    files = tool.list_available_files()
    print(f"CSV files: {files['csv']}")
    print(f"JSON files: {files['json']}")
    
    # Test CSV search
    print("\n" + "="*80)
    print("Testing CSV search...")
    print("="*80)
    
    csv_result = tool.search_csv(
        filename="edge_costs.csv",
        search_column="deployment_type",
        search_term="edge",
        max_rows=5
    )
    
    print(f"\nTitle: {csv_result.title}")
    print(f"Relevance: {csv_result.relevance_score:.2f}")
    print(f"Rows found: {csv_result.row_count}")
    print(f"\nContent:\n{csv_result.content}")
    
    # Test JSON search
    print("\n" + "="*80)
    print("Testing JSON search...")
    print("="*80)
    
    json_result = tool.search_json(
        filename="education_constraints.json",
        json_path=["regulatory_compliance", "FERPA"]
    )
    
    print(f"\nTitle: {json_result.title}")
    print(f"Relevance: {json_result.relevance_score:.2f}")
    print(f"\nContent:\n{json_result.content[:300]}...")