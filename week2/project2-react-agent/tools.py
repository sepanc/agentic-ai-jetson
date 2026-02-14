"""
Tool definitions following the 3-part interface:
1. Name (string)
2. Description (string) - what LLM reads to decide when to use
3. Function (callable) - actual implementation
"""

from datetime import datetime
from typing import Callable
import re

class Tool:
    """Base tool with name, description, and function."""
    
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
    
    def execute(self, input_str: str) -> str:
        """Execute tool with input and return string result."""
        try:
            # Universal input sanitization
            cleaned_input = self._sanitize_input(input_str)
            result = self.func(cleaned_input)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def _sanitize_input(input_str: str) -> str:
        """
        Sanitize LLM input by removing common issues:
        - Surrounding quotes (", ', `, etc.)
        - Extra whitespace
        - Newlines
        """
        if not isinstance(input_str, str):
            input_str = str(input_str)
        
        # Strip whitespace and newlines
        cleaned = input_str.strip()
        
        # Remove surrounding quotes (single, double, backticks)
        quote_chars = ['"', "'", '`']
        for quote in quote_chars:
            if cleaned.startswith(quote) and cleaned.endswith(quote):
                cleaned = cleaned[1:-1].strip()
        
        return cleaned


def calculator_fn(expression: str) -> float:
    """
    Safely evaluate mathematical expressions.
    Handles common LLM formatting issues.
    """
    # Already sanitized by Tool.execute(), but double-check
    expression = expression.strip()
    
    # Security: Only allow safe characters
    allowed_pattern = r'^[\d\s\+\-\*/\(\)\.\*]+$'
    if not re.match(allowed_pattern, expression):
        # Provide helpful error with allowed characters
        raise ValueError(
            f"Expression contains invalid characters. "
            f"Allowed: numbers, +, -, *, /, (, ), . "
            f"Got: '{expression}'"
        )
    
    # Evaluate safely (restricted environment)
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Cannot evaluate '{expression}': {e}")


DOCUMENTS = {
    "exam_info": """
    The NCP-AAI certification exam is scheduled for February 21, 2026.
    The exam covers 5 modules: Agent Abstraction, Structured Output, 
    Retrieval Mechanisms, Multi-Agent Systems, and Final Assessment.
    The exam duration is 90 minutes and costs $199.
    """,
    "project_info": """
    Week 1: RAG fundamentals (completed)
    Week 2: Multi-agent systems and vision integration
    Week 3: Advanced orchestration patterns
    Total project timeline: 90 days to job-ready
    """
}

def search_document_fn(query: str) -> str:
    """
    Search pre-loaded documents.
    Returns helpful message if nothing found.
    """
    query = query.lower().strip()
    
    if not query:
        return "Error: Empty search query"
    
    # Search across all documents
    results = []
    for doc_name, content in DOCUMENTS.items():
        # Check if any query word appears in content
        if any(keyword in content.lower() for keyword in query.split()):
            results.append(f"[{doc_name}]: {content.strip()}")
    
    if results:
        return "\n\n".join(results)
    else:
        return (
            f"No documents found matching '{query}'. "
            f"Available documents: {', '.join(DOCUMENTS.keys())}"
        )


def get_datetime_fn(format_type: str = "date") -> str:
    """
    Get current date/time in specified format.
    Default to date to avoid confusion.
    """
    format_type = format_type.lower().strip()
    now = datetime.now()
    
    if format_type in ["date", "d"]:
        return now.strftime("%Y-%m-%d")
    elif format_type in ["time", "t"]:
        return now.strftime("%H:%M:%S")
    elif format_type in ["iso", "full"]:
        return now.isoformat()
    else:
        # Default to date if unclear
        return now.strftime("%Y-%m-%d")


WEB_DATABASE = {
    "nvidia stock": "NVIDIA (NVDA) trading at $138.45",
    "weather": "San Francisco: Sunny, 72°F",
    "python": "Python 3.12 latest stable version",
}

def web_search_fn(query: str) -> str:
    """
    Simulated web search.
    Returns helpful message if no results.
    """
    query = query.lower().strip()
    
    if not query:
        return "Error: Empty search query"
    
    # Find best match
    for key, response in WEB_DATABASE.items():
        if any(keyword in query for keyword in key.split()):
            return response
    
    return (
        f"No web results for '{query}'. "
        f"Try: {', '.join(WEB_DATABASE.keys())}"
    )


# Tool definitions with comprehensive descriptions
AVAILABLE_TOOLS = [
    Tool(
        name="calculator",
        description="""
        Performs arithmetic calculations.
        
        WHEN TO USE: Math operations (addition, subtraction, multiplication, division, percentages).
        
        INPUT FORMAT: Math expression WITHOUT quotes. Examples:
          - 88 * 0.25
          - 100 / 4
          - (31 - 30) + 21
        
        CRITICAL: Input should NOT have surrounding quotes.
        WRONG: "88 * 0.25"
        RIGHT: 88 * 0.25
        
        OUTPUT: Numeric result as string.
        
        COMMON MISTAKES TO AVOID:
        - Don't put quotes around the expression
        - Don't use % symbol (use * 0.01 instead)
        """,
        func=calculator_fn
    ),
    
    Tool(
        name="search_document",
        description="""
        Search pre-loaded documents for information.
        
        WHEN TO USE: Questions about project timeline, exam details, certification info.
        
        INPUT FORMAT: Keywords without quotes. Examples:
          - exam date
          - week 2
          - certification cost
        
        OUTPUT: Matching document excerpts or "No documents found".
        
        AVAILABLE DOCUMENTS: exam_info, project_info
        """,
        func=search_document_fn
    ),
    
    Tool(
        name="get_datetime",
        description="""
        Get current date and time.
        
        WHEN TO USE: Need today's date, current time, or timestamp.
        
        INPUT FORMAT: Format type (optional). Examples:
          - date  → 2026-01-30
          - time  → 15:30:45
          - iso   → 2026-01-30T15:30:45.123456
          - (empty) → defaults to date
        
        OUTPUT: Formatted date/time string.
        
        NOTE: Call this ONCE per task, not repeatedly.
        """,
        func=get_datetime_fn
    ),
    
    Tool(
        name="web_search",
        description="""
        Search the web (simulated).
        
        WHEN TO USE: Current info, stock prices, weather.
        
        INPUT FORMAT: Natural query. Examples:
          - nvidia stock
          - weather
          - python
        
        OUTPUT: Search result or "No results".
        """,
        func=web_search_fn
    ),
]


def get_tool_by_name(tool_name: str) -> Tool:
    """Get tool by name, case-insensitive."""
    tool_name = tool_name.strip().lower()
    for tool in AVAILABLE_TOOLS:
        if tool.name.lower() == tool_name:
            return tool
    available = [t.name for t in AVAILABLE_TOOLS]
    raise ValueError(f"Tool '{tool_name}' not found. Available: {', '.join(available)}")


def get_tools_description() -> str:
    """Get formatted description of all tools."""
    descriptions = []
    for tool in AVAILABLE_TOOLS:
        descriptions.append(f"Tool: {tool.name}\n{tool.description.strip()}")
    return "\n\n".join(descriptions)