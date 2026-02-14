"""
Extended tool set for multi-tool integration.

8 diverse tool types:
1. Calculator - Math operations
2. Document Search - Local knowledge
3. DateTime - Current date/time
4. Web Search - External info (simulated)
5. Database Query - SQL operations (simulated)
6. API Call - REST endpoints (simulated)
7. File Write - Save data to files
8. File Read - Load data from files
"""

from datetime import datetime, timedelta
from typing import Callable, Dict, Any, List
import re
import json
import os


class Tool:
    """Base tool with name, description, and function."""
    
    def __init__(self, name: str, description: str, func: Callable, category: str = "general"):
        self.name = name
        self.description = description
        self.func = func
        self.category = category  # For grouping tools
    
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
        """Remove quotes, whitespace, newlines."""
        if not isinstance(input_str, str):
            input_str = str(input_str)
        
        cleaned = input_str.strip()
        
        # Remove surrounding quotes
        for quote in ['"', "'", '`']:
            if cleaned.startswith(quote) and cleaned.endswith(quote):
                cleaned = cleaned[1:-1].strip()
        
        return cleaned


# ============================================================================
# CATEGORY 1: COMPUTATION TOOLS
# ============================================================================

def calculator_fn(expression: str) -> float:
    """Safely evaluate mathematical expressions."""
    expression = expression.strip()
    
    if not re.match(r'^[\d\s\+\-\*/\(\)\.\*]+$', expression):
        raise ValueError(f"Invalid characters. Allowed: numbers, +, -, *, /, (, ), .")
    
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Cannot evaluate: {e}")

def calculate_days_until_fn(target_date: str) -> str:
    """
    Calculate days from today until target date.
    
    Input format: "YYYY-MM-DD" (e.g., "2026-02-21")
    Output: Number of days
    """
    try:
        target = datetime.strptime(target_date.strip(), "%Y-%m-%d")
        today = datetime.now()
        delta = target - today
        days = delta.days
        
        return f"{days} days until {target_date}"
    except Exception as e:
        return f"Error: Invalid date format. Use YYYY-MM-DD. {e}"


# ============================================================================
# CATEGORY 2: INFORMATION RETRIEVAL TOOLS
# ============================================================================

DOCUMENTS = {
    "exam_info": """
    The NCP-AAI certification exam is scheduled for February 21, 2026.
    The exam covers 5 modules and costs $199.
    """,
    "project_info": """
    Week 1: RAG fundamentals (completed)
    Week 2: Multi-agent systems and vision integration (in progress)
    Week 3: Advanced orchestration patterns
    """,
    "customer_data": """
    Customer segments:
    - Enterprise: 45 customers, avg spend $15,000/month
    - SMB: 120 customers, avg spend $2,500/month  
    - Startup: 380 customers, avg spend $500/month
    """
}

def search_document_fn(query: str) -> str:
    """Search pre-loaded documents."""
    query = query.lower().strip()
    
    if not query:
        return "Error: Empty search query"
    
    results = []
    for doc_name, content in DOCUMENTS.items():
        if any(keyword in content.lower() for keyword in query.split()):
            results.append(f"[{doc_name}]: {content.strip()}")
    
    if results:
        return "\n\n".join(results)
    else:
        return f"No documents found matching '{query}'"


def get_datetime_fn(format_type: str = "date") -> str:
    """Get current date/time."""
    format_type = format_type.lower().strip()
    now = datetime.now()
    
    if format_type in ["date", "d"]:
        return now.strftime("%Y-%m-%d")
    elif format_type in ["time", "t"]:
        return now.strftime("%H:%M:%S")
    else:
        return now.isoformat()


def web_search_fn(query: str) -> str:
    """Simulated web search."""
    query = query.lower().strip()
    
    web_db = {
        "nvidia stock": "NVIDIA (NVDA) trading at $138.45",
        "weather": "San Francisco: Sunny, 72°F",
        "ai news": "OpenAI announces GPT-5, Google releases Gemini 2.0"
    }
    
    for key, response in web_db.items():
        if any(keyword in query for keyword in key.split()):
            return response
    
    return f"No web results for '{query}'"


# ============================================================================
# CATEGORY 3: DATABASE TOOLS (Simulated)
# ============================================================================

# Simulated database tables
DATABASE = {
    "customers": [
        {"id": 1, "name": "Acme Corp", "segment": "Enterprise", "spend": 15000},
        {"id": 2, "name": "Tech Startup Inc", "segment": "Startup", "spend": 500},
        {"id": 3, "name": "MidSize LLC", "segment": "SMB", "spend": 2500},
        {"id": 4, "name": "Big Enterprise", "segment": "Enterprise", "spend": 18000},
        {"id": 5, "name": "Small Biz", "segment": "SMB", "spend": 1200},
    ],
    "orders": [
        {"order_id": 101, "customer_id": 1, "amount": 5000, "date": "2026-01-15"},
        {"order_id": 102, "customer_id": 1, "amount": 3000, "date": "2026-01-20"},
        {"order_id": 103, "customer_id": 4, "amount": 8000, "date": "2026-01-25"},
        {"order_id": 104, "customer_id": 3, "amount": 1500, "date": "2026-01-28"},
    ]
}

def database_query_fn(query: str) -> str:
    """
    Simulated SQL query execution.
    
    Supports simple queries like:
    - "SELECT * FROM customers"
    - "SELECT * FROM customers WHERE segment = 'Enterprise'"
    - "SELECT * FROM orders WHERE amount > 1000"
    """
    query = query.strip()
    
    # Parse simple SELECT queries
    if not query.upper().startswith("SELECT"):
        return "Error: Only SELECT queries supported"
    
    # Extract table name
    table_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
    if not table_match:
        return "Error: Cannot parse table name"
    
    table_name = table_match.group(1).lower()
    
    if table_name not in DATABASE:
        return f"Error: Table '{table_name}' not found. Available: {list(DATABASE.keys())}"
    
    # Get data
    data = DATABASE[table_name]
    
    # Simple WHERE filter (very basic parsing)
    where_match = re.search(r'WHERE\s+(\w+)\s*(=|>|<)\s*["\']?(\w+)["\']?', query, re.IGNORECASE)
    if where_match:
        field, op, value = where_match.groups()
        
        # Try to convert value to number if possible
        try:
            value = float(value)
        except ValueError:
            pass
        
        # Filter data
        filtered = []
        for row in data:
            if field in row:
                row_value = row[field]
                if op == '=' and row_value == value:
                    filtered.append(row)
                elif op == '>' and isinstance(row_value, (int, float)) and row_value > value:
                    filtered.append(row)
                elif op == '<' and isinstance(row_value, (int, float)) and row_value < value:
                    filtered.append(row)
        
        data = filtered
    
    # Return as JSON
    return json.dumps(data, indent=2)


# ============================================================================
# CATEGORY 4: API TOOLS (Simulated)
# ============================================================================

# Simulated API responses
API_ENDPOINTS = {
    "users/profile": {
        "user_id": 42,
        "name": "Senthil",
        "role": "AI Engineer",
        "projects": ["RAG System", "Vision Agent", "Multi-Agent Orchestration"]
    },
    "analytics/summary": {
        "total_users": 545,
        "active_users": 412,
        "conversion_rate": 0.756,
        "revenue": 125000
    },
    "status/health": {
        "status": "healthy",
        "uptime": "99.9%",
        "last_check": "2026-01-30T15:30:00Z"
    }
}

def api_call_fn(endpoint: str) -> str:
    """
    Simulated REST API call.
    
    Input format: endpoint path (e.g., "users/profile", "analytics/summary")
    """
    endpoint = endpoint.strip().lower()
    
    if endpoint in API_ENDPOINTS:
        return json.dumps(API_ENDPOINTS[endpoint], indent=2)
    else:
        available = list(API_ENDPOINTS.keys())
        return f"Error: Endpoint '{endpoint}' not found. Available: {available}"


# ============================================================================
# CATEGORY 5: FILE OPERATIONS
# ============================================================================

# In-memory file system (simulated)
FILE_SYSTEM = {}
OUTPUT_DIR = "output_files"

# Ensure directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
def file_write_fn(file_path_and_content: str) -> str:
    """
    Write content to REAL file on disk.
    
    Input format: "filename.txt|content to write"
    Files are saved to ./output_files/ directory
    """
    if '|' not in file_path_and_content:
        return "Error: Format must be 'filename.txt|content'. Missing pipe separator."
    
    parts = file_path_and_content.split('|', 1)
    filename = parts[0].strip()
    content = parts[1].strip() if len(parts) > 1 else ""
    
    if not filename:
        return "Error: Filename cannot be empty"
    
    # Sanitize filename (security)
    filename = os.path.basename(filename)  # Prevent path traversal
    
    # Write to actual file
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        abs_path = os.path.abspath(filepath)
        return f"Successfully wrote {len(content)} characters to {filename} (saved at: {abs_path})"
    
    except Exception as e:
        return f"Error writing file: {str(e)}"


def file_read_fn(filename: str) -> str:
    """
    Read content from REAL file on disk.
    
    Input format: filename (e.g., "report.txt")
    Reads from ./output_files/ directory
    """
    filename = filename.strip()
    
    # Sanitize filename
    filename = os.path.basename(filename)
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(filepath):
        available = os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else []
        return f"Error: File '{filename}' not found. Available: {available}"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return f"Content of {filename}:\n{content}"
    
    except Exception as e:
        return f"Error reading file: {str(e)}"

# ============================================================================
# TOOL REGISTRY
# ============================================================================

AVAILABLE_TOOLS = [
    # Computation
    Tool(
        name="calculate_days_until",
        description="""
        Calculate days from today until a target date.
        
        WHEN: Questions about "days until", "time until", "how long until"
        INPUT: Target date in format YYYY-MM-DD (e.g., "2026-02-21")
        OUTPUT: Number of days
        CATEGORY: computation
        
        Example: "2026-02-21" → "22 days until 2026-02-21"
        """,
        func=calculate_days_until_fn,
        category="computation"
    ),
    Tool(
        name="calculator",
        description="""
        Performs arithmetic calculations.
        WHEN: Math operations (percentages, sums, averages).
        INPUT: Expression WITHOUT quotes (e.g., 88 * 0.25)
        OUTPUT: Numeric result.
        CATEGORY: computation
        """,
        func=calculator_fn,
        category="computation"
    ),
    # Information Retrieval
    Tool(
        name="search_document",
        description="""
        Search pre-loaded documents.
        WHEN: Questions about projects, exams, customers.
        INPUT: Keywords (e.g., "exam date", "customer segments")
        OUTPUT: Matching document excerpts.
        CATEGORY: information
        """,
        func=search_document_fn,
        category="information"
    ),
    
    Tool(
        name="get_datetime",
        description="""
        Get current date/time.
        WHEN: Need today's date or current time.
        INPUT: "date", "time", or "iso"
        OUTPUT: Formatted date/time string.
        CATEGORY: information
        """,
        func=get_datetime_fn,
        category="information"
    ),
    
    Tool(
        name="web_search",
        description="""
        Search web for current info (simulated).
        WHEN: Stock prices, weather, news.
        INPUT: Query (e.g., "nvidia stock", "weather")
        OUTPUT: Search results.
        CATEGORY: information
        """,
        func=web_search_fn,
        category="information"
    ),
    
    # Database
    Tool(
        name="database_query",
        description="""
        Execute SQL query on database (simulated).
        WHEN: Need customer data, orders, analytics from database.
        INPUT: SQL SELECT query (e.g., "SELECT * FROM customers WHERE segment = 'Enterprise'")
        OUTPUT: JSON array of results.
        TABLES: customers, orders
        CATEGORY: database
        """,
        func=database_query_fn,
        category="database"
    ),
    
    # API
    Tool(
        name="api_call",
        description="""
        Call REST API endpoint (simulated).
        WHEN: Need user profiles, analytics, system status.
        INPUT: Endpoint path (e.g., "users/profile", "analytics/summary")
        OUTPUT: JSON response.
        ENDPOINTS: users/profile, analytics/summary, status/health
        CATEGORY: api
        """,
        func=api_call_fn,
        category="api"
    ),
    
    # File Operations
    Tool(
        name="file_write",
        description="""
        Write content to file.
        WHEN: Save reports, export data, create documents.
        INPUT: "filename.txt|content to write" (pipe separator)
        OUTPUT: Confirmation message.
        CATEGORY: file_ops
        IMPORTANT: Use pipe | to separate filename from content.
        """,
        func=file_write_fn,
        category="file_ops"
    ),
    
    Tool(
        name="file_read",
        description="""
        Read content from file.
        WHEN: Load previously saved reports or data.
        INPUT: Filename (e.g., "report.txt")
        OUTPUT: File contents.
        CATEGORY: file_ops
        """,
        func=file_read_fn,
        category="file_ops"
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


def get_tools_by_category(category: str) -> List[Tool]:
    """Get all tools in a category."""
    return [t for t in AVAILABLE_TOOLS if t.category == category]