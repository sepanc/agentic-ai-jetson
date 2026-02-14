"""Test all tools work correctly."""

from tools import AVAILABLE_TOOLS, get_tool_by_name

def test_calculator():
    print("Testing calculator...")
    calc = get_tool_by_name("calculator")
    
    assert calc.execute("2 + 2") == "4.0"
    assert calc.execute("100 / 4") == "25.0"
    assert calc.execute("22 * 0.25") == "5.5"
    
    print("✅ Calculator tests passed")

def test_search_document():
    print("Testing search_document...")
    search = get_tool_by_name("search_document")
    
    result = search.execute("exam date")
    assert "February 21, 2026" in result
    
    result = search.execute("week 1")
    assert "RAG" in result
    
    print("✅ Document search tests passed")

def test_get_datetime():
    print("Testing get_datetime...")
    dt = get_tool_by_name("get_datetime")
    
    result = dt.execute("date")
    assert "2026-01-30" in result  # Today's date
    
    print("✅ DateTime tests passed")

def test_web_search():
    print("Testing web_search...")
    web = get_tool_by_name("web_search")
    
    result = web.execute("nvidia stock")
    assert "NVDA" in result or "NVIDIA" in result
    
    print("✅ Web search tests passed")

if __name__ == "__main__":
    test_calculator()
    test_search_document()
    test_get_datetime()
    test_web_search()
    print("\n🎉 All tool tests passed!")