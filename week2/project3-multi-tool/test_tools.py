"""Test all extended tools."""

from tools_extended import AVAILABLE_TOOLS, get_tool_by_name, FILE_SYSTEM
import json

def test_calculator():
    print("\n📊 Testing Calculator...")
    calc = get_tool_by_name("calculator")
    
    result = calc.execute("100 * 0.25")
    assert "25.0" in result
    print(f"  ✅ 100 * 0.25 = {result}")

def test_database():
    print("\n💾 Testing Database Query...")
    db = get_tool_by_name("database_query")
    
    # Test 1: All customers
    result = db.execute("SELECT * FROM customers")
    data = json.loads(result)
    assert len(data) == 5
    print(f"  ✅ Found {len(data)} customers")
    
    # Test 2: Filtered query
    result = db.execute("SELECT * FROM customers WHERE segment = 'Enterprise'")
    data = json.loads(result)
    assert len(data) == 2
    print(f"  ✅ Found {len(data)} Enterprise customers")

def test_api():
    print("\n🌐 Testing API Call...")
    api = get_tool_by_name("api_call")
    
    result = api.execute("users/profile")
    data = json.loads(result)
    assert "user_id" in data
    print(f"  ✅ API returned: {data['name']}")

def test_file_operations():
    print("\n📁 Testing File Operations...")
    
    # Write file
    writer = get_tool_by_name("file_write")
    result = writer.execute("test.txt|Hello from multi-tool agent!")
    assert "Success" in result
    print(f"  ✅ Write: {result}")
    
    # Read file
    reader = get_tool_by_name("file_read")
    result = reader.execute("test.txt")
    assert "Hello from multi-tool agent!" in result
    print(f"  ✅ Read: File contains expected content")

def test_search_and_datetime():
    print("\n🔍 Testing Search & DateTime...")
    
    search = get_tool_by_name("search_document")
    result = search.execute("exam")
    assert "February 21, 2026" in result
    print(f"  ✅ Document search found exam info")
    
    dt = get_tool_by_name("get_datetime")
    result = dt.execute("date")
    assert "2026-01-30" in result
    print(f"  ✅ DateTime: {result}")

if __name__ == "__main__":
    print("="*60)
    print("Testing Extended Tool Set (8 tools)")
    print("="*60)
    
    test_calculator()
    test_database()
    test_api()
    test_file_operations()
    test_search_and_datetime()
    
    print("\n" + "="*60)
    print("🎉 All tool tests passed!")
    print("="*60)
    print(f"\nTotal tools available: {len(AVAILABLE_TOOLS)}")
    print("Categories: computation, information, database, api, file_ops")