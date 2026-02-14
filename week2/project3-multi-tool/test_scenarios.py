"""
Complex multi-tool test scenarios.

Each scenario requires chaining 3+ tools.
"""

from react_agent_multi import MultiToolReActAgent
import json


def print_scenario_header(number: int, title: str, description: str):
    """Print formatted scenario header."""
    print("\n" + "="*80)
    print(f"SCENARIO {number}: {title}")
    print("="*80)
    print(f"Description: {description}")
    print("="*80 + "\n")


def scenario_1_database_calc_file():
    """
    Scenario 1: Database → Calculator → File
    
    Query customers, calculate average spend, save to file.
    Expected tools: database_query, calculator, file_write
    """
    print_scenario_header(
        1,
        "Database Analysis with File Export",
        "Find Enterprise customers, calculate average spend, save report"
    )
    
    agent = MultiToolReActAgent(max_steps=10)
    
    query = """Find all Enterprise customers from the database, 
    calculate their average spend, and save a report to 'enterprise_report.txt' 
    with the results."""
    
    summary = agent.run(query)
    
    print("\n📊 Results:")
    print(f"  Tools used: {' → '.join(summary.tools_used)}")
    print(f"  Categories: {', '.join(summary.tool_categories)}")
    print(f"  Success: {'✅' if summary.success else '❌'}")
    print(f"  Answer: {summary.final_answer}")
    
    return summary


def scenario_2_api_search_calc():
    """
    Scenario 2: API → Document Search → Calculator
    
    Get analytics from API, search for context, calculate percentage.
    Expected tools: api_call, search_document, calculator
    """
    print_scenario_header(
        2,
        "API Data Analysis with Context",
        "Get analytics, find customer context, calculate conversion rate"
    )
    
    agent = MultiToolReActAgent(max_steps=10)
    
    query = """Get the analytics summary from the API, 
    then search documents for customer segment information,
    and calculate what percentage of total users are active users."""
    
    summary = agent.run(query)
    
    print("\n📊 Results:")
    print(f"  Tools used: {' → '.join(summary.tools_used)}")
    print(f"  Categories: {', '.join(summary.tool_categories)}")
    print(f"  Success: {'✅' if summary.success else '❌'}")
    print(f"  Answer: {summary.final_answer}")
    
    return summary


def scenario_3_database_filter_save():
    """
    Scenario 3: Database → Database → File
    
    Query orders, filter by amount, save results.
    Expected tools: database_query (2x), file_write
    """
    print_scenario_header(
        3,
        "Database Query with Filtering",
        "Find high-value orders (>$1000) and save to file"
    )
    
    agent = MultiToolReActAgent(max_steps=10)
    
    query = """Query the orders table for all orders with amount greater than 1000,
    then save the results to 'high_value_orders.txt'."""
    
    summary = agent.run(query)
    
    print("\n📊 Results:")
    print(f"  Tools used: {' → '.join(summary.tools_used)}")
    print(f"  Categories: {', '.join(summary.tool_categories)}")
    print(f"  Success: {'✅' if summary.success else '❌'}")
    print(f"  Answer: {summary.final_answer}")
    
    return summary


def scenario_4_datetime_calc_search():
    """
    Scenario 4: DateTime → Calculator → Document Search
    
    Get current date, calculate days until exam, verify exam info.
    Expected tools: get_datetime, calculator, search_document
    """
    print_scenario_header(
        4,
        "Date Calculation with Verification",
        "Calculate days until NCP-AAI exam and verify exam details"
    )
    
    agent = MultiToolReActAgent(max_steps=10)
    
    query = """Find out how many days until February 21, 2026 from today,
    and then search documents to verify when the NCP-AAI exam is scheduled."""
    
    summary = agent.run(query)
    
    print("\n📊 Results:")
    print(f"  Tools used: {' → '.join(summary.tools_used)}")
    print(f"  Categories: {', '.join(summary.tool_categories)}")
    print(f"  Success: {'✅' if summary.success else '❌'}")
    print(f"  Answer: {summary.final_answer}")
    
    return summary


def scenario_5_file_read_calc():
    """
    Scenario 5: File Read → Calculator
    
    Read saved report, perform calculation on the data.
    Expected tools: file_read, calculator
    """
    print_scenario_header(
        5,
        "File Analysis",
        "Read previously saved enterprise report and analyze data"
    )
    
    # First ensure file exists from scenario 1
    print("Note: This scenario requires running Scenario 1 first to create the file.\n")
    
    agent = MultiToolReActAgent(max_steps=10)
    
    query = """Read the file 'enterprise_report.txt' and tell me what's in it."""
    
    summary = agent.run(query)
    
    print("\n📊 Results:")
    print(f"  Tools used: {' → '.join(summary.tools_used)}")
    print(f"  Categories: {', '.join(summary.tool_categories)}")
    print(f"  Success: {'✅' if summary.success else '❌'}")
    print(f"  Answer: {summary.final_answer[:200]}...")
    
    return summary


def run_all_scenarios():
    """Run all test scenarios and summarize results."""
    print("\n" + "🚀"*40)
    print("MULTI-TOOL INTEGRATION TEST SUITE")
    print("🚀"*40)
    
    results = []
    
    # Run all scenarios
    results.append(("Database → Calc → File", scenario_1_database_calc_file()))
    results.append(("API → Search → Calc", scenario_2_api_search_calc()))
    results.append(("Database → Database → File", scenario_3_database_filter_save()))
    results.append(("DateTime → Calc → Search", scenario_4_datetime_calc_search()))
    results.append(("File Read → Analysis", scenario_5_file_read_calc()))
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    
    for name, summary in results:
        status = "✅ PASS" if summary.success else "❌ FAIL"
        tools_count = len(summary.tools_used)
        categories_count = len(summary.tool_categories)
        
        print(f"\n{name:35s} {status}")
        print(f"  Steps: {summary.total_steps:2d} | Tools: {tools_count} | Categories: {categories_count}")
        print(f"  Chain: {' → '.join(summary.tools_used)}")
    
    # Statistics
    total_scenarios = len(results)
    passed = sum(1 for _, s in results if s.success)
    
    print("\n" + "="*80)
    print(f"Overall: {passed}/{total_scenarios} scenarios passed ({passed/total_scenarios*100:.0f}%)")
    print("="*80)
    
    # Tool usage statistics
    all_tools_used = []
    all_categories = []
    for _, summary in results:
        all_tools_used.extend(summary.tools_used)
        all_categories.extend(summary.tool_categories)
    
    print("\n📊 Tool Usage Statistics:")
    from collections import Counter
    tool_counts = Counter(all_tools_used)
    for tool, count in tool_counts.most_common():
        print(f"  {tool:20s}: {count} times")
    
    print("\n📦 Category Coverage:")
    category_counts = Counter(all_categories)
    for category, count in category_counts.most_common():
        print(f"  {category:20s}: {count} times")


if __name__ == "__main__":
    run_all_scenarios()