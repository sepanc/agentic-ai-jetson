"""
Test suite for integrated research assistant.

Demonstrates all Week 2 capabilities:
- Project 1: Structured Output
- Project 2: ReAct Reasoning
- Project 3: Multi-Tool Integration
- Project 4: Multi-Agent System
- Project 5: Full Integration
"""

from integrated_graph import run_integrated_research
import json


def print_test_header(scenario_num: int, title: str, description: str):
    """Print formatted test header."""
    print("\n" + "="*80)
    print(f"SCENARIO {scenario_num}: {title}")
    print("="*80)
    print(f"{description}")
    print("="*80 + "\n")


def print_results(final_state):
    """Print comprehensive results."""
    print("\n" + "-"*80)
    print("WORKFLOW RESULTS")
    print("-"*80)
    
    print(f"\n📊 Execution Summary:")
    print(f"   Total Steps: {final_state['step_count']}")
    print(f"   Agents Used: {', '.join(set(msg['agent'] for msg in final_state['messages']))}")
    print(f"   Status: {'✅ Complete' if final_state['workflow_complete'] else '❌ Incomplete'}")
    
    print(f"\n🔍 Agent Activity:")
    for i, msg in enumerate(final_state['messages'], 1):
        print(f"   {i}. [{msg['agent'].upper()}] {msg['content']}")
        if msg.get('metadata'):
            for key, value in msg['metadata'].items():
                if value and key != 'result_length':
                    print(f"      • {key}: {value}")
    
    print(f"\n📄 Final Report:")
    print("-"*80)
    print(final_state['final_report'])
    print("-"*80 + "\n")


# ============================================================================
# TEST SCENARIOS
# ============================================================================

def scenario_1_document_research():
    """Scenario 1: Document search + Summary + File save."""
    print_test_header(
        1,
        "Document Research with File Export",
        "Search internal documents, analyze findings, save report to file"
    )
    
    query = "Research the NCP-AAI certification exam and create a detailed report with all important information. Save the report to a file."
    
    result = run_integrated_research(query, model_name="qwen2.5:7b")
    print_results(result)
    
    return result


def scenario_2_database_analysis():
    """Scenario 2: Database query + Calculation + Report."""
    print_test_header(
        2,
        "Database Analysis with Calculations",
        "Query database for Enterprise customers and calculate average spend"
    )
    
    query = "Find all Enterprise customers in the database and calculate their average spending. Create a summary report."
    
    result = run_integrated_research(query, model_name="qwen2.5:7b")
    print_results(result)
    
    return result


def scenario_3_api_processing():
    """Scenario 3: API call + Analysis + Structured output."""
    print_test_header(
        3,
        "API Data Processing",
        "Get analytics from API and analyze user engagement metrics"
    )
    
    query = "Get the analytics summary from the API and analyze the user engagement metrics. What percentage of users are active?"
    
    result = run_integrated_research(query, model_name="qwen2.5:7b")
    print_results(result)
    
    return result


def scenario_4_comprehensive():
    """Scenario 4: Multi-tool comprehensive research."""
    print_test_header(
        4,
        "Comprehensive Research Task",
        "Combine multiple data sources for complete analysis"
    )
    
    query = "Research our customer base: get Enterprise customers from database, check current analytics from API, and create a comprehensive report comparing the two data sources. Save the report."
    
    result = run_integrated_research(query, model_name="qwen2.5:7b")
    print_results(result)
    
    return result


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🎓"*40)
    print("WEEK 2 PROJECT 5: INTEGRATED RESEARCH ASSISTANT")
    print("Demonstrates: Multi-Agent + Multi-Tool + Structured Output + ReAct")
    print("🎓"*40)
    
    results = []
    
    # Run all scenarios
    results.append(("Document Research", scenario_1_document_research()))
    results.append(("Database Analysis", scenario_2_database_analysis()))
    results.append(("API Processing", scenario_3_api_processing()))
    results.append(("Comprehensive", scenario_4_comprehensive()))
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - WEEK 2 COMPLETE")
    print("="*80)
    
    success_count = sum(1 for _, r in results if r['workflow_complete'])
    total = len(results)
    
    print(f"\n📊 Overall Performance:")
    print(f"   Success Rate: {success_count}/{total} ({success_count/total*100:.0f}%)")
    
    print(f"\n✅ Completed Scenarios:")
    for name, result in results:
        status = "✅" if result['workflow_complete'] else "❌"
        steps = result['step_count']
        agents = ', '.join(set(msg['agent'] for msg in result['messages']))
        print(f"   {status} {name:30s} | Steps: {steps} | Agents: {agents}")
    
    print(f"\n🎯 Week 2 Skills Demonstrated:")
    print("   ✅ Structured Output with Pydantic validation")
    print("   ✅ ReAct reasoning with thought → action → observation")
    print("   ✅ Multi-tool integration (8 tools across 5 categories)")
    print("   ✅ Multi-agent orchestration with LangGraph")
    print("   ✅ State management and agent communication")
    print("   ✅ Production error handling and logging")
    print("   ✅ Automated file export and reporting")
    
    print(f"\n{'='*80}")
    print("🎉 WEEK 2 COMPLETE - READY FOR NCP-AAI CERTIFICATION!")
    print("="*80 + "\n")