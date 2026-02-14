"""Test multi-agent system with various scenarios."""

from multi_agent_graph import run_multi_agent_workflow


def test_scenario(query: str, scenario_name: str):
    """Run single test scenario."""
    print("\n" + "="*80)
    print(f"TEST: {scenario_name}")
    print("="*80)
    print(f"Query: {query}\n")
    
    # Run workflow
    final_state = run_multi_agent_workflow(query, model_name="qwen2.5:7b")
    
    # Print results
    print("\n" + "-"*80)
    print("WORKFLOW SUMMARY")
    print("-"*80)
    print(f"Steps: {final_state['step_count']}")
    
    # FIXED: Access dict keys instead of Pydantic attributes
    agents_used = set(msg['agent'] for msg in final_state['messages'])
    print(f"Agents used: {', '.join(agents_used)}")
    
    print(f"Complete: {'✅ Yes' if final_state['workflow_complete'] else '❌ No'}")
    
    # Print agent activity log
    print("\n" + "-"*80)
    print("AGENT ACTIVITY LOG")
    print("-"*80)
    for i, msg in enumerate(final_state['messages'], 1):
        print(f"{i}. [{msg['agent'].upper()}] {msg['content']}")
        if msg.get('metadata'):
            print(f"   Metadata: {msg['metadata']}")
    
    print("\n" + "-"*80)
    print("FINAL REPORT")
    print("-"*80)
    print(final_state['final_report'])
    print("-"*80 + "\n")


if __name__ == "__main__":
    print("\n" + "🤖"*40)
    print("MULTI-AGENT SYSTEM TESTS")
    print("🤖"*40)
    
    # Test 1: Document research
    test_scenario(
        "Find information about the NCP-AAI exam and summarize the key details",
        "Document Research & Summary"
    )
    
    # Test 2: Database analysis
    test_scenario(
        "Get Enterprise customers from database and calculate their average spend",
        "Database Analysis"
    )
    
    # Test 3: API data processing
    test_scenario(
        "Get analytics from API and calculate the conversion rate percentage",
        "API Data Processing"
    )
    
    print("\n✅ All multi-agent tests complete!")