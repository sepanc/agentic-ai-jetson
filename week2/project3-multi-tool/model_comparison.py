"""
Model Comparison Test Suite (FIXED VERSION)

Improvements:
- Proper success detection (validates answer quality)
- Detects empty finish answers
- Measures answer completeness
- Better metrics and reporting
"""

from react_agent_multi import MultiToolReActAgent
from typing import List, Dict
import time
import json
import re


class ModelComparison:
    """Compare multiple models on same test scenarios."""
    
    def __init__(self, models: List[str]):
        """
        Initialize comparison.
        
        Args:
            models: List of model names (e.g., ["llama3.2:3b", "llama3.1:8b"])
        """
        self.models = models
        self.results = {}
    
    def run_scenario(self, model_name: str, query: str, scenario_name: str, expected_pattern: str = None) -> Dict:
        """
        Run single scenario on specific model.
        
        Args:
            model_name: Name of model to test
            query: Query to run
            scenario_name: Name of scenario
            expected_pattern: Regex pattern expected in answer (optional)
        """
        print(f"\n🔬 Testing {model_name} on: {scenario_name}")
        
        # Create agent with specific model
        agent = MultiToolReActAgent(
            ollama_base_url="http://localhost:11434",
            max_steps=10
        )
        
        # Override model (patch the LLM)
        from langchain_ollama import ChatOllama
        from react_agent_multi import ReActResponse
        
        base_llm = ChatOllama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.1,
            num_predict=512,
            format="json",
        )
        agent.structured_llm = base_llm.with_structured_output(ReActResponse)
        
        # Run and time it
        start_time = time.time()
        summary = agent.run(query)
        elapsed_time = time.time() - start_time
        
        # Analyze results with IMPROVED success detection
        has_loop = self._detect_loop(summary.tools_used)
        has_real_answer = self._validate_answer_quality(summary.final_answer, query, expected_pattern)
        answer_complete = self._check_answer_completeness(summary.final_answer)
        
        result = {
            "model": model_name,
            "scenario": scenario_name,
            "success": has_real_answer,  # FIXED: Proper validation
            "steps": summary.total_steps,
            "tools_used": summary.tools_used,
            "categories": summary.tool_categories,
            "has_loop": has_loop,
            "time_seconds": round(elapsed_time, 2),
            "final_answer": summary.final_answer[:150],
            "answer_length": len(summary.final_answer),
            "answer_complete": answer_complete,
            "pattern_match": expected_pattern and bool(re.search(expected_pattern, summary.final_answer, re.IGNORECASE)) if expected_pattern else None
        }
        
        # Better status reporting
        status = "✅ PASS" if has_real_answer else "❌ FAIL"
        if not has_real_answer and answer_complete:
            status = "⚠️  PARTIAL"  # Has answer but wrong/incomplete
        
        print(f"  {status} | Steps: {summary.total_steps} | Time: {elapsed_time:.1f}s | Loop: {'Yes' if has_loop else 'No'} | Len: {len(summary.final_answer)}")
        if not has_real_answer:
            print(f"  ⮕ Answer: {summary.final_answer[:100]}")
        
        return result
    
    def _validate_answer_quality(self, answer: str, query: str, expected_pattern: str = None) -> bool:
        """
        Validate that answer is real and meaningful.
        
        Returns:
            True if answer is valid, False otherwise
        """
        if not answer or not answer.strip():
            return False  # Empty answer
        
        # Reject error/failure messages (generic non-answers from the agent)
        fail_indicators = [
            "Error",
            "Unable to complete",
            "all tool calls failed",
            "Unable to answer",
            "Task completed",  # Generic non-answer
        ]
        
        for indicator in fail_indicators:
            if indicator in answer:
                return False
        
        # Must have minimum content
        if len(answer.strip()) < 3:
            return False  # Too short to be meaningful
        
        # If expected pattern provided, check it
        if expected_pattern:
            if not re.search(expected_pattern, answer, re.IGNORECASE):
                return False
        
        return True
    
    def _check_answer_completeness(self, answer: str) -> bool:
        """
        Check if answer appears complete (not just partial info).
        
        Returns:
            True if answer seems complete
        """
        if not answer or len(answer.strip()) < 5:
            return False
        
        # Has some substance
        return len(answer.strip()) >= 10
    
    def _detect_loop(self, tools_used: List[str]) -> bool:
        """Detect if same tool used 3+ times in a row."""
        if len(tools_used) < 3:
            return False
        
        for i in range(len(tools_used) - 2):
            if tools_used[i] == tools_used[i+1] == tools_used[i+2]:
                return True
        
        return False
    
    def compare_models(self, scenarios: List[Dict[str, str]]):
        """
        Compare all models on all scenarios.
        
        Args:
            scenarios: List of {"name": str, "query": str, "expected_pattern": str (optional)}
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON TEST SUITE (FIXED VERSION)")
        print("="*80)
        print(f"Models: {', '.join(self.models)}")
        print(f"Scenarios: {len(scenarios)}")
        print("="*80)
        
        all_results = []
        
        for model in self.models:
            print(f"\n{'='*80}")
            print(f"🤖 Testing Model: {model}")
            print(f"{'='*80}")
            
            for scenario in scenarios:
                result = self.run_scenario(
                    model_name=model,
                    query=scenario["query"],
                    scenario_name=scenario["name"],
                    expected_pattern=scenario.get("expected_pattern")
                )
                all_results.append(result)
        
        # Analyze and compare
        self._print_comparison_summary(all_results, scenarios)
        
        return all_results
    
    def _print_comparison_summary(self, results: List[Dict], scenarios: List[Dict]):
        """Print comprehensive comparison with fixed metrics."""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY (ACCURATE METRICS)")
        print("="*80)
        
        # Group by model
        by_model = {}
        for result in results:
            model = result["model"]
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(result)
        
        # Per-model statistics
        print("\n📊 Per-Model Performance:")
        print("-" * 80)
        
        for model in self.models:
            model_results = by_model.get(model, [])
            
            # Calculate accurate metrics
            true_success_count = sum(1 for r in model_results if r["success"])
            success_rate = (true_success_count / len(model_results) * 100) if model_results else 0
            
            avg_steps = sum(r["steps"] for r in model_results) / len(model_results) if model_results else 0
            loop_count = sum(1 for r in model_results if r["has_loop"])
            avg_time = sum(r["time_seconds"] for r in model_results) / len(model_results) if model_results else 0
            avg_answer_len = sum(r["answer_length"] for r in model_results) / len(model_results) if model_results else 0
            
            # New metrics
            empty_answers = sum(1 for r in model_results if r["answer_length"] == 0)
            partial_answers = sum(1 for r in model_results if r["answer_complete"] and not r["success"])
            
            print(f"\n{model}:")
            print(f"  ✅ True Success:   {success_rate:.0f}% ({true_success_count}/{len(model_results)})")
            print(f"  📏 Avg Steps:      {avg_steps:.1f}")
            print(f"  🔄 Loop Count:     {loop_count}/{len(model_results)} scenarios")
            print(f"  ⏱️  Avg Time:       {avg_time:.1f}s per scenario")
            print(f"  📝 Avg Answer Len: {avg_answer_len:.0f} chars")
            print(f"  ⚠️  Empty Answers:  {empty_answers}/{len(model_results)}")
            if partial_answers > 0:
                print(f"  ⚠️  Partial/Wrong:  {partial_answers}/{len(model_results)}")
        
        # Head-to-head comparison
        print("\n" + "="*80)
        print("HEAD-TO-HEAD SCENARIO COMPARISON")
        print("="*80)
        
        for scenario in scenarios:
            print(f"\n📋 {scenario['name']}:")
            print("-" * 70)
            
            for model in self.models:
                model_result = next((r for r in results if r["model"] == model and r["scenario"] == scenario["name"]), None)
                
                if model_result:
                    status = "✅" if model_result["success"] else "❌"
                    loop_indicator = "🔄" if model_result["has_loop"] else "  "
                    empty_indicator = "📭" if model_result["answer_length"] == 0 else "  "
                    
                    print(f"  {status} {loop_indicator} {empty_indicator} {model:20s} | Steps: {model_result['steps']:2d} | Time: {model_result['time_seconds']:5.1f}s | Tools: {len(model_result['tools_used'])} | Ans: {model_result['answer_length']:3d} chars")
        
        # Winner analysis
        print("\n" + "="*80)
        print("🏆 WINNER ANALYSIS")
        print("="*80)
        
        if by_model:
            # Overall winner (by true success rate)
            best_model = max(by_model.keys(), key=lambda m: sum(1 for r in by_model[m] if r["success"]))
            best_success_rate = sum(1 for r in by_model[best_model] if r["success"]) / len(by_model[best_model]) * 100
            
            print(f"\n🥇 Best Overall (True Success): {best_model} ({best_success_rate:.0f}% success rate)")
            
            # Fewest loops
            fewest_loops_model = min(by_model.keys(), key=lambda m: sum(1 for r in by_model[m] if r["has_loop"]))
            loop_count = sum(1 for r in by_model[fewest_loops_model] if r["has_loop"])
            
            print(f"🎯 Fewest Loops: {fewest_loops_model} ({loop_count}/{len(by_model[fewest_loops_model])} scenarios)")
            
            # Most efficient (fewest steps)
            most_efficient = min(by_model.keys(), key=lambda m: sum(r["steps"] for r in by_model[m]) / len(by_model[m]))
            avg_steps = sum(r["steps"] for r in by_model[most_efficient]) / len(by_model[most_efficient])
            
            print(f"⚡ Most Efficient: {most_efficient} ({avg_steps:.1f} avg steps)")
            
            # Best answer quality (longest average answer)
            best_answers = max(by_model.keys(), key=lambda m: sum(r["answer_length"] for r in by_model[m]) / len(by_model[m]))
            avg_len = sum(r["answer_length"] for r in by_model[best_answers]) / len(by_model[best_answers])
            
            print(f"📝 Best Answer Quality: {best_answers} ({avg_len:.0f} avg chars)")
        
        # Problem identification
        print("\n" + "="*80)
        print("🔍 PROBLEM IDENTIFICATION")
        print("="*80)
        
        for model in self.models:
            model_results = by_model.get(model, [])
            empty_count = sum(1 for r in model_results if r["answer_length"] == 0)
            loop_count = sum(1 for r in model_results if r["has_loop"])
            
            print(f"\n{model}:")
            if empty_count > 0:
                print(f"  ⚠️  Empty answer problem: {empty_count}/{len(model_results)} cases")
                print(f"     → Needs 'finish with answer' prompt fix")
            
            if loop_count > len(model_results) / 2:
                print(f"  ⚠️  High loop rate: {loop_count}/{len(model_results)} cases")
                print(f"     → Needs better stopping logic")
            
            if loop_count == 0:
                print(f"  ✅ No loops detected - good stopping behavior")


# ============================================================================
# TEST SCENARIOS WITH EXPECTED PATTERNS
# ============================================================================

COMPARISON_SCENARIOS = [
    {
        "name": "Simple Math",
        "query": "What is 25% of 88?",
        "expected_pattern": r"22(\.0)?"  # Should contain "22" or "22.0"
    },
    {
        "name": "Database → File",
        "query": "Query all Enterprise customers from the database and save results to 'test_customers.txt'",
        "expected_pattern": r"(Successfully|wrote|saved)"  # Should mention success
    },
    {
        "name": "Multi-Step Planning",
        "query": "Get current date, calculate days until February 21, 2026, then search documents for exam info",
        "expected_pattern": r"(February|exam|21|2026)"  # Should mention exam date
    },
    {
        "name": "API → Calculator",
        "query": "Get analytics summary from API and calculate the percentage of active users",
        "expected_pattern": r"(75|0\.75|percent)"  # Should mention ~75%
    },
]


if __name__ == "__main__":
    # Models to compare
    MODELS_TO_TEST = [
        "llama3.2:3b",      # Current baseline
        "llama3.1:8b",      # Upgrade option 1
        "qwen2.5:7b",       # Upgrade option 2 (best for tools)
    ]
    
    print("\n🚀 Model Comparison Starting (FIXED VERSION)...")
    print(f"Testing {len(MODELS_TO_TEST)} models on {len(COMPARISON_SCENARIOS)} scenarios")
    print(f"Total tests: {len(MODELS_TO_TEST) * len(COMPARISON_SCENARIOS)}")
    
    # Check which models are available
    import subprocess
    try:
        available_models = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        ).stdout
        
        missing_models = []
        for model in MODELS_TO_TEST:
            if model not in available_models:
                missing_models.append(model)
        
        if missing_models:
            print("\n⚠️  Missing models (run these commands first):")
            for model in missing_models:
                print(f"  ollama pull {model}")
            print("\nContinuing with available models only...\n")
            MODELS_TO_TEST = [m for m in MODELS_TO_TEST if m not in missing_models]
    except:
        print("⚠️  Could not check available models, proceeding with all...")
    
    if not MODELS_TO_TEST:
        print("❌ No models available! Pull at least one model first.")
        exit(1)
    
    # Run comparison
    comparison = ModelComparison(MODELS_TO_TEST)
    results = comparison.compare_models(COMPARISON_SCENARIOS)
    
    # Save results to JSON
    with open("model_comparison_results_fixed.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to: model_comparison_results_fixed.json")