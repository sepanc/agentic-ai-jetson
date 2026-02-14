"""
ReAct (Reason + Act) Agent - Two Implementations

Version 1: Regex parsing (fragile, educational)
Version 2: Structured output with Pydantic (production-grade)

Pattern: Thought → Action → Observation → Thought → Action → ...
"""

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Literal
import logging
import re
import json

from tools import AVAILABLE_TOOLS, get_tool_by_name, get_tools_description

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================

class ReActResponse(BaseModel):
    """Structured response for a single ReAct step."""
    thought: str = Field(description="Your reasoning about what to do next")
    action: Literal["calculator", "search_document", "get_datetime", "web_search", "finish"] = Field(
        description="Tool to use or 'finish' if done"
    )
    action_input: str = Field(description="Input for the selected tool or final answer")


class ReActStep(BaseModel):
    """Single step in ReAct reasoning loop."""
    thought: str
    action: str
    action_input: str
    observation: str


# ============================================================================
# VERSION 1: REGEX PARSING (ORIGINAL - FRAGILE)
# ============================================================================

class ReActAgent:
    """
    ReAct agent with regex parsing.
    
    Educational implementation showing the fragility of text parsing.
    """
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434", max_steps: int = 5):
        """
        Initialize ReAct agent with regex parsing.
        
        Args:
            ollama_base_url: Ollama server URL
            max_steps: Maximum reasoning steps
        """
        self.llm = ChatOllama(
            model="llama3.2:3b",
            base_url=ollama_base_url,
            temperature=0.1,
            num_predict=512,
        )
        self.max_steps = max_steps
        self.steps: List[ReActStep] = []
    
    def run(self, query: str) -> str:
        """Run ReAct loop to answer query."""
        logger.info(f"🎯 [REGEX VERSION] Query: {query}")
        self.steps = []
        
        for step_num in range(1, self.max_steps + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Step {step_num}/{self.max_steps}")
            logger.info(f"{'='*60}")
            
            # Generate Thought + Action
            thought, action, action_input = self._generate_next_action(query)
            
            logger.info(f"💭 Thought: {thought}")
            logger.info(f"🔧 Action: {action}")
            logger.info(f"📥 Input: {action_input}")
            
            # Check if agent wants to finish
            if action.lower() == "finish":
                logger.info(f"✅ Final Answer: {action_input}")
                return action_input
            
            # Execute action and get observation
            observation = self._execute_action(action, action_input)
            logger.info(f"👁️  Observation: {observation}")
            
            # Store step
            self.steps.append(ReActStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation
            ))
        
        logger.warning(f"⚠️  Reached max steps ({self.max_steps})")
        return f"Unable to complete task within {self.max_steps} steps."
    
    def _generate_next_action(self, query: str) -> Tuple[str, str, str]:
        """Generate next action using regex parsing."""
        prompt = self._build_react_prompt(query)
        
        response = self.llm.invoke(prompt)
        raw_text = response.content
        
        logger.info(f"🤖 LLM Response:\n{raw_text[:300]}...")
        
        # Parse with regex (FRAGILE!)
        thought, action, action_input = self._parse_react_response(raw_text)
        
        return thought, action, action_input
    
    def _build_react_prompt(self, query: str) -> str:
        """Build ReAct prompt."""
        tools_desc = get_tools_description()
        
        history = ""
        for i, step in enumerate(self.steps, 1):
            history += f"""
Step {i}:
Thought: {step.thought}
Action: {step.action}
Action Input: {step.action_input}
Observation: {step.observation}
"""
        
        prompt = f"""You are a reasoning agent that solves problems step-by-step using available tools.

AVAILABLE TOOLS:
{tools_desc}

INSTRUCTIONS:
Answer the user's question by reasoning through the problem step-by-step.
For each step, provide:
1. Thought: Your reasoning about what to do next
2. Action: The tool to use (calculator, search_document, get_datetime, web_search, or finish)
3. Action Input: The input for that tool

When you have the final answer, use:
Action: finish
Action Input: [your final answer]

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
Thought: [your reasoning]
Action: [tool name or "finish"]
Action Input: [input for the tool or final answer]

CONVERSATION HISTORY:
{history if history else "(no previous steps)"}

USER QUESTION: {query}

Now provide your next Thought, Action, and Action Input:"""

        return prompt
    
    def _parse_react_response(self, response: str) -> Tuple[str, str, str]:
        """
        Parse LLM response using regex (FRAGILE!).
        
        Returns:
            (thought, action, action_input)
        """
        # Extract Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|$)', response, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else "No thought provided"
        
        # Extract Action
        action_match = re.search(r'Action:\s*(.+?)(?=\nAction Input:|$)', response, re.DOTALL | re.IGNORECASE)
        action = action_match.group(1).strip() if action_match else "finish"
        
        # Extract Action Input
        action_input_match = re.search(r'Action Input:\s*(.+?)(?=\n\n|$)', response, re.DOTALL | re.IGNORECASE)
        action_input = action_input_match.group(1).strip() if action_input_match else ""
        
        return thought, action, action_input
    
    def _execute_action(self, action: str, action_input: str) -> str:
        """Execute tool action."""
        if action.lower() == "finish":
            return "Task completed"
        
        try:
            tool = get_tool_by_name(action.strip())
            observation = tool.execute(action_input)
            return observation
        except ValueError:
            available_tools = [t.name for t in AVAILABLE_TOOLS]
            return f"Error: Tool '{action}' not found. Available: {', '.join(available_tools)}"
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    def get_steps_summary(self) -> str:
        """Get formatted summary of reasoning steps."""
        summary = []
        for i, step in enumerate(self.steps, 1):
            summary.append(f"""
Step {i}:
  Thought: {step.thought}
  Action: {step.action}({step.action_input})
  Observation: {step.observation}
""")
        return "\n".join(summary)


# ============================================================================
# VERSION 2: STRUCTURED OUTPUT (PRODUCTION-GRADE)
# ============================================================================

class ReActAgentStructured:
    """
    ReAct agent with Pydantic structured output.
    
    Production-grade implementation using schema validation.
    """
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434", max_steps: int = 5):
        """
        Initialize ReAct agent with structured output.
        
        Args:
            ollama_base_url: Ollama server URL
            max_steps: Maximum reasoning steps
        """
        base_llm = ChatOllama(
            model="llama3.2:3b",
            base_url=ollama_base_url,
            temperature=0.1,
            num_predict=512,
            format="json",  # Force JSON output
        )
        
        # Bind Pydantic schema to LLM
        self.structured_llm = base_llm.with_structured_output(ReActResponse)
        self.max_steps = max_steps
        self.steps: List[ReActStep] = []
    
    def run(self, query: str) -> str:
        """Run ReAct loop to answer query."""
        logger.info(f"🎯 [STRUCTURED VERSION] Query: {query}")
        self.steps = []
        
        for step_num in range(1, self.max_steps + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Step {step_num}/{self.max_steps}")
            logger.info(f"{'='*60}")
            
            # Generate Thought + Action (returns structured object)
            try:
                thought, action, action_input = self._generate_next_action(query)
            except Exception as e:
                logger.error(f"Failed to generate action: {e}")
                return f"Error: Unable to generate valid action - {str(e)}"
            
            logger.info(f"💭 Thought: {thought}")
            logger.info(f"🔧 Action: {action}")
            logger.info(f"📥 Input: {action_input}")
            
            # Check if agent wants to finish
            if action.lower() == "finish":
                logger.info(f"✅ Final Answer: {action_input}")
                return action_input
            
            # Execute action and get observation
            observation = self._execute_action(action, action_input)
            logger.info(f"👁️  Observation: {observation}")
            
            # Store step
            self.steps.append(ReActStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation
            ))
        
        logger.warning(f"⚠️  Reached max steps ({self.max_steps})")
        return f"Unable to complete task within {self.max_steps} steps."
    
    def _generate_next_action(self, query: str) -> Tuple[str, str, str]:
        """
        Generate next action using structured output.
        
        Returns ReActResponse object with validated fields.
        """
        prompt = self._build_react_prompt(query)
        
        # LLM returns ReActResponse object directly (no parsing!)
        response: ReActResponse = self.structured_llm.invoke(prompt)
        
        logger.info(f"✅ Structured response validated")
        
        return response.thought, response.action, response.action_input
    
    def _build_react_prompt(self, query: str) -> str:
        """Build ReAct prompt for structured output."""
        tools_desc = get_tools_description()
        
        history = ""
        for i, step in enumerate(self.steps, 1):
            history += f"""
Step {i}:
Thought: {step.thought}
Action: {step.action}
Action Input: {step.action_input}
Observation: {step.observation}
"""
        
        prompt = f"""You are a reasoning agent that solves problems step-by-step using available tools.

AVAILABLE TOOLS:
{tools_desc}

INSTRUCTIONS:
Return a JSON object with your next step containing:
- thought: Your reasoning about what to do next
- action: Tool name (calculator, search_document, get_datetime, web_search) or "finish"
- action_input: Input for the tool or final answer

CONVERSATION HISTORY:
{history if history else "(no previous steps)"}

USER QUESTION: {query}

Return your reasoning as a JSON object with thought, action, and action_input fields."""

        return prompt
    
    def _execute_action(self, action: str, action_input: str) -> str:
        """Execute tool action."""
        if action.lower() == "finish":
            return "Task completed"
        
        try:
            tool = get_tool_by_name(action.strip())
            observation = tool.execute(action_input)
            return observation
        except ValueError:
            available_tools = [t.name for t in AVAILABLE_TOOLS]
            return f"Error: Tool '{action}' not found. Available: {', '.join(available_tools)}"
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    def get_steps_summary(self) -> str:
        """Get formatted summary of reasoning steps."""
        summary = []
        for i, step in enumerate(self.steps, 1):
            summary.append(f"""
Step {i}:
  Thought: {step.thought}
  Action: {step.action}({step.action_input})
  Observation: {step.observation}
""")
        return "\n".join(summary)


# ============================================================================
# COMPARISON TESTS
# ============================================================================

def run_comparison_test(query: str, test_name: str):
    """Run same query on both versions and compare."""
    print("\n" + "="*80)
    print(f"TEST: {test_name}")
    print("="*80)
    print(f"Query: {query}\n")
    
    # Test Version 1: Regex parsing
    print("--- VERSION 1: Regex Parsing ---")
    agent_v1 = ReActAgent(ollama_base_url="http://localhost:11434", max_steps=5)
    try:
        answer_v1 = agent_v1.run(query)
        success_v1 = True
        print(f"✅ Answer: {answer_v1}")
    except Exception as e:
        success_v1 = False
        answer_v1 = None
        print(f"❌ Failed: {e}")
    
    # Test Version 2: Structured output
    print("\n--- VERSION 2: Structured Output ---")
    agent_v2 = ReActAgentStructured(ollama_base_url="http://localhost:11434", max_steps=5)
    try:
        answer_v2 = agent_v2.run(query)
        success_v2 = True
        print(f"✅ Answer: {answer_v2}")
    except Exception as e:
        success_v2 = False
        answer_v2 = None
        print(f"❌ Failed: {e}")
    
    # Comparison
    print("\n--- COMPARISON ---")
    print(f"V1 (Regex) Success: {success_v1}")
    print(f"V2 (Structured) Success: {success_v2}")
    print(f"V1 Steps: {len(agent_v1.steps)}")
    print(f"V2 Steps: {len(agent_v2.steps)}")
    
    return success_v1, success_v2


if __name__ == "__main__":
    print("\n" + "🚀"*40)
    print("ReAct Agent Comparison: Regex vs Structured Output")
    print("🚀"*40)
    
    # Track success rates
    results = []
    
    # Test 1: Simple calculation
    s1, s2 = run_comparison_test(
        "What is 25% of 88?",
        "Simple Math Question"
    )
    results.append(("Simple Math", s1, s2))
    
    # Test 2: Multi-step reasoning
    s1, s2 = run_comparison_test(
        "What is 25% of the number of days until February 21, 2026?",
        "Multi-Step Reasoning (Date + Calculation)"
    )
    results.append(("Multi-Step", s1, s2))
    
    # Test 3: Information retrieval
    s1, s2 = run_comparison_test(
        "When is the NCP-AAI exam?",
        "Document Search"
    )
    results.append(("Doc Search", s1, s2))
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    v1_successes = sum(1 for _, s1, _ in results if s1)
    v2_successes = sum(1 for _, _, s2 in results if s2)
    total_tests = len(results)
    
    print(f"\nRegex Version:      {v1_successes}/{total_tests} tests passed ({v1_successes/total_tests*100:.0f}%)")
    print(f"Structured Version: {v2_successes}/{total_tests} tests passed ({v2_successes/total_tests*100:.0f}%)")
    
    print("\n📊 Test Details:")
    for test_name, s1, s2 in results:
        v1_icon = "✅" if s1 else "❌"
        v2_icon = "✅" if s2 else "❌"
        print(f"  {test_name:20s} | Regex: {v1_icon} | Structured: {v2_icon}")
    
    print("\n💡 Key Takeaway:")
    if v2_successes > v1_successes:
        print("Structured output provides more reliable results!")
    elif v2_successes == v1_successes:
        print("Both approaches worked, but structured output is type-safe and production-ready.")
    
    print("\n" + "="*80)