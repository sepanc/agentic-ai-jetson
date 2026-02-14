#!/usr/bin/env python3
"""
CLI runner for multi-source research agent
"""

import argparse
from pathlib import Path
from src.agents.research_agent import ResearchAgent


def main():
    parser = argparse.ArgumentParser(
        description="Multi-source research agent for AI/edge deployment queries"
    )
    parser.add_argument(
        "query",
        type=str,
        help="Research question to investigate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="research_report.md",
        help="Output file for report (default: research_report.md)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:3b",
        help="Ollama model to use (default: llama3.2:3b)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Initialize agent
    print(f"Initializing research agent with model: {args.model}")
    agent = ResearchAgent(model=args.model)
    
    # Execute research
    print(f"\nResearching: {args.query}\n")
    report = agent.research(args.query)
    
    # Display results
    print("\n" + "="*80)
    print(report.to_markdown())
    print("="*80)
    
    # Save to file
    output_path = Path(args.output)
    output_path.write_text(report.to_markdown())
    print(f"\n✓ Report saved to: {output_path}")
    
    # Summary stats
    print(f"\n📊 Research Summary:")
    print(f"   Category: {report.category}")
    print(f"   Confidence: {report.confidence:.1%}")
    print(f"   Sources queried: {', '.join(report.sources_queried)}")
    print(f"   Total sources: {len(report.sources)}")
    print(f"   Key findings: {len(report.key_findings)}")
    print(f"   Recommendations: {len(report.recommendations)}")


if __name__ == "__main__":
    main()