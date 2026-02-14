#!/bin/bash

# Run research agent in Docker on Jetson
# Usage: ./run-research-docker.sh "Your research query"

QUERY="$1"

if [ -z "$QUERY" ]; then
  echo "Usage: ./run-research-docker.sh 'Your research query'"
  exit 1
fi

# Load environment variables (OLLAMA_BASE_URL, SERPER_API_KEY, etc.)
if [ -f .env ]; then source .env; fi

# Run Docker container (set OLLAMA_BASE_URL in .env for remote Ollama)
docker run --rm \
  --runtime nvidia \
  --network host \
  -e OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://localhost:11434} \
  -e OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.2:3b} \
  -e SERPER_API_KEY=${SERPER_API_KEY} \
  -v $(pwd)/data/chroma_db:/app/data/chroma_db \
  -v $(pwd)/data/knowledge_base:/app/data/knowledge_base \
  -v $(pwd)/data/deployment_metrics:/app/data/deployment_metrics \
  -v $(pwd)/output:/app/output \
  research-agent:jetson \
  "$QUERY" \
  --output /app/output/research_report.md

echo ""
echo "✅ Report saved to: output/research_report.md"