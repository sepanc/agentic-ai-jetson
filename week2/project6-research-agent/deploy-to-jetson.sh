#!/bin/bash

# Deployment script for Jetson Orin Nano
set -e

JETSON_IP="192.168.40.100"
JETSON_USER="senthil"
PROJECT_NAME="research-agent"
JETSON_PATH="/ssd/projects/$PROJECT_NAME"

echo "========================================"
echo "Deploying Research Agent to Jetson"
echo "========================================"

# Step 1: Create project directory on Jetson
echo "📁 Creating project directory on Jetson..."
ssh ${JETSON_USER}@${JETSON_IP} "mkdir -p ${JETSON_PATH}"

# Step 2: Copy project files (excluding .venv and __pycache__)
echo "📦 Copying project files..."
rsync -avz --progress \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='output' \
  --exclude='research_report.md' \
  ./ ${JETSON_USER}@${JETSON_IP}:${JETSON_PATH}/

# Step 3: Copy .env file (with API keys)
echo "🔑 Copying .env file..."
scp .env ${JETSON_USER}@${JETSON_IP}:${JETSON_PATH}/.env

# Step 4: Build Docker image on Jetson
echo "🐳 Building Docker image on Jetson..."
ssh ${JETSON_USER}@${JETSON_IP} << EOF
cd ${JETSON_PATH}
docker build -t research-agent:jetson .
EOF

echo "✅ Deployment complete!"
echo ""
echo "To run on Jetson:"
echo "  ssh ${JETSON_USER}@${JETSON_IP}"
echo "  cd ${JETSON_PATH}"
echo "  ./run-research-docker.sh 'Your query here'"