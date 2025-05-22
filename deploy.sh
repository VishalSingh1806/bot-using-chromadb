#!/bin/bash

set -e  # ⛑️ Exit immediately if any command fails

echo "📥 Pulling latest changes from Git..."
git reset --hard HEAD      # Discard local changes
git clean -fd              # Remove untracked files/folders
git pull origin master       # Or replace 'main' with your branch name

# 🔁 Step 1: Stop and remove existing container (if running)
echo "🛑 Stopping and removing any existing container..."
docker stop recircle-chatbot-container-master 2>/dev/null || true
docker rm recircle-chatbot-container-master 2>/dev/null || true

# 🔨 Step 2: Build the Docker image
echo "🔨 Building the Docker image..."
docker build -t recircle-chatbot-master .

# 🚀 Step 3: Run the container with volume mount for persistent DB
echo "🚀 Running the container..."
docker run -d \
  --name recircle-chatbot-container-master \
  -p 8000:8000 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  recircle-chatbot-master

# ✅ Done
echo "✅ Deployment complete. Container is running on port 8000."
