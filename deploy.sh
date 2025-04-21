#!/bin/bash

# 🔁 Step 1: Stop and remove existing container (if running)
echo "🛑 Stopping and removing any existing container..."
docker stop recircle-chatbot-container 2>/dev/null
docker rm recircle-chatbot-container 2>/dev/null

# 🔨 Step 2: Build the Docker image
echo "🔨 Building the Docker image..."
docker build -t recircle-chatbot .

# 🚀 Step 3: Run the container with volume mount for persistent DB
echo "🚀 Running the container..."
docker run -d \
  --name recircle-chatbot-container \
  -p 8000:8000 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  recircle-chatbot

# ✅ Done
echo "✅ Deployment complete. Container is running on port 8000."
