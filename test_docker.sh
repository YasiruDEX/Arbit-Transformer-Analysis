#!/bin/bash

# Docker Build and Test Script for Render Deployment
# This script helps you test the Docker configuration locally before deploying

echo "============================================"
echo "🐳 Docker Build & Test for Render"
echo "============================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="thermal-api"
CONTAINER_NAME="thermal-api-test"
PORT=8000

echo ""
echo "Step 1: Cleaning up old containers and images..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

echo ""
echo "Step 2: Building Docker image..."
echo "⏳ This may take 3-5 minutes on first build..."
if docker build -t $IMAGE_NAME .; then
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

echo ""
echo "Step 3: Checking image size..."
IMAGE_SIZE=$(docker images $IMAGE_NAME --format "{{.Size}}")
echo "📦 Image size: $IMAGE_SIZE"
if [[ $(docker images $IMAGE_NAME --format "{{.Size}}" | grep -oE '[0-9.]+' | head -1 | cut -d'.' -f1) -gt 2000 ]]; then
    echo -e "${YELLOW}⚠️  Warning: Image is large (>2GB). Consider optimization.${NC}"
else
    echo -e "${GREEN}✓ Image size is reasonable${NC}"
fi

echo ""
echo "Step 4: Starting container..."
if docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:$PORT \
    -e PORT=$PORT \
    $IMAGE_NAME; then
    echo -e "${GREEN}✓ Container started${NC}"
else
    echo -e "${RED}✗ Failed to start container${NC}"
    exit 1
fi

echo ""
echo "Step 5: Waiting for API to be ready..."
sleep 5

echo ""
echo "Step 6: Testing API endpoints..."

# Test 1: Health check
echo "Testing /health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:$PORT/health)
if [[ $HEALTH_RESPONSE == *"healthy"* ]]; then
    echo -e "${GREEN}✓ Health check passed${NC}"
    echo "  Response: $HEALTH_RESPONSE" | head -c 100
    echo "..."
else
    echo -e "${RED}✗ Health check failed${NC}"
    echo "  Response: $HEALTH_RESPONSE"
fi

echo ""

# Test 2: Root endpoint
echo "Testing / endpoint..."
ROOT_RESPONSE=$(curl -s http://localhost:$PORT/)
if [[ $ROOT_RESPONSE == *"Thermal Anomaly Detection API"* ]]; then
    echo -e "${GREEN}✓ Root endpoint working${NC}"
else
    echo -e "${YELLOW}⚠️  Root endpoint response unexpected${NC}"
fi

echo ""

# Test 3: Config endpoint
echo "Testing /config endpoint..."
CONFIG_RESPONSE=$(curl -s http://localhost:$PORT/config)
if [[ $CONFIG_RESPONSE == *"detection"* ]]; then
    echo -e "${GREEN}✓ Config endpoint working${NC}"
else
    echo -e "${YELLOW}⚠️  Config endpoint response unexpected${NC}"
fi

echo ""
echo "Step 7: Checking container logs..."
echo "----------------------------------------"
docker logs $CONTAINER_NAME --tail 20
echo "----------------------------------------"

echo ""
echo "============================================"
echo "🎉 Docker Test Complete!"
echo "============================================"
echo ""
echo "Container is running at: http://localhost:$PORT"
echo "Interactive docs: http://localhost:$PORT/docs"
echo ""
echo "Commands:"
echo "  View logs:      docker logs $CONTAINER_NAME"
echo "  Stop container: docker stop $CONTAINER_NAME"
echo "  Remove:         docker rm $CONTAINER_NAME"
echo "  Shell access:   docker exec -it $CONTAINER_NAME /bin/bash"
echo ""
echo "Next steps:"
echo "  1. Test API with sample images"
echo "  2. If everything works, push to GitHub"
echo "  3. Deploy to Render.com"
echo ""
echo "Keep container running? (y/n)"
read -r KEEP_RUNNING

if [[ $KEEP_RUNNING != "y" ]]; then
    echo "Stopping and removing container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    echo -e "${GREEN}✓ Cleanup complete${NC}"
fi
