#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

# Install backend
uv sync --all-packages

# Install frontend
cd app/frontend
npm install --silent
npm run build
cd ../..

# Start Flask (serves React build + API)
echo "Starting at http://127.0.0.1:5000"
uv run ga-app
