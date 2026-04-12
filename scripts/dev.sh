#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

cleanup() {
  echo ""
  echo "Shutting down..."
  kill $FLASK_PID 2>/dev/null
  wait $FLASK_PID 2>/dev/null
  echo "Done."
}
trap cleanup EXIT INT TERM

# Install backend
uv sync --all-packages

# Install frontend
cd app/frontend
npm install --silent
npm run build
cd ../..

# Start Flask (serves React build + API)
echo "Starting at http://127.0.0.1:5000"
echo "Press Ctrl+C to stop."
uv run ga-app &
FLASK_PID=$!
wait $FLASK_PID
