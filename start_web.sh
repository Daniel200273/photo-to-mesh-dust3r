#!/bin/bash

# Start FewShot-NeRF Web Application
# This script launches both the FastAPI backend and React frontend

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$PROJECT_ROOT/web"

echo "🚀 Starting FewShot-NeRF Web Application..."
echo "Project root: $PROJECT_ROOT"

# Activate virtual environment
echo "📦 Activating virtual environment..."
source "$PROJECT_ROOT/.venv/bin/activate"

# Install/update backend dependencies
echo "📥 Installing backend dependencies..."
cd "$WEB_DIR/backend"
pip install -q -r requirements.txt 2>&1 | grep -v "already satisfied" || true

# Install/update frontend dependencies (if needed)
echo "📥 Installing frontend dependencies..."
cd "$WEB_DIR/frontend"
if [ ! -d "node_modules" ]; then
    echo "  → Running npm install (first time)..."
    npm install --silent 2>&1 | tail -5 || npm install 2>&1 | tail -10
else
    echo "  → Using existing node_modules"
fi

echo ""
echo "✅ Dependencies installed!"
echo ""

# Start backend in background
echo "🔧 Starting FastAPI backend on http://127.0.0.1:8000..."
cd "$WEB_DIR/backend"
python app.py > "$WEB_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 2

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start. Check log:"
    cat "$WEB_DIR/backend.log"
    exit 1
fi

echo "✅ Backend started!"
echo ""

# Start frontend
echo "🎨 Starting React frontend on http://localhost:3000..."
echo "   Open http://localhost:3000 in your browser"
echo ""
cd "$WEB_DIR/frontend"
npm run dev

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null || true" EXIT
