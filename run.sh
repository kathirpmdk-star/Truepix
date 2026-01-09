#!/bin/bash

# TruePix Run Script
# Start both backend and frontend in separate terminals

echo "🚀 Starting TruePix Application"
echo "================================"
echo ""

# Check if setup has been run
if [ ! -d "backend/venv" ]; then
    echo "⚠️  Backend virtual environment not found"
    echo "Please run ./setup.sh first"
    exit 1
fi

if [ ! -d "frontend/node_modules" ]; then
    echo "⚠️  Frontend dependencies not found"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Start backend in background
echo "🔧 Starting Backend Server..."
cd backend
source venv/bin/activate
python main.py &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Start frontend
echo "🎨 Starting Frontend Server..."
cd frontend
npm start &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"
cd ..

echo ""
echo "✅ Both servers are starting!"
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for user interrupt
trap "echo ''; echo '🛑 Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit 0" INT

wait
