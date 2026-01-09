#!/bin/bash

# TruePix Setup Script
# Automated setup for backend and frontend

echo "🎯 TruePix - Automated Setup Script"
echo "===================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+"
    exit 1
fi

# Check Node version
echo "📋 Checking Node.js version..."
node --version

if [ $? -ne 0 ]; then
    echo "❌ Node.js is not installed. Please install Node.js 16+"
    exit 1
fi

echo ""
echo "✅ Prerequisites met!"
echo ""

# Backend Setup
echo "🔧 Setting up Backend..."
echo "------------------------"

cd backend

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit backend/.env with your Supabase credentials"
fi

echo "✅ Backend setup complete!"
echo ""

cd ..

# Frontend Setup
echo "🎨 Setting up Frontend..."
echo "------------------------"

cd frontend

# Install dependencies
echo "Installing Node.js dependencies..."
npm install

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
fi

echo "✅ Frontend setup complete!"
echo ""

cd ..

# Create model weights directory
echo "📁 Creating model directory..."
mkdir -p model/weights

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Configure Supabase (Optional for demo):"
echo "   - Edit backend/.env with your credentials"
echo "   - Or use demo mode (works without Supabase)"
echo ""
echo "2. Start Backend:"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
echo "3. Start Frontend (in new terminal):"
echo "   cd frontend"
echo "   npm start"
echo ""
echo "4. Open Browser:"
echo "   http://localhost:3000"
echo ""
echo "🚀 Happy Hacking!"
