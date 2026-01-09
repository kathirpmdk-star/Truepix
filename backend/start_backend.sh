#!/bin/bash

# TruePix Backend Startup Script
# Installs dependencies and starts the FastAPI server

echo "========================================="
echo "🚀 TruePix Backend Startup"
echo "========================================="

# Navigate to backend directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# TruePix Backend Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Optional: Supabase Configuration (for cloud storage)
# SUPABASE_URL=your_supabase_url
# SUPABASE_KEY=your_supabase_key
# SUPABASE_BUCKET=truepix-images
EOF
fi

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Starting TruePix API server..."
echo ""

# Start the server using the new main file
python main_new.py
