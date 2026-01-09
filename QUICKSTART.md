# 🚀 Quick Start Guide

Welcome to **TruePix**! This guide will get you up and running in 5 minutes.

## Prerequisites

- Python 3.9+
- Node.js 16+
- Terminal/Command Line

## Option 1: Automated Setup (Recommended)

### macOS/Linux

```bash
# Make script executable
chmod +x setup.sh

# Run setup
./setup.sh

# Start application (in separate terminals)
# Terminal 1 - Backend:
cd backend
source venv/bin/activate
python main.py

# Terminal 2 - Frontend:
cd frontend
npm start
```

### Windows

```powershell
# Backend
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python main.py

# Frontend (new terminal)
cd frontend
npm install
copy .env.example .env
npm start
```

## Option 2: Manual Setup

### Backend

```bash
cd backend

# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate
# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start server
python main.py
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env

# Start development server
npm start
```

## Access the Application

- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Demo Mode

The app works **without Supabase** in demo mode:
- ✅ Image upload functional
- ✅ AI detection works
- ✅ Platform simulation active
- ⚠️ Images not permanently stored

## Optional: Supabase Setup

1. Create account at [supabase.com](https://supabase.com)
2. Create new project
3. Go to Storage → Create bucket `truepix-images` (public)
4. Copy credentials to `backend/.env`:
   ```env
   SUPABASE_URL=your_project_url
   SUPABASE_KEY=your_anon_key
   ```

## Test the Application

1. **Upload Test Image**: Use any JPG/PNG
2. **View Results**: Check prediction and confidence
3. **Test Platforms**: Click "Test Platform Stability"
4. **Compare**: See how different platforms affect results

## Troubleshooting

### Port Already in Use

Backend (8000):
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

Frontend (3000):
```bash
# Use different port
PORT=3001 npm start
```

### Python Dependencies Fail

```bash
# Upgrade pip
pip install --upgrade pip

# Install individually
pip install fastapi uvicorn torch torchvision
```

### Node Dependencies Fail

```bash
# Clear cache
npm cache clean --force

# Reinstall
rm -rf node_modules package-lock.json
npm install
```

## Need Help?

See full documentation in [README.md](README.md)

## 🎉 You're Ready!

Upload an image and start detecting AI-generated content!
