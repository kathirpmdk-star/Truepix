# 🔧 TruePix - Troubleshooting Guide

Common issues and solutions for running TruePix

---

## 🐍 Backend Issues

### Issue: `ModuleNotFoundError: No module named 'fastapi'`

**Cause**: Python dependencies not installed

**Solution**:
```bash
cd backend
source venv/bin/activate  # Activate virtual environment first!
pip install -r requirements.txt
```

### Issue: `Address already in use` on port 8000

**Cause**: Another process using port 8000

**Solution**:
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill -9

# Or use different port
API_PORT=8001 python main.py

# Windows
netstat -ano | findstr :8000
taskkill /PID <process_id> /F
```

### Issue: `Torch not compiled with CUDA enabled`

**Cause**: GPU support not configured (expected on CPU)

**Solution**: This is normal! The app works on CPU. To use GPU:
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: `Supabase connection failed`

**Cause**: Supabase credentials missing or invalid

**Solution**: App works in demo mode without Supabase!
- Images stored temporarily
- Mock URLs generated
- All features functional

To enable Supabase:
1. Create account at supabase.com
2. Create project and storage bucket
3. Update `backend/.env`:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

### Issue: Model loading slow

**Cause**: First-time download of pre-trained weights

**Solution**: Wait 1-2 minutes on first run. Subsequent runs are faster.

---

## ⚛️ Frontend Issues

### Issue: `npm: command not found`

**Cause**: Node.js not installed

**Solution**:
```bash
# macOS
brew install node

# Or download from nodejs.org
```

### Issue: Port 3000 already in use

**Cause**: Another React app running

**Solution**:
```bash
# Use different port
PORT=3001 npm start

# Or kill existing process
lsof -ti:3000 | xargs kill -9
```

### Issue: `Cannot connect to backend`

**Cause**: Backend not running or wrong URL

**Solution**:
1. Verify backend is running: http://localhost:8000
2. Check `frontend/.env`:
```env
REACT_APP_API_URL=http://localhost:8000
```
3. Restart frontend:
```bash
npm start
```

### Issue: Image upload fails with CORS error

**Cause**: CORS middleware not configured

**Solution**: Backend already has CORS enabled. If issue persists:
1. Stop backend
2. Check `backend/main.py` has:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
3. Restart backend

### Issue: `npm install` fails

**Cause**: Package conflicts or corrupted cache

**Solution**:
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

---

## 🖼️ Image Processing Issues

### Issue: "Invalid image file" error

**Cause**: Unsupported format or corrupted image

**Solution**:
- Use JPG or PNG only
- File size < 10MB
- Minimum size: 32x32 pixels
- Try different image

### Issue: Analysis takes too long

**Cause**: Large image or CPU processing

**Solution**:
- Use smaller images (< 2MB recommended)
- Backend resizes to 224x224 automatically
- Consider GPU acceleration for production

### Issue: Platform simulation fails

**Cause**: Insufficient memory or network issue

**Solution**:
```bash
# Check available memory
free -h  # Linux
vm_stat  # macOS

# Reduce image size before upload
```

---

## 🚀 Setup Issues

### Issue: `Permission denied` when running setup.sh

**Cause**: Script not executable

**Solution**:
```bash
chmod +x setup.sh run.sh
./setup.sh
```

### Issue: Python version too old

**Cause**: Python 3.8 or earlier

**Solution**:
```bash
# Check version
python3 --version

# Install Python 3.9+
# macOS
brew install python@3.9

# Ubuntu/Debian
sudo apt-get install python3.9
```

### Issue: Virtual environment not activating

**Cause**: Wrong activation command

**Solution**:
```bash
# macOS/Linux
source venv/bin/activate

# Windows Command Prompt
venv\Scripts\activate.bat

# Windows PowerShell
venv\Scripts\Activate.ps1

# Verify activation
which python  # Should show venv/bin/python
```

---

## 🗄️ Storage Issues

### Issue: Images not saving

**Cause**: Supabase not configured (expected in demo mode)

**Solution**: This is normal! Demo mode provides mock URLs.

For persistent storage:
1. Setup Supabase (free tier)
2. Create bucket: `truepix-images`
3. Make bucket public
4. Update `.env` with credentials

### Issue: "Bucket not found"

**Cause**: Supabase bucket not created

**Solution**:
1. Go to Supabase Dashboard
2. Storage → New Bucket
3. Name: `truepix-images`
4. Make public: ✅
5. Restart backend

---

## 🧪 Testing Issues

### Issue: `test_api.py` fails

**Cause**: Backend not running or test image missing

**Solution**:
```bash
# 1. Start backend first
cd backend
python main.py

# 2. In new terminal, run tests
python test_api.py

# 3. Provide test image path in test_api.py:
test_image = "path/to/your/image.jpg"
```

### Issue: API returns 500 error

**Cause**: Server error (check logs)

**Solution**:
1. Check backend terminal for error messages
2. Common causes:
   - Model file missing
   - PIL/Pillow image processing error
   - Numpy version conflict
3. Restart backend after fixing

---

## 🌐 Browser Issues

### Issue: Landing page not loading

**Cause**: Build failed or wrong URL

**Solution**:
```bash
# Rebuild frontend
cd frontend
npm run build

# Check console for errors (F12)
# Verify URL: http://localhost:3000
```

### Issue: Styles not applied

**Cause**: CSS not loading

**Solution**:
1. Clear browser cache (Ctrl+Shift+R)
2. Check browser console for 404 errors
3. Verify CSS files in `frontend/src/components/`

### Issue: Upload button not responding

**Cause**: JavaScript error

**Solution**:
1. Open browser console (F12)
2. Check for errors
3. Verify `ImageUpload.js` exists
4. Restart frontend

---

## 🔐 Environment Variables

### Issue: `.env` file not loading

**Cause**: Wrong file location or name

**Solution**:
```bash
# Backend .env location
backend/.env  # NOT root/.env

# Frontend .env location
frontend/.env

# Verify with:
ls -la backend/.env
ls -la frontend/.env

# If missing, copy from template:
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

### Issue: API_URL not updating

**Cause**: Frontend not reading .env

**Solution**:
```bash
# .env must start with REACT_APP_
REACT_APP_API_URL=http://localhost:8000

# Restart frontend after .env changes
# Stop: Ctrl+C
# Start: npm start
```

---

## 💻 macOS Specific

### Issue: `xcrun: error: invalid active developer path`

**Cause**: Xcode command-line tools missing

**Solution**:
```bash
xcode-select --install
```

### Issue: Python SSL certificate error

**Cause**: macOS SSL certificates

**Solution**:
```bash
# Run the Install Certificates command
/Applications/Python\ 3.9/Install\ Certificates.command
```

---

## 🪟 Windows Specific

### Issue: PowerShell script execution disabled

**Cause**: Execution policy restricted

**Solution**:
```powershell
# Run as Administrator
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: `python` command not found

**Cause**: Python not in PATH

**Solution**:
1. Reinstall Python from python.org
2. Check "Add Python to PATH" during installation
3. Or use `py` command instead:
```bash
py -m venv venv
py main.py
```

---

## 🐧 Linux Specific

### Issue: `pip` not found

**Cause**: pip not installed

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install python3-pip

# CentOS/RHEL
sudo yum install python3-pip
```

### Issue: Port permission denied

**Cause**: Ports < 1024 require sudo

**Solution**:
```bash
# Use port > 1024 (already default 8000)
# Or run with sudo (not recommended)
```

---

## 🔍 Debug Mode

### Enable Verbose Logging

**Backend**:
```python
# In backend/main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Frontend**:
```bash
# Run with debug
REACT_APP_DEBUG=true npm start
```

### Check API Documentation

Visit: http://localhost:8000/docs
- Interactive API testing
- See request/response formats
- Test endpoints directly

---

## 📞 Getting Help

### 1. Check Documentation
- `README.md` - Main docs
- `QUICKSTART.md` - Setup guide
- `FILE_STRUCTURE.md` - Project layout

### 2. Search Error Messages
- Google the exact error message
- Check Stack Overflow
- Search GitHub issues

### 3. Debug Systematically
1. Verify prerequisites installed
2. Check both terminals running
3. Verify URLs correct
4. Check browser console
5. Read error messages carefully

### 4. Fresh Start
```bash
# Nuclear option - restart everything
cd Truepix

# Stop all processes (Ctrl+C in both terminals)

# Backend
cd backend
deactivate  # Exit venv
rm -rf venv  # Remove venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py

# Frontend (new terminal)
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

---

## ✅ Quick Diagnostic

Run this checklist:

```bash
# 1. Python version
python3 --version  # Should be 3.9+

# 2. Node version
node --version  # Should be 16+

# 3. Backend dependencies
cd backend
pip list | grep fastapi  # Should show fastapi

# 4. Frontend dependencies
cd frontend
npm list react  # Should show react@18.2.0

# 5. Backend running
curl http://localhost:8000  # Should return JSON

# 6. Frontend running
curl http://localhost:3000  # Should return HTML

# If all pass, app should work!
```

---

## 🆘 Still Having Issues?

1. **Read error messages carefully** - They usually tell you what's wrong
2. **Check both terminal windows** - Backend and frontend logs
3. **Try the demo without Supabase** - Should still work
4. **Use smaller test images** - < 1MB JPG files
5. **Restart everything** - Sometimes that's all you need

Remember: The app is designed to work in demo mode without external services!

---

**Most common issue**: Forgetting to activate virtual environment!

**Solution**: Always `source venv/bin/activate` before running backend
