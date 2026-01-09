# 🎯 TruePix - Project Overview

## 🏆 Hackathon Project: AI Image Detection Platform

**Built**: January 2026  
**Duration**: 24-hour hackathon  
**Purpose**: Detect AI-generated images with explanations

---

## 📦 What's Included

### Complete Full-Stack Application

✅ **Backend (FastAPI + Python)**
- `/backend/main.py` - FastAPI server with CORS
- `/backend/model_inference.py` - CNN-based AI detector
- `/backend/platform_simulator.py` - Social media transformations
- `/backend/storage_manager.py` - Supabase integration
- `/backend/utils.py` - Image processing utilities
- `/backend/test_api.py` - API testing script

✅ **Frontend (React.js)**
- `/frontend/src/App.js` - Main application component
- `/frontend/src/components/LandingPage.js` - Hero landing page
- `/frontend/src/components/ImageUpload.js` - Drag-and-drop upload
- `/frontend/src/components/ResultsPanel.js` - Analysis display
- `/frontend/src/components/PlatformSimulation.js` - Platform testing
- All CSS files with gradient styling

✅ **Machine Learning**
- `/model/train_model.py` - Training script reference
- `/model/weights/README.md` - Model setup guide
- EfficientNet-B0 architecture
- Grad-CAM integration for explainability

✅ **Documentation**
- `README.md` - Comprehensive documentation
- `QUICKSTART.md` - 5-minute setup guide
- `CONTRIBUTING.md` - Contribution guidelines
- `LICENSE` - MIT License

✅ **Automation**
- `setup.sh` - Automated setup script
- `run.sh` - One-command startup
- `.env.example` files for configuration

---

## 🎨 Key Features Implemented

### 1. AI Detection Engine
- Binary classification (AI vs Real)
- Confidence scoring (0-100%)
- Risk levels (High/Medium/Uncertain)
- Human-readable explanations

### 2. Visual Cue Analysis
Detects:
- Unnatural hand structures
- Asymmetrical facial features
- Over-smooth textures
- Lighting inconsistencies
- Repeated patterns
- Perfect symmetry (AI hallmark)

### 3. Platform Simulation
Tests stability across:
- **WhatsApp**: 512px, 40% quality
- **Instagram**: 1080px, 70% quality
- **Facebook**: 960px, 60% quality

Computes stability score (0-100%)

### 4. Beautiful UI/UX
- Blue-to-cyan gradient background
- Robot vs Human hero section
- Animated components
- Responsive design
- Real-time loading states
- Clear visual feedback

---

## 🚀 Quick Start

```bash
# 1. Run automated setup
chmod +x setup.sh
./setup.sh

# 2. Start backend (Terminal 1)
cd backend
source venv/bin/activate
python main.py

# 3. Start frontend (Terminal 2)
cd frontend
npm start

# 4. Open browser
# http://localhost:3000
```

---

## 📊 Architecture

```
┌─────────────┐
│   Browser   │
│  (React.js) │
└──────┬──────┘
       │
       │ HTTP/REST
       │
┌──────▼──────┐
│   FastAPI   │
│   Backend   │
└──────┬──────┘
       │
       ├──────► Supabase Storage (Images)
       │
       ├──────► PyTorch Model (Inference)
       │
       └──────► Platform Simulator
```

---

## 🔧 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React.js 18.2 | UI/UX |
| Backend | FastAPI 0.108 | REST API |
| ML | PyTorch + timm | AI Detection |
| Storage | Supabase | Object Storage |
| Server | Uvicorn | ASGI Server |
| Styling | CSS3 | Animations |

---

## 📁 Project Structure

```
Truepix/
├── backend/              # Python FastAPI server
│   ├── main.py          # API endpoints
│   ├── model_inference.py
│   ├── platform_simulator.py
│   ├── storage_manager.py
│   ├── utils.py
│   ├── test_api.py
│   └── requirements.txt
│
├── frontend/            # React application
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── App.js
│   │   └── *.css
│   └── package.json
│
├── model/              # ML model & training
│   ├── train_model.py
│   └── weights/
│
├── setup.sh           # Automated setup
├── run.sh            # Start script
├── README.md         # Full documentation
├── QUICKSTART.md     # Quick guide
├── CONTRIBUTING.md   # Contribution guide
├── LICENSE           # MIT License
└── .gitignore
```

---

## ✨ Highlights

### What Makes This Special

1. **Explainability First**
   - Not just "AI or Real"
   - Clear reasons WHY
   - Visual cue identification

2. **Robustness Testing**
   - Platform simulation
   - Stability scoring
   - Real-world compression

3. **Production-Ready Structure**
   - Clean separation of concerns
   - Modular components
   - Easy to extend

4. **Demo-Ready**
   - Works without Supabase
   - Mock mode for testing
   - Pre-trained model fallback

5. **Well-Documented**
   - Comprehensive README
   - Code comments
   - API documentation
   - Setup guides

---

## 🎯 Use Cases

- **Content Moderation**: Flag AI-generated content
- **News Verification**: Check photo authenticity
- **Social Media**: Detect manipulated images
- **Research**: Study AI generation patterns
- **Education**: Learn about AI detection

---

## ⚠️ Important Notes

### Not 100% Accurate

This is a **detection tool**, not absolute proof:
- Use as guidance
- Combine with human verification
- Consider context
- Don't use for legal decisions alone

### Model Training Required

For production:
- Train on 50k+ labeled images
- Use CIFAKE, DiffusionDB datasets
- Fine-tune for 20+ epochs
- Test on diverse AI models

### Demo Mode

Current implementation:
- Uses pre-trained ImageNet weights
- Functional but less accurate
- Perfect for hackathon demo
- Replace with trained weights for production

---

## 🚀 Deployment Recommendations

### Frontend
- **Vercel** - Zero config
- **Netlify** - CI/CD integration
- **GitHub Pages** - Free hosting

### Backend
- **Railway** - Easy Python deployment
- **Render** - Free tier available
- **AWS Lambda** - Serverless option
- **DigitalOcean** - Simple VPS

### Storage
- **Supabase** - Free 1GB
- **Cloudinary** - Image CDN
- **AWS S3** - Scalable storage

### Model
- **Hugging Face** - Model hosting
- **TorchServe** - Production serving
- **ONNX Runtime** - Optimized inference

---

## 📈 Future Roadmap

### Phase 1: Core Improvements
- [ ] Train production model
- [ ] Add Grad-CAM visualization
- [ ] Improve explanation quality
- [ ] Optimize inference speed

### Phase 2: Features
- [ ] Batch processing
- [ ] History dashboard
- [ ] API authentication
- [ ] Rate limiting
- [ ] Multi-language support

### Phase 3: Advanced
- [ ] Multi-model ensemble
- [ ] EXIF metadata analysis
- [ ] Video detection
- [ ] Browser extension
- [ ] Mobile apps

---

## 🏆 Hackathon Success Criteria

✅ **Functional MVP**: Complete working application  
✅ **Explainability**: Clear reasons for predictions  
✅ **Innovation**: Platform stability testing  
✅ **UI/UX**: Beautiful, intuitive interface  
✅ **Documentation**: Comprehensive guides  
✅ **Demo-Ready**: Works out of the box  
✅ **Code Quality**: Clean, commented, modular  

---

## 📞 Support

**Documentation**: See `README.md`  
**Quick Start**: See `QUICKSTART.md`  
**Contributing**: See `CONTRIBUTING.md`  
**Issues**: Open GitHub issue  

---

## 🎉 Get Started Now!

```bash
# Clone and run
git clone <repo-url>
cd Truepix
./setup.sh

# Start coding!
```

---

**Built with ❤️ for AI transparency**

*TruePix - AI or Real? You decide.*
