import React, { useState } from 'react';
import './App.css';
import LandingPage from './components/LandingPage';
import ImageUpload from './components/ImageUpload';
import ResultsPanel from './components/ResultsPanel';
import PlatformSimulation from './components/PlatformSimulation';

function App() {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showSimulation, setShowSimulation] = useState(false);

  const handleImageUpload = async (file) => {
    setLoading(true);
    setAnalysisResult(null);
    setShowSimulation(false);
    
    try {
      // Create preview URL
      const imageUrl = URL.createObjectURL(file);
      setUploadedImage(imageUrl);
      
      // Analyze image
      const formData = new FormData();
      formData.append('file', file);
      
      const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      const response = await fetch(`${API_URL}/analyze-image`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Analysis failed');
      }
      
      const result = await response.json();
      setAnalysisResult(result);
    } catch (error) {
      console.error('Error analyzing image:', error);
      alert('Failed to analyze image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setUploadedImage(null);
    setAnalysisResult(null);
    setShowSimulation(false);
  };

  return (
    <div className="App">
      {!uploadedImage ? (
        <LandingPage onUpload={handleImageUpload} />
      ) : (
        <div className="analysis-container">
          <div className="header">
            <h1 className="logo">TRUEPIX</h1>
            <button className="reset-btn" onClick={handleReset}>
              ← New Image
            </button>
          </div>
          
          <div className="content-grid">
            <div className="image-section">
              <h2>Uploaded Image</h2>
              <img src={uploadedImage} alt="Uploaded" className="uploaded-image" />
            </div>
            
            <div className="results-section">
              {loading ? (
                <div className="loading">
                  <div className="spinner"></div>
                  <p>Analyzing image...</p>
                </div>
              ) : analysisResult ? (
                <>
                  <ResultsPanel result={analysisResult} />
                  
                  <button 
                    className="simulation-btn"
                    onClick={() => setShowSimulation(!showSimulation)}
                  >
                    {showSimulation ? '← Back to Results' : '🔍 Test Platform Stability'}
                  </button>
                  
                  {showSimulation && (
                    <PlatformSimulation 
                      originalImage={uploadedImage}
                      originalResult={analysisResult}
                    />
                  )}
                </>
              ) : null}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
