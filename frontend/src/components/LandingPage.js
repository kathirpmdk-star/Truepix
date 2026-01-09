import React, { useRef } from 'react';
import './LandingPage.css';
import ImageUpload from './ImageUpload';

function LandingPage({ onUpload }) {
  return (
    <div className="landing-page">
      <div className="hero-section">
        <div className="hero-content">
          <div className="hero-image-container">
            {/* Left side - Robot */}
            <div className="character robot">
              <div className="robot-face">
                <div className="robot-eye left"></div>
                <div className="robot-eye right"></div>
                <div className="robot-mouth"></div>
              </div>
            </div>
            
            {/* Center - Logo */}
            <div className="logo-container">
              <h1 className="main-logo">TRUEPIX</h1>
              <p className="tagline">AI or Real?</p>
            </div>
            
            {/* Right side - Human */}
            <div className="character human">
              <div className="human-face">
                <div className="human-eye left"></div>
                <div className="human-eye right"></div>
                <div className="human-mouth"></div>
              </div>
            </div>
          </div>
          
          <p className="description">
            Upload an image to detect if it's AI-generated or a real photograph.
            <br />
            Get instant analysis with detailed explanations.
          </p>
          
          <ImageUpload onUpload={onUpload} />
          
          <div className="features">
            <div className="feature">
              <span className="icon">🔍</span>
              <span>Deep Analysis</span>
            </div>
            <div className="feature">
              <span className="icon">📊</span>
              <span>Confidence Score</span>
            </div>
            <div className="feature">
              <span className="icon">💡</span>
              <span>Clear Explanations</span>
            </div>
            <div className="feature">
              <span className="icon">📱</span>
              <span>Platform Testing</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default LandingPage;
