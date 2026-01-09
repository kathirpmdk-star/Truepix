import React from 'react';
import './LandingPage.css';
import ImageUpload from './ImageUpload';

function LandingPage({ onUpload }) {
  return (
    <div className="landing-page">
      {/* Left Edge - Robot (Fixed Position) */}
      <div className="edge-character robot-edge">
        <img src="/images/robot.png" alt="AI Robot" className="edge-img" />
      </div>
      
      {/* Right Edge - Human (Fixed Position) */}
      <div className="edge-character human-edge">
        <img src="/images/human.png" alt="Real Human" className="edge-img" />
      </div>

      {/* Main Content Center */}
      <div className="hero-section">
        <div className="hero-content">
          {/* Logo & Title */}
          <div className="logo-section">
            <h1 className="main-logo">TRUEPIX</h1>
            <div className="logo-underline"></div>
            <p className="tagline">Detect AI-Generated Images with Confidence</p>
          </div>
          
          {/* Description */}
          <p className="description">
            Upload an image to instantly detect if it's AI-generated or authentic.
            <br />
            Get detailed analysis with confidence scores and visual explanations.
          </p>
          
          {/* Upload Component */}
          <ImageUpload onUpload={onUpload} />
          
          {/* Feature Grid */}
          <div className="features">
            <div className="feature">
              <div className="feature-icon-wrapper">
                <svg className="feature-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M21 21L15 15M17 10C17 13.866 13.866 17 10 17C6.13401 17 3 13.866 3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <h3>Deep Analysis</h3>
              <p>Advanced neural network models trained on millions of images</p>
            </div>
            <div className="feature">
              <div className="feature-icon-wrapper">
                <svg className="feature-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M9 11L12 14L22 4M21 12V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <h3>Confidence Metrics</h3>
              <p>Precise probability scores with detailed classification reports</p>
            </div>
            <div className="feature">
              <div className="feature-icon-wrapper">
                <svg className="feature-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M13 2L3 14H12L11 22L21 10H12L13 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <h3>Explainable AI</h3>
              <p>Transparent insights into detection factors and reasoning</p>
            </div>
            <div className="feature">
              <div className="feature-icon-wrapper">
                <svg className="feature-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M2 12H22" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M12 2C14.5013 4.73835 15.9228 8.29203 16 12C15.9228 15.708 14.5013 19.2616 12 22C9.49872 19.2616 8.07725 15.708 8 12C8.07725 8.29203 9.49872 4.73835 12 2V2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <h3>Platform Simulation</h3>
              <p>Test detection stability across multiple social platforms</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default LandingPage;
