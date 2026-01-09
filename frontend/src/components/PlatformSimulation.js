import React, { useState, useEffect } from 'react';
import './PlatformSimulation.css';

function PlatformSimulation({ originalImage, originalResult }) {
  const [simulationData, setSimulationData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedPlatform, setSelectedPlatform] = useState('original');

  const runSimulation = async () => {
    setLoading(true);
    try {
      // Convert image URL to blob
      const response = await fetch(originalImage);
      const blob = await response.blob();
      
      // Create form data
      const formData = new FormData();
      formData.append('file', blob, 'image.jpg');
      
      // Call platform simulation API
      const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      const apiResponse = await fetch(`${API_URL}/api/simulate-platforms`, {
        method: 'POST',
        body: formData,
      });
      
      if (!apiResponse.ok) {
        throw new Error('Simulation failed');
      }
      
      const data = await apiResponse.json();
      setSimulationData(data);
    } catch (error) {
      console.error('Error running simulation:', error);
      alert('Failed to run platform simulation. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const platforms = [
    { id: 'original', name: 'Original', icon: '📷' },
    { id: 'whatsapp', name: 'WhatsApp', icon: '💬' },
    { id: 'instagram', name: 'Instagram', icon: '📸' },
    { id: 'facebook', name: 'Facebook', icon: '👥' }
  ];

  const getCurrentResult = () => {
    if (!simulationData) return null;
    
    if (selectedPlatform === 'original') {
      return simulationData.original_result;
    }
    return simulationData.platform_results[selectedPlatform];
  };

  useEffect(() => {
    runSimulation();
  }, []);

  if (loading) {
    return (
      <div className="platform-simulation">
        <div className="loading">
          <div className="spinner"></div>
          <p>Running platform simulations...</p>
        </div>
      </div>
    );
  }

  if (!simulationData) {
    return null;
  }

  const currentResult = getCurrentResult();
  const stabilityColor = simulationData.stability_score >= 75 ? '#4CAF50' : 
                         simulationData.stability_score >= 50 ? '#ffaa00' : '#ff4444';

  return (
    <div className="platform-simulation">
      <h2>Platform Stability Test</h2>
      
      <div className="stability-score">
        <div className="stability-header">
          <span>Stability Score</span>
          <span 
            className="stability-value"
            style={{ color: stabilityColor }}
          >
            {simulationData.stability_score.toFixed(1)}%
          </span>
        </div>
        <div className="stability-bar">
          <div 
            className="stability-fill"
            style={{ 
              width: `${simulationData.stability_score}%`,
              backgroundColor: stabilityColor
            }}
          ></div>
        </div>
        <p className="stability-description">
          {simulationData.stability_score >= 75 
            ? '✅ High stability - consistent predictions across platforms'
            : simulationData.stability_score >= 50
            ? '⚠️ Medium stability - some variation across platforms'
            : '❌ Low stability - significant variation across platforms'}
        </p>
      </div>
      
      <div className="platform-buttons">
        {platforms.map(platform => (
          <button
            key={platform.id}
            className={`platform-btn ${selectedPlatform === platform.id ? 'active' : ''}`}
            onClick={() => setSelectedPlatform(platform.id)}
          >
            <span className="platform-icon">{platform.icon}</span>
            <span className="platform-name">{platform.name}</span>
          </button>
        ))}
      </div>
      
      {currentResult && (
        <div className="platform-result">
          <h3>{platforms.find(p => p.id === selectedPlatform)?.name} Result</h3>
          
          <div className="result-card">
            <div className="result-prediction">
              <span className={`prediction-label ${currentResult.prediction === 'AI-Generated' ? 'ai' : 'real'}`}>
                {currentResult.prediction === 'AI-Generated' ? '🤖' : '📷'} {currentResult.prediction}
              </span>
            </div>
            
            <div className="result-confidence">
              <span className="label">Confidence:</span>
              <span className="value">{currentResult.confidence.toFixed(1)}%</span>
            </div>
            
            <div className="result-explanation">
              <h4>Analysis:</h4>
              {currentResult.explanation.split(' | ').map((line, index) => (
                <p key={index}>• {line}</p>
              ))}
            </div>
          </div>
          
          <div className="platform-info">
            <p className="info-text">
              {selectedPlatform === 'whatsapp' && 
                '💬 WhatsApp: Aggressive compression (512px, Quality 40) to reduce data usage'}
              {selectedPlatform === 'instagram' && 
                '📸 Instagram: Moderate compression (1080px, Quality 70) for feed optimization'}
              {selectedPlatform === 'facebook' && 
                '👥 Facebook: Balanced compression (960px, Quality 60) for fast loading'}
              {selectedPlatform === 'original' && 
                '📷 Original: Unmodified image as uploaded'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default PlatformSimulation;
