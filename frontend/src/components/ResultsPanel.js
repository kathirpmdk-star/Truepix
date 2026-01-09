import React from 'react';
import './ResultsPanel.css';

function ResultsPanel({ result }) {
  const isAI = result.prediction === 'AI-Generated';
  const riskColor = {
    'High': '#ff4444',
    'Medium': '#ffaa00',
    'Uncertain': '#888888'
  };

  return (
    <div className="results-panel">
      <h2>Analysis Results</h2>
      
      <div className={`prediction-badge ${isAI ? 'ai' : 'real'}`}>
        <span className="badge-icon">{isAI ? '🤖' : '📷'}</span>
        <span className="badge-text">{result.prediction}</span>
      </div>
      
      <div className="confidence-section">
        <div className="confidence-header">
          <span>Confidence</span>
          <span className="confidence-value">{result.confidence.toFixed(1)}%</span>
        </div>
        <div className="confidence-bar">
          <div 
            className="confidence-fill"
            style={{ 
              width: `${result.confidence}%`,
              backgroundColor: isAI ? '#ff6b6b' : '#4CAF50'
            }}
          ></div>
        </div>
      </div>
      
      <div className="risk-section">
        <span className="risk-label">Risk Level:</span>
        <span 
          className="risk-badge"
          style={{ backgroundColor: riskColor[result.risk_level] }}
        >
          {result.risk_level}
        </span>
      </div>
      
      <div className="explanation-section">
        <h3>🔍 Detailed Analysis</h3>
        <div className="explanation-text">
          {result.explanation.split(' | ').map((line, index) => (
            <p key={index} className="explanation-line">
              <span className="bullet">•</span> {line}
            </p>
          ))}
        </div>
      </div>
      
      <div className="metadata-section">
        <p className="metadata-item">
          <strong>Image ID:</strong> {result.metadata.image_id}
        </p>
        <p className="metadata-item">
          <strong>Analyzed:</strong> {new Date(result.metadata.timestamp).toLocaleString()}
        </p>
        <p className="metadata-item">
          <strong>Model:</strong> {result.metadata.model_version}
        </p>
      </div>
      
      <div className="disclaimer">
        ⚠️ This is an AI prediction and may not be 100% accurate. Use as a guideline, not definitive proof.
      </div>
    </div>
  );
}

export default ResultsPanel;
