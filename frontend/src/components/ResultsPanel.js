import React from 'react';
import './ResultsPanel.css';

function ResultsPanel({ result }) {
  const isAI = result.prediction === 'AI-Generated';
  const isUncertain = result.prediction === 'Uncertain';
  
  // Backend returns confidence as 0-100 percentage
  const confidencePercent = result.confidence.toFixed(1);
  
  // Use confidence_category from backend (High/Medium/Low)
  const confidenceLevel = result.confidence_category || 'Medium';
  
  const levelColor = {
    'High': '#4CAF50',
    'Medium': '#ffaa00',
    'Low': '#ff4444'
  };

  return (
    <div className="results-panel">
      <h2>Analysis Results</h2>
      
      <div className={`prediction-badge ${isAI ? 'ai' : isUncertain ? 'uncertain' : 'real'}`}>
        <span className="badge-icon">
          {isAI ? '🤖' : isUncertain ? '❓' : '📷'}
        </span>
        <span className="badge-text">{result.prediction}</span>
      </div>
      
      <div className="confidence-section">
        <div className="confidence-header">
          <span>Confidence</span>
          <span className="confidence-value">{confidencePercent}%</span>
        </div>
        <div className="confidence-bar">
          <div 
            className="confidence-fill"
            style={{ 
              width: `${confidencePercent}%`,
              backgroundColor: isAI ? '#ff6b6b' : isUncertain ? '#888' : '#4CAF50'
            }}
          ></div>
        </div>
      </div>
      
      <div className="risk-section">
        <span className="risk-label">Confidence Level:</span>
        <span 
          className="risk-badge"
          style={{ backgroundColor: levelColor[confidenceLevel] }}
        >
          {confidenceLevel}
        </span>
      </div>
      
      {result.metadata?.raw_scores && (
        <div className="scores-section">
          <h3>📊 Model Probabilities</h3>
          <div className="score-item">
            <span>Real Photo:</span>
            <span className="score-value">
              {(result.metadata.raw_scores.real * 100).toFixed(1)}%
            </span>
          </div>
          <div className="score-item">
            <span>AI-Generated:</span>
            <span className="score-value">
              {(result.metadata.raw_scores.ai_generated * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      )}
      
      <div className="explanation-section">
        <h3>🔍 Detailed Analysis</h3>
        <div className="explanation-text">
          {result.explanation.split(' | ').map((part, idx) => (
            <p key={idx} className="analysis-detail">{part}</p>
          ))}
        </div>
      </div>
      
      {result.metadata && (
        <div className="metadata-section">
          <p className="metadata-item">
            <strong>Model:</strong> {result.metadata.model_version}
          </p>
          {result.metadata.timestamp && (
            <p className="metadata-item">
              <strong>Analyzed:</strong> {new Date(result.metadata.timestamp).toLocaleString()}
            </p>
          )}
        </div>
      )}
      
      <div className="disclaimer">
        ⚠️ This is an AI prediction and may not be 100% accurate. Use as a guideline, not definitive proof.
      </div>
    </div>
  );
}

export default ResultsPanel;
