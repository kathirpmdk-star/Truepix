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
    'High': '#4ade80',
    'Medium': '#fbbf24',
    'Low': '#f87171'
  };

  return (
    <div className="results-panel">
      <div className="results-header">
        <h2>Detection Results</h2>
      </div>
      
      <div className={`prediction-card ${isAI ? 'ai' : isUncertain ? 'uncertain' : 'real'}`}>
        <div className="prediction-status">
          <span className="status-label">Classification</span>
          <span className="status-value">{result.prediction}</span>
        </div>
        
        <div className="confidence-display">
          <div className="confidence-header">
            <span className="confidence-label">Confidence Score</span>
            <span className="confidence-value">{confidencePercent}%</span>
          </div>
          <div className="confidence-bar">
            <div 
              className="confidence-fill"
              style={{ 
                width: `${confidencePercent}%`,
                backgroundColor: isAI ? '#6366f1' : isUncertain ? '#64748b' : '#4ade80'
              }}
            ></div>
          </div>
        </div>
        
        <div className="confidence-level">
          <span className="level-label">Assessment</span>
          <span 
            className="level-badge"
            style={{ 
              backgroundColor: levelColor[confidenceLevel],
              color: '#0a0e1a'
            }}
          >
            {confidenceLevel} Confidence
          </span>
        </div>
      </div>
      
      {result.metadata?.raw_scores && (
        <div className="probabilities-card">
          <h3 className="card-title">Model Probabilities</h3>
          <div className="probability-grid">
            <div className="probability-item">
              <span className="prob-label">Real Photo</span>
              <div className="prob-bar-container">
                <div 
                  className="prob-bar real"
                  style={{ width: `${(result.metadata.raw_scores.real * 100).toFixed(1)}%` }}
                ></div>
                <span className="prob-value">
                  {(result.metadata.raw_scores.real * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            <div className="probability-item">
              <span className="prob-label">AI-Generated</span>
              <div className="prob-bar-container">
                <div 
                  className="prob-bar ai"
                  style={{ width: `${(result.metadata.raw_scores.ai_generated * 100).toFixed(1)}%` }}
                ></div>
                <span className="prob-value">
                  {(result.metadata.raw_scores.ai_generated * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
      
      <div className="analysis-card">
        <h3 className="card-title">Technical Analysis</h3>
        <div className="analysis-content">
          {result.explanation.split(' | ').map((part, idx) => (
            <div key={idx} className="analysis-item">
              <div className="analysis-indicator"></div>
              <p className="analysis-text">{part}</p>
            </div>
          ))}
        </div>
      </div>
      
      {result.metadata && (
        <div className="metadata-card">
          <div className="metadata-grid">
            <div className="metadata-item">
              <span className="meta-label">Model Version</span>
              <span className="meta-value">{result.metadata.model_version}</span>
            </div>
            {result.metadata.timestamp && (
              <div className="metadata-item">
                <span className="meta-label">Analyzed</span>
                <span className="meta-value">
                  {new Date(result.metadata.timestamp).toLocaleString()}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
      
      <div className="disclaimer-card">
        <p>This is an AI-powered prediction based on multiple analysis techniques. Results should be used as guidance and may not be 100% accurate in all cases.</p>
      </div>
    </div>
  );
}

export default ResultsPanel;
