import React from 'react';
import './ResultsPanel.css';

function ResultsPanel({ result }) {
  const isAI = result.prediction === 'AI-Generated';
  
  // Calculate confidence percentage (our backend returns 0-1, convert to percentage)
  const confidencePercent = (result.confidence * 100).toFixed(1);
  
  // Determine risk level based on final_score
  const getRiskLevel = (score) => {
    if (score >= 0.7) return 'High';
    if (score >= 0.5) return 'Medium';
    return 'Uncertain';
  };
  
  const riskLevel = getRiskLevel(result.final_score);
  
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
          <span className="confidence-value">{confidencePercent}%</span>
        </div>
        <div className="confidence-bar">
          <div 
            className="confidence-fill"
            style={{ 
              width: `${confidencePercent}%`,
              backgroundColor: isAI ? '#ff6b6b' : '#4CAF50'
            }}
          ></div>
        </div>
      </div>
      
      <div className="risk-section">
        <span className="risk-label">Risk Level:</span>
        <span 
          className="risk-badge"
          style={{ backgroundColor: riskColor[riskLevel] }}
        >
          {riskLevel}
        </span>
      </div>
      
      <div className="scores-section">
        <h3>📊 Individual Scores</h3>
        <div className="score-item">
          <span>CNN Analysis:</span>
          <span className="score-value">{(result.individual_scores.cnn_score * 100).toFixed(1)}%</span>
        </div>
        <div className="score-item">
          <span>FFT Analysis:</span>
          <span className="score-value">{(result.individual_scores.fft_score * 100).toFixed(1)}%</span>
        </div>
        <div className="score-item">
          <span>Noise Analysis:</span>
          <span className="score-value">{(result.individual_scores.noise_score * 100).toFixed(1)}%</span>
        </div>
        <div className="score-item final-score">
          <span><strong>Final Score:</strong></span>
          <span className="score-value"><strong>{(result.final_score * 100).toFixed(1)}%</strong></span>
        </div>
      </div>
      
      <div className="explanation-section">
        <h3>🔍 Detailed Analysis</h3>
        <div className="explanation-text">
          <p className="main-explanation">{result.explanation}</p>
          
          {result.detailed_analysis && (
            <div className="detailed-breakdown">
              {result.detailed_analysis.cnn && (
                <p className="analysis-detail">
                  <strong>CNN:</strong> {result.detailed_analysis.cnn}
                </p>
              )}
              {result.detailed_analysis.fft && (
                <p className="analysis-detail">
                  <strong>FFT:</strong> {result.detailed_analysis.fft}
                </p>
              )}
              {result.detailed_analysis.noise && (
                <p className="analysis-detail">
                  <strong>Noise:</strong> {result.detailed_analysis.noise}
                </p>
              )}
            </div>
          )}
        </div>
      </div>
      
      <div className="metadata-section">
        <p className="metadata-item">
          <strong>Image ID:</strong> {result.image_id}
        </p>
        <p className="metadata-item">
          <strong>Analyzed:</strong> {new Date(result.timestamp).toLocaleString()}
        </p>
        <p className="metadata-item">
          <strong>Processing Time:</strong> {result.processing_time.toFixed(2)}s
        </p>
      </div>
      
      <div className="disclaimer">
        ⚠️ This is an AI prediction and may not be 100% accurate. Use as a guideline, not definitive proof.
      </div>
    </div>
  );
}

export default ResultsPanel;
