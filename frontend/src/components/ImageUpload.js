import React, { useRef } from 'react';
import './ImageUpload.css';

function ImageUpload({ onUpload }) {
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Validate file type
      const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
      if (!validTypes.includes(file.type)) {
        alert('Please upload a JPG or PNG image');
        return;
      }
      
      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        return;
      }
      
      onUpload(file);
    }
  };

  const handleClick = () => {
    fileInputRef.current.click();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
      if (validTypes.includes(file.type)) {
        onUpload(file);
      } else {
        alert('Please upload a JPG or PNG image');
      }
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  return (
    <div className="image-upload">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/jpg,image/png"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />
      
      <div 
        className="upload-area"
        onClick={handleClick}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <div className="upload-icon">📤</div>
        <p className="upload-text">Click to upload or drag & drop</p>
        <p className="upload-hint">JPG or PNG (max 10MB)</p>
      </div>
    </div>
  );
}

export default ImageUpload;
