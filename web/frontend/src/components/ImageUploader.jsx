import React, { useState } from 'react'
import './ImageUploader.css'

export default function ImageUploader({ onFilesSelected }) {
  const [dragActive, setDragActive] = useState(false)
  const [selectedCount, setSelectedCount] = useState(0)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedCount(e.dataTransfer.files.length)
      onFilesSelected(e.dataTransfer.files)
    }
  }

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedCount(e.target.files.length)
      onFilesSelected(e.target.files)
    }
  }

  return (
    <div className="uploader-container">
      <div className="uploader-card">
        <div
          className={`dropzone ${dragActive ? 'active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            id="file-input"
            multiple
            accept="image/jpeg,image/png,image/heic"
            onChange={handleChange}
            className="file-input"
          />
          
          <label htmlFor="file-input" className="dropzone-label">
            <div className="dropzone-icon">📸</div>
            <h2>Upload Your Photos</h2>
            <p>Drag and drop 3+ images here, or click to select</p>
            <p className="dropzone-hint">Supports: JPG, PNG, HEIC</p>
            {selectedCount > 0 && (
              <p className="dropzone-selected">✓ {selectedCount} images selected</p>
            )}
          </label>
        </div>

        <div className="uploader-info">
          <h3>📝 Tips for best results:</h3>
          <ul>
            <li>Use 10-15 images for optimal 3D reconstruction</li>
            <li>Capture the object from different angles</li>
            <li>Ensure good lighting and clear view of the object</li>
            <li>Keep camera settings consistent (same zoom level)</li>
            <li>Avoid blurry photos</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
