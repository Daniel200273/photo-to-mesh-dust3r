import React, { useState, useEffect } from 'react'
import axios from 'axios'
import ImageUploader from './components/ImageUploader'
import EffectSelector from './components/EffectSelector'
import MeshViewer from './components/MeshViewer'
import './App.css'

export default function App() {
  const [step, setStep] = useState('upload') // 'upload' | 'effect' | 'preview' | 'error'
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [selectedEffect, setSelectedEffect] = useState('low_poly')
  const [meshData, setMeshData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [downloadUrl, setDownloadUrl] = useState('')
  const [effects, setEffects] = useState({})

  useEffect(() => {
    // Fetch available effects
    fetchEffects()
  }, [])

  const fetchEffects = async () => {
    try {
      const response = await axios.get('/api/effects')
      setEffects(response.data.effects)
    } catch (err) {
      console.error('Error fetching effects:', err)
    }
  }

  const handleImagesSelected = (files) => {
    setUploadedFiles(Array.from(files))
    setStep('effect')
  }

  const handleProcessImages = async () => {
    if (uploadedFiles.length === 0) {
      setError('Please select at least 3 images')
      return
    }

    setLoading(true)
    setError('')

    try {
      const formData = new FormData()
      uploadedFiles.forEach(file => {
        formData.append('files', file)
      })
      formData.append('effect', selectedEffect)

      const response = await axios.post('/api/process', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      if (response.data.success) {
        setMeshData(response.data.mesh_data)
        setDownloadUrl(response.data.download_url)
        setStep('preview')
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Error processing images')
      setStep('error')
    } finally {
      setLoading(false)
    }
  }

  const resetApp = () => {
    setStep('upload')
    setUploadedFiles([])
    setMeshData(null)
    setError('')
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🎨 FewShot-NeRF Studio</h1>
        <p>Transform your photos into stunning 3D models</p>
      </header>

      <main className="app-main">
        {step === 'upload' && (
          <ImageUploader onFilesSelected={handleImagesSelected} />
        )}

        {step === 'effect' && (
          <div className="effect-container">
            <div className="preview-section">
              <h2>Selected Images ({uploadedFiles.length})</h2>
              <div className="image-grid">
                {uploadedFiles.map((file, idx) => (
                  <div key={idx} className="image-thumb">
                    <img src={URL.createObjectURL(file)} alt={`Preview ${idx + 1}`} />
                  </div>
                ))}
              </div>
            </div>

            <EffectSelector
              effects={effects}
              selectedEffect={selectedEffect}
              onEffectChange={setSelectedEffect}
              loading={loading}
              onProcess={handleProcessImages}
            />
          </div>
        )}

        {step === 'preview' && meshData && (
          <div className="preview-container">
            <MeshViewer meshData={meshData} />
            <div className="preview-controls">
              <a href={downloadUrl} download className="btn btn-download">
                ⬇️ Download Model
              </a>
              <button onClick={resetApp} className="btn btn-reset">
                ✏️ Start Over
              </button>
            </div>
          </div>
        )}

        {step === 'error' && (
          <div className="error-container">
            <h2>⚠️ Error</h2>
            <p>{error}</p>
            <button onClick={resetApp} className="btn btn-reset">
              Try Again
            </button>
          </div>
        )}

        {loading && (
          <div className="loading-overlay">
            <div className="spinner"></div>
            <p>Processing your 3D model... This may take a minute</p>
          </div>
        )}
      </main>
    </div>
  )
}
