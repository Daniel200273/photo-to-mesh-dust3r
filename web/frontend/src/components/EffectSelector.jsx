import React from 'react'
import './EffectSelector.css'

const EFFECT_DESCRIPTIONS = {
  low_poly: 'Clean, geometric lowpoly aesthetic with smooth shading',
  voxel: 'Minecraft-style blocky voxel representation',
  soft_voxel: 'Smooth voxel look with sphere nodes',
  hologram: 'Neon wireframe hologram effect'
}

export default function EffectSelector({
  effects,
  selectedEffect,
  onEffectChange,
  loading,
  onProcess
}) {
  return (
    <div className="effect-selector">
      <h2>Choose an Effect</h2>
      <div className="effects-grid">
        {Object.entries(effects).map(([key, label]) => (
          <div
            key={key}
            className={`effect-option ${selectedEffect === key ? 'active' : ''}`}
            onClick={() => !loading && onEffectChange(key)}
          >
            <div className="effect-icon">
              {key === 'low_poly' && '🎯'}
              {key === 'voxel' && '⚒️'}
              {key === 'soft_voxel' && '🔷'}
              {key === 'hologram' && '✨'}
            </div>
            <h3>{label}</h3>
            <p>{EFFECT_DESCRIPTIONS[key]}</p>
          </div>
        ))}
      </div>

      <button
        className="btn-process"
        onClick={onProcess}
        disabled={loading}
      >
        {loading ? '⏳ Processing...' : '🚀 Generate 3D Model'}
      </button>
    </div>
  )
}
