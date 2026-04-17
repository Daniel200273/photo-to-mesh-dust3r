import React, { useEffect, useRef } from 'react'
import * as THREE from 'three'
import './MeshViewer.css'

export default function MeshViewer({ meshData }) {
  const containerRef = useRef(null)
  const sceneRef = useRef(null)
  const rendererRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current || !meshData) return

    // Scene setup
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0xf5f5f5)
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    )
    camera.position.z = 100

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight)
    renderer.setPixelRatio(window.devicePixelRatio)
    containerRef.current.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Create mesh from data
    const geometry = new THREE.BufferGeometry()
    
    // Vertices
    const vertices = new Float32Array(meshData.vertices)
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3))

    // Colors
    const colors = new Uint8Array(meshData.colors)
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3, true))

    // Triangles (indices)
    const indices = new Uint32Array(meshData.triangles)
    geometry.setIndex(new THREE.BufferAttribute(indices, 1))

    // Normals
    if (meshData.normals) {
      const normals = new Float32Array(meshData.normals)
      geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3))
    } else {
      geometry.computeVertexNormals()
    }

    // Center and scale geometry
    geometry.computeBoundingBox()
    const center = new THREE.Vector3()
    geometry.boundingBox.getCenter(center)
    geometry.translate(-center.x, -center.y, -center.z)

    const size = geometry.boundingBox.getSize(new THREE.Vector3()).length()
    const scale = 100 / size
    geometry.scale(scale, scale, scale)

    // Material
    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      flatShading: false,
    })

    const mesh = new THREE.Mesh(geometry, material)
    scene.add(mesh)

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
    directionalLight.position.set(100, 100, 100)
    scene.add(directionalLight)

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4)
    directionalLight2.position.set(-100, -100, 100)
    scene.add(directionalLight2)

    // Mouse controls
    let mouse = { x: 0, y: 0 }
    let isDragging = false

    renderer.domElement.addEventListener('mousedown', () => {
      isDragging = true
    })
    renderer.domElement.addEventListener('mouseup', () => {
      isDragging = false
    })
    renderer.domElement.addEventListener('mousemove', (e) => {
      mouse.x = (e.clientX / window.innerWidth) * 2 - 1
      mouse.y = -(e.clientY / window.innerHeight) * 2 + 1
    })

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate)

      if (isDragging) {
        mesh.rotation.x += mouse.y * 0.01
        mesh.rotation.y += mouse.x * 0.01
      } else {
        mesh.rotation.x += 0.0005
        mesh.rotation.y += 0.001
      }

      renderer.render(scene, camera)
    }

    animate()

    // Handle window resize
    const handleResize = () => {
      if (!containerRef.current) return
      const width = containerRef.current.clientWidth
      const height = containerRef.current.clientHeight
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height)
    }

    window.addEventListener('resize', handleResize)

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
      renderer.dispose()
      geometry.dispose()
      material.dispose()
      if (containerRef.current && renderer.domElement.parentNode === containerRef.current) {
        containerRef.current.removeChild(renderer.domElement)
      }
    }
  }, [meshData])

  return (
    <div className="mesh-viewer-container">
      <div className="mesh-viewer" ref={containerRef} />
      <div className="viewer-info">
        <p>📊 Vertices: {meshData?.vertex_count?.toLocaleString()}</p>
        <p>📐 Triangles: {meshData?.triangle_count?.toLocaleString()}</p>
        <p>💡 Tip: Drag to rotate, scroll to zoom</p>
      </div>
    </div>
  )
}
