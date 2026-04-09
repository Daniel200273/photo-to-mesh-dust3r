If we want this to run on average laptops without excessive work, we must avoid deep-learning-based 3D geometry generation (like DreamFusion or Instruct-NeRF2NeRF), which requires massive GPUs.
Instead, we can achieve incredible "filters" (both texture and structure) using **two lightweight tricks**:

1. **Geometric Filters (Structure):** Using Open3D to mathematically alter the mesh (e.g., Voxelization for a "Minecraft" look, or Decimation for a "Retro PS1 Low-Poly" look).
2. **Texture Filters (Style):** Running a fast, lightweight 2D style transfer on your input photos *before* they go into DUSt3R, or manipulating the vertex colors of the final mesh.

---

### **🗺️ Project Roadmap: End-to-End Stylized 3D Reconstruction**

### **Phase 1: Data Preparation & Point Cloud Generation (✅ Completed)**

*The foundation of the pipeline, turning sparse 2D images into 3D spatial data.*

- ✅ **Task 1.1:** Develop image normalization script to standardize user inputs.
- ✅ **Task 1.2:** Integrate DUSt3R for few-shot, unposed image feature extraction.
- ✅ **Task 1.3:** Implement Global Alignment to merge uncalibrated views.
- ✅ **Task 1.4:** Extract point coordinates and confidence masks.
- ✅ **Task 1.5:** Export raw unstructured Point Cloud (`.ply`).
- ✅ **Task 1.6:** Implement hardware-agnostic fallback (CUDA -> MPS -> CPU) for cross-platform compatibility.

### **Phase 2: Mesh Reconstruction & Isolation (⏳ In Progress)**

*Turning floating dots into a solid, distinct 3D object.*

- ⏳ **Task 2.1:** Statistical noise removal (cleaning floating artifacts).
- ⏳ **Task 2.2:** RANSAC Plane Segmentation (deleting the floor/table).
- ⏳ **Task 2.3:** DBSCAN Clustering (isolating the primary object from background walls).
- ⏳ **Task 2.4:** Normal estimation and Poisson Surface Reconstruction (creating the solid triangle mesh).
- 📅 **Task 2.5 (The Missing Link): Vertex Color Mapping.** *Note: Poisson reconstruction creates a blank, grey mesh. We will need to write a quick function to transfer the colors from your DUSt3R point cloud onto the new Poisson mesh vertices so your object isn't completely grey.*

### **Phase 3: 3D "Filters" and Stylization (📅 Planned)**

*The creative engine. Applying mathematical and visual filters to the 3D data.*

- 📅 **Task 3.1: The "Low-Poly / Retro" Filter (Structural)**
    - Use Open3D's `simplify_quadric_decimation` to drastically reduce the polygon count, giving the object a blocky, retro video game aesthetic.
- 📅 **Task 3.2: The "Voxel / Minecraft" Filter (Structural)**
    - Convert the mesh into a Voxel Grid using Open3D, turning smooth curves into discrete 3D cubes.
- 📅 **Task 3.3: The "Wireframe / Hologram" Filter (Visual)**
    - Extract only the edges of the mesh and color them neon green/blue to simulate a sci-fi hologram.
- 📅 **Task 3.4 (Optional): The "AI Painter" Filter (Texture)**
    - Apply a lightweight OpenCV color-quantization (toon shader) to the images *before* DUSt3R processes them, resulting in a 3D model that looks like a 2D painting or comic book.

### **Phase 4: Pipeline Automation & Export (📅 Planned)**

*Wrapping the codebase into a user-friendly tool.*

- 📅 **Task 4.1:** Consolidate the pipeline into a single Master Script where a user can run `python main.py --input data/ --filter low_poly`.
- 📅 **Task 4.2:** Standardize the final export formats (`.obj` with `.ply` vertex colors) so they can be opened in standard 3D viewers (Windows 3D Viewer, Blender).
- 📅 **Task 4.3:** Finalize `README.md` documentation for group presentation.