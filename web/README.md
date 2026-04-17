# FewShot-NeRF Web Application

Complete web interface for 3D reconstruction with interactive effect selection.

## 📁 Structure

```
web/
├── backend/           # FastAPI server
│   ├── app.py        # Main backend application
│   ├── requirements.txt
│   └── ...uploads & results stored here during runtime
├── frontend/         # React + Vite frontend
│   ├── src/
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
├── uploads/          # Temporary image uploads
└── results/          # Processed 3D models
```

## 🚀 Quick Start

### Option 1: Automated (Recommended)
```bash
chmod +x start_web.sh
./start_web.sh
```
This will:
- Activate your `.venv`
- Install Python + Node.js dependencies (cached, won't re-download)
- Start backend on `http://127.0.0.1:8000`
- Start frontend on `http://localhost:3000`

### Option 2: Manual

**Terminal 1 - Backend:**
```bash
source .venv/bin/activate
cd web/backend
pip install -r requirements.txt
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd web/frontend
npm install  # Only needed once
npm run dev
```

Then open: `http://localhost:3000`

## 🔄 Workflow

1. **Upload images** (3+ photos of your object from different angles)
2. **Select effect** (low_poly, voxel, soft_voxel, or hologram)
3. **Generate** - Backend runs:
   - DUSt3R reconstruction (uses cached model from `dust3r/checkpoints/`)
   - Stylization effect from your `src/stylize.py`
4. **Preview** - Interactive 3D viewer powered by Three.js
5. **Download** - Save your mesh as PLY file

## ⚙️ Key Features

- **No dependency re-downloads**: Model cached in `dust3r/checkpoints/`
- **Organized folder structure**: Web code isolated in `web/` directory
- **Reuses existing code**: Directly imports your `pipeline.py` and `stylize.py`
- **Progress feedback**: Real-time processing status
- **Interactive viewer**: Drag to rotate, built-in model statistics

## 🎨 Available Effects

| Effect | Description |
|--------|-------------|
| **Low Poly** | Clean geometric style with smooth shading |
| **Voxel** | Minecraft-style blocky voxel grid |
| **Soft Voxel** | Smooth voxel representation with sphere nodes |
| **Hologram** | Neon wireframe with cyan edges |

## 🛠️ Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill the process
kill -9 <PID>
```

### Frontend won't load
```bash
# Clear Vite cache
rm -rf web/frontend/.vite
npm run dev
```

### HuggingFace issues (shouldn't happen now)
The setup avoids HuggingFace downloads by:
- Using pre-downloaded model in `dust3r/checkpoints/`
- Setting `HF_HUB_OFFLINE=1` in environment (optional)

### Out of memory during processing
- Use smaller images (< 2MB each)
- Reduce number of images
- Process on GPU machine if available

## 📊 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Check backend status |
| `/api/effects` | GET | List available effects |
| `/api/process` | POST | Upload images & generate model |
| `/api/download/{job_id}` | GET | Download result PLY file |
| `/api/jobs/{job_id}` | GET | Get job metadata |

## 🔧 Development

### Modifying effects
Edit `src/stylize.py`, changes auto-load on backend restart

### Adding new effects
1. Create function in `src/stylize.py`
2. Import in `web/backend/app.py`
3. Add to `EFFECTS` dict and `apply_effect()` switch

### Updating UI
React frontend hot-reloads during `npm run dev`

## 📝 Notes

- Uploads are stored in `web/uploads/` (temporary)
- Results saved in `web/results/` with job ID
- Each job creates a separate directory
- Old jobs can be manually cleaned up
