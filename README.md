# Few-Shot 3D Reconstruction Pipeline

This pipeline uses DUSt3R (Dense and Unconstrained Stereo 3D Reconstruction) to generate 3D point clouds from sparse, uncalibrated image sets (e.g., 10-15 images) without relying on traditional feature matching like COLMAP. It is designed to run on Windows, Mac, and Linux, and will automatically scale down to use the CPU if you only have integrated graphics.

## 🛠️ Setup Instructions

### 1. Set up your Python Environment

First create the environment, for example, if you use conda, like this:

```bash
conda create --name cv-env python=3.10 -y
conda activate cv-env
```

and install the libraries:

```bash
pip install -r requirements.txt
```

If not present in the root directory, run the following lines:

```bash
# Clone the dust3r logic
git clone --recursive [https://github.com/naver/dust3r.git](https://github.com/naver/dust3r.git)

# Download the model weights (requires curl)
mkdir -p dust3r/checkpoints
curl -L -o dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth [https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth)
```

### 2. Run the pipeline

YOu can load images of the object/scene you want to reconstruct into `data/raw_data/`>
Then you just need to run the pipeline. From the root:

```bash
python src/pipeline.py
```

Once finished without errors, you can visualize the result:

```bash
python src/visualize.py
```
