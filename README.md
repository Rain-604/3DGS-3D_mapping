# AI-GS Infill: Unobserved Angle Hallucination for FastGS
**Hackathon Submission 2026**

<p align="center">
    <i>An AI-driven point cloud generator plugin built on top of the FastGS framework to resolve missing geometry and floating artifacts in sparse drone datasets.</i>
</p>

## 🚀 Project Overview

Standard 3D Gaussian Splatting relies purely on observed pixels, leaving the unobserved backsides of structures empty or plagued by floating artifacts. The **AI-GS Infill** plugin solves this by intercepting the COLMAP initialization phase, applying monocular depth estimation (MiDaS), and back-projecting synthesized points into the unobserved regions before optimization begins.

This project is built upon **FastGS**, a general acceleration framework that supercharges 3D Gaussian Splatting training. By leveraging FastGS, our pipeline inherits blazing fast training speeds—achieving state-of-the-art results within 100 seconds.

### ✨ Key Features of the Hackathon Build
- **Intelligent Point Cloud Infill:** Uses MiDaS to predict depth from 2D images, injecting up to 450,000 synthesized 3D points into unseen areas.
- **Automated Pipeline (`automation.py`):** Handles environment setup, COLMAP conversion, and training seamlessly.
- **Optimized Rendering (`render.py`):** Generates novel view sequences that can be easily stitched into high-quality fly-through MP4s.
- **FastGS Foundation:** Inherits a memory-efficient architecture with a low GPU memory footprint and strict control over Gaussian efficiency.

## 📦 Environment Setup

To run this plugin, you must satisfy the base FastGS hardware requirements, which include a CUDA-ready GPU with Compute Capability 7.0+ and 24 GB VRAM (an NVIDIA RTX4090 is recommended). 

1. **Clone and Activate:**
```bash
conda env create --file environment.yml
conda activate fastgs
```

2. **Install Custom Dependencies:**
```bash
pip install timm websockets
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization_fastgs
pip install --no-build-isolation ./submodules/simple-knn
```

## 🚀 Running the AI-GS Infill Pipeline

1. **Format the Dataset:** Ensure your dataset contains the required COLMAP structures in `Input/scene_DJI/`.
2. **Train the Model:** The modified FastGS pipeline will automatically load the AI weights, merge the point clouds, and train the Gaussians:
```bash
python train.py -s ../Input/scene_DJI -m ../Output/"Floder_name"
```
3. **Render the Output:** Generate the novel view frames:
```bash
python render.py -m ../Output/"Floder_name"
```
4. **Stitch the Video:** Combine the rendered `.png` frames into a final demo video:
```bash
ffmpeg -framerate 30 -pattern_type glob -i '../Output/DJI_model/train/ours_30000/renders/*.png' -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p ../Output/DJI_model/hackathon_demo.mp4
```

## 🎯 Future Work: 3D Generative Models
While monocular depth handles minor occlusions, true view-extrapolation requires hallucinating completely unseen textures. Future iterations of this pipeline will integrate diffusion-based generative AI (like Stable Video 3D) to synthesize novel 2D images of unobserved building facades before routing them through COLMAP and FastGS.

## 🙏 Acknowledgements
This hackathon project is built upon [FastGS](https://github.com/fastgs/FastGS), which itself builds upon 3DGS, Taming-3DGS, Speedy-Splat, and Abs-GS. We extend our gratitude to the authors of these frameworks for their outstanding contributions to the 3D reconstruction community. All original MIT licensing and copyright notices for FastGS are preserved in the root directory.
