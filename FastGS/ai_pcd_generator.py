# ai_pcd_generator.py
import torch
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray

class AIDepthEstimator:
    def __init__(self, device='cuda'):
        self.device = device
        print("[AI-GS Infill] Loading AI Monocular Depth Model (MiDaS)...")
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self.device)
            self.model.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform
            print("[AI-GS Infill] AI Model loaded successfully!")
        except Exception as e:
            print(f"[AI-GS Infill] Warning: Failed to load online weights ({e}).")
            self.model = None

    @torch.no_grad()
    def predict_depth(self, img_rgb: np.ndarray) -> np.ndarray:
        if self.model is None:
            h, w, _ = img_rgb.shape
            return np.linspace(1.0, 5.0, h)[:, None].repeat(w, axis=1)

        input_batch = self.transform(img_rgb).to(self.device)
        prediction = self.model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        depth = prediction.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = 1.0 + depth * 4.0 
        return depth

class AIPointCloudGenerator:
    def __init__(self, device='cuda'):
        self.device = device
        self.depth_estimator = AIDepthEstimator(device=device)

    def backproject_image_to_3d(self, image_np: np.ndarray, R: np.ndarray, T: np.ndarray, 
                                FovX: float, FovY: float, sample_stride: int = 4) -> tuple:
        h, w, _ = image_np.shape
        depth = self.depth_estimator.predict_depth(image_np)

        fx = w / (2.0 * np.tan(FovX / 2.0))
        fy = h / (2.0 * np.tan(FovY / 2.0))
        cx, cy = w / 2.0, h / 2.0

        u_coords = np.arange(0, w, sample_stride)
        v_coords = np.arange(0, h, sample_stride)
        u, v = np.meshgrid(u_coords, v_coords)
        
        z = depth[v, u]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        pts_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = (image_np[v, u] / 255.0).reshape(-1, 3)

        pts_world = (pts_cam - T) @ R
        return pts_world, colors

    def filter_outliers(self, points: np.ndarray, colors: np.ndarray):
        if len(points) == 0:
            return points, colors
        mean_dist = np.mean(np.linalg.norm(points, axis=1))
        mask = np.linalg.norm(points, axis=1) < (mean_dist * 2.5)
        return points[mask], colors[mask]

def augment_scene_with_ai(cameras, original_pcd: BasicPointCloud, sample_rate: int = 4) -> BasicPointCloud:
    print("\n" + "="*60)
    print("🚀 [AI-GS Infill Plugin] Synthesizing AI Point Clouds for Unseen Views")
    print("="*60)

    generator = AIPointCloudGenerator()
    ai_points_list = []
    ai_colors_list = []

    step = max(1, len(cameras) // 5) 

    for idx in range(0, len(cameras), step):
        cam = cameras[idx]
        if hasattr(cam, 'original_image'):
            img = cam.original_image.permute(1, 2, 0).cpu().numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
        else:
            continue

        pts, cols = generator.backproject_image_to_3d(img, cam.R, cam.T, cam.FoVx, cam.FoVy, sample_rate)
        ai_points_list.append(pts)
        ai_colors_list.append(cols)

    if not ai_points_list:
        return original_pcd

    all_ai_pts = np.concatenate(ai_points_list, axis=0)
    all_ai_cols = np.concatenate(ai_colors_list, axis=0)
    all_ai_pts, all_ai_cols = generator.filter_outliers(all_ai_pts, all_ai_cols)

    if original_pcd is not None and len(original_pcd.points) > 0:
        merged_pts = np.concatenate([original_pcd.points, all_ai_pts], axis=0)
        merged_cols = np.concatenate([original_pcd.colors, all_ai_cols], axis=0)
    else:
        merged_pts = all_ai_pts
        merged_cols = all_ai_cols

    normals = np.zeros_like(merged_pts)
    print(f"[AI-GS Infill] Injected {len(all_ai_pts)} AI-generated points.")
    print(f"[AI-GS Infill] Total Point Cloud size: {len(merged_pts)} points.")
    print("="*60 + "\n")

    return BasicPointCloud(points=merged_pts, colors=merged_cols, normals=normals)