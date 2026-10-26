import os
import subprocess
import shutil

BASE_INPUT_DIR = "../Input"
BASE_OUTPUT_DIR = "../Output"

print("============================================================")
print("🚀 STARTING BATCH AI-GS INFILL HACKATHON PIPELINE")
print("============================================================")

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
scene_folders = [f for f in os.listdir(BASE_INPUT_DIR) if os.path.isdir(os.path.join(BASE_INPUT_DIR, f))]

if not scene_folders:
    print(f"No scene folders found in {BASE_INPUT_DIR}")
    exit(0)

for scene_name in scene_folders:
    input_dir = os.path.join(BASE_INPUT_DIR, scene_name)
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"{scene_name}_model")
    final_video_path = os.path.join(output_dir, "hackathon_demo.mp4")
    final_ply_path = os.path.join(output_dir, f"{scene_name}_final_cloud.ply")

    print(f"\n" + "="*60)
    print(f"🔎 Checking Scene: {scene_name}")
    print("="*60)

    if os.path.exists(final_video_path) and os.path.exists(final_ply_path):
        print(f"✅ Video and point cloud already exist for {scene_name}. Skipping.")
        continue

    print(f"⚙️ Processing scene: {scene_name}...")

    # STEP 1: COLMAP Conversion
    print("\n--- STEP 1: Generating Initial Point Cloud (COLMAP) ---")
    if not os.path.exists(os.path.join(input_dir, "sparse")):
        subprocess.run(["python", "convert.py", "-s", input_dir], check=True)
    else:
        print("Sparse folder already exists. Skipping COLMAP conversion.")

    # STEP 2: Train Model
    print("\n--- STEP 2: Training FastGS with AI-GS Infill Plugin ---")
    subprocess.run(["python", "train.py", "-s", input_dir, "-m", output_dir], check=True)

    # STEP 3: Render Video
    print("\n--- STEP 3: Rendering Final Fly-through Video ---")
    subprocess.run(["python", "render_demo.py", "-m", output_dir], check=True)

    # STEP 4: Extract Final Point Cloud
    print("\n--- STEP 4: Extracting Final Point Cloud ---")
    pc_dir = os.path.join(output_dir, "point_cloud")
    if os.path.exists(pc_dir):
        # Find the highest iteration folder (usually iteration_30000)
        iterations = [int(d.split('_')[-1]) for d in os.listdir(pc_dir) if d.startswith("iteration_")]
        if iterations:
            max_iter = max(iterations)
            src_ply = os.path.join(pc_dir, f"iteration_{max_iter}", "point_cloud.ply")
            
            if os.path.exists(src_ply):
                shutil.copy2(src_ply, final_ply_path)
                print(f"Extracted final point cloud to: {final_ply_path}")

    print(f"\n✅ Scene '{scene_name}' complete!")

print("\n" + "="*60)
print("🎉 ALL SCENES PROCESSED SUCCESSFULLY!")
print("="*60)