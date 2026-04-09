"""Generate comparison image pairs for the react-compare-slider teaser."""
import numpy as np
import os

def save_png(arr, path):
    """Save 2D array as PNG."""
    from PIL import Image
    if arr.ndim == 2:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10) * 255
        img = Image.fromarray(arr.astype(np.uint8), mode='L')
    elif arr.ndim == 3:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10) * 255
        img = Image.fromarray(arr.astype(np.uint8))
    img.save(path)
    print(f"  Saved {path}")

out_dir = "/data/yjh/agent-imaging-website/public/images/compare"
os.makedirs(out_dir, exist_ok=True)

# --- Task 44: pyDHM (already has PNG pairs) ---
import shutil
src = "/data/yjh/website_assets/Task_44_pyDHM-master"
shutil.copy(f"{src}/input_hologram.png", f"{out_dir}/task44_input.png")
shutil.copy(f"{src}/recon_phase.png", f"{out_dir}/task44_recon.png")
shutil.copy(f"{src}/gt_phase.png", f"{out_dir}/task44_gt.png")
print("Task 44: pyDHM done")

# --- Task 01: SIM deconvolution ---
try:
    from PIL import Image
    src1 = "/data/yjh/website_assets/Task_01_sim"
    inp = np.array(Image.open(f"{src1}/input.tif")).astype(float)
    gt = np.array(Image.open(f"{src1}/gt_output.tif")).astype(float)
    rec = np.array(Image.open(f"{src1}/recon_output.tif")).astype(float)
    save_png(inp, f"{out_dir}/task01_input.png")
    save_png(gt, f"{out_dir}/task01_gt.png")
    save_png(rec, f"{out_dir}/task01_recon.png")
    print("Task 01: SIM done")
except Exception as e:
    print(f"Task 01 failed: {e}")

# --- Task 75: TIGRE CT ---
try:
    src75 = "/data/yjh/website_assets/Task_75_TIGRE"
    data = np.load(f"{src75}/gt_output.npy")
    if data.ndim == 3:
        mid = data.shape[0] // 2
        gt_slice = data[mid]
    else:
        gt_slice = data
    save_png(gt_slice, f"{out_dir}/task75_gt.png")
    
    rec_data = np.load(f"{src75}/recon_output.npy")
    if rec_data.ndim == 3:
        mid = rec_data.shape[0] // 2
        rec_slice = rec_data[mid]
    else:
        rec_slice = rec_data
    save_png(rec_slice, f"{out_dir}/task75_recon.png")
    print("Task 75: TIGRE CT done")
except Exception as e:
    print(f"Task 75 failed: {e}")

# --- Task 76: CDI ---
try:
    src76 = "/data/yjh/website_assets/Task_76_CDI"
    gt76 = np.load(f"{src76}/gt_output.npy")
    rec76 = np.load(f"{src76}/recon_output.npy")
    # CDI: take amplitude
    if np.iscomplexobj(gt76):
        gt76 = np.abs(gt76)
    if np.iscomplexobj(rec76):
        rec76 = np.abs(rec76)
    if gt76.ndim == 3:
        gt76 = gt76[gt76.shape[0]//2]
    if rec76.ndim == 3:
        rec76 = rec76[rec76.shape[0]//2]
    save_png(gt76, f"{out_dir}/task76_gt.png")
    save_png(rec76, f"{out_dir}/task76_recon.png")
    print("Task 76: CDI done")
except Exception as e:
    print(f"Task 76 failed: {e}")

print("\nAll compare images generated!")
print("Files:", os.listdir(out_dir))
