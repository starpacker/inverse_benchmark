import sys
import json
import numpy as np

def compute_psnr(gt, pred, data_range=1.0):
    mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
    if mse < 1e-15:
        return float('inf')
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)

def compute_ssim_simple(gt, pred, data_range=1.0):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    mu_gt = np.mean(gt)
    mu_pred = np.mean(pred)
    sigma_gt_sq = np.var(gt)
    sigma_pred_sq = np.var(pred)
    sigma_cross = np.mean((gt - mu_gt) * (pred - mu_pred))
    num = (2 * mu_gt * mu_pred + C1) * (2 * sigma_cross + C2)
    den = (mu_gt ** 2 + mu_pred ** 2 + C1) * (sigma_gt_sq + sigma_pred_sq + C2)
    return float(num / den)

def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python eval_script.py prediction.npy"}))
        sys.exit(1)

    pred_path = sys.argv[1]
    gt_path = "dataset/gt_output.npy"

    try:
        gt = np.load(gt_path, allow_pickle=False)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load ground truth: {str(e)}"}))
        sys.exit(1)

    try:
        pred = np.load(pred_path, allow_pickle=False)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load prediction: {str(e)}"}))
        sys.exit(1)

    # Handle NaN/Inf
    pred = np.where(np.isfinite(pred), pred, 0.0)
    gt = np.where(np.isfinite(gt), gt, 0.0)

    # Shape matching: squeeze singleton dimensions
    def squeeze_to_match(pred, gt):
        # Try squeezing leading/trailing singletons
        while pred.ndim > gt.ndim:
            if pred.shape[0] == 1:
                pred = pred[0]
            elif pred.shape[-1] == 1:
                pred = pred[..., 0]
            else:
                break
        while gt.ndim > pred.ndim:
            if gt.shape[0] == 1:
                gt = gt[0]
            elif gt.shape[-1] == 1:
                gt = gt[..., 0]
            else:
                break
        return pred, gt

    pred, gt = squeeze_to_match(pred, gt)

    if pred.shape != gt.shape:
        print(json.dumps({
            "error": "Shape mismatch",
            "pred_shape": list(pred.shape),
            "gt_shape": list(gt.shape)
        }))
        sys.exit(1)

    # Clip to [0, 1] range
    pred = np.clip(pred.astype(np.float64), 0.0, 1.0)
    gt = gt.astype(np.float64)
    # Also clip gt just in case
    gt = np.clip(gt, 0.0, 1.0)

    data_range = 1.0

    # Compute PSNR
    psnr_val = compute_psnr(gt, pred, data_range)

    # Compute SSIM
    ssim_val = None
    try:
        from skimage.metrics import structural_similarity
        # For multi-channel data (e.g., shape (2, 600, 600)), handle channel_axis
        if gt.ndim == 3:
            # Determine appropriate win_size
            min_spatial = min(gt.shape[1], gt.shape[2])
            if min_spatial >= 7:
                win_size = 7
            else:
                win_size = max(3, min_spatial if min_spatial % 2 == 1 else min_spatial - 1)

            # Check if first dimension is channels (small) vs spatial
            if gt.shape[0] <= 4:
                # Treat as channel_axis=0
                ssim_val = structural_similarity(
                    gt, pred, data_range=data_range,
                    channel_axis=0, win_size=win_size
                )
            else:
                # Treat as 3D volume, compute per-slice and average
                ssim_vals = []
                for i in range(gt.shape[0]):
                    s = structural_similarity(
                        gt[i], pred[i], data_range=data_range, win_size=win_size
                    )
                    ssim_vals.append(s)
                ssim_val = float(np.mean(ssim_vals))
        elif gt.ndim == 2:
            min_dim = min(gt.shape)
            if min_dim >= 7:
                win_size = 7
            else:
                win_size = max(3, min_dim if min_dim % 2 == 1 else min_dim - 1)
            ssim_val = structural_similarity(
                gt, pred, data_range=data_range, win_size=win_size
            )
        else:
            # Fallback for other dims
            ssim_val = compute_ssim_simple(gt, pred, data_range)
    except ImportError:
        # Fallback: compute simplified SSIM
        if gt.ndim == 3:
            ssim_vals = []
            for i in range(gt.shape[0]):
                ssim_vals.append(compute_ssim_simple(gt[i], pred[i], data_range))
            ssim_val = float(np.mean(ssim_vals))
        else:
            ssim_val = compute_ssim_simple(gt, pred, data_range)
    except Exception as e:
        ssim_val = compute_ssim_simple(gt, pred, data_range)

    # Handle inf PSNR
    if not np.isfinite(psnr_val):
        psnr_val = 100.0  # Cap at very high value

    result = {
        "psnr": round(float(psnr_val), 4),
        "ssim": round(float(ssim_val), 6)
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main()