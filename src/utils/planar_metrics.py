import torch
import numpy as np
import cv2


def compute_reprojection_errors(pts0, pts1, H_rel):
    """
    Calculates pixel distance between pts0 projected by H_rel and pts1.
    """
    N = len(pts0)
    if N == 0:
        return np.array([])

    ones = np.ones((N, 1))
    pts0_h = np.hstack([pts0, ones])

    # Project pts0 -> pts1 using Relative Homography
    pts1_proj_h = (H_rel @ pts0_h.T).T

    # Normalize (x/z, y/z)
    # Standardized epsilon to 1e-8 for numerical stability
    pts1_proj = pts1_proj_h[:, :2] / (pts1_proj_h[:, 2:] + 1e-8)

    # Euclidean distance
    errors = np.linalg.norm(pts1 - pts1_proj, axis=1)
    return errors


def calculate_corner_error(pts0, pts1, H_rel, img_size=(256, 256), pixel_threshold=3):
    """
    Estimates homography from matches and calculates corner displacement error.
    Returns np.inf if RANSAC fails or too few points.
    """
    if len(pts0) < 4:
        return np.inf

    try:
        # Estimate Homography using RANSAC
        # Using threshold 3.0 to match the standard recall_threshold consistency
        H_pred, _ = cv2.findHomography(pts0, pts1, cv2.RANSAC, pixel_threshold)
        if H_pred is None:
            return np.inf

        w, h = img_size
        # Define 4 image corners
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)

        # Transform corners using GT and Predicted Homographies
        corners_gt = cv2.perspectiveTransform(corners, H_rel)
        corners_pred = cv2.perspectiveTransform(corners, H_pred)

        # Average distance between predicted and GT corners
        return np.mean(np.linalg.norm(corners_gt - corners_pred, axis=2))
    except Exception:
        return np.inf


def compute_planar_metrics(data, thresholds=[1, 2, 3], recall_threshold=3, img_size=(256, 256)):
    """
    Calculates full suite of geometric metrics for Planar/Satellite scenes.

    Metrics:
    1. Pose_Success_10px (Pose Recall) -> % of images with Corner Error < 10px (The "Executive Summary" metric)
    2. MMA (Mean Matching Accuracy)    -> Precision @ Threshold (Standard for HPatches)
    3. Num_Inliers                     -> Count of correct matches (Robustness)
    4. Feature_Recall                  -> % of GT matches recovered
    5. MRE (Mean Reprojection Error)   -> Pixel Precision of raw matches

    Expected keys in 'data' dictionary:
    - 'mkpts0_f', 'mkpts1_f': Matched keypoints (N, 2)
    - 'm_bids': Batch indices for matches (N,)
    - 'H0', 'H1': Absolute homographies for view 0 and 1 (B, 3, 3)
    - 'conf_matrix_gt': Ground truth supervision mask (B, H_c, W_c) [Optional]
    """

    # --- 1. Extract Data & Normalize Inputs ---
    all_pts0 = data['mkpts0_f'].detach().cpu().numpy()
    all_pts1 = data['mkpts1_f'].detach().cpu().numpy()
    all_bids = data['m_bids'].detach().cpu().numpy()

    H0_batch = data['H0'].detach().cpu().numpy()
    H1_batch = data['H1'].detach().cpu().numpy()

    batch_size = H0_batch.shape[0]

    # Extract GT Match Counts (Required for Feature Recall)
    gt_matches_per_batch = None
    if 'conf_matrix_gt' in data:
        gt_matches_per_batch = data['conf_matrix_gt'].sum(dim=(1, 2)).detach().cpu().numpy()
    elif 'num_gt_matches' in data:
        gt_matches_per_batch = data['num_gt_matches'].detach().cpu().numpy()

    # Initialize per-pixel errors array
    full_reproj_errors = np.full(len(all_pts0), np.nan, dtype=np.float32)

    # Storage for batch results
    metrics = {f"MMA@{t}": [] for t in thresholds}
    metrics["MRE"] = []
    metrics["Corner_Error"] = []
    metrics["Num_Matches"] = []
    metrics["Num_Inliers"] = []
    metrics["Feature_Recall"] = []

    per_batch_errors = []

    # --- 2. Iterate Batch ---
    for b in range(batch_size):
        mask = (all_bids == b)
        pts0 = all_pts0[mask]
        pts1 = all_pts1[mask]

        # Get GT count for Recall
        n_gt = gt_matches_per_batch[b] if gt_matches_per_batch is not None else 0

        # Handle failure cases (Insufficient matches)
        if len(pts0) < 4:
            for t in thresholds:
                metrics[f"MMA@{t}"].append(0.0)
            metrics["MRE"].append(np.nan)
            metrics["Corner_Error"].append(np.inf)  # Inf = Failure for Corner Error
            metrics["Num_Matches"].append(len(pts0))
            metrics["Num_Inliers"].append(0)  # Zero inliers on failure
            metrics["Feature_Recall"].append(0.0 if n_gt > 0 else np.nan)
            per_batch_errors.append(None)
            continue

        H0 = H0_batch[b]
        H1 = H1_batch[b]

        # Robust Relative Homography Calculation
        try:
            H0_inv = np.linalg.inv(H0)
            H_rel = H1 @ H0_inv
        except np.linalg.LinAlgError:
            for t in thresholds:
                metrics[f"MMA@{t}"].append(np.nan)
            metrics["MRE"].append(np.nan)
            metrics["Corner_Error"].append(np.inf)
            metrics["Num_Matches"].append(len(pts0))
            metrics["Num_Inliers"].append(0)
            metrics["Feature_Recall"].append(np.nan)
            per_batch_errors.append(None)
            continue

        # --- 3. Compute Metrics ---
        reproj_errors = compute_reprojection_errors(pts0, pts1, H_rel)

        # Store for visualization
        full_reproj_errors[mask] = reproj_errors
        per_batch_errors.append(reproj_errors)

        # MRE
        metrics["MRE"].append(np.mean(reproj_errors))

        # MMA
        for t in thresholds:
            metrics[f"MMA@{t}"].append(np.mean(reproj_errors < t))

        # Corner Error
        metrics["Corner_Error"].append(calculate_corner_error(pts0, pts1, H_rel, img_size))

        # Num Matches
        metrics["Num_Matches"].append(len(pts0))

        # NEW: Num Inliers (Matches < 3px error)
        n_inliers = np.sum(reproj_errors < recall_threshold)
        metrics["Num_Inliers"].append(n_inliers)

        # Feature Recall - Use np.nan for undefined cases (no GT)
        n_correct = np.sum(reproj_errors < recall_threshold)
        if n_gt > 0:
            batch_recall = n_correct / n_gt
        else:
            batch_recall = np.nan
        metrics["Feature_Recall"].append(batch_recall)

    # --- 4. Final Aggregation ---
    device = data['m_bids'].device
    data['planar_reproj_errs'] = torch.from_numpy(full_reproj_errors).to(device)

    final_metrics = {
        "MRE": np.nanmean(metrics["MRE"]),
        "Corner_Error_Mean": np.nanmean(metrics["Corner_Error"]),  # Renamed
        "Num_Matches": np.mean(metrics["Num_Matches"]),
        "Num_Inliers": np.mean(metrics["Num_Inliers"]),  # <--- NEW
        "Feature_Recall": np.nanmean(metrics["Feature_Recall"])
    }

    # --- NEW: Calculate Homography Success Rate (Pose Recall) ---
    # This corresponds to "Recall" in many Homography benchmarks (e.g., HPatches)
    # Success = Corner Error < 10 pixels
    ce_arr = np.array(metrics["Corner_Error"])
    # Replace Inf/NaN with huge value so they count as failures
    ce_arr[~np.isfinite(ce_arr)] = 1e9
    final_metrics["Pose_Success_10px"] = np.mean(ce_arr < 10.0)

    # Aggregate MMA
    for t in thresholds:
        final_metrics[f"MMA@{t}"] = np.nanmean(metrics[f"MMA@{t}"])

    return final_metrics


def compute_planar_metrics_raw(data, thresholds=[1, 2, 3], recall_threshold=3, img_size=(256, 256)):
    """
    Calculates metrics for a batch but returns LISTS of values (per-sample)
    instead of aggregated means.
    """
    # --- 1. Extract Data ---
    all_pts0 = data['mkpts0_f'].detach().cpu().numpy()
    all_pts1 = data['mkpts1_f'].detach().cpu().numpy()
    all_bids = data['m_bids'].detach().cpu().numpy()
    H0_batch = data['H0'].detach().cpu().numpy()
    H1_batch = data['H1'].detach().cpu().numpy()
    batch_size = H0_batch.shape[0]

    # Ground Truth handling
    gt_matches_per_batch = None
    if 'conf_matrix_gt' in data:
        gt_matches_per_batch = data['conf_matrix_gt'].sum(dim=(1, 2)).detach().cpu().numpy()
    elif 'num_gt_matches' in data:
        gt_matches_per_batch = data['num_gt_matches'].detach().cpu().numpy()

    # Per-pixel errors for visualization
    full_reproj_errors = np.full(len(all_pts0), np.nan, dtype=np.float32)

    # Initialize Lists
    metrics = {f"MMA@{t}": [] for t in thresholds}
    metrics["MRE"] = []
    metrics["Corner_Error"] = []
    metrics["Num_Matches"] = []
    metrics["Num_Inliers"] = []
    metrics["Feature_Recall"] = []
    metrics["Pose_Success_10px"] = []  # Explicit list for success (1/0)

    # --- 2. Iterate Batch ---
    for b in range(batch_size):
        mask = (all_bids == b)
        pts0 = all_pts0[mask]
        pts1 = all_pts1[mask]
        n_gt = gt_matches_per_batch[b] if gt_matches_per_batch is not None else 0

        # --- Failure Case ---
        if len(pts0) < 4:
            for t in thresholds: metrics[f"MMA@{t}"].append(0.0)
            metrics["MRE"].append(None)  # Use None for JSON safety
            metrics["Corner_Error"].append(None)
            metrics["Num_Matches"].append(len(pts0))
            metrics["Num_Inliers"].append(0)
            metrics["Feature_Recall"].append(0.0 if n_gt > 0 else None)
            metrics["Pose_Success_10px"].append(0)
            continue

        H0 = H0_batch[b]
        H1 = H1_batch[b]

        try:
            H0_inv = np.linalg.inv(H0)
            H_rel = H1 @ H0_inv
        except np.linalg.LinAlgError:
            for t in thresholds: metrics[f"MMA@{t}"].append(None)
            metrics["MRE"].append(None)
            metrics["Corner_Error"].append(None)
            metrics["Num_Matches"].append(len(pts0))
            metrics["Num_Inliers"].append(0)
            metrics["Feature_Recall"].append(None)
            metrics["Pose_Success_10px"].append(0)
            continue

        # --- Metrics Calculation ---
        reproj_errors = compute_reprojection_errors(pts0, pts1, H_rel)
        full_reproj_errors[mask] = reproj_errors

        metrics["MRE"].append(float(np.mean(reproj_errors)))

        for t in thresholds:
            metrics[f"MMA@{t}"].append(float(np.mean(reproj_errors < t)))

        # Corner Error & Success
        c_err = calculate_corner_error(pts0, pts1, H_rel, img_size, pixel_threshold=3)

        if np.isinf(c_err):
            metrics["Corner_Error"].append(None)
            metrics["Pose_Success_10px"].append(0)
        else:
            metrics["Corner_Error"].append(float(c_err))
            metrics["Pose_Success_10px"].append(1 if c_err < 10.0 else 0)

        metrics["Num_Matches"].append(int(len(pts0)))

        n_inliers = np.sum(reproj_errors < recall_threshold)
        metrics["Num_Inliers"].append(int(n_inliers))

        if n_gt > 0:
            metrics["Feature_Recall"].append(float(n_inliers / n_gt))
        else:
            metrics["Feature_Recall"].append(None)

    # --- 3. Return Raw Lists ---
    device = data['m_bids'].device
    data['planar_reproj_errs'] = torch.from_numpy(full_reproj_errors).to(device)

    # Return the dictionary of LISTS directly.
    return metrics