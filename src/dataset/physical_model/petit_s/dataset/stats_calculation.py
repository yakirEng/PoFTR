import numpy as np
import tarfile
import io
from tqdm import tqdm
from pathlib import Path
import os
import json


def update_welford_stats(existing_aggregate, new_values):
    """
    Updates running mean and variance using Welford's online algorithm.
    This is memory-efficient and numerically stable.

    Args:
        existing_aggregate (tuple): (count, mean, M2) for existing data.
        new_values (np.ndarray): A flat numpy array of new data points.

    Returns:
        tuple: Updated (count, mean, M2).
    """
    (count, mean, M2) = existing_aggregate

    # Ensure new_values is iterable
    if not hasattr(new_values, '__iter__'):
        new_values = [new_values]

    for x in np.nditer(new_values):
        x_float = float(x)  # Ensure we're working with floats
        count += 1
        delta = x_float - mean
        mean += delta / count
        delta2 = x_float - mean
        M2 += delta * delta2

    return (count, mean, M2)


def finalize_welford_stats(existing_aggregate):
    """
    Calculates final mean and standard deviation from Welford's aggregates.

    Args:
        existing_aggregate (tuple): (count, mean, M2).

    Returns:
        tuple: (final_mean, final_std). Returns (0, 0) if count is 0.
    """
    (count, mean, M2) = existing_aggregate
    if count < 2:
        # Cannot compute variance with < 2 points
        # Return mean, but 0.0 for std dev
        return (mean, 0.0)

    final_mean = mean
    # Use population standard deviation (N), not sample (N-1)
    variance = M2 / count
    final_std = np.sqrt(variance)

    return (final_mean, final_std)


def process_tar_file(tar_path, agg_mono, agg_t_fpa, agg_pan_temp):
    """
    Processes a single .tar file and updates the running aggregate statistics
    for our three data keys: mono_image, t_fpa, and pan_temp_map.
    """
    try:
        with tarfile.open(tar_path, 'r') as tar:
            npz_files = [m for m in tar.getmembers() if m.name.endswith('.npz')]

            if not npz_files:
                print(f"Warning: No .npz files found in {tar_path}")
                return agg_mono, agg_t_fpa, agg_pan_temp

            with tqdm(total=len(npz_files), desc=f"Reading {os.path.basename(tar_path)}", leave=False) as pbar:
                for member in npz_files:
                    file_buffer = tar.extractfile(member).read()

                    with io.BytesIO(file_buffer) as f:
                        try:
                            data = np.load(f)

                            # --- Update stats for our 3 keys ---

                            # 1. mono_image (pan image)
                            if 'mono_image' in data:
                                # .ravel() flattens the image array
                                agg_mono = update_welford_stats(agg_mono, data['mono_image'].ravel())

                            # 2. t_fpa (scalar value)
                            if 't_fpa' in data:
                                # t_fpa is a scalar, pass it as a single-item array
                                agg_t_fpa = update_welford_stats(agg_t_fpa, [data['t_fpa']])

                            # 3. pan_temp_map (ground truth)
                            if 'pan_temp_map' in data:
                                agg_pan_temp = update_welford_stats(agg_pan_temp, data['pan_temp_map'].ravel())

                        except Exception as e:
                            print(f"\nWarning: Could not process file {member.name} in {tar_path}. Error: {e}")
                            continue

                    # --- Update the tqdm progress bar description ---
                    (count_m, mean_m, M2_m) = agg_mono
                    (count_t, mean_t, M2_t) = agg_t_fpa
                    (count_p, mean_p, M2_p) = agg_pan_temp

                    std_m = np.sqrt(M2_m / count_m) if count_m > 1 else 0.0
                    std_t = np.sqrt(M2_t / count_t) if count_t > 1 else 0.0
                    std_p = np.sqrt(M2_p / count_p) if count_p > 1 else 0.0

                    pbar.set_description(
                        f"Reading {os.path.basename(tar_path)} | "
                        f"Pan(μ={mean_m:.2f}, σ={std_m:.2f}) | "
                        f"FPA(μ={mean_t:.2f}, σ={std_t:.2f}) | "
                        f"GT(μ={mean_p:.2f}, σ={std_p:.2f})"
                    )
                    pbar.update(1)

    except Exception as e:
        print(f"\nError: Could not open or read tar file {tar_path}. Error: {e}")

    return agg_mono, agg_t_fpa, agg_pan_temp


def calculate_kfold_stats():
    """
    Main function to find and process all .tar files across all k-folds.
    Calculates stats ONLY on the 'train' split for each fold.
    """

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # This should be the *base* directory containing the fold folders
    BASE_DATA_PATH = Path(r"F:\yakirs_thesis\thesis_code\data")

    # These are the sub-paths from your PetitsDs class
    BASE_MODEL = "petits"  # Or whatever config['proj']['base_model'] was
    SEED = 42  # Or whatever config['data']['master_seed'] was
    WL_SETTING = "9um"  # Or whatever config["data"]["wl"] was
    # -----------------------------------------------------------------

    K_FOLDS = 5
    DATASET_TYPE = "train"  # We ONLY calculate stats on the training set

    # This is the root where the 'webdataset' folder lives
    WDS_ROOT = BASE_DATA_PATH / BASE_MODEL / "webdataset" / f"seed_{SEED}"

    # This is where the final stats.json will be saved
    OUTPUT_FILE_PATH = WDS_ROOT / f"stats_{WL_SETTING}.json"

    print(f"Starting k-fold statistics calculation...")
    print(f"Base Webdataset Path: {WDS_ROOT}")
    print(f"Output file will be: {OUTPUT_FILE_PATH}\n")

    all_stats = {}

    for k in range(K_FOLDS):
        print(f"--- Processing Fold {k} ---")

        # 1. Construct the path to this fold's training directory
        fold_train_dir = WDS_ROOT / f"fold_{k}" / WL_SETTING / DATASET_TYPE

        if not fold_train_dir.is_dir():
            print(f"Error: Directory not found, skipping: {fold_train_dir}")
            continue

        # 2. Find all .tar files in this *specific* directory
        tar_files = sorted(fold_train_dir.glob("shard-*.tar"))

        if not tar_files:
            print(f"Error: No 'shard-*.tar' files found in: {fold_train_dir}")
            continue

        print(f"Found {len(tar_files)} tar files for fold {k} training set.")

        # 3. Initialize Welford's aggregates *for this fold*
        agg_mono = (0, 0.0, 0.0)
        agg_t_fpa = (0, 0.0, 0.0)
        agg_pan_temp = (0, 0.0, 0.0)

        # 4. Loop over all tar files *for this fold*
        for tar_path in tqdm(tar_files, desc=f"Processing Fold {k}"):
            agg_mono, agg_t_fpa, agg_pan_temp = process_tar_file(
                tar_path, agg_mono, agg_t_fpa, agg_pan_temp
            )

        # 5. Finalize and store stats for this fold
        mean_m, std_m = finalize_welford_stats(agg_mono)
        mean_t, std_t = finalize_welford_stats(agg_t_fpa)
        mean_p, std_p = finalize_welford_stats(agg_pan_temp)

        fold_stats = {
            "mono_image": {
                "mean": mean_m,
                "std": std_m,
                "count": agg_mono[0]
            },
            "t_fpa": {
                "mean": mean_t,
                "std": std_t,
                "count": agg_t_fpa[0]
            },
            "pan_temp_map": {
                "mean": mean_p,
                "std": std_p,
                "count": agg_pan_temp[0]
            }
        }

        all_stats[f"fold_{k}"] = fold_stats

        print(f"\nStats for Fold {k}:")
        print(json.dumps(fold_stats, indent=2))
        print("-" * 20)

    # 6. Save all combined stats to the single JSON file
    if all_stats:
        try:
            with open(OUTPUT_FILE_PATH, 'w') as f:
                json.dump(all_stats, f, indent=4)
            print(f"\n=======================================================")
            print(f"Successfully saved all k-fold statistics to: {OUTPUT_FILE_PATH}")
            print(f"=======================================================")
        except Exception as e:
            print(f"\nError: Could not save final stats to {OUTPUT_FILE_PATH}. Error: {e}")
    else:
        print("\nNo statistics were calculated. No file saved.")


# --- Main execution block ---
if __name__ == "__main__":
    calculate_kfold_stats()