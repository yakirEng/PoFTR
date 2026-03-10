import numpy as np
import tarfile
import io
from tqdm import tqdm
from pathlib import Path
import os
import json
from loguru import logger

from src.configs.poftr_configs import get_config
from src.utils.misc import lower_config



def update_welford_stats(existing_aggregate, new_values):
    """
    Updates running stats using Chan's method for combining batches.
    Orders of magnitude faster than iterative Welford.
    """
    (n_a, mean_a, m2_a) = existing_aggregate

    # 1. Calculate stats for the new batch (Chunk B) using fast numpy vectorization
    n_b = new_values.size
    if n_b == 0:
        return existing_aggregate

    mean_b = np.mean(new_values)
    # Sum of squared differences from the batch mean
    m2_b = np.sum((new_values - mean_b) ** 2)

    # 2. Combine existing global stats (Chunk A) with new batch (Chunk B)
    n_new = n_a + n_b
    delta = mean_b - mean_a

    mean_new = mean_a + delta * (n_b / n_new)
    m2_new = m2_a + m2_b + (delta ** 2) * (n_a * n_b / n_new)

    return (n_new, mean_new, m2_new)


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
        return (mean, 0.0)  # Cannot compute variance with < 2 points

    mean = mean
    variance = M2 / count
    std_dev = np.sqrt(variance)

    return (mean, std_dev)


def process_tar_file(tar_path, phys0_stats, phys1_stats, image0_stats, image1_stats):
    """
    Processes a single .tar file using the explicit 'mask0'/'mask1' from disk.
    """
    try:
        with tarfile.open(tar_path, 'r') as tar:
            npz_files = [m for m in tar.getmembers() if m.name.endswith('.npz')]

            if not npz_files:
                return phys0_stats, phys1_stats, image0_stats, image1_stats

            for member in npz_files:
                file_buffer = tar.extractfile(member).read()

                with io.BytesIO(file_buffer) as f:
                    try:
                        data = np.load(f)

                        # --- IMAGE 0 / PHYS 0 ---
                        if 'image0' in data and 'phys0' in data:
                            # 1. Get the Mask
                            if 'mask0' in data:
                                mask0 = data['mask0'].astype(bool)
                            else:
                                raise ValueError("No mask0 found in data")

                            # 2. Filter using Mask
                            valid_phys = data['phys0'][mask0]
                            if valid_phys.size > 0:
                                phys0_stats = update_welford_stats(phys0_stats, valid_phys)

                            valid_image = data['image0'][mask0]
                            if valid_image.size > 0:
                                image0_stats = update_welford_stats(image0_stats, valid_image)

                        # --- IMAGE 1 / PHYS 1 ---
                        if 'image1' in data and 'phys1' in data:
                            if 'mask1' in data:
                                mask1 = data['mask1'].astype(bool)
                            else:
                                raise ValueError("No mask1 found in data")

                            valid_phys = data['phys1'][mask1]
                            if valid_phys.size > 0:
                                phys1_stats = update_welford_stats(phys1_stats, valid_phys)

                            valid_image = data['image1'][mask1]
                            if valid_image.size > 0:
                                image1_stats = update_welford_stats(image1_stats, valid_image)

                    except Exception as e:
                        print(f"\nWarning: Error processing {member.name} in {tar_path}: {e}")
                        continue

    except Exception as e:
        print(f"\nError opening tar {tar_path}: {e}")

    return phys0_stats, phys1_stats, image0_stats, image1_stats


def calculate_datasets_stats(directory_path):
    """
    Main function to find and process all .tar files in a directory.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Provided path is not a directory: {directory_path}")
        return

    print(f"Scanning for .tar files in directory: {directory_path}")

    tar_files = [f for f in os.listdir(directory_path) if f.endswith('.tar')]

    if not tar_files:
        print("Error: No .tar files found in the specified directory.")
        return

    print(f"Found {len(tar_files)} tar files to process.")

    # Initialize Welford's algorithm aggregates
    phys0_stats = (0, 0.0, 0.0)
    phys1_stats = (0, 0.0, 0.0)

    image0_stats = (0, 0.0, 0.0)
    image1_stats = (0, 0.0, 0.0)

    # Loop over all found tar files
    pbar = tqdm(tar_files, desc="Initializing... ", leave=False)
    for i, tar_file_name in enumerate(pbar):
        pbar.set_description(f"Processing {tar_file_name}... ")
        tar_file_path = os.path.join(directory_path, tar_file_name)
        phys0_stats, phys1_stats, image0_stats, image1_stats = process_tar_file(
            tar_file_path, phys0_stats, phys1_stats, image0_stats, image1_stats
        )


    stats_data = {}

    # --- Finalize and Print Statistics for phys0 ---
    total_count_phys0 = phys0_stats[0]
    if total_count_phys0 >= 0:
        final_mean_phys0, final_std_phys0 = finalize_welford_stats(phys0_stats)
        print("\n--- Statistics for phys0 ---")
        print(f"Total valid data points found: {total_count_phys0}")
        print(f"Global Mean (μ):                {final_mean_phys0:.6f}")
        print(f"Global Standard Deviation (σ):  {final_std_phys0:.6f}")
        stats_data['phys0'] = {
            'mean': float(final_mean_phys0),
            'std': float(final_std_phys0),
            'count': float(total_count_phys0)
        }
    else:
        print("\nNo valid data found for phys0")

    # --- Finalize and Print Statistics for phys1 ---
    total_count_phys1 = phys1_stats[0]
    if total_count_phys1 >= 0:
        final_mean_phys1, final_std_phys1 = finalize_welford_stats(phys1_stats)
        print("\n--- Statistics for phys1 ---")
        print(f"Total valid data points found: {total_count_phys1}")
        print(f"Global Mean (μ):                {final_mean_phys1:.6f}")
        print(f"Global Standard Deviation (σ):  {final_std_phys1:.6f}")
        stats_data['phys1'] = {
            'mean': float(final_mean_phys1),
            'std': float(final_std_phys1),
            'count': float(total_count_phys1)
        }
    else:
        print("\nNo valid data found for phys1")

    # --- Finalize and Print Statistics for image0  ---
    total_count_image0 = image0_stats[0]
    if total_count_image0 >= 0:
        final_mean_image0, final_std_image0 = finalize_welford_stats(image0_stats)
        print("\n--- Statistics for image0 ---")
        print(f"Total valid data points found: {total_count_image0}")
        print(f"Global Mean (μ):                {final_mean_image0:.6f}")
        print(f"Global Standard Deviation (σ):  {final_std_image0:.6f}")
        stats_data['image0'] = {
            'mean': float(final_mean_image0),
            'std': float(final_std_image0),
            'count': float(total_count_image0)
        }
    else:
        print("\nNo valid data found for image0")

    # --- Finalize and Print Statistics for phy ---
    total_count_image1 = image1_stats[0]
    if total_count_image1 >= 0:
        final_mean_image1, final_std_image1 = finalize_welford_stats(image1_stats)
        print("\n--- Statistics for image1 ---")
        print(f"Total valid data points found: {total_count_image1}")
        print(f"Global Mean (μ):                {final_mean_image1:.6f}")
        print(f"Global Standard Deviation (σ):  {final_std_image1:.6f}")
        stats_data['image1'] = {
            'mean': float(final_mean_image1),
            'std': float(final_std_image1),
            'count': float(total_count_image1)
        }
    else:
        print("\nNo valid data found for image1")


    # --- Save statistics to a JSON file ---
    if stats_data:
        output_path = os.path.join(directory_path.parent, f'stats.json')
        try:
            with open(output_path, 'w') as f:
                json.dump(stats_data, f, indent=4)
            print(f"\nSuccessfully saved statistics to: {output_path}")
        except Exception as e:
            print(f"\nError: Could not save statistics to {output_path}. Error: {e}")



def main():
    config = lower_config(get_config())
    os.chdir(config["poftr"]["proj"]["cwd"])
    distribution_types = ["truncnorm"]
    dataset_versions = ["9um_11um", "9um_pan", "11um_pan"]
    ablation_types = ["standard"] # noised_priors, standard
    for distribution_type in distribution_types:
        for dataset_version in dataset_versions:
            for ablation_type in ablation_types:
                if ablation_type != "standard":
                    dataset_str = dataset_version + f"_{ablation_type}"
                    directory_to_process = Path(rf"F:\yakirs_thesis\thesis_code\data\simulated\datasets\{distribution_type}\ablations\{dataset_str}\train")
                else:
                    directory_to_process = Path(rf"F:\yakirs_thesis\thesis_code\data\simulated\datasets\{distribution_type}\{dataset_version}\train")
                logger.info(f"Starting statistics calculation for directory: {directory_to_process}")
                calculate_datasets_stats(directory_to_process)

if __name__ == "__main__":
    main()

