import csv
from pathlib import Path


def create_master_results_csv(output_path="results.csv"):
    # Columns: Run Name | Wavelength | Fold | LPIPS (Lower is better) | PSNR | RMSE
    # Data taken from your final Test Set screenshots (images 10328a and 103206)
    headers = ["run_name", "wl", "fold_idx", "val_lpips", "val_psnr", "val_rmse"]

    data = [
        # --- 9um Band ---
        ["31_12_16_47", "9um", 0, 0.070638, 35.530, 0.1005],
        ["29_12_15_35", "9um", 1, 0.097427, 34.150, 0.1177],
        ["29_12_15_35", "9um", 2, 0.070612, 35.492, 0.1009],
        ["31_12_20_53", "9um", 3, 0.069436, 35.585, 0.0998],  # Best 9um
        ["29_12_15_37", "9um", 4, 0.097902, 34.083, 0.1186],

        # --- 11um Band ---
        ["31_12_20_56", "11um", 0, 0.126305, 32.809, 0.1374],
        ["29_12_15_39", "11um", 1, 0.107616, 33.722, 0.1237],  # Best 11um
        ["31_12_16_53", "11um", 2, 0.118215, 33.289, 0.1300],
        ["30_12_13_26", "11um", 3, 0.110197, 33.624, 0.1251],
        ["30_12_13_27", "11um", 4, 0.131530, 32.753, 0.1383],
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)

    print(f"[OK] Created {output_path} with {len(data)} rows.")


import csv
import numpy as np
from pathlib import Path


def create_master_results_csv(output_path="results.csv"):
    """Creates the raw results CSV from your final test set numbers."""
    # Columns: Run Name | Wavelength | Fold | LPIPS | PSNR | RMSE
    headers = ["run_name", "wl", "fold_idx", "val_lpips", "val_psnr", "val_rmse"]

    data = [
        # --- 9um Band ---
        ["31_12_16_47", "9um", 0, 0.070638, 35.530, 0.1005],
        ["29_12_15_35", "9um", 1, 0.097427, 34.150, 0.1177],
        ["29_12_15_35", "9um", 2, 0.070612, 35.492, 0.1009],
        ["31_12_20_53", "9um", 3, 0.069436, 35.585, 0.0998],  # Best 9um
        ["29_12_15_37", "9um", 4, 0.097902, 34.083, 0.1186],

        # --- 11um Band ---
        ["31_12_20_56", "11um", 0, 0.126305, 32.809, 0.1374],
        ["29_12_15_39", "11um", 1, 0.107616, 33.722, 0.1237],  # Best 11um
        ["31_12_16_53", "11um", 2, 0.118215, 33.289, 0.1300],
        ["30_12_13_26", "11um", 3, 0.110197, 33.624, 0.1251],
        ["30_12_13_27", "11um", 4, 0.131530, 32.753, 0.1383],
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)

    print(f"[OK] Created {output_path} with {len(data)} rows.")


def print_results_table(csv_path="results.csv"):
    """Reads the CSV, calculates Mean/Std, and prints a formatted text table."""

    # 1. Read Data
    data = {"9um": {"lpips": [], "rmse": [], "psnr": []},
            "11um": {"lpips": [], "rmse": [], "psnr": []}}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wl = row["wl"]
            data[wl]["lpips"].append(float(row["val_lpips"]))
            data[wl]["rmse"].append(float(row["val_rmse"]))
            data[wl]["psnr"].append(float(row["val_psnr"]))

    # 2. Print Header
    print("\n" + "=" * 75)
    print(f"{'QUANTITATIVE RESULTS (Mean ± Std)':^75}")
    print("=" * 75)
    print(f"{'Spectral Band':<15} | {'LPIPS (↓)':<18} | {'RMSE (↓)':<18} | {'PSNR (↑)':<15}")
    print("-" * 75)

    # 3. Calculate and Print Rows
    for wl in ["9um", "11um"]:
        # Calculate stats
        lpips_m, lpips_s = np.mean(data[wl]["lpips"]), np.std(data[wl]["lpips"], ddof=1)
        rmse_m, rmse_s = np.mean(data[wl]["rmse"]), np.std(data[wl]["rmse"], ddof=1)
        psnr_m, psnr_s = np.mean(data[wl]["psnr"]), np.std(data[wl]["psnr"], ddof=1)

        # Format strings
        l_str = f"{lpips_m:.2f} ± {lpips_s:.2f}"
        r_str = f"{rmse_m:.2f} ± {rmse_s:.2f}"
        p_str = f"{psnr_m:.2f} ± {psnr_s:.2f}"

        print(f"{wl:<15} | {l_str:<18} | {r_str:<18} | {p_str:<15}")

    print("=" * 75 + "\n")


if __name__ == "__main__":
    csv_file = "results.csv"
    # create_master_results_csv(csv_file)
    print_results_table(csv_file)
