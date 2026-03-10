import numpy as np
import glob
import os
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from scipy.spatial import cKDTree

def nmea_to_dd(val):
    """Converts NMEA format (DDMM.MMMMM) to Decimal Degrees."""
    degrees = int(val / 100)
    minutes = val - (degrees * 100)
    return degrees + (minutes / 60.0)

def geodetic_to_ecef(lat, lon, alt):
    """Converts Decimal Degrees and Altitude to ECEF (X, Y, Z) in meters."""
    # WGS84 Ellipsoid Constants
    a = 6378137.0
    e2 = 0.00669437999014

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    # Radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * sin_lat**2)

    X = (N + alt) * cos_lat * np.cos(lon_rad)
    Y = (N + alt) * cos_lat * np.sin(lon_rad)
    Z = (N * (1 - e2) + alt) * sin_lat

    return X, Y, Z

def load_and_prepare_data(wavelength_dir):
    """Loads npz files from a directory and extracts IDs and 3D coordinates."""
    # Ensure you are targeting the directory properly
    files = glob.glob(os.path.join(wavelength_dir, '*.npz'))

    indices = []
    coordinates = []

    for file_path in tqdm(files):
        # Extract the image ID from the filename (e.g., 'pan/123.npz' -> '123')
        basename = os.path.basename(file_path)
        idx = os.path.splitext(basename)[0]

        # Load the file safely
        with np.load(file_path) as data:
            lat_nmea = float(data['lat'])
            long_nmea = float(data['long'])
            alt = float(data['alt'])

            # 1. Convert NMEA to Decimal Degrees
            lat_dd = nmea_to_dd(lat_nmea)
            long_dd = nmea_to_dd(long_nmea)

            # 2. Convert Geodetic to 3D Cartesian (Meters)
            x, y, z = geodetic_to_ecef(lat_dd, long_dd, alt)

            indices.append(idx)
            coordinates.append((x, y, z))

    return np.array(indices), np.array(coordinates)

def find_closest_pairs(dir_a, dir_b, distance_threshold=100.0):
    """Finds image pairs between two directories within a max distance (meters)."""
    print(f"Loading data from {dir_a}...")
    indices_a, coords_a = load_and_prepare_data(dir_a)

    print(f"Loading data from {dir_b}...")
    indices_b, coords_b = load_and_prepare_data(dir_b)

    if len(coords_a) == 0 or len(coords_b) == 0:
        print("One or both directories are empty.")
        return []

    print("Building KD-Tree and searching...")
    # Build the search index for List A
    tree_a = cKDTree(coords_a)

    # Query List B against List A
    distances, closest_indices_in_a = tree_a.query(coords_b, distance_upper_bound=distance_threshold)

    pairs = []
    for idx_in_b, idx_in_a in tqdm(enumerate(closest_indices_in_a)):
        # SciPy returns the length of list A (out of bounds) if no match is found within the threshold
        if idx_in_a != tree_a.n:
            img_id_a = indices_a[idx_in_a]
            img_id_b = indices_b[idx_in_b]
            distance = distances[idx_in_b]
            pairs.append((img_id_a, img_id_b, distance))

    return pairs


def save_matches_to_csv(matched_pairs, csv_dir, wl0, wl1):
    # 1. Convert the list of tuples into a DataFrame
    # Assuming matched_pairs looks like: [('idx_pan', 'idx_9000nm', distance), ...]
    output_filename = f"sorted_matched_pairs_{wl0}_{wl1}.csv"
    df = pd.DataFrame(matched_pairs, columns=[wl1, wl0, 'Distance_meters'])

    # 2. Sort by distance (lowest to highest)
    df_sorted = df.sort_values(by='Distance_meters', ascending=True)

    print(df_sorted.head(10))
    # 3. Save to CSV without the index column
    df_sorted.to_csv(csv_dir / output_filename, index=False)

    print(f"Successfully saved {len(df_sorted)} sorted pairs to {output_filename}")



# --- Execution ---
if __name__ == "__main__":
    os.chdir(r"F:\yakirs_thesis\thesis_code")
    csv_dir = Path(r"F:\yakirs_thesis\thesis_code\sim2real\csv_outputs")
    csv_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(r"F:\yakirs_thesis\data\aerial_ds\land_frames")
    wl0 = "11000nm"
    wl1 = "pan"
    matched_pairs = find_closest_pairs(data_dir/wl1, data_dir/wl0, distance_threshold=50.0)
    save_matches_to_csv(matched_pairs, csv_dir, wl0, wl1)

