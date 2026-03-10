import numpy as np
import torch
# from src.physical_model.petit_s.dataset.augment import aug

def numpy2tensor(*numpy_arrays):
    tensors = [torch.from_numpy(array).float() for array in numpy_arrays]
    return tuple(tensors)

def expand_dims(*numpy_arrays):
    np_channelized = [np.expand_dims(array, axis=0) for array in numpy_arrays]
    return tuple(np_channelized)


def norm_global_zscore(x, mean, std, eps=1e-8):
    """
    Normalizes an entire array 'x' using pre-calculated
    global mean and std (Z-score).

    This is used for 'mono_image', 't_fpa', and 'pan_temp_map'.

    :param x: Input array (can be scalar or multi-dimensional)
    :param mean: Pre-calculated global mean for this data type.
    :param std: Pre-calculated global standard deviation.
    :param eps: Epsilon to avoid division by zero.
    :return: Normalized array (float32)
    """
    # Ensure x is float32 for normalization
    x = x.astype(np.float32)

    # Calculate z-score
    normed_x = (x - mean) / (std + eps)

    return normed_x


def normalize_sample(sample, fold_stats):
    """
    Applies global Z-score normalization to each part of the
    data sample using the stats for the *current* fold.

    :param sample: A dictionary containing 'mono_image', 't_fpa', 'pan_temp_map'.
    :param fold_stats: A dictionary for the *specific fold* (e.g., the value of stats['fold_0']).
    :return: The sample with normalized data.
    """

    # 1. Normalize 'mono_image' (Input 1)
    stats_mono = fold_stats['mono_image']
    sample['mono_image'] = norm_global_zscore(
        sample['mono_image'],
        mean=stats_mono['mean'],
        std=stats_mono['std']
    )

    # 2. Normalize 't_fpa' (Input 2)
    stats_t_fpa = fold_stats['t_fpa']
    sample['t_fpa'] = norm_global_zscore(
        sample['t_fpa'],
        mean=stats_t_fpa['mean'],
        std=stats_t_fpa['std']
    )

    # 3. Normalize 'pan_temp_map' (Ground Truth)
    stats_gt = fold_stats['pan_temp_map']
    sample['pan_temp_map'] = norm_global_zscore(
        sample['pan_temp_map'],
        mean=stats_gt['mean'],
        std=stats_gt['std']
    )

    return sample

def denormalize_sample(sample, fold_stats):
    """
    Reverts global Z-score normalization for each part of the
    data sample using the stats for the *current* fold.

    :param sample: A dictionary containing 'mono_image', 't_fpa', 'pan_temp_map'.
    :param fold_stats: A dictionary for the *specific fold* (e.g., the value of stats['fold_0']).
    :return: The sample with denormalized data.
    """

    # 1. Denormalize 'mono_image' (Input 1)
    stats_mono = fold_stats['mono_image']
    sample['mono_image'] = sample['mono_image'] * (stats_mono['std']) + stats_mono['mean']

    # 2. Denormalize 't_fpa' (Input 2)
    stats_t_fpa = fold_stats['t_fpa']
    sample['t_fpa'] = sample['t_fpa'] * (stats_t_fpa['std']) + stats_t_fpa['mean']

    # 3. Denormalize 'pan_temp_map' (Ground Truth)
    stats_gt = fold_stats['pan_temp_map']
    sample['pan_temp_map'] = sample['pan_temp_map'] * (stats_gt['std']) + stats_gt['mean']

    return sample



def to_tensor(sample_dict):
    sample_dict['pan_temp_map'] = np.expand_dims(sample_dict['pan_temp_map'], axis=0)
    sample_dict['mono_image'] = np.expand_dims(sample_dict['mono_image'], axis=0)

    for key, value in sample_dict.items():
        if key in ('image_name', 'idx'):
            if key == 'image_name' and isinstance(value, np.ndarray):
                # collapse 0-d or 1-d string array to str or list[str]
                if value.ndim == 0:
                    sample_dict[key] = str(value.item())
                else:
                    sample_dict[key] = [str(v) for v in value.tolist()]
            elif key == 'image_name' and isinstance(value, (np.str_, bytes)):
                sample_dict[key] = str(value)
            continue
        if isinstance(value, np.ndarray):
            # If it's a string array
            if np.issubdtype(value.dtype, np.str_):
                # Convert string arrays to a list of strings
                sample_dict[key] = tuple(value)
                continue

            # Only copy if necessary
            if not value.flags.writeable:
                value = value.copy()

            sample_dict[key] = torch.from_numpy(value).float()

        elif isinstance(value, list):
            sample_dict[key] = torch.tensor(value, dtype=torch.float32)

        elif isinstance(value, (int, float)):
            sample_dict[key] = torch.tensor(value, dtype=torch.float32)

    return sample_dict