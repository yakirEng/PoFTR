import cv2
import numpy as np
import math
from itertools import chain
import socket
import torch
# from lightning.fabric.utilities import rank_zero_only
from loguru import logger

def center_crop(image: np.ndarray, crop_size: tuple) -> np.ndarray:
    """Crops the image around the center to the desired size (no padding)."""
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size

    # Calculate the center of the image
    center_h, center_w = h // 2, w // 2

    # Calculate the cropping area
    crop_top = center_h - crop_h // 2
    crop_bottom = center_h + crop_h // 2
    crop_left = center_w - crop_w // 2
    crop_right = center_w + crop_w // 2

    # Crop the image
    return image[crop_top:crop_bottom, crop_left:crop_right]

def calculate_iou(config, lt1, lt2):
    # Bounding box coordinates and dimensions
    x1, y1 = lt1[0], lt1[1]
    w1, h1 = config.sample_shape[1], config.sample_shape[2]
    x2, y2,  = lt2[0], lt2[1]
    w2, h2 = config.sample_shape[1], config.sample_shape[2]

    # Calculate the coordinates of the intersection rectangle
    intersection_x1 = max(x1, x2)
    intersection_y1 = max(y1, y2)
    intersection_x2 = min(x1 + w1, x2 + w2)
    intersection_y2 = min(y1 + h1, y2 + h2)

    # Calculate the area of intersection rectangle
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)

    # Calculate the area of both bounding boxes
    area_bbox1 = w1 * h1
    area_bbox2 = w2 * h2

    # Calculate the Union area
    union_area = area_bbox1 + area_bbox2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def norm_01(image):

    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val)
    return normalized

def _convert_conf2masks(conf_matrix):
    dim = int(math.isqrt(conf_matrix.shape[0]))
    mask0, mask1 = np.zeros((dim, dim)), np.zeros((dim, dim))


def iou_is_ok(config, lt1, lt2):
    iou = calculate_iou(config, lt1, lt2)
    if iou < config.min_iou or iou > config.max_iou:
        return False
    else:
        return True

def in_tile_boundries(config, tile, lt1):
    x, y = lt1[0], lt1[1]
    if x + config.sample_shape[1] > tile.shape[0] or y + config.sample_shape[2] > tile.shape[1]:
        return False
    elif x < 0 or y < 0:
        return False
    else:
        return True

def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def resize_tuple(vec, scale_percent):
    np_vec = np.array(vec)
    resized = np_vec * scale_percent // 100
    return resized


def nonzero_norm01(img):
    """
    Normalize the image to [0, 1] range only for non-zero pixels.
    Zeros (assumed to be invalid/background) remain zero.
    """
    normed = np.zeros_like(img, dtype=np.float32)
    mask = img > 0

    if np.any(mask):
        min_val = img[mask].min()
        max_val = img[mask].max()
        normed[mask] = (img[mask] - min_val) / (max_val - min_val + 1e-8)

    return normed

def flattenList(x):
    return list(chain(*x))


import os
import contextlib
import joblib
from typing import Union
from loguru import logger
from itertools import chain

import torch
from yacs.config import CfgNode as CN
from pytorch_lightning.utilities import rank_zero_only


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def upper_config(dict_cfg):
    if not isinstance(dict_cfg, dict):
        return dict_cfg
    return {k.upper(): upper_config(v) for k, v in dict_cfg.items()}


def log_on(condition, message, level):
    if condition:
        assert level in ['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL']
        logger.log(level, message)


def get_rank_zero_only_logger(logger):
    if rank_zero_only.rank == 0:
        return logger
    else:
        for _level in logger._core.levels.keys():
            level = _level.lower()
            setattr(logger, level,
                    lambda x: None)
        logger._log = lambda x: None
    return logger


def setup_gpus(gpus: Union[str, int]) -> int:
    """ A temporary fix for pytorch-lighting 1.3.x """
    gpus = str(gpus)
    gpu_ids = []

    if ',' not in gpus:
        n_gpus = int(gpus)
        return n_gpus if n_gpus != -1 else torch.cuda.device_count()
    else:
        gpu_ids = [i.strip() for i in gpus.split(',') if i != '']

    # setup environment variables
    visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    if visible_devices is None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in gpu_ids)
        visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
        logger.warning(
            f'[Temporary Fix] manually set CUDA_VISIBLE_DEVICES when specifying gpus to use: {visible_devices}')
    else:
        logger.warning('[Temporary Fix] CUDA_VISIBLE_DEVICES already set by user or the main process.')
    return len(gpu_ids)


def flattenList(x):
    return list(chain(*x))


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument

    Usage:
        with tqdm_joblib(tqdm(desc="My calculation", total=10)) as progress_bar:
            Parallel(n_jobs=16)(delayed(sqrt)(i**2) for i in range(10))

    When iterating over a generator, directly use of tqdm is also a solutin (but monitor the task queuing, instead of finishing)
        ret_vals = Parallel(n_jobs=args.world_size)(
                    delayed(lambda x: _compute_cov_score(pid, *x))(param)
                        for param in tqdm(combinations(image_ids, 2),
                                          desc=f'Computing cov_score of [{pid}]',
                                          total=len(image_ids)*(len(image_ids)-1)/2))
    Src: https://stackoverflow.com/a/58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

