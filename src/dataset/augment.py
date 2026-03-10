import albumentations as A
import cv2


aug = A.Compose(
  [
    # 1) Geometric
    A.Affine(
      translate_percent={"x":(-0.02,0.02),"y":(-0.02,0.02)},
      scale=(0.95,1.05),
      rotate=(-3,3),
      interpolation=cv2.INTER_LINEAR,
      border_mode=cv2.BORDER_REFLECT,
      p=0
    ),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),

    # 2) Photometric
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
        A.RandomGamma(gamma_limit=(85,115), p=1),
        A.Sharpen(alpha=(0.1,0.3), lightness=(0.7,1.3), p=1),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),
    ], p=0.6),

    A.OneOf([
    A.GaussianBlur(blur_limit=(3,5), p=1),
    A.MedianBlur(blur_limit=(3,5), p=1),
    A.GaussNoise(std_range=(0.05, 0.15), noise_scale_factor=0.5, p=1),
    ], p=0.4),

  ],
  keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
  additional_targets={'image1':'image',
                      'keypoints1':'keypoints'}
)


def filter_pairs_in_frame(kp0, kp1, width=256, height=256):
    """
    kp0, kp1: (N,2) float arrays of (x,y)
    width, height: image dimensions
    Returns filtered kp0, kp1 of the same new length M <= N.
    """
    # Check each point is inside [0..w-1]×[0..h-1]
    in0 = (
        (kp0[:,0] >= 0) & (kp0[:,0] <  width) &
        (kp0[:,1] >= 0) & (kp0[:,1] <  height)
    )
    in1 = (
        (kp1[:,0] >= 0) & (kp1[:,0] <  width) &
        (kp1[:,1] >= 0) & (kp1[:,1] <  height)
    )
    keep = in0 & in1   # only pairs where both endpoints are valid

    return kp0[keep], kp1[keep]
