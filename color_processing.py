"""
Color processing module for Image Processing HW2.

Implements:
  - 3A: Local color histogram modification using Algorithm A on
        R, G, B individually or all three together.
  - 3B: Augmentation of color images generating 20 images per ROI.
"""

import numpy as np

from histogram import algorithm_a, rotate_90, rotate_180, rotate_270


# ---------------------------------------------------------------------------
# 3A: Local color histogram modification
# ---------------------------------------------------------------------------

def apply_algorithm_a_color(
    roi: np.ndarray,
    channels: str = "all",
) -> np.ndarray:
    """Apply Algorithm A to one or more channels of a colour ROI.

    Parameters
    ----------
    roi : np.ndarray
        3-D uint8 array of shape (H, W, 3) in BGR or RGB order.
    channels : str
        Which channel(s) to process.  Accepted values:
          - "R" or "r"  → process channel index 0 (first channel)
          - "G" or "g"  → process channel index 1 (second channel)
          - "B" or "b"  → process channel index 2 (third channel)
          - "all"       → process all three channels independently

        The caller is responsible for passing the array in a consistent
        channel order (e.g. always RGB or always BGR).  The string labels
        "R", "G", "B" map to indices 0, 1, 2 regardless.

    Returns
    -------
    np.ndarray
        Modified 3-D uint8 array with the same shape as *roi*.
    """
    result = roi.copy()
    ch_map = {"r": 0, "g": 1, "b": 2}

    if channels.lower() == "all":
        for idx in range(3):
            result[:, :, idx] = algorithm_a(roi[:, :, idx])
    elif channels.lower() in ch_map:
        idx = ch_map[channels.lower()]
        result[:, :, idx] = algorithm_a(roi[:, :, idx])
    else:
        raise ValueError(
            f"Invalid channels value '{channels}'. "
            "Expected 'R', 'G', 'B', or 'all'."
        )

    return result


# ---------------------------------------------------------------------------
# 3B: Augmentation of colour images (20 images per ROI)
# ---------------------------------------------------------------------------

def augment_color(roi: np.ndarray) -> list:
    """Generate 20 augmented colour images from a single colour ROI.

    Image set (in order):
      1.  Original
      2.  Original rotated 90°
      3.  Original rotated 180°
      4.  Original rotated 270°
      5.  (A) on R + original orientation
      6.  (A) on R + 90°
      7.  (A) on R + 180°
      8.  (A) on R + 270°
      9.  (A) on G + original orientation
      10. (A) on G + 90°
      11. (A) on G + 180°
      12. (A) on G + 270°
      13. (A) on B + original orientation
      14. (A) on B + 90°
      15. (A) on B + 180°
      16. (A) on B + 270°
      17. (A) on R+G+B + original orientation
      18. (A) on R+G+B + 90°
      19. (A) on R+G+B + 180°
      20. (A) on R+G+B + 270°

    Parameters
    ----------
    roi : np.ndarray
        3-D uint8 colour array (H, W, 3).

    Returns
    -------
    list of (np.ndarray, str)
        Each element is (image_array, label_string).
    """
    a_r   = apply_algorithm_a_color(roi, "R")
    a_g   = apply_algorithm_a_color(roi, "G")
    a_b   = apply_algorithm_a_color(roi, "B")
    a_all = apply_algorithm_a_color(roi, "all")

    images = [
        (roi.copy(),          "Original"),
        (rotate_90(roi),      "Original 90°"),
        (rotate_180(roi),     "Original 180°"),
        (rotate_270(roi),     "Original 270°"),
        (a_r,                 "A(R)"),
        (rotate_90(a_r),      "A(R) 90°"),
        (rotate_180(a_r),     "A(R) 180°"),
        (rotate_270(a_r),     "A(R) 270°"),
        (a_g,                 "A(G)"),
        (rotate_90(a_g),      "A(G) 90°"),
        (rotate_180(a_g),     "A(G) 180°"),
        (rotate_270(a_g),     "A(G) 270°"),
        (a_b,                 "A(B)"),
        (rotate_90(a_b),      "A(B) 90°"),
        (rotate_180(a_b),     "A(B) 180°"),
        (rotate_270(a_b),     "A(B) 270°"),
        (a_all,               "A(R+G+B)"),
        (rotate_90(a_all),    "A(R+G+B) 90°"),
        (rotate_180(a_all),   "A(R+G+B) 180°"),
        (rotate_270(a_all),   "A(R+G+B) 270°"),
    ]
    return images
