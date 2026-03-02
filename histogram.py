"""
Histogram processing module for Image Processing HW2.

Implements:
  - Algorithm A: Modified histogram stretching with Optimal Thresholding (1A)
  - Algorithm B: Local histogram stretching on 4 quarters of the ROI (1B)
  - Grey-level image augmentation generating 12 images per ROI (1C)
  - Recursive local stretching with homogeneity criterion (1D - extra credit)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Optimal Thresholding (used by Algorithm A)
# ---------------------------------------------------------------------------

def optimal_threshold(region: np.ndarray) -> float:
    """Compute Optimal Threshold T for a greyscale region (2-D uint8 array).

    Algorithm:
      1. Initialise T = (max + min) / 2.
      2. Split pixels into "dark" (I < T) and "bright" (I >= T).
      3. T_new = (mean_dark + mean_bright) / 2.
      4. Repeat until |T_new - T| < 0.5 (converged).

    Returns
    -------
    float
        The optimal threshold value.
    """
    pixels = region.flatten().astype(float)
    T = (float(pixels.min()) + float(pixels.max())) / 2.0

    for _ in range(1000):  # safety cap on iterations
        dark = pixels[pixels < T]
        bright = pixels[pixels >= T]

        if dark.size == 0 or bright.size == 0:
            # Degenerate case: all pixels are the same or split fails
            break

        mean_dark = dark.mean()
        mean_bright = bright.mean()
        T_new = (mean_dark + mean_bright) / 2.0

        if abs(T_new - T) < 0.5:
            T = T_new
            break
        T = T_new

    return T


# ---------------------------------------------------------------------------
# Algorithm A: Modified histogram stretching
# ---------------------------------------------------------------------------

def algorithm_a(region: np.ndarray) -> np.ndarray:
    """Apply Algorithm A (modified histogram stretching) to a greyscale region.

    Only "dark" pixels (intensity < T, where T is the Optimal Threshold) are
    modified.  Pixels with intensity >= T remain unchanged.

    Steps
    -----
    A1. Compute T via Optimal Thresholding.
    A2. Identify dark pixels (I < T).  Compute Imin / Imax of dark pixels,
        then c = 1.05 * Imin and d = 0.95 * Imax, clamped to [0, 255].
    A3. Apply piecewise linear mapping to dark pixels only:
          I' = 0             if I <= c
          I' = 255*(I-c)/(d-c)  if c < I < d
          I' = 255           if I >= d

    Edge cases
    ----------
    - No dark pixels (all I >= T):  region is returned unchanged; a message
      is printed to stdout.
    - c >= d (degenerate stretch range):  dark pixels are mapped to 0; a
      message is printed.

    Parameters
    ----------
    region : np.ndarray
        2-D uint8 greyscale array (H x W).

    Returns
    -------
    np.ndarray
        Modified 2-D uint8 array of the same shape.
    """
    result = region.copy().astype(np.float64)
    T = optimal_threshold(region)

    dark_mask = region < T

    if not dark_mask.any():
        # No dark pixels — leave region unchanged
        print(
            f"[Algorithm A] No dark pixels found (T={T:.2f}). "
            "ROI left unchanged."
        )
        return region.copy()

    dark_pixels = region[dark_mask].astype(float)
    I_min = dark_pixels.min()
    I_max = dark_pixels.max()

    # Step A2: compute c and d, clamped to [0, 255]
    c = float(np.clip(1.05 * I_min, 0, 255))
    d = float(np.clip(0.95 * I_max, 0, 255))

    if c >= d:
        # Degenerate range — map all dark pixels to 0
        print(
            f"[Algorithm A] Degenerate stretch range (c={c:.2f} >= d={d:.2f}). "
            "Dark pixels mapped to 0."
        )
        result[dark_mask] = 0.0
        return np.clip(result, 0, 255).astype(np.uint8)

    # Step A3: piecewise linear mapping on dark pixels
    vals = result[dark_mask]
    mapped = np.where(
        vals <= c,
        0.0,
        np.where(
            vals >= d,
            255.0,
            255.0 * (vals - c) / (d - c),
        ),
    )
    result[dark_mask] = mapped

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Algorithm B: Local histogram stretching (4 quarters)
# ---------------------------------------------------------------------------

def algorithm_b(region: np.ndarray) -> np.ndarray:
    """Apply Algorithm B: split ROI into 4 quarters and apply Algorithm A
    independently to each quarter.

    Splitting convention when dimensions are odd:
      - Top-left quarter:  rows [0, H//2), cols [0, W//2)
      - Top-right quarter: rows [0, H//2), cols [W//2, W)
      - Bottom-left:       rows [H//2, H), cols [0, W//2)
      - Bottom-right:      rows [H//2, H), cols [W//2, W)
    The extra row/column (when H or W is odd) is assigned to the
    bottom / right quarter respectively.

    Parameters
    ----------
    region : np.ndarray
        2-D uint8 greyscale array (H x W).

    Returns
    -------
    np.ndarray
        Modified 2-D uint8 array of the same shape.
    """
    H, W = region.shape
    mh = H // 2
    mw = W // 2

    result = region.copy()

    # Quarter indices: (row_start, row_end, col_start, col_end)
    quarters = [
        (0,  mh, 0,  mw),   # top-left
        (0,  mh, mw, W),    # top-right
        (mh, H,  0,  mw),   # bottom-left
        (mh, H,  mw, W),    # bottom-right
    ]

    for r0, r1, c0, c1 in quarters:
        sub = region[r0:r1, c0:c1]
        if sub.size == 0:
            continue
        result[r0:r1, c0:c1] = algorithm_a(sub)

    return result


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rotate_90(image: np.ndarray) -> np.ndarray:
    """Rotate image 90 degrees counter-clockwise."""
    return np.rot90(image, k=1)


def rotate_180(image: np.ndarray) -> np.ndarray:
    """Rotate image 180 degrees."""
    return np.rot90(image, k=2)


def rotate_270(image: np.ndarray) -> np.ndarray:
    """Rotate image 270 degrees counter-clockwise (90 clockwise)."""
    return np.rot90(image, k=3)


# ---------------------------------------------------------------------------
# 1C: Grey-level augmentation (12 images per ROI)
# ---------------------------------------------------------------------------

def augment_greyscale(roi: np.ndarray) -> list:
    """Generate 12 augmented greyscale images from a single ROI.

    Image set (in order):
      1.  Original
      2.  Original rotated 90°
      3.  Original rotated 180°
      4.  Original rotated 270°
      5.  (A) processed
      6.  (A) rotated 90°
      7.  (A) rotated 180°
      8.  (A) rotated 270°
      9.  (B) processed
      10. (B) rotated 90°
      11. (B) rotated 180°
      12. (B) rotated 270°

    Parameters
    ----------
    roi : np.ndarray
        2-D uint8 greyscale array.

    Returns
    -------
    list of (np.ndarray, str)
        Each element is (image_array, label_string).
    """
    a_proc = algorithm_a(roi)
    b_proc = algorithm_b(roi)

    images = [
        (roi.copy(),         "Original"),
        (rotate_90(roi),     "Original 90°"),
        (rotate_180(roi),    "Original 180°"),
        (rotate_270(roi),    "Original 270°"),
        (a_proc,             "Alg-A"),
        (rotate_90(a_proc),  "Alg-A 90°"),
        (rotate_180(a_proc), "Alg-A 180°"),
        (rotate_270(a_proc), "Alg-A 270°"),
        (b_proc,             "Alg-B"),
        (rotate_90(b_proc),  "Alg-B 90°"),
        (rotate_180(b_proc), "Alg-B 180°"),
        (rotate_270(b_proc), "Alg-B 270°"),
    ]
    return images


# ---------------------------------------------------------------------------
# 1D (Extra credit): Recursive local stretching
# ---------------------------------------------------------------------------

def _recursive_stretch(region: np.ndarray, p_percent: float) -> np.ndarray:
    """Recursively apply Algorithm A to sub-regions that are not homogeneous.

    Homogeneity criterion:  std(region) < (P/100) * mean(region)
    Stopping criterion:
      - Region is homogeneous, OR
      - Region dimensions < 10 x 10 pixels.

    When a region meets either stopping criterion, Algorithm A is applied to
    that region.

    Parameters
    ----------
    region : np.ndarray
        2-D uint8 greyscale array.
    p_percent : float
        Homogeneity threshold (percentage of mean).

    Returns
    -------
    np.ndarray
        Processed 2-D uint8 array of the same shape.
    """
    H, W = region.shape

    # Stopping criterion 1: too small
    if H < 10 or W < 10:
        return algorithm_a(region)

    pixels = region.astype(float).flatten()
    mean_val = pixels.mean()
    std_val = pixels.std()

    # Stopping criterion 2: homogeneous
    # Guard against zero mean (avoid division by zero).
    if mean_val == 0 or std_val < (p_percent / 100.0) * mean_val:
        return algorithm_a(region)

    # Not homogeneous and large enough — split into 4 and recurse
    mh = H // 2
    mw = W // 2

    result = region.copy()
    quarters = [
        (0,  mh, 0,  mw),
        (0,  mh, mw, W),
        (mh, H,  0,  mw),
        (mh, H,  mw, W),
    ]
    for r0, r1, c0, c1 in quarters:
        sub = region[r0:r1, c0:c1]
        if sub.size == 0:
            continue
        result[r0:r1, c0:c1] = _recursive_stretch(sub, p_percent)

    return result


def algorithm_d(roi: np.ndarray, p_percent: float) -> np.ndarray:
    """Algorithm D: Recursive local stretching (1D extra credit).

    Parameters
    ----------
    roi : np.ndarray
        2-D uint8 greyscale array.
    p_percent : float
        Homogeneity threshold P (%).

    Returns
    -------
    np.ndarray
        Processed 2-D uint8 array of the same shape.
    """
    return _recursive_stretch(roi, p_percent)
