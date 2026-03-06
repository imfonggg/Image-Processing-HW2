"""
main.py — Image Processing HW2 interactive application.

Usage
-----
    python main.py [--image PATH] [--mode {grey,color}]

If --image is omitted the script will open a file-chooser dialog.

Workflow
--------
1. Load the image.
2. The user draws up to 10 rectangular ROIs by clicking-and-dragging
   on the displayed image.  Press ENTER / SPACE to confirm the current
   ROI set, or press 'q' to quit without processing.
3. For each ROI the program:
   - (grey mode) runs Algorithm A, B and generates the 12-image
     augmentation set (1A, 1B, 1C).
   - (color mode) runs Algorithm A on individual channels and all
     channels, then generates the 20-image augmentation set (3A, 3B).
4. Results are displayed in a Matplotlib figure grid and optionally
   saved to an output directory.

Keyboard shortcuts while selecting ROIs
----------------------------------------
ENTER or SPACE : process the selected ROIs
c              : clear all selected ROIs
q              : quit

Extra-credit 1D (recursive stretching)
---------------------------------------
Pass --mode grey_recursive --p_percent <value> to enable recursive
local stretching (Algorithm D).
"""

import argparse
import os
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from histogram import algorithm_a, algorithm_b, algorithm_d, augment_greyscale
from color_processing import apply_algorithm_a_color, augment_color


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_ROIS = 10
SUPPORTED_MODES = {"grey", "color", "grey_recursive"}


def load_image(path: str) -> np.ndarray:
    """Load image from *path* using OpenCV (returns BGR uint8)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def parse_bool(value: str) -> bool:
    """Parse a text boolean value used in parameter files."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_params_file(params_path: str) -> dict:
    """Parse key=value parameters from a text file.

    Supported keys: image, mode, p_percent, rois, histograms, save_dir.
    Lines starting with '#' or empty lines are ignored.
    """
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameter file not found: {params_path}")

    config = {}
    supported_keys = {"image", "mode", "p_percent", "rois", "histograms", "save_dir"}

    with open(params_path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                raise ValueError(
                    f"Invalid parameter line {line_no} in {params_path}: '{raw_line.rstrip()}'. "
                    "Expected format key=value."
                )

            key, value = [part.strip() for part in line.split("=", 1)]
            if key not in supported_keys:
                raise ValueError(
                    f"Unknown parameter key '{key}' on line {line_no}. "
                    f"Supported keys: {', '.join(sorted(supported_keys))}."
                )

            if key == "mode":
                value = value.lower()
                if value not in SUPPORTED_MODES:
                    raise ValueError(
                        f"Invalid mode '{value}' on line {line_no}. "
                        f"Expected one of: {', '.join(sorted(SUPPORTED_MODES))}."
                    )
            elif key == "p_percent":
                value = float(value)
            elif key == "histograms":
                value = parse_bool(value)

            config[key] = value

    return config


def resolve_config(args: argparse.Namespace, params: dict) -> dict:
    """Merge defaults, parameter file values, and CLI overrides."""
    mode = args.mode if args.mode is not None else params.get("mode", "grey")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Expected one of {sorted(SUPPORTED_MODES)}")

    return {
        "image": args.image if args.image is not None else params.get("image"),
        "mode": mode,
        "p_percent": (
            args.p_percent if args.p_percent is not None else params.get("p_percent", 10.0)
        ),
        "rois": args.rois if args.rois is not None else params.get("rois"),
        "histograms": (
            args.histograms if args.histograms is not None else params.get("histograms", False)
        ),
        "save_dir": args.save_dir if args.save_dir is not None else params.get("save_dir"),
    }


def select_rois_interactively(image_bgr: np.ndarray) -> list:
    """Open an OpenCV window and let the user draw up to MAX_ROIS ROIs.

    Returns
    -------
    list of (x, y, w, h) tuples in image coordinates.
    """
    rois = []
    window = "Select ROIs (press ENTER/SPACE to confirm, 'c' to clear, 'q' to quit)"

    # OpenCV selectROIs returns an array of (x, y, w, h)
    cv_rois = cv2.selectROIs(window, image_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    for roi in cv_rois[:MAX_ROIS]:
        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        if w > 0 and h > 0:
            rois.append((x, y, w, h))
    return rois


def extract_roi(image: np.ndarray, roi: tuple) -> np.ndarray:
    """Extract the ROI sub-array from *image*.

    Parameters
    ----------
    image : np.ndarray  (H, W) or (H, W, C)
    roi   : (x, y, w, h)
    """
    x, y, w, h = roi
    return image[y: y + h, x: x + w]


def display_images(images: list, title: str = "", save_dir: str = None):
    """Display a list of (array, label) pairs in a Matplotlib grid.

    Parameters
    ----------
    images   : list of (np.ndarray, str)
    title    : overall figure title
    save_dir : if provided, save the figure as a PNG file there
    """
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for ax, (img, label) in zip(axes, images):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            # Convert BGR → RGB for display
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(label, fontsize=8)
        ax.axis("off")

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "-")
        path = os.path.join(save_dir, f"{safe_title}.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")

    plt.show()


def display_histograms(roi_grey: np.ndarray, processed: np.ndarray, title: str = ""):
    """Show before/after histograms for a greyscale ROI (optional debug)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(roi_grey.flatten(), bins=256, range=(0, 255), color="blue", alpha=0.7)
    axes[0].set_title("Before")
    axes[0].set_xlabel("Intensity")
    axes[0].set_ylabel("Count")

    axes[1].hist(processed.flatten(), bins=256, range=(0, 255), color="orange", alpha=0.7)
    axes[1].set_title("After")
    axes[1].set_xlabel("Intensity")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Processing pipelines
# ---------------------------------------------------------------------------

def process_grey_roi(
    roi_grey: np.ndarray,
    roi_index: int,
    p_percent: float = None,
    show_histograms: bool = False,
    save_dir: str = None,
):
    """Run greyscale augmentation pipeline for one ROI.

    Generates 12 augmented images (1C).  If p_percent is provided,
    also runs Algorithm D and shows its result (1D).
    """
    print(f"\n--- ROI {roi_index + 1} (grey, {roi_grey.shape}) ---")

    # 1C: 12-image augmentation
    images = augment_greyscale(roi_grey)
    display_images(
        images,
        title=f"ROI {roi_index + 1} — Grey Augmentation (12 images)",
        save_dir=save_dir,
    )

    # Optional: show histograms for Algorithm A result
    if show_histograms:
        a_proc = algorithm_a(roi_grey)
        display_histograms(
            roi_grey, a_proc,
            title=f"ROI {roi_index + 1} — Histogram before/after Algorithm A",
        )

    # 1D extra credit
    if p_percent is not None:
        print(f"  Running Algorithm D (recursive, P={p_percent}%)…")
        d_proc = algorithm_d(roi_grey, p_percent)
        display_images(
            [(d_proc, f"Alg-D (P={p_percent}%)")],
            title=f"ROI {roi_index + 1} — Recursive Stretching (Alg-D)",
            save_dir=save_dir,
        )


def process_color_roi(
    roi_bgr: np.ndarray,
    roi_index: int,
    save_dir: str = None,
):
    """Run colour augmentation pipeline for one ROI.

    Generates 20 augmented images (3B).
    """
    print(f"\n--- ROI {roi_index + 1} (colour, {roi_bgr.shape}) ---")
    images = augment_color(roi_bgr)
    display_images(
        images,
        title=f"ROI {roi_index + 1} — Color Augmentation (20 images)",
        save_dir=save_dir,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Image Processing HW2 — Histogram & Color Augmentation"
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help=(
            "Path to a parameter file (key=value per line). "
            "CLI arguments override values from this file."
        ),
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=sorted(SUPPORTED_MODES),
        default=None,
        help=(
            "Processing mode: 'grey' (1A–1C), 'color' (3A–3B), "
            "or 'grey_recursive' (1D extra credit)."
        ),
    )
    _P_DEFAULT = 10.0
    parser.add_argument(
        "--p_percent", "-p",
        type=float,
        default=None,
        help=(
            "Homogeneity threshold P (%%) for recursive stretching "
            f"(only used in grey_recursive mode). Default: {_P_DEFAULT}"
        ),
    )
    parser.add_argument(
        "--rois",
        type=str,
        default=None,
        help=(
            "Comma-separated list of ROIs as x,y,w,h tuples, e.g. "
            "'10,20,100,80;200,50,120,90'.  If omitted, an interactive "
            "OpenCV ROI selector is opened."
        ),
    )
    parser.add_argument(
        "--histograms",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show before/after histograms for each ROI (grey mode).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save output figures. If omitted, figures are only displayed.",
    )
    return parser.parse_args()


def parse_rois_string(rois_str: str) -> list:
    """Parse a semicolon-separated list of 'x,y,w,h' ROI specs."""
    rois = []
    for part in rois_str.split(";"):
        part = part.strip()
        if not part:
            continue
        vals = [int(v) for v in part.split(",")]
        if len(vals) != 4:
            raise ValueError(f"Invalid ROI spec '{part}': expected x,y,w,h")
        rois.append(tuple(vals))
    return rois[:MAX_ROIS]


def main():
    args = parse_args()
    params = parse_params_file(args.params) if args.params else {}
    config = resolve_config(args, params)

    # ---- Load image ----
    if config["image"] is None:
        # Try a simple Tkinter file dialog if available
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                title="Select input image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
            )
            root.destroy()
            if not path:
                print("No image selected. Exiting.")
                sys.exit(0)
        except Exception:
            print("No --image argument provided and no file dialog available. Exiting.")
            sys.exit(1)
    else:
        path = config["image"]

    print(f"Loading image: {path}")
    image_bgr = load_image(path)
    print(f"  Image size: {image_bgr.shape[1]} x {image_bgr.shape[0]}")

    # ---- Select ROIs ----
    if config["rois"]:
        rois = parse_rois_string(config["rois"])
        print(f"Using {len(rois)} ROI(s) from configuration (--rois or --params).")
    else:
        print("Opening interactive ROI selector…")
        print("  Draw ROI(s) by clicking and dragging.")
        print("  Press ENTER/SPACE to confirm, 'c' to clear, 'q' to quit.")
        rois = select_rois_interactively(image_bgr)

    if not rois:
        print("No ROIs selected. Exiting.")
        sys.exit(0)

    print(f"Processing {len(rois)} ROI(s)…")

    # ---- Process each ROI ----
    for i, roi_coords in enumerate(rois):
        x, y, w, h = roi_coords
        # Clamp to image boundaries
        H_img, W_img = image_bgr.shape[:2]
        x = max(0, min(x, W_img - 1))
        y = max(0, min(y, H_img - 1))
        w = max(1, min(w, W_img - x))
        h = max(1, min(h, H_img - y))

        roi_bgr = extract_roi(image_bgr, (x, y, w, h))

        if config["mode"] in ("grey", "grey_recursive"):
            roi_grey = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            p = config["p_percent"] if config["mode"] == "grey_recursive" else None
            process_grey_roi(
                roi_grey,
                roi_index=i,
                p_percent=p,
                show_histograms=config["histograms"],
                save_dir=config["save_dir"],
            )
        else:  # color
            process_color_roi(
                roi_bgr,
                roi_index=i,
                save_dir=config["save_dir"],
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
