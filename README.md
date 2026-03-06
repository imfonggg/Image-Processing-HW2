# Image-Processing-HW2

A small Python image processing project that lets you select up to 10 rectangular regions of interest (ROIs) from an input image, then runs histogram based processing and generates augmentation results for each ROI. Results are displayed in Matplotlib and can optionally be saved to disk.

## Features

Greyscale mode

- Select ROI(s) and generate a 12 image augmentation set for each ROI
- Optional before and after histogram visualization
- Optional recursive local stretching mode using a homogeneity threshold

Color mode

- Select ROI(s) and generate a 20 image augmentation set for each ROI

Input options

- Provide an image path via command line, or choose a file using a dialog
- Provide ROIs via command line, or draw them interactively in an OpenCV window

## Requirements

Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies include numpy, opencv-python, matplotlib, and Pillow.

## Usage

Run with an image path

```bash
python main.py --image path/to/image.jpg --mode grey
```

Open a file dialog to pick an image

```bash
python main.py
```

Modes

```bash
python main.py --mode grey
python main.py --mode color
python main.py --mode grey_recursive --p_percent 10
```

Use a parameter file (key=value per line)

```bash
python main.py --params sample_params.txt
```

CLI options override values loaded from the parameter file.

Provide ROIs without interactive selection

- Format is semicolon separated ROIs, each ROI is x,y,w,h

```bash
python main.py --image path/to/image.jpg --mode grey --rois "10,20,100,80;200,50,120,90"
```

Optional flags

```bash
python main.py --image path/to/image.jpg --mode grey --histograms
python main.py --image path/to/image.jpg --mode color --save_dir outputs
```

## Parameter file format

Use one `key=value` pair per line. Empty lines and lines starting with `#` are ignored.

Supported keys:

- `image`: input image path
- `mode`: `grey`, `color`, or `grey_recursive`
- `p_percent`: float value for Algorithm D homogeneity threshold (used in `grey_recursive`)
- `rois`: semicolon-separated ROI list, each ROI as `x,y,w,h`
- `histograms`: `true`/`false` (also accepts `1/0`, `yes/no`, `on/off`)
- `save_dir`: directory to save output figures

Example:

```text
# sample_params.txt
image=images/sample1.jpg
mode=grey
rois=10,20,120,120;220,40,130,110
histograms=true
save_dir=outputs/run1
```

## Files

- main.py: entry point, ROI selection, visualization, and running pipelines
- histogram.py: histogram based algorithms and greyscale augmentation helpers
- color_processing.py: color processing and augmentation helpers
- test_image_processing.py: tests
- requirements.txt: Python dependencies
