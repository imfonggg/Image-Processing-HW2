Image-Processing-HW2
====================

Project Overview
----------------
This project supports image augmentation based on histogram modification and color processing within user-defined rectangular Regions of Interest (ROIs). The program supports up to 10 ROIs per input image and includes both grayscale and color pipelines.

Implemented Functions
---------------------
1) Histogram modification (grayscale)
   - Algorithm A: Modified histogram stretching with Optimal Thresholding
   - Algorithm B: Local histogram stretching by splitting ROI into 4 quarters
   - Algorithm D (extra credit): Recursive local stretching using homogeneity criterion

2) Grayscale augmentation
   - Generates 12 images per ROI:
     - Original + rotations (90, 180, 270)
     - Algorithm A result + rotations
     - Algorithm B result + rotations

3) Color processing
   - Local color histogram modification using Algorithm A on:
     - R only, G only, B only, or all channels
   - Uses OpenCV BGR channel order internally: B=0, G=1, R=2

4) Color augmentation
   - Generates 20 images per ROI:
     - Original + rotations
     - A(R) + rotations
     - A(G) + rotations
     - A(B) + rotations
     - A(all channels) + rotations

Build / Execution Instructions
------------------------------
1) Install dependencies:
   pip install -r requirements.txt

2) Run with image path:
   python main.py --image path/to/image.jpg --mode grey

3) Run color mode:
   python main.py --image path/to/image.jpg --mode color

4) Run recursive grayscale (extra credit):
   python main.py --image path/to/image.jpg --mode grey_recursive --p_percent 10

5) Run without image argument (opens file dialog):
   python main.py

6) Provide ROIs by command line (skip interactive ROI selector):
   python main.py --image path/to/image.jpg --mode grey --rois "10,20,100,80;200,50,120,90"

7) Save output figures:
   python main.py --image path/to/image.jpg --mode color --save_dir outputs

8) Optional histogram visualization (grey mode):
   python main.py --image path/to/image.jpg --mode grey --histograms

9) Run using parameter file:
   python main.py --params sample_params.txt

10) Run automated tests:
    python -m pytest -v test_image_processing.py

Parameter File Format
---------------------
The program supports a text parameter file with one key=value pair per line.
Empty lines and lines starting with # are ignored.

Supported keys and meanings:
- image      : Input image path
- mode       : Processing mode (grey, color, or grey_recursive)
- p_percent  : Float homogeneity threshold P (%) for recursive mode
- rois       : Semicolon-separated ROI list, each ROI as x,y,w,h
- histograms : true/false (also accepts 1/0, yes/no, on/off)
- save_dir   : Output directory for saved result figures

Parameter order:
- No fixed order is required.

Example parameter file:
# sample_params.txt
image=images/sample1.jpg
mode=grey
rois=10,20,120,120;220,40,130,110
histograms=true
save_dir=outputs/run1

Project Files
-------------
- main.py                 : CLI entry point, ROI selection, display/saving pipelines
- histogram.py            : Optimal thresholding, Algorithm A/B/D, grayscale augmentation
- color_processing.py     : Color-channel processing and color augmentation
- test_image_processing.py: Unit tests
- requirements.txt        : Python package dependencies
- sample_params.txt       : Example parameter-file configuration

Notes
-----
- ROI processing supports 1 to 10 ROIs per image.
- In non-interactive environments, figures can be saved with --save_dir.
- Output images are suitable for report evidence and submission artifacts.
