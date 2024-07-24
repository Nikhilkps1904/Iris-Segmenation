# Iris Detection and Segmentation

This repository contains a Python script for iris detection and segmentation in eye images. The script uses OpenCV to process images, detect the iris and pupil, and segment the iris region.

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed Python 3.6 or later
* You have a Windows/Linux/Mac machine

## Installing Iris Detection and Segmentation

To install the required libraries, follow these steps:

1. Clone this repository to your local machine:
   ```
   git clone https://github.com/Nikhilkps1904/Iris-Segmenation.git
   cd  Iris-Segmenation
   ```

2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install opencv-python numpy matplotlib
   ```

## Using Iris Detection and Segmentation

To use the iris detection script, follow these steps:

1. Place your eye image in the same directory as the script, or update the file path in the `main()` function.

2. Run the script:
   ```
   python main.py 
   ```
  or
  ```
  python main.ipynb
  ```
3. The script will display four images:
   - Original Grayscale
   - Filtered Image
   - Detected Iris and Pupil
   - Segmented Iris

## Customizing the Script

You can customize the script by modifying the following parameters:

- In the `detect_iris_and_pupil()` function:
  - Adjust the `param2` range in the for loop to fine-tune circle detection
  - Modify `minRadius` and `maxRadius` to change the size range of detected circles

- In the `advanced_preprocess()` function:
  - Adjust the CLAHE parameters (`clipLimit` and `tileGridSize`)
  - Modify the bilateral filter and blur parameters

## Troubleshooting

If the script fails to detect the iris or pupil, it will display diagnostic information to help you understand why. Check the console output for any error messages.

## Contact

If you want to contact me, you can reach me at `<nikhilkps1538@gmail.com>`.

## License

This project uses the following license: [MIT License](<link_to_license>).
