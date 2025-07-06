# Fish Scale Analysis Tool

A Python-based tool for analyzing fish scales and measuring growth rings using computer vision techniques. This tool provides both automatic and manual measurement capabilities for fish scale analysis in fisheries research.

## Features

- **Automatic Scale Detection**: Uses computer vision to automatically detect scale boundaries and measurement lines
- **Manual Mode**: Allows manual placement of measurement points when automatic detection fails
- **Interactive Interface**: Real-time visualization with zoom, pan, and point placement capabilities
- **Multiple Measurement Modes**: 
  - Automatic edge detection
  - Manual scale definition
  - Manual measurement line placement
- **Data Export**: Saves measurements to CSV format with detailed metadata
- **Batch Processing**: Process multiple images sequentially
- **Scale Calibration**: Automatic and manual scale calibration (pixels per mm)

## Requirements

- Python 3.7+
- OpenCV 4.x
- NumPy
- Standard Python libraries (os, csv, datetime, dataclasses, typing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BDX786/fish-scale-analysis.git
cd fish-scale-analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
```

3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`

4. Install dependencies:
```bash
pip install opencv-python numpy
```

## Usage

### Basic Usage

1. Place your fish scale images in the `Test images` folder
2. Run the program:
```bash
python Poissons.py
```

3. Follow the on-screen instructions to analyze each image

### Controls

#### Mouse Controls
- **Left Click**: Place center point or measurement points
- **Mouse Wheel**: Zoom in/out
- **Mouse Movement**: Pan when zoomed in

#### Keyboard Controls
- **'v'**: Validate current point and move to next step
- **'f'**: Finish and save measurements (moves to next image)
- **'u'**: Undo last point/validation
- **'a'/'d'**: Adjust measurement line angle (left/right)
- **'m'**: Toggle manual mode (disable edge detection)
- **'s'**: Switch to manual scale mode (define scale manually)
- **'b'**: Go to previous image
- **'n'**: Go to next image
- **'q'**: Quit program

### Workflow

1. **Scale Detection**: The program attempts automatic scale detection
2. **Center Point**: Click to place the center point of the scale
3. **Validation**: Press 'v' to validate the center point
4. **Measurement Points**: Click along the measurement line to place points
5. **Validation**: Press 'v' after each point to validate
6. **Finish**: Press 'f' to save measurements and move to next image

### Modes

#### Automatic Mode
- Automatically detects scale boundaries using edge detection
- Automatically detects scale calibration marks
- Measurement line extends to detected scale edge

#### Manual Mode ('m' key)
- Disables automatic edge detection
- Allows placement of measurement line to image boundaries
- Useful when automatic detection fails

#### Manual Scale Mode ('s' key)
- Click two points 3.33mm apart to define scale
- Overrides automatic scale detection
- Useful for images with unclear scale marks

## Output

### Files Generated

- **Processed/**: Contains processed images with measurements overlaid
- **Treated/**: Original images moved after processing
- **scale_measurements.csv**: CSV file with all measurements

### CSV Format

The output CSV contains the following columns:
- `Image_Name`: Name of the processed image
- `Fish_ID`: Extracted from image filename (format: fishid_scalenumber)
- `Scale_Number`: Scale number from image filename
- `Date`: Processing date
- `Time`: Processing time
- `Scale_px_per_mm`: Scale calibration (pixels per millimeter)
- `Center_to_First_Point_mm`: Distance from center to first measurement point
- `Point1_to_Point2_mm`: Distance between consecutive measurement points
- `Point2_to_Point3_mm`: Distance between consecutive measurement points
- ... (up to 10 measurement points)
- `Total_Length_mm`: Total measurement line length

## File Structure

```
fish-scale-analysis/
├── Poissons.py              # Main analysis program
├── README.md                # This file
├── Test images/             # Input images folder
├── Processed/               # Output processed images
├── Treated/                 # Moved original images
├── scale_measurements.csv   # Measurement results
└── .venv/                   # Virtual environment (if used)
```

## Image Requirements

- **Format**: PNG, JPG, JPEG, BMP
- **Naming**: `fishid_scalenumber.ext` (e.g., `9001_1.png`)
- **Quality**: Clear, well-lit images with visible scale structure
- **Scale**: Should include calibration marks or known reference distance

## Troubleshooting

### Common Issues

1. **"No module named 'cv2'"**
   - Install OpenCV: `pip install opencv-python`

2. **Automatic detection fails**
   - Try manual mode ('m' key)
   - Ensure good image quality and lighting
   - Check that scale boundaries are clearly visible

3. **Scale calibration issues**
   - Use manual scale mode ('s' key)
   - Click two points exactly 3.33mm apart

4. **Program crashes on image load**
   - Check image format and file integrity
   - Ensure images are not corrupted

### Tips for Best Results

- Use high-quality, well-lit images
- Ensure scale boundaries are clearly visible
- Place center point accurately at scale center
- Use manual mode for difficult images
- Calibrate scale manually if automatic detection fails

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- Cassandra Wicht for her proper explanations of the procedures to automate and methodoology.
- OpenCV community for computer vision capabilities

## Contact

For questions or issues, please open an issue on GitHub or contact Hugo Bordereaux.

---

**Note**: This tool is designed for fisheries research and fish scale analysis. Ensure proper training and validation when using for scientific research. 
