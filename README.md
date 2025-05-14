# Harry Potter Invisibility Cloak

A computer vision project that creates a real-time invisibility cloak effect using OpenCV, similar to Harry Potter's invisibility cloak. The program uses color detection to make a blue-colored cloth appear invisible by replacing it with the background.

## Features

- Real-time invisibility effect using webcam
- Enhanced blue color detection for better results
- Smooth blending and edge processing
- Multiple blue shade detection (light and dark blue)
- High-quality camera settings
- Background averaging for stability

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/harry-potter-cloak.git
cd harry-potter-cloak
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the program:
```bash
python harry_potter_cloak.py
```

2. Follow the on-screen instructions:
   - Step away from the camera view during the 3-second countdown
   - After background capture, step into view wearing a blue cloth/cloak
   - The blue cloth will appear invisible, showing the background instead
   - Press 'q' to quit the program

## How it Works

1. The program captures a background image when you're not in the frame
2. It then detects blue-colored objects in real-time using HSV color space
3. The detected blue areas are replaced with the previously captured background
4. Advanced image processing techniques are used to create smooth transitions

## Tips for Best Results

- Use a solid blue cloth or blanket
- Ensure good lighting conditions
- Keep the background static
- Avoid wearing blue clothing other than the cloak
- Stand at a reasonable distance from the camera

## License

This project is open source and available under the MIT License. 