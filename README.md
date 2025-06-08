# CAPTCHA Prediction

This project implements a CAPTCHA recognition system using deep learning. It processes CAPTCHA images, segments individual characters, and predicts them using a trained neural network model.

## Features

- Image preprocessing and cleaning
- Character segmentation
- Deep learning-based character recognition
- Support for numerical CAPTCHAs

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- Keras
- PIL (Python Imaging Library)

## Usage

1. Place your CAPTCHA image as `img.png` in the project directory
2. Run the script:
```bash
python final.py
```

The script will:
1. Clean and preprocess the image
2. Segment individual characters
3. Predict each character using the trained model
4. Output the predicted sequence

## Model

The project uses a trained Keras model (`my_model.keras`) for character recognition. The model is trained to recognize numerical digits.

## Note

Make sure to have the required model file (`my_model.keras`) in the project directory before running the script. 