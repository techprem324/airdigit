<div align="center">

<img src="assets/banner.svg" alt="AirDigit — a little room to think" width="100%">

# AirDigit

### Draw a digit. Let a neural network take a guess.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CNN-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-1f4d43?style=flat-square)](LICENSE)

<br>

<img src="assets/demo.gif" alt="AirDigit drawing and recognizing the number seven" width="92%">

</div>

AirDigit is a small, browser-based MNIST experiment. Draw a number with a mouse or touchscreen, or try drawing in the air with a webcam. A compact PyTorch convolutional neural network predicts the digit and shows its confidence for each of the ten classes.

## Run locally

Python 3.12 is recommended.

```powershell
git clone https://github.com/techprem324/airdigit.git
cd airdigit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python web_app.py --host 127.0.0.1 --port 8000
```

Open [http://localhost:8000](http://localhost:8000). Keep the server running in the terminal while using the app. Stop it with `Ctrl+C`. If PowerShell blocks activation, run `.venv\Scripts\python.exe web_app.py --host 127.0.0.1 --port 8000` instead.

The webcam hand-tracking option needs camera permission and a secure context. `localhost` is allowed by browsers. Drawing with a mouse or touchscreen does not require the camera.

## How it works

1. The browser renders mouse, touch, or hand-tracked strokes into a white-on-black image.
2. `POST /api/predict` crops the strokes, scales and centers them to MNIST's 28 × 28 input, and runs the included CNN.
3. The page displays the predicted digit, ten class probabilities, and the normalized image the model received.

The API sends no drawing to a database. The image is submitted to the web server for inference and is not written to disk by the app.

The recorded MNIST training run reached **99.11% test accuracy**. That score is measured on the MNIST test set; freehand and air-drawn digits may be less accurate.

## Controls

| Action | Control |
| --- | --- |
| Draw | Click and drag, or touch and drag |
| Undo the last stroke | `Z` or **Undo** |
| Clear the canvas | `C` or **Clear** |
| Save the drawing | **Save image** |
| Air drawing | Select **Air draw**, allow camera access, and point with one finger |
| Clear in air-draw mode | Hold a closed fist until the progress ring completes |

## Model and training

The CNN uses four convolutional layers, batch normalization, two max-pooling stages, and a small classification head. Its weights are in `models/digit_cnn.pt`.

For training and data preparation:

```bash
pip install -r requirements.txt
python train.py --epochs 10
```

Training downloads MNIST through torchvision. It is not required to run the included web application.

Run the automated tests with:

```bash
python -m unittest discover -s tests -v
```

## Project layout

```text
airdigit/
├── assets/
│   ├── banner.svg             # Project cover artwork
│   ├── demo.gif               # Animated app preview
│   └── training_curves.png    # Curves from the included training log
├── models/
│   ├── digit_cnn.pt           # Pre-trained model
│   └── training_log.json      # Metadata and recorded training history
├── tests/
│   └── test_app.py            # Preprocessing and HTTP API tests
├── .github/
│   └── workflows/
│       └── test.yml           # Automated tests on GitHub Actions
├── web/
│   ├── app.js                 # Drawing, camera, and prediction UI
│   ├── index.html             # Accessible app page
│   └── style.css              # Responsive visual design
├── model.py                   # CNN definition and checkpoint loading
├── preprocess.py              # MNIST-compatible image preparation
├── train.py                   # MNIST training entry point
├── web_app.py                 # Web server and prediction API
├── requirements.txt           # Local and training dependencies
└── requirements-web.txt       # Lightweight web server and test dependencies
```

## License

This project is released under the [MIT License](LICENSE).
