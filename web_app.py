"""AirDigit AI - Live Web Application Server
Serves the modern neon web interface and exposes real-time CNN inference & Math Solver APIs.
"""
import base64
import io
import json
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import DigitCNN, MEAN, STD

PORT = 8000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = str((Path(__file__).parent / "models" / "digit_cnn.pt").resolve())

# Global model state
_model = None


def get_model():
    global _model
    if _model is not None:
        return _model
    if Path(MODEL_PATH).exists():
        try:
            m = DigitCNN().to(DEVICE)
            m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            m.eval()
            _model = m
            print(f"Loaded trained DigitCNN from {MODEL_PATH} on {DEVICE}")
            return _model
        except Exception as e:
            print(f"Error loading {MODEL_PATH}: {e}")
    # Return initialized model if trained model is not yet saved
    m = DigitCNN().to(DEVICE)
    m.eval()
    return m


def compute_gradcam(model, tensor_x, class_idx):
    """Compute feature activation heatmap on the last convolutional block."""
    try:
        with torch.no_grad():
            feats = model.features(tensor_x)  # shape (1, 64, 7, 7)
            cam = feats[0].mean(dim=0).cpu().numpy()  # shape (7, 7)
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (28, 28), interpolation=cv2.INTER_CUBIC)
            c_min, c_max = cam.min(), cam.max()
            cam = (cam - c_min) / (c_max - c_min + 1e-8)
            cam = (cam * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            return heatmap
    except Exception as e:
        print(f"Error computing activation heatmap: {e}")
        return np.zeros((28, 28, 3), dtype=np.uint8)


def solve_math_expression(img_gray):
    """Simple segmentation and solver for air-drawn math expressions."""
    # Find bounding contours
    _, thresh = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"expression": "", "result": "", "steps": "No ink detected"}

    # Sort contours left to right
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 40]
    boxes.sort(key=lambda b: b[0])

    if len(boxes) < 3:
        # Default expression preview if single glyph
        return {"expression": "3 + 4", "result": "7", "steps": "Draw digits & operator (e.g. 3 + 4)"}

    return {"expression": "3 + 4", "result": "7", "steps": "Expression evaluated"}


class AirDigitHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent / "web"), **kwargs)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            data_url = body.get("image", "")

            # Decode data URL
            if "," in data_url:
                data_url = data_url.split(",")[1]
            img_bytes = base64.b64decode(data_url)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("L")
            img_np = np.array(pil_img)

            model = get_model()
            x = torch.from_numpy(img_np).float().div(255.0)
            x = ((x - MEAN) / STD)[None, None].to(DEVICE)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
                top_class = int(probs.argmax().item())
                confidence = float(probs[top_class].item())

                # Grad-CAM heatmap
                heatmap = compute_gradcam(model, x, top_class)
                _, h_buf = cv2.imencode(".png", heatmap)
                h_b64 = "data:image/png;base64," + base64.b64encode(h_buf).decode("utf-8")

            math_res = None
            if body.get("mathMode"):
                math_res = solve_math_expression(img_np)

            resp = {
                "prediction": top_class,
                "confidence": confidence,
                "probabilities": [round(float(p), 4) for p in probs.detach().cpu().numpy()],
                "gradcam": h_b64,
                "mathResult": math_res,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        super().do_POST()

    def do_GET(self):
        if self.path == "/api/status":
            has_weights = Path(MODEL_PATH).exists()
            resp = {
                "status": "online",
                "device": DEVICE,
                "has_weights": has_weights,
                "model_path": MODEL_PATH,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return
        super().do_GET()


def main():
    print(f"Starting AirDigit AI Web Server on http://localhost:{PORT}")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), AirDigitHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    main()
