"""Automated verification and test suite for AirDigit AI."""
import base64
import io
import json
import urllib.request
import numpy as np
import torch
import cv2
from PIL import Image

from model import DigitCNN, MEAN, STD
from app import preprocess, fingers_up

class MockLandmark:
    def __init__(self, y):
        self.y = y

def test_model_architecture():
    m = DigitCNN()
    x = torch.randn(2, 1, 28, 28)
    out = m(x)
    assert out.shape == (2, 10), f"Expected shape (2, 10), got {out.shape}"
    print("[PASS] DigitCNN architecture test passed!")

def test_model_weights():
    m = DigitCNN()
    m.load_state_dict(torch.load("models/digit_cnn.pt", map_location="cpu"))
    m.eval()
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        out = m(x)
        probs = torch.softmax(out, dim=1)
    assert probs.shape == (1, 10)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)
    print("[PASS] Model weights loading and inference test passed!")

def test_preprocess_pipeline():
    # Synthetic canvas with a drawn box
    canvas = np.zeros((600, 800), dtype=np.uint8)
    cv2.circle(canvas, (400, 300), 50, 255, 18)
    res = preprocess(canvas)
    assert res is not None, "Preprocessing returned None on valid stroke"
    assert res.shape == (28, 28), f"Expected shape (28, 28), got {res.shape}"
    print("[PASS] Preprocessing pipeline test passed!")

def test_gestures():
    # Case 1: Index up, middle/ring/pinky down
    # tips=(8, 12, 16, 20), pips=(6, 10, 14, 18)
    lm = [MockLandmark(0.5)] * 21
    # Index extended: lm[8].y < lm[6].y - 0.02
    lm[6] = MockLandmark(0.5)
    lm[8] = MockLandmark(0.3)   # extended
    lm[10] = MockLandmark(0.5)
    lm[12] = MockLandmark(0.6)  # curled
    lm[14] = MockLandmark(0.5)
    lm[16] = MockLandmark(0.6)  # curled
    lm[18] = MockLandmark(0.5)
    lm[20] = MockLandmark(0.6)  # curled

    up = fingers_up(lm)
    assert up == [1, 0, 0, 0], f"Expected index up [1,0,0,0], got {up}"
    print("[PASS] Gesture state machine test passed!")

def test_web_api():
    # 1. Test /api/status
    req = urllib.request.urlopen("http://localhost:8000/api/status")
    status = json.loads(req.read().decode())
    assert status.get("status") == "online"
    assert status.get("has_weights") is True
    print(f"[PASS] Web API Status test passed: {status}")

    # 2. Test /api/predict
    dummy = np.zeros((28, 28), dtype=np.uint8)
    cv2.putText(dummy, "7", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    pil = Image.fromarray(dummy)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    post_data = json.dumps({"image": b64, "mathMode": True}).encode()
    req = urllib.request.Request(
        "http://localhost:8000/api/predict",
        data=post_data,
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    pred = json.loads(res.read().decode())
    assert "prediction" in pred
    assert "probabilities" in pred
    assert len(pred["probabilities"]) == 10
    assert "gradcam" in pred
    print(f"[PASS] Web API Prediction & Grad-CAM test passed! Top prediction: {pred['prediction']}, Conf: {pred['confidence']:.2f}")

if __name__ == "__main__":
    print("\n--- Running AirDigit AI Test Suite ---")
    test_model_architecture()
    test_model_weights()
    test_preprocess_pipeline()
    test_gestures()
    test_web_api()
    print("\n==========================================")
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("==========================================\n")
