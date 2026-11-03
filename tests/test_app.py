"""Tests that need no trained model and no GPU.  Run:  python -m unittest discover -s tests -v"""
import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import base64
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess import normalise, to_mnist  # noqa: E402
from web_app import make_server  # noqa: E402


def drawn_digit(size=112, thickness=8):
    """A white-on-black 'seven', like the picture the browser sends."""
    img = np.zeros((size, size), np.uint8)
    cv2.line(img, (16, 16), (96, 16), 255, thickness)
    cv2.line(img, (96, 16), (44, 98), 255, thickness)
    return img


def as_data_url(img):
    ok, png = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(png).decode()


class PreprocessTests(unittest.TestCase):
    def test_output_is_28x28_uint8(self):
        out = to_mnist(drawn_digit())
        self.assertEqual(out.shape, (28, 28))
        self.assertEqual(out.dtype, np.uint8)

    def test_digit_fits_20px_box_and_is_centred(self):
        out = to_mnist(drawn_digit())
        ys, xs = np.nonzero(out > 32)
        self.assertLessEqual(max(ys.max() - ys.min(), xs.max() - xs.min()) + 1, 22)
        m = cv2.moments(out)
        self.assertAlmostEqual(m["m10"] / m["m00"], 13.5, delta=1.0)
        self.assertAlmostEqual(m["m01"] / m["m00"], 13.5, delta=1.0)

    def test_size_of_the_drawing_does_not_matter(self):
        small = to_mnist(cv2.resize(drawn_digit(), (40, 40), interpolation=cv2.INTER_AREA))
        big = to_mnist(cv2.resize(drawn_digit(), (400, 400), interpolation=cv2.INTER_CUBIC))
        diff = np.abs(small.astype(int) - big.astype(int)).mean()
        self.assertLess(diff, 25)

    def test_blank_image_returns_none(self):
        self.assertIsNone(to_mnist(np.zeros((112, 112), np.uint8)))

    def test_normalise_shape_and_range(self):
        x = normalise(to_mnist(drawn_digit()))
        self.assertEqual(x.shape, (1, 1, 28, 28))
        self.assertAlmostEqual(float(x.min()), -0.1307 / 0.3081, places=3)


class FakePredictor:
    """Stands in for the CNN so the HTTP layer can be tested without torch."""

    def __init__(self, ready=True):
        self.ready = ready
        self.error = None if ready else "No trained model found."

    def status(self):
        return {"ready": self.ready, "device": "cpu", "message": self.error, "meta": {"test_acc": 0.995}}

    def predict(self, img28):
        probs = [0.01] * 10
        probs[7] = 0.91
        return probs


class ServerTests(unittest.TestCase):
    def start(self, predictor):
        server = make_server(predictor, "127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def post(self, base, payload, raw=None):
        data = raw if raw is not None else json.dumps(payload).encode()
        req = urllib.request.Request(base + "/api/predict", data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_serves_the_page(self):
        base = self.start(FakePredictor())
        with urllib.request.urlopen(base + "/") as r:
            self.assertIn(b"airdigit", r.read())
        with urllib.request.urlopen(base + "/app.js") as r:
            self.assertIn("javascript", r.headers["Content-Type"])

    def test_status(self):
        base = self.start(FakePredictor())
        with urllib.request.urlopen(base + "/api/status") as r:
            body = json.loads(r.read())
        self.assertTrue(body["ready"])
        self.assertEqual(body["meta"]["test_acc"], 0.995)

    def test_health_check_tracks_model_readiness(self):
        healthy = self.start(FakePredictor())
        with urllib.request.urlopen(healthy + "/health") as r:
            self.assertEqual(json.loads(r.read()), {"ready": True})

        unhealthy = self.start(FakePredictor(ready=False))
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(unhealthy + "/health")
        self.assertEqual(error.exception.code, 503)

    def test_predict_ok(self):
        base = self.start(FakePredictor())
        status, body = self.post(base, {"image": as_data_url(drawn_digit())})
        self.assertEqual(status, 200)
        self.assertEqual(body["prediction"], 7)
        self.assertEqual(len(body["probabilities"]), 10)
        self.assertTrue(body["input"].startswith("data:image/png;base64,"))

    def test_predict_without_model_is_503(self):
        base = self.start(FakePredictor(ready=False))
        status, body = self.post(base, {"image": as_data_url(drawn_digit())})
        self.assertEqual(status, 503)
        self.assertIn("No trained model", body["error"])

    def test_bad_requests(self):
        base = self.start(FakePredictor())
        self.assertEqual(self.post(base, None, raw=b"not json")[0], 400)
        self.assertEqual(self.post(base, {"image": "data:image/png;base64,AAAA"})[0], 400)
        self.assertEqual(self.post(base, {"image": as_data_url(np.zeros((112, 112), np.uint8))})[0], 422)


if __name__ == "__main__":
    unittest.main()
