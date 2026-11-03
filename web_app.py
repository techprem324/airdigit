"""airdigit web server.

Serves the interface in web/ and exposes two JSON endpoints:

    GET  /api/status    is a trained model loaded, and which device it runs on
    POST /api/predict   {"image": "<data URL of a white-on-black PNG>"}

Run it with:  python web_app.py   then open http://localhost:8000
"""
import argparse
import base64
import json
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from preprocess import normalise, to_mnist

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
MAX_BODY = 2 * 1024 * 1024  # a 112x112 PNG is a few KB; this is generous

# Some Windows setups map .js to text/plain, which browsers refuse to run.
SimpleHTTPRequestHandler.extensions_map.update({".js": "text/javascript", ".css": "text/css"})


class Predictor:
    """Loads the trained CNN once and answers predictions. torch is imported lazily."""

    def __init__(self, model_path, device=None):
        self.path = Path(model_path)
        self.model = None
        self.meta = {}
        self.device = "cpu"
        self.error = None
        self._torch = None
        self._load(device)

    def _load(self, device):
        if not self.path.exists():
            self.error = (f"No trained model found at {self.path}. "
                          "Run `python train.py`, then restart the server.")
            return
        try:
            import torch
            from model import load_checkpoint
        except ImportError as exc:
            self.error = f"PyTorch is not installed ({exc}). Run `pip install -r requirements.txt`."
            return
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model, self.meta = load_checkpoint(self.path, self.device)
        except Exception as exc:  # corrupted or incompatible file
            self.error = f"Could not load {self.path.name}: {exc}"

    @property
    def ready(self) -> bool:
        return self.model is not None

    def status(self) -> dict:
        return {"ready": self.ready, "device": self.device, "message": self.error, "meta": self.meta}

    def predict(self, img28: np.ndarray) -> list:
        torch = self._torch
        x = torch.from_numpy(normalise(img28)).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        return probs.cpu().tolist()


class Handler(SimpleHTTPRequestHandler):
    predictor = None  # set by make_server

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    # Only log problems, not every static file.
    def log_request(self, code="-", size="-"):
        if str(code).isdigit() and int(code) >= 400:
            super().log_request(code, size)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Bypass-Tunnel-Reminder")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Bypass-Tunnel-Reminder")
        self.end_headers()

    def do_GET(self):
        clean_path = self.path.split("?")[0]
        if clean_path == "/api/status":
            return self.send_json(200, self.predictor.status())
        if clean_path == "/health":
            ready = self.predictor.ready
            return self.send_json(200 if ready else 503, {"ready": ready})
        super().do_GET()

    def do_POST(self):
        clean_path = self.path.split("?")[0]
        if clean_path != "/api/predict":
            return self.send_json(404, {"error": "Not found."})

        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            return self.send_json(400, {"error": "Send a PNG under 2 MB."})

        try:
            data_url = json.loads(self.rfile.read(length))["image"]
            raw = base64.b64decode(data_url.split(",", 1)[-1], validate=True)
            gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        except (ValueError, KeyError, TypeError):
            gray = None
        if gray is None:
            return self.send_json(400, {"error": "The image could not be read."})

        if not self.predictor.ready:
            return self.send_json(503, {"error": self.predictor.error})

        img28 = to_mnist(gray)
        if img28 is None:
            return self.send_json(422, {"error": "There is no ink in the image."})

        probs = self.predictor.predict(img28)
        top = int(np.argmax(probs))
        ok, png = cv2.imencode(".png", img28)
        self.send_json(200, {
            "prediction": top,
            "confidence": probs[top],
            "probabilities": [round(p, 4) for p in probs],
            "input": "data:image/png;base64," + base64.b64encode(png).decode("ascii") if ok else None,
        })


def make_server(predictor, host="0.0.0.0", port=8000) -> ThreadingHTTPServer:
    class Bound(Handler):
        pass

    Bound.predictor = predictor
    return ThreadingHTTPServer((host, port), Bound)


def main():
    ap = argparse.ArgumentParser(description="Run the airdigit web app.")
    ap.add_argument("--host", default="0.0.0.0", help="use 0.0.0.0 to open it to your local network")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=str(ROOT / "models" / "digit_cnn.pt"))
    ap.add_argument("--open", action="store_true", help="open the browser when the server starts")
    args = ap.parse_args()

    predictor = Predictor(args.model)
    if predictor.ready:
        meta = predictor.meta
        score = f", test accuracy {meta['test_acc'] * 100:.2f}%" if "test_acc" in meta else ""
        print(f"Model loaded on {predictor.device}{score}")
    else:
        print(f"Warning: {predictor.error}")

    try:
        server = make_server(predictor, args.host, args.port)
    except OSError:
        raise SystemExit(f"Port {args.port} is already in use. Try: python web_app.py --port {args.port + 1}")

    url = f"http://localhost:{args.port}"
    print(f"airdigit is running at {url}   (Ctrl+C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
