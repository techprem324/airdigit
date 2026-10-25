"""AirDigit AI - draw digits in the air with your index finger, a CNN reads them in real time.

Gestures
    Index finger up            -> DRAW
    Index + middle finger up   -> HOVER (move without drawing)
    Closed fist (hold 0.8 s)   -> CLEAR canvas
Keys: [c] clear   [s] save screenshot   [q] quit
"""
import argparse
import math
import random
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from model import DigitCNN, MEAN, STD

W, H, PANEL_W = 800, 600, 380
INK_THICKNESS = 22
PANEL_BG = (30, 20, 16)
ACCENT = (255, 210, 0)        # BGR
CORE = (255, 245, 190)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_B = cv2.FONT_HERSHEY_DUPLEX


# --------------------------------------------------------------------------- model helpers
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(path: str, device: str) -> DigitCNN:
    if not Path(path).exists():
        raise SystemExit(f"Model file '{path}' not found. Run:  python train.py")
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def preprocess(canvas: np.ndarray):
    """Turn the drawn canvas into an MNIST-style 28x28 image (crop, scale to 20px, centre by mass)."""
    ys, xs = np.nonzero(canvas > 40)
    if len(xs) < 80:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = canvas[y0:y1 + 1, x0:x1 + 1]
    h, w = crop.shape
    if max(h, w) < 40:
        return None
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    img = np.zeros((28, 28), np.uint8)
    oy, ox = (28 - nh) // 2, (28 - nw) // 2
    img[oy:oy + nh, ox:ox + nw] = crop
    m = cv2.moments(img)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        img = cv2.warpAffine(img, np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]]), (28, 28))
    return img


@torch.no_grad()
def predict(model, img: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(img).float().div(255.0)
    x = ((x - MEAN) / STD)[None, None].to(device)
    return torch.softmax(model(x), dim=1)[0].cpu().numpy()


# --------------------------------------------------------------------------- hand helpers
def fingers_up(lm) -> list:
    """[index, middle, ring, pinky] -> 1 if the finger is extended (assumes upright hand)."""
    tips, pips = (8, 12, 16, 20), (6, 10, 14, 18)
    return [int(lm[t].y < lm[p].y - 0.02) for t, p in zip(tips, pips)]


# --------------------------------------------------------------------------- drawing helpers
def put(img, text, org, scale=0.5, color=(220, 220, 220), thick=1, font=FONT):
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


def draw_panel(disp, preview, mode, fps, clear_progress, has_ink):
    panel = np.full((H, PANEL_W, 3), PANEL_BG, np.uint8)
    cv2.line(panel, (0, 0), (0, H), ACCENT, 2)
    put(panel, "AIRDIGIT AI", (24, 42), 0.9, ACCENT, 2, FONT_B)
    put(panel, "real-time CNN digit recognition", (24, 66), 0.45, (150, 150, 150))

    top = int(np.argmax(disp))
    conf = float(disp[top])
    if has_ink and conf > 0.05:
        put(panel, str(top), (28, 200), 5.0, CORE, 8, FONT_B)
        put(panel, f"{conf * 100:5.1f}%", (130, 180), 0.9, ACCENT, 2, FONT_B)
        put(panel, "confidence", (132, 202), 0.45, (150, 150, 150))
    else:
        put(panel, "?", (40, 200), 5.0, (80, 70, 60), 8, FONT_B)
        put(panel, "draw a digit", (130, 180), 0.6, (150, 150, 150))

    # What the network actually sees (28x28 input)
    prev = cv2.resize(preview, (112, 112), interpolation=cv2.INTER_NEAREST)
    prev = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)
    panel[90:202, 250:362] = prev
    cv2.rectangle(panel, (250, 90), (362, 202), (90, 80, 70), 1)
    put(panel, "model input", (262, 218), 0.4, (150, 150, 150))

    # Animated probability bars
    y0, bar_max = 245, PANEL_W - 120
    for i in range(10):
        y = y0 + i * 26
        is_top = (i == top and has_ink and conf > 0.05)
        col = ACCENT if is_top else (110, 95, 80)
        put(panel, str(i), (24, y + 14), 0.55, CORE if is_top else (170, 170, 170), 1, FONT_B)
        cv2.rectangle(panel, (50, y), (50 + bar_max, y + 16), (50, 40, 34), -1)
        cv2.rectangle(panel, (50, y), (50 + int(disp[i] * bar_max), y + 16), col, -1)
        put(panel, f"{disp[i] * 100:4.1f}", (58 + bar_max, y + 13), 0.4, (170, 170, 170))

    # Status
    status_col = {"DRAWING": (120, 255, 120), "HOVER": (0, 200, 255), "FIST": (80, 80, 255)}.get(mode, (150, 150, 150))
    put(panel, f"MODE  {mode}", (24, 530), 0.6, status_col, 1, FONT_B)
    put(panel, f"FPS   {fps:4.1f}", (24, 552), 0.5, (170, 170, 170))
    if clear_progress > 0:
        cv2.rectangle(panel, (150, 540), (350, 556), (50, 40, 34), -1)
        cv2.rectangle(panel, (150, 540), (150 + int(200 * clear_progress), 556), (80, 80, 255), -1)
        put(panel, "clearing...", (150, 534), 0.4, (80, 80, 255))
    put(panel, "1 finger: draw | 2 fingers: hover | fist: clear", (24, 580), 0.4, (120, 120, 120))
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default="models/digit_cnn.pt")
    args = ap.parse_args()

    device = pick_device()
    model = load_model(args.model, device)
    hands = mp.solutions.hands.Hands(
        max_num_hands=1, model_complexity=0,
        min_detection_confidence=0.7, min_tracking_confidence=0.6,
    )
    connections = mp.solutions.hands.HAND_CONNECTIONS

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try --camera 1")

    canvas = np.zeros((H, W), np.uint8)
    probs, disp = np.zeros(10), np.zeros(10)
    preview = np.zeros((28, 28), np.uint8)
    particles = []
    prev, fist_t = None, None
    ink_dirty, has_ink = False, False
    frame_i, fps, t_prev = 0, 0.0, time.time()
    Path("assets").mkdir(exist_ok=True)

    def clear_all():
        nonlocal prev, ink_dirty, has_ink
        canvas[:] = 0
        probs[:] = 0
        preview[:] = 0
        prev, ink_dirty, has_ink = None, False, False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(cv2.flip(frame, 1), (W, H))
        now = time.time()
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        mode, pts, clear_progress = "NO HAND", None, 0.0
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            pts = [(int(p.x * W), int(p.y * H)) for p in lm]
            up = fingers_up(lm)
            if up == [1, 0, 0, 0]:
                mode = "DRAWING"
            elif up[0] and up[1]:
                mode = "HOVER"
            elif not any(up):
                mode = "FIST"
            else:
                mode = "IDLE"

        # --- drawing logic
        if mode == "DRAWING":
            tip = pts[8]
            pos = tip if prev is None else (
                int(prev[0] + 0.55 * (tip[0] - prev[0])),
                int(prev[1] + 0.55 * (tip[1] - prev[1])),
            )
            if prev is not None:
                cv2.line(canvas, prev, pos, 255, INK_THICKNESS, cv2.LINE_AA)
                ink_dirty = has_ink = True
            prev = pos
            for _ in range(2):
                a, s = random.uniform(0, 2 * math.pi), random.uniform(1, 4)
                particles.append([pos[0], pos[1], math.cos(a) * s, math.sin(a) * s, 1.0])
        else:
            prev = None

        # --- fist hold = clear
        if mode == "FIST":
            if fist_t is None:
                fist_t = now
            if now - fist_t > 0.8:
                clear_all()
                fist_t = float("inf")       # wait until the fist is released
            elif fist_t != float("inf"):
                clear_progress = (now - fist_t) / 0.8
        else:
            fist_t = None

        # --- inference (every 3rd frame, only when ink changed)
        if ink_dirty and frame_i % 3 == 0:
            img = preprocess(canvas)
            if img is not None:
                preview = img
                probs = predict(model, img, device)
            else:
                probs = np.zeros(10)
            ink_dirty = False
        disp += 0.2 * (probs - disp)          # smooth animated bars

        # --- render camera view
        glow = cv2.GaussianBlur(canvas, (0, 0), 9)
        layer = np.zeros_like(frame)
        layer[..., 0] = glow
        layer[..., 1] = (glow * 0.8).astype(np.uint8)
        layer[..., 2] = (glow * 0.1).astype(np.uint8)
        view = cv2.add(cv2.convertScaleAbs(frame, alpha=0.55), layer)
        view[canvas > 128] = CORE

        # particles
        alive = []
        for p in particles:
            p[0] += p[2]; p[1] += p[3]; p[4] -= 0.06
            if p[4] > 0:
                col = tuple(int(c * p[4]) for c in ACCENT)
                cv2.circle(view, (int(p[0]), int(p[1])), max(1, int(3 * p[4])), col, -1, cv2.LINE_AA)
                alive.append(p)
        particles = alive

        # hand skeleton + pulsing fingertip
        if pts:
            for a, b in connections:
                cv2.line(view, pts[a], pts[b], (120, 100, 70), 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(view, p, 3, (200, 180, 140), -1, cv2.LINE_AA)
            r = int(12 + 4 * math.sin(now * 8))
            ring_col = (120, 255, 120) if mode == "DRAWING" else (0, 200, 255)
            cv2.circle(view, pts[8], r, ring_col, 2, cv2.LINE_AA)

        # bounding box + label around the drawn digit
        if has_ink and probs.max() > 0:
            nz = cv2.findNonZero(canvas)
            if nz is not None:
                x, y, w, h = cv2.boundingRect(nz)
                cv2.rectangle(view, (x - 14, y - 14), (x + w + 14, y + h + 14), ACCENT, 2, cv2.LINE_AA)
                label = f"{int(np.argmax(probs))}  {probs.max() * 100:.0f}%"
                cv2.rectangle(view, (x - 14, y - 46), (x + 110, y - 14), ACCENT, -1)
                put(view, label, (x - 6, y - 22), 0.7, (20, 20, 20), 2, FONT_B)

        # fps
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        out = np.hstack([view, draw_panel(disp, preview, mode, fps, clear_progress, has_ink)])
        cv2.imshow("AirDigit AI", out)
        frame_i += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            clear_all()
        elif key == ord("s"):
            path = f"assets/screenshot_{int(now)}.png"
            cv2.imwrite(path, out)
            print("Saved", path)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
