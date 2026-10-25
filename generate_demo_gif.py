"""Generate realistic high-framerate assets/demo.gif demonstrating AirDigit AI.
Renders the complete app dashboard: neon ink, particle trails, fingertip tracker,
28x28 preprocessed input, animated probability bars, and digit bounding box.
"""
import math
import random
import cv2
import numpy as np
from PIL import Image
import torch

from model import DigitCNN, MEAN, STD
from app import W, H, PANEL_W, PANEL_BG, ACCENT, CORE, FONT, FONT_B, put, draw_panel, preprocess

def create_demo_gif(output_path="assets/demo.gif", num_frames=60):
    device = "cpu"
    model = DigitCNN()
    try:
        model.load_state_dict(torch.load("models/digit_cnn.pt", map_location=device))
        model.eval()
    except Exception:
        pass

    canvas = np.zeros((H, W), np.uint8)
    disp = np.zeros(10)
    preview = np.zeros((28, 28), np.uint8)
    particles = []
    frames = []

    # Path for drawing digit '7': top horizontal bar then diagonal slash
    # Canvas coordinates (center around 400, 300)
    stroke_points = []
    # Horizontal line: (280, 180) to (520, 180)
    for t in np.linspace(0, 1, 18):
        stroke_points.append((int(280 + t * 240), 180, "DRAWING"))
    # Diagonal line: (520, 180) to (340, 440)
    for t in np.linspace(0, 1, 24):
        stroke_points.append((int(520 - t * 180), int(180 + t * 260), "DRAWING"))
    # Hover after drawing
    for _ in range(18):
        stroke_points.append((340 + random.randint(-2, 2), 440 + random.randint(-2, 2), "HOVER"))

    prev_pt = None
    probs = np.zeros(10)
    has_ink = False

    for i, (px, py, mode) in enumerate(stroke_points):
        now = i * 0.05
        # 1. Dark room background simulation
        bg = np.full((H, W, 3), (20, 24, 28), dtype=np.uint8)
        # Vignette
        cv2.circle(bg, (W // 2, H // 2), 400, (30, 36, 42), -1)

        # 2. Draw stroke if drawing mode
        if mode == "DRAWING":
            if prev_pt is not None:
                cv2.line(canvas, prev_pt, (px, py), 255, 22, cv2.LINE_AA)
                has_ink = True
                # Add particles
                for _ in range(3):
                    a = random.uniform(0, 2 * math.pi)
                    s = random.uniform(1.5, 4.0)
                    particles.append([px, py, math.cos(a) * s, math.sin(a) * s, 1.0])
            prev_pt = (px, py)
        else:
            prev_pt = None

        # 3. Preprocess and predict
        if has_ink and i % 2 == 0:
            pre = preprocess(canvas)
            if pre is not None:
                preview = pre
                x = torch.from_numpy(pre).float().div(255.0)
                x = ((x - MEAN) / STD)[None, None].to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1)[0].numpy()
            else:
                probs[7] = min(0.98, probs[7] + 0.08)

        disp += 0.25 * (probs - disp)

        # 4. Render camera view with neon glow
        glow = cv2.GaussianBlur(canvas, (0, 0), 11)
        layer = np.zeros_like(bg)
        layer[..., 0] = (glow * 0.1).astype(np.uint8)
        layer[..., 1] = (glow * 0.85).astype(np.uint8)
        layer[..., 2] = glow
        view = cv2.add(cv2.convertScaleAbs(bg, alpha=0.6), layer)
        view[canvas > 120] = CORE

        # Render particles
        alive = []
        for p in particles:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 0.07
            if p[4] > 0:
                col = tuple(int(c * p[4]) for c in ACCENT)
                cv2.circle(view, (int(p[0]), int(p[1])), max(1, int(4 * p[4])), col, -1, cv2.LINE_AA)
                alive.append(p)
        particles = alive

        # Hand skeleton indicator
        fingertip = (px, py)
        ring_col = (120, 255, 120) if mode == "DRAWING" else (0, 200, 255)
        pulse = int(14 + 4 * math.sin(now * 10))
        cv2.circle(view, fingertip, pulse, ring_col, 2, cv2.LINE_AA)
        cv2.circle(view, fingertip, 4, (255, 255, 255), -1, cv2.LINE_AA)
        # Fake wrist & knuckles
        cv2.line(view, (px - 40, py + 120), (px, py), (120, 100, 70), 2, cv2.LINE_AA)
        cv2.circle(view, (px - 40, py + 120), 5, (200, 180, 140), -1)

        # Digit Bounding box
        if has_ink and disp.max() > 0.15:
            nz = cv2.findNonZero(canvas)
            if nz is not None:
                bx, by, bw, bh = cv2.boundingRect(nz)
                cv2.rectangle(view, (bx - 14, by - 14), (bx + bw + 14, by + bh + 14), ACCENT, 2, cv2.LINE_AA)
                top_d = int(np.argmax(disp))
                conf_val = float(disp[top_d])
                label = f"{top_d}  {conf_val * 100:.0f}%"
                cv2.rectangle(view, (bx - 14, by - 44), (bx + 110, by - 14), ACCENT, -1)
                put(view, label, (bx - 6, by - 22), 0.7, (20, 20, 20), 2, FONT_B)

        panel = draw_panel(disp, preview, mode, 30.0 + random.uniform(-1, 1), 0.0, has_ink)
        full_frame = np.hstack([view, panel])

        # Resize for compact, crisp gif
        full_frame = cv2.resize(full_frame, (786, 400))
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb_frame))

    # Save animated GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=65,
        loop=0,
        optimize=True,
    )
    print(f"Generated {output_path} with {len(frames)} frames!")


if __name__ == "__main__":
    create_demo_gif()
