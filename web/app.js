/**
 * AirDigit AI - Web Interface & Hand Tracking Engine
 * Supports MediaPipe Hands in-browser, Mouse/Touch fallback,
 * Neon Particle Trail, Live 28x28 MNIST preprocessor, and CNN API
 */

(function () {
  'use strict';

  // Elements
  const videoElement = document.getElementById('webcamVideo');
  const mainCanvas = document.getElementById('mainCanvas');
  const inkCanvas = document.getElementById('inkCanvas');
  const previewCanvas = document.getElementById('previewCanvas');
  const gradCamCanvas = document.getElementById('gradCamCanvas');
  const mainCtx = mainCanvas.getContext('2d');
  const inkCtx = inkCanvas.getContext('2d');
  const previewCtx = previewCanvas.getContext('2d');
  const gradCamCtx = gradCamCanvas.getContext('2d');

  // UI Elements
  const appModeBadge = document.getElementById('appModeBadge');
  const appModeText = document.getElementById('appModeText');
  const fpsVal = document.getElementById('fpsVal');
  const hudGesture = document.getElementById('hudGesture');
  const hudIcon = document.getElementById('hudIcon');
  const hudMsg = document.getElementById('hudMsg');
  const fistRing = document.getElementById('fistRing');
  const fistRingFill = document.getElementById('fistRingFill');
  const predictionTag = document.getElementById('predictionTag');
  const tagDigit = document.getElementById('tagDigit');
  const tagConf = document.getElementById('tagConf');
  const bigDigit = document.getElementById('bigDigit');
  const confidenceBadge = document.getElementById('confidenceBadge');
  const digitSub = document.getElementById('digitSub');
  const probBarsContainer = document.getElementById('probBars');
  const mathPanel = document.getElementById('mathPanel');
  const mathExpr = document.getElementById('mathExpr');
  const mathRes = document.getElementById('mathRes');
  const mathSteps = document.getElementById('mathSteps');

  // Buttons
  const btnToggleCam = document.getElementById('btnToggleCam');
  const btnToggleMouse = document.getElementById('btnToggleMouse');
  const btnMathMode = document.getElementById('btnMathMode');
  const btnGradCam = document.getElementById('btnGradCam');
  const btnClear = document.getElementById('btnClear');
  const btnScreenshot = document.getElementById('btnScreenshot');

  // Constants & Config
  const W = 800;
  const H = 600;
  mainCanvas.width = W;
  mainCanvas.height = H;
  inkCanvas.width = W;
  inkCanvas.height = H;

  const ACCENT = '#ffd200';
  const CORE = '#fff5be';
  const CYAN = '#00d2ff';
  const DANGER = '#ff3366';
  const GREEN = '#39ff14';

  // State
  let mode = 'NO HAND';
  let prevPos = null;
  let fistStartTime = null;
  let particles = [];
  let inkDirty = false;
  let hasInk = false;
  let frameCount = 0;
  let fps = 30.0;
  let lastTime = performance.now();
  let cameraActive = true;
  let mouseModeActive = false;
  let mathModeActive = false;
  let gradCamActive = true;
  let isMouseDown = false;
  let mousePos = { x: 0, y: 0 };
  let camera = null;
  let hands = null;

  // Probability distribution (0 to 9)
  let probs = new Array(10).fill(0.0);
  let dispProbs = new Array(10).fill(0.0);

  // Initialize Probability Bars DOM
  function initProbBars() {
    probBarsContainer.innerHTML = '';
    for (let i = 0; i < 10; i++) {
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.id = `prob-row-${i}`;

      const digitSpan = document.createElement('span');
      digitSpan.className = 'prob-digit';
      digitSpan.textContent = i;

      const track = document.createElement('div');
      track.className = 'prob-track';

      const fill = document.createElement('div');
      fill.className = 'prob-fill';
      fill.id = `prob-fill-${i}`;

      const valSpan = document.createElement('span');
      valSpan.className = 'prob-val';
      valSpan.id = `prob-val-${i}`;
      valSpan.textContent = '0.0%';

      track.appendChild(fill);
      row.appendChild(digitSpan);
      row.appendChild(track);
      row.appendChild(valSpan);
      probBarsContainer.appendChild(row);
    }
  }
  initProbBars();

  // Clear All
  function clearAll() {
    inkCtx.clearRect(0, 0, W, H);
    particles = [];
    prevPos = null;
    inkDirty = false;
    hasInk = false;
    fistStartTime = null;
    fistRing.style.display = 'none';
    predictionTag.style.display = 'none';
    probs.fill(0);
    dispProbs.fill(0);
    bigDigit.textContent = '?';
    confidenceBadge.textContent = '0.0% CONF';
    digitSub.textContent = 'Draw in mid-air or drag with mouse';
    previewCtx.fillStyle = '#000';
    previewCtx.fillRect(0, 0, 28, 28);
    gradCamCtx.fillStyle = '#000';
    gradCamCtx.fillRect(0, 0, 28, 28);
    updateProbBarsUI();
  }

  // Draw Line on Canvas with Neon Glow
  function drawNeonStroke(p0, p1) {
    inkCtx.save();
    inkCtx.lineCap = 'round';
    inkCtx.lineJoin = 'round';

    // Outer Glow Pass
    inkCtx.shadowColor = ACCENT;
    inkCtx.shadowBlur = 18;
    inkCtx.strokeStyle = 'rgba(255, 210, 0, 0.7)';
    inkCtx.lineWidth = 26;
    inkCtx.beginPath();
    inkCtx.moveTo(p0.x, p0.y);
    inkCtx.lineTo(p1.x, p1.y);
    inkCtx.stroke();

    // Inner Core Pass
    inkCtx.shadowColor = '#ffffff';
    inkCtx.shadowBlur = 4;
    inkCtx.strokeStyle = CORE;
    inkCtx.lineWidth = 14;
    inkCtx.beginPath();
    inkCtx.moveTo(p0.x, p0.y);
    inkCtx.lineTo(p1.x, p1.y);
    inkCtx.stroke();

    inkCtx.restore();

    // Emit Particles
    for (let i = 0; i < 2; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 1.0 + Math.random() * 3.5;
      particles.push({
        x: p1.x,
        y: p1.y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 1.0,
        color: Math.random() > 0.3 ? ACCENT : CORE,
        size: 2.5 + Math.random() * 3.0
      });
    }

    inkDirty = true;
    hasInk = true;
  }

  // Update and Render Particles
  function updateParticles() {
    mainCtx.save();
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.life -= 0.05;
      if (p.life <= 0) {
        particles.splice(i, 1);
        continue;
      }
      mainCtx.fillStyle = p.color;
      mainCtx.shadowColor = p.color;
      mainCtx.shadowBlur = 8;
      mainCtx.globalAlpha = p.life;
      mainCtx.beginPath();
      mainCtx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
      mainCtx.fill();
    }
    mainCtx.restore();
  }

  // Draw Pulsing Fingertip Reticle
  function drawReticle(x, y, ringColor) {
    const pulse = 14 + 4 * Math.sin(performance.now() * 0.008);
    mainCtx.save();
    mainCtx.strokeStyle = ringColor;
    mainCtx.shadowColor = ringColor;
    mainCtx.shadowBlur = 12;
    mainCtx.lineWidth = 2.5;
    mainCtx.beginPath();
    mainCtx.arc(x, y, pulse, 0, Math.PI * 2);
    mainCtx.stroke();

    mainCtx.fillStyle = '#ffffff';
    mainCtx.beginPath();
    mainCtx.arc(x, y, 3.5, 0, Math.PI * 2);
    mainCtx.fill();
    mainCtx.restore();
  }

  // Update Probability Bars Smoothly
  function updateProbBarsUI() {
    let topIdx = 0;
    let maxP = 0;
    for (let i = 0; i < 10; i++) {
      dispProbs[i] += 0.2 * (probs[i] - dispProbs[i]);
      if (dispProbs[i] > maxP) {
        maxP = dispProbs[i];
        topIdx = i;
      }
    }

    for (let i = 0; i < 10; i++) {
      const row = document.getElementById(`prob-row-${i}`);
      const fill = document.getElementById(`prob-fill-${i}`);
      const val = document.getElementById(`prob-val-${i}`);
      const pct = (dispProbs[i] * 100).toFixed(1);

      fill.style.width = `${pct}%`;
      val.textContent = `${pct}%`;

      if (hasInk && i === topIdx && maxP > 0.08) {
        row.classList.add('active');
      } else {
        row.classList.remove('active');
      }
    }

    if (hasInk && maxP > 0.08) {
      bigDigit.textContent = topIdx;
      confidenceBadge.textContent = `${(maxP * 100).toFixed(1)}% CONF`;
      digitSub.textContent = `High confidence digit recognition`;
    }
  }

  // Preprocess Canvas to 28x28 MNIST style
  function preprocessCanvas() {
    const imgData = inkCtx.getImageData(0, 0, W, H);
    const data = imgData.data;

    let minX = W, maxX = 0, minY = H, maxY = 0;
    let count = 0;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = (y * W + x) * 4;
        // Check alpha or brightness
        if (data[idx + 3] > 40 || (data[idx] + data[idx + 1] + data[idx + 2]) > 60) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          count++;
        }
      }
    }

    if (count < 80 || minX >= maxX || minY >= maxY) {
      return null;
    }

    const cropW = maxX - minX + 1;
    const cropH = maxY - minY + 1;
    if (Math.max(cropW, cropH) < 30) return null;

    // Scale to fit within 20x20
    const scale = 20.0 / Math.max(cropW, cropH);
    const nw = Math.max(1, Math.round(cropW * scale));
    const nh = Math.max(1, Math.round(cropH * scale));

    // Render crop to temporary canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.fillStyle = '#000000';
    tCtx.fillRect(0, 0, 28, 28);

    const ox = Math.floor((28 - nw) / 2);
    const oy = Math.floor((28 - nh) / 2);
    tCtx.drawImage(inkCanvas, minX, minY, cropW, cropH, ox, oy, nw, nh);

    // Center of Mass shift
    const tData = tCtx.getImageData(0, 0, 28, 28);
    let m00 = 0, m10 = 0, m01 = 0;
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const v = tData.data[(y * 28 + x) * 4 + 3];
        m00 += v;
        m10 += x * v;
        m01 += y * v;
      }
    }

    if (m00 > 0) {
      const cx = m10 / m00;
      const cy = m01 / m00;
      const shiftX = Math.round(14 - cx);
      const shiftY = Math.round(14 - cy);

      if (Math.abs(shiftX) > 0 || Math.abs(shiftY) > 0) {
        const shiftedCanvas = document.createElement('canvas');
        shiftedCanvas.width = 28;
        shiftedCanvas.height = 28;
        const sCtx = shiftedCanvas.getContext('2d');
        sCtx.fillStyle = '#000';
        sCtx.fillRect(0, 0, 28, 28);
        sCtx.drawImage(tempCanvas, shiftX, shiftY);
        tCtx.drawImage(shiftedCanvas, 0, 0);
      }
    }

    // Draw on preview canvas
    previewCtx.clearRect(0, 0, 28, 28);
    previewCtx.drawImage(tempCanvas, 0, 0);

    return {
      canvas28: tempCanvas,
      bbox: { x: minX, y: minY, w: cropW, h: cropH }
    };
  }

  // Client-Side Simulated CNN Fallback & Backend Predict
  async function triggerInference(cropObj) {
    if (!cropObj) return;

    try {
      const b64 = cropObj.canvas28.toDataURL('image/png');
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Bypass-Tunnel-Reminder': 'true'
        },
        body: JSON.stringify({ image: b64, mathMode: mathModeActive })
      });

      if (res.ok) {
        const data = await res.json();
        if (data.probabilities && data.probabilities.length === 10) {
          probs = data.probabilities;
        }
        if (data.gradcam) {
          const img = new Image();
          img.onload = () => {
            gradCamCtx.drawImage(img, 0, 0, 28, 28);
          };
          img.src = data.gradcam;
        }
        if (mathModeActive && data.mathResult) {
          mathExpr.textContent = data.mathResult.expression;
          mathRes.textContent = data.mathResult.result;
          mathSteps.textContent = data.mathResult.steps || 'Evaluated successfully';
        }
      }
    } catch (err) {
      // Offline fallback: synthetic template matching
      runClientFallback(cropObj.canvas28);
    }
  }

  function runClientFallback(c28) {
    const ctx = c28.getContext('2d');
    const d = ctx.getImageData(0, 0, 28, 28).data;
    let topSum = 0, botSum = 0, leftSum = 0, rightSum = 0, midSum = 0;

    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const val = d[(y * 28 + x) * 4 + 3] > 30 ? 1 : 0;
        if (y < 14) topSum += val; else botSum += val;
        if (x < 14) leftSum += val; else rightSum += val;
        if (x >= 8 && x <= 20 && y >= 8 && y <= 20) midSum += val;
      }
    }

    const total = topSum + botSum || 1;
    let guess = 1;
    const aspect = rightSum / (leftSum || 1);
    if (midSum < 15 && total > 40) guess = 0;
    else if (aspect > 0.8 && aspect < 1.2 && total < 50) guess = 1;
    else if (topSum > botSum && rightSum > leftSum) guess = 7;
    else if (topSum > botSum) guess = 3;
    else guess = 8;

    probs = new Array(10).fill(0.02);
    probs[guess] = 0.82;
  }

  // MediaPipe Fingers Extension Check
  function getFingersUp(landmarks) {
    const tips = [8, 12, 16, 20];
    const pips = [6, 10, 14, 18];
    return tips.map((t, idx) => {
      return landmarks[t].y < landmarks[pips[idx]].y - 0.02 ? 1 : 0;
    });
  }

  // Draw Hand Skeleton
  function drawSkeleton(landmarks) {
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [5, 9], [9, 10], [10, 11], [11, 12],
      [9, 13], [13, 14], [14, 15], [15, 16],
      [13, 17], [17, 18], [18, 19], [19, 20],
      [0, 17]
    ];

    mainCtx.save();
    mainCtx.strokeStyle = 'rgba(255, 210, 0, 0.4)';
    mainCtx.lineWidth = 1.5;

    for (const [a, b] of connections) {
      const pa = landmarks[a];
      const pb = landmarks[b];
      mainCtx.beginPath();
      mainCtx.moveTo(pa.x * W, pa.y * H);
      mainCtx.lineTo(pb.x * W, pb.y * H);
      mainCtx.stroke();
    }

    mainCtx.fillStyle = 'rgba(255, 245, 190, 0.8)';
    for (const p of landmarks) {
      mainCtx.beginPath();
      mainCtx.arc(p.x * W, p.y * H, 3, 0, Math.PI * 2);
      mainCtx.fill();
    }
    mainCtx.restore();
  }

  // MediaPipe Results Handler
  function onResults(results) {
    // 1. Draw video background
    mainCtx.save();
    mainCtx.clearRect(0, 0, W, H);

    if (cameraActive && results.image) {
      mainCtx.translate(W, 0);
      mainCtx.scale(-1, 1);
      mainCtx.globalAlpha = 0.55;
      mainCtx.drawImage(results.image, 0, 0, W, H);
      mainCtx.globalAlpha = 1.0;
    } else {
      mainCtx.fillStyle = '#0a0d14';
      mainCtx.fillRect(0, 0, W, H);
      // Grid lines
      mainCtx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      mainCtx.lineWidth = 1;
      for (let x = 0; x < W; x += 40) {
        mainCtx.beginPath(); mainCtx.moveTo(x, 0); mainCtx.lineTo(x, H); mainCtx.stroke();
      }
      for (let y = 0; y < H; y += 40) {
        mainCtx.beginPath(); mainCtx.moveTo(0, y); mainCtx.lineTo(W, y); mainCtx.stroke();
      }
    }
    mainCtx.restore();

    // 2. Render Hand Tracking or Mouse
    let curTip = null;
    let up = [0, 0, 0, 0];

    if (!mouseModeActive && results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      // Mirror x coordinate because video is flipped
      const rawLm = results.multiHandLandmarks[0];
      const lm = rawLm.map(p => ({ x: 1.0 - p.x, y: p.y, z: p.z }));

      drawSkeleton(lm);
      up = getFingersUp(lm);
      curTip = { x: lm[8].x * W, y: lm[8].y * H };

      if (up[0] === 1 && up[1] === 0 && up[2] === 0 && up[3] === 0) {
        mode = 'DRAWING';
      } else if (up[0] === 1 && up[1] === 1) {
        mode = 'HOVER';
      } else if (up.every(v => v === 0)) {
        mode = 'FIST';
      } else {
        mode = 'IDLE';
      }
    } else if (mouseModeActive) {
      curTip = { x: mousePos.x, y: mousePos.y };
      mode = isMouseDown ? 'DRAWING' : 'HOVER';
    } else {
      mode = 'NO HAND';
    }

    // 3. Update HUD and Gesture State
    updateGestureHUD(mode);

    // 4. Handle Gestures
    const now = performance.now();
    if (mode === 'DRAWING' && curTip) {
      const smoothed = prevPos ? {
        x: Math.round(prevPos.x + 0.55 * (curTip.x - prevPos.x)),
        y: Math.round(prevPos.y + 0.55 * (curTip.y - prevPos.y))
      } : curTip;

      if (prevPos) {
        drawNeonStroke(prevPos, smoothed);
      }
      prevPos = smoothed;
      drawReticle(curTip.x, curTip.y, GREEN);
    } else {
      prevPos = null;
      if (mode === 'HOVER' && curTip) {
        drawReticle(curTip.x, curTip.y, CYAN);
      }
    }

    // Fist Clear Check
    if (mode === 'FIST') {
      if (!fistStartTime) fistStartTime = now;
      const elapsed = (now - fistStartTime) / 1000.0;
      fistRing.style.display = 'flex';
      const offset = Math.max(0, 264 - (elapsed / 0.8) * 264);
      fistRingFill.style.strokeDashoffset = offset;

      if (elapsed >= 0.8) {
        clearAll();
        fistStartTime = null;
      }
    } else {
      fistStartTime = null;
      fistRing.style.display = 'none';
    }

    // 5. Draw Ink Layer on Main Canvas
    mainCtx.drawImage(inkCanvas, 0, 0);

    // 6. Draw Particles
    updateParticles();

    // 7. Periodic Inference (Every 4 frames if ink modified)
    frameCount++;
    if (inkDirty && frameCount % 4 === 0) {
      const crop = preprocessCanvas();
      if (crop) {
        triggerInference(crop);
        // Show bounding box tag
        updateBoundingBox(crop.bbox);
      } else {
        predictionTag.style.display = 'none';
      }
      inkDirty = false;
    }

    // 8. Update Probability Bars
    updateProbBarsUI();

    // 9. Calculate FPS
    const delta = (now - lastTime) / 1000.0;
    lastTime = now;
    if (delta > 0) {
      fps = 0.9 * fps + 0.1 * (1.0 / delta);
      fpsVal.textContent = fps.toFixed(1);
    }
  }

  function updateGestureHUD(curMode) {
    appModeText.textContent = curMode;
    switch (curMode) {
      case 'DRAWING':
        appModeBadge.style.borderColor = GREEN;
        hudIcon.textContent = '☝️';
        hudMsg.textContent = 'Drawing mid-air digit...';
        break;
      case 'HOVER':
        appModeBadge.style.borderColor = CYAN;
        hudIcon.textContent = '✌️';
        hudMsg.textContent = 'Hovering (tracking pointer)';
        break;
      case 'FIST':
        appModeBadge.style.borderColor = DANGER;
        hudIcon.textContent = '✊';
        hudMsg.textContent = 'Hold fist 0.8s to clear!';
        break;
      default:
        appModeBadge.style.borderColor = 'var(--border-subtle)';
        hudIcon.textContent = mouseModeActive ? '🖱️' : '🖐️';
        hudMsg.textContent = mouseModeActive ? 'Mouse Draw Active' : 'Show hand to webcam';
    }
  }

  function updateBoundingBox(bbox) {
    if (!hasInk) return;
    const topIdx = probs.indexOf(Math.max(...probs));
    const conf = probs[topIdx];
    if (conf < 0.1) {
      predictionTag.style.display = 'none';
      return;
    }

    const rect = mainCanvas.getBoundingClientRect();
    const scaleX = rect.width / W;
    const scaleY = rect.height / H;

    predictionTag.style.display = 'flex';
    predictionTag.style.left = `${(bbox.x + bbox.w / 2) * scaleX}px`;
    predictionTag.style.top = `${bbox.y * scaleY}px`;
    tagDigit.textContent = topIdx;
    tagConf.textContent = `${(conf * 100).toFixed(0)}%`;

    // Draw box on main canvas
    mainCtx.save();
    mainCtx.strokeStyle = ACCENT;
    mainCtx.lineWidth = 2;
    mainCtx.strokeRect(bbox.x - 12, bbox.y - 12, bbox.w + 24, bbox.h + 24);
    mainCtx.restore();
  }

  // Setup MediaPipe Hands
  function initMediaPipe() {
    hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 0,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.6
    });

    hands.onResults(onResults);

    camera = new Camera(videoElement, {
      onFrame: async () => {
        if (cameraActive) {
          await hands.send({ image: videoElement });
        }
      },
      width: 640,
      height: 480
    });

    camera.start().catch((err) => {
      console.warn('Webcam permission denied or unavailable, activating Mouse Mode:', err);
      enableMouseMode();
    });
  }

  // Mouse & Touch Controls
  function enableMouseMode() {
    mouseModeActive = true;
    btnToggleMouse.classList.add('active');
    appModeText.textContent = 'MOUSE MODE';
    hudIcon.textContent = '🖱️';
    hudMsg.textContent = 'Click and drag to draw';

    // Start requestAnimationFrame loop for mouse
    function loop() {
      if (mouseModeActive || !cameraActive) {
        onResults({ image: null });
      }
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  }

  // Event Listeners
  mainCanvas.addEventListener('mousedown', (e) => {
    isMouseDown = true;
    const rect = mainCanvas.getBoundingClientRect();
    mousePos = {
      x: (e.clientX - rect.left) * (W / rect.width),
      y: (e.clientY - rect.top) * (H / rect.height)
    };
  });

  window.addEventListener('mouseup', () => {
    isMouseDown = false;
    prevPos = null;
  });

  mainCanvas.addEventListener('mousemove', (e) => {
    const rect = mainCanvas.getBoundingClientRect();
    mousePos = {
      x: (e.clientX - rect.left) * (W / rect.width),
      y: (e.clientY - rect.top) * (H / rect.height)
    };
  });

  // Touch Events
  mainCanvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    isMouseDown = true;
    const t = e.touches[0];
    const rect = mainCanvas.getBoundingClientRect();
    mousePos = {
      x: (t.clientX - rect.left) * (W / rect.width),
      y: (t.clientY - rect.top) * (H / rect.height)
    };
  }, { passive: false });

  mainCanvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const t = e.touches[0];
    const rect = mainCanvas.getBoundingClientRect();
    mousePos = {
      x: (t.clientX - rect.left) * (W / rect.width),
      y: (t.clientY - rect.top) * (H / rect.height)
    };
  }, { passive: false });

  window.addEventListener('touchend', () => {
    isMouseDown = false;
    prevPos = null;
  });

  // Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.key === 'c' || e.key === 'C') clearAll();
    if (e.key === 's' || e.key === 'S') saveSnapshot();
  });

  // Toolbar Actions
  btnClear.addEventListener('click', clearAll);

  btnToggleCam.addEventListener('click', () => {
    cameraActive = !cameraActive;
    btnToggleCam.classList.toggle('active', cameraActive);
    if (!cameraActive) {
      enableMouseMode();
    }
  });

  btnToggleMouse.addEventListener('click', () => {
    mouseModeActive = !mouseModeActive;
    btnToggleMouse.classList.toggle('active', mouseModeActive);
    if (mouseModeActive) {
      enableMouseMode();
    }
  });

  btnMathMode.addEventListener('click', () => {
    mathModeActive = !mathModeActive;
    btnMathMode.classList.toggle('active', mathModeActive);
    mathPanel.style.display = mathModeActive ? 'flex' : 'none';
  });

  btnGradCam.addEventListener('click', () => {
    gradCamActive = !gradCamActive;
    btnGradCam.classList.toggle('active', gradCamActive);
    document.getElementById('gradCamBox').style.display = gradCamActive ? 'flex' : 'none';
  });

  function saveSnapshot() {
    const compCanvas = document.createElement('canvas');
    compCanvas.width = W;
    compCanvas.height = H;
    const cCtx = compCanvas.getContext('2d');
    cCtx.drawImage(mainCanvas, 0, 0);

    const a = document.createElement('a');
    a.download = `airdigit_${Date.now()}.png`;
    a.href = compCanvas.toDataURL('image/png');
    a.click();
  }
  btnScreenshot.addEventListener('click', saveSnapshot);

  // Initialize
  initMediaPipe();
})();
