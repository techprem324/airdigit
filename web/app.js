/* airdigit: draw a digit with the mouse, a finger, or your hand on camera.
   The browser keeps the strokes, renders a clean 112x112 picture of them, and
   asks the server's CNN what digit it is. */
(function () {
  'use strict';

  var W = 1200, H = 900;      // internal drawing resolution (the canvas scales with CSS)
  var BRUSH = 24;             // visible pen width, in internal pixels
  var MIN_EXTENT = 30;        // smallest drawing we will try to read
  var MASK = 112, BOX = 80, MASK_PEN = 8;  // clean picture sent to the model

  var $ = function (id) { return document.getElementById(id); };
  var stage = $('stage'), video = $('video'), inkCanvas = $('ink'), fxCanvas = $('fx');
  var ink = inkCanvas.getContext('2d'), fx = fxCanvas.getContext('2d');
  inkCanvas.width = fxCanvas.width = W;
  inkCanvas.height = fxCanvas.height = H;

  var els = {
    chip: $('chip'), hint: $('hint'), ring: $('clearRing'), ringFill: $('ringFill'),
    overlay: $('overlay'), overlayText: $('overlayText'), overlayRetry: $('overlayRetry'), overlayMouse: $('overlayMouse'),
    undo: $('undoBtn'), clear: $('clearBtn'), save: $('saveBtn'),
    mark: $('mark'), digit: $('digit'), verdict: $('verdict'), chart: $('chart'),
    seen: $('seen'),
    themeBtn: $('themeBtn')
  };

  var state = {
    mode: 'mouse',        // 'mouse' | 'camera'
    strokes: [],          // [{pts: [{x, y}]}]
    current: null,        // stroke being drawn
    seq: 0,               // bumped on clear/undo so stale replies are dropped
    busy: false,
    again: false,
    lastPredictTime: 0,
    predictTimer: null,
    cameraStarted: false,
    cameraController: null,
    hands: null,
    fistHoldStart: 0,
    drawingFromCamera: false
  };

  /* ---------- Theme Handling ---------- */
  function initTheme() {
    var stored = document.documentElement.dataset.theme;
    try {
      var saved = localStorage.getItem('airdigit-theme');
      if (saved === 'light' || saved === 'dark') stored = saved;
    } catch (e) {}
    document.documentElement.dataset.theme = stored;
    updateThemeButton();
    els.themeBtn.addEventListener('click', function () {
      var current = document.documentElement.dataset.theme;
      var next = (current === 'dark') ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('airdigit-theme', next); } catch (e) {}
      updateThemeButton();
      redraw();
    });
  }

  function updateThemeButton() {
    var isDark = document.documentElement.dataset.theme === 'dark';
    els.themeBtn.setAttribute('aria-label', isDark ? 'Switch to light theme' : 'Switch to dark theme');
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function inkColor() {
    return state.mode === 'camera' ? '#ffffff' : cssVar('--ink');
  }

  function mid(p1, p2) {
    return { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
  }

  /* ---------- Drawing Engine ---------- */
  function paintStroke(c, pts, map, width, color) {
    if (!pts || pts.length === 0) return;
    c.lineCap = c.lineJoin = 'round';
    c.lineWidth = width;
    c.strokeStyle = c.fillStyle = color;
    var p = pts.map(map);
    if (p.length === 1) {
      c.beginPath();
      c.arc(p[0].x, p[0].y, width / 2, 0, Math.PI * 2);
      c.fill();
      return;
    }
    c.beginPath();
    c.moveTo(p[0].x, p[0].y);
    for (var i = 1; i < p.length - 1; i++) {
      var xc = (p[i].x + p[i + 1].x) / 2;
      var yc = (p[i].y + p[i + 1].y) / 2;
      c.quadraticCurveTo(p[i].x, p[i].y, xc, yc);
    }
    c.lineTo(p[p.length - 1].x, p[p.length - 1].y);
    c.stroke();
  }

  var identity = function (p) { return p; };

  function redraw() {
    ink.clearRect(0, 0, W, H);
    var color = inkColor();
    state.strokes.forEach(function (s) {
      paintStroke(ink, s.pts, identity, BRUSH, color);
    });
    updateButtons();
  }

  function updateButtons() {
    var hasStrokes = state.strokes.length > 0;
    els.undo.disabled = !hasStrokes;
    els.clear.disabled = !hasStrokes;
    els.save.disabled = !hasStrokes;
    if (hasStrokes) {
      els.hint.style.display = 'none';
    } else {
      els.hint.style.display = 'block';
    }
  }

  /* ---------- Pointer / Mouse / Touch Event Listeners ---------- */
  function getCanvasCoords(e) {
    var rect = inkCanvas.getBoundingClientRect();
    var clientX = e.clientX !== undefined ? e.clientX : (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
    var clientY = e.clientY !== undefined ? e.clientY : (e.touches && e.touches[0] ? e.touches[0].clientY : 0);
    return {
      x: ((clientX - rect.left) / rect.width) * W,
      y: ((clientY - rect.top) / rect.height) * H
    };
  }

  var isDrawingMouse = false;

  function onDrawingStart(e) {
    if (state.mode !== 'mouse') return;
    isDrawingMouse = true;
    var pt = getCanvasCoords(e);
    state.current = { pts: [pt] };
    state.strokes.push(state.current);
    redraw();
  }

  function onDrawingMove(e) {
    if (state.mode !== 'mouse' || !isDrawingMouse || !state.current) return;
    var pt = getCanvasCoords(e);
    state.current.pts.push(pt);
    redraw();
    debouncedPredict();
  }

  function onDrawingEnd(e) {
    if (state.mode !== 'mouse' || !isDrawingMouse) return;
    isDrawingMouse = false;
    state.current = null;
    redraw();
    predictNow();
  }

  inkCanvas.addEventListener('pointerdown', function (e) {
    try { inkCanvas.setPointerCapture(e.pointerId); } catch(err) {}
    onDrawingStart(e);
  });
  inkCanvas.addEventListener('pointermove', onDrawingMove);
  inkCanvas.addEventListener('pointerup', onDrawingEnd);
  inkCanvas.addEventListener('pointercancel', onDrawingEnd);

  /* ---------- Model Prediction ---------- */
  function getDrawingBounds() {
    var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    var count = 0;
    state.strokes.forEach(function (s) {
      s.pts.forEach(function (p) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
        count++;
      });
    });
    if (count < 2) return null;
    return { minX: minX, maxX: maxX, minY: minY, maxY: maxY, w: maxX - minX, h: maxY - minY };
  }

  function renderCleanModelImage() {
    var b = getDrawingBounds();
    if (!b || Math.max(b.w, b.h) < MIN_EXTENT) return null;

    var off = document.createElement('canvas');
    off.width = MASK;
    off.height = MASK;
    var ctx = off.getContext('2d');
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, MASK, MASK);

    var scale = BOX / Math.max(b.w, b.h);
    var cx = (b.minX + b.maxX) / 2;
    var cy = (b.minY + b.maxY) / 2;

    var map = function (p) {
      return {
        x: (p.x - cx) * scale + (MASK / 2),
        y: (p.y - cy) * scale + (MASK / 2)
      };
    };

    state.strokes.forEach(function (s) {
      paintStroke(ctx, s.pts, map, MASK_PEN, '#ffffff');
    });

    return off.toDataURL('image/png');
  }

  function debouncedPredict() {
    clearTimeout(state.predictTimer);
    state.predictTimer = setTimeout(predictNow, 160);
  }

  function predictNow() {
    var dataUrl = renderCleanModelImage();
    if (!dataUrl) return;

    var curSeq = ++state.seq;
    if (state.busy) {
      state.again = true;
      return;
    }
    state.busy = true;

    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl })
    })
      .then(function (r) {
        if (!r.ok) throw new Error('Prediction error ' + r.status);
        return r.json();
      })
      .then(function (data) {
        if (curSeq !== state.seq) return;
        renderPrediction(data);
      })
      .catch(function (err) {
        console.warn(err);
        if (curSeq === state.seq) {
          els.verdict.textContent = 'Could not get a prediction. Check the model connection and try again.';
        }
      })
      .finally(function () {
        state.busy = false;
        if (state.again) {
          state.again = false;
          predictNow();
        }
      });
  }

  function renderPrediction(data) {
    if (!data || data.prediction === undefined) return;
    var digit = data.prediction;
    var conf = Math.round(data.confidence * 100);

    els.digit.textContent = digit;
    els.mark.dataset.state = 'active';

    if (conf >= 60) {
      els.verdict.textContent = conf + '% sure.';
    } else {
      var sorted = data.probabilities.map(function (p, i) { return { p: p, i: i }; })
        .sort(function (a, b) { return b.p - a.p; });
      els.verdict.textContent = 'Could be ' + sorted[0].i + ' or ' + sorted[1].i + ' (' + conf + '% sure).';
    }

    renderChart(data.probabilities, digit);

    if (data.input) {
      var img = new Image();
      img.onload = function () {
        var sctx = els.seen.getContext('2d');
        sctx.clearRect(0, 0, 28, 28);
        sctx.drawImage(img, 0, 0, 28, 28);
      };
      img.src = data.input;
    }
  }

  function resetPrediction() {
    els.digit.textContent = '–';
    els.mark.dataset.state = 'empty';
    els.verdict.textContent = 'Draw a digit to get started.';
    renderChart([0,0,0,0,0,0,0,0,0,0], -1);
    var sctx = els.seen.getContext('2d');
    sctx.fillStyle = '#000000';
    sctx.fillRect(0, 0, 28, 28);
  }

  function renderChart(probs, topDigit) {
    els.chart.innerHTML = '';
    for (var i = 0; i <= 9; i++) {
      var p = (probs && probs[i] !== undefined) ? probs[i] : 0;
      var col = document.createElement('div');
      col.className = 'chart-bar-col';

      var bar = document.createElement('div');
      bar.className = 'chart-bar' + (i === topDigit ? ' top' : '');
      bar.style.height = Math.max(3, Math.round(p * 85)) + 'px';

      var lbl = document.createElement('span');
      lbl.className = 'chart-label' + (i === topDigit ? ' top' : '');
      lbl.textContent = i;

      col.appendChild(bar);
      col.appendChild(lbl);
      els.chart.appendChild(col);
    }
  }

  /* ---------- Action Buttons ---------- */
  els.undo.addEventListener('click', function () {
    if (state.strokes.length === 0) return;
    state.strokes.pop();
    state.seq++;
    redraw();
    if (state.strokes.length === 0) {
      resetPrediction();
    } else {
      predictNow();
    }
  });

  els.clear.addEventListener('click', function () {
    state.strokes = [];
    state.current = null;
    state.seq++;
    redraw();
    resetPrediction();
  });

  els.save.addEventListener('click', function () {
    if (state.strokes.length === 0) return;
    var a = document.createElement('a');
    a.download = 'airdigit-' + Date.now() + '.png';
    a.href = inkCanvas.toDataURL('image/png');
    a.click();
  });

  /* ---------- Keyboard Shortcuts ---------- */
  window.addEventListener('keydown', function (e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    var key = e.key.toLowerCase();
    if (key === 'z') {
      els.undo.click();
    } else if (key === 'c') {
      els.clear.click();
    }
  });

  /* ---------- Mode Switching & Camera Tracking ---------- */
  var modeButtons = document.querySelectorAll('.mode-button');
  modeButtons.forEach(function (btn) {
    btn.addEventListener('click', function () {
      var mode = btn.dataset.mode;
      if (mode === state.mode) return;
      modeButtons.forEach(function (b) { b.setAttribute('aria-pressed', b === btn ? 'true' : 'false'); });
      setMode(mode);
    });
  });

  function setMode(mode) {
    state.mode = mode;
    stage.dataset.mode = mode;
    redraw();
    if (mode === 'camera') {
      startCamera();
    } else {
      stopCamera();
    }
  }

  function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showOverlay('Your browser does not support camera access.');
      return;
    }
    state.cameraStarted = true;
    loadMediaPipe();
  }

  function stopCamera() {
    state.cameraStarted = false;
    if (state.cameraController) {
      state.cameraController.stop();
      state.cameraController = null;
    }
    if (state.hands) {
      state.hands.close();
      state.hands = null;
    }
    video.srcObject = null;
    fx.clearRect(0, 0, W, H);
    els.overlay.hidden = true;
  }

  function showOverlay(msg) {
    els.overlayText.textContent = msg;
    els.overlay.hidden = false;
  }

  els.overlayRetry.addEventListener('click', function () {
    els.overlay.hidden = true;
    startCamera();
  });

  els.overlayMouse.addEventListener('click', function () {
    els.overlay.hidden = true;
    modeButtons[0].click();
  });

  function loadMediaPipe() {
    if (window.Hands) {
      setupHands();
      return;
    }
    var s1 = document.createElement('script');
    s1.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js';
    var s2 = document.createElement('script');
    s2.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js';
    s1.onerror = s2.onerror = function () {
      showOverlay('Hand tracking could not load. Check your internet connection, or draw with mouse instead.');
    };
    document.head.appendChild(s1);
    document.head.appendChild(s2);
    s2.onload = setupHands;
  }

  function setupHands() {
    if (!state.cameraStarted) return;
    if (!window.Hands || !window.Camera) {
      state.cameraStarted = false;
      showOverlay('Hand tracking could not start. Draw with mouse instead.');
      return;
    }
    var hands = new window.Hands({
      locateFile: function (file) {
        return 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/' + file;
      }
    });
    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.65,
      minTrackingConfidence: 0.65
    });
    state.hands = hands;
    hands.onResults(onHandResults);

    var cam = new window.Camera(video, {
      onFrame: async function () {
        if (state.cameraStarted) {
          await hands.send({ image: video });
        }
      },
      width: 640,
      height: 480
    });
    state.cameraController = cam;
    cam.start().catch(function (err) {
      state.cameraStarted = false;
      showOverlay('Camera permission denied or camera not found: ' + err.message);
    });
  }

  function onHandResults(results) {
    fx.clearRect(0, 0, W, H);
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      state.fistHoldStart = 0;
      els.ring.hidden = true;
      if (state.drawingFromCamera && state.current) {
        state.drawingFromCamera = false;
        state.current = null;
        predictNow();
      }
      return;
    }

    var lm = results.multiHandLandmarks[0];
    // Invert X because camera video is flipped horizontally
    var indexTip = { x: (1.0 - lm[8].x) * W, y: lm[8].y * H };
    var indexPip = { x: (1.0 - lm[6].x) * W, y: lm[6].y * H };
    var middleTip = { x: (1.0 - lm[12].x) * W, y: lm[12].y * H };
    var middlePip = { x: (1.0 - lm[10].x) * W, y: lm[10].y * H };
    var ringTip = { x: (1.0 - lm[16].x) * W, y: lm[16].y * H };
    var ringPip = { x: (1.0 - lm[14].x) * W, y: lm[14].y * H };
    var pinkyTip = { x: (1.0 - lm[20].x) * W, y: lm[20].y * H };
    var pinkyPip = { x: (1.0 - lm[18].x) * W, y: lm[18].y * H };

    var indexUp = lm[8].y < lm[6].y - 0.02;
    var middleUp = lm[12].y < lm[10].y - 0.02;
    var ringUp = lm[16].y < lm[14].y - 0.02;
    var pinkyUp = lm[20].y < lm[18].y - 0.02;

    var isFist = !indexUp && !middleUp && !ringUp && !pinkyUp;
    var isDrawing = indexUp && !middleUp && !ringUp && !pinkyUp;
    var isHover = indexUp && middleUp && !ringUp && !pinkyUp;

    // Draw pointer indicator on fx canvas
    fx.beginPath();
    fx.arc(indexTip.x, indexTip.y, isDrawing ? 12 : 7, 0, Math.PI * 2);
    fx.fillStyle = isDrawing ? '#ff4b4b' : 'rgba(255, 255, 255, 0.7)';
    fx.fill();
    fx.strokeStyle = '#ffffff';
    fx.lineWidth = 2;
    fx.stroke();

    // Fist clear gesture
    if (isFist) {
      if (!state.fistHoldStart) state.fistHoldStart = Date.now();
      var elapsed = Date.now() - state.fistHoldStart;
      var prog = Math.min(1.0, elapsed / 800.0);
      els.ring.hidden = false;
      els.ringFill.style.strokeDashoffset = 1.0 - prog;
      if (prog >= 1.0) {
        els.clear.click();
        state.fistHoldStart = Date.now() + 600; // prevent repeated clearing
      }
    } else {
      state.fistHoldStart = 0;
      els.ring.hidden = true;
    }

    // Drawing gesture
    if (isDrawing) {
      if (!state.drawingFromCamera || !state.current) {
        state.drawingFromCamera = true;
        state.current = { pts: [indexTip] };
        state.strokes.push(state.current);
      } else {
        state.current.pts.push(indexTip);
      }
      redraw();
      debouncedPredict();
    } else {
      if (state.drawingFromCamera && state.current) {
        state.drawingFromCamera = false;
        state.current = null;
        redraw();
        predictNow();
      }
    }
  }

  /* ---------- Initialisation ---------- */
  initTheme();
  renderChart([0,0,0,0,0,0,0,0,0,0], -1);
  updateButtons();

})();
