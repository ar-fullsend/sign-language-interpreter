/**
 * Sign Language Interpreter
 * MediaPipe Hand Landmarker + TensorFlow.js KNN Classifier
 * Privacy-first: all processing stays in the browser.
 */

import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// DOM
const video = document.getElementById("webcam");
const canvas = document.getElementById("output");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const sampleCountEl = document.getElementById("sample-count");

const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnSwitch = document.getElementById("btn-switch");
const btnSpeak = document.getElementById("btn-speak");
const btnAdd = document.getElementById("btn-add");
const btnClear = document.getElementById("btn-clear");
const labelInput = document.getElementById("label-input");

// State
let handLandmarker = null;
let drawingUtils = null;
let classifier = null;
let stream = null;
let running = false;
let facingMode = "user"; // or "environment"
let lastVideoTime = -1;
let currentPrediction = null;
let sampleCount = 0;

// Prediction smoothing
const PREDICTION_THRESHOLD = 0.65;
const STABLE_FRAMES = 4;
let recentPredictions = [];

// ─────────────────────────────────────────────
// MediaPipe setup
// ─────────────────────────────────────────────
async function createHandLandmarker() {
  statusEl.textContent = "Loading MediaPipe…";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  drawingUtils = new DrawingUtils(ctx);
  statusEl.textContent = "MediaPipe ready";
}

// ─────────────────────────────────────────────
// KNN Classifier
// ─────────────────────────────────────────────
function createClassifier() {
  classifier = knnClassifier.create();
  updateSampleCount();
}

function landmarksToTensor(landmarks) {
  // Flatten 21 landmarks × 3 (x,y,z) → 63 values
  // Normalize relative to wrist (landmark 0) for translation invariance
  const wrist = landmarks[0];
  const features = [];
  for (const lm of landmarks) {
    features.push(lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z);
  }
  return tf.tensor(features);
}

function updateSampleCount() {
  if (!classifier) {
    sampleCount = 0;
  } else {
    const counts = classifier.getClassExampleCount();
    sampleCount = Object.values(counts).reduce((a, b) => a + b, 0);
  }
  sampleCountEl.textContent = `${sampleCount} sample${sampleCount === 1 ? "" : "s"}`;
  btnClear.disabled = sampleCount === 0;
}

// ─────────────────────────────────────────────
// Camera
// ─────────────────────────────────────────────
async function startCamera() {
  if (stream) stopCamera();

  try {
    statusEl.textContent = "Requesting camera…";
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    // Match canvas to video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    running = true;
    btnStart.disabled = true;
    btnStop.disabled = false;
    btnSwitch.disabled = false;
    btnAdd.disabled = false;
    btnSpeak.disabled = false;

    statusEl.textContent = facingMode === "user" ? "Front camera" : "Rear camera";
    requestAnimationFrame(detectFrame);
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Camera error: " + (err.message || "permission denied");
    alert("Could not access camera. Please allow camera permissions and try again.");
  }
}

function stopCamera() {
  running = false;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  predictionEl.textContent = "—";
  confidenceEl.textContent = "";
  currentPrediction = null;
  recentPredictions = [];

  btnStart.disabled = false;
  btnStop.disabled = true;
  btnSwitch.disabled = true;
  btnAdd.disabled = true;
  btnSpeak.disabled = true;
  statusEl.textContent = "Camera off";
}

async function switchCamera() {
  facingMode = facingMode === "user" ? "environment" : "user";
  if (running) {
    await startCamera();
  }
}

// ─────────────────────────────────────────────
// Detection loop
// ─────────────────────────────────────────────
async function detectFrame() {
  if (!running || !handLandmarker) return;

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    const results = handLandmarker.detectForVideo(video, performance.now());

    // Clear + draw video is handled by the <video> element; we only overlay
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.landmarks && results.landmarks.length > 0) {
      for (const landmarks of results.landmarks) {
        drawingUtils.drawConnectors(
          landmarks,
          HandLandmarker.HAND_CONNECTIONS,
          { color: "#4f8cff", lineWidth: 3 }
        );
        drawingUtils.drawLandmarks(landmarks, {
          color: "#3dd68c",
          lineWidth: 1,
          radius: 4
        });
      }

      // Use the first detected hand for classification
      const landmarks = results.landmarks[0];
      await classify(landmarks);
    } else {
      // No hand → decay prediction
      recentPredictions.push(null);
      if (recentPredictions.length > STABLE_FRAMES) recentPredictions.shift();
      updateStablePrediction();
    }
  }

  if (running) requestAnimationFrame(detectFrame);
}

async function classify(landmarks) {
  if (!classifier || sampleCount === 0) {
    predictionEl.textContent = "Train a sign…";
    confidenceEl.textContent = "";
    return;
  }

  const tensor = landmarksToTensor(landmarks);
  try {
    const result = await classifier.predictClass(tensor);
    tensor.dispose();

    const conf = result.confidences[result.label] || 0;
    if (conf >= PREDICTION_THRESHOLD) {
      recentPredictions.push({ label: result.label, confidence: conf });
    } else {
      recentPredictions.push(null);
    }

    if (recentPredictions.length > STABLE_FRAMES) recentPredictions.shift();
    updateStablePrediction();
  } catch (e) {
    tensor.dispose();
    console.warn("Classification error", e);
  }
}

function updateStablePrediction() {
  // Require the same non-null label for the last N frames
  const valid = recentPredictions.filter(Boolean);
  if (valid.length < STABLE_FRAMES) {
    // keep previous or clear
    return;
  }

  const last = valid[valid.length - 1];
  const allSame = valid.every((p) => p.label === last.label);

  if (allSame) {
    currentPrediction = last;
    predictionEl.textContent = last.label;
    confidenceEl.textContent = `${Math.round(last.confidence * 100)}% confidence`;
  }
}

// ─────────────────────────────────────────────
// Training helpers
// ─────────────────────────────────────────────
async function addSample() {
  const label = labelInput.value.trim().toUpperCase();
  if (!label) {
    labelInput.focus();
    return;
  }
  if (!running || !handLandmarker) return;

  // Capture current frame landmarks
  const results = handLandmarker.detectForVideo(video, performance.now());
  if (!results.landmarks || results.landmarks.length === 0) {
    statusEl.textContent = "No hand detected – try again";
    return;
  }

  const landmarks = results.landmarks[0];
  const tensor = landmarksToTensor(landmarks);
  classifier.addExample(tensor, label);
  tensor.dispose();

  updateSampleCount();
  statusEl.textContent = `Added sample for "${label}"`;
  // brief visual feedback
  setTimeout(() => {
    if (running) statusEl.textContent = facingMode === "user" ? "Front camera" : "Rear camera";
  }, 900);
}

function clearSamples() {
  if (!classifier) return;
  classifier.clearAllClasses();
  updateSampleCount();
  predictionEl.textContent = "—";
  confidenceEl.textContent = "";
  currentPrediction = null;
  recentPredictions = [];
  statusEl.textContent = "All samples cleared";
}

// ─────────────────────────────────────────────
// Speech
// ─────────────────────────────────────────────
function speak() {
  if (!currentPrediction || !window.speechSynthesis) return;
  const utter = new SpeechSynthesisUtterance(currentPrediction.label);
  utter.rate = 0.95;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utter);
}

// ─────────────────────────────────────────────
// Event listeners
// ─────────────────────────────────────────────
btnStart.addEventListener("click", startCamera);
btnStop.addEventListener("click", stopCamera);
btnSwitch.addEventListener("click", switchCamera);
btnSpeak.addEventListener("click", speak);
btnAdd.addEventListener("click", addSample);
btnClear.addEventListener("click", clearSamples);

labelInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") addSample();
});

// ─────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────
(async function init() {
  try {
    createClassifier();
    await createHandLandmarker();
    statusEl.textContent = "Ready – press Start Camera";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Failed to load models";
    alert("Could not load MediaPipe / TensorFlow models. Check your connection and try again.");
  }
})();
