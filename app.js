/**
 * ASL Translator
 * MediaPipe Hand Landmarker + built-in ASL geometry + optional TF.js KNN
 * Live ML always updates; hold-to-lock commits letters into words.
 */

import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ── DOM ──────────────────────────────────────
const video = document.getElementById("webcam");
const canvas = document.getElementById("output");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const sampleCountEl = document.getElementById("sample-count");
const liveBadge = document.getElementById("live-badge");
const holdFill = document.getElementById("hold-fill");
const spellingEl = document.getElementById("spelling");
const sentenceEl = document.getElementById("sentence");
const activityLogEl = document.getElementById("activity-log");
const statLetters = document.getElementById("stat-letters");
const statWords = document.getElementById("stat-words");
const statDict = document.getElementById("stat-dict");

const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnSwitch = document.getElementById("btn-switch");
const btnSpeak = document.getElementById("btn-speak");
const btnSpace = document.getElementById("btn-space");
const btnBackspace = document.getElementById("btn-backspace");
const btnClearText = document.getElementById("btn-clear-text");
const btnClearLog = document.getElementById("btn-clear-log");
const btnAdd = document.getElementById("btn-add");
const btnClear = document.getElementById("btn-clear");
const labelInput = document.getElementById("label-input");
const btnSpeedNovice = document.getElementById("speed-novice");
const btnSpeedExpert = document.getElementById("speed-expert");
const chipHold = document.getElementById("chip-hold");

// ── State ────────────────────────────────────
let handLandmarker = null;
let drawingUtils = null;
let classifier = null;
let stream = null;
let running = false;
let facingMode = "user";
let lastVideoTime = -1;
let lastTimestamp = 0;
let currentPrediction = null;
let sampleCount = 0;
let handVisible = false;
let rafId = 0;

// Live display smoothing (short)
const DISPLAY_HISTORY = 5;
let displayVotes = [];

// Hold-to-commit timings (overridden by capture mode)
const CAPTURE_MODES = {
  // Slower lock — easier while learning
  novice: {
    holdMs: 1000,
    cooldownMs: 550,
    wordGapMs: 1300,
    spaceHoldMs: 850,
    lockMinConf: 0.72,
    label: "Novice",
    hint: "Hold a letter ~1.0s to lock it"
  },
  // Fast lock — target under half a second
  expert: {
    holdMs: 350,
    cooldownMs: 220,
    wordGapMs: 700,
    spaceHoldMs: 400,
    lockMinConf: 0.68,
    label: "Expert",
    hint: "Hold a letter ~0.35s to lock it"
  }
};

let captureMode = "novice";
let HOLD_MS = CAPTURE_MODES.novice.holdMs;
let COOLDOWN_MS = CAPTURE_MODES.novice.cooldownMs;
let WORD_GAP_MS = CAPTURE_MODES.novice.wordGapMs;
let SPACE_HOLD_MS = CAPTURE_MODES.novice.spaceHoldMs;
let LOCK_MIN_CONF = CAPTURE_MODES.novice.lockMinConf;

let holdLabel = null;
let holdStartedAt = 0;
let lastCommittedLabel = null;
let lastCommitAt = 0;
let lastHandSeenAt = 0;
let spaceHoldStartedAt = 0;
let wordFinalizedAfterGap = false;

// Composition
let spelling = "";
/** @type {{ text: string, known: boolean }[]} */
let words = [];
let letterCount = 0;
let wordCount = 0;
let dictHitCount = 0;

const DICTIONARY = new Set(
  `
  A AN THE I ME MY YOU YOUR WE US OUR HE HIM HIS SHE HER THEY THEM THEIR
  IT ITS THIS THAT THESE THOSE IS ARE WAS WERE BE BEEN BEING AM
  DO DOES DID DONE HAVE HAS HAD WILL WOULD CAN COULD SHOULD MAY MIGHT MUST
  NOT NO YES OK OKAY HI HELLO HEY BYE GOOD BAD BIG SMALL NEW OLD
  AND OR BUT IF SO AS AT THAN THEN WHEN WHERE WHY HOW WHAT WHO
  IN ON AT TO FOR OF FROM WITH BY UP OUT OFF OVER INTO ABOUT
  LOVE LIKE WANT NEED HELP PLEASE THANKS SORRY
  NAME HOME WORK FOOD WATER TIME DAY NIGHT TODAY NOW LATER
  MOM DAD FAMILY FRIEND SCHOOL TEACH LEARN READ WRITE SIGN ASL
  HAPPY SAD ANGRY TIRED FINE GREAT NICE COOL TRUE FALSE
  GO COME SEE LOOK HEAR SPEAK TALK WALK RUN STOP START OPEN CLOSE
  CAT DOG BOOK CAR HOUSE PHONE MONEY RED BLUE GREEN BLACK WHITE
  ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN
  WORLD APPLE BANANA MILK BREAD COFFEE TEA PIZZA
  SUN MOON STAR RAIN WIND FIRE EARTH
  LEFT RIGHT MORE LESS SAME DIFFERENT
  TEACHER STUDENT DOCTOR NURSE
  COMPUTER INTERNET EMAIL CODE GAME MUSIC MOVIE
  MORNING AFTERNOON EVENING WEEKEND
  `
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => w.toUpperCase())
);

const PHRASES = [
  "I LOVE YOU",
  "THANK YOU",
  "HOW ARE YOU",
  "WHAT IS YOUR NAME",
  "NICE TO MEET YOU",
  "YOU ARE WELCOME",
  "GOOD MORNING",
  "GOOD NIGHT",
  "SEE YOU LATER"
];

// ── MediaPipe ────────────────────────────────
async function createHandLandmarker() {
  statusEl.textContent = "Loading MediaPipe…";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  const modelPath =
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: modelPath, delegate },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.35,
    minHandPresenceConfidence: 0.35,
    minTrackingConfidence: 0.35
  });

  // CPU first on problematic GPUs (Surface/WSL/ARM often flake on GPU)
  let used = "CPU";
  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, opts("GPU"));
    used = "GPU";
  } catch (e) {
    console.warn("GPU MediaPipe failed, using CPU", e);
    handLandmarker = await HandLandmarker.createFromOptions(vision, opts("CPU"));
    used = "CPU";
  }

  // If GPU "succeeded" but is known flaky, keep it; we can recreate on demand later
  drawingUtils = new DrawingUtils(ctx);
  statusEl.textContent = `MediaPipe ready (${used})`;
  return used;
}

async function recreateLandmarkerCpu() {
  try {
    statusEl.textContent = "Switching MediaPipe to CPU…";
    if (handLandmarker) {
      try {
        handLandmarker.close();
      } catch (_) {
        /* ignore */
      }
    }
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    const modelPath =
      "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: modelPath, delegate: "CPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.35,
      minHandPresenceConfidence: 0.35,
      minTrackingConfidence: 0.35
    });
    lastTimestamp = 0;
    lastVideoTime = -1;
    statusEl.textContent = "MediaPipe ready (CPU)";
    addLogEntry({
      type: "system",
      icon: "CPU",
      title: "Switched to CPU tracking",
      detail: formatTime()
    });
  } catch (e) {
    console.error("CPU recreate failed", e);
  }
}

let framesWithoutHand = 0;
let triedCpuFallback = false;

// ── KNN ──────────────────────────────────────
function createClassifier() {
  if (typeof knnClassifier === "undefined") return;
  classifier = knnClassifier.create();
  updateSampleCount();
}

function landmarksToTensor(landmarks) {
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const scale =
    Math.hypot(middleMcp.x - wrist.x, middleMcp.y - wrist.y, (middleMcp.z || 0) - (wrist.z || 0)) ||
    1e-6;
  const features = [];
  for (const lm of landmarks) {
    features.push(
      (lm.x - wrist.x) / scale,
      (lm.y - wrist.y) / scale,
      ((lm.z || 0) - (wrist.z || 0)) / scale
    );
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
  if (sampleCountEl) {
    sampleCountEl.textContent = `${sampleCount} sample${sampleCount === 1 ? "" : "s"}`;
  }
  if (btnClear) btnClear.disabled = sampleCount === 0;
}

// ── Geometry + polished similar-sign ML ───────
// Continuous finger features + scored templates + confusion-group
// disambiguation + temporal hysteresis (reduces A/S/E, U/V/R, O/C flicker).

function dist2(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

/** Map value in [lo,hi] → 0..1 (clamped). */
function normRange(v, lo, hi) {
  if (hi <= lo) return 0;
  return clamp01((v - lo) / (hi - lo));
}

/** Soft peak score: 1 at ideal, falls off toward lo/hi edges. */
function peakScore(v, ideal, tol) {
  return clamp01(1 - Math.abs(v - ideal) / tol);
}

// Temporal hysteresis state for similar-sign stability
let stickyLabel = null;
let stickyKind = "letter";
let stickyScore = 0;
let stickyFrames = 0;
let challengerLabel = null;
let challengerFrames = 0;

function resetRecognitionMemory() {
  stickyLabel = null;
  stickyKind = "letter";
  stickyScore = 0;
  stickyFrames = 0;
  challengerLabel = null;
  challengerFrames = 0;
}

/**
 * Continuous 0..1 extension: tip vs PIP distance from wrist.
 * ~0 fist curl, ~1 fully extended.
 */
function fingerExtension(lm, tipIdx, pipIdx, scale) {
  const wrist = lm[0];
  const tip = lm[tipIdx];
  const pip = lm[pipIdx];
  const tipD = dist2(tip, wrist) / scale;
  const pipD = dist2(pip, wrist) / scale;
  // Ratio tip/pip: extended ~1.25+, curled ~0.85-
  const ratio = tipD / (pipD + 1e-6);
  return clamp01((ratio - 0.85) / 0.5);
}

/**
 * Extract rich features for disambiguating similar ASL letters.
 */
function extractHandFeatures(lm) {
  const wrist = lm[0];
  const scale = dist2(wrist, lm[9]) || 1e-6;
  const n = (a, b) => dist2(a, b) / scale;

  const thumbTip = lm[4];
  const thumbIp = lm[3];
  const thumbMcp = lm[2];
  const indexTip = lm[8];
  const middleTip = lm[12];
  const ringTip = lm[16];
  const pinkyTip = lm[20];
  const indexMcp = lm[5];
  const middleMcp = lm[9];
  const ringMcp = lm[13];
  const pinkyMcp = lm[17];
  const indexPip = lm[6];
  const middlePip = lm[10];
  const ringPip = lm[14];
  const pinkyPip = lm[18];

  const palm = {
    x: (indexMcp.x + pinkyMcp.x) / 2,
    y: (indexMcp.y + pinkyMcp.y) / 2
  };

  const index = fingerExtension(lm, 8, 6, scale);
  const middle = fingerExtension(lm, 12, 10, scale);
  const ring = fingerExtension(lm, 16, 14, scale);
  const pinky = fingerExtension(lm, 20, 18, scale);

  // Thumb: openness away from palm + how high vs fist
  const thumbOpen = clamp01((n(thumbTip, palm) - n(thumbIp, palm) + 0.15) / 0.55);
  const thumbBeside = clamp01(1 - n(thumbTip, indexMcp) / 0.55); // near index MCP = A-like
  const thumbOverFingers = clamp01(1 - n(thumbTip, indexPip) / 0.4); // S-like
  const thumbBetweenIM = clamp01(
    1 - Math.min(n(thumbTip, indexPip), n(thumbTip, middlePip)) / 0.38
  ); // T-like
  const thumbUnder3 = clamp01(1 - n(thumbTip, ringPip) / 0.45); // M-ish
  const thumbUnder2 = clamp01(1 - n(thumbTip, middlePip) / 0.42); // N-ish

  const indexMiddleSpread = n(indexTip, middleTip);
  const middleRingSpread = n(middleTip, ringTip);
  const ringPinkySpread = n(ringTip, pinkyTip);
  const thumbIndexSpread = n(thumbTip, indexTip);
  const thumbMiddleDist = n(thumbTip, middleTip);
  const thumbIndexDist = n(thumbTip, indexTip);

  // Circle metrics (O vs C vs fist)
  const tips = [indexTip, middleTip, ringTip, pinkyTip];
  const tipToThumb = tips.map((t) => n(t, thumbTip));
  const circleAvg = tipToThumb.reduce((a, b) => a + b, 0) / 4;
  const circleMax = Math.max(...tipToThumb);
  const circleMin = Math.min(...tipToThumb);
  const circleVar = circleMax - circleMin;
  const tipCluster =
    (n(indexTip, middleTip) + n(middleTip, ringTip) + n(ringTip, pinkyTip)) / 3;
  const tipFromMcp =
    (n(indexTip, indexMcp) + n(middleTip, middleMcp) + n(ringTip, ringMcp) + n(pinkyTip, pinkyMcp)) /
    4;

  // Index direction (D upright vs G sideways)
  const indexDx = Math.abs(indexTip.x - indexMcp.x);
  const indexDy = Math.abs(indexTip.y - indexMcp.y);
  const indexHorizontal = indexDx / (indexDx + indexDy + 1e-6);

  // Hook for X: index partially extended
  const indexHook = index > 0.25 && index < 0.65 && middle < 0.35;

  const meanExt = (index + middle + ring + pinky) / 4;
  const extBits = {
    index: index > 0.55,
    middle: middle > 0.55,
    ring: ring > 0.55,
    pinky: pinky > 0.55,
    thumb: thumbOpen > 0.45
  };
  const extCount = [extBits.index, extBits.middle, extBits.ring, extBits.pinky].filter(
    Boolean
  ).length;

  return {
    scale,
    n,
    index,
    middle,
    ring,
    pinky,
    thumbOpen,
    thumbBeside,
    thumbOverFingers,
    thumbBetweenIM,
    thumbUnder3,
    thumbUnder2,
    indexMiddleSpread,
    middleRingSpread,
    ringPinkySpread,
    thumbIndexSpread,
    thumbMiddleDist,
    thumbIndexDist,
    circleAvg,
    circleMax,
    circleMin,
    circleVar,
    tipCluster,
    tipFromMcp,
    indexHorizontal,
    indexHook,
    meanExt,
    extBits,
    extCount,
    lm,
    thumbTip,
    indexTip,
    middleTip,
    ringTip,
    pinkyTip,
    indexMcp,
    middleMcp,
    indexPip,
    middlePip,
    ringPip,
    pinkyPip
  };
}

/**
 * Score how well continuous finger extensions match a target pattern.
 * pattern: { index, middle, ring, pinky, thumb } each 0..1 ideal, weight optional.
 */
function scoreFingerPattern(feat, pattern, tol = 0.45) {
  const keys = ["index", "middle", "ring", "pinky"];
  let sum = 0;
  let wsum = 0;
  for (const k of keys) {
    if (pattern[k] === undefined) continue;
    const w = pattern.w?.[k] ?? 1;
    sum += w * peakScore(feat[k], pattern[k], tol);
    wsum += w;
  }
  if (pattern.thumb !== undefined) {
    const w = pattern.w?.thumb ?? 1.1;
    sum += w * peakScore(feat.thumbOpen, pattern.thumb, tol);
    wsum += w;
  }
  return wsum ? sum / wsum : 0;
}

function kindForLabel(label) {
  if (label === "5" || label === "OPEN") return "space";
  if (label === "1" || label === "3" || label === "4") return "number";
  return "letter";
}

/**
 * Score every letter; return ranked list. Heavy focus on similar-sign separation.
 */
function scoreAllLetters(feat) {
  /** @type {{ label: string, score: number }[]} */
  const scores = [];

  const add = (label, score) => {
    if (score > 0.15) scores.push({ label, score: clamp01(score) });
  };

  // ── Distinctive / easy shapes ─────────────────
  // Y: thumb+pinky up, others down
  add(
    "Y",
    scoreFingerPattern(feat, { index: 0, middle: 0, ring: 0, pinky: 1, thumb: 1 }, 0.4) *
      (feat.pinky > 0.55 && feat.index < 0.4 ? 1 : 0.5)
  );

  // I: pinky only
  add(
    "I",
    scoreFingerPattern(feat, { index: 0, middle: 0, ring: 0, pinky: 1, thumb: 0.2 }, 0.4) *
      (feat.pinky > 0.55 && feat.index < 0.35 && feat.middle < 0.35 ? 1.05 : 0.45)
  );

  // L: index + thumb, wide angle
  {
    let s = scoreFingerPattern(feat, { index: 1, middle: 0, ring: 0, pinky: 0, thumb: 1 }, 0.4);
    s *= peakScore(feat.thumbIndexSpread, 0.75, 0.4);
    s *= feat.index > 0.55 && feat.middle < 0.4 ? 1 : 0.4;
    add("L", s);
  }

  // ── Two-finger family: U V H R K (hard cluster) ─
  if (feat.index > 0.45 && feat.middle > 0.45 && feat.ring < 0.5 && feat.pinky < 0.5) {
    const spread = feat.indexMiddleSpread;
    // V: clear spread
    add(
      "V",
      scoreFingerPattern(feat, { index: 1, middle: 1, ring: 0, pinky: 0, thumb: 0.25 }, 0.4) *
        peakScore(spread, 0.5, 0.28) *
        (spread >= 0.32 ? 1.08 : 0.55)
    );
    // U: together, thumb in
    add(
      "U",
      scoreFingerPattern(feat, { index: 1, middle: 1, ring: 0, pinky: 0, thumb: 0.15 }, 0.4) *
        peakScore(spread, 0.18, 0.2) *
        (spread < 0.3 && feat.thumbOpen < 0.45 ? 1.1 : 0.5)
    );
    // H: like U but often more horizontal / thumb out a bit — treat as U-sideways proxy
    add(
      "H",
      scoreFingerPattern(feat, { index: 1, middle: 1, ring: 0, pinky: 0, thumb: 0.55 }, 0.45) *
        peakScore(spread, 0.2, 0.22) *
        (feat.thumbOpen > 0.4 && spread < 0.32 ? 1.05 : 0.45)
    );
    // R: tips very close / crossed
    add(
      "R",
      scoreFingerPattern(feat, { index: 0.9, middle: 0.9, ring: 0, pinky: 0, thumb: 0.2 }, 0.45) *
        peakScore(spread, 0.12, 0.15) *
        (spread < 0.22 ? 1.05 : 0.4)
    );
    // K: V-ish with thumb between (touching middle)
    add(
      "K",
      scoreFingerPattern(feat, { index: 1, middle: 1, ring: 0, pinky: 0, thumb: 0.7 }, 0.45) *
        peakScore(spread, 0.35, 0.25) *
        peakScore(feat.thumbMiddleDist, 0.28, 0.25) *
        (feat.thumbOpen > 0.4 && spread > 0.22 && spread < 0.55 ? 1.05 : 0.45)
    );
  }

  // W / 3
  if (feat.index > 0.5 && feat.middle > 0.5 && feat.ring > 0.5 && feat.pinky < 0.45) {
    if (feat.thumbOpen > 0.4) {
      add("W", scoreFingerPattern(feat, { index: 1, middle: 1, ring: 1, pinky: 0, thumb: 0.7 }, 0.4));
    } else {
      add("3", scoreFingerPattern(feat, { index: 1, middle: 1, ring: 1, pinky: 0, thumb: 0.15 }, 0.4));
    }
  }

  // F: OK circle + 3 up
  {
    let s = scoreFingerPattern(feat, { index: 0.15, middle: 1, ring: 1, pinky: 1, thumb: 0.6 }, 0.4);
    s *= peakScore(feat.thumbIndexDist, 0.18, 0.22);
    s *= feat.middle > 0.55 && feat.ring > 0.55 && feat.pinky > 0.5 ? 1.05 : 0.4;
    add("F", s);
  }

  // B vs 5 (open hand family)
  {
    const openFingers = scoreFingerPattern(
      feat,
      { index: 1, middle: 1, ring: 1, pinky: 1 },
      0.38
    );
    add("B", openFingers * peakScore(feat.thumbOpen, 0.15, 0.35) * (feat.thumbOpen < 0.4 ? 1.1 : 0.5));
    add("5", openFingers * peakScore(feat.thumbOpen, 0.85, 0.35) * (feat.thumbOpen > 0.45 ? 1.1 : 0.5));
  }

  // ── Index-only family: D G X 1 ──
  if (feat.index > 0.45 && feat.middle < 0.45 && feat.ring < 0.45 && feat.pinky < 0.45) {
    const upright = 1 - feat.indexHorizontal;
    add(
      "1",
      scoreFingerPattern(feat, { index: 1, middle: 0, ring: 0, pinky: 0, thumb: 0.1 }, 0.4) *
        peakScore(feat.thumbOpen, 0.15, 0.35) *
        upright
    );
    add(
      "D",
      scoreFingerPattern(feat, { index: 1, middle: 0, ring: 0, pinky: 0, thumb: 0.55 }, 0.4) *
        upright *
        (feat.thumbOpen > 0.35 ? 1.05 : 0.55) *
        // D: thumb near middle finger curl
        peakScore(Math.min(feat.thumbMiddleDist, feat.n(feat.thumbTip, feat.middlePip)), 0.25, 0.3)
    );
    add(
      "G",
      scoreFingerPattern(feat, { index: 1, middle: 0, ring: 0, pinky: 0, thumb: 0.7 }, 0.45) *
        peakScore(feat.indexHorizontal, 0.75, 0.35) *
        (feat.indexHorizontal > 0.55 ? 1.1 : 0.45)
    );
    if (feat.indexHook) {
      add(
        "X",
        scoreFingerPattern(feat, { index: 0.45, middle: 0, ring: 0, pinky: 0, thumb: 0.3 }, 0.4) * 0.95
      );
    }
  }

  // ── Circle family: O vs C (strict separation) ──
  {
    const curled =
      feat.index < 0.45 && feat.middle < 0.45 && feat.ring < 0.45 && feat.pinky < 0.45;
    if (curled) {
      // O: tips form tight ring on thumb, mid extension from MCP (hole)
      let o =
        peakScore(feat.circleAvg, 0.22, 0.16) *
        peakScore(feat.circleVar, 0.1, 0.15) *
        peakScore(feat.tipFromMcp, 0.42, 0.2) *
        peakScore(feat.tipCluster, 0.22, 0.18) *
        (feat.circleMax < 0.4 ? 1.05 : 0.5) *
        (feat.tipFromMcp > 0.28 ? 1.05 : 0.45);
      // Penalize if looks like fist (tips too close to MCP)
      if (feat.tipFromMcp < 0.25) o *= 0.35;
      // Penalize if thumb is "beside fist" A-style more than circle
      if (feat.thumbBeside > 0.7 && feat.circleAvg > 0.28) o *= 0.5;
      add("O", o);

      // C: larger open gap, still curved
      let c =
        peakScore(feat.thumbIndexDist, 0.65, 0.28) *
        peakScore(feat.circleAvg, 0.55, 0.25) *
        peakScore(feat.tipFromMcp, 0.45, 0.22) *
        (feat.thumbIndexDist > 0.42 && feat.thumbIndexDist < 1.0 ? 1.05 : 0.4) *
        (feat.thumbOpen > 0.35 ? 1 : 0.55);
      // Must not be tight O
      if (feat.circleAvg < 0.32) c *= 0.4;
      add("C", c);
    }
  }

  // ── Fist family: A E S T M N (hardest cluster) ──
  if (feat.meanExt < 0.4 && feat.extCount === 0) {
    const fistTight = peakScore(feat.tipFromMcp, 0.22, 0.2) * (feat.tipFromMcp < 0.4 ? 1 : 0.5);

    // A: fist + thumb alongside (not over fingers)
    add(
      "A",
      fistTight *
        peakScore(feat.thumbOpen, 0.55, 0.35) *
        peakScore(feat.thumbBeside, 0.75, 0.35) *
        (1 - feat.thumbOverFingers * 0.7) *
        (feat.thumbOpen > 0.35 ? 1.05 : 0.55)
    );

    // E: fist, thumb tucked, fingertips on thumb side
    add(
      "E",
      fistTight *
        peakScore(feat.thumbOpen, 0.15, 0.3) *
        (1 - feat.thumbBeside * 0.5) *
        peakScore(feat.circleAvg, 0.35, 0.25) *
        (feat.thumbOpen < 0.4 ? 1.05 : 0.5)
    );

    // S: fist, thumb over closed fingers
    add(
      "S",
      fistTight *
        peakScore(feat.thumbOverFingers, 0.8, 0.35) *
        peakScore(feat.thumbOpen, 0.25, 0.35) *
        (feat.thumbOverFingers > 0.45 ? 1.1 : 0.5)
    );

    // T: thumb between index & middle
    add(
      "T",
      fistTight *
        peakScore(feat.thumbBetweenIM, 0.8, 0.3) *
        peakScore(feat.thumbOpen, 0.35, 0.35) *
        (feat.thumbBetweenIM > 0.5 ? 1.08 : 0.45)
    );

    // M: thumb under 3 fingers (toward pinky side)
    add(
      "M",
      fistTight *
        peakScore(feat.thumbUnder3, 0.75, 0.35) *
        peakScore(feat.thumbOpen, 0.2, 0.3) *
        (feat.thumbUnder3 > 0.5 && feat.thumbUnder3 >= feat.thumbUnder2 ? 1.05 : 0.45)
    );

    // N: thumb under 2 fingers
    add(
      "N",
      fistTight *
        peakScore(feat.thumbUnder2, 0.75, 0.35) *
        peakScore(feat.thumbOpen, 0.2, 0.3) *
        (feat.thumbUnder2 > 0.5 && feat.thumbUnder2 > feat.thumbUnder3 + 0.05 ? 1.05 : 0.45)
    );
  }

  // Sort high → low
  scores.sort((a, b) => b.score - a.score);
  return scores;
}

/**
 * Within a confusion group, re-rank using specialized margins so similar
 * signs don't flip on noise.
 */
function disambiguateSimilar(ranked, feat) {
  if (!ranked.length) return ranked;

  const by = Object.fromEntries(ranked.map((r) => [r.label, r.score]));
  const bump = (label, delta) => {
    if (by[label] === undefined) return;
    by[label] = clamp01(by[label] + delta);
  };

  // Fist group: pick single winner with clearer thumb geometry
  const fist = ["A", "E", "S", "T", "M", "N"];
  const fistPresent = fist.filter((l) => by[l] !== undefined);
  if (fistPresent.length >= 2 && feat.meanExt < 0.42) {
    // Strong priors from thumb placement
    if (feat.thumbOpen > 0.45 && feat.thumbBeside > 0.55 && feat.thumbOverFingers < 0.45) {
      bump("A", 0.12);
      bump("S", -0.1);
      bump("E", -0.08);
    } else if (feat.thumbOverFingers > 0.55) {
      bump("S", 0.14);
      bump("A", -0.1);
      bump("T", -0.05);
    } else if (feat.thumbBetweenIM > 0.55 && feat.thumbOpen < 0.5) {
      bump("T", 0.14);
      bump("A", -0.08);
      bump("S", -0.06);
    } else if (feat.thumbOpen < 0.35) {
      bump("E", 0.1);
      bump("A", -0.08);
    }
    if (feat.thumbUnder3 > feat.thumbUnder2 + 0.08 && feat.thumbOpen < 0.4) {
      bump("M", 0.12);
      bump("N", -0.06);
    } else if (feat.thumbUnder2 > feat.thumbUnder3 + 0.05 && feat.thumbOpen < 0.4) {
      bump("N", 0.12);
      bump("M", -0.06);
    }
  }

  // Two-finger: V vs U vs R vs K vs H
  const two = ["V", "U", "R", "K", "H"];
  if (two.some((l) => by[l] !== undefined)) {
    const sp = feat.indexMiddleSpread;
    if (sp >= 0.34) {
      bump("V", 0.12);
      bump("U", -0.12);
      bump("R", -0.1);
      bump("H", -0.08);
    } else if (sp <= 0.2) {
      bump("U", 0.08);
      bump("R", sp < 0.15 ? 0.1 : 0.02);
      bump("V", -0.14);
      if (feat.thumbOpen > 0.45) {
        bump("H", 0.08);
        bump("U", -0.05);
      }
    }
    if (feat.thumbOpen > 0.5 && sp > 0.22 && sp < 0.5 && feat.thumbMiddleDist < 0.4) {
      bump("K", 0.12);
      bump("V", -0.06);
    }
  }

  // O vs C
  if (by.O !== undefined || by.C !== undefined) {
    if (feat.circleAvg < 0.3 && feat.tipFromMcp > 0.3) {
      bump("O", 0.1);
      bump("C", -0.15);
    } else if (feat.thumbIndexDist > 0.48) {
      bump("C", 0.12);
      bump("O", -0.15);
    }
    // Fist-like → kill O/C
    if (feat.tipFromMcp < 0.24) {
      bump("O", -0.25);
      bump("C", -0.15);
    }
  }

  // B vs 5
  if (by.B !== undefined || by["5"] !== undefined) {
    if (feat.thumbOpen > 0.45) {
      bump("5", 0.1);
      bump("B", -0.12);
    } else {
      bump("B", 0.1);
      bump("5", -0.12);
    }
  }

  // D vs G vs 1
  if (by.D !== undefined || by.G !== undefined || by["1"] !== undefined) {
    if (feat.indexHorizontal > 0.58) {
      bump("G", 0.12);
      bump("D", -0.1);
      bump("1", -0.08);
    } else if (feat.thumbOpen < 0.3) {
      bump("1", 0.1);
      bump("D", -0.08);
    } else {
      bump("D", 0.08);
      bump("1", -0.06);
    }
  }

  // Rebuild ranked
  return Object.entries(by)
    .map(([label, score]) => ({ label, score }))
    .sort((a, b) => b.score - a.score);
}

/**
 * Require a clear margin between #1 and #2 for confusable pairs.
 */
function applyMarginGate(ranked) {
  if (ranked.length < 2) return ranked;
  const confusable = new Set([
    "A|S",
    "A|E",
    "A|T",
    "S|T",
    "S|E",
    "E|T",
    "M|N",
    "U|V",
    "U|R",
    "V|K",
    "U|H",
    "O|C",
    "O|E",
    "B|5",
    "D|1",
    "D|G",
    "W|3"
  ]);
  const a = ranked[0];
  const b = ranked[1];
  const pair = [a.label, b.label].sort().join("|");
  const need = confusable.has(pair) ? 0.06 : 0.03;
  if (a.score - b.score < need) {
    // Ambiguous: keep leader but lower confidence so we don't lock
    return [{ label: a.label, score: a.score * 0.85, ambiguous: true }, ...ranked.slice(1)];
  }
  return ranked;
}

/**
 * Temporal hysteresis: don't flip to a similar letter without sustained evidence.
 */
function applyHysteresis(label, score, kind) {
  const SIMILAR = {
    A: ["E", "S", "T", "M", "N"],
    E: ["A", "S", "T", "O"],
    S: ["A", "E", "T"],
    T: ["A", "S", "N"],
    M: ["N", "A", "S"],
    N: ["M", "T", "S"],
    U: ["V", "R", "H", "K"],
    V: ["U", "K", "R"],
    R: ["U", "V"],
    K: ["V", "U"],
    H: ["U", "V"],
    O: ["C", "E", "A"],
    C: ["O", "E"],
    B: ["5"],
    "5": ["B"],
    D: ["1", "G", "X"],
    G: ["D", "1"],
    "1": ["D", "G"],
    W: ["3"],
    "3": ["W"]
  };

  if (!stickyLabel) {
    stickyLabel = label;
    stickyKind = kind;
    stickyScore = score;
    stickyFrames = 1;
    challengerLabel = null;
    challengerFrames = 0;
    return { label, score, kind };
  }

  if (label === stickyLabel) {
    stickyFrames += 1;
    stickyScore = stickyScore * 0.7 + score * 0.3;
    challengerLabel = null;
    challengerFrames = 0;
    return { label: stickyLabel, score: Math.max(stickyScore, score), kind: stickyKind };
  }

  const related = SIMILAR[stickyLabel] || [];
  const isSimilar = related.includes(label);
  // Need stronger / longer evidence to switch among similar signs
  const needFrames = isSimilar ? 4 : 2;
  const needMargin = isSimilar ? 0.08 : 0.04;

  if (label === challengerLabel) {
    challengerFrames += 1;
  } else {
    challengerLabel = label;
    challengerFrames = 1;
  }

  if (challengerFrames >= needFrames && score >= stickyScore + needMargin) {
    stickyLabel = label;
    stickyKind = kind;
    stickyScore = score;
    stickyFrames = 1;
    challengerLabel = null;
    challengerFrames = 0;
    return { label, score, kind };
  }

  // Hold sticky prediction while challenger builds up
  return {
    label: stickyLabel,
    score: stickyScore * 0.95,
    kind: stickyKind,
    pending: label
  };
}

/**
 * Main recognizer entry: features → scores → disambiguate → margin → hysteresis.
 */
function recognizeAsl(lm) {
  const feat = extractHandFeatures(lm);
  let ranked = scoreAllLetters(feat);
  ranked = disambiguateSimilar(ranked, feat);
  ranked = applyMarginGate(ranked);

  if (!ranked.length || ranked[0].score < 0.38) {
    // Don't immediately clear sticky on one weak frame
    if (stickyLabel && stickyFrames > 2) {
      stickyFrames -= 1;
      return {
        label: stickyLabel,
        confidence: Math.max(0.4, stickyScore * 0.9),
        kind: stickyKind,
        sourceHint: "sticky"
      };
    }
    resetRecognitionMemory();
    return { label: "…", confidence: 0.3, kind: "unknown" };
  }

  const top = ranked[0];
  const kind = kindForLabel(top.label);
  const stable = applyHysteresis(top.label, top.score, kind);

  // Confidence: blend score + margin over runner-up
  const second = ranked[1]?.score ?? 0;
  const margin = top.score - second;
  let confidence = clamp01(stable.score * 0.75 + margin * 0.9 + 0.15);
  if (top.ambiguous) confidence *= 0.85;
  if (stable.pending) confidence *= 0.92;

  // Cap O confidence unless score is excellent (historical over-trigger)
  if (stable.label === "O" && top.score < 0.7) confidence = Math.min(confidence, 0.7);

  return {
    label: stable.label,
    confidence,
    kind: stable.kind,
    alt: ranked[1] ? ranked[1].label : null,
    pending: stable.pending || null,
    margin
  };
}

// ── Activity / text UI ───────────────────────
function formatTime(d = new Date()) {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function addLogEntry({ type, icon, title, detail, known }) {
  const empty = document.getElementById("log-empty");
  if (empty) empty.remove();

  const item = document.createElement("div");
  item.className = `log-item ${type}${known ? " known" : ""}`;
  item.innerHTML = `
    <div class="log-icon">${escapeHtml(String(icon).slice(0, 4))}</div>
    <div class="log-main"><span class="tag">${escapeHtml(type)}</span>${escapeHtml(title)}</div>
    <div class="log-meta">${escapeHtml(detail || formatTime())}</div>
  `;
  activityLogEl.prepend(item);
  while (activityLogEl.children.length > 80) {
    activityLogEl.removeChild(activityLogEl.lastChild);
  }
}

function updateStats() {
  statLetters.textContent = String(letterCount);
  statWords.textContent = String(wordCount);
  statDict.textContent = String(dictHitCount);
}

function updateComposeUI() {
  if (spelling.length === 0) {
    spellingEl.innerHTML = `<span class="placeholder">Letters appear here…</span>`;
  } else {
    spellingEl.innerHTML = `${escapeHtml(spelling)}<span class="cursor"></span>`;
  }

  if (words.length === 0) {
    sentenceEl.innerHTML = `<span class="placeholder">Completed words appear here…</span>`;
  } else {
    sentenceEl.innerHTML = words
      .map(
        (w) =>
          `<span class="word-chip${w.known ? "" : " unknown"}">${escapeHtml(w.text)}</span>`
      )
      .join("");
  }

  const hasText = spelling.length > 0 || words.length > 0;
  btnSpace.disabled = !running || spelling.length === 0;
  btnBackspace.disabled = !running || spelling.length === 0;
  btnClearText.disabled = !hasText;
  btnSpeak.disabled = !hasText;
}

function isDictionaryWord(text) {
  return DICTIONARY.has(text.toUpperCase());
}

function isLetterLabel(label) {
  return typeof label === "string" && /^[A-Z]$/i.test(label);
}

function isWholeWordLabel(label) {
  return typeof label === "string" && /^[A-Z]{2,}$/i.test(label.replace(/\s+/g, ""));
}

function commitLetter(letter, confidence) {
  const ch = letter.toUpperCase();
  if (!/^[A-Z]$/.test(ch)) return;

  const now = performance.now();
  if (lastCommittedLabel === ch && now - lastCommitAt < COOLDOWN_MS) return;

  spelling += ch;
  letterCount += 1;
  lastCommittedLabel = ch;
  lastCommitAt = now;
  wordFinalizedAfterGap = false;

  addLogEntry({
    type: "letter",
    icon: ch,
    title: `Locked “${ch}”`,
    detail: `${Math.round(confidence * 100)}% · “${spelling}” · ${formatTime()}`
  });
  updateStats();
  updateComposeUI();
  flashStatus(`+${ch}`);
}

function commitWholeWordGesture(label, confidence) {
  const text = label.toUpperCase().replace(/[^A-Z ]/g, "").trim();
  if (!text || text.length < 2) return;
  if (spelling.length > 0) finalizeWord("before gesture");

  const known = isDictionaryWord(text) || PHRASES.includes(text);
  words.push({ text, known });
  wordCount += 1;
  if (known) dictHitCount += 1;

  addLogEntry({
    type: "word",
    icon: known ? "✓" : "W",
    title: `Word “${text}”`,
    detail: `${Math.round(confidence * 100)}% · gesture · ${formatTime()}`,
    known
  });
  updateStats();
  updateComposeUI();
  flashStatus(`Word: ${text}`);
}

function finalizeWord(reason = "pause") {
  if (!spelling) return;
  const text = spelling.toUpperCase();
  spelling = "";
  const known = isDictionaryWord(text);
  words.push({ text, known });
  wordCount += 1;
  if (known) dictHitCount += 1;
  maybeMergePhrase();

  addLogEntry({
    type: "word",
    icon: known ? "✓" : "?",
    title: known ? `Word “${text}”` : `Spelled “${text}”`,
    detail: `${reason} · ${known ? "dictionary" : "unknown"} · ${formatTime()}`,
    known
  });
  lastCommittedLabel = null;
  wordFinalizedAfterGap = true;
  updateStats();
  updateComposeUI();
  flashStatus(known ? `Word: ${text}` : `Spelled: ${text}`);
}

function maybeMergePhrase() {
  if (words.length < 2) return;
  for (let n = Math.min(4, words.length); n >= 2; n--) {
    const joined = words
      .slice(-n)
      .map((w) => w.text)
      .join(" ");
    if (PHRASES.includes(joined)) {
      words.splice(-n, n, { text: joined, known: true });
      dictHitCount += 1;
      addLogEntry({
        type: "word",
        icon: "✓",
        title: `Phrase “${joined}”`,
        detail: `merged · ${formatTime()}`,
        known: true
      });
      return;
    }
  }
}

function backspaceLetter() {
  if (!spelling) return;
  const removed = spelling.slice(-1);
  spelling = spelling.slice(0, -1);
  letterCount = Math.max(0, letterCount - 1);
  addLogEntry({ type: "system", icon: "⌫", title: `Removed “${removed}”`, detail: formatTime() });
  updateStats();
  updateComposeUI();
}

function clearText() {
  spelling = "";
  words = [];
  updateComposeUI();
  addLogEntry({ type: "system", icon: "CLR", title: "Cleared sentence", detail: formatTime() });
}

function clearLog() {
  activityLogEl.innerHTML =
    '<div class="log-empty" id="log-empty">Log cleared.<br />Fingerspell again — letters and words show up here.</div>';
  letterCount = 0;
  wordCount = 0;
  dictHitCount = 0;
  updateStats();
}

let statusFlashTimer = null;
function flashStatus(msg) {
  if (!running) return;
  statusEl.textContent = msg;
  clearTimeout(statusFlashTimer);
  statusFlashTimer = setTimeout(() => {
    if (!running) return;
    statusEl.textContent =
      (facingMode === "user" ? "Front" : "Rear") +
      (handVisible ? " · hand detected" : " · no hand");
  }, 800);
}

// ── Live prediction + hold commit ────────────
function showLive(pred) {
  currentPrediction = pred;
  if (!pred) {
    predictionEl.textContent = handVisible ? "…" : "—";
    confidenceEl.textContent = handVisible ? "Reading hand…" : "Show your hand";
    holdFill.style.width = "0%";
    return;
  }
  predictionEl.textContent = pred.label;
  const src = pred.source === "trained" ? "custom" : "live ASL";
  let meta = `${Math.round(pred.confidence * 100)}% · ${src}`;
  if (pred.alt && pred.alt !== pred.label) {
    meta += ` · vs ${pred.alt}`;
  }
  if (pred.pending && pred.pending !== pred.label) {
    meta += ` · checking ${pred.pending}`;
  }
  confidenceEl.textContent = meta;
}

/**
 * Light display vote on top of recognizer hysteresis.
 * Prefer sticky recognizer output; only smooth single-frame glitches.
 */
function voteDisplay(pred) {
  if (!pred || pred.label === "…") {
    displayVotes.push(null);
    if (displayVotes.length > DISPLAY_HISTORY) displayVotes.shift();
    return pred;
  }
  displayVotes.push(pred.label);
  if (displayVotes.length > DISPLAY_HISTORY) displayVotes.shift();

  // If last 3 frames mostly agree with current, keep current (hysteresis already did work)
  const recent = displayVotes.filter(Boolean).slice(-3);
  if (recent.length >= 2) {
    const same = recent.filter((l) => l === pred.label).length;
    if (same >= 2) return pred;
  }
  return pred;
}

function resetHold() {
  holdLabel = null;
  holdStartedAt = 0;
  spaceHoldStartedAt = 0;
  holdFill.style.width = "0%";
}

function applyCaptureMode(mode, { silent = false } = {}) {
  if (!CAPTURE_MODES[mode]) mode = "novice";
  captureMode = mode;
  const cfg = CAPTURE_MODES[mode];
  HOLD_MS = cfg.holdMs;
  COOLDOWN_MS = cfg.cooldownMs;
  WORD_GAP_MS = cfg.wordGapMs;
  SPACE_HOLD_MS = cfg.spaceHoldMs;
  LOCK_MIN_CONF = cfg.lockMinConf;

  // Reset in-progress hold so timing doesn't jump mid-lock
  resetHold();

  if (btnSpeedNovice && btnSpeedExpert) {
    btnSpeedNovice.classList.toggle("active", mode === "novice");
    btnSpeedExpert.classList.toggle("active", mode === "expert");
    btnSpeedNovice.setAttribute("aria-pressed", mode === "novice" ? "true" : "false");
    btnSpeedExpert.setAttribute("aria-pressed", mode === "expert" ? "true" : "false");
  }
  if (chipHold) chipHold.textContent = cfg.hint;

  try {
    localStorage.setItem("asl-capture-mode", mode);
  } catch (_) {
    /* ignore */
  }

  if (!silent) {
    addLogEntry({
      type: "system",
      icon: mode === "expert" ? "⚡" : "🐢",
      title: `${cfg.label} capture`,
      detail: `Lock in ~${(cfg.holdMs / 1000).toFixed(2)}s · ${formatTime()}`
    });
    if (running) {
      flashStatus(`${cfg.label}: ${(cfg.holdMs / 1000).toFixed(2)}s lock`);
    } else if (statusEl && !statusEl.textContent.startsWith("Failed")) {
      // keep ready message informative
    }
  }
}

function processHold(pred) {
  if (!pred || pred.label === "…" || pred.kind === "unknown") {
    resetHold();
    return;
  }

  const now = performance.now();
  const label = pred.label.toUpperCase();

  // Space: open hand only ends a word if we are spelling
  if (pred.kind === "space" || label === "5") {
    holdLabel = null;
    holdStartedAt = 0;
    if (spelling.length === 0) {
      holdFill.style.width = "0%";
      confidenceEl.textContent = `${Math.round(pred.confidence * 100)}% · open hand (space when spelling)`;
      return;
    }
    if (!spaceHoldStartedAt) spaceHoldStartedAt = now;
    const p = Math.min(1, (now - spaceHoldStartedAt) / SPACE_HOLD_MS);
    holdFill.style.width = `${p * 100}%`;
    confidenceEl.textContent = `Space ${Math.round(p * 100)}% · hold open hand`;
    if (p >= 1) {
      finalizeWord("open hand");
      spaceHoldStartedAt = 0;
      lastCommittedLabel = "__SPACE__";
      lastCommitAt = now;
      holdFill.style.width = "0%";
    }
    return;
  }

  spaceHoldStartedAt = 0;

  // Don't lock weak / ambiguous reads (stops O spam from partial poses)
  if (pred.confidence < LOCK_MIN_CONF) {
    resetHold();
    confidenceEl.textContent = `${Math.round(pred.confidence * 100)}% · hold clearer pose to lock`;
    return;
  }

  // Trained whole words
  if (pred.source === "trained" && isWholeWordLabel(label)) {
    tickHold(label, pred, now, () => {
      commitWholeWordGesture(label, pred.confidence);
      resetHold();
    });
    return;
  }

  // Letters only (numbers show live but don't auto-spell)
  if (isLetterLabel(label) && pred.kind === "letter") {
    tickHold(label, pred, now, () => {
      commitLetter(label, pred.confidence);
      resetHold();
    });
    return;
  }

  // Numbers / other — show only, don't spell
  holdFill.style.width = "0%";
}

function tickHold(label, pred, now, onDone) {
  if (label === lastCommittedLabel && now - lastCommitAt < COOLDOWN_MS) {
    holdFill.style.width = "0%";
    confidenceEl.textContent = `${Math.round(pred.confidence * 100)}% · ready for next sign`;
    return;
  }

  if (holdLabel !== label) {
    holdLabel = label;
    holdStartedAt = now;
  }

  const p = Math.min(1, (now - holdStartedAt) / HOLD_MS);
  holdFill.style.width = `${p * 100}%`;
  if (p < 1) {
    confidenceEl.textContent = `${Math.round(pred.confidence * 100)}% · locking ${Math.round(p * 100)}%`;
  } else {
    onDone();
  }
}

// ── Camera ───────────────────────────────────
async function startCamera() {
  if (stream) stopCamera(false);

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
    // Critical: playsInline + mute for autoplay policies
    video.muted = true;
    video.setAttribute("playsinline", "true");
    await video.play();

    // Wait for dimensions
    if (!video.videoWidth) {
      await new Promise((resolve, reject) => {
        const t = setTimeout(() => reject(new Error("Video metadata timeout")), 5000);
        video.onloadedmetadata = () => {
          clearTimeout(t);
          resolve();
        };
      }).catch(() => {});
    }

    // Match canvas to actual video resolution (required for correct landmarks)
    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;
    canvas.width = w;
    canvas.height = h;

    // Mirror preview only (does not affect MediaPipe pixel buffer)
    const mirror = facingMode === "user";
    video.style.transform = mirror ? "scaleX(-1)" : "none";
    canvas.style.transform = mirror ? "scaleX(-1)" : "none";

    running = true;
    lastVideoTime = -1;
    lastTimestamp = 0;
    displayVotes = [];
    framesWithoutHand = 0;
    resetHold();
    resetRecognitionMemory();
    liveBadge.hidden = false;

    btnStart.disabled = true;
    btnStop.disabled = false;
    btnSwitch.disabled = false;
    btnAdd.disabled = false;
    updateComposeUI();

    statusEl.textContent = (facingMode === "user" ? "Front" : "Rear") + " · looking for hands…";
    addLogEntry({
      type: "system",
      icon: "CAM",
      title: "Camera started",
      detail: `${w}×${h} · ${formatTime()}`
    });

    loop();
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Camera error: " + (err.message || "permission denied");
    alert("Could not access camera. Allow camera permission and try again.");
  }
}

function stopCamera(log = true) {
  running = false;
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = 0;
  }
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  video.style.transform = "none";
  canvas.style.transform = "none";
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  predictionEl.textContent = "—";
  confidenceEl.textContent = "";
  currentPrediction = null;
  displayVotes = [];
  handVisible = false;
  resetHold();
  resetRecognitionMemory();
  liveBadge.hidden = true;

  btnStart.disabled = false;
  btnStop.disabled = true;
  btnSwitch.disabled = true;
  btnAdd.disabled = true;
  updateComposeUI();
  statusEl.textContent = "Camera off";

  if (log) {
    addLogEntry({ type: "system", icon: "OFF", title: "Camera stopped", detail: formatTime() });
  }
}

async function switchCamera() {
  facingMode = facingMode === "user" ? "environment" : "user";
  if (running) await startCamera();
}

// ── Detection loop (never dies on error) ─────
function loop() {
  if (!running) return;
  rafId = requestAnimationFrame(tick);
}

async function tick() {
  if (!running || !handLandmarker) return;

  try {
    await processFrame();
  } catch (err) {
    console.warn("frame error", err);
  }

  if (running) loop();
}

async function processFrame() {
  if (video.readyState < 2) return;

  // Some browsers report same currentTime for multiple rAFs — still run at least ~30fps wall clock
  const now = performance.now();
  const videoTime = video.currentTime;
  const timeAdvanced = videoTime !== lastVideoTime;
  if (!timeAdvanced && now - lastTimestamp < 30) return;
  lastVideoTime = videoTime;

  let ts = now;
  if (ts <= lastTimestamp) ts = lastTimestamp + 1;
  lastTimestamp = ts;

  // Keep canvas synced if resolution changes
  if (video.videoWidth && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  const results = handLandmarker.detectForVideo(video, ts);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const hands = results.landmarks || [];
  if (hands.length > 0) {
    framesWithoutHand = 0;
    handVisible = true;
    lastHandSeenAt = performance.now();
    wordFinalizedAfterGap = false;

    for (const landmarks of hands) {
      try {
        drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
          color: "#4f8cff",
          lineWidth: 3
        });
        drawingUtils.drawLandmarks(landmarks, {
          color: "#3dd68c",
          lineWidth: 1,
          radius: 4
        });
      } catch (drawErr) {
        // Fallback: simple dots if DrawingUtils fails
        ctx.fillStyle = "#3dd68c";
        for (const p of landmarks) {
          ctx.beginPath();
          ctx.arc(p.x * canvas.width, p.y * canvas.height, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    const pred = await classifyHand(hands[0]);
    const smoothed = voteDisplay(pred);
    showLive(smoothed);
    if (smoothed) processHold(smoothed);

    if (!String(statusEl.textContent).startsWith("+") && !String(statusEl.textContent).startsWith("Word")) {
      statusEl.textContent =
        (facingMode === "user" ? "Front" : "Rear") + " · hand detected";
    }
  } else {
    handVisible = false;
    framesWithoutHand += 1;
    displayVotes.push(null);
    if (displayVotes.length > DISPLAY_HISTORY) displayVotes.shift();
    // Clear sticky letter after hand is gone briefly so next pose starts fresh
    if (framesWithoutHand > 12) resetRecognitionMemory();
    showLive(null);
    resetHold();

    // Auto CPU fallback if GPU never sees a hand
    if (!triedCpuFallback && framesWithoutHand > 90 && running) {
      triedCpuFallback = true;
      await recreateLandmarkerCpu();
    }

    if (
      spelling.length > 0 &&
      lastHandSeenAt > 0 &&
      performance.now() - lastHandSeenAt >= WORD_GAP_MS &&
      !wordFinalizedAfterGap
    ) {
      finalizeWord("hand left frame");
    }

    if (!String(statusEl.textContent).startsWith("+") && !String(statusEl.textContent).startsWith("Word")) {
      statusEl.textContent =
        (facingMode === "user" ? "Front" : "Rear") + " · no hand in view";
    }
  }
}

async function classifyHand(landmarks) {
  // Optional trained KNN
  if (classifier && sampleCount > 0 && typeof tf !== "undefined") {
    let tensor;
    try {
      tensor = landmarksToTensor(landmarks);
      const result = await classifier.predictClass(tensor, 3);
      tensor.dispose();
      const conf = result.confidences[result.label] || 0;
      if (conf >= 0.5) {
        return {
          label: String(result.label).toUpperCase(),
          confidence: conf,
          source: "trained",
          kind: isLetterLabel(result.label) ? "letter" : "word"
        };
      }
    } catch (e) {
      try {
        if (tensor) tensor.dispose();
      } catch (_) {
        /* ignore */
      }
      console.warn("KNN error", e);
    }
  }

  const builtIn = recognizeAsl(landmarks);
  if (!builtIn) return null;
  return {
    ...builtIn,
    source: "builtin",
    // preserve disambiguation metadata for UI
    alt: builtIn.alt,
    pending: builtIn.pending,
    margin: builtIn.margin
  };
}

// ── Training ─────────────────────────────────
async function addSample() {
  const label = labelInput.value.trim().toUpperCase();
  if (!label) {
    labelInput.focus();
    return;
  }
  if (!running || !handLandmarker) return;
  if (!classifier || typeof tf === "undefined") {
    alert("Custom training unavailable. Built-in ASL still works.");
    return;
  }

  let ts = performance.now();
  if (ts <= lastTimestamp) ts = lastTimestamp + 1;
  lastTimestamp = ts;

  const results = handLandmarker.detectForVideo(video, ts);
  if (!results.landmarks || results.landmarks.length === 0) {
    statusEl.textContent = "No hand – try again";
    return;
  }

  const tensor = landmarksToTensor(results.landmarks[0]);
  classifier.addExample(tensor, label);
  tensor.dispose();
  updateSampleCount();
  statusEl.textContent = `Added “${label}” sample`;
  addLogEntry({
    type: "system",
    icon: "+",
    title: `Trained “${label}”`,
    detail: `${sampleCount} total · ${formatTime()}`
  });
}

function clearSamples() {
  if (!classifier) return;
  classifier.clearAllClasses();
  updateSampleCount();
  addLogEntry({ type: "system", icon: "CLR", title: "Cleared custom samples", detail: formatTime() });
}

function speakSentence() {
  if (!window.speechSynthesis) return;
  const parts = words.map((w) => w.text);
  if (spelling) parts.push(spelling);
  const text = parts.join(" ").trim();
  if (!text) return;
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 0.95;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utter);
  addLogEntry({ type: "system", icon: "🔊", title: `Spoke “${text}”`, detail: formatTime() });
}

// ── Events ───────────────────────────────────
btnStart.addEventListener("click", startCamera);
btnStop.addEventListener("click", () => stopCamera(true));
btnSwitch.addEventListener("click", switchCamera);
btnSpeak.addEventListener("click", speakSentence);
btnSpace.addEventListener("click", () => finalizeWord("space button"));
btnBackspace.addEventListener("click", backspaceLetter);
btnClearText.addEventListener("click", clearText);
btnClearLog.addEventListener("click", clearLog);
btnAdd.addEventListener("click", addSample);
btnClear.addEventListener("click", clearSamples);

btnSpeedNovice.addEventListener("click", () => applyCaptureMode("novice"));
btnSpeedExpert.addEventListener("click", () => applyCaptureMode("expert"));

labelInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") addSample();
});

document.addEventListener("keydown", (e) => {
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
  if (e.key === "Backspace") {
    e.preventDefault();
    backspaceLetter();
  } else if (e.key === " " || e.key === "Enter") {
    e.preventDefault();
    finalizeWord(e.key === "Enter" ? "enter" : "space");
  }
});

// ── Boot ─────────────────────────────────────
(async function init() {
  try {
    let saved = "novice";
    try {
      saved = localStorage.getItem("asl-capture-mode") || "novice";
    } catch (_) {
      /* ignore */
    }
    applyCaptureMode(saved === "expert" ? "expert" : "novice", { silent: true });

    createClassifier();
    updateComposeUI();
    updateStats();
    const backend = await createHandLandmarker();
    statusEl.textContent = "Ready – press Start Camera";
    predictionEl.textContent = "—";
    const cfg = CAPTURE_MODES[captureMode];
    confidenceEl.textContent = `${cfg.label} mode · lock in ~${(cfg.holdMs / 1000).toFixed(2)}s`;
    addLogEntry({
      type: "system",
      icon: "✓",
      title: "Models ready",
      detail: `${backend} · ${cfg.label} ~${(cfg.holdMs / 1000).toFixed(2)}s · ${formatTime()}`
    });
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Failed to load models";
    alert(
      "Could not load MediaPipe models.\n\n" +
        (err && err.message ? err.message : err) +
        "\n\nCheck internet and refresh."
    );
  }
})();
