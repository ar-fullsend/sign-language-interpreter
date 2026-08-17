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
    lockMinConf: 0.65,
    label: "Novice",
    hint: "Hold a letter ~1.0s to lock it"
  },
  // Fast lock — target under half a second
  expert: {
    holdMs: 350,
    cooldownMs: 220,
    wordGapMs: 700,
    spaceHoldMs: 400,
    lockMinConf: 0.6,
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

// ── ASL letter recognition (reliable + light disambiguation) ──
// Priority rules for common signs, continuous features for close pairs,
// short hysteresis so letters switch quickly but don't thrash.

function dist2(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

// Light temporal memory (2 frames for most switches)
let stickyLabel = null;
let stickyKind = "letter";
let stickyFrames = 0;
let challengerLabel = null;
let challengerFrames = 0;

function resetRecognitionMemory() {
  stickyLabel = null;
  stickyKind = "letter";
  stickyFrames = 0;
  challengerLabel = null;
  challengerFrames = 0;
}

/**
 * Finger extended? Tip farther from wrist than PIP (rotation-tolerant).
 * Also requires tip reasonably past PIP along the finger chain.
 */
function isFingerUp(lm, tipIdx, pipIdx, mcpIdx) {
  const wrist = lm[0];
  const tip = lm[tipIdx];
  const pip = lm[pipIdx];
  const mcp = lm[mcpIdx];
  const tipW = dist2(tip, wrist);
  const pipW = dist2(pip, wrist);
  const tipM = dist2(tip, mcp);
  const pipM = dist2(pip, mcp);
  // Forgiving thresholds so common poses register as "up"
  return tipW > pipW * 1.05 && tipM > pipM * 1.05;
}

/**
 * Thumb "out" (A, L, Y, 5) vs tucked (B, E, S).
 * Compares tip distance from palm center vs IP joint.
 */
function isThumbOut(lm) {
  const tip = lm[4];
  const ip = lm[3];
  const indexMcp = lm[5];
  const pinkyMcp = lm[17];
  const palm = {
    x: (indexMcp.x + pinkyMcp.x) * 0.5,
    y: (indexMcp.y + pinkyMcp.y) * 0.5
  };
  // Thumb out if tip is meaningfully farther from palm than IP
  return dist2(tip, palm) > dist2(ip, palm) * 1.08;
}

/** Thumb near index MCP (classic A position). */
function thumbBesideFist(lm, scale) {
  return dist2(lm[4], lm[5]) / scale < 0.55;
}

/** Thumb tip over closed fingers (S-ish). */
function thumbOverFist(lm, scale) {
  return dist2(lm[4], lm[6]) / scale < 0.38;
}

function kindForLabel(label) {
  if (label === "5") return "space";
  if (label === "1" || label === "3") return "number";
  return "letter";
}

/**
 * Core recognizer: ordered rules for high-recall common letters,
 * then disambiguation within confusable groups.
 */
function recognizeAsl(lm) {
  const wrist = lm[0];
  const scale = dist2(wrist, lm[9]) || 1e-6;
  const n = (i, j) => dist2(lm[i], lm[j]) / scale;

  const thumb = isThumbOut(lm);
  const index = isFingerUp(lm, 8, 6, 5);
  const middle = isFingerUp(lm, 12, 10, 9);
  const ring = isFingerUp(lm, 16, 14, 13);
  const pinky = isFingerUp(lm, 20, 18, 17);

  const imSpread = n(8, 12);
  const thumbIndex = n(4, 8);
  const thumbMiddle = n(4, 12);
  const tipFromMcp =
    (n(8, 5) + n(12, 9) + n(16, 13) + n(20, 17)) / 4;
  const circleAvg = (n(8, 4) + n(12, 4) + n(16, 4) + n(20, 4)) / 4;

  const indexDx = Math.abs(lm[8].x - lm[5].x);
  const indexDy = Math.abs(lm[8].y - lm[5].y);
  const indexHoriz = indexDx > indexDy * 1.15;

  /** @type {{ label: string, confidence: number, kind?: string } | null} */
  let best = null;
  const set = (label, confidence, kind) => {
    const k = kind || kindForLabel(label);
    if (!best || confidence > best.confidence) {
      best = { label, confidence, kind: k };
    }
  };

  // ═══════════════════════════════════════════
  // 1) Distinctive high-priority shapes (common)
  // ═══════════════════════════════════════════

  // Y — shaka
  if (thumb && pinky && !index && !middle && !ring) {
    set("Y", 0.94);
  }

  // I — pinky only
  if (pinky && !index && !middle && !ring) {
    set("I", thumb ? 0.82 : 0.93);
  }

  // L — index + thumb, others down, wide angle
  if (index && thumb && !middle && !ring && !pinky && thumbIndex > 0.4) {
    set("L", 0.93);
  }

  // V / U / H / R / K — index+middle up
  if (index && middle && !ring && !pinky) {
    if (imSpread >= 0.3) {
      // Spread → V (or K if thumb clearly between)
      if (thumb && thumbMiddle < 0.4 && imSpread < 0.55) set("K", 0.8);
      else set("V", 0.93);
    } else if (imSpread < 0.22) {
      // Together → U / H / R
      if (imSpread < 0.14) set("R", 0.78);
      else if (thumb) set("H", 0.84);
      else set("U", 0.9);
    } else {
      // Mid spread
      if (thumb && thumbMiddle < 0.42) set("K", 0.78);
      else set("V", 0.82);
    }
  }

  // W / 3 — three fingers
  if (index && middle && ring && !pinky) {
    set(thumb ? "W" : "3", thumb ? 0.9 : 0.86);
  }

  // F — OK: thumb~index tip, other three up
  if (middle && ring && pinky && !index && thumbIndex < 0.4) {
    set("F", 0.9);
  }

  // B — four up, thumb in
  if (index && middle && ring && pinky && !thumb) {
    set("B", 0.92);
  }

  // 5 — open hand (space)
  if (index && middle && ring && pinky && thumb) {
    set("5", 0.93);
  }

  // D / G / 1 — index only among four fingers
  if (index && !middle && !ring && !pinky) {
    // Don't steal L (already handled when thumb + wide)
    if (!(thumb && thumbIndex > 0.4)) {
      if (indexHoriz && thumb) set("G", 0.84);
      else if (thumb) set("D", 0.88);
      else set("1", 0.9);
    }
  }

  // ═══════════════════════════════════════════
  // 2) Fist family — A is the default fist+thumb
  // ═══════════════════════════════════════════
  if (!index && !middle && !ring && !pinky) {
    // Prefer A whenever thumb is out beside a closed hand (most common)
    if (thumb) {
      if (thumbOverFist(lm, scale) && !thumbBesideFist(lm, scale)) {
        set("S", 0.8);
      } else if (n(4, 6) < 0.35 && n(4, 10) < 0.4 && tipFromMcp < 0.35) {
        // T-ish: thumb jammed into fist top
        set("T", 0.76);
      } else {
        // A: default fist + thumb out (high recall)
        const conf = thumbBesideFist(lm, scale) ? 0.92 : 0.86;
        set("A", conf);
      }
    } else {
      // Thumb tucked
      if (n(4, 14) < 0.4 || n(4, 10) < 0.38) {
        // under fingers → M/N approx
        set(n(4, 14) < n(4, 10) ? "M" : "N", 0.74);
      } else {
        set("E", 0.84);
      }
    }

    // O only if clearly a ring (not a fist) — strict, low priority
    if (
      circleAvg < 0.34 &&
      tipFromMcp > 0.3 &&
      tipFromMcp < 0.55 &&
      n(8, 12) < 0.35 &&
      best &&
      (best.label === "A" || best.label === "E" || best.label === "S")
    ) {
      // Only override fist if circle is strong
      if (circleAvg < 0.28 && tipFromMcp > 0.32) {
        set("O", 0.8);
      }
    }
  }

  // ═══════════════════════════════════════════
  // 3) C — open curve (not fist, not fully extended)
  // ═══════════════════════════════════════════
  if (!index && !middle && thumb && tipFromMcp > 0.28) {
    if (thumbIndex > 0.45 && thumbIndex < 1.05 && circleAvg > 0.35 && circleAvg < 0.75) {
      set("C", 0.8);
    }
  }

  // ═══════════════════════════════════════════
  // 4) Soft fallbacks so something always shows
  // ═══════════════════════════════════════════
  if (!best) {
    const up = [index, middle, ring, pinky].filter(Boolean).length;
    if (up === 0 && thumb) {
      best = { label: "A", confidence: 0.72, kind: "letter" };
    } else if (up >= 4) {
      best = { label: thumb ? "5" : "B", confidence: 0.7, kind: thumb ? "space" : "letter" };
    } else if (up === 1 && index) {
      best = { label: thumb ? "D" : "1", confidence: 0.7, kind: thumb ? "letter" : "number" };
    } else if (up === 2 && index && middle) {
      best = { label: "V", confidence: 0.7, kind: "letter" };
    } else {
      best = { label: "…", confidence: 0.35, kind: "unknown" };
    }
  }

  // Light hysteresis: only for confusable pairs, 2 frames
  const stable = applyLightHysteresis(best.label, best.confidence, best.kind);

  return {
    label: stable.label,
    confidence: clamp01(stable.confidence),
    kind: stable.kind,
    alt: stable.pending || null,
    pending: stable.pending || null
  };
}

function applyLightHysteresis(label, confidence, kind) {
  const CONFUSABLE = {
    A: ["E", "S", "T", "O"],
    E: ["A", "S", "O"],
    S: ["A", "E", "T"],
    U: ["V", "R", "H"],
    V: ["U", "K"],
    O: ["C", "A", "E"],
    C: ["O"],
    B: ["5"],
    "5": ["B"],
    D: ["1", "G", "L"],
    "1": ["D"]
  };

  if (label === "…" || kind === "unknown") {
    return { label, confidence, kind };
  }

  if (!stickyLabel) {
    stickyLabel = label;
    stickyKind = kind;
    stickyFrames = 1;
    challengerLabel = null;
    challengerFrames = 0;
    return { label, confidence, kind };
  }

  if (label === stickyLabel) {
    stickyFrames += 1;
    challengerLabel = null;
    challengerFrames = 0;
    return { label: stickyLabel, confidence, kind: stickyKind };
  }

  // Distinctive letters switch immediately (Y, L, I, V, 5, B, W…)
  const confusable = (CONFUSABLE[stickyLabel] || []).includes(label);
  const need = confusable ? 2 : 1;

  if (label === challengerLabel) challengerFrames += 1;
  else {
    challengerLabel = label;
    challengerFrames = 1;
  }

  if (challengerFrames >= need) {
    stickyLabel = label;
    stickyKind = kind;
    stickyFrames = 1;
    challengerLabel = null;
    challengerFrames = 0;
    return { label, confidence, kind };
  }

  return {
    label: stickyLabel,
    confidence: confidence * 0.95,
    kind: stickyKind,
    pending: label
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
