# Changelog

All notable changes to this project are documented in this file.

## [0.2.7] — 2026-08-17

### Fixed (Chrome mobile stacking)
- Detection no longer sits under `<video>` (Chrome compositing ignored z-index overlays)
- **Seeing now** panel is in normal document flow directly under the camera
- Mobile dock is **`position: fixed`** bottom (Chrome-safe; no sticky + nested overflow)
- Single scroll container on mobile; video isolated in its own layer
- Compact fixed dock with letter/word + spelling + actions always visible

## [0.2.6] — 2026-08-17

### Changed (mobile UX)
- Detection panel + mobile dock for live letter/word visibility
- Gesture guide collapsed by default; guides moved below translation
- Portrait camera framing on phones

## [0.2.5] — 2026-08-16

### Fixed
- **HandLandmarker create failed** on GitHub Pages / Surface:
  - Self-host full MediaPipe **WASM** under `wasm/` (same-origin)
  - Try `createFromOptions`, `createFromModelPath`, and `createFromModelBuffer`
  - Fresh `Uint8Array` copies (buffers can be detached after a failed attempt)
  - Surface the **real underlying error** lines in the alert
  - Add `.nojekyll` so Pages always serves binary assets as static files

## [0.2.4] — 2026-08-16

### Fixed
- Hand tracking model load failures on Pages / restricted networks:
  - Fetch model as **bytes** (`modelAssetBuffer`) with multi-host fallbacks
  - Multiple WASM CDNs (jsDelivr + unpkg)
  - Removed fragile full-file preflight + GPU swap that could kill a working CPU model
  - Clearer on-screen load progress and console diagnostics

## [0.2.3] — 2026-08-16

### Added
- **Whole-word ASL gestures** with pose history + motion detection:
  - **THANK YOU** (chin → out/down)
  - **HELLO / HI** (wave)
  - **I LOVE YOU** (ILY handshape)
  - **YES** (fist nod), **NO** (two-finger shake), **BYE** (wave)
  - **GOOD** (thumbs up), **OK** (OK handshape)
- Word gestures commit into the sentence bar (and speak once for feedback)
- On-page gesture guide listing how to produce each sign
- Letter vs word fusion so ILY/THANK YOU win over confusable letters

## [0.2.2] — 2026-08-16

### Fixed (GitHub Pages ML)
- Import MediaPipe via explicit `vision_bundle.mjs` (bare package URL failed as ESM on Pages)
- Host `models/hand_landmarker.task` **same-origin** (no dependency on Google Storage for the model)
- Resolve model URL with `import.meta.url` so `/sign-language-interpreter/` paths work
- WASM path uses trailing slash; **CPU-first** load for broader device support
- Detection loop no longer dies if the model isn’t ready yet
- Start Camera waits/retries model load; clearer load errors in the activity log
- `<base href="./">` for correct relative assets on project Pages

## [0.2.1] — 2026-08-16

### Changed
- Rebuilt letter recognizer for **high recall on common signs** (A, L, V, Y, B, 5, I, W…)
- **A** is the default result for fist + thumb out (no longer lost to over-strict scoring)
- **O** only fires on a clear fingertip ring so it doesn’t steal A/E/S
- Short hysteresis (1–2 frames) instead of sticky multi-frame gates
- Slightly lower lock confidence thresholds (Novice 0.65 / Expert 0.60)

### Docs
- README: live demo / GitHub Pages section and deploy steps
- Architecture notes updated to match the rule-based recognizer

## [0.2.0] — 2026-08-16

### Added
- Two-column UI: main stage + **Activity** sidebar (letter/word log + stats)
- Letter **hold-to-lock** spelling buffer and sentence composition
- Word finalize via open hand, hand-left-frame gap, or Space/Enter
- Dictionary matching and common phrase merge (`I LOVE YOU`, `THANK YOU`, …)
- **Novice / Expert** capture-speed toggle (persisted in `localStorage`)
- Built-in geometry recognizer with similar-sign handling
- MediaPipe GPU path with automatic **CPU fallback**
- Speak full sentence; backspace letter; clear text / clear log
- Optional collapsible custom KNN training panel

### Changed
- App identity: **ASL Translator** (fingerspell → words)
- Live prediction always updates; lock bar only commits spelling
- Open hand acts as space only while spelling is in progress

### Fixed
- Detection loop no longer dies on draw/classify errors
- Canvas / video sizing for correct landmark overlay
- Weak predictions no longer auto-lock into spelling

## [0.1.0] — 2026-08-16

### Added
- Initial privacy-first web sign language interpreter
- MediaPipe Hand Landmarker + TensorFlow.js KNN
- Camera start/stop/switch, basic train UI
