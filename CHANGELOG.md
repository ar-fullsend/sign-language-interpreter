# Changelog

All notable changes to this project are documented in this file.

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
