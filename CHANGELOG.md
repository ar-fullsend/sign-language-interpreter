# Changelog

All notable changes to this project are documented in this file.

## [0.2.0] — 2026-08-16

### Added
- Two-column UI: main stage + **Activity** sidebar (letter/word log + stats)
- Letter **hold-to-lock** spelling buffer and sentence composition
- Word finalize via open hand, hand-left-frame gap, or Space/Enter
- Dictionary matching and common phrase merge (`I LOVE YOU`, `THANK YOU`, …)
- **Novice / Expert** capture-speed toggle (persisted in `localStorage`)
- Built-in geometry recognizer with continuous finger features
- Similar-sign disambiguation (fist, two-finger, circle, open, index families)
- Temporal hysteresis and margin gate to reduce letter flicker
- MediaPipe GPU path with automatic **CPU fallback**
- Speak full sentence; backspace letter; clear text / clear log
- Optional collapsible custom KNN training panel

### Changed
- App identity: **ASL Translator** (fingerspell → words)
- Live prediction always updates; lock bar only commits spelling
- Open hand acts as space only while spelling is in progress
- Stricter **O** scoring (no longer matches every relaxed hand)

### Fixed
- Detection loop no longer dies on draw/classify errors
- Canvas / video sizing for correct landmark overlay
- Weak predictions no longer auto-lock into spelling

## [0.1.0] — 2026-08-16

### Added
- Initial privacy-first web sign language interpreter
- MediaPipe Hand Landmarker + TensorFlow.js KNN
- Camera start/stop/switch, basic train UI
