# ASL Translator

**Privacy-first American Sign Language fingerspelling → text**, running entirely in your browser.

Point a camera at your hand. MediaPipe tracks 21 landmarks. An on-device scorer turns poses into letters, locks them into spelling, builds words against a dictionary, and keeps an activity log — **no video leaves the device**.

> Camera + MediaPipe + geometry ML + optional TensorFlow.js KNN · zero backend required.

---

## Features

### Live recognition
- Real-time **hand landmark** tracking (MediaPipe Hand Landmarker)
- **Built-in ASL letter detector** — works with **zero training**
- **High-recall priority rules** for common letters first (**A, L, V, Y, B, 5, I, W…**)
- Light disambiguation for look-alikes:
  - Fist: **A · E · S · T · M · N** (A is default fist + thumb out)
  - Two-finger: **U · V · H · R · K**
  - Circle: **O · C** (O only when a clear ring — won’t steal A)
  - Open hand: **B · 5**
  - Index: **D · G · 1**
- Short **hysteresis** (1–2 frames) so letters switch quickly without thrashing
- GPU MediaPipe with **automatic CPU fallback** (Surface / ARM friendly)

### Whole-word gestures (built-in)
| Sign | How to produce |
|------|----------------|
| **THANK YOU** | Flat hand at chin → move forward & down |
| **HELLO / HI** | Open hand wave |
| **I LOVE YOU** | ILY handshape (thumb + index + pinky) |
| **YES** | Fist nodding |
| **NO** | Two-finger side shake |
| **BYE** | Open hand wave (more flicks) |
| **GOOD** | Thumbs up (hold) |
| **OK** | OK circle + three up (hold) |

Motion signs lock as soon as the movement completes; static shapes use a short hold.

### Spelling → words
- Hold a steady **letter** to lock it into the spelling buffer
- **Open hand (5)** = space / end word (when already spelling)
- Drop hand from frame to finalize a spelled word
- Dictionary match + phrase merge for fingerspelled text
- Sentence bar with known vs unknown word chips
- **Speak sentence** via Web Speech API

### Capture speed
| Mode | Letter lock | Best for |
|------|-------------|----------|
| **Novice** | ~1.0s | Learning clear poses |
| **Expert** | ~0.35s | Fast fingerspelling |

Toggle lives in the top bar; preference is saved in `localStorage`.

### Activity sidebar
- Live log of locked letters, completed words, gestures, and system events
- Stats: letters · words · dictionary hits

### Optional custom training
- TensorFlow.js **KNN classifier** for custom labels (e.g. `HELLO`, `THANKS`)
- Multi-letter trained labels commit as whole-word gestures
- Collapsed “train custom signs” panel so the main UI stays simple

### Privacy & UX
- **100% client-side** — no uploads, no accounts
- Front / rear camera switch
- Mirrored selfie preview (live `<video>` under a **transparent** landmark canvas)
- Skeleton overlay so you always know tracking is alive
- **Mobile-first detection UI**
  - Large **Seeing now** panel **below** the camera (not over the photo)
  - Fixed bottom dock with letter/word, spelling, words, Space / ⌫ / Speak
  - Safari/iOS safe-area + `webkit-playsinline` support

---

## Live demo (GitHub Pages)

**https://ar-fullsend.github.io/sign-language-interpreter/**

Deployed automatically on every push to `main` via **GitHub Actions** (`.github/workflows/pages.yml`).  
HTTPS is required for camera access.

## Quick start (local)

```bash
git clone https://github.com/ar-fullsend/sign-language-interpreter.git
cd sign-language-interpreter

# Any static server works (camera needs http://localhost or HTTPS)
python3 -m http.server 8000
# or: npx serve .
```

Open **http://localhost:8000/** → **Start Camera** → allow permissions.

---

## How to use

1. Click **Start Camera** and allow camera access.
2. Confirm the **blue/green skeleton** on your hand (`hand detected` in the status chip).
3. Pick **Novice** or **Expert** capture speed.
4. Hold a clear ASL letter until the **hold meter** fills — letter locks into **Spelling**.
5. End a word with:
   - open hand (**5**), or
   - drop hand from frame, or
   - **Space** button / keyboard Space / Enter
6. Known words light up as dictionary matches; the **Activity** sidebar records everything.
7. **Speak sentence** reads the full line aloud.
8. Optional: expand **Train custom signs**, type a label, add several samples while holding the pose.

### Good starter letters

| Letter | Pose (rough) |
|--------|----------------|
| **L** | Index up + thumb out |
| **V** | Peace / two fingers spread |
| **Y** | Shaka (thumb + pinky) |
| **B** | Flat hand, thumb tucked |
| **A** | Fist, thumb beside |
| **5** | Open hand (space when spelling) |
| **I** | Pinky only |

Tips: good lighting · full hand in frame · steady pose · thumb placement matters a lot for **A / S / E / T**.

---

## Architecture

```
Webcam frame
    ↓
MediaPipe Hand Landmarker  →  21 landmarks (x, y, z)
    ↓
Finger up / thumb out (rotation-tolerant geometry)
    ↓
Priority rules (common letters: A, L, V, Y, B, 5, …)
    ↓
Light disambiguation for confusable pairs + short hysteresis
    ↓
Live UI  ·  hold-to-lock spelling  ·  word finalize  ·  activity log
    ↓
Optional: TensorFlow.js KNN (custom trained labels override when present)
```

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Hand tracking | [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (Hand Landmarker) |
| Built-in letters | Geometry rules + light similar-sign disambiguation (vanilla JS) |
| Custom signs | TensorFlow.js + `@tensorflow-models/knn-classifier` |
| Camera | `navigator.mediaDevices.getUserMedia` |
| Speech | Web Speech API (`speechSynthesis`) |
| UI | Vanilla HTML / CSS / JS (no build step) |

---

## Project structure

```
.
├── index.html                      # Shell: stage + activity sidebar + controls
├── style.css                       # Layout, dark UI, responsive sidebar
├── app.js                          # Camera, MediaPipe, recognizer, spelling, log
├── models/
│   └── hand_landmarker.task        # MediaPipe hand model (~7.5MB)
├── wasm/                           # MediaPipe WASM runtime (self-hosted for Pages)
│   ├── vision_wasm_internal.js
│   ├── vision_wasm_internal.wasm
│   ├── vision_wasm_nosimd_internal.js
│   └── vision_wasm_nosimd_internal.wasm
├── .nojekyll                       # Required so GitHub Pages serves wasm/ as static files
├── CHANGELOG.md
└── README.md
```

No bundler, no npm install required to run — open via a static file server.

> **GitHub Pages note:** model + WASM are **same-origin** under `models/` and `wasm/` so tracking does not depend on Google Storage or jsDelivr for the core runtime (CDN remains as fallback).

---

## Configuration (in `app.js`)

Capture modes:

| Mode | `holdMs` | `cooldownMs` | `wordGapMs` | `spaceHoldMs` |
|------|----------|--------------|-------------|---------------|
| Novice | 1000 | 550 | 1300 | 850 |
| Expert | 350 | 220 | 700 | 400 |

Dictionary and phrase lists live near the top of `app.js` (`DICTIONARY`, `PHRASES`) — extend them for your classroom or product.

---

## Browser support

- Chromium-based browsers (Edge, Chrome) recommended for camera + WebGL/WASM
- Requires **secure context**: `localhost` or **HTTPS** (camera will not work on plain LAN IPs in most browsers)
- Works on desktop and mobile; Surface / ARM devices may use the **CPU** MediaPipe path automatically

---

## Deploying to GitHub Pages

1. Push `main` with `index.html` at the repo root (already the case).
2. Repo **Settings → Pages → Build and deployment**
   - Source: **Deploy from a branch**
   - Branch: **main** / folder **/** (root)
3. After ~1 minute, open  
   `https://ar-fullsend.github.io/sign-language-interpreter/`
4. Optional: repo **About → Website** → paste that URL so it shows on the repo home page.

Every `git push` to `main` republishes the site.

## Extending

- Swap geometry rules for a trained TF.js LayersModel on landmark sequences
- Add motion letters **J** / **Z** with short trajectory buffers
- Dual-hand + full-body pose for conversational ASL (not only fingerspelling)
- Export / import KNN samples as JSON
- Ship as a PWA (service worker + install prompt)

---

## Privacy

- Video is processed **only in the browser**
- No analytics, no auth, no server-side inference in this repo
- Custom samples and capture-mode preference stay in local browser storage if you use those features

---

## Credits

Built as a privacy-first ASL fingerspelling foundation on **MediaPipe** + optional **TensorFlow.js**, with emphasis on similar-sign separation and an intuitive spell → word → sentence UX.

## License

MIT
