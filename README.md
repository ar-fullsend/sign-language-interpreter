# Sign Language Interpreter

A **privacy-first** web application that uses your device's **front or rear camera** and machine learning to interpret sign language (ASL alphabet + custom gestures) in real time.

Everything runs entirely in the browser — no video or data is ever sent to a server.

**Live demo:** Enable GitHub Pages on this repo (Settings → Pages → Deploy from main branch) or open `index.html` via any static file server.

## Features

- 📷 **Front / Rear camera** switching (`facingMode: user` / `environment`)
- ✋ Real-time hand landmark detection via **MediaPipe Hand Landmarker**
- 🧠 On-device machine learning with **TensorFlow.js KNN Classifier**
- 🎯 Train your own signs live (add samples for any label)
- 🗣️ Optional text-to-speech readout of predictions
- 📱 Mobile-friendly responsive UI
- 🔒 100% client-side processing

## Quick Start

```bash
# Clone
git clone https://github.com/ar-fullsend/sign-language-interpreter.git
cd sign-language-interpreter

# Serve locally (required for camera + modules)
npx serve .
# or
python -m http.server 8000
```

Open the URL shown (usually http://localhost:3000 or :8000).

## How to Use

1. Click **Start Camera** and grant permission.
2. Use **Switch Camera** to toggle front/rear (on devices that support it).
3. Hold a sign steady in view of the camera.
4. To train custom signs:
   - Type a label (e.g. `HELLO`, `A`, `THUMBS_UP`)
   - Click **Add Sample** several times while holding the pose
   - The KNN classifier will start predicting once enough samples exist
5. Click **Speak** to hear the current prediction via the Web Speech API.

## Architecture

```
Webcam frame
    ↓
MediaPipe Hand Landmarker  →  21 × 3D landmarks (x, y, z)
    ↓
Normalize + flatten to feature vector
    ↓
TensorFlow.js KNN Classifier
    ↓
Predicted label + confidence
    ↓
UI + optional TTS
```

## Tech Stack

| Component              | Library / API                          |
|------------------------|----------------------------------------|
| Hand tracking          | MediaPipe Tasks Vision (Hand Landmarker) |
| Classification         | TensorFlow.js + `@tensorflow-models/knn-classifier` |
| Camera                 | `navigator.mediaDevices.getUserMedia`  |
| Speech                 | Web Speech API (`speechSynthesis`)     |
| UI                     | Vanilla HTML / CSS / JS                |

## Project Structure

```
.
├── index.html          # Main application
├── style.css           # Styles
├── app.js              # Core logic (camera, MediaPipe, KNN, UI)
└── README.md
```

## Extending

- Replace the KNN with a custom TensorFlow.js LayersModel trained on ASL alphabet landmarks (many open-source models exist).
- Add sequence buffering + LSTM / Transformer for continuous word-level recognition.
- Export / import the KNN dataset via `localStorage` or downloadable JSON.
- Add dual-hand support and pose (full body) for richer context.

## Credits & Inspiration

Built as a clean, modern foundation inspired by several excellent open-source browser ASL projects that combine MediaPipe + TensorFlow.js.

## License

MIT
