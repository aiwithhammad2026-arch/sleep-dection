# 😴 AI Sleep & Drowsiness Detection System with Automatic Alarm

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-007FFF?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Pygame](https://img.shields.io/badge/Pygame-Audio%20Alarm-yellow?logo=python)](https://www.pygame.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, real-time Computer Vision system that monitors eye closure through a webcam. Using **MediaPipe Face Mesh** and the **Eye Aspect Ratio (EAR)** algorithm, the system accurately distinguishes between natural blinking and prolonged eye closure (drowsiness). When eyes remain closed for more than **3 seconds**, an audible emergency alarm triggers automatically and persists until the eyes open again.

---

## 📋 Table of Contents
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Project Architecture](#-project-architecture)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Running the Project](#-running-the-project)
- [Controls & Interface](#-controls--interface)
- [Customization](#-customization)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## ✨ Key Features

- **Real-Time Facial Landmark Tracking**: Tracks 468+ 3D facial landmarks at 30+ FPS using Google MediaPipe Face Mesh.
- **Accurate Eye Aspect Ratio (EAR)**: 6-point Euclidean distance formula to compute precise eyelid aperture.
- **Smart Blink Discrimination**: Normal eye blinks (< 1.0 second) do not trigger false alerts.
- **Progressive Alert System**:
  - `AWAKE` (Green): Normal open eyes or brief blinks.
  - `DROWSY` (Orange/Yellow): Eyes closed between 1.0s and 3.0s.
  - `ALARM ACTIVE` (Red): Eyes closed for $\ge$ 3.0s with flashing visual and loud auditory alarm.
- **Zero-Dependency Audio Synthesis**: Synthesizes a loud, pulsating two-tone alarm in-memory via Pygame & NumPy without needing external `.mp3` or `.wav` files.
- **Instant Alarm Cutoff**: Automatically silences the alarm the instant eyes reopen.
- **Live On-Screen HUD**: Displays current EAR, closure duration, countdown progress bar, and real-time eye contour overlays.

---

## 🧠 How It Works

### Eye Aspect Ratio (EAR)
The system calculates the ratio between the vertical eyelid distances and horizontal eye width:

$$\text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \times ||p_1 - p_4||}$$

Where $p_1, \dots, p_6$ are standard 2D facial landmark coordinates on the eye perimeter:
- **$p_1, p_4$**: Outer and inner eye corners.
- **$p_2, p_3, p_5, p_6$**: Top and bottom eyelid landmark pairs.

```
          p2      p3
           •------•
   p1 •              • p4
           •------•
          p6      p5
```

- When eyes are **open**, EAR is typically $\ge 0.25 - 0.35$.
- When eyes are **closed**, EAR drops below $0.22$.

---

## 🏗️ Project Architecture

```
sleep-detection-system/
├── main.py              # Complete application code (Vision, Detection & Alarm)
├── requirements.txt     # Python dependencies list
├── .gitignore           # Git ignore configuration
└── README.md            # Comprehensive project documentation
```

---

## 💻 System Requirements

- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Python 3.10, 3.11, or 3.12 (Recommended for full MediaPipe compatibility)
- **Hardware**: Working USB or integrated webcam, audio speakers / headphones

---

## 🚀 Installation

### 1. Clone or Download the Repository
```bash
git clone https://github.com/aiwithhammad2026-arch/aiwithhammad2026-arch.git
cd aiwithhammad2026-arch
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install directly:
```bash
pip install opencv-python mediapipe numpy pygame
```

---

## 🎮 Running the Project

Run the application with:

```bash
python main.py
```

### Quick Test:
1. Ensure your face is visible within the webcam frame.
2. Observe the green status indicator (`STATUS: AWAKE`) and real-time EAR reading.
3. Close your eyes:
   - **0.0s – 1.0s**: Treated as a natural blink (no alarm).
   - **1.0s – 3.0s**: Yellow/Orange warning (`STATUS: DROWSY`).
   - **3.0s+**: Loud alarm sounds and flashing red screen alert appears (`STATUS: ALARM ACTIVE`).
4. Reopen your eyes: The alarm immediately stops and returns to `AWAKE`.

---

## 🎛️ Controls & Interface

| Key | Action |
| :--- | :--- |
| `q` | Quit the application cleanly |
| `ESC` | Quit the application cleanly |

### On-Screen Display Elements:
- **Status Box**: Current state (`AWAKE`, `DROWSY`, `ALARM ACTIVE`, or `NO FACE DETECTED`).
- **EAR Metric**: Live Eye Aspect Ratio value and active detection threshold.
- **Closed Time & Progress Bar**: Visual countdown of closure duration towards the 3.0-second limit.
- **Eye Contours**: Green outlines when open, red outlines when closed.
- **Flashing Warning**: Prominent screen flash during active alarm.

---

## ⚙️ Customization

You can easily adjust sensitivity settings at the top of [`main.py`](main.py):

```python
# Eye Aspect Ratio threshold (increase if eyes trigger closed too easily)
EAR_THRESHOLD = 0.22

# Time in seconds of continuous eye closure before sounding alarm
DROWSINESS_TIME_LIMIT = 3.0

# Warning threshold duration before alarm sounds
DROWSY_WARNING_TIME = 1.0
```

### Custom Alarm Sound
To use your own sound file instead of the built-in synthesizer:
1. Place any audio file named **`alarm.wav`** in the project folder.
2. The application will automatically detect and play your file.

---

## ❓ Troubleshooting

| Issue | Solution |
| :--- | :--- |
| `[ERROR] Missing required library: cv2` | Run `python -m pip install -r requirements.txt`. |
| `No matching distribution found for mediapipe` | MediaPipe may not have prebuilt wheels for Python 3.13 yet. Use Python 3.10, 3.11, or 3.12 (`py -3.11 -m pip install -r requirements.txt`). |
| `[ERROR] Cannot access the webcam` | Ensure no other app (Zoom, Teams, Camera) is using the webcam, or change `cv2.VideoCapture(0)` to `1` in `main.py`. |
| No sound playing | Check that your system audio/volume is turned up and that audio output is connected. |

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - feel free to use, modify, and distribute it.
