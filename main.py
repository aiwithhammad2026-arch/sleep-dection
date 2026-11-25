"""
AI Sleep Detection System with Automatic Alarm
==============================================
A real-time computer vision system that monitors eye closure via webcam.
If eyes remain closed for more than 3 seconds (drowsiness detected),
it triggers an audible alarm until the user opens their eyes.

Dependencies:
    pip install opencv-python mediapipe numpy pygame
"""

import sys
import time
import math
import io
import wave
import os

# Check required third-party libraries gracefully
try:
    import cv2
    import numpy as np
    import mediapipe as mp
    import pygame
except ImportError as e:
    missing_module = getattr(e, "name", str(e))
    print(f"\n[ERROR] Missing required library: {missing_module}")
    print("\nPlease install the required libraries with:")
    print("    pip install opencv-python mediapipe numpy pygame\n")
    sys.exit(1)


# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
# Eye Aspect Ratio (EAR) threshold below which eye is considered closed
EAR_THRESHOLD = 0.22

# Time in seconds of continuous eye closure before triggering the alarm
DROWSINESS_TIME_LIMIT = 3.0

# Warning threshold (shows DROWSY status before alarm triggers)
DROWSY_WARNING_TIME = 1.0

# Landmark indices for Left and Right eyes in MediaPipe Face Mesh
# Format for 6 points: [outer_corner, top_outer, top_inner, inner_corner, bottom_inner, bottom_outer]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


# ==========================================
# AUDIO / ALARM INITIALIZATION
# ==========================================
def create_alarm_sound():
    """
    Creates an alarm sound. If 'alarm.wav' exists in the working directory,
    it loads that file. Otherwise, it synthesizes an alarm tone in-memory
    using NumPy and the built-in wave module so no external audio file is required.
    """
    # 1. Check for custom audio file
    if os.path.exists("alarm.wav"):
        try:
            return pygame.mixer.Sound("alarm.wav")
        except Exception as err:
            print(f"[WARNING] Could not load alarm.wav: {err}. Generating synthetic alarm.")

    # 2. Synthesize a loud pulsating alarm tone in-memory
    sample_rate = 44100
    duration = 0.6  # seconds per loop
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Dual frequency siren (900 Hz and 1300 Hz) with amplitude pulsing
    frequency = np.where(t < duration / 2, 900, 1300)
    waveform = np.sin(2 * np.pi * frequency * t)

    # Convert to 16-bit PCM integer audio
    pcm_data = (waveform * 32767 * 0.85).astype(np.int16)

    # Write into an in-memory WAV container
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)       # Mono
        wav_file.setsampwidth(2)       # 16-bit (2 bytes)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data.tobytes())

    wav_buffer.seek(0)
    return pygame.mixer.Sound(wav_buffer)


# ==========================================
# EYE ASPECT RATIO (EAR) CALCULATION
# ==========================================
def calculate_ear(landmarks, eye_indices, img_w, img_h):
    """
    Calculates the Eye Aspect Ratio (EAR) using 6 landmark coordinates.
    Formula: EAR = (|p2 - p6| + |p3 - p5|) / (2.0 * |p1 - p4|)
    """
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))

    p1, p2, p3, p4, p5, p6 = coords

    # Vertical eye distances
    d_vertical_1 = np.linalg.norm(p2 - p6)
    d_vertical_2 = np.linalg.norm(p3 - p5)

    # Horizontal eye distance
    d_horizontal = np.linalg.norm(p1 - p4)

    if d_horizontal == 0:
        return 0.0

    ear = (d_vertical_1 + d_vertical_2) / (2.0 * d_horizontal)
    return ear


def draw_eye_contours(frame, landmarks, eye_indices, img_w, img_h, color):
    """Draws lines connecting the eye landmarks for visual feedback."""
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * img_w), int(lm.y * img_h)))
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)


# ==========================================
# MAIN APPLICATION LOOP
# ==========================================
def main():
    print("=" * 60)
    print("        AI Sleep Detection System with Automatic Alarm        ")
    print("=" * 60)

    # Initialize Pygame mixer for audio
    sound_enabled = True
    alarm_sound = None
    try:
        pygame.mixer.init()
        alarm_sound = create_alarm_sound()
        print("[INFO] Audio system initialized successfully.")
    except Exception as e:
        sound_enabled = False
        print(f"[WARNING] Could not initialize audio mixer ({e}). System will run without sound.")

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access the webcam.")
        print("Please verify that your webcam is connected and not in use by another app.")
        if sound_enabled:
            pygame.mixer.quit()
        return

    # Tracking variables
    closed_start_time = None
    alarm_playing = False
    status = "AWAKE"
    status_color = (0, 255, 0)  # Green

    print("[INFO] Starting video stream... Press 'q' or 'ESC' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from camera. Exiting...")
            break

        # Flip horizontally for natural mirror display
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        avg_ear = 0.0
        elapsed_closed_time = 0.0

        if results.multi_face_landmarks:
            mesh_landmarks = results.multi_face_landmarks[0].landmark

            # Compute EAR for both eyes
            right_ear = calculate_ear(mesh_landmarks, RIGHT_EYE_INDICES, w, h)
            left_ear = calculate_ear(mesh_landmarks, LEFT_EYE_INDICES, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            # Eye closure logic
            if avg_ear < EAR_THRESHOLD:
                if closed_start_time is None:
                    closed_start_time = time.time()

                elapsed_closed_time = time.time() - closed_start_time

                # Determine status based on duration
                if elapsed_closed_time >= DROWSINESS_TIME_LIMIT:
                    status = "ALARM ACTIVE"
                    status_color = (0, 0, 255)  # Red
                    # Trigger alarm
                    if sound_enabled and alarm_sound and not alarm_playing:
                        alarm_sound.play(loops=-1)
                        alarm_playing = True
                elif elapsed_closed_time >= DROWSY_WARNING_TIME:
                    status = "DROWSY"
                    status_color = (0, 165, 255)  # Orange / Yellow
                else:
                    # Brief eye blink (< 1.0s) is considered normal
                    status = "AWAKE"
                    status_color = (0, 255, 0)  # Green
            else:
                # Eyes are open: reset timer and immediately stop alarm
                closed_start_time = None
                status = "AWAKE"
                status_color = (0, 255, 0)  # Green

                if alarm_playing and alarm_sound:
                    alarm_sound.stop()
                    alarm_playing = False

            # Draw eye landmarks on face
            contour_color = (0, 0, 255) if avg_ear < EAR_THRESHOLD else (0, 255, 0)
            draw_eye_contours(frame, mesh_landmarks, RIGHT_EYE_INDICES, w, h, contour_color)
            draw_eye_contours(frame, mesh_landmarks, LEFT_EYE_INDICES, w, h, contour_color)

        else:
            # No face detected
            status = "NO FACE DETECTED"
            status_color = (180, 180, 180)
            closed_start_time = None
            if alarm_playing and alarm_sound:
                alarm_sound.stop()
                alarm_playing = False

        # ==========================================
        # ON-SCREEN DISPLAY (DASHBOARD)
        # ==========================================
        # Top status panel overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (380, 140), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (15, 15), (380, 140), status_color, 2)

        # Status text
        cv2.putText(frame, f"STATUS: {status}", (30, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, status_color, 2)

        # Eye Aspect Ratio (EAR) metric
        cv2.putText(frame, f"EAR: {avg_ear:.2f} (Threshold: {EAR_THRESHOLD:.2f})", (30, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

        # Closed time counter & progress bar
        cv2.putText(frame, f"Closed Time: {elapsed_closed_time:.1f}s / {DROWSINESS_TIME_LIMIT:.1f}s", (30, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

        # Progress bar showing time until alarm
        progress = min(1.0, elapsed_closed_time / DROWSINESS_TIME_LIMIT)
        bar_x, bar_y, bar_w, bar_h = 30, 118, 320, 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        if progress > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), status_color, -1)

        # Flashing border alert when alarm is active
        if status == "ALARM ACTIVE" and int(time.time() * 4) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
            cv2.putText(frame, "! WAKE UP !", (w // 2 - 140, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

        # Bottom instruction bar
        cv2.putText(frame, "Press 'q' or 'ESC' to exit", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show frame
        cv2.imshow("AI Sleep Detection System", frame)

        # Check key press (q or ESC to exit)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Clean up resources
    print("[INFO] Shutting down...")
    if alarm_playing and alarm_sound:
        alarm_sound.stop()
    if sound_enabled:
        pygame.mixer.quit()
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed cleanly.")


if __name__ == "__main__":
    main()
