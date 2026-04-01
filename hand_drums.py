import numpy as np
import sounddevice as sd
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os
import time

MODEL_PATH = "hand_landmarker.task"

# ─────────────────────────────────────────────
# Robust Preprocessing Pipeline
# ─────────────────────────────────────────────
class FramePreprocessor:
    """Normalize lighting using CLAHE + auto brightness to help MediaPipe in bad conditions."""
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def process(self, frame):
        # Convert to LAB color space for perceptual brightness adjustment
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to luminance channel only, preserving color
        l = self.clahe.apply(l)
        
        # Auto-brightness: shift mean luminance toward 128
        mean_l = np.mean(l)
        target = 128
        if mean_l > 10:
            alpha = target / mean_l
            alpha = max(0.8, min(1.5, alpha))  # Clamp to sane range
            l = np.clip(l * alpha, 0, 255).astype(np.uint8)
        
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# Temporal Landmark Smoother
# ─────────────────────────────────────────────
class LandmarkSmoother:
    """Exponential moving average on landmark positions to kill jitter."""
    def __init__(self, alpha=0.6):
        self.alpha = alpha  # Higher = more responsive, lower = more smooth
        self.prev = {}  # key -> (x, y)
    
    def smooth(self, hand_idx, landmark_idx, x, y):
        key = (hand_idx, landmark_idx)
        if key in self.prev:
            px, py = self.prev[key]
            x = self.alpha * x + (1 - self.alpha) * px
            y = self.alpha * y + (1 - self.alpha) * py
        self.prev[key] = (x, y)
        return x, y

    def clear(self, hand_idx=None):
        if hand_idx is None:
            self.prev.clear()
        else:
            to_remove = [k for k in self.prev if k[0] == hand_idx]
            for k in to_remove:
                del self.prev[k]


# ─────────────────────────────────────────────
# Presence Tracker (anti-flicker)
# ─────────────────────────────────────────────
class PresenceTracker:
    """Requires a hand to be detected for N consecutive frames before 'confirming' it,
       and keeps it alive for N frames after it disappears to prevent flickering."""
    def __init__(self, confirm_frames=3, hold_frames=5):
        self.confirm_frames = confirm_frames
        self.hold_frames = hold_frames
        self.detect_streak = {}   # hand_idx -> consecutive detection count
        self.absence_count = {}   # hand_idx -> frames since last seen
        self.confirmed = {}       # hand_idx -> bool

    def update(self, detected_hand_indices):
        all_indices = set(list(self.detect_streak.keys()) + list(detected_hand_indices))
        
        for idx in all_indices:
            if idx in detected_hand_indices:
                self.detect_streak[idx] = self.detect_streak.get(idx, 0) + 1
                self.absence_count[idx] = 0
                if self.detect_streak[idx] >= self.confirm_frames:
                    self.confirmed[idx] = True
            else:
                self.absence_count[idx] = self.absence_count.get(idx, 0) + 1
                self.detect_streak[idx] = 0
                if self.absence_count[idx] > self.hold_frames:
                    self.confirmed[idx] = False

    def is_confirmed(self, hand_idx):
        return self.confirmed.get(hand_idx, False)


# ─────────────────────────────────────────────
# Drum Synthesis (unchanged)
# ─────────────────────────────────────────────
def generate_kick():
    sr = 44100
    t = np.linspace(0, 0.4, int(sr * 0.4))
    freq = 60 + 150 * np.exp(-20 * t)
    phase = np.cumsum(2 * np.pi * freq / sr)
    wave = np.sin(phase)
    env = np.exp(-12 * t)
    return (wave * env).astype(np.float32)

def generate_808():
    sr = 44100
    t = np.linspace(0, 1.5, int(sr * 1.5))
    freq = 45 + 50 * np.exp(-3 * t)
    phase = np.cumsum(2 * np.pi * freq / sr)
    wave = np.sin(phase)
    wave = np.tanh(wave * 4.0) / 2.0
    env = np.exp(-1.5 * t)
    return (wave * env).astype(np.float32)

def generate_snare():
    sr = 44100
    t = np.linspace(0, 0.35, int(sr * 0.35))
    freq = 250 * np.exp(-25 * t) + 150
    phase = np.cumsum(2 * np.pi * freq / sr)
    tone = np.sin(phase) * np.exp(-20 * t)
    noise = np.random.uniform(-1, 1, len(t))
    noise_env = np.exp(-15 * t)
    wave = tone * 0.5 + noise * noise_env * 0.5
    return wave.astype(np.float32)

def generate_hihat():
    sr = 44100
    t = np.linspace(0, 0.1, int(sr * 0.1))
    noise = np.random.uniform(-1, 1, len(t))
    noise = np.diff(noise, prepend=0)
    env = np.exp(-40 * t)
    return (noise * env * 0.4).astype(np.float32)

DRUMS = {
    0: {"name": "HI-HAT", "sound": generate_hihat(), "color": (50, 255, 255)},
    1: {"name": "SNARE",  "sound": generate_snare(), "color": (50, 50, 255)},
    2: {"name": "KICK",   "sound": generate_kick(),  "color": (50, 255, 50)},
    3: {"name": "808",    "sound": generate_808(),    "color": (255, 50, 255)}
}

class DrumMachine:
    def __init__(self):
        self.active_sounds = []
        self.stream = sd.OutputStream(channels=1, samplerate=44100, callback=self.callback)
        self.stream.start()

    def play(self, pad_idx):
        self.active_sounds.append({"sound": DRUMS[pad_idx]["sound"], "pos": 0})

    def callback(self, outdata, frames, time_info, status):
        outdata.fill(0)
        finished = []
        for s in self.active_sounds:
            snd = s["sound"]
            pos = s["pos"]
            chunk = min(frames, len(snd) - pos)
            outdata[:chunk, 0] += snd[pos:pos+chunk]
            s["pos"] += chunk
            if s["pos"] >= len(snd):
                finished.append(s)
        for s in finished:
            self.active_sounds.remove(s)
        outdata[:, 0] = np.clip(outdata[:, 0], -1.0, 1.0)

    def close(self):
        self.stream.stop()
        self.stream.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model not found. Run hand_magic.py first.")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)

    preprocessor = FramePreprocessor()
    smoother = LandmarkSmoother(alpha=0.55)
    presence = PresenceTracker(confirm_frames=3, hold_frames=4)
    machine = DrumMachine()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("====================================")
    print("  AR Hand Drum Kit (Enhanced)       ")
    print("====================================")
    print("Controls:")
    print(" - Strike DOWNWARDS into the pads!")
    print(" - Press 'Q' to Quit")
    print("====================================")

    prev_y = {}
    num_pads = 4
    pad_hit_flash = [0.0] * num_pads
    pad_cooldown = [0.0] * num_pads  # Per-pad cooldown timer
    COOLDOWN_FRAMES = 8  # Minimum frames between triggers on same pad

    show_debug = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # 1. PREPROCESS: Normalize lighting before MediaPipe sees it
        processed = preprocessor.process(frame)
        
        # Dark background for display
        display = cv2.convertScaleAbs(frame, alpha=0.4, beta=10)
        overlay = np.zeros_like(display, dtype=np.uint8)
        
        pad_width = w // num_pads
        pad_y_start = h // 2 
        
        # 2. DETECT on preprocessed frame
        img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)
        
        # 3. UPDATE PRESENCE TRACKER
        detected_indices = set()
        if results.hand_landmarks:
            for i in range(len(results.hand_landmarks)):
                detected_indices.add(i)
        presence.update(detected_indices)
        
        # Tick cooldowns
        for i in range(num_pads):
            if pad_cooldown[i] > 0:
                pad_cooldown[i] -= 1

        # 4. PROCESS CONFIRMED HANDS ONLY
        if results.hand_landmarks:
            for hand_idx, landmarks in enumerate(results.hand_landmarks):
                if not presence.is_confirmed(hand_idx):
                    continue  # Skip flickering/unconfirmed hands
                
                # 5. SMOOTH landmark positions
                raw_ix = landmarks[8].x * w
                raw_iy = landmarks[8].y * h
                ix, iy = smoother.smooth(hand_idx, 8, raw_ix, raw_iy)
                ix, iy = int(ix), int(iy)
                
                # Strike detection
                key = f"hand_{hand_idx}"
                if key in prev_y:
                    old_iy = prev_y[key]
                    
                    # Crossed the strike line going downward
                    if iy > pad_y_start and old_iy <= pad_y_start:
                        pad_hit = min(num_pads - 1, max(0, ix // pad_width))
                        
                        # 6. COOLDOWN CHECK: prevent double triggers
                        if pad_cooldown[pad_hit] <= 0:
                            machine.play(pad_hit)
                            pad_hit_flash[pad_hit] = 1.0
                            pad_cooldown[pad_hit] = COOLDOWN_FRAMES
                            cv2.circle(overlay, (ix, pad_y_start), 60, (255, 255, 255), -1)
                
                prev_y[key] = iy

                # Draw finger
                cv2.circle(overlay, (ix, iy), 12, (255, 255, 255), -1)
                cv2.circle(overlay, (ix, iy), 25, (0, 255, 255), 3)
        else:
            smoother.clear()

        # Draw Pads
        for i in range(num_pads):
            px1 = i * pad_width
            px2 = (i + 1) * pad_width
            color = DRUMS[i]["color"]
            
            if pad_hit_flash[i] > 0:
                c = tuple(int(ch * pad_hit_flash[i]) for ch in color)
                cv2.rectangle(overlay, (px1, pad_y_start), (px2, h), c, -1)
                pad_hit_flash[i] = max(0.0, pad_hit_flash[i] - 0.12)
            
            cv2.rectangle(overlay, (px1, pad_y_start), (px2, h), color, 4)
            cv2.line(overlay, (px1, pad_y_start + 50), (px2, pad_y_start + 50), tuple(int(ch*0.3) for ch in color), 1)
            cv2.putText(overlay, DRUMS[i]["name"], (px1 + 20, pad_y_start + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 4, cv2.LINE_AA)
            cv2.putText(overlay, DRUMS[i]["name"], (px1 + 20, pad_y_start + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        # Strike line
        cv2.line(overlay, (0, pad_y_start), (w, pad_y_start), (255, 255, 255), 4)

        display = cv2.addWeighted(overlay, 0.85, display, 1.0, 0)
        
        # HUD
        cv2.putText(display, "AR NEON BEATS (Enhanced)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 4)
        cv2.putText(display, "Strike downwards through the line!", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Debug overlay toggle
        if show_debug:
            cv2.putText(display, "[DEBUG] CLAHE + EMA + Presence + Cooldown Active", (50, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            n_hands = len(results.hand_landmarks) if results.hand_landmarks else 0
            confirmed = sum(1 for i in range(n_hands) if presence.is_confirmed(i))
            cv2.putText(display, f"Detected: {n_hands} | Confirmed: {confirmed}", (50, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("AR Hand Drums", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('d') or key == ord('D'):
            show_debug = not show_debug

    machine.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
