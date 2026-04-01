import numpy as np
import sounddevice as sd
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os
import random

MODEL_PATH = "hand_landmarker.task"

# C Major Scale 
SCALE = [
    ("C5", 523.25, (255, 100, 255)), # Pinkish
    ("B4", 493.88, (150, 100, 255)), # Purple
    ("A4", 440.00, (50, 150, 255)),  # Blue
    ("G4", 392.00, (50, 255, 200)),  # Cyan
    ("F4", 349.23, (50, 255, 100)),  # Green
    ("E4", 329.63, (200, 255, 50)),  # Yellow-Green
    ("D4", 293.66, (255, 200, 50)),  # Orange
    ("C4", 261.63, (255, 50, 100))   # Red
]

SONGS = [
    {
        "name": "Ode To Joy",
        "notes": ["E4","E4","F4","G4","G4","F4","E4","D4","C4","C4","D4","E4","E4","D4","D4"]
    },
    {
        "name": "Twinkle Twinkle",
        "notes": ["C4","C4","G4","G4","A4","A4","G4","F4","F4","E4","E4","D4","D4","C4"]
    },
    {
        "name": "Mario Intro",
        "notes": ["E5","E5","E5","C5","E5","G4"]
    }
]

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-6, 6)
        self.vy = random.uniform(-6, 6)
        self.life = 1.0
        self.color = color
        self.size = random.uniform(4, 10)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 0.05
        self.size *= 0.95

class AudioSynth:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.target_freq = 0.0
        self.current_freq = 0.0
        self.target_vol = 0.0
        self.current_vol = 0.0
        self.phase = 0.0
        
        self.stream = sd.OutputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=self._audio_callback
        )
        self.stream.start()

    def set_target(self, freq, vol):
        self.target_freq = freq
        self.target_vol = vol

    def _audio_callback(self, outdata, frames, time, status):
        freqs = np.linspace(self.current_freq, self.target_freq, frames)
        vols = np.linspace(self.current_vol, self.target_vol, frames)
        
        dPhase = 2.0 * np.pi * freqs / self.sample_rate
        phases = self.phase + np.cumsum(dPhase)
        self.phase = phases[-1] % (2.0 * np.pi)
        
        wave = np.sin(phases) * 0.6 + np.sin(phases * 2) * 0.2 + np.sin(phases * 3) * 0.1
        outdata[:, 0] = wave * vols
        
        self.current_freq = self.target_freq
        self.current_vol = self.target_vol

    def close(self):
        self.stream.stop()
        self.stream.close()

def draw_rounded_rect(img, pt1, pt2, color, thickness=cv2.FILLED, r=10):
    x1, y1 = pt1
    x2, y2 = pt2
    if thickness == cv2.FILLED:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, cv2.FILLED)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, cv2.FILLED)
        cv2.circle(img, (x1+r, y1+r), r, color, cv2.FILLED)
        cv2.circle(img, (x2-r, y1+r), r, color, cv2.FILLED)
        cv2.circle(img, (x1+r, y2-r), r, color, cv2.FILLED)
        cv2.circle(img, (x2-r, y2-r), r, color, cv2.FILLED)
    else:
        cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
        cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
        cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
        cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
        cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+r, y2-r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r, y2-r), (r, r), 0, 0, 90, color, thickness)

def main():
    if not os.path.exists(MODEL_PATH):
        print("ERROR: hand_landmarker.task not found!")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,         
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)
    synth = AudioSynth()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_song_idx = 0
    song_progress = 0
    note_played_latch = False
    
    num_zones = len(SCALE)
    particles = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        
        frame = cv2.flip(frame, 1)
        # Apply a subtle dark vignette/tint to the raw frame to make neon colors pop
        frame = cv2.convertScaleAbs(frame, alpha=0.6, beta=10)
        
        h, w, c = frame.shape
        zone_height = (h - 160) // num_zones
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)
        
        vol = 0.0
        active_zone = -1
        ix, iy, tx, ty = 0, 0, 0, 0

        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            ix = int(landmarks[8].x * w)
            iy = int(landmarks[8].y * h)
            tx = int(landmarks[4].x * w)
            ty = int(landmarks[4].y * h)
            
            dist = math.sqrt((tx - ix)**2 + (ty - iy)**2)
            max_dist = 220.0
            vol = max(0.0, min(1.0, (dist - 25) / max_dist))
            
            # Active zone from Y offset 140
            y_offset = iy - 140
            active_zone = int(y_offset // zone_height)
            active_zone = max(0, min(num_zones - 1, active_zone))
            
            freq = SCALE[active_zone][1]
            synth.set_target(freq, vol)
            
            # Hand glowing connection line
            thickness = max(2, int(vol * 20))
            cv2.line(frame, (tx, ty), (ix, iy), (255, 255, 255), thickness, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), 12 + int(vol*5), SCALE[active_zone][2], cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), 10, (200, 200, 200), cv2.FILLED, cv2.LINE_AA)
            
            # Spawn continuous particles if playing loud
            if vol > 0.5 and random.random() < 0.3:
                particles.append(Particle(ix, iy, SCALE[active_zone][2]))
        else:
            synth.set_target(synth.current_freq, 0.0)

        # Update and draw particles
        for p in particles:
            p.update()
            if p.life > 0:
                c = tuple(int(x * p.life) for x in p.color)
                cv2.circle(frame, (int(p.x), int(p.y)), max(1, int(p.size)), c, cv2.FILLED, cv2.LINE_AA)
        particles = [p for p in particles if p.life > 0]

        # Draw UI (Glassmorphism overlay)
        overlay = np.zeros_like(frame, dtype=np.uint8)
        
        # 1. Floating Sci-Fi Keys (Left side)
        key_x = 30
        key_w = 120
        for i in range(num_zones):
            y1 = 140 + i * zone_height
            y2 = y1 + zone_height - 10
            color = SCALE[i][2]
            
            if i == active_zone and vol > 0.1:
                # Active glowing key
                draw_rounded_rect(overlay, (key_x, y1), (key_x + key_w + int(vol*40), y2), color, cv2.FILLED, 10)
                cv2.putText(overlay, SCALE[i][0], (key_x + 20 + int(vol*10), y2 - int(zone_height*0.3)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
            else:
                # Inactive key
                dim_color = tuple(int(c * 0.3) for c in color)
                draw_rounded_rect(overlay, (key_x, y1), (key_x + key_w, y2), dim_color, cv2.FILLED, 10)
                draw_rounded_rect(overlay, (key_x, y1), (key_x + key_w, y2), color, 2, 10)
                cv2.putText(overlay, SCALE[i][0], (key_x + 20, y2 - int(zone_height*0.3)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # 2. Modern Volume LED Meter (Right side)
        vol_x = w - 80
        vol_y1 = h - 60
        vol_y2 = 140
        num_leds = 25
        led_h = (vol_y1 - vol_y2) / num_leds
        active_leds = int(vol * num_leds)
        
        # Draw meter background panel
        draw_rounded_rect(overlay, (vol_x - 15, vol_y2 - 20), (vol_x + 55, vol_y1 + 20), (20, 20, 20), cv2.FILLED, 15)
        cv2.putText(overlay, "VOL", (vol_x - 5, vol_y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        
        for i in range(num_leds):
            led_y = vol_y1 - int((i + 1) * led_h) + 2
            if i < active_leds:
                r = int(255 * (i / num_leds))
                g = int(255 * (1 - i / num_leds))
                c = (0, g, r) # BGR
                draw_rounded_rect(overlay, (vol_x, led_y), (vol_x + 40, led_y + int(led_h) - 4), c, cv2.FILLED, 3)
            else:
                draw_rounded_rect(overlay, (vol_x, led_y), (vol_x + 40, led_y + int(led_h) - 4), (40, 40, 40), cv2.FILLED, 3)

        # 3. Interactive Song Guide Panel (Top)
        song = SONGS[current_song_idx]
        target_note = song["notes"][song_progress] if song_progress < len(song["notes"]) else "DONE"
        
        # Panel Background
        draw_rounded_rect(overlay, (key_x, 20), (w - 30, 110), (30, 30, 40), cv2.FILLED, 15)
        draw_rounded_rect(overlay, (key_x, 20), (w - 30, 110), (100, 150, 255), 2, 15)
        
        cv2.putText(overlay, f"S: Switch Song | R: Reset | Playing: {song['name']}", 
                    (key_x + 20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # Render Notes
        text_x = key_x + 20
        for i, n in enumerate(song["notes"]):
            if i < song_progress:
                c = (100, 100, 100) # Played
                scale = 0.7
                thick = 2
            elif i == song_progress:
                c = (0, 255, 255) # Current Target (Yellow)
                scale = 1.2
                thick = 3
                # Draw glowing box around target note
                cv2.rectangle(overlay, (text_x-5, 60), (text_x + int(45*scale), 100), (0, 200, 200), 2)
            else:
                c = (255, 255, 255) # Upcoming
                scale = 0.8
                thick = 2
            
            cv2.putText(overlay, n, (text_x, 90), cv2.FONT_HERSHEY_SIMPLEX, scale, c, thick, cv2.LINE_AA)
            text_x += int(55 * scale)

        # Song Progression Logic
        if target_note != "DONE" and vol > 0.4:
            if SCALE[active_zone][0] == target_note:
                if not note_played_latch:
                    song_progress += 1
                    note_played_latch = True
                    # Burst particles on success
                    for _ in range(20):
                        particles.append(Particle(ix, iy, (0, 255, 255)))
        else:
            if vol < 0.15:
                note_played_latch = False

        if target_note == "DONE":
            cv2.putText(overlay, "SONG COMPLETE! Perfect!", (w//2 - 150, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

        # Blend overlay onto frame (Glassmorphism look)
        frame = cv2.addWeighted(overlay, 0.85, frame, 1.0, 0)

        cv2.imshow("AR Hand Music", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            current_song_idx = (current_song_idx + 1) % len(SONGS)
            song_progress = 0
            note_played_latch = False
        elif key == ord('r') or key == ord('R'):
            song_progress = 0
            note_played_latch = False

    synth.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
