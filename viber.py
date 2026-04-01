import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import os
import time
import random
from collections import deque

MODEL_PATH = "hand_landmarker.task"

# Hand skeleton connections (MediaPipe hand landmark pairs)
SKELETON = [
    (0,1),(1,2),(2,3),(3,4),      # Thumb
    (0,5),(5,6),(6,7),(7,8),      # Index
    (0,9),(9,10),(10,11),(11,12), # Middle
    (0,13),(13,14),(14,15),(15,16),# Ring
    (0,17),(17,18),(18,19),(19,20),# Pinky
    (5,9),(9,13),(13,17),          # Palm cross
]

NEON = [
    (255, 50, 150), (50, 255, 200), (255, 200, 50),
    (200, 50, 255), (50, 200, 255), (100, 255, 50), (255, 100, 50),
]

# ─────────────────────────────────────────────
# Particle
# ─────────────────────────────────────────────
class Particle:
    __slots__ = ['x','y','vx','vy','life','max_life','color','size','gravity','drag']
    def __init__(self, x, y, vx, vy, life, color, size, gravity=0.3, drag=0.98):
        self.x=x;self.y=y;self.vx=vx;self.vy=vy
        self.life=life;self.max_life=life;self.color=color;self.size=size
        self.gravity=gravity;self.drag=drag
    def update(self):
        self.x+=self.vx;self.y+=self.vy
        self.vy+=self.gravity;self.vx*=self.drag;self.vy*=self.drag
        self.life-=1
    def alive(self): return self.life>0
    def alpha(self): return max(0,self.life/self.max_life)

class ShockwaveRing:
    def __init__(self, x, y, color, max_r=200, speed=12):
        self.x=x;self.y=y;self.color=color;self.r=10;self.max_r=max_r;self.speed=speed;self.life=1.0
    def update(self):
        self.r+=self.speed;self.life=max(0,1.0-self.r/self.max_r)
    def alive(self): return self.life>0
    def draw(self, img):
        if self.life<=0:return
        c=tuple(int(ch*self.life) for ch in self.color)
        cv2.circle(img,(int(self.x),int(self.y)),int(self.r),c,max(1,int(4*self.life)),cv2.LINE_AA)

# ─────────────────────────────────────────────
# Lightning Arc Generator
# ─────────────────────────────────────────────
def draw_lightning(img, x1, y1, x2, y2, color, segments=8, jitter=20):
    """Draw a jagged lightning bolt between two points."""
    pts = [(x1, y1)]
    dx = x2 - x1
    dy = y2 - y1
    for i in range(1, segments):
        t = i / segments
        mx = x1 + dx * t + random.uniform(-jitter, jitter)
        my = y1 + dy * t + random.uniform(-jitter, jitter)
        pts.append((int(mx), int(my)))
    pts.append((x2, y2))
    
    # Glow layer (thick, dim)
    for i in range(1, len(pts)):
        glow_c = tuple(int(c * 0.3) for c in color)
        cv2.line(img, pts[i-1], pts[i], glow_c, 5, cv2.LINE_AA)
    # Core layer (thin, bright)
    for i in range(1, len(pts)):
        cv2.line(img, pts[i-1], pts[i], color, 2, cv2.LINE_AA)
    # White hot center
    for i in range(1, len(pts)):
        cv2.line(img, pts[i-1], pts[i], (255,255,255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# Plasma Beam (between two hands)
# ─────────────────────────────────────────────
def draw_plasma_beam(img, x1, y1, x2, y2, t):
    """Draw a thick, animated plasma energy beam."""
    dist = math.hypot(x2-x1, y2-y1)
    if dist < 30: return
    
    segs = max(10, int(dist / 15))
    dx, dy = x2-x1, y2-y1
    
    # Multiple beam layers for thickness
    for layer in range(3):
        jitter = 8 + layer * 6
        pts = []
        for i in range(segs + 1):
            frac = i / segs
            wave = math.sin(frac * 12 + t * 10 + layer) * jitter
            perp_x = -dy / max(1, dist) * wave
            perp_y = dx / max(1, dist) * wave
            px = int(x1 + dx * frac + perp_x)
            py = int(y1 + dy * frac + perp_y)
            pts.append((px, py))
        
        colors = [(255, 200, 50), (50, 200, 255), (200, 50, 255)]
        alpha = [0.6, 0.4, 0.25][layer]
        c = tuple(int(ch * alpha) for ch in colors[layer])
        thick = [4, 7, 10][layer]
        
        for i in range(1, len(pts)):
            cv2.line(img, pts[i-1], pts[i], c, thick, cv2.LINE_AA)
    
    # White core
    core_pts = []
    for i in range(segs + 1):
        frac = i / segs
        wave = math.sin(frac * 12 + t * 10) * 5
        perp_x = -dy / max(1, dist) * wave
        perp_y = dx / max(1, dist) * wave
        core_pts.append((int(x1 + dx*frac + perp_x), int(y1 + dy*frac + perp_y)))
    for i in range(1, len(core_pts)):
        cv2.line(img, core_pts[i-1], core_pts[i], (255,255,255), 2, cv2.LINE_AA)

    # Energy orbs at connection points
    pulse = 0.5 + 0.5 * math.sin(t * 8)
    r = int(15 + pulse * 10)
    cv2.circle(img, (int(x1), int(y1)), r, (255, 200, 50), -1, cv2.LINE_AA)
    cv2.circle(img, (int(x2), int(y2)), r, (255, 200, 50), -1, cv2.LINE_AA)
    cv2.circle(img, (int(x1), int(y1)), r-3, (255,255,255), -1, cv2.LINE_AA)
    cv2.circle(img, (int(x2), int(y2)), r-3, (255,255,255), -1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# CLAHE
# ─────────────────────────────────────────────
class Preprocessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    def process(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        l = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# Ghost / Afterimage System
# ─────────────────────────────────────────────
class GhostTrail:
    def __init__(self, max_ghosts=6):
        self.ghosts = deque(maxlen=max_ghosts)  # list of landmark snapshots
    
    def capture(self, landmarks, w, h):
        pts = []
        for lm in landmarks:
            pts.append((int(lm.x * w), int(lm.y * h)))
        self.ghosts.append({"pts": pts, "time": time.time()})
    
    def draw(self, img, color):
        now = time.time()
        for i, ghost in enumerate(self.ghosts):
            age = now - ghost["time"]
            if age > 0.4: continue
            alpha = max(0, 1.0 - age / 0.4) * 0.4
            c = tuple(int(ch * alpha) for ch in color)
            pts = ghost["pts"]
            # Draw skeleton
            for a, b in SKELETON:
                if a < len(pts) and b < len(pts):
                    cv2.line(img, pts[a], pts[b], c, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────
# Gesture helpers
# ─────────────────────────────────────────────
def fingers_up(lm):
    return [lm[t].y < lm[p].y for t, p in [(8,6),(12,10),(16,14),(20,18)]]

def is_fist(lm): return not any(fingers_up(lm))
def is_open(lm): return all(fingers_up(lm))
def is_peace(lm):
    fu = fingers_up(lm)
    return fu[0] and fu[1] and not fu[2] and not fu[3]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    if not os.path.exists(MODEL_PATH):
        print("ERROR: hand_landmarker.task not found!")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE)
    detector = vision.HandLandmarker.create_from_options(options)

    preprocessor = Preprocessor()
    ghost_trails = [GhostTrail(max_ghosts=8), GhostTrail(max_ghosts=8)]
    particles = []
    shockwaves = []
    
    # Velocity trackers
    prev_palms = [None, None]
    gesture_cooldowns = [0, 0]
    
    # Gravity well particles
    orbit_particles = []

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 50)
    print("  ⚡ THE ULTIMATE VIBER v2 ⚡")
    print("=" * 50)
    print("Just move your hands. Everything is always on.")
    print()
    print("ALWAYS ON:")
    print("  ⚡ Neon skeleton + glowing joints")
    print("  ⚡ Lightning arcs between fingertips")
    print("  ⚡ Ghost afterimage trail")
    print("  ⚡ Comet particles from movement")
    print("  ⚡ Gravity well orbiting your palm")
    print()
    print("TWO HANDS:")
    print("  🔥 Plasma energy beam connecting both hands!")
    print()
    print("GESTURES:")
    print("  ✊ Fist       → Shockwave + energy burst")
    print("  🖐 Open Palm  → Firework explosion")
    print("  ✌ Peace      → Confetti shower")
    print()
    print("  Q - Quit")
    print("=" * 50)

    ghost_timer = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        t = time.time()

        processed = preprocessor.process(frame)
        display = cv2.convertScaleAbs(frame, alpha=0.35, beta=5)
        overlay = np.zeros_like(display, dtype=np.uint8)

        img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)

        palm_positions = []  # For plasma beam
        ghost_timer += 1

        if results.hand_landmarks:
            for hi, landmarks in enumerate(results.hand_landmarks):
                if hi >= 2: break
                
                hand_color = NEON[hi * 3 % len(NEON)]
                
                # Get all landmark pixel positions
                pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in range(21)]
                
                palm_x = int(np.mean([pts[i][0] for i in [0,5,9,13,17]]))
                palm_y = int(np.mean([pts[i][1] for i in [0,5,9,13,17]]))
                palm_positions.append((palm_x, palm_y))
                
                # Velocity
                speed = 0
                if prev_palms[hi] is not None:
                    dx = palm_x - prev_palms[hi][0]
                    dy = palm_y - prev_palms[hi][1]
                    speed = math.hypot(dx, dy)
                prev_palms[hi] = (palm_x, palm_y)
                
                # ═══════════════════════════════════════
                # EFFECT 1: NEON SKELETON (always on)
                # ═══════════════════════════════════════
                pulse = 0.7 + 0.3 * math.sin(t * 5 + hi * 2)
                bone_color = tuple(int(c * pulse) for c in hand_color)
                
                for a, b in SKELETON:
                    # Glow
                    cv2.line(overlay, pts[a], pts[b], tuple(int(c*0.4) for c in bone_color), 6, cv2.LINE_AA)
                    # Core
                    cv2.line(overlay, pts[a], pts[b], bone_color, 3, cv2.LINE_AA)
                    # White center
                    cv2.line(overlay, pts[a], pts[b], (255,255,255), 1, cv2.LINE_AA)
                
                # Glowing joints
                for i in range(21):
                    r = 8 if i in [4,8,12,16,20] else 5  # Bigger for fingertips
                    cv2.circle(overlay, pts[i], r+4, tuple(int(c*0.3) for c in hand_color), -1, cv2.LINE_AA)
                    cv2.circle(overlay, pts[i], r, hand_color, -1, cv2.LINE_AA)
                    cv2.circle(overlay, pts[i], max(1,r-3), (255,255,255), -1, cv2.LINE_AA)
                
                # ═══════════════════════════════════════
                # EFFECT 2: LIGHTNING BETWEEN FINGERTIPS (always on)
                # ═══════════════════════════════════════
                tips = [4, 8, 12, 16, 20]
                # Lightning between adjacent fingertips
                for i in range(len(tips)-1):
                    if random.random() < 0.6:  # Flicker effect
                        lc = random.choice([(200, 200, 255), (150, 200, 255), (255, 200, 150)])
                        dist = math.hypot(pts[tips[i]][0]-pts[tips[i+1]][0], pts[tips[i]][1]-pts[tips[i+1]][1])
                        jit = max(5, int(dist * 0.15))
                        draw_lightning(overlay, pts[tips[i]][0], pts[tips[i]][1],
                                       pts[tips[i+1]][0], pts[tips[i+1]][1], lc, segments=6, jitter=jit)
                
                # Thumb to pinky arc
                if random.random() < 0.4:
                    draw_lightning(overlay, pts[4][0], pts[4][1], pts[20][0], pts[20][1],
                                   (100, 200, 255), segments=10, jitter=25)
                
                # ═══════════════════════════════════════
                # EFFECT 3: GHOST AFTERIMAGES (always on)
                # ═══════════════════════════════════════
                if ghost_timer % 3 == 0:
                    ghost_trails[hi].capture(landmarks, w, h)
                ghost_trails[hi].draw(overlay, hand_color)
                
                # ═══════════════════════════════════════
                # EFFECT 4: GRAVITY WELL (orbiting particles around palm)
                # ═══════════════════════════════════════
                # Spawn orbit particles
                if random.random() < 0.4:
                    angle = random.uniform(0, 2*math.pi)
                    dist_r = random.uniform(40, 80)
                    ox = palm_x + dist_r * math.cos(angle)
                    oy = palm_y + dist_r * math.sin(angle)
                    orbit_particles.append({
                        "x": ox, "y": oy, "angle": angle, "dist": dist_r,
                        "speed": random.uniform(3, 6), "life": random.randint(30, 60),
                        "palm_idx": hi, "color": random.choice(NEON), "size": random.uniform(2, 5)
                    })
                
                # ═══════════════════════════════════════
                # EFFECT 5: COMET PARTICLES (movement-based, easy)
                # ═══════════════════════════════════════
                if speed > 5:
                    count = min(8, int(speed / 4))
                    for _ in range(count):
                        fi = random.choice([4,8,12,16,20])
                        particles.append(Particle(
                            pts[fi][0] + random.uniform(-5,5),
                            pts[fi][1] + random.uniform(-5,5),
                            random.uniform(-3,3), random.uniform(-3,3),
                            random.randint(10, 25), random.choice(NEON),
                            random.uniform(2, 5), gravity=0.05, drag=0.96))
                
                # ═══════════════════════════════════════
                # EFFECT 6: PALM ENERGY AURA
                # ═══════════════════════════════════════
                aura_pulse = 0.5 + 0.5 * math.sin(t * 6 + hi * 3)
                aura_r = 50 + int(aura_pulse * 25) + int(min(speed, 30) * 2)
                cv2.circle(overlay, (palm_x, palm_y), aura_r, tuple(int(c*0.15) for c in hand_color), -1, cv2.LINE_AA)
                cv2.circle(overlay, (palm_x, palm_y), int(aura_r*0.6), tuple(int(c*0.25) for c in hand_color), -1, cv2.LINE_AA)
                cv2.circle(overlay, (palm_x, palm_y), int(aura_r*0.3), tuple(int(c*0.35) for c in hand_color), -1, cv2.LINE_AA)
                
                # ═══════════════════════════════════════
                # GESTURES (very easy triggers)
                # ═══════════════════════════════════════
                if gesture_cooldowns[hi] > 0:
                    gesture_cooldowns[hi] -= 1
                elif speed > 3:
                    if is_fist(landmarks):
                        shockwaves.append(ShockwaveRing(palm_x, palm_y, (0,255,255), max_r=300, speed=18))
                        for _ in range(30):
                            a = random.uniform(0, 2*math.pi)
                            s = random.uniform(5, 15)
                            particles.append(Particle(palm_x, palm_y,
                                s*math.cos(a), s*math.sin(a),
                                random.randint(20,40), (255,255,255),
                                random.uniform(3,6), gravity=0.1))
                        gesture_cooldowns[hi] = 10
                        
                    elif is_open(landmarks):
                        for _ in range(50):
                            a = random.uniform(0, 2*math.pi)
                            s = random.uniform(5, 20)
                            particles.append(Particle(palm_x, palm_y,
                                s*math.cos(a), s*math.sin(a),
                                random.randint(25,50), random.choice(NEON),
                                random.uniform(3,7), gravity=0.15))
                        shockwaves.append(ShockwaveRing(palm_x, palm_y, hand_color, max_r=200, speed=12))
                        gesture_cooldowns[hi] = 12
                        
                    elif is_peace(landmarks):
                        for _ in range(35):
                            particles.append(Particle(
                                pts[8][0]+random.uniform(-30,30), pts[8][1]-30,
                                random.uniform(-8,8), random.uniform(-12,-3),
                                random.randint(30,60), random.choice(NEON),
                                random.uniform(2,5), gravity=0.2, drag=0.97))
                        gesture_cooldowns[hi] = 10
                
                # Ambient sparkles (always)
                for _ in range(2):
                    fi = random.choice([4,8,12,16,20])
                    particles.append(Particle(
                        pts[fi][0], pts[fi][1],
                        random.uniform(-1,1), random.uniform(-3,-0.5),
                        random.randint(8,18), (255,255,255),
                        random.uniform(1,3), gravity=-0.08))

        # ═══════════════════════════════════════
        # EFFECT 7: PLASMA BEAM (two hands visible)
        # ═══════════════════════════════════════
        if len(palm_positions) >= 2:
            p1, p2 = palm_positions[0], palm_positions[1]
            draw_plasma_beam(overlay, p1[0], p1[1], p2[0], p2[1], t)
            
            # Beam particles
            for _ in range(5):
                frac = random.random()
                bx = p1[0] + (p2[0]-p1[0]) * frac + random.uniform(-15,15)
                by = p1[1] + (p2[1]-p1[1]) * frac + random.uniform(-15,15)
                particles.append(Particle(bx, by,
                    random.uniform(-2,2), random.uniform(-4,-1),
                    random.randint(8,15), random.choice([(255,200,50),(50,200,255),(200,50,255)]),
                    random.uniform(2,4), gravity=-0.05, drag=0.95))

        # Update orbit particles
        alive_orbits = []
        for op in orbit_particles:
            op["angle"] += op["speed"] * 0.05
            op["dist"] *= 0.97  # Spiral inward
            op["life"] -= 1
            
            # Get current palm position
            if op["palm_idx"] < len(palm_positions):
                px, py = palm_positions[op["palm_idx"]]
            else:
                op["life"] = 0
                continue
            
            op["x"] = px + op["dist"] * math.cos(op["angle"])
            op["y"] = py + op["dist"] * math.sin(op["angle"])
            
            if op["life"] > 0 and op["dist"] > 5:
                alive_orbits.append(op)
                alpha = op["life"] / 60
                c = tuple(int(ch * alpha) for ch in op["color"])
                sz = max(1, int(op["size"] * alpha))
                cv2.circle(overlay, (int(op["x"]), int(op["y"])), sz+2, tuple(int(x*0.3) for x in c), -1)
                cv2.circle(overlay, (int(op["x"]), int(op["y"])), sz, c, -1)
        orbit_particles = alive_orbits

        # Update regular particles
        alive = []
        for p in particles:
            p.update()
            if p.alive():
                alive.append(p)
                a = p.alpha()
                c = tuple(int(ch*a) for ch in p.color)
                sz = max(1, int(p.size * a))
                cv2.circle(overlay, (int(p.x),int(p.y)), sz+2, tuple(int(x*0.3) for x in c), -1)
                cv2.circle(overlay, (int(p.x),int(p.y)), sz, c, -1)
        particles = alive[:500]  # Cap for performance

        # Shockwaves
        alive_sw = []
        for sw in shockwaves:
            sw.update()
            if sw.alive():
                alive_sw.append(sw)
                sw.draw(overlay)
        shockwaves = alive_sw

        # Composite
        display = cv2.addWeighted(overlay, 1.0, display, 1.0, 0)

        # HUD
        cv2.rectangle(display, (0,0), (w,40), (10,10,15), -1)
        cv2.putText(display, "THE ULTIMATE VIBER v2", (30, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 50), 2, cv2.LINE_AA)
        n = len(results.hand_landmarks) if results.hand_landmarks else 0
        cv2.putText(display, f"Hands: {n} | FX: {len(particles)+len(orbit_particles)} | Q: Quit",
                    (550, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

        cv2.imshow("The Ultimate Viber v2", display)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
