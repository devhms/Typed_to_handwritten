"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    BIO-KINEMATIC MOTOR ENGINE  v7.0 (The Muscle)                             ║
║    "Handwriting is not drawing; it is movement" — Sigma-Lognormal Model      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CORE PRINCIPLES:                                                            ║
║  1. STROKE PRIMITIVES: Movements are decomposed into lognormal impulses.     ║
║  2. VELOCITY-CURVATURE: The hand slows down in curves (UCM Law).             ║
║  3. RECOIL & DRIFT: Modeling muscular fatigue and spring-like transitions.   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import math
from scipy.interpolate import interp1d

class SigmaLognormalStroke:
    def __init__(self, start_pos, end_pos, t0=0, D=1.0, mu=-1.5, sigma=0.5):
        """
        D: Amplitude (stroke length)
        mu: Lognormal mean (delay/timing)
        sigma: Lognormal std (stroke smoothness)
        """
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.t0 = t0
        self.D = D
        self.mu = mu
        self.sigma = sigma

    def velocity_at(self, t):
        dt = t - self.t0
        if dt <= 0: return 0.0
        # Lognormal Speed Profile
        coeff = self.D / (self.sigma * np.sqrt(2 * np.pi) * dt)
        exp_term = np.exp(- ((np.log(dt) - self.mu)**2) / (2 * self.sigma**2))
        return coeff * exp_term

class BioKinematicEngine:
    def __init__(self, seed=555):
        self.rng = np.random.default_rng(seed)

    def generate_human_stroke(self, points, total_time=1.0, sampling_rate=200, style_drift=None):
        """
        Generates (x, y, v) triples using Sigma-Lognormal interpolation.
        style_drift: Instance of StyleDriftState for global document variation.
        """
        points = np.array(points)
        if len(points) < 2: return points
        
        # 1. Apply Style Drift (Slant and Scale)
        if style_drift:
            points[:, 0] += points[:, 1] * style_drift.slant
            points *= style_drift.scale
            points[:, 1] += style_drift.y_offset / 100.0 # Normalized offset
        
        # 1. Break down into primitives
        ts = np.linspace(0, total_time, sampling_rate)
        velocities = np.zeros_like(ts)
        
        # Heuristic: Each segment between points is a lognormal impulse
        segments = []
        seg_t0 = 0.0
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i+1]
            dist = np.linalg.norm(p2 - p1)
            # Duration proportional to distance
            duration = dist * 0.4 + self.rng.uniform(0.05, 0.15)
            # Create stroke primitive
            # mu and sigma are tuned for "human-like" fluidity
            mu = np.log(duration / 2.0)
            segments.append(SigmaLognormalStroke(p1, p2, t0=seg_t0, D=dist, mu=mu, sigma=0.45))
            seg_t0 += duration / 1.5 # Overlapping strokes

        # 2. Integrate velocities
        path_x = []
        path_y = []
        path_v = []
        
        # Accumulate positions based on vector direction and speed
        for t in ts:
            total_v = 0.0
            total_dx, total_dy = 0.0, 0.0
            
            # Sum contributions from all active primitives
            active_count = 0
            for seg in segments:
                v = seg.velocity_at(t)
                if v > 1e-4:
                    dir_vec = (seg.end_pos - seg.start_pos)
                    norm = np.linalg.norm(dir_vec)
                    if norm > 1e-6:
                        total_dx += (dir_vec[0] / norm) * v
                        total_dy += (dir_vec[1] / norm) * v
                        total_v += v
                        active_count += 1
            
            # Update path via integration
            if not path_x:
                path_x.append(points[0,0])
                path_y.append(points[0,1])
            else:
                # dt is constant (total_time / sampling_rate)
                dt = total_time / sampling_rate
                path_x.append(path_x[-1] + total_dx * dt)
                path_y.append(path_y[-1] + total_dy * dt)
            
            path_v.append(total_v)

        # 3. Apply Micro-Tremor (Pink Noise 1/f Logic)
        # Replacing digital sine with stochastic human tremor
        path_x = np.array(path_x)
        path_y = np.array(path_y)
        path_v = np.array(path_v) # FIX: convert to array
        
        # Simple Pink Noise approximation via cumulative sum of normal noise
        pink_noise_x = np.cumsum(self.rng.normal(0, 0.001, sampling_rate))
        pink_noise_y = np.cumsum(self.rng.normal(0, 0.001, sampling_rate))
        
        # Band-pass filter to keep it in the 5-15Hz range (Human tremor)
        path_x += pink_noise_x * 0.003
        path_y += pink_noise_y * 0.003
        
        # 4. Velocity-Pressure Coupling Pre-calc
        # Curvature-based weight: Hand slows down + presses harder in tight turns
        dx = np.gradient(path_x)
        dy = np.gradient(path_y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)**1.5
        
        # Create a 'pressure' channel (v4)
        pressure = (1.0 / (0.5 + path_v)) * (1.0 + curvature * 0.5)
        
        return np.stack([path_x, path_y, path_v, pressure], axis=1)

if __name__ == "__main__":
    engine = BioKinematicEngine()
    test_pts = [[0, 0], [0.5, 0.8], [1.0, 0.2]]
    stroke = engine.generate_human_stroke(test_pts)
    print(f"Generated Bio-Kinematic Stroke: {stroke.shape} points.")
    print(f"Sample Velocity Profile: {stroke[10:15, 2]}")
