"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    BIO-KINEMATIC MOTOR ENGINE  v8.0 (The Muscle)                             ║
║    "Handwriting is not drawing; it is movement" — Sigma-Lognormal Model      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CORE PRINCIPLES:                                                            ║
║  1. STROKE PRIMITIVES: Movements are decomposed into lognormal impulses.     ║
║  2. TWO-THIRDS POWER LAW: Angular velocity scales with curvature.            ║
║  3. FRACTAL TREMOR: 1/f noise models neuromuscular oscillation.              ║
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

    def _generate_fractal_noise(self, length, alpha=1.5):
        """Generates 1/f^alpha noise using spectral filtering."""
        white_noise = self.rng.normal(0, 1, length)
        freqs = np.fft.rfftfreq(length)
        # Filter: 1 / f^(alpha/2)
        # Replace 0 frequency with small value to avoid division by zero
        weights = np.where(freqs == 0, 1.0, freqs**(-alpha/2.0))
        weights[0] = 0 # DC component
        fft_white = np.fft.rfft(white_noise)
        fft_filtered = fft_white * weights
        noise = np.fft.irfft(fft_filtered, n=length)
        return noise / (np.std(noise) + 1e-8)

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

                # 3. Apply Micro-Tremor (Fractal 1/f Logic)
        # v8.0: Replacing cumulative sum with true spectral fractal noise
        path_x = np.array(path_x)
        path_y = np.array(path_y)
        path_v = np.array(path_v)
        
        tremor_x = self._generate_fractal_noise(sampling_rate, alpha=1.2)
        tremor_y = self._generate_fractal_noise(sampling_rate, alpha=1.2)
        
        # Scale tremor by velocity (more tremor at lower speeds/starts)
        tremor_scale = 0.002 * (1.0 + 1.0 / (0.1 + path_v))
        path_x += tremor_x * tremor_scale
        path_y += tremor_y * tremor_scale
        
        # 4. Two-Thirds Power Law & Pressure Coupling
        # v8.0: Re-calculating curvature for power-law velocity adjustment
        dx = np.gradient(path_x)
        dy = np.gradient(path_y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature kappa
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)**1.5
        
        # Apply Power Law slowdown: V_adj = V * (1 + kappa)^-0.33
        velocity_modulation = (1.0 + curvature)**(-0.33)
        path_v *= velocity_modulation
        
        # Create a 'pressure' channel (v8.0)
        # Pressure increases with curvature and decreases with velocity
        pressure = (1.2 / (1.0 + path_v)) * (1.0 + curvature * 0.4)
        
        # 5. Non-linear Baseline Drift (The Pivot)
        # Simulating the arm pivoting around the elbow/wrist
        drift_t = np.linspace(0, 1, sampling_rate)
        pivot_drift = 0.05 * np.sin(drift_t * np.pi) # Simple arc
        path_y += pivot_drift
        
        return np.stack([path_x, path_y, path_v, pressure], axis=1)

if __name__ == "__main__":
    engine = BioKinematicEngine()
    test_pts = [[0, 0], [0.5, 0.8], [1.0, 0.2]]
    stroke = engine.generate_human_stroke(test_pts)
    print(f"Generated Bio-Kinematic Stroke: {stroke.shape} points.")
    print(f"Sample Velocity Profile: {stroke[10:15, 2]}")
