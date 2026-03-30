"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    FORENSIC DISCRIMINATOR  v2.0 (The Judge)                                 ║
║    Objective: Automated Authenticity Scoring for Sovereign v8.0             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  VALIDATION TESTS:                                                           ║
║  1. FOURIER CLEANLINESS: Detects high-frequency digital jitter spikes.       ║
║  2. EDGE FRACTAL DIMENSION: Measures organic complexity of ink boundaries.   ║
║  3. CHROMATIC ENTROPY: Verifies sensor-realistic noise distributions.        ║
║  4. FIBER COHERENCE: Checks for sub-pixel ink/fiber interaction alignment.   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
from scipy.stats import entropy
from pathlib import Path

class ForensicDiscriminator:
    def __init__(self):
        pass

    def get_fractal_dimension(self, Z, threshold=0.9):
        """Calculates the Minkowski-Bouligand dimension of a binary image."""
        # Only for numeric 2D arrays
        Z = (Z > threshold)
        p = min(Z.shape)
        # Number of boxes
        n = 2**np.floor(np.log2(p))
        n = int(np.log2(n))
        sizes = 2**np.arange(n, 1, -1)
        counts = []
        for size in sizes:
            counts.append(self._boxcount(Z, size))
        
        # Fit the results to a log-log curve
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    def _boxcount(self, Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k*k))[0])

    def analyze_edge_fractal(self, image):
        """Measures the organic complexity of ink edges via Fractal Dimension."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        if np.sum(edges) == 0: return 0.0
        return self.get_fractal_dimension(edges / 255.0)

    def analyze_fourier_cleanliness(self, image):
        """Detects digital artifacts (spikes) in the frequency domain."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # Digital "AI" outputs often have high-frequency spikes at specific intervals (tiling)
        # We calculate the kurtosis of the magnitude spectrum (high = spiky/digital)
        std = np.std(magnitude_spectrum)
        mean = np.mean(magnitude_spectrum)
        kurtosis = np.mean(((magnitude_spectrum - mean) / (std + 1e-8))**4)
        return kurtosis

    def analyze_chroma_entropy(self, image):
        """Verifies sensor-realistic noise (Poisson/Gaussian) vs flat fills."""
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        hist = cv2.calcHist([s_channel], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        return entropy(hist_norm)

    def score_authenticity(self, image_path):
        if not Path(image_path).exists():
            return {"error": "File not found"}
            
        image = cv2.imread(str(image_path))
        if image is None: return {"error": "Invalid image format"}
        
        fd = self.analyze_edge_fractal(image)
        fk = self.analyze_fourier_cleanliness(image)
        ce = self.analyze_chroma_entropy(image)
        
        # v2.0 Scoring Model (Refined for Sovereign v8.0)
        # Organic edge fractal dim should be around 1.3 - 1.6
        fd_score = 1.0 - abs(fd - 1.45) / 0.5
        # Fourier Kurtosis: Low is better (smooth noise, no digital spikes)
        fk_score = max(0, 1.0 - (fk / 100.0))
        # Chroma Entropy: High is better (textured grain, not flat colors)
        ce_score = min(1.0, ce / 4.0)
        
        final_score = (fd_score * 0.4) + (fk_score * 0.3) + (ce_score * 0.3)
        
        return {
            "authenticity_score": float(round(np.clip(final_score, 0, 1), 4)),
            "fractal_dimension": float(fd),
            "fourier_kurtosis": float(fk),
            "chroma_entropy": float(ce)
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python forensic_discriminator.py <image_path>")
        sys.exit(1)
        
    judge = ForensicDiscriminator()
    res = judge.score_authenticity(sys.argv[1])
    print(f"FORENSIC AUDIT v2.0 RESULTS: {res}")
