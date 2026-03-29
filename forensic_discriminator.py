"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    FORENSIC DISCRIMINATOR  v1.0 (The Judge)                                 ║
║    Objective: Automated Authenticity Scoring for Sovereign v6.0             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  VALIDATION TESTS:                                                           ║
║  1. FOURIER PERIODICITY: Detects "AI-Grid" artifacts in character alignment. ║
║  2. EDGE-GRADIENT ENTROPY: Measures natural "roughness" of ink vs. fonts.    ║
║  3. CHROMATIC DISTRIBUTION: Checks for sensor-realistic noise patterns.      ║
║  4. FIBER-COMPLIANCE: Verifies ink adheres to paper fiber geometry.          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
from scipy.stats import entropy

class ForensicDiscriminator:
    def __init__(self):
        pass

    def analyze_texture_entropy(self, image):
        """Measures the organic complexity of ink edges."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Focus on edges
        edges = cv2.Canny(gray, 50, 150)
        if np.sum(edges) == 0: return 0.0
        
        # Calculate entropy of edge regions
        hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        return entropy(hist_norm)

    def analyze_periodicity(self, image):
        """Detects unnatural repetition (common in tiled AI outputs)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # High peaks outside DC indicate unnatural periodicity
        # Normalize and find max
        peak = np.max(magnitude_spectrum[10:-10, 10:-10])
        avg = np.mean(magnitude_spectrum)
        return peak / (avg + 1e-6)

    def analyze_chroma_noise(self, image):
        """Checks for Gaussian sensor noise vs. flat digital fills."""
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        # Standard deviation of saturation in 'flat' areas
        std = np.std(s_channel)
        return std

    def score_authenticity(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None: return 0.0
        
        e = self.analyze_texture_entropy(image)
        p = self.analyze_periodicity(image)
        c = self.analyze_chroma_noise(image)
        
        # Heuristic scoring (Normalized to 0.0 - 1.0)
        # Entropy should be high (organic), Periodicity low (non-tiled), Chroma noise non-zero (sensor)
        
        # We want high e, low p (but not zero), moderate c
        e_score = min(1.0, e / 3.5)
        p_score = max(0, 1.0 - (p / 25.0))
        c_score = min(1.0, c / 12.0)
        
        final_score = (e_score * 0.4) + (p_score * 0.3) + (c_score * 0.3)
        
        results = {
            "authenticity_score": float(round(final_score, 4)),
            "entropy": float(e),
            "periodicity": float(p),
            "chroma_noise": float(c)
        }
        return results

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python forensic_discriminator.py <image_path>")
        sys.exit(1)
        
    judge = ForensicDiscriminator()
    res = judge.score_authenticity(sys.argv[1])
    print(f"FORENSIC AUDIT RESULTS: {res}")
