"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    SOVEREIGN WRITER PROFILE SYSTEM v10.0 (The Legacy)                        ║
║    Objective: Cross-Document Stylistic Persistence                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
from pathlib import Path
import numpy as np

class WriterProfile:
    def __init__(self, name="Default", seed=None):
        self.name = name
        self.rng = np.random.default_rng(seed)
        
        # Stylistic DNA (Locked per profile)
        self.dna = {
            "avg_slant": self.rng.uniform(-0.1, 0.1),
            "drift_speed": self.rng.uniform(0.002, 0.008),
            "baseline_stability": self.rng.uniform(0.5, 2.5), # Variance in Y
            "variant_bias": self.rng.uniform(0, 1.0, size=(5,)).tolist(), # Preferred allographs
            "pressure_bias": self.rng.uniform(0.8, 1.2),
            "word_spacing_mean": self.rng.uniform(0.6, 0.9)
        }

    def save(self, folder="profiles"):
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = Path(folder) / f"{self.name.lower()}.json"
        with open(path, 'w') as f:
            json.dump(self.dna, f, indent=2)
        print(f"Profile '{self.name}' saved to {path}")

    @classmethod
    def load(cls, name, folder="profiles"):
        path = Path(folder) / f"{name.lower()}.json"
        if not path.exists():
            return cls(name=name)
        with open(path, 'r') as f:
            data = json.load(f)
        profile = cls(name=name)
        profile.dna = data
        return profile

# PRESETS (The 2026 Sovereign Personas)
def get_persona(type="Architect"):
    if type == "Architect":
        p = WriterProfile("Architect", seed=505)
        p.dna["avg_slant"] = 0.0
        p.dna["baseline_stability"] = 0.2
        p.dna["word_spacing_mean"] = 0.7
    elif type == "Messy":
        p = WriterProfile("Messy", seed=909)
        p.dna["avg_slant"] = -0.15
        p.dna["baseline_stability"] = 5.0
        p.dna["word_spacing_mean"] = 1.1
    else: # Balanced
        p = WriterProfile("Student", seed=101)
    return p
