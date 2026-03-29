"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    SOVEREIGN BRAIN  v6.0 (The Architect)                                    ║
║    Objective: Autonomous Optimization of Forensic Authenticity               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  STRATEGY:                                                                   ║
║  1. PERTURB: Randomly adjust PBI parameters (jitter, pressure, depletion).   ║
║  2. RENDER: Synthesize a test document with new params.                      ║
║  3. DISCRIMINATE: Get authenticity score from ForensicDiscriminator.         ║
║  4. EVOLVE: Keep parameters if score increases.                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import numpy as np
from pathlib import Path
from forensic_discriminator import ForensicDiscriminator
from phase5_forensic_synthesis import compose_forensic_assignment

class SovereignBrain:
    def __init__(self):
        self.judge = ForensicDiscriminator()
        self.config_path = Path("brain/sovereign_config.json")
        self.findings_path = Path("brain/v6_findings.md")
        self.best_params = self.load_config()
        self.best_score = 0.0

    def load_config(self):
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {
            "jitter": 0.03,
            "slant": 0.12,
            "hand_waviness": 0.015,
            "waviness_freq": 12.0,
            "ink_depletion_rate": 0.00018,
            "ink_refill_rate": 0.0012,
            "fiber_skip_sens": 0.25,
            "thickness_base": 4.5,
            "alpha_base": 215
        }

    def save_config(self, params):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(params, indent=2))

    def update_findings(self, exp_id, params, score):
        with open(self.findings_path, "a") as f:
            f.write(f"\n- **Exp {exp_id:03d}**: Score {score:.4f} | Params: {list(params.values())[:3]}...")
            if score > self.best_score:
                f.write(" [NEW BEST]")

    def run_experiment(self, exp_id):
        # 1. Perturb
        current_params = self.best_params.copy()
        for key in current_params:
            if isinstance(current_params[key], float):
                perturb = np.random.normal(1.0, 0.05) # 5% variation
                current_params[key] *= perturb
        
        # 2. Render
        test_path = Path(f"brain/exp_{exp_id:03d}_render.png")
        compose_forensic_assignment(
            title="Sovereign v6.0 Optimization",
            body="This sentence is being analyzed by the Sovereign Brain.\nIt is searching for the global optimum of human-ness.",
            output_path=test_path,
            params=current_params
        )
        
        # 3. Discriminate
        results = self.judge.score_authenticity(test_path)
        score = results["authenticity_score"]
        
        # 4. Evolve
        if score > self.best_score:
            print(f"  [FOUND OPTIMUM] Exp {exp_id}: {score:.4f} > {self.best_score:.4f}")
            self.best_score = score
            self.best_params = current_params
            self.save_config(current_params)
            return True
        return False

    def autonomous_loop(self, iterations=10):
        print(f"🚀 Sovereign Brain initializing with {iterations} iterations...")
        for i in range(iterations):
            success = self.run_experiment(i)
            # Update findings
            self.update_findings(i, self.best_params if success else self.best_params, self.best_score)
        print(f"✅ Optimization complete. Best Score: {self.best_score:.4f}")

if __name__ == "__main__":
    brain = SovereignBrain()
    brain.autonomous_loop(5)
