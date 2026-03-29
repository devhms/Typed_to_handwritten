"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    SOVEREIGN ALLOGRAPHIC MASTERPIECE v9.0 (The Human)                        ║
║    Objective: Final Elimination of Allographic Uniformity                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
from pathlib import Path
from thixotropic_pbi import ThixotropicPBI
from scene_synthesizer import SceneSynthesizer
from allographic_engine import AllographicEngine, StyleDriftState
from bio_kinematic_engine import BioKinematicEngine
from phase7_masterpiece_synthesis import apply_forensic_smudge

def compose_v9_allographic_photo(title, body, output_path, params=None):
    canvas_w, canvas_h = 2480, 3508
    ink_canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    
    # Engines
    allograph_gen = AllographicEngine()
    drift = StyleDriftState()
    bio_engine = BioKinematicEngine()
    pbi_renderer = ThixotropicPBI(params)
    
    # Assets
    fb_p = Path("assets/fiber_map.png")
    fiber_map = cv2.imread(str(fb_p), 0) if fb_p.exists() else None
    
    margin_left = 320
    curr_y = 350
    line_spacing = 110
    
    lines = [title] + [l.strip() for l in body.split('\n') if l.strip()]
    
    for i, line_text in enumerate(lines):
        curr_x = margin_left + np.random.normal(0, 15)
        char_size = 85 if i > 0 else 120 # Header larger
        
        for char in line_text:
            if char == ' ':
                curr_x += char_size * 0.65
                continue
            
            # 1. Selection & Mutation (v9.0 Key)
            skeleton = allograph_gen.get_variant(char)
            
            # 2. Bio-Kinematic Motor Path with Style Drift
            stroke_3d = bio_engine.generate_human_stroke(skeleton, style_drift=drift)
            
            # Scale and Offset
            stroke_3d[:, 0] = stroke_3d[:, 0] * char_size + curr_x
            stroke_3d[:, 1] = stroke_3d[:, 1] * char_size + curr_y
            
            # 3. Thixotropic Render
            pbi_renderer.render_bio_stroke(stroke_3d, ink_canvas, char_size, fiber_map)
            
            curr_x += char_size * 0.88 + np.random.uniform(-3, 5)
            drift.update() # Constant style drift
            
        curr_y += line_spacing + np.random.normal(0, 4)
        if curr_y > canvas_h - 400: break

    # --- Blending & Scene ---
    paper_p = Path("assets/photorealistic_substrate.jpg")
    doc_layer = cv2.imread(str(paper_p)) if paper_p.exists() else np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 250
    doc_layer = cv2.resize(doc_layer, (canvas_w, canvas_h))
    
    ink_canvas = apply_forensic_smudge(ink_canvas)
    
    # Blend Ink into Paper
    alpha = (ink_canvas[:, :, 3] / 255.0)[:, :, None]
    doc_float = doc_layer.astype(float)
    ink_rgb = ink_canvas[:, :, :3].astype(float)
    doc_result = doc_float * (1.0 - alpha) + (doc_float * ink_rgb / 255.0) * alpha
    doc_result = np.clip(doc_result, 0, 255).astype(np.uint8)
    doc_rgba = cv2.cvtColor(doc_result, cv2.COLOR_RGB2RGBA)
    doc_rgba[:, :, 3] = 255
    
    # Scene Synthesis (Desktop v8.0 logic)
    brain_dir = Path(r"C:\Users\hafiz\.gemini\antigravity\brain\daa17e3b-d91c-4a28-8942-996855579593")
    desk_files = list(brain_dir.glob("forensic_desk_background_*.png"))
    desk_p = desk_files[0] if desk_files else Path("assets/desk_fallback.jpg")
    
    synthesizer = SceneSynthesizer(str(desk_p))
    final_res = synthesizer.synthesize_photo(doc_rgba, output_path)
    
    return final_res

if __name__ == "__main__":
    TITLE = "SOVEREIGN v9.0 INDISTINGUISHABLE"
    BODY = "This is the final state of the allographic engine.\nEvery 'a', 'e', and 'o' is structuraly unique.\nThe hand drifts, the slant oscillates, the spacing fluctuates.\nThere is zero uniformity because there is zero digital repetition."
    out_p = Path("assignments/v9_allographic_human.jpg")
    compose_v9_allographic_photo(TITLE, BODY, out_p)
    print(f"v9.0 Allographic Masterpiece generated: {out_p}")
