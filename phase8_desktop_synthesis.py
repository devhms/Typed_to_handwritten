"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    SOVEREIGN DESKTOP MASTERPIECE v8.0 (The Reality)                          ║
║    Objective: 100% Screenshot-Indistinguishable Forensic Synthesis           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  VALIDATION:                                                                 ║
║  1. BIO-KINEMATIC: Human motor paths.                                        ║
║  2. THIXOTROPIC: Speed-sensitive ink.                                         ║
║  3. SCENE CONTEXT: Authentic desk environment + shadows.                     ║
║  4. MOBILE ISP: Realistic lens and sensor artifacts.                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
from pathlib import Path
from thixotropic_pbi import ThixotropicPBI
from scene_synthesizer import SceneSynthesizer
from phase7_masterpiece_synthesis import apply_forensic_smudge, apply_paper_crush

def compose_v8_desktop_photo(title, body, output_path, params=None):
    # 1. Generate High-Fidelity Sovereign Document Line-Art
    canvas_w, canvas_h = 2480, 3508
    ink_layer = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    
    # Paper Base (Photorealistic Substrate)
    fb_p = Path("assets/fiber_map.png")
    fiber_map = cv2.imread(str(fb_p), 0) if fb_p.exists() else None
    
    # We use a white background for the ink layer initially
    ink_canvas = np.ones((canvas_h, canvas_w, 4), dtype=np.uint8) * 255
    ink_canvas[:, :, 3] = 0 # Transparent background
    
    renderer = ThixotropicPBI(params)
    
    margin_left = 320
    line_y_start = 320
    line_spacing = 108
    
    lines = [title] + [l.strip() for l in body.split('\n') if l.strip()]
    for i, txt in enumerate(lines):
        curr_y = line_y_start + i * line_spacing
        if curr_y > canvas_h - 400: break
        
        # Subtle horizontal jitter for human margin variation
        drift_x = np.random.normal(0, 10)
        
        if i == 0: # Header
            renderer.compose_masterpiece(None, txt.upper(), ink_canvas, (margin_left + 400, curr_y-40), char_size=110, fiber_map=fiber_map)
            continue
            
        renderer.compose_masterpiece(None, txt, ink_canvas, (margin_left + drift_x, curr_y), char_size=78, fiber_map=fiber_map)

    # 2. Add realistic paper texture to the doc layer
    paper_p = Path("assets/photorealistic_substrate.jpg")
    doc_layer = cv2.imread(str(paper_p)) if paper_p.exists() else np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 250
    doc_layer = cv2.resize(doc_layer, (canvas_w, canvas_h))
    
    # Forensic Smudge
    ink_canvas = apply_forensic_smudge(ink_canvas)
    
    # Blend Ink into Paper
    alpha = (ink_canvas[:, :, 3] / 255.0)[:, :, None]
    ink_rgb = ink_canvas[:, :, :3].astype(float)
    doc_float = doc_layer.astype(float)
    
    # Multiply Blend
    doc_result = doc_float * (1.0 - alpha) + (doc_float * ink_rgb / 255.0) * alpha
    doc_result = np.clip(doc_result, 0, 255).astype(np.uint8)
    
    # Convert to RGBA for the scene synthesizer
    doc_rgba = cv2.cvtColor(doc_result, cv2.COLOR_RGB2RGBA)
    doc_rgba[:, :, 3] = 255
    
    # 3. Transpose into Scene
    # Find the desk background (looks for the generated artifact)
    brain_dir = Path(r"C:\Users\hafiz\.gemini\antigravity\brain\daa17e3b-d91c-4a28-8942-996855579593")
    desk_files = list(brain_dir.glob("forensic_desk_background_*.png"))
    desk_p = desk_files[0] if desk_files else Path("assets/desk_fallback.jpg")
    
    synthesizer = SceneSynthesizer(str(desk_p))
    final_res = synthesizer.synthesize_photo(doc_rgba, output_path)
    
    return final_res

if __name__ == "__main__":
    TITLE = "SOVEREIGN v8.0 FINAL SUBMISSION"
    BODY = "Dear Professor,\nEnclosed is the completed assignment on Sovereign Biomechanics.\nThis document was synthesized with 100% human motor-path fidelity.\nEvery artifact, from ink-drag to paper-cink, is forensically verified.\nThank you for this opportunity to demonstrate the v8.0 engine."
    out_p = Path("assignments/v8_final_submission.jpg")
    compose_v8_desktop_photo(TITLE, BODY, out_p)
    print(f"v8.0 Masterpiece generated: {out_p}")
