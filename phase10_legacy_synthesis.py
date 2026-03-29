"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    SOVEREIGN LEGACY SYNTHESIS v10.0 (The Infinite Writer)                    ║
║    Objective: Automated, Multi-Page, Forensic-Grade Assignments              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
from pathlib import Path
from thixotropic_pbi import ThixotropicPBI
from scene_synthesizer import SceneSynthesizer
from allographic_engine import AllographicEngine, StyleDriftState
from bio_kinematic_engine import BioKinematicEngine
from writer_profile import WriterProfile, get_persona
from phase7_masterpiece_synthesis import apply_forensic_smudge, apply_paper_crush

def compose_legacy_page(title, body_lines, output_path, profile=None, page_num=1):
    canvas_w, canvas_h = 2480, 3508
    ink_canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    
    # 1. Initialize Engines with Profile DNA
    if profile is None:
        profile = get_persona("Student")
        
    allograph_gen = AllographicEngine() # Could be extended to use profile.dna["variant_bias"]
    drift = StyleDriftState(seed=101 + page_num)
    
    # Custom drift parameters from profile
    drift.slant = profile.dna["avg_slant"]
    
    bio_engine = BioKinematicEngine()
    pbi_renderer = ThixotropicPBI() # profile.dna["pressure_bias"] could go here
    
    # Assets
    fb_p = Path("assets/fiber_map.png")
    fiber_map = cv2.imread(str(fb_p), 0) if fb_p.exists() else None
    
    margin_left = 320
    curr_y = 350
    line_spacing = 110
    
    for i, line_text in enumerate(body_lines):
        # Apply profile-based margin and baseline instability
        curr_x = margin_left + np.random.normal(0, 15 * profile.dna["baseline_stability"])
        char_size = 85 if (i > 0 or page_num > 1) else 120 # Header only on page 1
        
        for char in line_text:
            if char == ' ':
                curr_x += char_size * profile.dna["word_spacing_mean"]
                continue
            
            # Selection & Mutant Motor Path
            skeleton = allograph_gen.get_variant(char)
            stroke_3d = bio_engine.generate_human_stroke(skeleton, style_drift=drift)
            
            # Scale and Offset
            stroke_3d[:, 0] = stroke_3d[:, 0] * char_size + curr_x
            stroke_3d[:, 1] = stroke_3d[:, 1] * char_size + curr_y + np.random.normal(0, profile.dna["baseline_stability"])
            
            # Physically-Based Ink
            pbi_renderer.render_bio_stroke(stroke_3d, ink_canvas, char_size, fiber_map)
            
            curr_x += char_size * 0.88 + np.random.uniform(-2, 4)
            drift.update()
            
        curr_y += line_spacing + np.random.normal(0, 5 * profile.dna["baseline_stability"])
        if curr_y > canvas_h - 400: break

    # --- Forensic Compositing ---
    paper_p = Path("assets/photorealistic_substrate.jpg")
    doc_layer = cv2.imread(str(paper_p)) if paper_p.exists() else np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 250
    doc_layer = cv2.resize(doc_layer, (canvas_w, canvas_h))
    
    # Smudge and Crush
    ink_canvas = apply_forensic_smudge(ink_canvas, seed=202+page_num)
    doc_layer = apply_paper_crush(doc_layer, ink_canvas[:, :, 3], seed=303+page_num)
    
    # Blend Ink (Multiply)
    alpha = (ink_canvas[:, :, 3] / 255.0)[:, :, None]
    ink_rgb = ink_canvas[:, :, :3].astype(float)
    doc_float = doc_layer.astype(float)
    doc_result = doc_float * (1.0 - alpha) + (doc_float * ink_rgb / 255.0) * alpha
    doc_result = np.clip(doc_result, 0, 255).astype(np.uint8)
    
    # Scene Synthesis
    desk_p = Path("assets/photorealistic_substrate.jpg") # Real desk should be here
    # For now, we output the raw high-fidelity document
    cv2.imwrite(str(output_path), doc_result, [cv2.IMWRITE_JPEG_QUALITY, 98])
    return doc_result

def generate_legacy_assignment(title, body_text, output_dir, persona="Student"):
    profile = get_persona(persona)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple Pagination
    all_lines = [l.strip() for l in body_text.split('\n') if l.strip()]
    lines_per_page = 22
    pages = [all_lines[i:i + lines_per_page] for i in range(0, len(all_lines), lines_per_page)]
    
    results = []
    for i, page_lines in enumerate(pages):
        page_num = i + 1
        out_p = output_dir / f"assignment_page_{page_num:02d}.jpg"
        print(f"  Rendering Page {page_num}...")
        res = compose_legacy_page(title if page_num == 1 else f"{title} (cont.)", page_lines, out_p, profile, page_num)
        results.append(out_p)
        
    return results

if __name__ == "__main__":
    TITLE = "HISTORY ASSIGNMENT: THE INDUSTRIAL REVOLUTION"
    # Long text to trigger pagination
    BODY = (("The Industrial Revolution began in Great Britain in the late 18th century.\n" * 10) + 
            "\n" + 
            ("It changed the world forever by introducing steam power and factories.\n" * 20))
    generate_legacy_assignment(TITLE, BODY, "output/legacy_test", persona="Architect")
