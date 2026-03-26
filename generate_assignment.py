"""
generate_assignment.py — Effortless Typed-to-Handwritten Assignment Generator
-----------------------------------------------------------------------------
Takes a plain text file and outputs a photorealistic image of a handwritten assignment.

Usage:
    python generate_assignment.py my_assignment.txt
"""

import sys
import os
from pathlib import Path
from phase2_ink_synthesis import render_page
from phase3_degradation import degrade_image

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_assignment.py <input_text_file>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    text_content = input_file.read_text(encoding="utf-8")
    
    # Setup output directory
    output_dir = Path("assignments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_h_path = output_dir / f"{input_file.stem}_handwritten_raw.png"
    final_h_path = output_dir / f"{input_file.stem}_handwritten_photo.png"

    print(f"\n[1/2] Rendering text to paper using 'Human Error' model...")
    # Render with the approved paper texture and realistic cross-outs
    render_page(
        title = input_file.stem.replace("_", " ").title(),
        body_text = text_content,
        output_path = raw_h_path
    )

    print(f"[2/2] Applying photorealistic camera degradation (Augraphy 8.2)...")
    # Apply 'heavy' degradation to simulate a real phone capture
    degrade_image(
        input_path = raw_h_path,
        output_path = final_h_path,
        severity = "heavy"
    )

    print(f"\nSUCCESS! Your handwritten assignment is ready:")
    print(f"  Final Image: {final_h_path}")
    print(f"  Opening result...")
    
    # Try to open the result automatically (Windows)
    try:
        os.startfile(final_h_path)
    except:
        pass

if __name__ == "__main__":
    main()
