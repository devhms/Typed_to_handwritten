"""
generate_assignment.py — Notebook Renderer Assignment Generator
---------------------------------------------------------------
Takes a plain text file and outputs a multi-page handwritten notebook assignment.

Usage:
    python generate_assignment.py <input.txt>
"""

import sys
import os
from pathlib import Path
from notebook_renderer import render_notebook_page, render_notebook_multi_page

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_assignment.py <input_text_file>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    text_content = input_file.read_text(encoding="utf-8")
    
    # Extract title from first line
    lines = text_content.strip().split('\n')
    title = lines[0].strip() if lines else input_file.stem.replace("_", " ").title()
    body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else text_content
    
    # Output directory
    output_dir = Path("assignments") / input_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  NOTEBOOK RENDERER -- GENERATING ASSIGNMENT")
    print(f"  ------------------------------------------")
    print(f"  Title:   {title}")
    print(f"  Output:  {output_dir}")
    print(f"  ------------------------------------------\n")

    results = render_notebook_multi_page(
        body_text=body,
        output_dir=str(output_dir),
        title=title,
        seed=42,
    )

    print(f"\n  SUCCESS! All pages generated in {output_dir}")
    for i, p in enumerate(results):
        print(f"  Page {i+1}: {p}")
        
    # Open the first page automatically
    if results:
        try:
            os.startfile(results[0])
        except Exception:
            pass

if __name__ == "__main__":
    main()
