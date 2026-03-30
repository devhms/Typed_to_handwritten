import sys
from pathlib import Path
from notebook_renderer import render_notebook_page

def main():
    input_file = Path("waterfall_summary.txt")
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    text_content = input_file.read_text(encoding="utf-8")
    
    # Extract title from first line or use a default
    lines = text_content.strip().split('\n')
    title = lines[0].strip() if lines else "Waterfall Model Analysis"
    body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else text_content
    
    output_path = Path("assignments/waterfall_masterpiece.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 GENERATING ABSOLUTE PHYSICALITY MASTERPIECE")
    print(f"----------------------------------------------")
    print(f"Title:  {title}")
    print(f"File:   {input_file}")
    print(f"Output: {output_path}")
    print(f"----------------------------------------------")
    
    render_notebook_page(
        body_text=body,
        output_path=str(output_path),
        title=title,
        seed=42,
        masterpiece=True
    )
    
    print(f"\n✅ SUCCESS! Masterpiece assignment generated: {output_path}")

if __name__ == "__main__":
    main()
