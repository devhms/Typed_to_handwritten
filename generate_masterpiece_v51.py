import sys
from pathlib import Path
from notebook_renderer import render_notebook_page, NotebookConfig

def generate_style_audit(style='neat', output_name='masterpiece_v51.png'):
    input_file = Path("letter_draft.txt")
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    text_content = input_file.read_text(encoding="utf-8")
    
    # Extract title from first line or use a default
    lines = text_content.strip().split('\n')
    title = lines[0].strip() if lines else "A Letter to Amna"
    body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else text_content
    
    output_path = Path(f"assignments/{output_name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configuration with style
    cfg = NotebookConfig(style=style)
    
    print(f"\n🚀 GENERATING MASTERPIECE v5.1 — Style: {style.upper()}")
    print(f"-------------------------------------------------------")
    print(f"Content: {input_file}")
    print(f"Output:  {output_path}")
    print(f"-------------------------------------------------------")
    
    render_notebook_page(
        body_text=body,
        output_path=str(output_path),
        title=title,
        seed=777,
        config=cfg,
        masterpiece=True
    )
    
    print(f"\n✅ SUCCESS! {style.capitalize()} Masterpiece generated: {output_path}")

if __name__ == "__main__":
    # Generate both for audit
    generate_style_audit(style='neat', output_name='masterpiece_v51_neat.png')
    generate_style_audit(style='messy', output_name='masterpiece_v51_messy.png')
