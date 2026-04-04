from pathlib import Path

from notebook_renderer import NotebookConfig, render_notebook_multi_page, render_notebook_page


def test_render_notebook_page_accepts_legacy_body_text_signature(tmp_path: Path):
    out_path = tmp_path / "legacy.png"
    result = render_notebook_page(
        body_text="Legacy body text call should still work.",
        title="Legacy Title",
        output_path=str(out_path),
        seed=11,
    )

    assert result == str(out_path)
    assert out_path.exists()


def test_render_notebook_page_does_not_mutate_input_config(tmp_path: Path):
    out_path = tmp_path / "preview.png"
    cfg = NotebookConfig(style="neat", drift_intensity=9.9, variation_magnitude=0.007)

    before = (cfg.drift_intensity, cfg.variation_magnitude)
    render_notebook_page(
        body_text="Configuration isolation test.",
        output_path=str(out_path),
        seed=21,
        config=cfg,
    )
    after = (cfg.drift_intensity, cfg.variation_magnitude)

    assert out_path.exists()
    assert before == after


def test_render_notebook_multi_page_returns_paths(tmp_path: Path):
    output_dir = tmp_path / "pages"
    text = " ".join(["word"] * 2600)

    pages = render_notebook_multi_page(
        body_text=text,
        output_dir=str(output_dir),
        title="Paged",
        seed=42,
        max_chars_per_page=700,
    )

    assert len(pages) >= 2
    for path in pages:
        assert Path(path).exists()
