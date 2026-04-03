import pytest
import numpy as np
from PIL import Image

from notebook_renderer import HandwritingRenderer, NotebookConfig


FAST_SAMPLE = {
    "dx": 0.0,
    "dy": 0.0,
    "rotation": 0.0,
    "scale": 1.0,
    "component_idx": 0,
}

TRANSFORMED_SAMPLE = {
    "dx": 0.6,
    "dy": 0.0,
    "rotation": 0.0,
    "scale": 1.0,
    "component_idx": 0,
}


def _make_renderer(masterpiece_overlay_alpha: float = 0.0) -> HandwritingRenderer:
    cfg = NotebookConfig(
        variation_magnitude=0.0,
        body_font_size=62,
        masterpiece_overlay_alpha=masterpiece_overlay_alpha,
    )
    return HandwritingRenderer(cfg, seed=123)


def _render_preview_top(char: str, sample: dict, threshold: int) -> int:
    renderer = _make_renderer()
    renderer._mixture.sample = lambda: sample
    canvas = Image.new("RGBA", (420, 260), (0, 0, 0, 0))
    renderer.render_char(canvas, char, x=160, y_baseline=170)
    alpha = np.array(canvas)[:, :, 3]
    ys = np.where(alpha > threshold)[0]
    assert ys.size > 0, f"No rendered pixels for '{char}' in preview mode"
    return int(ys.min())


def _render_masterpiece_point_top(char: str, sample: dict) -> int:
    renderer = _make_renderer(masterpiece_overlay_alpha=0.0)
    renderer._mixture.sample = lambda: sample

    captured = {"points": None}

    def _capture_points(stroke_v, canvas, groove_canvas=None, fiber_map=None):
        captured["points"] = np.array(stroke_v, copy=True)

    renderer.pbi_engine.render_bio_stroke = _capture_points

    masterpiece_canvas = np.zeros((260, 420, 4), dtype=np.uint8)
    groove_canvas = np.zeros((260, 420), dtype=np.uint8)
    fiber_map = np.zeros((260, 420), dtype=np.uint8)

    canvas = Image.new("RGBA", (420, 260), (0, 0, 0, 0))
    renderer.render_char(
        canvas,
        char,
        x=160,
        y_baseline=170,
        masterpiece_canvas=masterpiece_canvas,
        groove_canvas=groove_canvas,
        fiber_map=fiber_map,
    )

    points = captured["points"]
    assert points is not None and len(points) > 0, (
        f"No PBI points captured for '{char}'"
    )
    return int(np.min(points[:, 1]))


@pytest.mark.parametrize("char", ["f", "h", "k", "l", "b", "Thequick"])
def test_preview_transformed_path_preserves_upper_extent(char: str):
    top_fast = _render_preview_top(char, FAST_SAMPLE, threshold=0)
    top_transformed = _render_preview_top(char, TRANSFORMED_SAMPLE, threshold=0)
    assert abs(top_fast - top_transformed) <= 1


@pytest.mark.parametrize("char", ["f", "h", "k", "l", "b", "Thequick"])
def test_masterpiece_transformed_path_preserves_upper_extent(char: str):
    top_preview_mask = _render_preview_top(char, TRANSFORMED_SAMPLE, threshold=50)
    top_masterpiece_points = _render_masterpiece_point_top(char, TRANSFORMED_SAMPLE)
    assert abs(top_preview_mask - top_masterpiece_points) <= 1


@pytest.mark.parametrize("char", ["f", "h", "k", "l", "b", "Thequick"])
def test_masterpiece_overlay_keeps_upper_extent(char: str):
    renderer_overlay = _make_renderer(masterpiece_overlay_alpha=0.40)
    renderer_overlay._mixture.sample = lambda: TRANSFORMED_SAMPLE

    canvas = Image.new("RGBA", (420, 260), (0, 0, 0, 0))
    masterpiece_canvas = np.zeros((260, 420, 4), dtype=np.uint8)
    groove_canvas = np.zeros((260, 420), dtype=np.uint8)
    fiber_map = np.zeros((260, 420), dtype=np.uint8)

    renderer_overlay.render_char(
        canvas,
        char,
        x=160,
        y_baseline=170,
        masterpiece_canvas=masterpiece_canvas,
        groove_canvas=groove_canvas,
        fiber_map=fiber_map,
    )

    alpha = np.array(canvas)[:, :, 3]
    ys = np.where(alpha > 0)[0]
    assert ys.size > 0, f"No overlay pixels for '{char}' in masterpiece mode"

    top_preview_fast = _render_preview_top(char, FAST_SAMPLE, threshold=0)
    assert abs(int(ys.min()) - top_preview_fast) <= 1
