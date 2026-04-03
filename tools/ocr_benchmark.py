import argparse
import json
import re
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import easyocr

from notebook_renderer import NotebookConfig, render_notebook_page


BENCHMARK_SAMPLES = [
    "The quick brown fox jumps over the lazy dog near the market square.",
    "Please submit the assignment before Friday and keep margins neat.",
    "Students should revise chapter seven and practice geometry proofs.",
    "Reliable software requires careful testing, review, and documentation.",
    "Bright sunlight through the window can reduce contrast in old notebooks.",
    "Our teacher highlighted grammar mistakes and asked for a clean rewrite.",
    "Accurate records help every department track progress during the semester.",
    "Many people prefer black ink, but blue ink is easier on ruled paper.",
    "Keep each heading short and leave one blank line before the next section.",
    "Engineers often sketch ideas quickly before writing the final explanation.",
    "Typing speed varies widely when the keyboard layout changes unexpectedly.",
    "Good handwriting balances rhythm, spacing, pressure, and baseline control.",
]

OCR_ALLOWLIST = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.-'"


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)

    prev = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        curr = [i]
        for j, hyp_item in enumerate(hyp, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            substitute = prev[j - 1] + (0 if ref_item == hyp_item else 1)
            curr.append(min(ins, delete, substitute))
        prev = curr
    return prev[-1]


def error_rate(ref_units: Sequence[str], hyp_units: Sequence[str]) -> dict:
    dist = levenshtein_distance(ref_units, hyp_units)
    denom = max(1, len(ref_units))
    return {
        "distance": int(dist),
        "reference_length": int(len(ref_units)),
        "rate": float(dist / denom),
    }


def preprocess_image_competitor_v1(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    return cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)


def preprocess_image_upscale(image_bgr, scale: float):
    return cv2.resize(
        image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )


def assemble_text_competitor_v1(ocr_boxes) -> str:
    items = []
    for item in ocr_boxes:
        if len(item) < 3:
            continue
        bbox, text, conf = item[0], str(item[1]), float(item[2])
        if conf < 0.20 or not text.strip():
            continue

        xs = [float(pt[0]) for pt in bbox]
        ys = [float(pt[1]) for pt in bbox]
        x_min = min(xs)
        y_min = min(ys)
        y_max = max(ys)
        h = max(1.0, y_max - y_min)
        items.append(
            {
                "text": text,
                "x": x_min,
                "y": y_min,
                "h": h,
            }
        )

    if not items:
        return ""

    median_h = statistics.median([it["h"] for it in items])
    line_threshold = max(8.0, 0.65 * median_h)
    items.sort(key=lambda it: (it["y"], it["x"]))

    lines = []
    for it in items:
        if not lines:
            lines.append([it])
            continue
        line_y = statistics.mean([x["y"] for x in lines[-1]])
        if abs(it["y"] - line_y) <= line_threshold:
            lines[-1].append(it)
        else:
            lines.append([it])

    ordered_tokens = []
    for line in lines:
        line.sort(key=lambda it: it["x"])
        ordered_tokens.extend(it["text"] for it in line)

    return " ".join(ordered_tokens).strip()


def ocr_easyocr(
    reader: easyocr.Reader,
    image_path: Path,
    preprocess_mode: str,
    preprocessed_image_path: Path | None,
) -> str:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return ""

    if preprocess_mode == "none":
        paragraph_results = reader.readtext(image_bgr, detail=0, paragraph=True)
        if isinstance(paragraph_results, list):
            return " ".join(str(x) for x in paragraph_results)
        return str(paragraph_results or "")

    if preprocess_mode == "competitor_v1":
        image_bgr = preprocess_image_competitor_v1(image_bgr)
        if preprocessed_image_path is not None:
            preprocessed_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preprocessed_image_path), image_bgr)

    read_kwargs = {}
    sort_text = False
    if preprocess_mode == "competitor_v3":
        image_bgr = preprocess_image_upscale(image_bgr, 1.5)
        read_kwargs = {
            "decoder": "beamsearch",
            "beamWidth": 5,
            "allowlist": OCR_ALLOWLIST,
        }
        if preprocessed_image_path is not None:
            preprocessed_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preprocessed_image_path), image_bgr)
    elif preprocess_mode == "competitor_v4":
        read_kwargs = {
            "decoder": "beamsearch",
            "beamWidth": 5,
            "allowlist": OCR_ALLOWLIST,
        }
    elif preprocess_mode == "competitor_v5":
        read_kwargs = {
            "decoder": "greedy",
            "allowlist": OCR_ALLOWLIST,
        }
        sort_text = True

    ocr_boxes = reader.readtext(image_bgr, detail=1, paragraph=False, **read_kwargs)
    if preprocess_mode in {
        "competitor_v1",
        "competitor_v2",
        "competitor_v3",
        "competitor_v4",
    }:
        assembled = assemble_text_competitor_v1(ocr_boxes)
        if assembled:
            return assembled

    if sort_text and ocr_boxes:
        assembled = assemble_text_competitor_v1(ocr_boxes)
        if assembled:
            return assembled

    # Fallback for robustness.
    if ocr_boxes:
        tokens = []
        for item in ocr_boxes:
            if isinstance(item, dict):
                text_val = item.get("text", "")
                if text_val:
                    tokens.append(str(text_val))
                continue
            if isinstance(item, (list, tuple)):
                try:
                    _, text_val, *_ = item
                except Exception:
                    continue
                if text_val:
                    tokens.append(str(text_val))
        return " ".join(tokens)

    results = reader.readtext(image_bgr, detail=0, paragraph=True)
    if isinstance(results, list):
        return " ".join(str(x) for x in results)
    return str(results or "")


def build_config(style: str) -> NotebookConfig:
    return NotebookConfig(
        page_w=1400,
        page_h=960,
        style=style,
        line_spacing=92,
        first_line_y=260,
        margin_x=125,
        text_start_x=170,
        text_max_x=1290,
        body_font_size=60,
        header_font_size=56,
        title_font_size=62,
        word_spacing_base=16,
    )


def run_benchmark(
    runs: int,
    seed: int,
    mode: str,
    style: str,
    output_dir: Path,
    label: str,
    preprocess_mode: str,
) -> dict:
    images_dir = output_dir / "images" / label
    images_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir = output_dir / "preprocessed" / label

    cfg = build_config(style=style)
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    per_sample = []
    started_at = time.time()

    for i in range(runs):
        text = BENCHMARK_SAMPLES[i % len(BENCHMARK_SAMPLES)]
        sample_id = f"sample_{i:03d}"
        image_path = images_dir / f"{sample_id}_{mode}.png"

        blocks = [{"type": "paragraph", "content": text}]
        render_notebook_page(
            document_blocks=blocks,
            output_path=str(image_path),
            seed=seed + i,
            config=cfg,
            masterpiece=(mode == "masterpiece"),
        )

        preprocessed_image_path = None
        if preprocess_mode != "none":
            preprocessed_image_path = preprocessed_dir / f"{sample_id}_{mode}.png"

        ocr_text = ocr_easyocr(
            reader,
            image_path,
            preprocess_mode=preprocess_mode,
            preprocessed_image_path=preprocessed_image_path,
        )

        ref_norm = normalize_text(text)
        ocr_norm = normalize_text(ocr_text)

        cer_stats = error_rate(list(ref_norm), list(ocr_norm))
        wer_stats = error_rate(ref_norm.split(), ocr_norm.split())

        per_sample.append(
            {
                "sample_id": sample_id,
                "seed": seed + i,
                "image_path": str(image_path),
                "reference_text": text,
                "ocr_text": ocr_text,
                "reference_normalized": ref_norm,
                "ocr_normalized": ocr_norm,
                "cer": cer_stats,
                "wer": wer_stats,
            }
        )

    cer_values = [row["cer"]["rate"] for row in per_sample]
    wer_values = [row["wer"]["rate"] for row in per_sample]

    ended_at = time.time()

    return {
        "meta": {
            "label": label,
            "engine": "easyocr",
            "mode": mode,
            "style": style,
            "preprocess_mode": preprocess_mode,
            "runs": runs,
            "seed": seed,
            "started_at_unix": started_at,
            "ended_at_unix": ended_at,
            "duration_seconds": ended_at - started_at,
        },
        "aggregate": {
            "cer_mean": float(statistics.mean(cer_values)) if cer_values else 0.0,
            "cer_median": float(statistics.median(cer_values)) if cer_values else 0.0,
            "wer_mean": float(statistics.mean(wer_values)) if wer_values else 0.0,
            "wer_median": float(statistics.median(wer_values)) if wer_values else 0.0,
        },
        "per_sample": per_sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OCR benchmark with CER/WER JSON report."
    )
    parser.add_argument(
        "--runs", type=int, default=12, help="Number of rendered samples."
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--mode",
        choices=["preview", "masterpiece"],
        default="preview",
        help="Render mode used for benchmarking.",
    )
    parser.add_argument(
        "--style",
        choices=["neat", "messy"],
        default="neat",
        help="Notebook style profile.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/ocr_benchmark"),
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional run label used in report filename and images directory.",
    )
    parser.add_argument(
        "--preprocess",
        choices=[
            "none",
            "competitor_v1",
            "competitor_v2",
            "competitor_v3",
            "competitor_v4",
            "competitor_v5",
        ],
        default="none",
        help="OCR-side preprocessing pipeline.",
    )

    args = parser.parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    label = args.label or f"{args.mode}-{args.style}-{timestamp}"

    report = run_benchmark(
        runs=args.runs,
        seed=args.seed,
        mode=args.mode,
        style=args.style,
        output_dir=args.output_dir,
        label=label,
        preprocess_mode=args.preprocess,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"report_{label}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    aggregate = report["aggregate"]
    print(f"[OCR-BENCH] report: {report_path}")
    print(f"[OCR-BENCH] CER mean: {aggregate['cer_mean']:.4f}")
    print(f"[OCR-BENCH] WER mean: {aggregate['wer_mean']:.4f}")


if __name__ == "__main__":
    main()
