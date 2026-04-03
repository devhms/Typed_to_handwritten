import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import cv2
import easyocr


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein_distance(ref, hyp) -> int:
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


def cer_wer(reference: str, hypothesis: str):
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    cer_dist = levenshtein_distance(list(ref_norm), list(hyp_norm))
    wer_dist = levenshtein_distance(ref_norm.split(), hyp_norm.split())

    cer = cer_dist / max(1, len(ref_norm))
    wer = wer_dist / max(1, len(ref_norm.split()))
    return cer, wer


def preprocess_none(image_bgr):
    return image_bgr


def preprocess_upscale(image_bgr, scale: float):
    return cv2.resize(
        image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )


def preprocess_clahe(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def assemble_sorted_text(ocr_boxes):
    items = []
    for item in ocr_boxes:
        if len(item) < 3:
            continue
        bbox, text, conf = item[0], str(item[1]), float(item[2])
        if not text.strip():
            continue

        xs = [float(pt[0]) for pt in bbox]
        ys = [float(pt[1]) for pt in bbox]
        x_min = min(xs)
        y_min = min(ys)
        y_max = max(ys)
        items.append(
            {
                "text": text,
                "x": x_min,
                "y": y_min,
                "h": max(1.0, y_max - y_min),
                "conf": conf,
            }
        )

    if not items:
        return "", 0.0

    median_h = statistics.median([it["h"] for it in items])
    line_threshold = max(8.0, 0.6 * median_h)
    items.sort(key=lambda it: (it["y"], it["x"]))

    lines = []
    for it in items:
        if not lines:
            lines.append([it])
            continue
        last_y = statistics.mean([row["y"] for row in lines[-1]])
        if abs(it["y"] - last_y) <= line_threshold:
            lines[-1].append(it)
        else:
            lines.append([it])

    out_tokens = []
    confs = []
    for line in lines:
        line.sort(key=lambda it: it["x"])
        out_tokens.extend([it["text"] for it in line])
        confs.extend([it["conf"] for it in line])

    return " ".join(out_tokens).strip(), float(statistics.mean(confs)) if confs else 0.0


def read_paragraph(reader, image_bgr, kwargs):
    result = reader.readtext(image_bgr, detail=0, paragraph=True, **kwargs)
    if isinstance(result, list):
        return " ".join(str(x) for x in result).strip(), 0.0
    return str(result or ""), 0.0


def read_sorted_boxes(reader, image_bgr, kwargs):
    boxes = reader.readtext(image_bgr, detail=1, paragraph=False, **kwargs)
    return assemble_sorted_text(boxes)


def evaluate_method(reader, samples, method):
    cer_scores = []
    wer_scores = []
    conf_scores = []

    for row in samples:
        image_path = Path(row["image_path"])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        prep = method["preprocess"](image_bgr)
        text, conf = method["read"](reader, prep, method["kwargs"])
        cer, wer = cer_wer(row["reference_text"], text)

        cer_scores.append(cer)
        wer_scores.append(wer)
        conf_scores.append(conf)

    if not cer_scores:
        return None

    return {
        "method": method["name"],
        "cer_mean": float(statistics.mean(cer_scores)),
        "wer_mean": float(statistics.mean(wer_scores)),
        "conf_mean": float(statistics.mean(conf_scores)) if conf_scores else 0.0,
    }


def evaluate_ensemble(reader, samples):
    cer_scores = []
    wer_scores = []

    for row in samples:
        image_path = Path(row["image_path"])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        raw_txt, raw_conf = read_sorted_boxes(
            reader,
            image_bgr,
            {"decoder": "beamsearch", "beamWidth": 5},
        )
        up15_txt, up15_conf = read_sorted_boxes(
            reader,
            preprocess_upscale(image_bgr, 1.5),
            {"decoder": "beamsearch", "beamWidth": 5},
        )

        chosen = up15_txt if up15_conf > raw_conf + 0.02 else raw_txt
        cer, wer = cer_wer(row["reference_text"], chosen)
        cer_scores.append(cer)
        wer_scores.append(wer)

    return {
        "method": "ensemble_raw_vs_up15_by_conf",
        "cer_mean": float(statistics.mean(cer_scores)),
        "wer_mean": float(statistics.mean(wer_scores)),
        "conf_mean": 0.0,
    }


def build_methods():
    allow = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.-'"
    return [
        {
            "name": "baseline_paragraph",
            "preprocess": preprocess_none,
            "read": read_paragraph,
            "kwargs": {},
        },
        {
            "name": "sorted_greedy",
            "preprocess": preprocess_none,
            "read": read_sorted_boxes,
            "kwargs": {"decoder": "greedy"},
        },
        {
            "name": "sorted_greedy_allowlist",
            "preprocess": preprocess_none,
            "read": read_sorted_boxes,
            "kwargs": {"decoder": "greedy", "allowlist": allow},
        },
        {
            "name": "up15_sorted_greedy",
            "preprocess": lambda im: preprocess_upscale(im, 1.5),
            "read": read_sorted_boxes,
            "kwargs": {"decoder": "greedy"},
        },
        {
            "name": "sorted_beam5",
            "preprocess": preprocess_none,
            "read": read_sorted_boxes,
            "kwargs": {"decoder": "beamsearch", "beamWidth": 5},
        },
        {
            "name": "sorted_beam5_mag15",
            "preprocess": preprocess_none,
            "read": read_sorted_boxes,
            "kwargs": {"decoder": "beamsearch", "beamWidth": 5, "mag_ratio": 1.5},
        },
        {
            "name": "sorted_beam5_mag20",
            "preprocess": preprocess_none,
            "read": read_sorted_boxes,
            "kwargs": {"decoder": "beamsearch", "beamWidth": 5, "mag_ratio": 2.0},
        },
        {
            "name": "sorted_beam5_allowlist",
            "preprocess": preprocess_none,
            "read": read_sorted_boxes,
            "kwargs": {"decoder": "beamsearch", "beamWidth": 5, "allowlist": allow},
        },
        {
            "name": "up15_sorted_beam5",
            "preprocess": lambda im: preprocess_upscale(im, 1.5),
            "read": read_sorted_boxes,
            "kwargs": {"decoder": "beamsearch", "beamWidth": 5},
        },
        {
            "name": "up20_sorted_beam5",
            "preprocess": lambda im: preprocess_upscale(im, 2.0),
            "read": read_sorted_boxes,
            "kwargs": {"decoder": "beamsearch", "beamWidth": 5},
        },
        {
            "name": "clahe_sorted_beam5",
            "preprocess": preprocess_clahe,
            "read": read_sorted_boxes,
            "kwargs": {"decoder": "beamsearch", "beamWidth": 5},
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep OCR techniques on existing benchmark report images."
    )
    parser.add_argument(
        "--report", type=Path, required=True, help="Path to benchmark report JSON."
    )
    parser.add_argument(
        "--profile",
        choices=["quick", "full"],
        default="quick",
        help="Technique set to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples evaluated from the report.",
    )
    args = parser.parse_args()

    if not args.report.exists():
        print(f"Report not found: {args.report}")
        sys.exit(1)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    samples = report.get("per_sample", [])
    if args.max_samples is not None and args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        print("No samples found in report.")
        sys.exit(1)

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    all_methods = build_methods()
    if args.profile == "quick":
        wanted = {
            "baseline_paragraph",
            "sorted_greedy",
            "sorted_greedy_allowlist",
            "up15_sorted_greedy",
        }
        methods = [m for m in all_methods if m["name"] in wanted]
    else:
        methods = all_methods

    results = []
    for method in methods:
        row = evaluate_method(reader, samples, method)
        if row is not None:
            results.append(row)

    if args.profile == "full":
        results.append(evaluate_ensemble(reader, samples))
    results.sort(key=lambda x: (x["cer_mean"], x["wer_mean"]))

    print(f"Technique sweep for {args.report.name}")
    for row in results:
        print(
            f"- {row['method']}: CER={row['cer_mean']:.4f}, WER={row['wer_mean']:.4f}, CONF={row['conf_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
