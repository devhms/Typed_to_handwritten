import argparse
import json
import sys
import urllib.error
import urllib.request


def _request(url: str, method: str = "GET", data: bytes | None = None, timeout: int = 30):
    req = urllib.request.Request(url=url, method=method, data=data)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def _assert_json(status: int, body: str, context: str) -> dict:
    if status < 200 or status >= 300:
        raise RuntimeError(f"{context} returned HTTP {status}: {body[:300]}")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{context} returned non-JSON response: {body[:300]}") from exc


def run_smoke(base_url: str) -> None:
    base = base_url.rstrip("/")

    # 1) health check
    health_status, health_body = _request(f"{base}/health", method="GET")
    health = _assert_json(health_status, health_body, "GET /health")
    if not health.get("success") or health.get("status") != "ok":
        raise RuntimeError(f"GET /health failed semantic validation: {health}")

    # 2) generate preview
    payload = {
        "text": "Smoke test line for Vercel deployment verification.",
        "preview": True,
        "config": {
            "style": "neat",
            "drift_intensity": 0.6,
            "variation_magnitude": 0.015,
        },
    }
    raw = json.dumps(payload).encode("utf-8")
    gen_status, gen_body = _request(f"{base}/generate", method="POST", data=raw, timeout=120)
    generated = _assert_json(gen_status, gen_body, "POST /generate")

    if not generated.get("success"):
        raise RuntimeError(f"POST /generate reported failure: {generated}")

    if not (generated.get("image_data_url") or generated.get("image_url")):
        raise RuntimeError(f"POST /generate missing image output fields: {generated}")

    print("[SMOKE] PASS")
    print(f"[SMOKE] base_url={base}")
    print(f"[SMOKE] runtime={health.get('runtime')}")
    print(f"[SMOKE] version={health.get('version')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-deploy smoke test for Typed_to_handwritten Vercel deployment."
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Deployment base URL, e.g. https://typed-to-handwritten.vercel.app",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_smoke(args.base_url)
        return 0
    except (RuntimeError, urllib.error.URLError) as exc:
        print(f"[SMOKE] FAIL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
