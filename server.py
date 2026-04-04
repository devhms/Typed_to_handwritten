import base64
import http.server
import json
import logging
import mimetypes
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from forensic_discriminator import ForensicDiscriminator
from notebook_renderer import NotebookConfig, render_notebook_page

PORT = 8000

MAX_REQUEST_BYTES = 1_000_000
LOGGER = logging.getLogger("handwritten.server")
STATIC_ROOT = Path(__file__).resolve().parent


def _as_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _parse_document_blocks(text: str) -> list[dict]:
    blocks: list[dict] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            blocks.append({"type": "header", "content": stripped[2:], "level": 1})
        elif stripped.startswith("## "):
            blocks.append({"type": "header", "content": stripped[3:], "level": 2})
        elif not stripped:
            blocks.append({"type": "spacer", "content": ""})
        else:
            blocks.append({"type": "paragraph", "content": stripped})
    return blocks


def _json_error(message: str) -> dict:
    return {"success": False, "error": message}


def _assignments_dir() -> Path:
    if os.getenv("VERCEL"):
        return Path(os.getenv("TMPDIR", "/tmp")) / "assignments"
    return Path("assignments")


def _handle_generation(data: dict) -> tuple[dict, int]:
    text = str(data.get("text", ""))
    is_preview = bool(data.get("preview", False))

    config_data = data.get("config", {})
    if not isinstance(config_data, dict):
        config_data = {}
    style = str(config_data.get("style", "neat"))

    blocks = _parse_document_blocks(text)

    try:
        cfg = NotebookConfig(style=style)
        if "drift_intensity" in config_data:
            cfg.drift_intensity = _as_float(config_data.get("drift_intensity"), cfg.drift_intensity)
        if "variation_magnitude" in config_data:
            cfg.variation_magnitude = _as_float(
                config_data.get("variation_magnitude"), cfg.variation_magnitude
            )
        if not is_preview:
            cfg.masterpiece_overlay_alpha = _as_float(
                config_data.get("masterpiece_overlay_alpha", 0.0),
                cfg.masterpiece_overlay_alpha,
            )
            cfg.masterpiece_preview_blend_alpha = _as_float(
                config_data.get("masterpiece_preview_blend_alpha", 0.0),
                cfg.masterpiece_preview_blend_alpha,
            )

        output_name = "notebook_preview.png" if is_preview else "notebook_final.png"
        assignments_dir = _assignments_dir()
        assignments_dir.mkdir(parents=True, exist_ok=True)
        final_path = assignments_dir / output_name

        render_notebook_page(
            document_blocks=blocks,
            output_path=str(final_path),
            seed=42,
            config=cfg,
            masterpiece=not is_preview,
        )

        audit_results = None
        if not is_preview:
            judge = ForensicDiscriminator()
            audit_results = judge.score_authenticity(final_path)

        response = {
            "success": True,
            "image_url": f"/assignments/{output_name}?t={os.path.getmtime(final_path)}",
            "audit": audit_results,
            "is_preview": is_preview,
        }

        if os.getenv("VERCEL"):
            encoded = base64.b64encode(final_path.read_bytes()).decode("ascii")
            response["image_data_url"] = f"data:image/png;base64,{encoded}"

        return response, 200
    except Exception as exc:
        LOGGER.exception("Generation failed")
        return {"success": False, "error": str(exc)}, 500


_STATUS_TEXT = {
    200: "OK",
    400: "Bad Request",
    404: "Not Found",
    405: "Method Not Allowed",
    413: "Payload Too Large",
    500: "Internal Server Error",
}


def _wsgi_response(start_response, status: int, body: bytes, content_type: str) -> list[bytes]:
    start_response(
        f"{status} {_STATUS_TEXT.get(status, 'OK')}",
        [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
            ("Access-Control-Allow-Origin", "*"),
        ],
    )
    return [body]


def _wsgi_json(start_response, payload: dict, status: int) -> list[bytes]:
    body = json.dumps(payload).encode("utf-8")
    return _wsgi_response(start_response, status, body, "application/json; charset=utf-8")


def _resolve_static_file(path_info: str) -> Path | None:
    request_path = unquote(path_info or "/")
    relative_path = request_path.lstrip("/") or "index.html"
    candidate = (STATIC_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(STATIC_ROOT)
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    return None


def _resolve_assignment_file(path_info: str) -> Path | None:
    request_path = unquote(path_info or "/")
    if not request_path.startswith("/assignments/"):
        return None

    rel = request_path[len("/assignments/") :]
    if not rel:
        return None

    assignments_root = _assignments_dir().resolve()
    candidate = (assignments_root / rel).resolve()
    try:
        candidate.relative_to(assignments_root)
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    return None


def _serve_file_wsgi(start_response, file_path: Path) -> list[bytes]:
    content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    return _wsgi_response(start_response, 200, file_path.read_bytes(), content_type)


def app(environ, start_response):
    method = str(environ.get("REQUEST_METHOD", "GET")).upper()
    path_info = str(environ.get("PATH_INFO", "/") or "/")

    if method == "OPTIONS":
        return _wsgi_response(start_response, 200, b"", "text/plain; charset=utf-8")

    if method == "POST" and path_info == "/generate":
        try:
            content_length = int(environ.get("CONTENT_LENGTH") or "0")
        except ValueError:
            content_length = 0

        if content_length <= 0:
            return _wsgi_json(start_response, _json_error("Empty request body"), 400)
        if content_length > MAX_REQUEST_BYTES:
            return _wsgi_json(start_response, _json_error("Request too large"), 413)

        payload = environ["wsgi.input"].read(content_length)
        try:
            data = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:
            return _wsgi_json(start_response, _json_error("Invalid JSON payload"), 400)

        response, status = _handle_generation(data)
        return _wsgi_json(start_response, response, status)

    if method in {"GET", "HEAD"}:
        normalized = "/index.html" if path_info in {"", "/"} else path_info
        file_path = _resolve_assignment_file(normalized) or _resolve_static_file(normalized)
        if file_path is None:
            return _wsgi_json(start_response, _json_error("Not Found"), 404)

        if method == "HEAD":
            content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            return _wsgi_response(start_response, 200, b"", content_type)

        return _serve_file_wsgi(start_response, file_path)

    return _wsgi_json(start_response, _json_error("Method Not Allowed"), 405)


class SovereignServerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        """Serve static files like index.html, assets, fonts."""
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        """Handle real-time generation and forensic auditing."""
        if self.path == "/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self._send_json(_json_error("Empty request body"), status=400)
                return
            if content_length > MAX_REQUEST_BYTES:
                self._send_json(_json_error("Request too large"), status=413)
                return

            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(_json_error("Invalid JSON payload"), status=400)
                return

            response, status = _handle_generation(data)
            self._send_json(response, status=status)
        else:
            self.send_error(404, "Not Found")

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


def run_dev_server(port: int = PORT) -> None:
    _assignments_dir().mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("\nSOVEREIGN FORENSIC DASHBOARD SERVER (v8.0)")
    print("-----------------------------------------------")
    print(f"Backend syncing on: http://localhost:{port}")
    print("Listening for parametric generation (THREADED)...")

    ThreadingHTTPServer.allow_reuse_address = True
    with ThreadingHTTPServer(("", port), SovereignServerHandler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    run_dev_server()
