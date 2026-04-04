import http.server
import json
import logging
import os
from http.server import ThreadingHTTPServer
from pathlib import Path

from forensic_discriminator import ForensicDiscriminator
from notebook_renderer import NotebookConfig, render_notebook_page

PORT = 8000

MAX_REQUEST_BYTES = 1_000_000
LOGGER = logging.getLogger("handwritten.server")


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

            text = str(data.get("text", ""))
            is_preview = bool(data.get("preview", False))

            # Parametric Config from Frontend
            config_data = data.get("config", {})
            if not isinstance(config_data, dict):
                config_data = {}
            style = str(config_data.get("style", "neat"))

            # ─────────────────────────────────────────────────────────────────────────────
            # UI-UX PRO MAX: BLOCK-BASED DOCUMENT PARSER (v8.5)
            # ─────────────────────────────────────────────────────────────────────────────
            # Instead of a simple split, we parse the buffer into a structured layout model.

            blocks = _parse_document_blocks(text)

            try:
                # 1. Update Config Object with Sliders
                cfg = NotebookConfig(style=style)
                if "drift_intensity" in config_data:
                    cfg.drift_intensity = _as_float(
                        config_data.get("drift_intensity"), cfg.drift_intensity
                    )
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

                # 2. Rendering Selection (Preview vs Masterpiece)
                output_name = (
                    "notebook_preview.png" if is_preview else "notebook_final.png"
                )  # Context-aware filenames
                final_path = (
                    Path("assignments") / output_name
                )  # Ensure high-resolution artifacts are stored centrally

                # Perform the Synthesis with the new Document Model
                render_notebook_page(
                    document_blocks=blocks,  # Pass the structured block model instead of flat text (v8.5 Refactor)
                    output_path=str(final_path),  # Final write location
                    seed=42,  # Static seed for deterministic forensic evaluation
                    config=cfg,  # The kinematic profile selected via the Bento UI
                    masterpiece=not is_preview,  # Elevate to PBI-Physics mode only for high-res exports
                )

                # 4. Forensic Audit (Skip or use lightweight mode for previews)
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
                self._send_json(response)

            except Exception as e:
                LOGGER.exception("Generation failed")
                self._send_json({"success": False, "error": str(e)}, status=500)
        else:
            self.send_error(404, "Not Found")

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


# Ensure directories exist
Path("assignments").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

print("\nSOVEREIGN FORENSIC DASHBOARD SERVER (v8.0)")
print("-----------------------------------------------")
print(f"Backend syncing on: http://localhost:{PORT}")
print("Listening for parametric generation (THREADED)...")

ThreadingHTTPServer.allow_reuse_address = True
with ThreadingHTTPServer(("", PORT), SovereignServerHandler) as httpd:
    httpd.serve_forever()
