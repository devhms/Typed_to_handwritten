import http.server
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import traceback

# Sovereign v8.0 Imports
from notebook_renderer import render_notebook_page, NotebookConfig
from forensic_discriminator import ForensicDiscriminator

PORT = 8000

class SovereignServerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        """Serve static files like index.html, assets, fonts."""
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        """Handle real-time generation and forensic auditing."""
        if self.path == "/generate":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            text = data.get('text', '')
            title = data.get('title', 'Assignment')
            is_preview = data.get('preview', False)
            
            # Parametric Config from Frontend
            config_data = data.get('config', {})
            style = config_data.get('style', 'neat')
            
            try:
                # 1. Update Config Object
                cfg = NotebookConfig(style=style)
                # Override with explicit sliders if provided
                if 'drift_intensity' in config_data:
                    cfg.drift_intensity = float(config_data['drift_intensity'])
                if 'variation_magnitude' in config_data:
                    cfg.variation_magnitude = float(config_data['variation_magnitude'])
                
                # 2. Extract title/body
                lines = text.strip().split('\n')
                title_line = lines[0].strip() if lines else title
                body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else text
                
                # 3. Render (Masterpiece mode only for final high-res)
                output_name = "notebook_preview.png" if is_preview else "notebook_final.png"
                final_path = Path("assignments") / output_name
                
                render_notebook_page(
                    body_text=body,
                    output_path=str(final_path),
                    title=title_line,
                    seed=42,
                    config=cfg,
                    masterpiece=not is_preview
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
                    "is_preview": is_preview
                }
                self._send_json(response)
                
            except Exception as e:
                traceback.print_exc()
                self._send_json({"success": False, "error": str(e)}, status=500)
        else:
            self.send_error(404, "Not Found")

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*') 
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

# Ensure directories exist
Path("assignments").mkdir(parents=True, exist_ok=True)

print(f"\n🚀 SOVEREIGN FORENSIC DASHBOARD SERVER (v8.0)")
print(f"-----------------------------------------------")
print(f"Backend syncing on: http://localhost:{PORT}")
print(f"Listening for parametric generation (THREADED)...")

ThreadingHTTPServer.allow_reuse_address = True
with ThreadingHTTPServer(("", PORT), SovereignServerHandler) as httpd:
    httpd.serve_forever()
