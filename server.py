import http.server
import socketserver
import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from notebook_renderer import render_notebook_page

PORT = 8000

class SyncBridgeHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        """Serve static files like index.html, assets, fonts."""
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        """Handle real-time generation requests."""
        if self.path == "/generate":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            text = data.get('text', '')
            title = data.get('title', 'Assignment')
            
            try:
                # Extract title from first line, body from rest
                lines = text.strip().split('\n')
                title_line = lines[0].strip() if lines else title
                body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else text
                
                final_path = Path("assignments/notebook_output.png")
                render_notebook_page(
                    body_text=body,
                    output_path=str(final_path),
                    title=title_line,
                    seed=42,
                )
                
                response = {
                    "success": True,
                    "image_url": f"/{final_path.as_posix()}?t={os.path.getmtime(final_path)}"
                }
                self._send_json(response)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({"success": False, "error": str(e)}, status=500)
        else:
            self.send_error(404, "Not Found")

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*') # Allow local dev
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

# Ensure directories exist
Path("assignments").mkdir(parents=True, exist_ok=True)

print(f"\n🚀 AUTHENTIC ASSIGNMENT SYNC BRIDGE")
print(f"-----------------------------------")
print(f"Server starting on: http://localhost:{PORT}")
print(f"Press Ctrl+C to stop.")

with socketserver.TCPServer(("", PORT), SyncBridgeHandler) as httpd:
    httpd.serve_forever()
