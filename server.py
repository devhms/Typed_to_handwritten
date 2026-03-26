import http.server
import socketserver
import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import phase2_ink_synthesis
import phase3_degradation

PORT = 8000

class SyncBridgeHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        """Serve static files like index.html, assets, fonts."""
        # Simple routing for SPA-like feel
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
            fast_mode = data.get('fast', True) # Skip Augraphy for real-time smoothness
            
            try:
                # 1. Write to temp text file for consistency
                input_file = Path("my_assignment.txt")
                input_file.write_text(text, encoding="utf-8")
                
                # 2. Render Page (Raw Handwriting)
                raw_path = Path("assignments/v_sync_raw.png")
                phase2_ink_synthesis.render_page(
                    title=title,
                    body_text=text,
                    output_path=raw_path
                )
                
                final_path = raw_path
                # 3. Apply Quality Degradation (Optional)
                if not fast_mode:
                    final_photo_path = Path("assignments/v_sync_photo.png")
                    phase3_degradation.degrade_image(
                        input_path=raw_path,
                        output_path=final_photo_path,
                        severity="medium"
                    )
                    final_path = final_photo_path
                
                # Success response with timestamp to bust cache
                response = {
                    "success": True,
                    "image_url": f"/{final_path.as_posix()}?t={os.path.getmtime(final_path)}"
                }
                self._send_json(response)
                
            except Exception as e:
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
