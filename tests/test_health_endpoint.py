import io
import json

from server import app


def _invoke_wsgi(path: str, method: str = "GET", body: bytes = b""):
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = dict(headers)

    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": io.BytesIO(body),
    }

    chunks = app(environ, start_response)
    payload = b"".join(chunks)
    return captured["status"], captured["headers"], payload


def test_health_get_returns_ok_json():
    status, headers, payload = _invoke_wsgi("/health", method="GET")

    assert status.startswith("200")
    assert headers["Content-Type"].startswith("application/json")

    body = json.loads(payload.decode("utf-8"))
    assert body["success"] is True
    assert body["status"] == "ok"
    assert body["service"] == "typed-to-handwritten"


def test_health_head_returns_200_without_body():
    status, headers, payload = _invoke_wsgi("/health", method="HEAD")

    assert status.startswith("200")
    assert headers["Content-Type"].startswith("application/json")
    assert payload == b""
