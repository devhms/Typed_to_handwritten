import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def _check_dependencies() -> None:
    try:
        import PIL  # noqa: F401
        import numpy  # noqa: F401
        import cv2  # noqa: F401
    except ImportError:
        print("[ERROR] Missing core dependencies (Pillow, numpy, opencv-python).")
        print("        Run: python -m pip install -r requirements.txt")
        sys.exit(1)


def main():
    print("\nSTARTING AUTHENTIC ASSIGNMENT GENERATOR")
    print("------------------------------------------")

    _check_dependencies()

    server_script = Path(__file__).resolve().parent / "server.py"
    if not server_script.exists():
        print(f"[ERROR] server.py not found at: {server_script}")
        sys.exit(1)

    print("[1/2] Launching Sync Bridge Server...")
    server_process = subprocess.Popen([sys.executable, str(server_script)])

    time.sleep(2)

    print("[2/2] Opening Web Interface...")
    webbrowser.open("http://localhost:8000")

    print("\nSYSTEM ONLINE")
    print("------------------------------------------")
    print("[INFO] Keep this terminal open while using the app.")
    print("[INFO] Press Ctrl+C in this terminal to shut down.")

    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\n\nSHUTTING DOWN...")
        server_process.terminate()
        print("Goodbye!")

if __name__ == "__main__":
    main()
