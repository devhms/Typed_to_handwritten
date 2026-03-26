import subprocess
import webbrowser
import time
import os
import sys
from pathlib import Path

def main():
    print("\n🌟 STARTING AUTHENTIC ASSIGNMENT GENERATOR")
    print("------------------------------------------")
    
    # Check dependencies
    try:
        import PIL
        import numpy
        import cv2
    except ImportError:
        print("[ERROR] Missing core dependencies (Pillow, numpy, opencv-python).")
        print("        Please run: pip install pillow numpy opencv-python")
        sys.exit(1)

    # Start the Sync Bridge Server
    print("[1/2] Launching Sync Bridge Server...")
    server_process = subprocess.Popen([sys.executable, "server.py"])
    
    # Wait for server to warm up
    time.sleep(2)
    
    # Open the UI in the default browser
    print("[2/2] Opening Premium Web Interface...")
    webbrowser.open("http://localhost:8000")
    
    print("\n✅ SYSTEM ONLINE")
    print("------------------------------------------")
    print("Type in the browser to see the 'Human Error' model")
    print("synchronize your handwriting on the ruling lines.")
    print("\n[INFO] Keep this terminal open while using the app.")
    print("[INFO] Press Ctrl+C in this terminal to shut down.")
    
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 SHUTTING DOWN...")
        server_process.terminate()
        print("Goodbye!")

if __name__ == "__main__":
    main()
