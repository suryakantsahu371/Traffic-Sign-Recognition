# run.py
import subprocess
import webbrowser
import os
import time
from threading import Thread
import sys

# =========================
# CONFIG PATHS (EDIT ONLY IF PROJECT MOVES)
# =========================
PROJECT_ROOT = r"C:\Users\CYBORG\Downloads\traffic_sign_project\traffic_sign_project"

BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
BACKEND_APP = os.path.join(BACKEND_DIR, "app.py")

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")  # must contain index.html
FRONTEND_PORT = 5500

# ⚠️ IMPORTANT: venv Python (this fixes Flask error)
VENV_PYTHON = r"D:\Environment\myenv\Scripts\python.exe"

# =========================
# START BACKEND (Flask)
# =========================
def run_backend():
    print("🚀 Starting Flask backend...")
    subprocess.Popen(
        [VENV_PYTHON, "app.py"],
        cwd=BACKEND_DIR
    )

# =========================
# START FRONTEND (HTTP SERVER)
# =========================
def run_frontend():
    print("🌐 Starting frontend HTTP server...")
    subprocess.Popen(
        [sys.executable, "-m", "http.server", str(FRONTEND_PORT)],
        cwd=FRONTEND_DIR
    )

# =========================
# OPEN BROWSER
# =========================
def open_browser():
    time.sleep(4)  # wait for servers
    url = f"http://127.0.0.1:{FRONTEND_PORT}/"
    print(f"🌟 Opening browser at {url}")
    webbrowser.open(url)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    Thread(target=run_backend, daemon=True).start()
    Thread(target=run_frontend, daemon=True).start()
    Thread(target=open_browser, daemon=True).start()

    # Keep script alive
    while True:
        time.sleep(1)
