"""Verification agent for Sisyphus cooling project.

Starts the Flask server (in a subprocess inside the current venv), waits until the /api/constants endpoint responds, then runs pytest and prints a coloured summary.

Usage:
    python verification_agent.py
"""

import subprocess
import time
import os
import signal
import sys
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SERVER_CMD = [sys.executable, "app.py"]
SERVER_PORT = 4000
TIMEOUT = 60  # seconds


def wait_for_server(url: str, timeout: int = TIMEOUT):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def main():
    env = os.environ.copy()
    env.setdefault("FLASK_ENV", "production")

    # Spawn server
    server_proc = subprocess.Popen(
        SERVER_CMD,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        print("⏳ Waiting for Flask server to be ready…")
        if not wait_for_server(f"http://localhost:{SERVER_PORT}/api/constants"):
            print("❌ Server did not start within timeout")
            sys.exit(1)
        print("✅ Server is up. Running pytest…")
        pytest_cmd = [sys.executable, "-m", "pytest", "-q", "test_api.py", "20250417.sisyphus_cooling_project/tests"]
        pytest_proc = subprocess.run(pytest_cmd, cwd=str(ROOT), capture_output=True, text=True)
        exit_code = pytest_proc.returncode
        print("—— pytest output ——")
        print(pytest_proc.stdout)
        print(pytest_proc.stderr)
    finally:
        print("🔻 Terminating server…")
        if server_proc.poll() is None:
            server_proc.send_signal(signal.SIGINT)
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        # print captured output
        out, _ = server_proc.communicate()
        print("—— server log ——")
        print(out)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
