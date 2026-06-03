import subprocess
import time
import sys
import socket


def is_port_in_use(port):
    """Check if a port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def run_app():
    # Check if ports are already in use
    if is_port_in_use(8000):
        print("ERROR: Port 8000 is already in use. Stop the existing backend first.")
        print("  Run:  taskkill /F /IM python.exe   (Windows)")
        sys.exit(1)
    if is_port_in_use(8501):
        print("ERROR: Port 8501 is already in use. Stop the existing frontend first.")
        print("  Run:  taskkill /F /IM python.exe   (Windows)")
        sys.exit(1)

    # Start Backend — inherit stdout so pipe buffer never fills up
    print("Starting FastAPI Backend...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
    )

    # Wait for backend to be ready (poll the port instead of a blind sleep)
    print("Waiting for backend to initialize...")
    for _ in range(30):          # up to 15 seconds
        time.sleep(0.5)
        if is_port_in_use(8000):
            print("Backend is ready.")
            break
    else:
        print("ERROR: Backend did not start in time.")
        backend_process.terminate()
        sys.exit(1)

    # Start Frontend — inherit stdout so pipe buffer never fills up
    print("Starting Streamlit Frontend...")
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend.py", "--server.port=8501"],
    )

    print("\n" + "=" * 50)
    print("Drug Prediction System is running!")
    print("Backend API : http://localhost:8000")
    print("Frontend UI : http://localhost:8501")
    print("=" * 50)
    print("Press Ctrl+C to stop both servers.\n")

    try:
        while True:
            if backend_process.poll() is not None:
                print("\nBackend stopped unexpectedly.")
                frontend_process.terminate()
                break
            if frontend_process.poll() is not None:
                print("\nFrontend stopped unexpectedly.")
                backend_process.terminate()
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    finally:
        backend_process.terminate()
        frontend_process.terminate()
        # Give them a moment to exit cleanly
        backend_process.wait()
        frontend_process.wait()
        print("Done.")


if __name__ == "__main__":
    run_app()
