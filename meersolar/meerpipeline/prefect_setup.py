import os
import socket
import subprocess
import time
import sys

# Configurable server host and port
server_host = "127.0.0.1"
server_port = 4200
server_url = f"http://{server_host}:{server_port}"

def is_prefect_server_running(host=server_host, port=server_port):
    """
    Check if a Prefect server is running on the given host and port.
    """
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False

def start_prefect_server():
    """
    Start the Prefect server in the background (non-blocking).
    Returns True if successfully started, False otherwise.
    """
    try:
        print("🚀 Starting MeerSOLAR Prefect server...")
        proc = subprocess.Popen(
            ["prefect", "server", "start"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"✅ MeerSOLAR Prefect server started with PID {proc.pid}")

        for _ in range(10):  # wait up to 10 seconds
            if is_prefect_server_running():
                print(f"🌐 MeerSOLAR Prefect server is now running at {server_url}")
                return True
            time.sleep(1)

        print(f"⚠️ Warning: Prefect server may not be fully ready yet. Check {server_url}")
        return is_prefect_server_running()

    except Exception as e:
        print(f"❌ Failed to start Prefect server: {e}")
        return False

def setup_prefect_server_mode():
    """
    Setup logic: if server isn't running, try to start it.
    Returns 0 on success, 1 on failure.
    """
    if not is_prefect_server_running():
        success = start_prefect_server()
        if not success:
            print("❌ Could not start Prefect server. Exiting...")
            return 1
    else:
        print(f"🟢 MeerSOLAR Prefect server already running at {server_url}")

    # Ensure Prefect is in server mode
    os.environ["PREFECT_API_MODE"] = "server"
    return 0

if __name__ == "__main__":
    sys.exit(setup_prefect_server_mode())

