import os
import types
import subprocess
import time
import socket
import signal
import argparse
import toml
from pathlib import Path
from meersolar.utils.basic_utils import *

# === CONFIG ===
cachedir = get_cachedir()
PREFECT_HOME = f"{cachedir}/prefect_home"
os.makedirs(PREFECT_HOME,exist_ok=True)
DB_URL = f"sqlite+aiosqlite:///{PREFECT_HOME}/prefect.db"
LOG_FILE = os.path.join(PREFECT_HOME, "server.log")
profile_path=os.path.join(PREFECT_HOME, "profiles.toml")
memo_path=os.path.join(PREFECT_HOME, "memo_store.toml")
storage=os.path.join(PREFECT_HOME, "storage")
os.makedirs(storage,exist_ok=True)
ENV_FILE = os.path.join(cachedir, "meersolar_prefect.env")
SERVER_HOST = "0.0.0.0"
SERVER_PORT = "4200"
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/api"
SERVER_DASHBOARD = f"http://{SERVER_HOST}:{SERVER_PORT}/dashboard"
profile_name="solarpipe"
pid_file = os.path.join(PREFECT_HOME, "server.pid")
logging_path=os.path.join(PREFECT_HOME,"logging.yml")


def write_prefect_profile():
    """
    Save prefect profile
    """
    # Load existing TOML config or start new
    if os.path.exists(profile_path):
        data = toml.load(profile_path)
    else:
        data = {}
    # Set active profile
    data["active"] = profile_name
    # Set config under [profiles.<profile_name>]
    if "profiles" not in data:
        data["profiles"] = {}
    data["profiles"][profile_name] = {
        "PREFECT_API_URL": SERVER_URL,
        "PREFECT_HOME": PREFECT_HOME,
        "PREFECT_API_DATABASE_CONNECTION_URL": DB_URL,
    }
    with open(profile_path, "w") as f:
        toml.dump(data, f)
    print(f"✅ Prefect profile '{profile_name}' written to {profile_path}")
    

def prefect_server_status():
    """
    Get prefect server status
    """
    try:
        with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=2):
            return True
    except OSError:
        return False


def get_prefect_env():
    """
    Get environment variables of prefect 
    """
    env = os.environ.copy()
    env["PREFECT_HOME"] = PREFECT_HOME
    env["PREFECT_API_MODE"] = "server"
    env["PREFECT_API_DATABASE_CONNECTION_URL"] = DB_URL
    env["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "false"
    env["PREFECT_API_URL"]=SERVER_URL
    env["PREFECT_PROFILE"] = profile_name
    env["PREFECT_PROFILES_PATH"]=profile_path
    env["PREFECT_LOCAL_STORAGE_PATH"]=storage
    env["PREFECT_LOGGING_SETTINGS_PATH"]=logging_path
    env["PREFECT_MEMO_STORE_PATH"]=memo_path
    return env


def save_prefect_env_to_file():
    """
    Save current Prefect server env config to a .env file for reuse.
    """
    with open(ENV_FILE, "w") as f:
        f.write(f"PREFECT_HOME={PREFECT_HOME}\n")
        f.write("PREFECT_API_MODE=server\n")
        f.write(f"PREFECT_API_DATABASE_CONNECTION_URL={DB_URL}\n")
        f.write("PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=false\n")
        f.write(f"PREFECT_API_URL={SERVER_URL}\n")
        f.write(f"PREFECT_PROFILE={profile_name}\n")
        f.write(f"PREFECT_PROFILES_PATH={profile_path}\n")
        f.write(f"PREFECT_LOCAL_STORAGE_PATH={storage}\n")
        f.write(f"PREFECT_LOGGING_SETTINGS_PATH={logging_path}\n")
        f.write(f"PREFECT_MEMO_STORE_PATH={memo_path}\n")
    print(f"📄 Saved Prefect server environment to {ENV_FILE}")
    if os.path.exists(f"{cachedir}/prefect.dashboard") is not True:
        with open(f"{cachedir}/prefect.dashboard","w") as f:
            f.write(f"{SERVER_DASHBOARD}")
    write_prefect_profile()


def start_server(show_config=False):
    """
    Start prefect server if it is not running
    """
    if prefect_server_status():
        print(f"🟢 Prefect server is already running at {SERVER_DASHBOARD}")
        if os.path.exists(f"{cachedir}/prefect.dashboard") is not True:
            with open(f"{cachedir}/prefect.dashboard","w") as f:
                f.write(f"{SERVER_DASHBOARD}")
        if show_config:
            show_prefect_config()
        os.makedirs(PREFECT_HOME, exist_ok=True)
        save_prefect_env_to_file()
        return 0
    print("🚀 Starting Prefect server...")
    if os.path.exists(pid_file):
        stop_prefect_server()
    with open(LOG_FILE, "w") as f:
        server_proc = subprocess.Popen(
            ["prefect", "server", "start", "--host",SERVER_HOST,"--port",SERVER_PORT],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=get_prefect_env(),
        )
    server_started=False
    for _ in range(30):  # wait up to 30s for the server to respond
        if prefect_server_status():
            if show_config:
                show_prefect_config()
            server_started=True
            break
        time.sleep(1)
    if server_started:
        with open(pid_file, "w") as pf:
            pf.write(str(server_proc.pid))
        os.makedirs(PREFECT_HOME, exist_ok=True)
        save_prefect_env_to_file()
        print(f"✅ Prefect server is now running at {SERVER_DASHBOARD}")
        if os.path.exists(f"{cachedir}/prefect.dashboard") is not True:
            with open(f"{cachedir}/prefect.dashboard","w") as f:
                f.write(f"{SERVER_DASHBOARD}")
        return 0
    else:
        print(f"⚠️ Server did not respond in time. Check logs at {LOG_FILE}")
        return 1


def stop_prefect_server():
    """
    Stop prefect server running in the current installation
    Note: it will only stop prefect server which is running from the current installation
    For this pipeline, default port (4250) is kept seperate from default prefect port 4200.
    """
    if not os.path.exists(pid_file):
        print("⚠️ No PID file found. Cannot stop Prefect server.")
        return
    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        print(f"🛑 Stopping Prefect server with PID {pid} ...")
        os.kill(pid, signal.SIGTERM)
        os.remove(pid_file)
        print("✅ Server stopped and PID file removed.")
    except ProcessLookupError:
        print(f"⚠️ No such process with PID {pid}. Removing stale PID file.")
        os.remove(pid_file)
    except Exception as e:
        print(f"❌ Error stopping server: {e}")


def show_prefect_config():
    """
    Print the effective Prefect config in this environment.
    """
    print("🔍 Prefect config in current environment:")
    subprocess.run(["prefect", "config", "view"], env=get_prefect_env())

# Exposing only functions
__all__ = [
    name
    for name, obj in globals().items()
    if isinstance(obj, types.FunctionType) and obj.__module__ == __name__
]
