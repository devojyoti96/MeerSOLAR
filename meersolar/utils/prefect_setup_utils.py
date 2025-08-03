import os
import types
import subprocess
import time
import socket
import signal
import argparse
import toml
from pathlib import Path
from dotenv import load_dotenv
from meersolar.utils.basic_utils import *


# === CONFIG ===
def prefect_config():
    cachedir = get_cachedir()
    PREFECT_HOME = f"{cachedir}/prefect_home"
    os.makedirs(PREFECT_HOME, exist_ok=True)

    DB_URL = f"sqlite+aiosqlite:///{PREFECT_HOME}/prefect.db"
    LOG_FILE = os.path.join(PREFECT_HOME, "server.log")
    profile_path = os.path.join(PREFECT_HOME, "profiles.toml")
    memo_path = os.path.join(PREFECT_HOME, "memo_store.toml")
    storage = os.path.join(PREFECT_HOME, "storage")
    os.makedirs(storage, exist_ok=True)
    ENV_FILE = os.path.join(cachedir, "meersolar_prefect.env")
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = "4250"
    SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/api"
    SERVER_DASHBOARD = f"http://{SERVER_HOST}:{SERVER_PORT}/dashboard"
    profile_name = "solarpipe"
    pid_file = os.path.join(PREFECT_HOME, "server.pid")
    logging_path = os.path.join(PREFECT_HOME, "logging.yml")

    return {
        "CACHEDIR": cachedir,
        "PREFECT_HOME": PREFECT_HOME,
        "DB_URL": DB_URL,
        "LOG_FILE": LOG_FILE,
        "PROFILE_PATH": profile_path,
        "MEMO_PATH": memo_path,
        "STORAGE": storage,
        "ENV_FILE": ENV_FILE,
        "SERVER_HOST": SERVER_HOST,
        "SERVER_PORT": SERVER_PORT,
        "SERVER_URL": SERVER_URL,
        "SERVER_DASHBOARD": SERVER_DASHBOARD,
        "PROFILE_NAME": profile_name,
        "PID_FILE": pid_file,
        "LOGGING_PATH": logging_path,
    }


def write_prefect_profile():
    """
    Save prefect profile
    """
    config = prefect_config()
    # Load existing TOML config or start new
    profile_path = config["PROFILE_PATH"]
    if os.path.exists(profile_path):
        data = toml.load(profile_path)
    else:
        data = {}
    # Set active profile
    profile_name = config["PROFILE_NAME"]
    data["active"] = profile_name
    # Set config under [profiles.<profile_name>]
    if "profiles" not in data:
        data["profiles"] = {}
    data["profiles"][profile_name] = {
        "PREFECT_API_URL": config["SERVER_URL"],
        "PREFECT_HOME": config["PREFECT_HOME"],
        "PREFECT_API_DATABASE_CONNECTION_URL": config["DB_URL"],
    }
    with open(profile_path, "w") as f:
        toml.dump(data, f)
    print(f"✅ Prefect profile '{profile_name}' written to {profile_path}")


def prefect_server_status():
    """
    Get prefect server status
    """
    config = prefect_config()
    try:
        with socket.create_connection(
            (config["SERVER_HOST"], config["SERVER_PORT"]), timeout=2
        ):
            return True
    except OSError:
        return False


def get_prefect_env():
    """
    Get environment variables of prefect
    """
    config = prefect_config()
    env = os.environ.copy()
    env["PREFECT_HOME"] = config["PREFECT_HOME"]
    env["PREFECT_API_MODE"] = "server"
    env["PREFECT_API_DATABASE_CONNECTION_URL"] = config["DB_URL"]
    env["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "false"
    env["PREFECT_API_URL"] = config["SERVER_URL"]
    env["PREFECT_PROFILE"] = config["PROFILE_NAME"]
    env["PREFECT_PROFILES_PATH"] = config["PROFILE_PATH"]
    env["PREFECT_LOCAL_STORAGE_PATH"] = config["STORAGE"]
    env["PREFECT_LOGGING_SETTINGS_PATH"] = config["LOGGING_PATH"]
    env["PREFECT_MEMO_STORE_PATH"] = config["MEMO_PATH"]
    return env


def save_prefect_env_to_file():
    """
    Save current Prefect server env config to a .env file for reuse.
    """
    config = prefect_config()
    cachedir = config["CACHEDIR"]
    with open(config["ENV_FILE"], "w") as f:
        f.write(f"PREFECT_HOME={config['PREFECT_HOME']}\n")
        f.write("PREFECT_API_MODE=server\n")
        f.write(f"PREFECT_API_DATABASE_CONNECTION_URL={config['DB_URL']}\n")
        f.write("PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=false\n")
        f.write(f"PREFECT_API_URL={config['SERVER_URL']}\n")
        f.write(f"PREFECT_PROFILE={config['PROFILE_NAME']}\n")
        f.write(f"PREFECT_PROFILES_PATH={config['PROFILE_PATH']}\n")
        f.write(f"PREFECT_LOCAL_STORAGE_PATH={config['STORAGE']}\n")
        f.write(f"PREFECT_LOGGING_SETTINGS_PATH={config['LOGGING_PATH']}\n")
        f.write(f"PREFECT_MEMO_STORE_PATH={config['MEMO_PATH']}\n")
    print(f"📄 Saved Prefect server environment to {config['ENV_FILE']}")
    if os.path.exists(f"{cachedir}/prefect.dashboard") is not True:
        with open(f"{cachedir}/prefect.dashboard", "w") as f:
            f.write(f"{config['SERVER_DASHBOARD']}")
    write_prefect_profile()


def start_server(show_config=False):
    """
    Start prefect server if it is not running
    """
    config = prefect_config()
    cachedir = config["CACHEDIR"]
    if prefect_server_status():
        print(f"🟢 Prefect server is already running at {config['SERVER_DASHBOARD']}")
        if os.path.exists(f"{cachedir}/prefect.dashboard") is not True:
            with open(f"{cachedir}/prefect.dashboard", "w") as f:
                f.write(f"{config['SERVER_DASHBOARD']}")
        if show_config:
            show_prefect_config()
        os.makedirs(config["PREFECT_HOME"], exist_ok=True)
        save_prefect_env_to_file()
        return 0
    print("🚀 Starting Prefect server...")
    pid_file = config["PID_FILE"]
    if os.path.exists(pid_file):
        stop_prefect_server()
    with open(config["LOG_FILE"], "w") as f:
        server_proc = subprocess.Popen(
            [
                "prefect",
                "server",
                "start",
                "--host",
                config["SERVER_HOST"],
                "--port",
                config["SERVER_PORT"],
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=get_prefect_env(),
        )
    server_started = False
    for _ in range(30):  # wait up to 30s for the server to respond
        if prefect_server_status():
            if show_config:
                show_prefect_config()
            server_started = True
            break
        time.sleep(1)
    if server_started:
        with open(pid_file, "w") as pf:
            pf.write(str(server_proc.pid))
        os.makedirs(config["PREFECT_HOME"], exist_ok=True)
        save_prefect_env_to_file()
        print(f"✅ Prefect server is now running at {config['SERVER_DASHBOARD']}")
        if os.path.exists(f"{cachedir}/prefect.dashboard") is not True:
            with open(f"{cachedir}/prefect.dashboard", "w") as f:
                f.write(f"{config['SERVER_DASHBOARD']}")
        return 0
    else:
        print(f"⚠️ Server did not respond in time. Check logs at {config['LOG_FILE']}")
        return 1


def stop_prefect_server():
    """
    Stop prefect server running in the current installation
    Note: it will only stop prefect server which is running from the current installation
    For this pipeline, default port (4250) is kept seperate from default prefect port 4200.
    """
    config = prefect_config()
    pid_file = config["PID_FILE"]
    cachedir = config["CACHEDIR"]
    try:
        if not os.path.exists(pid_file):
            print("⚠️ No PID file found. Cannot stop Prefect server.")
        else:
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
    finally:
        os.remove(config["ENV_FILE"])
        os.system(f"touch {config['ENV_FILE']}")
        os.remove(f"{cachedir}/prefect.dashboard")


def show_prefect_config():
    """
    Print the effective Prefect config in this environment.
    """
    config = prefect_config()
    load_dotenv(dotenv_path=config["ENV_FILE"], override=False)
    env = os.environ.copy()
    print("🔍 Prefect config in current environment ...")
    subprocess.run(["prefect", "config", "view"], env=env)


# Exposing only functions
__all__ = [
    name
    for name, obj in globals().items()
    if isinstance(obj, types.FunctionType) and obj.__module__ == __name__
]
