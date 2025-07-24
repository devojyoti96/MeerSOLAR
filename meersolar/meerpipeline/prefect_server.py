import argparse
import sys
from meersolar.utils import *

def cli():
    parser = argparse.ArgumentParser(description="Manage a local Prefect server.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")
    # Start
    start_parser = subparsers.add_parser("start", help="Start the Prefect server")
    start_parser.add_argument(
        "--show-config", action="store_true", help="Display Prefect config after startup"
    )
    # Stop
    subparsers.add_parser("stop", help="Stop the Prefect server")

    # Status
    subparsers.add_parser("status", help="Check if the Prefect server is running")

    # Env
    subparsers.add_parser("save_env", help="Save the Prefect environment to a .env file")

    # Config
    subparsers.add_parser("config", help="Print the current Prefect config")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    if args.command == "start":
        start_server(show_config=args.show_config)
    elif args.command == "stop":
        stop_prefect_server()
    elif args.command == "status":
        if prefect_server_status():
            print(f"🟢 Prefect server is running at {SERVER_DASHBOARD}")
        else:
            print("🔴 Prefect server is not running.")
    elif args.command == "save_env":
        save_prefect_env_to_file()
    elif args.command == "config":
        show_prefect_config()
    else:
        parser.print_help()

if __name__=="__main__":
    cli()
