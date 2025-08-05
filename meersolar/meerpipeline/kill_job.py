import psutil
import numpy as np
import argparse
import time
import sys
import os
import signal
from meersolar.utils import get_cachedir, drop_cache


def terminate_process_and_children(pid, grace_period=3.0):
    """
    Try to gracefully terminate a process tree, then force kill if needed.
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except Exception:
                pass
        parent.terminate()
        gone, alive = psutil.wait_procs([parent] + children, timeout=grace_period)
        for p in alive:
            try:
                p.kill()
            except Exception:
                pass
    except (psutil.NoSuchProcess, ProcessLookupError):
        pass


def force_kill_pids_with_children(pids, max_tries=10, wait_time=0.5):
    """
    Repeatedly try to terminate and then kill all PIDs (and their children) until none remain.
    """
    for attempt in range(max_tries):
        remaining = []
        for pid in np.atleast_1d(pids):
            try:
                terminate_process_and_children(int(pid))
            except Exception:
                remaining.append(pid)

        time.sleep(wait_time)
        remaining = [pid for pid in np.atleast_1d(pids) if psutil.pid_exists(int(pid))]

        if not remaining:
            break
        else:
            pids = remaining


def kill_meerjob():
    """
    Gracefully terminate all processes related to a MeerSOLAR job.
    """
    parser = argparse.ArgumentParser(description="Kill MeerSOLAR Job")
    parser.add_argument(
        "--jobid", type=str, required=True, help="MeerSOLAR Job ID to kill"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    cachedir = get_cachedir()
    jobfile_name = f"{cachedir}/main_pids_{args.jobid}.txt"

    try:
        results = np.loadtxt(jobfile_name, dtype="str", unpack=True)
        main_pid = int(results[1])
        msname = str(results[2])
        workdir = str(results[3])
        outdir = str(results[4])
    except Exception as e:
        print(f"Could not read job file: {e}")
        return

    print(f"Attempting to terminate main PID: {main_pid}")
    terminate_process_and_children(main_pid)

    pid_file = f"{cachedir}/pids/pids_{args.jobid}.txt"
    if os.path.exists(pid_file):
        try:
            pids = np.loadtxt(pid_file, unpack=True, dtype="int")
            print(f"Terminating worker PIDs: {pids}")
            force_kill_pids_with_children(pids)
        except Exception as e:
            print(f"Could not read or terminate PIDs from {pid_file}: {e}")

    os.system(f"rm -rf {workdir}/tmp_meersolar_*")

    print("Dropping caches...")
    drop_cache(msname)
    drop_cache(workdir)
    drop_cache(outdir)
    drop_cache(cachedir)
    print("Cleanup complete.")


if __name__ == "__main__":
    kill_meerjob()
