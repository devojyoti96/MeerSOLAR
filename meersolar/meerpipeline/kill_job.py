import psutil
import numpy as np
import argparse
import time
import sys
import os
import signal
from casatasks import casalog
from meersolar.utils import get_cachedir, drop_cache

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    pass


def kill_process_and_children(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except Exception:
                pass
        parent.kill()
    except (psutil.NoSuchProcess, ProcessLookupError):
        pass


def force_kill_pids_with_children(pids, max_tries=10, wait_time=0.5):
    """
    Repeatedly try to kill all PIDs (and their children) until none remain.
    """
    for attempt in range(max_tries):
        remaining = []
        for pid in np.atleast_1d(pids):
            try:
                kill_process_and_children(int(pid))
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
    Kill MeerSOLAR Job
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
    results = np.loadtxt(jobfile_name, dtype="str", unpack=True)
    main_pid = int(results[1])
    msname = str(results[2])
    workdir = str(results[3])
    outdir = str(results[4])
    pid_file = f"{cachedir}/pids/pids_{args.jobid}.txt"
    try:
        os.kill(int(main_pid), signal.SIGKILL)
    except BaseException:
        pass
    if os.path.exists(pid_file):
        pids = np.loadtxt(pid_file, unpack=True, dtype="int")
        force_kill_pids_with_children(pids)
    os.system(f"rm -rf {workdir}/tmp_meersolar_*")
    drop_cache(msname)
    drop_cache(workdir)
    drop_cache(outdir)
    drop_cache(cachedir)


if __name__ == "__main__":
    kill_meerjob()
