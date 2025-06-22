import os, sys, signal, argparse, numpy as np, time, psutil
from meersolar.pipeline.basic_func import *

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

        remaining = [pid for pid in np.atleast_1d(pids)
                     if psutil.pid_exists(int(pid))]

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
    datadir = get_datadir()
    jobfile_name = datadir + f"/main_pids_{args.jobid}.txt"
    results = np.loadtxt(jobfile_name, dtype="str", unpack=True)
    basedir = results[2]
    main_pid = results[1]
    pid_file = datadir + f"/pids/pids_{args.jobid}.txt"
    try:
        os.kill(int(main_pid), signal.SIGKILL)
    except:
        pass
    if os.path.exists(pid_file):
        pids = np.loadtxt(pid_file, unpack=True, dtype="int")
        force_kill_pids_with_children(pids)
       
if __name__=="__main__":
    kill_meerjob()
