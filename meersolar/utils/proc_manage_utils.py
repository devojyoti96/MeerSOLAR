import types
import resource
import psutil
import dask
import numpy as np
import warnings
import gc
import logging
import time
import glob
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from dask import delayed, compute, config
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
from datetime import datetime as dt, timedelta
from .basic_utils import *


#################################
# Process management
#################################
def get_nprocess_solarpipe(jobid):
    """
    Get numbers of processes currently running

    Parameters
    ----------
    workdir : str
        Work directory name
    jobid : int
        Job ID

    Returns
    -------
    int
        Number of running processes
    """
    cachedir = get_cachedir()
    pid_file = f"{cachedir}/pids/pids_{jobid}.txt"
    pids = np.loadtxt(pid_file, unpack=True)
    n_process = 0
    for pid in pids:
        if psutil.pid_exists(int(pid)):
            n_process += 1
    return n_process


def get_jobid():
    """
    Get Job ID with millisecond-level uniqueness.

    Returns
    -------
    int
        Job ID in the format YYYYMMDDHHMMSSmmm (milliseconds)
    """
    cachedir = get_cachedir()
    jobid_file = os.path.join(cachedir, "jobids.txt")
    if os.path.exists(jobid_file):
        prev_jobids = np.loadtxt(jobid_file, unpack=True, dtype="int64")
        if prev_jobids.size == 0:
            prev_jobids = []
        elif prev_jobids.size == 1:
            prev_jobids = [str(prev_jobids)]
        else:
            prev_jobids = [str(jid) for jid in prev_jobids]
    else:
        prev_jobids = []

    if len(prev_jobids) > 0:
        FORMAT = "%Y%m%d%H%M%S%f"
        CUTOFF = dt.utcnow() - timedelta(days=15)
        filtered_prev_jobids = []
        for job_id in prev_jobids:
            job_time = dt.strptime(job_id.ljust(20, "0"), FORMAT)  # pad if truncated
            if job_time >= CUTOFF or job_id == 0:  # Job ID 0 is always kept
                filtered_prev_jobids.append(job_id)
        prev_jobids = filtered_prev_jobids

    now = dt.utcnow()
    cur_jobid = (
        now.strftime("%Y%m%d%H%M%S") + f"{int(now.microsecond/1000):03d}"
    )  # ms = first 3 digits of microseconds
    prev_jobids.append(cur_jobid)

    job_ids_int = np.array(prev_jobids, dtype=np.int64)
    np.savetxt(jobid_file, job_ids_int, fmt="%d")

    return int(cur_jobid)


def save_main_process_info(pid, jobid, msname, workdir, outdir, cpu_frac, mem_frac):
    """
    Save main processes info

    Parameters
    ----------
    pid : int
        Main job process id
    jobid : int
        Job ID
    msname : str
        Main measurement set
    workdir : str
        Work directory
    outdir : str
        Output directory
    cpu_frac : float
        CPU fraction of the job
    mem_frac : float
        Memory fraction of the job

    Returns
    -------
    str
        Job info file name
    """
    cachedir = get_cachedir()
    prev_main_pids = glob.glob(f"{cachedir}/main_pids_*.txt")
    prev_jobids = [
        str(os.path.basename(i).rstrip(".txt").split("main_pids_")[-1])
        for i in prev_main_pids
    ]
    if len(prev_jobids) > 0:
        FORMAT = "%Y%m%d%H%M%S%f"
        CUTOFF = dt.utcnow() - timedelta(days=15)
        filtered_prev_jobids = []
        for i in range(len(prev_jobids)):
            job_id = prev_jobids[i]
            job_time = dt.strptime(job_id.ljust(20, "0"), FORMAT)  # pad if truncated
            if job_time < CUTOFF or job_id == 0:  # Job ID 0 is always kept
                filtered_prev_jobids.append(job_id)
            else:
                os.system(f"rm -rf {prev_main_pids[i]}")
                if os.path.exists(f"{cachedir}/pids/pids_{job_id}.txt"):
                    os.system(f"rm -rf {cachedir}/pids/pids_{job_id}.txt")
    main_job_file = f"{cachedir}/main_pids_{jobid}.txt"
    main_str = f"{jobid} {pid} {msname} {workdir} {outdir} {cpu_frac} {mem_frac}"
    with open(main_job_file, "w") as f:
        f.write(main_str)
    return main_job_file


def save_pid(pid, pid_file):
    """
    Save PID

    Parameters
    ----------
    pid : int
        Process ID
    pid_file : str
        File to save
    """
    if os.path.exists(pid_file):
        pids = np.loadtxt(pid_file, unpack=True, dtype="int")
        pids = np.append(pids, pid)
    else:
        pids = np.array([int(pid)])
    np.savetxt(pid_file, pids, fmt="%d")


def generate_activate_env(outfile="activate_env.sh"):
    """
    Generate a shell script that activates the current Python environment.

    This works for both Conda and virtualenv environments and is safe for use in
    non-interactive shells (e.g., Slurm batch jobs) by explicitly sourcing `conda.sh`.

    If conda is not found in $PATH, it will try loading either `anaconda` or `anaconda3` module.

    Parameters
    ----------
    outfile : str
        Path to the shell script to write (default: ./activate_env.sh).

    Returns
    -------
    str
        Output file name
    """
    outfile = Path(outfile).expanduser().resolve()
    putfile = os.path.abspath(outfile)
    lines = ["#!/bin/bash", ""]

    def module_exists(name):
        """Check if a module exists using 'module avail'."""
        try:
            subprocess.run(
                ["module", "avail", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return True
        except Exception:
            return False

    # Conda-based environment
    if "CONDA_DEFAULT_ENV" in os.environ:
        conda_env = os.environ["CONDA_DEFAULT_ENV"]
        lines.append("# === Activate Conda Environment Safely ===")
        lines.append("if ! command -v conda >/dev/null 2>&1; then")
        if module_exists("anaconda"):
            lines.append("    module load anaconda")
        elif module_exists("anaconda3"):
            lines.append("    module load anaconda3")
        else:
            lines.append("    echo 'No Conda module found (anaconda or anaconda3)'")
            lines.append("    exit 1")
        lines.append("fi")
        lines.append("source $(conda info --base)/etc/profile.d/conda.sh")
        lines.append(f"conda activate {conda_env}")
    # Virtualenv-based environment
    elif "VIRTUAL_ENV" in os.environ:
        venv_path = os.environ["VIRTUAL_ENV"]
        lines.append("# === Activate Virtualenv ===")
        lines.append(f"source {venv_path}/bin/activate")
    else:
        python_path = sys.executable
        lines.append(
            "# === No Conda/Virtualenv Detected — Using current Python directly ==="
        )
        lines.append(f"echo 'No Conda or virtualenv detected; using: {python_path}'")
        lines.append(f"export PATH={os.path.dirname(python_path)}:$PATH")
    # Write file
    with open(outfile, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(outfile, 0o755)
    print(f"Created activation script at: {outfile}")
    return outfile


def get_total_worker(cluster):
    """
    Get total workers in the cluster

    Parameters
    ----------
    cluster : dask.cluster
        Dask cluster

    Returns
    -------
    int
        Number of workers
    """
    return len(cluster.workers)


def scale_worker_and_wait(dask_cluster, nworker, timeout=60, poll_interval=1):
    """
    Scale worker and wait until it is done

    Parameters
    ----------
    dask_cluster : dask.cluster
        Dask cluster
    nworker : int
        Number of worker
    timeout : float, optional
        Timeout, show a warning and move
    poll_interval : float, optional
        Check interval in seconds
    """
    print(f"Start scaling to {nworker} workers")
    dask_cluster.scale(nworker)
    timeout = 60
    c = 0
    while c < timeout:
        if get_total_worker(dask_cluster) == nworker:
            print(f"Successfully scaled to {nworker} workers")
            return 0
        else:
            time.sleep(poll_interval)
            c += poll_interval
    print(f"Dask cluster did not scale to {nworker} within {timeout} seconds.")
    return 1


def wait_for_dask_workers(client, min_worker=1, timeout=60):
    """
    Wait until the Dask cluster has a minimum number of total and/or new workers.

    Parameters
    ----------
    client : dask.distributed.Client
        Dask client
    min_worker : int, optional
        Minimum new connected workers (default: 1)
    timeout : float, optional
        Maximum time to wait in seconds (default: 60)

    Raises
    ------
    TimeoutError
        If the required number of workers do not connect in time.
    """
    client.wait_for_workers(n_workers=min_worker, timeout=timeout)


def get_scheduler_name():
    """
    Get job scheduler available

    Returns
    -------
    str
        Scheduler name (local, pbs, slurm)
    """
    if shutil.which("sbatch"):
        return "slurm"
    elif shutil.which("bsub"):
        return "lsf"
    elif shutil.which("qhost"):
        return "sge"
    elif shutil.which("qsub"):
        return "pbs"
    elif shutil.which("condor_submit"):
        return "htcondor"
    elif shutil.which("msub"):
        return "mab"
    elif shutil.which("oarsub"):
        return "oar"
    else:
        return "local"


def get_local_dask_cluster(
    njobs,
    dask_dir,
    cpu_frac=0.8,
    mem_frac=0.8,
    ncpu=-1,
    mem=-1,
    spill_frac=0.7,
    verbose=True,
):
    """
    Create a local Dask cluster

    Parameters
    ----------
    njobs : int
        Number of MS tasks (ideally = number of MS files)
    dask_dir : str
        Dask temporary directory
    cpu_frac : float, optional
        Fraction of total CPUs to use
    mem_frac : float, optional
        Fraction of total memory to use
    ncpu : int, optional
        Number of CPUs to use (if specified, cpu_frac will be ignored)
    mem : float, optional
        Memory in GB to use (if specified, mem_frac will be ignored)
    spill_frac : float, optional
        Spill to disk at this fraction
    verbose : bool, optional
        Verbose (details of cluster)

    Returns
    -------
    client : dask.distributed.Client
        Dask client
    cluster : dask.distributed.LocalCluster
        Dask cluster
    str
        Dask directory
    """
    logging.getLogger("distributed").setLevel(logging.ERROR)
    print("Creating local cluster on the current node.")
    # Set up Dask working directories
    dask_dir = os.path.join(dask_dir.rstrip("/"), f"dask_{int(time.time())}")
    dask_dir_tmp = os.path.join(dask_dir, "tmp")
    os.makedirs(dask_dir_tmp, exist_ok=True)

    total_cpus = psutil.cpu_count(logical=True)
    total_mem = psutil.virtual_memory().total / 1024**3  # In GB

    # Override fractions if ncpu or mem is provided
    if ncpu > 0:
        cpu_frac = min(ncpu / total_cpus, 0.8)
    if mem > 0:
        mem_frac = min(mem / total_mem, 0.8)
    cpu_frac = min(cpu_frac, 0.8)
    mem_frac = min(mem_frac, 0.8)
    usable_mem = total_mem * mem_frac
    usable_cpus = int(total_cpus * cpu_frac)
    n_workers = max(1, usable_cpus)

    # Raise file descriptor limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < int(hard * 0.8):
        resource.setrlimit(resource.RLIMIT_NOFILE, (int(hard * 0.8), hard))

    dask.config.set(
        {
            "temporary-directory": dask_dir,
            "distributed.worker.memory.target": spill_frac,
            "distributed.worker.memory.spill": spill_frac + 0.1,
            "distributed.worker.memory.pause": spill_frac + 0.2,
            "distributed.worker.memory.terminate": spill_frac + 0.25,
        }
    )

    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        memory_limit=f"{usable_mem}GB",
        local_directory=dask_dir,
        dashboard_address=":0",
        processes=True,
        env={
            "TMPDIR": dask_dir_tmp,
            "TMP": dask_dir_tmp,
            "TEMP": dask_dir_tmp,
            "DASK_TEMPORARY_DIRECTORY": dask_dir_tmp,
            "MALLOC_TRIM_THRESHOLD_": "0",
            "PYTHONWARNINGS": "ignore::UserWarning:contextlib",
        },
    )
    client = Client(cluster, heartbeat_interval="5s")
    client.run_on_scheduler(gc.collect)
    if verbose:
        print("####################################################")
        print(f"Dask dashboard available at: {client.dashboard_link}")
        print("####################################################")
    return client, cluster, dask_dir


def get_slurm_dask_cluster(
    njobs,
    config_yaml,
    dask_dir,
    cpu_frac=0.8,
    mem_frac=0.8,
    ncpu=-1,
    mem=-1,
    spill_frac=0.7,
    verbose=True,
):
    """
    Launch a SLURMCluster using a YAML configuration and return a connected Dask client.

    Parameters
    ----------
    njobs : int
        Number of expected tasks (used for worker scaling)
    config_yaml : str
        Path to Dask SLURMCluster YAML configuration
    dask_dir : str
        Dask working directory (for temporary files)
    cpu_frac : float
        Fraction of total CPUs to use (ignored if ncpu > 0)
    mem_frac : float
        Fraction of total RAM to use (ignored if mem > 0)
    ncpu : int
        Total CPUs to use (overrides cpu_frac)
    mem : float
        Total memory (in GB) to use (overrides mem_frac)
    spill_frac : float
        Fraction of memory to spill to disk
    verbose : bool
        Print Dask dashboard URL and diagnostics

    Returns
    -------
    client : dask.distributed.Client
        Connected Dask client
    cluster : dask_jobqueue.SLURMCluster
        SLURM Dask cluster
    str
        Dask directory used
    """
    logging.getLogger("distributed").setLevel(logging.ERROR)

    dask_dir = os.path.join(dask_dir.rstrip("/"), f"dask_{int(time.time())}")
    dask_dir_tmp = os.path.join(dask_dir, "tmp")
    os.makedirs(dask_dir_tmp, exist_ok=True)

    total_cpus = psutil.cpu_count(logical=True)
    total_mem = psutil.virtual_memory().total / 1024**3  # in GB

    if ncpu > 0:
        cpu_frac = min(ncpu / total_cpus, 0.8)
    if mem > 0:
        mem_frac = min(mem / total_mem, 0.8)

    cpu_frac = min(cpu_frac, 0.8)
    mem_frac = min(mem_frac, 0.8)
    usable_mem = total_mem * mem_frac
    usable_cpus = int(total_cpus * cpu_frac)

    # Raise file descriptor limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < int(hard * 0.8):
        resource.setrlimit(resource.RLIMIT_NOFILE, (int(hard * 0.8), hard))

    dask.config.set(
        {
            "temporary-directory": dask_dir,
            "distributed.worker.memory.target": spill_frac,
            "distributed.worker.memory.spill": spill_frac + 0.1,
            "distributed.worker.memory.pause": spill_frac + 0.2,
            "distributed.worker.memory.terminate": spill_frac + 0.25,
        }
    )

    # Load cluster config from YAML
    with open(config_yaml, "r") as f:
        cluster_config = yaml.safe_load(f)
    dask.config.update(cluster_config, priority="new")

    # Initialize SLURM cluster
    cluster = SLURMCluster(
        local_directory=dask_dir_tmp,
        env_extra=[
            f"TMPDIR={dask_dir_tmp}",
            f"TMP={dask_dir_tmp}",
            f"TEMP={dask_dir_tmp}",
            f"DASK_TEMPORARY_DIRECTORY={dask_dir_tmp}",
            "MALLOC_TRIM_THRESHOLD_=0",
            "PYTHONWARNINGS=ignore::UserWarning:contextlib",
        ],
    )

    # Scale workers (1 per task/MS file ideally)
    cluster.scale(njobs)
    client = Client(cluster, heartbeat_interval="5s")
    client.run_on_scheduler(gc.collect)
    if verbose:
        print("####################################################")
        print(f"Dask dashboard available at: {client.dashboard_link}")
        print("####################################################")

    return client, cluster, dask_dir


# Exposing only functions
__all__ = [
    name
    for name, obj in globals().items()
    if isinstance(obj, types.FunctionType) and obj.__module__ == __name__
]
