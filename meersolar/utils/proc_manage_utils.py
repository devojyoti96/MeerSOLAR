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
from pathlib import Path
from dask import delayed, compute, config
from dask.distributed import Client, LocalCluster
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


def create_batch_script_nonhpc(cmd, workdir, basename, write_logfile=True):
    """
    Function to make a batch script not non-HPC environment

    Parameters
    ----------
    cmd : str
        Command to run
    workdir : str
        Work directory of the measurement set
    basename : str
        Base name of the batch files
    write_logfile : bool, optional
        Write log file or not

    Returns
    -------
    str
        Batch file name
    str
        Log file name
    """
    batch_file = workdir + "/" + basename + ".batch"
    cmd_batch = workdir + "/" + basename + "_cmd.batch"
    finished_touch_file = workdir + "/.Finished_" + basename
    os.system("rm -rf " + finished_touch_file + "*")
    finished_touch_file_error = finished_touch_file + "_1"
    finished_touch_file_success = finished_touch_file + "_0"
    cmd_file_content = f"{cmd}; exit_code=$?; if [ $exit_code -ne 0 ]; then touch {finished_touch_file_error}; else touch {finished_touch_file_success}; fi"
    if write_logfile:
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        logfile = workdir + "/logs/" + basename + ".log"
    else:
        logfile = "/dev/null"
    batch_file_content = f"""export PYTHONUNBUFFERED=1\nnohup sh {cmd_batch}> {logfile} 2>&1 &\nsleep 2\n rm -rf {batch_file}\n rm -rf {cmd_batch}"""
    if os.path.exists(cmd_batch):
        os.system("rm -rf " + cmd_batch)
    if os.path.exists(batch_file):
        os.system("rm -rf " + batch_file)
    with open(cmd_batch, "w") as cmd_batch_file:
        cmd_batch_file.write(cmd_file_content)
    with open(batch_file, "w") as b_file:
        b_file.write(batch_file_content)
    os.system("chmod a+rwx " + batch_file)
    os.system("chmod a+rwx " + cmd_batch)
    del cmd
    return workdir + "/" + basename + ".batch", logfile


def create_batch_script_slurm(
    cmd,
    workdir,
    basename,
    partition="general",
    nodes=1,
    cpus_per_task=1,
    ntasks_per_node=1,
    mem="4G",
    time="01:00:00",
    account=None,
    dependency=None,
    write_logfile=True,
):
    """
    Create a Slurm-compatible batch script.

    Parameters
    ----------
    cmd : str
        Command to run.
    workdir : str
        Work directory.
    basename : str
        Base name of batch script files. This will be Slurm job name as well (shown in `squeue`, `sacct`, etc.).
    partition : str, optional
        Name of the Slurm partition (queue) to submit to (e.g., "compute", "general").
    nodes : int, optional
        Number of physical nodes to request.
    cpus_per_task : int, optional
        Number of CPUs to allocate for each task (for multi-threaded tasks).
    ntasks_per_node : int, optional
        Number of tasks per node.
    mem : str, optional
        Total memory per node (e.g., "16G", "64000M").
    time : str, optional
        Maximum wall-clock time for the job in HH:MM:SS format.
    account : str, optional
        Slurm account to charge for the job (if required).
    dependency : str, optional
        Slurm job dependency (e.g., "afterok:12345") to delay job until others complete.
    write_logfile : bool, optional
        If True, write job output and error to {workdir}/logs/{basename}.log.
        If False, redirects output to /dev/null.

    Returns
    -------
    str
        Batch script filename.
    str
        Log file path (or /dev/null).
    """
    env_file = os.path.join(workdir, "activate_env.sh")
    # Generate the env file if it doesn't exist
    if not os.path.exists(env_file):
        generate_activate_env(outfile=env_file)
    os.makedirs(workdir, exist_ok=True)
    batch_file = os.path.join(workdir, f"{basename}.sbatch")
    cmd_batch = os.path.join(workdir, f"{basename}_cmd.batch")
    finished_touch_base = os.path.join(workdir, f".Finished_{basename}")
    finished_touch_error = finished_touch_base + "_1"
    finished_touch_success = finished_touch_base + "_0"
    # Remove old status files
    os.system(f"rm -rf {finished_touch_base}*")
    # === Command script content with environment activation ===
    cmd_file_content = "\n".join(
        [
            f"#!/bin/bash",
            f"source {env_file}",
            f"{cmd}",
            "exit_code=$?",
            f"if [ $exit_code -ne 0 ]; then touch {finished_touch_error};",
            f"else touch {finished_touch_success}; fi",
        ]
    )
    # Write log output
    if write_logfile:
        log_dir = os.path.join(workdir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        outputfile = os.path.join(log_dir, f"{basename}.log")
    else:
        outputfile = "/dev/null"
    # === SBATCH directives ===
    sbatch_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={basename}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks-per-node={ntasks_per_node}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
    ]
    if account:
        sbatch_lines.append(f"#SBATCH --account={account}")
    if dependency:
        sbatch_lines.append(f"#SBATCH --dependency={dependency}")
    if write_logfile:
        sbatch_lines.append(f"#SBATCH --output={outputfile}")
        sbatch_lines.append(f"#SBATCH --error={outputfile}")
    # === Batch script content ===
    batch_script_content = "\n".join(
        [
            *sbatch_lines,
            "",
            f"bash {cmd_batch}",
            f"rm -rf {batch_file}",
            f"rm -rf {cmd_batch}",
        ]
    )
    # Write files
    for f in [cmd_batch, batch_file]:
        if os.path.exists(f):
            os.remove(f)
    with open(cmd_batch, "w") as f:
        f.write(cmd_file_content)
    with open(batch_file, "w") as f:
        f.write(batch_script_content)
    os.chmod(cmd_batch, 0o777)
    os.chmod(batch_file, 0o777)
    return batch_file, outputfile


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


def wait_for_dask_workers(client, min_worker=1, timeout=60):
    """
    Wait for dask worker
    
    Parameters
    ----------
    client : dask.client
        Dask client
    min_worker : int, optional
        Minimum number of workers
    timeout : float, optional
        Timeout in seconds
    """
    start = time.time()
    while True:
        if len(client.scheduler_info()["workers"]) >= min_worker:
            break
        if time.time() - start > timeout:
            raise TimeoutError("No Dask workers connected within timeout.")
        time.sleep(2)
        
        
def get_dask_client(
    n_jobs,
    dask_dir,
    cpu_frac=0.8,
    mem_frac=0.8,
    ncpu=-1,
    mem=-1,
    spill_frac=0.6,
    min_mem_per_job=-1,
    min_cpu_per_job=1,
    only_cal=False,
    verbose=True,
):
    """
    Create a Dask client optimized for one-task-per-worker execution,
    where each worker is a separate process that can use multiple threads internally.

    Parameters
    ----------
    n_jobs : int
        Number of MS tasks (ideally = number of MS files)
    dask_dir : str
        Dask temporary directory
    cpu_frac : float, optional
        Fraction of total CPUs to use
    mem_frac : float, optinal;
        Fraction of total memory to use
    ncpu : int, optional
        Number of CPUs to use (if specified, cpu_frac will be ignored)
    mem : float, optional
        Memory in GB to use (if specified, mem_frac will be ignored)
    spill_frac : float, optional
        Spill to disk at this fraction
    min_mem_per_job : float, optional
        Minimum memory per job
    min_cpu_per_job : int, optional
        Minimum CPU threads per job
    only_cal : bool, optional
        Only calculate number of workers
    verbose : bool, optional
        Verbose (details of cluster)

    Returns
    -------
    client : dask.distributed.Client
        Dask clinet
    cluster : dask.distributed.LocalCluster
        Dask cluster
    n_workers : int
        Number of workers
    threads_per_worker : int
        Threads per worker to use
    str
        Dask directory
    """
    logging.getLogger("distributed").setLevel(logging.ERROR)
    # Create the Dask temporary working directory if it does not already exist
    dask_dir=dask_dir.rstrip("/")
    os.makedirs(dask_dir,exist_ok=True)
    dask_dir = tempfile.mkdtemp(prefix=f"{dask_dir}/dask_")
    os.makedirs(dask_dir, exist_ok=True)
    dask_dir_tmp = dask_dir + "/tmp"
    os.makedirs(dask_dir_tmp, exist_ok=True)

    # Detect total system resources
    total_cpus = psutil.cpu_count(logical=True)  # Total logical CPU cores
    total_mem = psutil.virtual_memory().total  # Total system memory (bytes)
    if ncpu >= 1:
        cpu_frac = float(ncpu / total_cpus)
    if mem > 0:
        mem_frac = float((mem * 1024**3) / total_mem)
    if cpu_frac > 0.8:
        if verbose:
            print(
                "Given CPU fraction is more than 80%. Resetting to 80% to avoid system crash."
            )
        cpu_frac = 0.8
    if mem_frac > 0.8:
        if verbose:
            print(
                "Given memory fraction is more than 80%. Resetting to 80% to avoid system crash."
            )
        mem_frac = 0.8

    ############################################
    # Wait until enough free CPU is available
    ############################################
    count = 0
    while True:
        available_cpu_pct = 100 - psutil.cpu_percent(
            interval=1
        )  # Percent CPUs currently free
        available_cpus = int(
            total_cpus * available_cpu_pct / 100.0
        )  # Number of free CPU cores
        usable_cpus = max(
            1, int(total_cpus * cpu_frac)
        )  # Target number of CPU cores we want available based on cpu_frac
        if available_cpus >= int(
            0.5 * usable_cpus
        ):  # Enough free CPUs (at-least more than 50%), exit loop
            usable_cpus = min(usable_cpus, available_cpus)
            break
        else:
            if count == 0:
                print("Waiting for available free CPUs...")
            time.sleep(5)  # Wait a bit and retry
        count += 1

    ############################################
    # Wait until enough free memory is available
    ############################################
    count = 0
    while True:
        available_mem = (
            psutil.virtual_memory().available
        )  # Current available system memory (bytes)
        usable_mem = total_mem * mem_frac  # Target usable memory based on mem_frac
        if (
            available_mem >= 0.5 * usable_mem
        ):  # Enough free memory, (at-least more than 50%) exit loop
            usable_mem = min(usable_mem, available_mem)
            break
        else:
            if count == 0:
                print("Waiting for available free memory...")
            time.sleep(5)  # Wait and retry
        count += 1

    ############################################
    # Calculate memory per worker
    ############################################
    mem_per_worker = usable_mem / n_jobs  # Assume initially one job per worker
    # Apply minimum memory per worker constraint
    min_mem_per_job = round(
        min_mem_per_job, 2
    )  # Ensure min_mem_per_job is a clean float
    if min_mem_per_job > 0 and mem_per_worker < (min_mem_per_job * 1024**3):
        # If calculated memory per worker is smaller than user-requested
        # minimum, adjust number of workers
        if verbose:
            print(
                f"Total memory per job is smaller than {min_mem_per_job} GB. Adjusting total number of workers to meet this."
            )
        mem_per_worker = (
            min_mem_per_job * 1024**3
        )  # Reset memory per worker to minimum allowed
        n_workers = min(
            n_jobs, int(usable_mem / mem_per_worker)
        )  # Reduce number of workers accordingly
    else:
        # Otherwise, just keep n_jobs workers
        n_workers = n_jobs

    #########################################
    # Cap number of workers to available CPUs
    n_workers = max(
        1, min(n_workers, int(usable_cpus / min_cpu_per_job))
    )  # Prevent CPU oversubscription
    # Recalculate final memory per worker based on capped n_workers
    mem_per_worker = usable_mem / n_workers
    # Calculate threads per worker
    threads_per_worker = max(
        1, usable_cpus // max(1, n_workers)
    )  # Each worker gets min_cpu_per_job or more threads

    ##########################################
    if not only_cal and verbose:
        print("#################################")
        print(
            f"Dask workers: {n_workers}, Threads per worker: {threads_per_worker}, Mem/worker: {round(mem_per_worker/(1024.0**3),2)} GB"
        )
        print("#################################")
    # Memory control settings
    swap = psutil.swap_memory()
    swap_gb = swap.total / 1024.0**3
    if swap_gb > 16:
        pass
    elif swap_gb > 4:
        spill_frac = 0.6
    else:
        spill_frac = 0.5

    if spill_frac > 0.7:
        spill_frac = 0.7
    if only_cal:
        return None, None, n_workers, threads_per_worker, round(mem_per_worker/(1024.0**3),2), dask_dir

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(int(hard * 0.8), hard)  # safe cap
    if soft < new_soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    dask.config.set({"temporary-directory": dask_dir})
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        # one python-thread per worker, in workers OpenMP threads can be used
        memory_limit=f"{round(mem_per_worker/(1024.0**3),2)}GB",
        local_directory=dask_dir,
        processes=True,  # one process per worker
        dashboard_address=":0",
        env={
            "TMPDIR": dask_dir_tmp,
            "TMP": dask_dir_tmp,
            "TEMP": dask_dir_tmp,
            "DASK_TEMPORARY_DIRECTORY": dask_dir,
            "MALLOC_TRIM_THRESHOLD_": "0",
        },  # Explicitly set for workers
    )
    client = Client(cluster, timeout="60s", heartbeat_interval="5s")
    dask.config.set(
        {
            "distributed.worker.memory.target": spill_frac,
            "distributed.worker.memory.spill": spill_frac + 0.1,
            "distributed.worker.memory.pause": spill_frac + 0.2,
            "distributed.worker.memory.terminate": spill_frac + 0.25,
        }
    )
    if verbose:
        print(f"Dask dashboard available at: {client.dashboard_link}")

    client.run_on_scheduler(gc.collect)
    return client, cluster, n_workers, threads_per_worker, round(mem_per_worker/(1024.0**3),2), dask_dir


def run_limited_memory_task(task, dask_dir="/tmp", timeout=30):
    """
    Run a task for a limited time, then kill and return memory usage.

    Parameters
    ----------
    task : dask.delayed
        Dask delayed task object
    timeout : int
        Time in seconds to let the task run

    Returns
    -------
    float
        Memory used by task (in GB)
    """
    dask.config.set({"temporary-directory": dask_dir})
    warnings.filterwarnings("ignore")
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        # one python-thread per worker, in workers OpenMP threads can be used
        local_directory=dask_dir,
        processes=True,  # one process per worker
        dashboard_address=":0",
    )
    client = Client(cluster)
    future = client.compute(task)
    start = time.time()

    def get_worker_memory_info():
        proc = psutil.Process()
        return {
            "rss_GB": proc.memory_info().rss / 1024**3,
            "vms_GB": proc.memory_info().vms / 1024**3,
        }

    while not future.done():
        if time.time() - start > timeout:
            try:
                mem_info = client.run(get_worker_memory_info)
                total_rss = sum(v["rss_GB"] for v in mem_info.values())
                per_worker_mem = total_rss
            except Exception as e:
                per_worker_mem = None
            future.cancel()
            client.close()
            cluster.close()
            return per_worker_mem
        time.sleep(1)
    mem_info = client.run(get_worker_memory_info)
    total_rss = sum(v["rss_GB"] for v in mem_info.values())
    per_worker_mem = total_rss
    client.close()
    cluster.close()
    return round(per_worker_mem, 2)


# Exposing only functions
__all__ = [
    name
    for name, obj in globals().items()
    if isinstance(obj, types.FunctionType) and obj.__module__ == __name__
]
