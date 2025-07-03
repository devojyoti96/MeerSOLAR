from .all_depend import *
from .basic_utils import *

#################################
# MeerSOLAR process management
#################################


def get_nprocess_meersolar(jobid):
    """
    Get numbers of MeerSOLAR processes currently running

    Parameters
    ----------
    workdir : str
        Work directory name
    jobid : int
        MeerSOLAR Job ID

    Returns
    -------
    int
        Number of running processes
    """
    meersolar_cachedir = get_meersolar_cachedir()
    pid_file = f"{meersolar_cachedir}/pids/pids_{jobid}.txt"
    pids = np.loadtxt(pid_file, unpack=True)
    n_process = 0
    for pid in pids:
        if psutil.pid_exists(int(pid)):
            n_process += 1
    return n_process


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


def get_jobid():
    """
    Get MeerSOLAR Job ID with millisecond-level uniqueness.

    Returns
    -------
    int
        Job ID in the format YYYYMMDDHHMMSSmmm (milliseconds)
    """
    meersolar_cachedir = get_meersolar_cachedir()
    jobid_file = os.path.join(meersolar_cachedir, "jobids.txt")
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
    Save MeerSOLAR main processes info

    Parameters
    ----------
    pid : int
        Main job process id
    jobid : int
        MeerSOLAR Job ID
    msname : str
        Main measurement set
    workdir : str
        Work directory
    outdir : str
        Output directory
    cpu_frac : float
        CPU fraction of the job
    mem_frac : float
        Mempry fraction of the job

    Returns
    -------
    str
        Job info file name
    """
    meersolar_cachedir = get_meersolar_cachedir()
    prev_main_pids = glob.glob(f"{meersolar_cachedir}/main_pids_*.txt")
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
                if os.path.exists(f"{meersolar_cachedir}/pids/pids_{job_id}.txt"):
                    os.system(f"rm -rf {meersolar_cachedir}/pids/pids_{job_id}.txt")
    main_job_file = f"{meersolar_cachedir}/main_pids_{jobid}.txt"
    main_str = f"{jobid} {pid} {msname} {workdir} {outdir} {cpu_frac} {mem_frac}"
    with open(main_job_file, "w") as f:
        f.write(main_str)
    return main_job_file


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
        outputfile = workdir + "/logs/" + basename + ".log"
    else:
        outputfile = "/dev/null"
    batch_file_content = f"""export PYTHONUNBUFFERED=1\nnohup sh {cmd_batch}> {outputfile} 2>&1 &\nsleep 2\n rm -rf {batch_file}\n rm -rf {cmd_batch}"""
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
    return workdir + "/" + basename + ".batch"


def get_dask_client(
    n_jobs,
    dask_dir,
    cpu_frac=0.8,
    mem_frac=0.8,
    spill_frac=0.6,
    min_mem_per_job=-1,
    min_cpu_per_job=1,
    only_cal=False,
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
    cpu_frac : float
        Fraction of total CPUs to use
    mem_frac : float
        Fraction of total memory to use
    spill_frac : float, optional
        Spill to disk at this fraction
    min_mem_per_job : float, optional
        Minimum memory per job
    min_cpu_per_job : int, optional
        Minimum CPU threads per job
    only_cal : bool, optional
        Only calculate number of workers

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
    """
    # Create the Dask temporary working directory if it does not already exist
    os.makedirs(dask_dir, exist_ok=True)
    dask_dir_tmp = dask_dir + "/tmp"
    os.makedirs(dask_dir_tmp, exist_ok=True)

    # Detect total system resources
    total_cpus = psutil.cpu_count(logical=True)  # Total logical CPU cores
    total_mem = psutil.virtual_memory().total  # Total system memory (bytes)
    if cpu_frac > 0.8:
        print(
            "Given CPU fraction is more than 80%. Resetting to 80% to avoid system crash."
        )
        cpu_frac = 0.8
    if mem_frac > 0.8:
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
    if not only_cal:
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
        final_mem_per_worker = round((mem_per_worker * spill_frac) / (1024.0**3), 2)
        return None, None, n_workers, threads_per_worker, final_mem_per_worker

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
        dashboard_address=None,
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

    client.run_on_scheduler(gc.collect)
    final_mem_per_worker = round((mem_per_worker * spill_frac) / (1024.0**3), 2)
    return client, cluster, n_workers, threads_per_worker, final_mem_per_worker


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
