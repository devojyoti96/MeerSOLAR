import time, os, numpy as np, psutil, dask, gc
from dask.distributed import Client, LocalCluster
from dask import delayed, compute, config

def get_datadir():
    """
    Get package data directory
    """
    from importlib.resources import files

    datadir_path = str(files("meersolar").joinpath("data"))
    return datadir_path
    
def get_dask_client(n_jobs, dask_dir="/tmp", cpu_frac=0.8, mem_frac=0.8, spill_frac = 0.6, min_mem_per_job=-1):
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
    if os.path.exists(dask_dir) == False:
        os.makedirs(dask_dir)
    
    # Detect total system resources
    total_cpus = psutil.cpu_count(logical=True)  # Total logical CPU cores
    total_mem = psutil.virtual_memory().total    # Total system memory (bytes)
    if cpu_frac>0.8:
        print ("Given CPU fraction is more than 80%. Resetting to 80% to avoid system crash.")
        cpu_frac=0.8
    if mem_frac>0.8:
        print ("Given memory fraction is more than 80%. Resetting to 80% to avoid system crash.")
        mem_frac=0.8
        
    ############################################
    # Wait until enough free CPU is available
    ############################################
    count = 0
    while True:
        available_cpu_pct = 100 - psutil.cpu_percent(interval=1)  # Percent CPUs currently free
        available_cpus = int(total_cpus * available_cpu_pct / 100.0)  # Number of free CPU cores
        usable_cpus = max(1, int(total_cpus * cpu_frac))  # Target number of CPU cores we want available based on cpu_frac
        if available_cpus >= int(0.5*usable_cpus): # Enough free CPUs (at-least more than 50%), exit loop
            usable_cpus=min(usable_cpus,available_cpus)
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
        available_mem = psutil.virtual_memory().available  # Current available system memory (bytes)
        usable_mem = total_mem * mem_frac  # Target usable memory based on mem_frac
        if available_mem >= usable_mem: # Enough free memory , exit loop
            usable_mem=min(usable_mem,available_mem)
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
    min_mem_per_job = round(min_mem_per_job, 2)  # Ensure min_mem_per_job is a clean float
    if min_mem_per_job > 0 and mem_per_worker < (min_mem_per_job * 1024**3):
        # If calculated memory per worker is smaller than user-requested minimum, adjust number of workers
        print(
            f"Total memory per job is smaller than {min_mem_per_job} GB. Adjusting total number of workers to meet this."
        )
        mem_per_worker = min_mem_per_job * 1024**3  # Reset memory per worker to minimum allowed
        n_workers = min(n_jobs, int(usable_mem / mem_per_worker))  # Reduce number of workers accordingly
    else:
        # Otherwise, just keep n_jobs workers
        n_workers = n_jobs
    #########################################
    # Cap number of workers to available CPUs
    n_workers = min(n_workers, usable_cpus)  # Prevent CPU oversubscription
    # Recalculate final memory per worker based on capped n_workers
    mem_per_worker = usable_mem / n_workers
    # Calculate threads per worker
    threads_per_worker = max(1, usable_cpus // max(1, n_workers))  # Each worker gets 1 or more threads
    ##########################################
    print("\n#################################")
    print(
        f"Dask workers: {n_workers}, Threads per worker: {threads_per_worker}, Mem/worker: {mem_per_worker/1e9:.2f} GB"
    )

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1, # one python-thread per worker, in workers OpenMP threads can be used
        memory_limit=f"{mem_per_worker/1e9:.2f}GB",
        local_directory=dask_dir,
        processes=True,  # one process per worker
        dashboard_address=":0",
    )
    client = Client(cluster)
    # Memory control settings
    swap = psutil.swap_memory()
    swap_gb = swap.total / 1024**3  
    if swap_gb>16:
        pass
    elif swap_gb>4:
        spill_frac=0.6
    else:
        spill_frac=0.5  
        
    if spill_frac>0.7:
        spill_frac=0.7    
    dask.config.set(
        {
            "distributed.worker.memory.target": spill_frac,
            "distributed.worker.memory.spill": 0.7,
            "distributed.worker.memory.pause": 0.8,
            "distributed.worker.memory.terminate": 0.95,
        }
    )

    client.run_on_scheduler(gc.collect)

    print(f"Dask Dashboard: {client.dashboard_link}")
    print("#################################\n")

    return client, cluster, n_workers, threads_per_worker   
    
def get_observatory_info(observatory_name):
    """
    Parameters
    ----------
    observatory_name : str
        Name of the observatory
    Returns
    -------
    float
        Observatory latitude
    float
        Observatory longitude
    float
        Observatory altitude
    """
    observatories = {
        "meerkat": {
            "latitude": -30.713,
            "longitude": 21.443,
            "altitude": 1038,
        },  # In meters
        "ugmrt": {
            "latitude": 19.096,
            "longitude": 74.050,
            "altitude": 650,
        },  # In meters
        "eovsa": {
            "latitude": 37.233,
            "longitude": -118.283,
            "altitude": 1222,
        },  # In meters
        "askap": {
            "latitude": -26.696,
            "longitude": 116.630,
            "altitude": 377,
        },  # In meters
        "fasr": {
            "latitude": 38.430,
            "longitude": -79.839,
            "altitude": 820,
        },  # Approximate value
        "skao-mid": {
            "latitude": -30.721,
            "longitude": 21.411,
            "altitude": 1060,
        },  # Approximate location
        "skao-low": {
            "latitude": -26.7033,
            "longitude": 116.6319,
            "altitude": 377,
        },  # Approximate location
        "jvla": {
            "latitude": 34.0784,
            "longitude": -107.6184,
            "altitude": 2124,
        },  # In meters
    }
    keys = list(observatories.keys())
    if observatory_name.lower() not in keys:
        print("Observatory: " + observatory_name + " is not in the list.\n")
        print(
            "Available observatories: MeerKAT, uGMRT, eOVSA, ASKAP, FASR, SKAO-MID, JVLA.\n"
        )
        return
    else:
        pos = observatories[observatory_name.lower()]
        lat = pos["latitude"]
        lon = pos["longitude"]
        alt = pos["altitude"]
        return lat, lon, alt
        
