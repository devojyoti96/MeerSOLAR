import logging
import psutil
import dask
import numpy as np
import argparse
import traceback
import time
import glob
import sys
import os
from casatasks import casalog
from casatools import msmetadata
from dask import delayed
from dask.distributed import Client
from meersolar.utils import *

logging.getLogger("distributed").setLevel(logging.WARNING)

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    pass

datadir = get_datadir()


def single_ms_flag(
    msname="",
    badspw="",
    bad_ants_str="",
    datacolumn="data",
    use_tfcrop=True,
    use_rflag=False,
    flagdimension="freqtime",
    flag_autocorr=True,
    n_threads=-1,
    memory_limit=-1,
    dry_run=False,
):
    """
    Flag on a single ms

    Parameters
    ----------
    msname : str
        Measurement set name
    badspw : str, optional
        Bad spectral window
    bad_ants_str : str, optional
        Bad antenna string
    datacolumn : str, optional
        Data column
    use_tfcrop : str, optional
        Use tfcrop or not
    use_rflag : str, optional
        Use rflag or not
    flagdimension : str, optional
        Flag dimension (only applicable for tfcrop)
    flag_autocorr : bool, optional
        Flag autocorrelations or not
    n_threads : int, optional
        Number of OpenMP threads
    memory_limit : float, optional
        Memory limit in GB
    dry_run : bool, optional
        Return the amount of pre-occupied memory in GB
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    try:
        ##############################
        # Flagging bad channels
        ##############################
        if badspw != "":
            print(f"Flagging bad spectral windows: {badspw}\n")
            try:
                with suppress_casa_output():
                    flagdata(
                        vis=msname,
                        mode="manual",
                        spw=badspw,
                        cmdreason="badchan",
                        flagbackup=False,
                    )
            except BaseException:
                pass

        ##############################
        # Flagging bad antennas
        ##############################
        if bad_ants_str != "":
            print(f"Flagging bad antenna: {bad_ants_str}\n")
            try:
                with suppress_casa_output():
                    flagdata(
                        vis=msname,
                        mode="manual",
                        antenna=bad_ants_str,
                        cmdreason="badant",
                        flagbackup=False,
                    )
            except BaseException:
                pass

        #################################
        # Clip zero amplitude data points
        #################################
        try:
            with suppress_casa_output():
                flagdata(
                    vis=msname,
                    mode="clip",
                    clipzeros=True,
                    datacolumn=datacolumn,
                    autocorr=flag_autocorr,
                    flagbackup=False,
                )
        except BaseException:
            pass

        #################################
        # Flag auto-correlations
        #################################
        if flag_autocorr:
            try:
                with suppress_casa_output():
                    flagdata(
                        vis=msname,
                        mode="manual",
                        autocorr=True,
                        datacolumn=datacolumn,
                        flagbackup=False,
                    )
            except BaseException:
                pass

        ####################################################
        # Check if required columns are present for residual
        ####################################################
        if datacolumn == "residual" or datacolumn == "RESIDUAL":
            modelcolumn_present = check_datacolumn_valid(
                msname, datacolumn="MODEL_DATA"
            )
            corcolumn_present = check_datacolumn_valid(
                msname, datacolumn="CORRECTED_DATA"
            )
            if modelcolumn_present == False or corcolumn_present == False:
                datacolumn = "corrected"
        elif datacolumn == "RESIDUAL_DATA":
            modelcolumn_present = check_datacolumn_valid(
                msname, datacolumn="MODEL_DATA"
            )
            datacolumn_present = check_datacolumn_valid(msname, datacolumn="DATA")
            if modelcolumn_present == False or datacolumn_present == False:
                datacolumn = "corrected"

        #################################################
        # Whether corrected data column is present or not
        #################################################
        if datacolumn == "corrected" or datacolumn == "CORRECTED_DATA":
            corcolumn_present = check_datacolumn_valid(
                msname, datacolumn="CORRECTED_DATA"
            )
            if not corcolumn_present:
                print(
                    "Corrected data column is chosen for flagging, but it is not present.\n"
                )
                return
            else:
                datacolumn = "corrected"

        #################################################
        # Whether data column is present or not
        #################################################
        if datacolumn == "data" or datacolumn == "DATA":
            datacolumn_present = check_datacolumn_valid(msname, datacolumn="DATA")
            if not datacolumn_present:
                print("Data column is chosen for flagging, but it is not present.\n")
                return
            else:
                datacolumn = "data"

        ###########################
        # Determinign time chunking
        ############################
        if use_tfcrop or use_rflag:
            nchunk = get_chunk_size(msname, memory_limit=memory_limit)
            if nchunk <= 1:
                ntime = "scan"
            else:
                msmd = msmetadata()
                msmd.open(msname)
                scan = np.unique(msmd.scannumbers())[0]
                times = msmd.timesforspws(0)
                msmd.close()
                total_time = np.nanmax(times) - np.nanmin(times)
                timeres = np.nanmin(np.diff(times))
                ntime = float(total_time / nchunk)
                if ntime < timeres:
                    ntime = timeres

        ##############
        # Tfcrop flag
        ##############
        if use_tfcrop:
            try:
                with suppress_casa_output():
                    flagdata(
                        vis=msname,
                        mode="tfcrop",
                        timefit="line",
                        freqfit="line",
                        extendflags=False,
                        flagdimension=flagdimension,
                        timecutoff=5.0,
                        freqcutoff=5.0,
                        extendpols=True,
                        growaround=False,
                        action="apply",
                        flagbackup=False,
                        overwrite=True,
                        writeflags=True,
                        datacolumn=datacolumn,
                        ntime=ntime,
                    )
            except BaseException:
                pass

        #############
        # Rflag flag
        #############
        try:
            with suppress_casa_output():
                flagdata(
                    vis=msname,
                    mode="rflag",
                    timefit="line",
                    freqfit="line",
                    extendflags=False,
                    timedevscale=5.0,
                    freqdevscale=5.0,
                    extendpols=True,
                    growaround=False,
                    action="apply",
                    flagbackup=False,
                    overwrite=True,
                    writeflags=True,
                    datacolumn=datacolumn,
                    ntime=ntime,
                )
        except BaseException:
            pass

        ##############
        # Extend flag
        ##############
        try:
            with suppress_casa_output():
                flagdata(
                    vis=msname,
                    mode="extend",
                    datacolumn="data",
                    clipzeros=True,
                    extendflags=False,
                    extendpols=True,
                    growtime=80.0,
                    growfreq=80.0,
                    growaround=False,
                    flagneartime=False,
                    flagnearfreq=False,
                    action="apply",
                    flagbackup=False,
                    overwrite=True,
                    writeflags=True,
                    ntime=ntime,
                )
        except BaseException:
            pass
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1


def do_flagging(
    msname,
    workdir,
    datacolumn="data",
    flag_bad_ants=True,
    flag_bad_spw=True,
    use_tfcrop=True,
    use_rflag=False,
    flagdimension="freqtime",
    flag_autocorr=True,
    flag_backup=True,
    cpu_frac=0.8,
    mem_frac=0.8,
    dask_addr=None,
):
    """
    Function to perform initial flagging

    Parameters
    ----------
    msname : str
        Name of the ms
    workdir : str
        Work directory
    datacolumn : str, optional
        Data column
    flag_bad_ants : bool, optional
        Flag bad antennas
    flag_bad_spw : bool, optional
        Flag bad channels
    use_tfcrop : bool, optional
        Use tfcrop or not
    use_rflag : bool, optional
        Use rflag or not
    flagdimension : str, optional
        Flag dimension (only for tfcrop)
    flag_autocorr : bool,optional
        Flag auto-correlations
    flag_backup : bool, optional
        Flag backup
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message
    """
    start_time = time.time()
    try:
        from casatasks import flagdata

        msname = msname.rstrip("/")
        mspath = os.path.dirname(os.path.abspath(msname))
        os.chdir(mspath)
        print("###########################")
        print("Flagging measurement set : ", msname)
        print("###########################\n")
        total_cpus = psutil.cpu_count(logical=True)
        ncpu = int(total_cpus * cpu_frac)
        correct_missing_col_subms(msname)
        print("Restoring all previous flags...")
        with suppress_casa_output():
            flagdata(vis=msname, mode="unflag", spw="0", flagbackup=False)
        fluxcal_field, fluxcal_scans = get_fluxcals(msname)
        if len(fluxcal_field) == 0:
            flag_bad_spw = False
            flag_bad_ants = False
        if flag_bad_spw:
            badspw = get_bad_chans(msname)
        else:
            bandspw = ""
        if flag_bad_ants:
            bad_ants, bad_ants_str = get_bad_ants(msname, fieldnames=fluxcal_field)
        else:
            bad_ants_str = ""
        ###########################
        # Dask local cluster setup
        ##########################
        if os.path.exists(msname + "/SUBMSS"):
            subms_list = glob.glob(msname + "/SUBMSS/*")
            for subms in subms_list:
                os.system(f"rm -rf {subms}/.flagversions")
        else:
            subms_list = [msname]
        ms_size_list = [get_column_size(ms) for ms in subms_list]
        mem_limit = max(ms_size_list)
        if dask_addr is None:
            dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = (
                get_dask_client(
                    len(subms_list),
                    dask_dir=workdir,
                    cpu_frac=cpu_frac,
                    mem_frac=mem_frac,
                    min_mem_per_job=mem_limit,
                )
            )
        else:
            _, _, n_jobs, n_threads, mem_limit, dask_dir = get_dask_client(
                len(subms_list),
                dask_dir=workdir,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                min_mem_per_job=mem_limit,
                only_cal=True,
            )
            os.system(f"rm -rf {dask_dir}")
            dask_client = Client(address=dask_addr)
        wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
        if flag_backup:
            do_flag_backup(msname, flagtype="flagdata")
        tasks = [
            delayed(single_ms_flag)(
                ms,
                badspw=badspw,
                bad_ants_str=bad_ants_str,
                datacolumn=datacolumn,
                use_tfcrop=use_tfcrop,
                use_rflag=use_rflag,
                flagdimension=flagdimension,
                flag_autocorr=flag_autocorr,
                n_threads=n_threads,
                memory_limit=mem_limit,
            )
            for ms in subms_list
        ]
        futures = dask_client.compute(tasks)
        results = list(dask_client.gather(futures))
        dask_client.close()
        if dask_addr is None:
            dask_cluster.close()
            os.system(f"rm -rf {dask_dir}")
        print("##################")
        print("Total time taken : ", time.time() - start_time)
        print("##################\n")
        return 0
    except Exception as e:
        traceback.print_exc()
        print("##################")
        print("Total time taken : " + str(time.time() - start_time) + "s")
        print("##################\n")
        return 1


def main(
    msname,
    workdir="",
    datacolumn="DATA",
    flag_bad_ants=True,
    flag_bad_spw=True,
    use_tfcrop=False,
    use_rflag=False,
    flag_autocorr=True,
    flagbackup=True,
    flagdimension="freqtime",
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    start_remote_log=False,
    dask_addr=None,
):
    """
    Run the flagging pipeline for a measurement set.

    Parameters
    ----------
    msname : str
        Path to the input measurement set (MS) to be flagged.
    workdir : str, optional
        Working directory to store logs and temporary files. If empty, defaults to
        `<msname>/workdir`. Default is "".
    datacolumn : str, optional
        Data column to be flagged (e.g., "DATA", "CORRECTED"). Default is "DATA".
    flag_bad_ants : bool, optional
        If True, flags known bad antennas using pre-defined heuristics. Default is True.
    flag_bad_spw : bool, optional
        If True, flags bad spectral windows based on statistics. Default is True.
    use_tfcrop : bool, optional
        If True, applies the `tfcrop` automated flagging algorithm. Default is False.
    use_rflag : bool, optional
        If True, applies the `rflag` automated flagging algorithm. Default is False.
    flag_autocorr : bool, optional
        If True, flags auto-correlations. Default is True.
    flagbackup : bool, optional
        If True, saves a flag backup before applying new flags. Default is True.
    flagdimension : str, optional
        Dimension over which to apply automated flagging (e.g., "freqtime"). Default is "freqtime".
    cpu_frac : float, optional
        Fraction of total CPU resources to use. Default is 0.8.
    mem_frac : float, optional
        Fraction of total memory resources to use. Default is 0.8.
    logfile : str or None, optional
        Path to the log file for saving logs. If None, logging to file is skipped.
    jobid : int, optional
        Numeric job ID used for PID tracking. Default is 0.
    start_remote_log : bool, optional
        Whether to enable remote logging using credentials in the workdir. Default is False.
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message
    """
    pid = os.getpid()
    cachedir = get_cachedir()
    save_pid(pid, f"{cachedir}/pids/pids_{jobid}.txt")

    if workdir == "":
        workdir = os.path.dirname(os.path.abspath(msname)) + "/workdir"
    os.makedirs(workdir, exist_ok=True)

    ############
    # Logger
    ############
    observer = None
    if (
        start_remote_log
        and os.path.exists(f"{workdir}/jobname_password.npy")
        and logfile is not None
    ):
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            observer = init_logger(
                "do_flagging", logfile, jobname=jobname, password=password
            )
    if observer == None:
        print("Remote link or jobname is blank. Not transmiting to remote logger.")
    ###########

    try:
        if msname and os.path.exists(msname):
            msg = do_flagging(
                msname,
                workdir,
                datacolumn=datacolumn,
                flag_bad_ants=flag_bad_ants,
                flag_bad_spw=flag_bad_spw,
                use_tfcrop=use_tfcrop,
                use_rflag=use_rflag,
                flagdimension=flagdimension,
                flag_autocorr=flag_autocorr,
                flag_backup=flagbackup,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                dask_addr=dask_addr,
            )
        else:
            print("Please provide correct measurement set.\n")
            msg = 1
    except Exception as e:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        drop_cache(msname)
        drop_cache(workdir)
        clean_shutdown(observer)
    return msg


def cli():
    usage = "Initial flagging of calibrator data"
    parser = argparse.ArgumentParser(
        description=usage, formatter_class=SmartDefaultsHelpFormatter
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument("msname", type=str, help="Name of measurement set")
    basic_args.add_argument(
        "--workdir", type=str, default="", help="Name of work directory"
    )
    basic_args.add_argument(
        "--datacolumn", type=str, default="DATA", help="Name of the datacolumn"
    )

    # Advanced switches
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--no_flag_bad_ants",
        dest="flag_bad_ants",
        action="store_false",
        help="Do not flag bad antennas",
    )
    adv_args.add_argument(
        "--no_flag_bad_spw",
        dest="flag_bad_spw",
        action="store_false",
        help="Do not flag bad spectral windows",
    )
    adv_args.add_argument(
        "--use_tfcrop", action="store_true", help="Use tfcrop flagging"
    )
    adv_args.add_argument("--use_rflag", action="store_true", help="Use rflag flagging")
    adv_args.add_argument(
        "--no_flag_autocorr",
        dest="flag_autocorr",
        action="store_false",
        help="Do not flag auto-correlations",
    )
    adv_args.add_argument(
        "--no_flagbackup",
        dest="flagbackup",
        action="store_false",
        help="Do not backup flags",
    )
    adv_args.add_argument(
        "--start_remote_log", action="store_true", help="Start remote logging"
    )

    # Resource management parameters
    hard_args = parser.add_argument_group(
        "###################\nHardware resource management parameters\n###################"
    )
    hard_args.add_argument(
        "--flagdimension", type=str, default="freqtime", help="Flag dimension"
    )
    hard_args.add_argument(
        "--cpu_frac", type=float, default=0.8, help="CPU fraction to use"
    )
    hard_args.add_argument(
        "--mem_frac", type=float, default=0.8, help="Memory fraction to use"
    )
    hard_args.add_argument("--logfile", type=str, default=None, help="Log file")
    hard_args.add_argument("--jobid", type=int, default=0, help="Job ID")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    msg = main(
        msname=args.msname,
        workdir=args.workdir,
        datacolumn=args.datacolumn,
        flag_bad_ants=args.flag_bad_ants,
        flag_bad_spw=args.flag_bad_spw,
        use_tfcrop=args.use_tfcrop,
        use_rflag=args.use_rflag,
        flag_autocorr=args.flag_autocorr,
        flagbackup=args.flagbackup,
        flagdimension=args.flagdimension,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        logfile=args.logfile,
        jobid=args.jobid,
        start_remote_log=args.start_remote_log,
    )
    return msg


if __name__ == "__main__":
    result = cli()
    print(f"Final msg : {result}")
    print("\n###################\nBasic flagging is finished.\n###################\n")
    os._exit(result)
