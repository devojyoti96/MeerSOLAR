import sys, traceback, time, gc
import os, numpy as np, copy, glob, argparse
from datetime import datetime as dt, timezone
from functools import partial
from meersolar.pipeline.basic_func import *
from dask import delayed, compute, config
from casatasks import casalog

try:
    logfile = casalog.logfile()
    os.system("rm -rf " + logfile)
except:
    pass


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
            except:
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
            except:
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
        except:
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
            except:
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
            if corcolumn_present == False:
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
            if datacolumn_present == False:
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
            except:
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
        except:
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
        except:
            pass
    except Exception as e:
        traceback.print_exc()
    finally:
        time.sleep(5)
        drop_cache(msname)
    return


def do_flagging(
    msname,
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
):
    """
    Function to perform initial flagging

    Parameters
    ----------
    msname : str
        Name of the ms
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
        task = delayed(single_ms_flag)(dry_run=True)
        mem_limit = run_limited_memory_task(task, dask_dir=mspath)
        dask_client, dask_cluster, n_jobs, n_threads, mem_limit = get_dask_client(
            len(subms_list),
            dask_dir=mspath,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit / 0.6,
        )
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
        results = compute(*tasks)
        dask_client.close()
        dask_cluster.close()
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
    finally:
        time.sleep(5)
        drop_cache(msname)


def main():
    usage = "Initial flagging of calibrator data"
    parser = argparse.ArgumentParser(
        description=usage, formatter_class=SmartDefaultsHelpFormatter
    )

    ## Essential parameters
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

    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )

    # Advanced switches
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

    ## Resource management parameters
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

    pid = os.getpid()
    save_pid(pid, datadir + f"/pids/pids_{args.jobid}.txt")

    if args.workdir == "" or not os.path.exists(args.workdir):
        workdir = os.path.dirname(os.path.abspath(args.msname)) + "/workdir"
    else:
        workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)

    observer = None
    if os.path.exists(f"{workdir}/jobname_password.npy") and args.logfile:
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(args.logfile):
            observer = init_logger(
                "do_flagging", args.logfile, jobname=jobname, password=password
            )

    try:
        if args.msname and os.path.exists(args.msname):
            msg = do_flagging(
                args.msname,
                datacolumn=args.datacolumn,
                flag_bad_ants=args.flag_bad_ants,
                flag_bad_spw=args.flag_bad_spw,
                use_tfcrop=args.use_tfcrop,
                use_rflag=args.use_rflag,
                flagdimension=args.flagdimension,
                flag_autocorr=args.flag_autocorr,
                flag_backup=args.flagbackup,
                cpu_frac=args.cpu_frac,
                mem_frac=args.mem_frac,
            )
        else:
            print("Please provide correct measurement set.\n")
            msg = 1
    except Exception as e:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        drop_cache(args.msname)
        drop_cache(args.workdir)
        clean_shutdown(observer)
    return msg


if __name__ == "__main__":
    result = main()
    print(f"Final msg : {result}")
    print("\n###################\nBasic flagging is finished.\n###################\n")
    os._exit(result)
