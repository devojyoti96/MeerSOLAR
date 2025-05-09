import sys, traceback, time, gc
import os, numpy as np, copy, glob
from optparse import OptionParser
from datetime import datetime as dt, timezone
from functools import partial
from meersolar.pipeline.basic_func import *
from dask import delayed, compute, config
from casatasks import casalog

logfile = casalog.logfile()
os.system("rm -rf " + logfile)


def single_ms_flag(
    msname="",
    badspw="",
    bad_ants_str="",
    datacolumn="DATA",
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
    ##############################
    # Flagging bad channels
    ##############################
    if badspw != "":
        flagdata(
            vis=msname,
            mode="manual",
            spw=badspw,
            cmdreason="badchan",
            flagbackup=False,
        )

    ##############################
    # Flagging bad antennas
    ##############################
    if bad_ants_str != "":
        flagdata(
            vis=msname,
            mode="manual",
            antenna=bad_ants_str,
            cmdreason="badant",
            flagbackup=False,
        )

    #################################
    # Clip zero amplitude data points
    #################################
    flagdata(
        vis=msname,
        mode="clip",
        clipzeros=True,
        datacolumn=datacolumn,
        autocorr=flag_autocorr,
        flagbackup=False,
    )

    #################################
    # Flag auto-correlations
    #################################
    if flag_autocorr:
        flagdata(
            vis=msname,
            mode="manual",
            autocorr=True,
            datacolumn=datacolumn,
            flagbackup=False,
        )

    ncol = 3
    ####################################################
    # Check if required columns are present for residual
    ####################################################
    if datacolumn == "residual" or datacolumn == "RESIDUAL":
        modelcolumn_present = check_datacolumn_valid(msname, datacolumn="MODEL_DATA")
        corcolumn_present = check_datacolumn_valid(msname, datacolumn="CORRECTED_DATA")
        if modelcolumn_present == False or corcolumn_present == False:
            datacolumn = "corrected"
    elif datacolumn == "RESIDUAL_DATA":
        modelcolumn_present = check_datacolumn_valid(msname, datacolumn="MODEL_DATA")
        datacolumn_present = check_datacolumn_valid(msname, datacolumn="DATA")
        if modelcolumn_present == False or datacolumn_present == False:
            datacolumn = "corrected"
    ########################################################
    # Whenther memory is sufficient for calculating residual
    ########################################################
    if (
        datacolumn == "residual"
        or datacolumn == "RESIDUAL"
        or datacolumn == "RESIDUAL_DATA"
    ):
        colsize = get_column_size(msname)
        if memory_limit > (3 * colsize):
            ncol = 4
        else:
            print(
                "Total available memory for this job is not sufficient to calculate residual."
            )
            if datacolumn == "residual":
                datacolumn = "corrected"
            else:
                datacolumn = "data"
    #################################################
    # Whether corrected data column is present or not
    #################################################
    if datacolumn == "corrected" or datacolumn == "CORRECTED_DATA":
        corcolumn_present = check_datacolumn_valid(msname, datacolumn="CORRECTED_DATA")
        if corcolumn_present == False:
            print(
                "Corrected data column is chosen for flagging, but it is not present."
            )
            return
    ##############
    # Tfcrop flag
    ##############
    if use_tfcrop:
        time_chunk, baseline_chunk = get_chunk_size(
            msname, memory_limit=memory_limit, ncol=ncol
        )
        if baseline_chunk == None or time_chunk == None:
            print("Memory limit is too small to work. Do not flagging.")

            return
        baseline_name_list = baseline_names(msname)
        if baseline_chunk > len(baseline_name_list):
            flagdata(
                vis=msname,
                mode="tfcrop",
                ntime="scan",
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
            )

        else:
            baseline_blocks = [
                ";".join(str(x) for x in baseline_name_list[i : i + baseline_chunk])
                for i in range(0, len(baseline_name_list), baseline_chunk)
            ]
            for ant_str in baseline_blocks:
                flagdata(
                    vis=msname,
                    mode="tfcrop",
                    ntime="scan",
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
                    antenna=ant_str,
                )

    #############
    # Rflag flag
    #############
    if use_rflag:
        time_chunk, baseline_chunk = get_chunk_size(
            msname, memory_limit=memory_limit, ncol=ncol
        )
        if baseline_chunk == None or time_chunk == None:
            print("Memory limit is too small to work. Do not flagging.")

            return
        baseline_name_list = baseline_names(msname)
        if baseline_chunk > len(baseline_name_list):
            flagdata(
                vis=msname,
                mode="rflag",
                ntime="scan",
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
            )

        else:
            baseline_blocks = [
                ";".join(str(x) for x in baseline_name_list[i : i + baseline_chunk])
                for i in range(0, len(baseline_name_list), baseline_chunk)
            ]
            for ant_str in baseline_blocks:
                flagdata(
                    vis=msname,
                    mode="rflag",
                    ntime="scan",
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
                    antenna=ant_str,
                )

    ##############
    # Extend flag
    ##############
    if use_tfcrop or use_rflag:
        time_chunk, baseline_chunk = get_chunk_size(
            msname, memory_limit=memory_limit, ncol=ncol
        )
        if baseline_chunk == None or time_chunk == None:
            print("Memory limit is too small to work. Do not flagging.")

            return
        baseline_name_list = baseline_names(msname)
        if baseline_chunk > len(baseline_name_list):
            flagdata(
                vis=msname,
                mode="extend",
                datacolumn="data",
                clipzeros=True,
                ntime="scan",
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
            )

        else:
            baseline_blocks = [
                ";".join(str(x) for x in baseline_name_list[i : i + baseline_chunk])
                for i in range(0, len(baseline_name_list), baseline_chunk)
            ]
            for ant_str in baseline_blocks:
                flagdata(
                    vis=msname,
                    mode="extend",
                    datacolumn=datacolumn,
                    clipzeros=True,
                    ntime="scan",
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
                    antenna=ant_str,
                )

    return


def do_flagging(
    msname,
    datacolumn="DATA",
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
        msname = msname.rstrip("/")
        mspath = os.path.dirname(os.path.abspath(msname))
        os.chdir(mspath)
        print("###########################")
        print("Flagging measurement set : ", msname)
        print("###########################\n")
        fluxcal_field, fluxcal_scans = get_fluxcals(msname)
        badspw = get_bad_chans(msname)
        bad_ants, bad_ants_str = get_bad_ants(msname, fieldnames=fluxcal_field)
        ###########################
        # Dask local cluster setup
        ##########################
        if os.path.exists(msname + "/SUBMSS"):
            subms_list = glob.glob(msname + "/SUBMSS/*")
        else:
            subms_list = [msname]
        task = delayed(single_ms_flag)(dry_run=True)
        mem_limit = run_limited_memory_task(task, dask_dir=mspath)
        dask_client, dask_cluster, n_jobs, n_threads = get_dask_client(
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


def main():
    usage = "Initial flagging of calibrator data"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--msname",
        dest="msname",
        default=None,
        help="Name of measurement set",
        metavar="Measurement Set",
    )
    parser.add_option(
        "--datacolumn",
        dest="datacolumn",
        default="DATA",
        help="Name of the datacolumn",
        metavar="String",
    )
    parser.add_option(
        "--print_casalog",
        dest="print_casalog",
        default=False,
        help="Print CASA log",
        metavar="Boolean",
    )
    parser.add_option(
        "--use_tfcrop",
        dest="use_tfcrop",
        default=False,
        help="Use tfcrop",
        metavar="Boolean",
    )
    parser.add_option(
        "--use_rflag",
        dest="use_rflag",
        default=False,
        help="Use rflag",
        metavar="Boolean",
    )
    parser.add_option(
        "--flagdimension",
        dest="flagdimension",
        default="freqtime",
        help="Flag dimension",
        metavar="String",
    )
    parser.add_option(
        "--flag_autocorr",
        dest="flag_autocorr",
        default=True,
        help="Flag autpo-correlations or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--flagbackup",
        dest="flagbackup",
        default=True,
        help="Flag backup",
        metavar="Boolean",
    )
    parser.add_option(
        "--cpu_frac",
        dest="cpu_frac",
        default=0.8,
        help="CPU fraction to use",
        metavar="Float",
    )
    parser.add_option(
        "--mem_frac",
        dest="mem_frac",
        default=0.8,
        help="Memory fraction to use",
        metavar="Float",
    )
    (options, args) = parser.parse_args()
    if eval(str(options.print_casalog)) == True:
        casalog.showconsole(True)
    if options.msname != None and os.path.exists(options.msname):
        try:
            result = do_flagging(
                options.msname,
                datacolumn=options.datacolumn,
                use_tfcrop=eval(str(options.use_tfcrop)),
                use_rflag=eval(str(options.use_rflag)),
                flagdimension=options.flagdimension,
                flag_autocorr=eval(str(options.flag_autocorr)),
                flag_backup=eval(str(options.flagbackup)),
                cpu_frac=float(options.cpu_frac),
                mem_frac=float(options.mem_frac),
            )
            return result
        except Exception as e:
            traceback.print_exc()
            return 1
    else:
        print("Please provide correct measurement set.\n")
        return 1


if __name__ == "__main__":
    result = main()
    print(f"Final msg : {result}")
    print("\n###################\nBasic flagging is finished.\n###################\n")
    os._exit(result)
