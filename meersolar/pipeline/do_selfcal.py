import logging
import resource
import psutil
import dask
import numpy as np
import argparse
import traceback
import time
import sys
import os
from casatasks import casalog
from casatools import msmetadata, table
from dask import delayed, compute
from functools import partial
from meersolar.utils.basic_utils import get_datadir, suppress_casa_output
from meersolar.utils.resource_utils import drop_cache, limit_threads
from meersolar.utils.logger_utils import init_logger, clean_shutdown, SmartDefaultsHelpFormatter, create_logger
from meersolar.utils.proc_manage_utils import run_limited_memory_task, get_dask_client, save_pid
from meersolar.utils.ms_metadata import check_datacolumn_valid
from meersolar.utils.flagging import get_unflagged_antennas
from meersolar.utils.imaging import calc_field_of_view, calc_cellsize, calc_sun_dia
from meersolar.utils.udocker_utils import check_udocker_container
from meersolar.utils.selfcal_utils import intensity_selfcal
logging.getLogger("distributed").setLevel(logging.WARNING)

try:
    logfile = casalog.logfile()
    os.system("rm -rf " + logfile)
except BaseException:
    pass
    
datadir = get_datadir()

def do_selfcal(
    msname="",
    workdir="",
    selfcaldir="",
    start_threshold=5,
    end_threshold=3,
    max_iter=100,
    max_DR=1000,
    min_iter=2,
    DR_convegerence_frac=0.3,
    uvrange="",
    minuv=0,
    solint="60s",
    weight="briggs",
    robust=0.0,
    do_apcal=True,
    min_tol_factor=1.0,
    applymode="calonly",
    solar_selfcal=True,
    ncpu=-1,
    mem=-1,
    dry_run=False,
    logfile="selfcal.log",
):
    """
    Do selfcal iterations and use convergence rules to stop

    Parameters
    ----------
    msname : str
        Name of the measurement set
    workdir : str
        Work directory
    selfcaldir : str
        Working directory
    start_threshold : int, optional
        Start CLEAN threhold
    end_threshold : int, optional
        End CLEAN threshold
    max_iter : int, optional
        Maximum numbers of selfcal iterations
    max_DR : float, optional
        Maximum dynamic range
    min_iter : int, optional
        Minimum numbers of seflcal iterations at different stages
    DR_convegerence_frac : float, optional
        Dynamic range fractional change to consider as converged
    uvrange : str, optional
        UV-range for calibration
    minuv : float, optionial
        Minimum UV-lambda to use in imaging
    solint : str, optional
        Solutions interval
    weight : str, optional
        Imaging weighting
    robust : float, optional
        Briggs weighting robust parameter (-1 to 1)
    do_apcal : bool, optional
        Perform ap-selfcal or not
    min_tol_factor : float, optional
         Minimum tolerable variation in temporal direction in percentage
    applymode : str, optional
        Solution apply mode
    solar_selfcal : bool, optional
        Whether is is solar selfcal or not
    ncpu : int, optional
        Number of CPU threads to use
    mem : float, optional
        Memory in GB to use
    logfile : str, optional
        Log file name

    Returns
    -------
    int
        Success message
    str
        Final caltable
    """
    limit_threads(n_threads=ncpu)
    from casatasks import split, flagdata, initweights, flagmanager

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    sub_observer = None
    logger, logfile = create_logger(
        os.path.basename(logfile).split(".log")[0], logfile, verbose=False
    )
    if os.path.exists(f"{workdir}/jobname_password.npy") and logfile is not None:
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            sub_observer = init_logger(
                "remotelogger_selfcal_{os.path.basename(msname).split('.ms')[0]}",
                logfile,
                jobname=jobname,
                password=password,
            )
    try:
        msname = os.path.abspath(msname.rstrip("/"))
        selfcaldir = selfcaldir.rstrip("/")
        os.makedirs(selfcaldir, exist_ok=True)

        os.chdir(selfcaldir)
        selfcalms = selfcaldir + "/selfcal_" + os.path.basename(msname)
        if os.path.exists(selfcalms):
            os.system("rm -rf " + selfcalms)
        if os.path.exists(selfcalms + ".flagversions"):
            os.system("rm -rf " + selfcalms + ".flagversions")

        ##############################
        # Restoring any previous flags
        ##############################
        with suppress_casa_output():
            flags = flagmanager(vis=msname, mode="list")
        keys = flags.keys()
        for k in keys:
            if k == "MS":
                pass
            else:
                version = flags[0]["name"]
                try:
                    with suppress_casa_output():
                        flagmanager(vis=msname, mode="restore", versionname=version)
                        flagmanager(vis=msname, mode="delete", versionname=version)
                except BaseException:
                    pass
        if os.path.exists(msname + ".flagversions"):
            os.system("rm -rf " + msname + ".flagversions")

        ##############################
        # Spliting corrected data
        ##############################
        hascor = check_datacolumn_valid(msname, datacolumn="CORRECTED_DATA")
        msmd = msmetadata()
        msmd.open(msname)
        scan = int(msmd.scannumbers()[0])
        field = int(msmd.fieldsforscan(scan)[0])
        freqMHz=msmd.meanfreq(0,unit="MHz")
        msmd.close()
        if hascor:
            logger.info(f"Spliting corrected data to ms : {selfcalms}")
            with suppress_casa_output():
                split(
                    vis=msname,
                    field=str(field),
                    scan=str(scan),
                    outputvis=selfcalms,
                    datacolumn="corrected",
                )
        else:
            logger.info(f"Spliting data to ms : {selfcalms}")
            with suppress_casa_output():
                split(
                    vis=msname,
                    field=str(field),
                    scan=str(scan),
                    outputvis=selfcalms,
                    datacolumn="data",
                )
        msname = selfcalms

        ##########################################
        # Initiate proper weighting
        ##########################################
        with suppress_casa_output():
            flagdata(
                vis=msname,
                mode="clip",
                clipzeros=True,
                datacolumn="data",
                flagbackup=False,
            )
        logger.info("Initiating weights ....")
        with suppress_casa_output():
            initweights(vis=msname, wtmode="ones", dowtsp=True)

        ############################################
        # Imaging and calibration parameters
        ############################################
        logger.info(f"Estimating imaging Parameters ...")
        cellsize = calc_cellsize(msname, 3)
        instrument_fov = calc_field_of_view(msname, FWHM=False)
        sun_size=calc_sun_dia(freqMHz)
        fov = min(instrument_fov, 2*sun_size*60) # 2 times sun size at that frequency  
        imsize = int(fov / cellsize)
        pow2 = np.ceil(np.log2(imsize)).astype("int")
        possible_sizes = []
        for p in range(pow2):
            for k in [3, 5]:
                possible_sizes.append(k * 2**p)
        possible_sizes = np.sort(np.array(possible_sizes))
        possible_sizes = possible_sizes[possible_sizes >= imsize]
        imsize = max(1024, int(possible_sizes[0]))
        unflagged_antenna_names, flag_frac_list = get_unflagged_antennas(msname)
        refant = unflagged_antenna_names[0]
       
        ############################################
        # Initiating selfcal Parameters
        ############################################
        logger.info(f"Estimating self-calibration parameters...")
        DR1 = 0.0
        DR2 = 0.0
        DR3 = 0.0
        RMS1 = -1.0
        RMS2 = -1.0
        RMS3 = -1.0
        num_iter = 0
        num_iter_after_ap = 0
        num_iter_fixed_sigma = 0
        last_sigma_DR1 = 0
        sigma_reduced_count = 0
        calmode = "p"
        threshold = start_threshold
        last_round_gaintable = ""
        use_previous_model = False
        os.system("rm -rf *_selfcal_present*")
        ###########################################
        # Starting selfcal loops
        ##########################################
        while True:
            ##################################
            # Selfcal round parameters
            ##################################
            logger.info("######################################")
            logger.info(
                f"Selfcal iteration : "
                + str(num_iter)
                + ", Threshold: "
                + str(threshold)
                + ", Calibration mode: "
                + str(calmode)
            )
            msg, gaintable, dyn, rms, final_image, final_model, final_residual = (
                intensity_selfcal(
                    msname,
                    logger,
                    selfcaldir,
                    cellsize,
                    imsize,
                    round_number=num_iter,
                    uvrange=uvrange,
                    minuv=minuv,
                    calmode=calmode,
                    solint=solint,
                    refant=str(refant),
                    applymode=applymode,
                    min_tol_factor=min_tol_factor,
                    threshold=threshold,
                    use_previous_model=use_previous_model,
                    weight=weight,
                    robust=robust,
                    use_solar_mask=solar_selfcal,
                    ncpu=ncpu,
                    mem=round(mem, 2),
                )
            )
            if msg == 1:
                if num_iter == 0:
                    logger.info(
                        f"No model flux is picked up in first round. Trying with lowest threshold.\n"
                    )
                    (
                        msg,
                        gaintable,
                        dyn,
                        rms,
                        final_image,
                        final_model,
                        final_residual,
                    ) = intensity_selfcal(
                        msname,
                        logger,
                        selfcaldir,
                        cellsize,
                        imsize,
                        round_number=num_iter,
                        uvrange=uvrange,
                        minuv=minuv,
                        calmode=calmode,
                        solint=solint,
                        refant=str(refant),
                        applymode=applymode,
                        min_tol_factor=min_tol_factor,
                        threshold=end_threshold,
                        use_previous_model=False,
                        weight=weight,
                        robust=robust,
                        use_solar_mask=solar_selfcal,
                        ncpu=ncpu,
                        mem=round(mem, 2),
                    )
                    if msg == 1:
                        os.system("rm -rf *_selfcal_present*")
                        time.sleep(5)
                        clean_shutdown(sub_observer)
                        return msg, []
                    else:
                        threshold = end_threshold
                else:
                    os.system("rm -rf *_selfcal_present*")
                    return msg, []
            if msg == 2:
                os.system("rm -rf *_selfcal_present*")
                time.sleep(5)
                clean_shutdown(sub_observer)
                return msg, []
            if num_iter == 0:
                DR1 = DR3 = DR2 = dyn
                RMS1 = RMS2 = RMS3 = rms
            elif num_iter == 1:
                DR3 = dyn
                RMS2 = RMS1
                RMS1 = rms
            else:
                DR1 = DR2
                DR2 = DR3
                DR3 = dyn
                RMS3 = RMS2
                RMS2 = RMS1
                RMS1 = rms
            logger.info(
                f"RMS based dynamic ranges: "
                + str(DR1)
                + ","
                + str(DR2)
                + ","
                + str(DR3)
            )
            logger.info(
                f"RMS of the images: " + str(RMS1) + "," + str(RMS2) + "," + str(RMS3)
            )
            if DR3 > 0.9 * DR2:
                use_previous_model = True
            else:
                use_previous_model = False

            #########################################################
            # If DR is decreasing (DR decrease in phase-only selfcal)
            #########################################################
            if (
                (DR3 < 0.85 * DR2 and DR3 < 0.9 * DR1 and DR2 > DR1)
                and calmode == "p"
                and num_iter > min_iter
            ):
                logger.info(f"Dynamic range decreasing in phase-only self-cal.")
                if do_apcal:
                    logger.info(f"Changed calmode to 'ap'.")
                    calmode = "ap"
                    if threshold > end_threshold and num_iter_fixed_sigma > min_iter:
                        threshold -= 1
                        sigma_reduced_count += 1
                        num_iter_fixed_sigma = 0
                else:
                    os.system("rm -rf *_selfcal_present*")
                    time.sleep(5)
                    clean_shutdown(sub_observer)
                    return 0, last_round_gaintable
            
            ##############################################################
            # If DR is decreasing (DR decrease in amplitude-phase selfcal)
            ##############################################################        
            if (
                (DR3 < 0.9 * DR2 and DR2 > 1.5 * DR1)
                and calmode == "ap"
                and num_iter_after_ap > min_iter
            ):
                logger.info(
                    f"Dynamic range is decreasing after minimum numbers of 'ap' rounds.\n"
                )
                os.system("rm -rf *_selfcal_present*")
                time.sleep(5)
                clean_shutdown(sub_observer)
                return 0, last_round_gaintable
                
            ###########################
            # If maximum DR has reached
            ###########################
            if DR3 >= max_DR and num_iter_after_ap > min_iter:
                logger.info(f"Maximum dynamic range is reached.\n")
                os.system("rm -rf *_selfcal_present*")
                time.sleep(5)
                clean_shutdown(sub_observer)
                return 0, gaintable
                
            ##########################
		    # If DR suddenly decreased
		    ##########################
            if DR3<0.7*DR2 and do_apcal==True and sigma_reduced_count>1:
                logger.info(
                    f"Dynamic range dropped suddenly. Using last round caltable as final.\n"
                )
                os.system("rm -rf *_selfcal_present*")
                time.sleep(5)
                clean_shutdown(sub_observer)
                return 0, last_round_gaintable      
            ###########################
            # Checking DR convergence
            ###########################
            # Condition 1 
            # (If DR did not increase after one round of sigma reduction, do not reduce sigma further and exit)
            ###########################
            elif (
                ((do_apcal and calmode == "ap") or do_apcal == False)
                and num_iter_fixed_sigma > min_iter
                and (
                    last_sigma_DR1 > 0
                    and abs(round(np.nanmedian([DR1, DR2, DR3]), 0) - last_sigma_DR1)
                    / last_sigma_DR1
                    < DR_convegerence_frac
                )
                and sigma_reduced_count > 1
            ):
                if threshold > end_threshold:
                    logger.info(
                        f"DR does not increase over last two changes in threshold, but minimum threshold has not reached yet.\n"
                    )
                    logger.info(
                        f"Starting final self-calibration rounds with threshold = "
                        + str(end_threshold)
                        + "sigma...\n"
                    )
                    threshold = end_threshold
                    sigma_reduced_count += 1
                    num_iter_fixed_sigma = 0
                    continue
                else:
                    logger.info(
                        f"Selfcal converged. DR does not increase over last two changes in threshold.\n"
                    )
                    os.system("rm -rf *_selfcal_present*")
                    time.sleep(5)
                    clean_shutdown(sub_observer)
                    return 0, gaintable                    
            else:
                ########################################
                # Condition 2
                # If DR does not increase a certain percentage
                ########################################
                if (
                    abs(DR1 - DR2) / DR2 < DR_convegerence_frac
                    and num_iter > min_iter
                    and threshold == end_threshold + 1
                ):
                    #####################################
                    # Change from phase only selfcal to amplitude-phase selfcal
                    #####################################
                    if do_apcal and calmode == "p":
                        logger.info(
                            f"Dynamic range converged. Changing calmode to 'ap'.\n"
                        )
                        calmode = "ap"
                        if num_iter_fixed_sigma > min_iter:
                            threshold -= 1
                            sigma_reduced_count += 1
                            num_iter_fixed_sigma = 0
                    ######################################
                    # Converged if already in apcal
                    ######################################
                    elif (
                        do_apcal and num_iter_after_ap > min_iter
                    ) or do_apcal == False:
                        logger.info(f"Self-calibration has converged.\n")
                        os.system("rm -rf *_selfcal_present*")
                        time.sleep(5)
                        clean_shutdown(sub_observer)
                        return 0, gaintable
                ######################################
                # Condition 3
                # Reducing threshold if not converged
                ######################################
                elif (
                    abs(DR1 - DR2) / DR2 < DR_convegerence_frac
                    and threshold > end_threshold
                    and num_iter_fixed_sigma > min_iter
                ):
                    threshold -= 1
                    logger.info(f"Reducing threshold to : " + str(threshold))
                    sigma_reduced_count += 1
                    num_iter_fixed_sigma = 0
                    if last_sigma_DR1 > 0:
                        last_sigma_DR1 = round(np.nanmean([DR1, DR2, DR3]), 0)
                    else:
                        last_sigma_DR1 = round(np.nanmean([DR1, DR2, DR3]), 0)
                #########################################
                # In apcal and maximum iteration has reached
                #########################################
                elif (do_apcal == False or (do_apcal and calmode == "ap")) and num_iter>min_iter and num_iter == max_iter:
                    logger.info(
                        f"Self-calibration is finished. Maximum iteration is reached.\n"
                    )
                    os.system("rm -rf *_selfcal_present*")
                    time.sleep(5)
                    clean_shutdown(sub_observer)
                    return 0, gaintable
            num_iter += 1
            last_round_gaintable = gaintable
            if calmode == "ap":
                num_iter_after_ap += 1
            num_iter_fixed_sigma += 1
    except Exception as e:
        traceback.print_exc()
        os.system("rm -rf *_selfcal_present*")
        time.sleep(5)
        clean_shutdown(sub_observer)
        return 1, []
    finally:
        time.sleep(5)
        drop_cache(msname)


def main():
    starttime = time.time()
    parser = argparse.ArgumentParser(
        description="Self-calibration", formatter_class=SmartDefaultsHelpFormatter
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mslist",
        type=str,
        help="Comma-separated list of measurement sets (required positional argument)",
    )
    basic_args.add_argument(
        "--workdir",
        type=str,
        default="",
        required=True,
        help="Working directory",
    )
    basic_args.add_argument(
        "--caldir",
        type=str,
        default="",
        required=True,
        help="Caltable directory",
    )

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced calibration and imaging parameters\n###################"
    )
    adv_args.add_argument(
        "--start_thresh",
        type=float,
        default=5,
        help="Starting CLEANing threshold",
        metavar="Float",
    )
    adv_args.add_argument(
        "--stop_thresh",
        type=float,
        default=3,
        help="Stop CLEANing threshold",
        metavar="Float",
    )
    adv_args.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum number of selfcal iterations",
        metavar="Integer",
    )
    adv_args.add_argument(
        "--max_DR",
        type=float,
        default=1000,
        help="Maximum dynamic range",
        metavar="Float",
    )
    adv_args.add_argument(
        "--min_iter",
        type=int,
        default=2,
        help="Minimum number of selfcal iterations",
        metavar="Integer",
    )
    adv_args.add_argument(
        "--conv_frac",
        type=float,
        default=0.3,
        help="Fractional change in DR to determine convergence",
        metavar="Float",
    )
    adv_args.add_argument("--solint", type=str, default="60s", help="Solution interval")
    adv_args.add_argument(
        "--uvrange",
        type=str,
        default="",
        help="Calibration UV-range (CASA format)",
    )
    adv_args.add_argument(
        "--minuv",
        type=float,
        default=0,
        help="Minimum UV-lambda used for imaging",
        metavar="Float",
    )
    adv_args.add_argument("--weight", type=str, default="briggs", help="Imaging weight")
    adv_args.add_argument(
        "--robust",
        type=float,
        default=0.0,
        help="Robust parameter for briggs weight",
        metavar="Float",
    )
    adv_args.add_argument(
        "--applymode",
        type=str,
        default="calonly",
        help="Solution apply mode",
        metavar="String",
    )
    adv_args.add_argument(
        "--min_tol_factor",
        type=float,
        default=1.0,
        help="Minimum tolerable variation in temporal direction in percentage",
        metavar="Float",
    )
    adv_args.add_argument(
        "--no_apcal",
        action="store_false",
        dest="do_apcal",
        help="Do not perform ap-selfcal",
    )
    adv_args.add_argument(
        "--no_solar_selfcal",
        action="store_false",
        dest="solar_selfcal",
        help="Do not perform solar self-calibration",
    )
    adv_args.add_argument(
        "--keep_backup",
        action="store_true",
        help="Keep backup of self-calibration rounds",
    )

    # Resource management parameters
    hard_args = parser.add_argument_group(
        "###################\nHardware resource management parameters\n###################"
    )
    hard_args.add_argument(
        "--cpu_frac",
        type=float,
        default=0.8,
        help="CPU fraction to use",
        metavar="Float",
    )
    hard_args.add_argument(
        "--mem_frac",
        type=float,
        default=0.8,
        help="Memory fraction to use",
        metavar="Float",
    )
    hard_args.add_argument("--jobid", type=int, default=0, help="Job ID")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    pid = os.getpid()
    save_pid(pid, datadir + f"/pids/pids_{args.jobid}.txt")
    
    mslist = str(args.mslist).split(",")
    if args.workdir == "" or os.path.exists(args.workdir) == False:
        workdir = os.path.dirname(os.path.abspath(mslist[0])) + "/workdir"
    else:
        workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)

    if args.caldir == "" or not os.path.exists(args.caldir):
        caldir = f"{workdir}/caltables"
    else:
        caldir = args.caldir
    os.makedirs(caldir, exist_ok=True)

    os.makedirs(workdir + "/logs", exist_ok=True)
    mainlog_file = workdir + "/logs/selfcal_targets.mainlog"
    mainlogger, mainlog_file = create_logger(
        os.path.basename(mainlog_file).split(".mainlog")[0], mainlog_file, verbose=False
    )
    observer = None
    if os.path.exists(f"{workdir}/jobname_password.npy") and mainlog_file is not None:
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(mainlog_file):
            observer = init_logger(
                "all_selfcal", mainlog_file, jobname=jobname, password=password
            )
    ###########################
    # WSClean container
    ###########################
    container_name = "meerwsclean"
    container_present = check_udocker_container(container_name)
    if not container_present:
        container_name = initialize_wsclean_container(name=container_name)
        if container_name is None:
            print(
                "Container {container_name} is not initiated. First initiate container and then run."
            )
            return 1
    try:
        if len(mslist) == 0:
            mainlogger.info("Please provide at-least one measurement set.")
            msg = 1
        else:
            task = delayed(do_selfcal)(dry_run=True)
            mem_limit = run_limited_memory_task(task, dask_dir=args.workdir)
            partial_do_selfcal = partial(
                do_selfcal,
                start_threshold=float(args.start_thresh),
                end_threshold=float(args.stop_thresh),
                max_iter=int(args.max_iter),
                max_DR=float(args.max_DR),
                min_iter=int(args.min_iter),
                DR_convegerence_frac=float(args.conv_frac),
                uvrange=str(args.uvrange),
                minuv=float(args.minuv),
                solint=str(args.solint),
                weight=str(args.weight),
                robust=float(args.robust),
                do_apcal=args.do_apcal,
                applymode=args.applymode,
                min_tol_factor=float(args.min_tol_factor),
                solar_selfcal=args.solar_selfcal,
            )

            ####################################
            # Filtering any corrupted ms
            #####################################
            filtered_mslist = []  # Filtering in case any ms is corrupted
            for ms in mslist:
                checkcol = check_datacolumn_valid(ms)
                if checkcol:
                    filtered_mslist.append(ms)
                else:
                    mainlogger.info(f"Issue in : {ms}")
                    os.system("rm -rf {ms}")
            mslist = filtered_mslist

            chanlist = []
            for ms in mslist:
                channame = (
                    os.path.basename(ms)
                    .split(".ms")[0]
                    .split("spw_")[-1]
                    .split("_time")[0]
                )
                if channame not in chanlist:
                    chanlist.append(channame)

            available_mem = psutil.virtual_memory().available / 1024**3
            if (mem_limit / 0.6) < 4 and available_mem > 4:
                min_mem_per_job = 4
            else:
                min_mem_per_job = mem_limit / 0.6

            ######################################
            # Resetting maximum file limit
            ######################################
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            new_soft_limit = max(soft_limit, int(0.8 * hard_limit))
            if soft_limit < new_soft_limit:
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))

            num_fd_list = []
            for ms in mslist:
                msmd = msmetadata()
                msmd.open(ms)
                times = msmd.timesforspws(0)
                timeres = np.diff(times)
                pos = np.where(timeres > 3 * np.nanmedian(timeres))[0]
                max_intervals = min(1, len(pos))
                freqs = msmd.chanfreqs(0, unit="MHz")
                freqres = np.diff(freqs)
                pos = np.where(freqres > 3 * np.nanmedian(freqres))[0]
                max_nchan = min(1, len(pos))
                msmd.close()
                per_job_fd = (
                    (max_nchan + 1) * max_intervals * 4 * 2
                )  # 4 types of images, 2 is fudge factor
                num_fd_list.append(per_job_fd)
            total_fd = max(num_fd_list) * len(mslist)
            n_jobs = max(1, int(new_soft_limit / total_fd))
            n_jobs = min(len(mslist), n_jobs)

            dask_client, dask_cluster, n_jobs, n_threads, mem_limit = get_dask_client(
                n_jobs,
                dask_dir=args.workdir,
                cpu_frac=float(args.cpu_frac),
                mem_frac=float(args.mem_frac),
                min_cpu_per_job=3,
                min_mem_per_job=min_mem_per_job,
            )
            tasks = []
            for ms in mslist:
                logfile = (
                    args.workdir
                    + "/logs/"
                    + os.path.basename(ms).split(".ms")[0]
                    + "_selfcal.log"
                )
                mainlogger.info(f"MS name: {ms}, Log file: {logfile}\n")
                tasks.append(
                    delayed(partial_do_selfcal)(
                        ms,
                        args.workdir,
                        args.workdir
                        + "/"
                        + os.path.basename(ms).split(".ms")[0]
                        + "_selfcal",
                        ncpu=n_threads,
                        mem=mem_limit,
                        logfile=logfile,
                    )
                )
            results = compute(*tasks)
            dask_client.close()
            dask_cluster.close()
            gcal_list = []
            for i in range(len(results)):
                r = results[i]
                msg = r[0]
                if msg != 0:
                    mainlogger.info(
                        f"Self-calibration was not successful for ms: {mslist[i]}."
                    )
                else:
                    gcal = r[1]
                    tb = table()
                    tb.open(gcal)
                    scan = np.unique(tb.getcol("SCAN_NUMBER"))[0]
                    tb.close()
                    final_gain_caltable = caldir + f"/selfcal_scan_{scan}.gcal"
                    os.system(f"cp -r {gcal} {final_gain_caltable}")
                    gcal_list.append(final_gain_caltable)
            if not args.keep_backup:
                for ms in mslist:
                    selfcaldir = (
                        args.workdir
                        + "/"
                        + os.path.basename(ms).split(".ms")[0]
                        + "_selfcal"
                    )
                    os.system("rm -rf " + selfcaldir)
            if len(gcal_list) > 0:
                mainlogger.info(f"Final selfcal caltables: {gcal_list}")
                mainlogger.info("################################################")
                mainlogger.info(f"Total time taken: {round(time.time()-starttime,2)}s")
                mainlogger.info("################################################")
                msg = 0
            else:
                mainlogger.info("No self-calibration is successful.")
                mainlogger.info("################################################")
                mainlogger.info(f"Total time taken: {round(time.time()-starttime,2)}s")
                mainlogger.info("################################################")
                msg = 1
    except Exception as e:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        for ms in mslist:
            drop_cache(ms)
        drop_cache(workdir)
        clean_shutdown(observer)
    return msg


if __name__ == "__main__":
    result = main()
    if result > 0:
        result = 1
    print("\n###################\nSelf-calibration is done.\n###################\n")
    os._exit(result)
