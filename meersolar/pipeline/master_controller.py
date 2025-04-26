import os, glob, psutil, copy, time, traceback
from meersolar.pipeline.basic_func import *
from casatasks import split

from casatasks import casalog

logfile = casalog.logfile()
os.system("rm -rf " + logfile)


def run_flag(msname, workdir, flag_calibrators=True, cpu_frac=0.8, mem_frac=0.8):
    """
    Run flagging jobs
    Parameters
    ----------
    msname: str
        Name of the measurement set
    workdir : str
        Working directory
    flag_calibrators : bool, optional
        Flag calibrator fields
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    Returns
    -------
    int
        Success message
    """
    print("###########################")
    print("Flagging ....")
    print("###########################\n")
    ########################
    # Calibrator ms flagging
    ########################
    msname = msname.rstrip("/")
    if flag_calibrators:
        flagfield_type = "cal"
        flagging_cmd = (
            "run_flag --msname "
            + msname
            + " --datacolumn DATA --use_tfcrop True --flagbackup True"
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
        )
    else:
        flagfield_type = "target"
        flagging_cmd = (
            "run_flag --msname "
            + msname
            + " --datacolumn DATA --use_tfcrop True --flagdimension freq --flagbackup True"
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
        )
    basename = f"flagging_{flagfield_type}_" + os.path.basename(msname).split(".ms")[0]
    if os.path.isdir(workdir + "/logs") == False:
        os.makedirs(workdir + "/logs")
    batch_file = create_batch_script_nonhpc(flagging_cmd, workdir, basename)
    print(flagging_cmd + "\n")
    os.system("bash " + batch_file)
    print("Waiting to finish flagging...\n")
    while True:
        finished_file = glob.glob(workdir + "/.Finished_" + basename + "*")
        if len(finished_file) > 0:
            break
        else:
            time.sleep(1)
    success_index = int(finished_file[0].split("_")[-1])
    if success_index == 0:
        print("Initial flagging is done successfully.\n")
        return 0
    else:
        print("Initial flagging is not done successfully.\n")
        return 1


def run_import_model(msname, workdir, cpu_frac=0.8, mem_frac=0.8):
    """
    Importing calibrator models
    Parameters
    ----------
    msname: str
        Name of the measurement set
    workdir : str
        Working directory
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    Returns
    -------
    int
        Success message
    """
    print("###########################")
    print("Importing model of calibrators....")
    print("###########################\n")
    msname = msname.rstrip("/")
    import_model_cmd = (
        "import_model --msname "
        + msname
        + " --workdir "
        + str(workdir)
        + " --cpu_frac "
        + str(cpu_frac)
        + " --mem_frac "
        + str(mem_frac)
    )
    basename = "modeling_" + os.path.basename(msname).split(".ms")[0]
    if os.path.isdir(workdir + "/logs") == False:
        os.makedirs(workdir + "/logs")
    batch_file = create_batch_script_nonhpc(import_model_cmd, workdir, basename)
    print(import_model_cmd + "\n")
    os.system("bash " + batch_file)
    print("Waiting to finish visibility simulation for calibrators...\n")
    while True:
        finished_file = glob.glob(workdir + "/.Finished_" + basename + "*")
        if len(finished_file) > 0:
            break
        else:
            time.sleep(1)
    success_index = int(finished_file[0].split("_")[-1])
    if success_index == 0:
        print("Visibility simulation for calibrator fields are done successfully.\n")
        return 0
    else:
        print(
            "Visibility simulation for calibrator fields are not done successfully.\n"
        )
        return 1


def run_basic_cal(msname, workdir, cpu_frac=0.8, mem_frac=0.8):
    """
    Perform basic calibration
    Parameters
    ----------
    msname: str
        Name of the measurement set
    workdir : str
        Working directory
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    Returns
    -------
    int
        Success message for basic calibration
    """
    print("###########################")
    print("Performing basic calibration .....")
    print("###########################\n")
    msname = msname.rstrip("/")
    cal_basename = "basic_cal"
    basic_cal_cmd = (
        "run_basic_cal --msname "
        + msname
        + " --workdir "
        + workdir
        + " --cpu_frac "
        + str(cpu_frac)
        + " --mem_frac "
        + str(mem_frac)
    )
    if os.path.isdir(workdir + "/logs") == False:
        os.makedirs(workdir + "/logs")
    batch_file = create_batch_script_nonhpc(basic_cal_cmd, workdir, cal_basename)
    print(basic_cal_cmd + "\n")
    os.system("bash " + batch_file)
    print("Waiting to finish calibration...\n")
    while True:
        finished_file = glob.glob(workdir + "/.Finished_" + cal_basename + "*")
        if len(finished_file) > 0:
            break
        else:
            time.sleep(1)
    success_index_cal = int(finished_file[0].split("_")[-1])
    if success_index_cal == 0:
        print("Basic calibration is done successfully.\n")
    else:
        print("Basic calibration is unsuccessful.\n")
    return success_index_cal


def run_noise_diode_cal(msname, workdir, cpu_frac=0.8, mem_frac=0.8):
    """
    Perform noise diode based flux calibration
    Parameters
    ----------
    msname: str
        Name of the measurement set
    workdir : str
        Working directory
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    Returns
    -------
    int
        Success message for noise diode based flux calibration
    """
    try:
        print("###########################")
        print("Performing noise diode based flux calibration .....")
        print("###########################\n")
        msname = msname.rstrip("/")
        noisecal_basename = "noise_cal"
        noise_cal_cmd = (
            "run_fluxcal --msname "
            + msname
            + " --workdir "
            + workdir
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
        )
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        batch_file = create_batch_script_nonhpc(
            noise_cal_cmd, workdir, noisecal_basename
        )
        print(noise_cal_cmd + "\n")
        os.system("bash " + batch_file)
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1


def run_partion(msname, workdir, partition_cal=True, cpu_frac=0.8, mem_frac=0.8):
    """
    Perform basic calibration
    Parameters
    ----------
    msname: str
        Name of the measurement set
    workdir : str
        Working directory
    partition_cal : bool, optional
        Partition calibrators
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    Returns
    -------
    int
        Success message
    """
    print("###########################")
    print("Partitioning measurement set ...")
    print("###########################\n")
    msname = msname.rstrip("/")
    msmd = msmetadata()
    msmd.open(msname)
    nchan = msmd.nchan(0)
    times = msmd.timesforfield(0)
    msmd.close()
    if len(times) == 1:
        timeres = msmd.exposuretime(scan)["value"]
    else:
        timeres = times[1] - times[0]
    if nchan > 1024:
        width = int(nchan / 1024)
        if width < 1:
            width = 1
    else:
        width = 1
    if timeres < 8:
        timebin = "8s"
    else:
        timebin = ""
    target_scans, cal_scans, f_scans, g_scans, p_scans = get_cal_target_scans(msname)
    if partition_cal:
        cal_scans_copy = copy.deepcopy(cal_scans)
        for s in cal_scans:
            noise_cal_scan = determine_noise_diode_cal_scan(msname, s)
            if noise_cal_scan:
                print(f"Removing noise-diode scan: {s}")
                cal_scans_copy.remove(s)
        cal_scans = copy.deepcopy(cal_scans_copy)
        cal_scans = ",".join([str(s) for s in cal_scans])
        calibrator_ms = workdir + "/calibrator.ms"
        split_cmd = f"run_partition --msname {msname} --outputms {calibrator_ms} --scans {cal_scans} --timebin {timebin} --width {width} --cpu_frac {cpu_frac} --mem_frac {mem_frac}"
        partition_field = "cal"
    else:
        target_scans = ",".join([str(s) for s in target_scans])
        target_ms = workdir + "/target.ms"
        split_cmd = f"run_partition --msname {msname} --outputms {target_ms} --scans {target_scans} --timebin {timebin} --width {width} --cpu_frac {cpu_frac} --mem_frac {mem_frac}"
        partition_field = "target"
    ####################################
    # Partition fields
    ####################################
    print("\n########################################")
    basename = (
        f"partition_{partition_field}_" + os.path.basename(msname).split(".ms")[0]
    )
    if os.path.isdir(workdir + "/logs") == False:
        os.makedirs(workdir + "/logs")
    batch_file = create_batch_script_nonhpc(split_cmd, workdir, basename)
    print(split_cmd + "\n")
    os.system("bash " + batch_file)
    print("Waiting to finish partitioning...\n")
    while True:
        finished_file = glob.glob(workdir + "/.Finished_" + basename + "*")
        if len(finished_file) > 0:
            break
        else:
            time.sleep(1)
    print("Partitioning is finished.\n")
    if partition_cal:
        if os.path.exists(calibrator_ms):
            print(f"Calibrator ms: {calibrator_ms}")
            return 0
        else:
            print(f"Calibrator fields could not be partitioned.")
            return 1
    else:
        if os.path.exists(target_ms):
            print(f"Target ms: {target_ms}")
            return 0
        else:
            print(f"Target fields could not be partitioned.")
            return 1


def run_target_split(
    msname,
    workdir,
    datacolumn="data",
    timeres=-1,
    freqres=-1,
    target_freq_chunk=-1,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Apply calibration solutions and split corrected target scans
    Parameters
    ----------
    msname: str
        Name of the measurement set
    workdir : str
        Working directory
    datacolumn : str, optional
        Data column
    timeres : float, optional
        Time bin to average in seconds
    freqres : float, optional
        Frequency averaging in MHz
    target_freq_chunk : float, optional
        Target frequency chunk in MHz
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    Returns
    -------
    int
        Success message for spliting target scans
    """
    try:
        print("###########################")
        print("spliting target scans .....")
        print("###########################\n")
        msname = msname.rstrip("/")
        if target_freq_chunk > 0:
            msmd = msmetadata()
            msmd.open(msname)
            chanres = msmd.chanres(0, unit="MHz")[0]
            msmd.close()
            spectral_chunk = int(target_freq_chunk / chanres)
            spectral_chunk = max(1, spectral_chunk)
        else:
            spectral_chunk = 1
        split_basename = "split_targets"
        split_cmd = (
            "run_target_split --msname "
            + msname
            + " --workdir "
            + workdir
            + " --datacolumn "
            + datacolumn
            + " --freqres "
            + str(freqres)
            + " --timeres "
            + str(timeres)
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
            + " --nchan_chunck "
            + str(spectral_chunk)
        )
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        batch_file = create_batch_script_nonhpc(split_cmd, workdir, split_basename)
        print(split_cmd + "\n")
        os.system("bash " + batch_file)
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1

def run_applycal_sol(
    target_mslist,
    workdir,
    caldir,
    use_only_bandpass=False,
    use_only_fluxcal=False,
    overwrite_datacolumn=False,
    apply_selfcal=False,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Apply calibration solutions on splited target scans
    Parameters
    ----------
    target_mslist: list
        Target measurement set list
    workdir : str
        Working directory
    caldir : str
        Caltable directory
    use_only_bandpass : bool
        Use only bandpass solutions
    use_only_fluxcal : bool
        Use only fluxcal solutions
    apply_selfcal : bool, optional
        Applying self-calibration solutions or not
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    overwrite_datacolumn : bool
        Overwrite data column or not
    Returns
    -------
    int
        Success message for applying calibration solutions and spliting target scans
    """
    try:
        if apply_selfcal:
            print("###########################")
            print("Applying basic and self-calibration solutions on target scans .....")
            print("###########################\n")
            applycal_basename = "apply_selfcal"
            use_only_bandpass=False
            use_only_fluxcal=False
        else:
            print("###########################")
            print("Applying basic calibration solutions on target scans .....")
            print("###########################\n")
            applycal_basename = "apply_basiccal"
        applycal_cmd = (
            "run_applycal --mslist "
            + ",".join(target_mslist)
            + " --workdir "
            + workdir
            + " --caldir "
            + caldir
            + " --use_only_bandpass "
            + str(use_only_bandpass)
            + " --use_only_fluxcal "
            + str(use_only_fluxcal)
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
            + " --overwrite_datacolumn "
            + str(overwrite_datacolumn)
            + " --include_selfcal "
            + str(apply_selfcal)
        )
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        batch_file = create_batch_script_nonhpc(
            applycal_cmd, workdir, applycal_basename
        )
        print(applycal_cmd + "\n")
        os.system("bash " + batch_file)
        print("Waiting to finish applycal...\n")
        while True:
            finished_file = glob.glob(workdir + "/.Finished_" + applycal_basename + "*")
            if len(finished_file) > 0:
                break
            else:
                time.sleep(1)
        success_index_cal = int(finished_file[0].split("_")[-1])
        if success_index_cal == 0:
            print("Applying calibration is done successfully.\n")
        else:
            print("Applying calibration is unsuccessful.\n")
        return success_index_cal
    except Exception as e:
        traceback.print_exc()
        return 1

def run_selfcal(
    mslist,
    workdir,
    cpu_frac=0.8,
    mem_frac=0.8,
    start_thresh=-1,
    stop_thresh=-1,
    max_iter=-1,
    max_DR=-1,
    min_iter=-1,
    conv_frac=-1,
    solint="",
    do_apcal="",
    solar_selfcal="",
    keep_backup="",
    uvrange="",
    minuv=-1,
    weight="",
    robust="",
    applymode="",
    gaintype="",
    min_fractional_bw=-1,
):
    """
    Self-calibration on target scans
    Parameters
    ----------
    mslist: list
        Target measurement set list
    workdir : str
        Working directory
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
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
    conv_frac : float, optional
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
    min_fractional_bw : float, optional
        Minimum fractional bandwidth of spectral chunk in percentage
    applymode : str, optional
        Solution apply mode
    gaintype : str, optional
        Gaintype, G or T
    solar_selfcal : bool, optional
        Whether is is solar selfcal or not
    Returns
    -------
    int
        Success message for self-calibration 
    """
    try:
        print("###########################")
        print("Performing self-calibration of target scans .....")
        print("###########################\n")
        selfcal_basename = "selfcal_targets"
        selfcal_cmd = "run_selfcal --mslist "+",".join(mslist)+" --workdir "+workdir+" --cpu_frac "+str(cpu_frac)+" --mem_frac "+str(mem_frac)
        if start_thresh>0:
            selfcal_cmd+=" --start_thresh "+str(start_thresh)
            if stop_thresh>0 and stop_thresh<start_thresh:
                selfcal_cmd+=" --stop_threshold "+str(stop_thresh)
        if max_iter>0:
            selfcal_cmd+=" --max_iter "+str(max_iter)
        if max_DR>0:
            selfcal_cmd+=" --max_DR "+str(max_DR)
        if min_iter>0:
            selfcal_cmd+=" --min_iter "+str(min_iter)
        if conv_frac>0:
            selfcal_cmd+=" --conv_frac "+str(conv_frac)
        if solint!="":
            selfcal_cmd+=" --solint "+solint
        if do_apcal!="":
            selfcal_cmd+=" --do_apcal "+str(do_apcal)
        if solar_selfcal!="":
            selfcal_cmd+=" --solar_selfcal "+str(solar_selfcal)
        if keep_backup!="":
            selfcal_cmd+=" --keep_backup "+str(keep_backup)
        if uvrange!="":
            selfcal_cmd+=" --uvrange "+str(uvrange)
        if minuv>0:
            selfcal_cmd+=" --minuv "+str(minuv)
        if weight!="":
            selfcal_cmd+=" --weight "+str(weight)
        if robust!="":
             selfcal_cmd+=" --robust "+str(robust)
        if applymode!="":
            selfcal_cmd+=" --applymode "+str(applymode)
        if gaintype!="":
            selfcal_cmd+=" --gaintype "+str(gaintype)
        if min_fractional_bw>0:
            selfcal_cmd+=" --min_fractional_bw "+str(min_fractional_bw)
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        batch_file = create_batch_script_nonhpc(
            selfcal_cmd, workdir, selfcal_basename
        )
        print(selfcal_cmd + "\n")
        os.system("bash " + batch_file)
        print("Waiting to finish self-calibration...\n")
        while True:
            finished_file = glob.glob(workdir + "/.Finished_" + selfcal_basename + "*")
            if len(finished_file) > 0:
                break
            else:
                time.sleep(1)
        success_index_cal = int(finished_file[0].split("_")[-1])
        if success_index_cal == 0:
            print("Self-calibration is done successfully.\n")
        else:
            print("Self-calibration is unsuccessful.\n")
        return success_index_cal
    except Exception as e:
        traceback.print_exc()
        return 1
        
def check_status(workdir, basename):
    """
    Check job status
    Parameters
    ----------
    workdir : str
        Work directory
    basename : str
        Basename
    Returns
    -------
    int
        Success code
    """
    finished_file = glob.glob(workdir + "/.Finished_" + basename + "*")
    if len(finished_file) > 0:
        success_index_split_target = int(finished_file[0].split("_")[-1])
        if success_index_split_target == 0:
            return 0
        else:
            return 1
    else:
        return 1


def master_control(
    msname,
    workdir,
    per_job_cpu=3,
    per_job_mem=2,
    do_reset_weight_flag=True,
    do_partition=True,
    do_flag=True,
    do_target_split=True,
    do_import_model=True,
    do_basic_cal=True,
    do_noise_cal=True,
    do_applycal=True,
    do_selfcal=True,
    do_apply_selfcal=True,
    solint=15,
    time_window=5,
    do_imaging=True,
    weight="briggs",
    robust=0.0,
    freqavg=-1,
    timeavg=-1,
    image_freqres=-1,
    image_timeres=-1,
    target_freq_chunk=-1,
    cpu_frac=0.8,
    mem_frac=0.8,
    n_nodes=1,
    verbose=False,
):
    print("###########################")
    print("Starting the pipeline .....")
    print("###########################\n")
    msname = msname.rstrip("/")
    ###################################
    # Preparing working directories
    ###################################
    if workdir == "":
        workdir = os.path.dirname(os.path.abspath(msname)) + "/workdir"
    if workdir[-1] == "/":
        workdir = workdir[:-1]
    if os.path.exists(workdir) == False:
        os.makedirs(workdir)
    os.chdir(workdir)
    caldir = workdir + "/caltables"
    selfcaldir = workdir + "/selfcaltables"
    frac_compute_use = 1.0  # Fraction of total allocated compute resource to use

    ##########################################################
    # Determining maximum allowed time and frequency averaging
    ##########################################################
    if freqavg > 0:
        max_freqres = calc_bw_smearing_freqwidth(msname)
        if freqavg > max_freqres:
            freqavg = round(max_freqres, 2)
    if timeavg > 0:
        max_timeres = min(
            calc_time_smearing_timewidth(msname), max_time_solar_smearing(msname)
        )
        if timeavg > max_timeres:
            timeavg = round(max_timeres, 2)

    #############################
    # Reset any previous weights
    ############################
    if do_reset_weight_flag:
        cpu_usage = psutil.cpu_percent(interval=1)  # Average over 1 second
        total_cpus = psutil.cpu_count(logical=True)
        available_cpus = int(total_cpus * (1 - cpu_usage / 100.0))
        available_cpus = max(1, available_cpus)  # Avoid zero workers
        reset_weights_and_flags(msname, n_threads=available_cpus)

    ##############################
    # Run partitioning jobs
    ##############################
    calibrator_msname = workdir + "/calibrator.ms"
    target_msname = workdir + "/target.ms"
    if do_partition or os.path.exists(workdir + "/calibrator.ms") == False:
        msg = run_partion(
            msname,
            workdir,
            cpu_frac=round(frac_compute_use * cpu_frac, 2),
            mem_frac=round(frac_compute_use * mem_frac, 2),
        )
        if msg != 0:
            print("!!!! WARNING: Error in partitioning calibrator fields. !!!!")
            return 1

    #########################################
    # Spliting target scans
    #########################################
    split_use_frac=1.0
    if do_target_split:
        if (do_flag==False and do_import_model==False and do_basic_cal==False) or do_noise_cal==False:
            pass
        else:
            split_use_frac=0.2
        msg = run_target_split(
            msname,
            workdir,
            datacolumn="data",
            timeres=timeavg,
            freqres=freqavg,
            target_freq_chunk=target_freq_chunk,
            cpu_frac=round(split_use_frac * cpu_frac, 2),
            mem_frac=round(split_use_frac * mem_frac, 2),
        )
        if split_use_frac==0.2:
            frac_compute_use = frac_compute_use - 0.2
        if msg != 0:
            print("!!!! WARNING: Error in running spliting target scans. !!!!")

    ##################################
    # Run flagging jobs on calibrators
    ##################################
    if do_flag:
        msg = run_flag(
            calibrator_msname,
            workdir,
            flag_calibrators=True,
            cpu_frac=round(frac_compute_use * cpu_frac, 2),
            mem_frac=round(frac_compute_use * mem_frac, 2),
        )
        if msg != 0:
            print("!!!! WARNING: Flagging error. !!!!")
            return 1
    if split_use_frac==0.2 and do_target_split and check_status(workdir, "split_targets") == 0:
        frac_compute_use += 0.2

    #################################
    # Import model
    #################################
    if do_import_model:
        fluxcal_fields, fluxcal_scans = get_fluxcals(calibrator_msname)
        phasecal_fields, phasecal_scans, phasecal_fluxes = get_phasecals(
            calibrator_msname
        )
        calibrator_field = fluxcal_fields + phasecal_fields
        msg = run_import_model(
            calibrator_msname,
            workdir,
            cpu_frac=round(frac_compute_use * cpu_frac, 2),
            mem_frac=round(frac_compute_use * mem_frac, 2),
        )  # Run model import
        if msg != 0:
            print(
                "!!!! WARNING: Error in importing calibrator models. Not continuing calibration. !!!!"
            )
            return 1
    if split_use_frac==0.2 and do_target_split and check_status(workdir, "split_targets") == 0:
        frac_compute_use += 0.2

    ########################################
    # Run noise-diode based flux calibration
    ########################################
    if do_noise_cal:
        frac_compute_use -= 0.2
        msg = run_noise_diode_cal(
            msname,
            workdir,
            cpu_frac=round(0.2 * cpu_frac, 2),
            mem_frac=round(0.2 * mem_frac, 2),
        )  # Run noise diode based flux calibration
        if msg != 0:
            print(
                "!!!! WARNING: Error in running noise-diode based flux calibration. Not continuing further. !!!!"
            )
            return 1
    if split_use_frac==0.2 and do_target_split and check_status(workdir, "split_targets") == 0:
        frac_compute_use += 0.2
    if do_noise_cal and check_status(workdir, "noise_cal") == 0:
        frac_compute_use += 0.2

    ###############################
    # Run basic calibration
    ###############################
    use_only_bandpass = False
    use_only_fluxcal = False
    if do_basic_cal:
        msg = run_basic_cal(
            calibrator_msname,
            workdir,
            cpu_frac=round(frac_compute_use * cpu_frac, 2),
            mem_frac=round(frac_compute_use * mem_frac, 2),
        )  # Run basic calibration
        if msg != 0:
            print(
                "!!!! WARNING: Error in basic calibration. Not continuing further. !!!!"
            )
            return 1
    else:
        if len(glob.glob(caldir + "/*.bcal")) == 0:
            print(f"No bandpass table is present in calibration directory : {caldir}.")
            return 1
        if (
            len(glob.glob(caldir + "/*.gcal")) == 0
            and len(glob.glob(caldir + "/*.fcal")) == 0
        ):
            print(
                f"No time-dependent gaintable is present in calibration directory : {caldir}. Applying only bandpass solutions."
            )
            use_only_bandpass = True
        if (
            len(glob.glob(caldir + "/*.gcal")) != 0
            and len(glob.glob(caldir + "/*.fcal")) == 0
        ):
            print(
                f"No time-dependent fluxscaled gaintable is present in calibration directory : {caldir}. Applying solutions only from fluxcal."
            )
            use_only_fluxcal = True

    #######################################
    # Check noise diode cal finished or not
    #######################################
    if do_noise_cal:
        print("Waiting to finish noise-diode based flux calibration...\n")
        noisecal_basename = "noise_cal"
        while True:
            finished_file = glob.glob(workdir + "/.Finished_" + noisecal_basename + "*")
            if len(finished_file) > 0:
                break
            else:
                time.sleep(1)
        success_index_noisecal = int(finished_file[0].split("_")[-1])
        if success_index_noisecal == 0:
            print("Noise-diode based flux-calibration is done successfully.\n")
            frac_compute_use += 0.2
        else:
            print(
                "!!!! WARNING: Error in noise-diode based flux calibration. Not continuing further. !!!!"
            )
            return 1

    #############################################
    # Check spliting target scans finished or not
    #############################################
    if do_target_split:
        print("Waiting to finish spliting of target scans...\n")
        split_basename = "split_targets"
        while True:
            finished_file = glob.glob(workdir + "/.Finished_" + split_basename + "*")
            if len(finished_file) > 0:
                break
            else:
                time.sleep(1)
        success_index_split_target = int(finished_file[0].split("_")[-1])
        if success_index_split_target == 0:
            print("Spliting target scans are done successfully.\n")
            if split_use_frac==0.2:
                frac_compute_use += 0.2
        else:
            print(
                "!!!! WARNING: Error in spliting target scans. Not continuing further. !!!!"
            )
            return 1

    target_mslist = glob.glob(workdir + "/target_scan*.ms")
    if len(target_mslist) == 0:
        print("No splited target scan ms are available in work directory.")
        return 1
    print(f"Target scan mslist : {[os.path.basename(i) for i in target_mslist]}")

    ########################################
    # Performing self-calibration
    ########################################
    if do_selfcal:
        target_mslist=sorted(target_mslist)
        selfcal_mslist = []
        spw_list=[]
        for ms in target_mslist:
            ms_spw=os.path.basename(ms).rstrip(".ms").split('_')[-1]
            if ms_spw not in spw_list:
                spw_list.append(ms_spw)
        chosen_spw=spw_list[int(len(spw_list)/2)]
        for ms in target_mslist:
            if chosen_spw in os.path.basename(ms):
                selfcal_mslist.append(ms)
        #########################################################
        # Applying solutions on target scans for self-calibration
        #########################################################
        if len(selfcal_mslist) > 0:
            caldir = workdir + "/caltables"
            msg = run_applycal_sol(
                selfcal_mslist,
                workdir,
                caldir,
                use_only_bandpass=use_only_bandpass,
                use_only_fluxcal=use_only_fluxcal,
                overwrite_datacolumn=False,
                apply_selfcal=False,
                cpu_frac=round(frac_compute_use * cpu_frac, 2),
                mem_frac=round(frac_compute_use * mem_frac, 2),
            )
            if msg != 0:
                print("!!!! WARNING: Error in applying solutions on target scans. Not continuing further. !!!!")
                return 1
            msg=run_selfcal(selfcal_mslist,workdir,cpu_frac=round(frac_compute_use*cpu_frac,2),mem_frac=round(frac_compute_use*mem_frac,2))
            if msg != 0:
                print("!!!! WARNING: Error in self-calibration on target scans. Not applying self-calibration. !!!!")
                do_apply_selfcal=False
        else:
            print ("!!!! WARNING: No measurement set is present for self-calibration. !!!!")
            do_apply_selfcal=False
            
    ########################################
    # Apply self-calibration
    ########################################
    if do_apply_selfcal:
        target_mslist=sorted(target_mslist)
        caldir = workdir + "/caltables"
        msg = run_applycal_sol(
            target_mslist,
            workdir,
            caldir,
            overwrite_datacolumn=False,
            apply_selfcal=True,
            cpu_frac=round(frac_compute_use * cpu_frac, 2),
            mem_frac=round(frac_compute_use * mem_frac, 2),
        )
        if msg != 0:
            print("!!!! WARNING: Error in applying self-calibration solutions on target scans. !!!!")
            
    #####################################
    # Imaging
    ######################################
    if do_imaging:
        print("Start imaging .....\n")
        ####################
        # Imaging parameters
        ####################
        imagedir = workdir + "/images"
        if os.path.exists(imagedir) == False:
            os.makedirs(imagedir)
        cellsize = calc_cellsize(msname, 3)
        imsize = int(7200 / cellsize)
        pow2 = round(np.log2(imsize / 10.0), 0)
        imsize = int((2**pow2) * 10)
        multiscale_scales = [
            str(s) for s in calc_multiscale_scales(msname, 3, max_scale=5)
        ]
        caldir = workdir + "/caltables"
        final_caltables = glob.glob(caldir + "/*")
        calibrator_caltables = []
        selfcaltables = []
        for gtable in final_caltables:
            if "selfcal" not in gtable:
                calibrator_caltables.append(gtable)
            else:
                selfcaltables.append(gtable)
        print("Basic calibration tables: " + ",".join(calibrator_caltables) + "\n")
        selfcaltables_mean_chan = []
        for selfcal_table in selfcaltables:
            start_chan = int(
                os.path.basename(selfcal_table).split("_selfcal")[0].split("_")[-2]
            )
            end_chan = int(
                os.path.basename(selfcal_table).split("_selfcal")[0].split("_")[-1]
            )
            selfcaltables_mean_chan.append(int((start_chan + end_chan) / 2))
        selfcaltables = np.array(selfcaltables)
        selfcaltables_mean_chan = np.array(selfcaltables_mean_chan)
        # Spectral channels
        msmd = msmetadata()
        msmd.open(msname)
        total_chan = msmd.nchan(0)
        total_bw = msmd.bandwidths(0) / 10**6  # In MHz
        ms_freqres = msmd.chanres(0)[0] / 10**6  # In MHz
        ms_timeres = msmd.exposuretime(scan=1)["value"]  # In second
        msmd.close()
        if image_freqres < ms_freqres:
            print(
                "Intended image spectral resolution is smaller than spectral resolution of the data. Hence, setting it to spectral resolution of the data.\n"
            )
            image_freqres = ms_freqres
        if image_timeres < ms_timeres:
            print(
                "Intended image temporal resolution is smaller than temporal resolution of the data. Hence, setting it to temporal resolution of the data.\n"
            )
            image_timeres = ms_timeres
        bad_chans = get_bad_chans(msname).split("0:")[-1]
        bad_chans = bad_chans.split(";")
        bad_chan_list = []
        for i in bad_chans:
            start_chan = int(i.split("~")[0])
            end_chan = int(i.split("~")[-1])
            for chan in range(start_chan, end_chan):
                bad_chan_list.append(chan)
        # Target scans and timeranges
        target_scans, cal_scans, f_scans, g_scans, p_scans = get_cal_target_scans(
            msname
        )
        max_chunk = 4096
        total_cpu = psutil.cpu_count()
        total_mem = psutil.virtual_memory().total / 1024**3
        max_jobs = min(int(total_cpu / 4), int(total_mem / 2))
        print("Maximum number of imaging jobs at a time: ", max_jobs)
        os.system("rm -rf " + workdir + "/.Finished_imaging*")
        job_count = 0
        finished_file_count = len(glob.glob(workdir + "/.Finished_imaging*"))
        # target_scans=[13]
        for scan in target_scans:
            final_imagedir = imagedir + "/scan_" + str(scan)
            if os.path.isdir(final_imagedir) == False:
                os.makedirs(final_imagedir)
            msmd.open(msname)
            times = msmd.timesforscan(int(scan))
            msmd.close()
            if image_timeres > 0:
                ntime = int((times[-1] - times[0]) / image_timeres)
            else:
                ntime = 1
            if image_freqres > 0:
                nchan = int(total_bw / image_freqres)
            else:
                nchan = 1
            ##################################
            # Determining chan and time chunks
            ##################################
            if ntime > max_chunk:
                ntime = max_chunk
                nchan = 1
            elif ntime * nchan > max_chunk:
                nchan = int(max_chunk / ntime)
                if nchan > max_chunk:
                    nchan = max_chunk
            ####################################
            # Loop over time and frequency
            ####################################
            for t in range(0, len(times), ntime):
                start_index = t
                end_index = t + ntime
                if end_index > len(times) - 1:
                    end_index = len(times) - 1
                timerange = (
                    mjdsec_to_timestamp(times[start_index], str_format=1)
                    + "~"
                    + mjdsec_to_timestamp(times[end_index], str_format=1)
                )
                for chan in range(0, total_chan, nchan):
                    start_chan = chan
                    end_chan = chan + nchan
                    if end_chan > total_chan:
                        end_chan = total_chan
                    good_chans = []
                    for c in range(start_chan, end_chan):
                        if c not in bad_chan_list:
                            good_chans.append(c)
                    if len(good_chans) == 0:
                        pass
                    else:
                        spw = str(min(good_chans)) + "~" + str(max(good_chans))
                        mean_chan = int((min(good_chans) + max(good_chans)) / 2)
                        pos = np.argmin(np.abs(selfcaltables_mean_chan - mean_chan))
                        selfcal_table = selfcaltables[pos]
                        print(
                            "Imaging for channel range: "
                            + str(start_chan)
                            + "~"
                            + str(end_chan)
                            + " and time range: "
                            + timerange
                            + ", scan: "
                            + str(scan)
                        )
                        imaging_args = [
                            "--msname " + msname,
                            "--imsize " + str(imsize),
                            "--cellsize " + str(cellsize),
                            "--spw " + spw,
                            "--timerange " + timerange,
                            "--weight " + str(weight),
                            "--imagedir " + final_imagedir,
                            "--calibrator_gaintables " + ",".join(calibrator_caltables),
                            "--selfcaltables " + selfcal_table,
                            "--freqres " + str(image_freqres),
                            "--timeres " + str(image_timeres),
                            "--minuv_l 200",
                            "--multiscale_scales " + ",".join(multiscale_scales),
                        ]
                        if weight == "briggs":
                            imaging_args.append("--robust " + str(robust))
                        ncpu = int((1 - psutil.cpu_percent()) * psutil.cpu_count())
                        mem = psutil.virtual_memory().available / 1024**3
                        imaging_args.append("--ncpu " + str(ncpu))
                        imaging_args.append("--mem " + str(mem))
                        imaging_cmd = "python3 do_imaging.py " + " ".join(imaging_args)
                        spw_str = str(start_chan) + "_" + str(end_chan)
                        time_str = "".join(
                            timerange.split("~")[0]
                            .split("/")[-1]
                            .split(".")[0]
                            .split(":")
                        ) + "".join(
                            timerange.split("~")[-1]
                            .split("/")[-1]
                            .split(".")[0]
                            .split(":")
                        )
                        basename = (
                            "imaging_"
                            + os.path.basename(msname).split(".ms")[0]
                            + "_spw_"
                            + spw_str
                            + "_time_"
                            + time_str
                            + "_scan_"
                            + str(scan)
                        )
                        batch_file = create_batch_script_nonhpc(
                            imaging_cmd, workdir, basename
                        )
                        print(imaging_cmd + "\n")
                        os.system("bash " + batch_file)
                        job_count += 1
                        count = 0
                        print("Number of jobs spawned: ", job_count)
                        while True:
                            if job_count >= max_jobs:
                                count += 1
                                cur_finished_file_count = len(
                                    glob.glob(workdir + "/.Finished_imaging*")
                                )
                                available_jobs = (
                                    cur_finished_file_count - finished_file_count
                                )
                                # print (available_jobs)
                                if available_jobs > 0:
                                    job_count = 0
                                    finished_file_count = cur_finished_file_count
                                    while True:
                                        free_cpu = (
                                            (100 - psutil.cpu_percent(interval=30))
                                            * psutil.cpu_count()
                                            / 100.0
                                        )
                                        free_mem = (
                                            psutil.virtual_memory().available / 1024**3
                                        )
                                        max_jobs = min(
                                            int(free_cpu / 4), int(free_mem / 2)
                                        )
                                        if max_jobs > 0:  # used_cpu<80 and used_mem<80:
                                            """total_cpu=psutil.cpu_count()*used_cpu/100.0)
                                            total_mem=psutil.virtual_memory().total/1024**3
                                            max_jobs=min(int(total_cpu/4),int(total_mem/5))
                                            """
                                            break
                                        elif count == 0:
                                            print(
                                                "Waiting for free CPUs and memory...\n"
                                            )
                                else:
                                    if count == 0:
                                        print(
                                            "Waiting for at-least one job to finish...\n"
                                        )
                                    time.sleep(30)
                            else:
                                break
