import os, glob, psutil, copy, time, traceback, argparse, socket
from multiprocessing import Process, Event
from meersolar.pipeline.basic_func import *
from meersolar.pipeline.init_data import init_meersolar_data
from meersolar.data.sendmail import send_paircars_notification as send_notification
from casatasks import casalog

try:
    logfile = casalog.logfile()
    os.system("rm -rf " + logfile)
except:
    pass


def run_flag(
    msname, workdir, flag_calibrators=True, jobid=0, cpu_frac=0.8, mem_frac=0.8
):
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
    jobid : int, optional
        Job ID
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
            f"run_flag {msname}"
            + " --datacolumn DATA --use_tfcrop"
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
            + " --workdir "
            + str(workdir)
            + " --jobid "
            + str(jobid)
        )
    else:
        flagfield_type = "target"
        flagging_cmd = (
            f"run_flag {msname}"
            + " --datacolumn DATA --use_tfcrop --flagdimension freq"
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
            + " --workdir "
            + str(workdir)
            + " --jobid "
            + str(jobid)
        )
    flag_basename = (
        f"flagging_{flagfield_type}_" + os.path.basename(msname).split(".ms")[0]
    )
    logfile = workdir + "/logs/" + flag_basename + ".log"
    flagging_cmd += f" --logfile {logfile}"
    os.makedirs(workdir + "/logs",exist_ok=True)
    batch_file = create_batch_script_nonhpc(flagging_cmd, workdir, flag_basename)
    print(flagging_cmd + "\n")
    os.system("bash " + batch_file)
    print("Waiting to finish flagging...\n")
    while True:
        finished_file = glob.glob(workdir + "/.Finished_" + flag_basename + "*")
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


def run_import_model(msname, workdir, jobid=0, cpu_frac=0.8, mem_frac=0.8):
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
        f"import_model {msname}"
        + " --workdir "
        + str(workdir)
        + " --cpu_frac "
        + str(cpu_frac)
        + " --mem_frac "
        + str(mem_frac)
        + " --jobid "
        + str(jobid)
    )
    model_basename = "modeling_" + os.path.basename(msname).split(".ms")[0]
    logfile = workdir + "/logs/" + model_basename + ".log"
    import_model_cmd += f" --logfile {logfile}"
    os.makedirs(workdir + "/logs",exist_ok=True)
    batch_file = create_batch_script_nonhpc(import_model_cmd, workdir, model_basename)
    print(import_model_cmd + "\n")
    os.system("bash " + batch_file)
    print("Waiting to finish visibility simulation for calibrators...\n")
    while True:
        finished_file = glob.glob(workdir + "/.Finished_" + model_basename + "*")
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


def run_basic_cal_jobs(
    msname,
    workdir,
    caldir,
    perform_polcal=False,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    keep_backup=False,
):
    """
    Perform basic calibration

    Parameters
    ----------
    msname: str
        Name of the measurement set
    workdir : str
        Working directory
    caldir : str
        Caltable directory
    perform_polcal : bool, optional
        Perform full polarization calibration
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    keep_backup : bool, optional
        Keep backups

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
        f"run_basic_cal {msname}"
        + " --workdir "
        + workdir
        + " --caldir "
        + caldir
        + " --cpu_frac "
        + str(cpu_frac)
        + " --mem_frac "
        + str(mem_frac)
        + " --jobid "
        + str(jobid)
    )
    if perform_polcal:
        basic_cal_cmd += " --perform_polcal"
    if keep_backup:
        basic_cal_cmd += " --keep_backup"
    logfile = workdir + "/logs/" + cal_basename + ".log"
    basic_cal_cmd += f" --logfile {logfile}"
    os.makedirs(workdir + "/logs",exist_ok=True)
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


def run_noise_diode_cal(
    msname, workdir, caldir, keep_backup=False, jobid=0, cpu_frac=0.8, mem_frac=0.8
):
    """
    Perform noise diode based flux calibration

    Parameters
    ----------
    msname: str
        Name of the measurement set
    workdir : str
        Working directory
    caldir : str
        Caltable directory
    keep_backup : bool, optional
        Keep backup
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
            f"run_fluxcal {msname}"
            + " --workdir "
            + workdir
            + " --caldir "
            + caldir
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
            + " --jobid "
            + str(jobid)
        )
        logfile = workdir + "/logs/" + noisecal_basename + ".log"
        noise_cal_cmd += f" --logfile {logfile}"
        os.makedirs(workdir + "/logs",exist_ok=True)
        batch_file = create_batch_script_nonhpc(
            noise_cal_cmd, workdir, noisecal_basename
        )
        print(noise_cal_cmd + "\n")
        os.system("bash " + batch_file)
        print("Waiting to finish noise-diode based flux calibration...\n")
        while True:
            finished_file = glob.glob(workdir + "/.Finished_" + noisecal_basename + "*")
            if len(finished_file) > 0:
                break
            else:
                time.sleep(1)
        success_index_noisecal = int(finished_file[0].split("_")[-1])
        if success_index_noisecal == 0:
            print("Noise-diode based flux-calibration is done successfully.\n")
            return 0
        else:
            return 1
    except Exception as e:
        traceback.print_exc()
        return 1


def run_partion(
    msname, workdir, split_fullpol=False, jobid=0, cpu_frac=0.8, mem_frac=0.8
):
    """
    Perform basic calibration

    Parameters
    ----------
    msname: str
        Name of the measurement set
    workdir : str
        Working directory
    split_fullpol : bool, optional
        Perform full polar split
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
    cal_scans_copy = copy.deepcopy(cal_scans)
    for s in cal_scans:
        noise_cal_scan = determine_noise_diode_cal_scan(msname, s)
        if noise_cal_scan:
            print(f"Removing noise-diode scan: {s}")
            cal_scans_copy.remove(s)
    cal_scans = copy.deepcopy(cal_scans_copy)
    cal_scans = ",".join([str(s) for s in cal_scans])
    calibrator_ms = workdir + "/calibrator.ms"
    split_cmd = f"run_partition {msname} --outputms {calibrator_ms} --scans {cal_scans} --timebin {timebin} --width {width} --cpu_frac {cpu_frac} --mem_frac {mem_frac} --workdir {workdir} --jobid {jobid}"
    if split_fullpol:
        split_cmd += " --split_fullpol"
    ####################################
    # Partition fields
    ####################################
    print("\n########################################")
    partition_basename = f"partition_cal"
    logfile = workdir + "/logs/" + partition_basename + ".log"
    split_cmd += f" --logfile {logfile}"
    os.makedirs(workdir + "/logs",exist_ok=True)
    batch_file = create_batch_script_nonhpc(split_cmd, workdir, partition_basename)
    print(split_cmd + "\n")
    os.system("bash " + batch_file)
    print("Waiting to finish partitioning...\n")
    while True:
        finished_file = glob.glob(workdir + "/.Finished_" + partition_basename + "*")
        if len(finished_file) > 0:
            break
        else:
            time.sleep(1)
    print("Partitioning is finished.\n")
    if os.path.exists(calibrator_ms):
        print(f"Calibrator ms: {calibrator_ms}")
        return 0
    else:
        print(f"Calibrator fields could not be partitioned.")
        return 1


def run_target_split_jobs(
    msname,
    workdir,
    datacolumn="data",
    spw="",
    timeres=-1,
    freqres=-1,
    target_freq_chunk=-1,
    n_spectral_chunk=-1,
    target_scans=[],
    prefix="targets",
    split_fullpol=False,
    merge_spws=False,
    time_window=-1,
    time_interval=-1,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    max_cpu_frac=0.8,
    max_mem_frac=0.8,
):
    """
    Split target scans

    Parameters
    ----------
    msname: str
        Name of the measurement set
    workdir : str
        Working directory
    datacolumn : str, optional
        Data column
    spw : str, optional
        Spectral window to split
    timeres : float, optional
        Time bin to average in seconds
    freqres : float, optional
        Frequency averaging in MHz
    target_freq_chunk : float, optional
        Target frequency chunk in MHz
    n_spectral_chunk : int, optional
        Number of spectral chunks to split
    target_scans : list, optional
        Target scans
    prefix : str, optional
        Prefix of splited targets
    split_fullpol : bool, optional
        Split full polar data or not
    merge_spws : bool, optional
        Merge spectral windows
    time_window : float, optional
        Time window in seconds
    time_interval : float, optional
        Time interval in seconds
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
        split_basename = f"split_{prefix}"
        split_cmd = (
            f"run_target_split {msname}"
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
            + " --spectral_chunk "
            + str(target_freq_chunk)
            + " --max_cpu_frac "
            + str(max_cpu_frac)
            + " --max_mem_frac "
            + str(max_mem_frac)
            + " --n_spectral_chunk "
            + str(n_spectral_chunk)
            + " --prefix "
            + str(prefix)
            + " --time_window "
            + str(time_window)
            + " --time_interval "
            + str(time_interval)
            + " --jobid "
            + str(jobid)
        )
        if split_fullpol:
            split_cmd += " --split_fullpol"
        if merge_spws:
            split_cmd += " --merge_spws"
        if spw != "":
            split_cmd = split_cmd + " --spw " + str(spw)
        if len(target_scans) > 0:
            split_cmd = (
                split_cmd + " --scans " + ",".join([str(s) for s in target_scans])
            )
        logfile = workdir + "/logs/" + split_basename + ".log"
        split_cmd += f" --logfile {logfile}"
        os.makedirs(workdir + "/logs",exist_ok=True)
        batch_file = create_batch_script_nonhpc(split_cmd, workdir, split_basename)
        print(split_cmd + "\n")
        os.system("bash " + batch_file)
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1


def run_solar_siderealcor_jobs(
    mslist,
    workdir,
    prefix="targets",
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    max_cpu_frac=0.8,
    max_mem_frac=0.8,
):
    """
    Apply sidereal motion correction of the Sun

    Parameters
    ----------
    mslist: str
        List of the measurement sets
    workdir : str
        Work directory
    prefix : str, optional
        Measurement set prefix
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    max_cpu_frac : float, optional
        Maximum CPU fraction to use
    max_mem_frac : float, optional
        Maximum memory fraction to use

    Returns
    -------
    int
        Success message
    """
    try:
        print("###########################")
        print("Correcting sidereal motion .....")
        print("###########################\n")
        mslist = ",".join(mslist)
        sidereal_basename = f"cor_sidereal_{prefix}"
        sidereal_cor_cmd = (
            f"run_solar_siderealcor {mslist}"
            + " --workdir "
            + str(workdir)
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
            + " --max_cpu_frac "
            + str(max_cpu_frac)
            + " --max_mem_frac "
            + str(max_mem_frac)
            + " --jobid "
            + str(jobid)
        )
        os.makedirs(workdir + "/logs",exist_ok=True)
        logfile = workdir + "/logs/" + sidereal_basename + ".log"
        sidereal_cor_cmd += f" --logfile {logfile}"
        batch_file = create_batch_script_nonhpc(
            sidereal_cor_cmd, workdir, sidereal_basename
        )
        print(sidereal_cor_cmd + "\n")
        os.system("bash " + batch_file)
        print("Waiting to finish sidereal correction...\n")
        while True:
            finished_file = glob.glob(workdir + "/.Finished_" + sidereal_basename + "*")
            if len(finished_file) > 0:
                break
            else:
                time.sleep(1)
        success_index_cal = int(finished_file[0].split("_")[-1])
        if success_index_cal == 0:
            print("Sidereal motion correction is done successfully.\n")
        else:
            print("Sidereal motion correction is unsuccessful.\n")
        return success_index_cal
    except Exception as e:
        traceback.print_exc()
        return 1


def run_apply_pbcor(
    imagedir,
    workdir,
    apply_parang=True,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Apply primary beam corrections on all images

    Parameters
    ----------
    imagedir: str
        Image directory name
    workdir : str
        Work directory
    apply_parang : bool, optional
        Apply parallactic angle correction
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use

    Returns
    -------
    int
        Success message for applying primary beam correction on all images
    """
    try:
        print("###########################")
        print("Applying primary beam corrections on all images .....")
        print("###########################\n")
        applypbcor_basename = "apply_pbcor"
        applypbcor_cmd = (
            f"run_meerpbcor {imagedir}"
            + " --workdir "
            + str(workdir)
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
            + " --jobid "
            + str(jobid)
        )
        if apply_parang == False:
            applypbcor_cmd += " --no_apply_parang"
        os.makedirs(workdir + "/logs",exist_ok=True)
        logfile = workdir + "/logs/" + applypbcor_basename + ".log"
        applypbcor_cmd += f" --logfile {logfile}"
        batch_file = create_batch_script_nonhpc(
            applypbcor_cmd, workdir, applypbcor_basename
        )
        print(applypbcor_cmd + "\n")
        os.system("bash " + batch_file)
        print("Waiting to finish apply primary beam correction...\n")
        while True:
            finished_file = glob.glob(
                workdir + "/.Finished_" + applypbcor_basename + "*"
            )
            if len(finished_file) > 0:
                break
            else:
                time.sleep(1)
        success_index_cal = int(finished_file[0].split("_")[-1])
        if success_index_cal == 0:
            print("Applying primary beam correction is done successfully.\n")
        else:
            print("Applying primary beam correction is unsuccessful.\n")
        return success_index_cal
    except Exception as e:
        traceback.print_exc()
        return 1


def run_apply_basiccal_sol(
    target_mslist,
    workdir,
    caldir,
    use_only_bandpass=False,
    overwrite_datacolumn=True,
    applymode="calflag",
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Apply basic calibration solutions on splited target scans

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
    applymode : str, optional
        Applycal mode
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
        print("###########################")
        print("Applying basic calibration solutions on target scans .....")
        print("###########################\n")
        applycal_basename = "apply_basiccal"
        mslist = ",".join(target_mslist)
        applycal_cmd = (
            f"run_apply_basiccal {mslist}"
            + " --workdir "
            + workdir
            + " --caldir "
            + caldir
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
            + " --applymode "
            + str(applymode)
            + " --jobid "
            + str(jobid)
        )
        if use_only_bandpass:
            applycal_cmd += " --use_only_bandpass"
        if overwrite_datacolumn:
            applycal_cmd += " --overwrite_datacolumn"
        os.makedirs(workdir + "/logs",exist_ok=True)
        logfile = workdir + "/logs/" + applycal_basename + ".log"
        applycal_cmd += f" --logfile {logfile}"
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
            print("Applying basic calibration is done successfully.\n")
        else:
            print("Applying basic calibration is unsuccessful.\n")
        return success_index_cal
    except Exception as e:
        traceback.print_exc()
        return 1


def run_apply_selfcal_sol(
    target_mslist,
    workdir,
    caldir,
    overwrite_datacolumn=True,
    applymode="calflag",
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Apply self-calibration solutions on splited target scans

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
    applymode : str, optional
        Applycal mode
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
        print("###########################")
        print("Applying self-calibration solutions on target scans .....")
        print("###########################\n")
        applycal_basename = "apply_selfcal"
        mslist = ",".join(target_mslist)
        applycal_cmd = (
            f"run_apply_selfcal {mslist}"
            + " --workdir "
            + workdir
            + " --caldir "
            + caldir
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
            + " --applymode "
            + str(applymode)
            + " --jobid "
            + str(jobid)
        )
        if overwrite_datacolumn:
            applycal_cmd += " --overwrite_datacolumn"
        os.makedirs(workdir + "/logs",exist_ok=True)
        logfile = workdir + "/logs/" + applycal_basename + ".log"
        applycal_cmd += f" --logfile {logfile}"
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
            print("Applying self-calibration is done successfully.\n")
        else:
            print("Applying self-calibration is unsuccessful.\n")
        return success_index_cal
    except Exception as e:
        traceback.print_exc()
        return 1


def run_selfcal_jobs(
    mslist,
    workdir,
    caldir,
    start_thresh=5.0,
    stop_thresh=3.0,
    max_iter=100,
    max_DR=1000,
    min_iter=2,
    conv_frac=0.3,
    solint="60s",
    do_apcal=True,
    solar_selfcal=True,
    keep_backup=False,
    uvrange="",
    minuv=0,
    weight="briggs",
    robust=0.0,
    applymode="calonly",
    min_tol_factor=10.0,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Self-calibration on target scans

    Parameters
    ----------
    mslist: list
        Target measurement set list
    workdir : str
        Working directory
    caldir : str
        Caltable directory
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
    weight : str, optional
        Image weighitng scheme
    robust : float, optional
        Robustness parameter for briggs weighting
    solint : str, optional
        Solutions interval
    do_apcal : bool, optional
        Perform ap-selfcal or not
    min_tol_factor : float, optional
        Minimum tolerance in temporal variation in imaging
    applymode : str, optional
        Solution apply mode
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
        mslist = ",".join(mslist)
        selfcal_cmd = (
            f"run_selfcal {mslist}"
            + " --workdir "
            + workdir
            + " --caldir "
            + caldir
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
            + " --jobid "
            + str(jobid)
        )
        if start_thresh > 0:
            selfcal_cmd += " --start_thresh " + str(start_thresh)
            if stop_thresh > 0 and stop_thresh < start_thresh:
                selfcal_cmd += " --stop_thresh " + str(stop_thresh)
        if max_iter > 0:
            selfcal_cmd += " --max_iter " + str(max_iter)
        if max_DR > 0:
            selfcal_cmd += " --max_DR " + str(max_DR)
        if min_iter > 0:
            selfcal_cmd += " --min_iter " + str(min_iter)
        if conv_frac > 0:
            selfcal_cmd += " --conv_frac " + str(conv_frac)
        if solint != "":
            selfcal_cmd += " --solint " + solint
        if uvrange != "":
            selfcal_cmd += " --uvrange " + str(uvrange)
        if minuv > 0:
            selfcal_cmd += " --minuv " + str(minuv)
        if applymode != "":
            selfcal_cmd += " --applymode " + str(applymode)
        if min_tol_factor > 0:
            selfcal_cmd += " --min_tol_factor " + str(min_tol_factor)
        if weight != "":
            selfcal_cmd += " --weight " + str(weight)
        if robust != "":
            selfcal_cmd += " --robust " + str(robust)
        if do_apcal == False:
            selfcal_cmd += " --no_apcal"
        if solar_selfcal == False:
            selfcal_cmd += " --no_solar_selfcal"
        if keep_backup:
            selfcal_cmd += " --keep_backup"
        os.makedirs(workdir + "/logs",exist_ok=True)
        batch_file = create_batch_script_nonhpc(selfcal_cmd, workdir, selfcal_basename)
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


def run_imaging_jobs(
    mslist,
    workdir,
    outdir,
    freqrange="",
    timerange="",
    minuv=-1,
    weight="briggs",
    robust=0.0,
    pol="IQUV",
    freqres=-1,
    timeres=-1,
    band="",
    threshold=1.0,
    use_multiscale=True,
    use_solar_mask=True,
    cutout_rsun=2.5,
    make_overlay=True,
    savemodel=False,
    saveres=False,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Self-calibration on target scans

    Parameters
    ----------
    mslist: list
        Target measurement set list
    workdir : str
        Working directory
    outdir : str
        Output image directory
    freqrange : str, optional
        Frequency range to image in MHz
    timerange : str, optional
        Time range to image (YYYY/MM/DD/hh:mm:ss~YYYY/MM/DD/hh:mm:ss)
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    minuv : float, optionial
        Minimum UV-lambda to use in imaging
    weight : str, optional
        Imaging weighting
    robust : float, optional
        Briggs weighting robust parameter (-1 to 1)
    pol : str, optional
        Stokes parameters to image
    freqres : float, optional
        Frequency resolution of spectral chunk in MHz (default : -1, no spectral chunking)
    timeres : float, optional
        Time resolution of temporal chunks in MHz (default : -1, no temporal chunking)
    band : str, optional
        Band name
    threshold : float, optional
        CLEAN threshold
    use_multiscale : bool, optional
        Use multiscale or not
    use_solar_mask : bool, optional
        Use solar mask or not
    cutout_rsun : float, optional
        Cutout image size from center in solar radii (default : 2.5 solar radii)
    make_overlay : bool, optional
        Make SUVI MeerKAT overlay
    savemodel : bool, optional
        Save model images or not
    saveres : bool, optional
        Save residual images or not

    Returns
    -------
    int
        Success message for imaging
    """
    try:
        print("###########################")
        print("Performing imaging of target scans .....")
        print("###########################\n")
        imaging_basename = "imaging_targets"
        mslist = ",".join(mslist)
        imaging_cmd = (
            f"run_imaging {mslist}"
            + " --workdir "
            + workdir
            + " --outdir "
            + outdir
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
            + " --pol "
            + str(pol)
            + " --freqres "
            + str(freqres)
            + " --timeres "
            + str(timeres)
            + " --weight "
            + weight
            + " --robust "
            + str(robust)
            + " --minuv_l "
            + str(minuv)
            + " --threshold "
            + str(threshold)
            + " --cutout_rsun "
            + str(cutout_rsun)
            + " --jobid "
            + str(jobid)
        )
        if use_solar_mask == False:
            imaging_cmd += " --no_solar_mask"
        if savemodel == False:
            imaging_cmd += " --no_savemodel"
        if saveres == False:
            imaging_cmd += " --no_saveres"
        if use_multiscale == False:
            imaging_cmd += " --no_multiscale"
        if make_overlay == False:
            imaging_cmd += " --no_make_overlay"
        if freqrange != "":
            imaging_cmd += " --freqrange " + str(freqrange)
        if timerange != "":
            imaging_cmd += " --timerange " + str(timerange)
        if band != "":
            imaging_cmd += " --band " + str(band)
        os.makedirs(workdir + "/logs",exist_ok=True)
        batch_file = create_batch_script_nonhpc(imaging_cmd, workdir, imaging_basename)
        print(imaging_cmd + "\n")
        os.system("bash " + batch_file)
        print("Waiting to finish imaging...\n")
        while True:
            finished_file = glob.glob(workdir + "/.Finished_" + imaging_basename + "*")
            if len(finished_file) > 0:
                break
            else:
                time.sleep(1)
        success_index_cal = int(finished_file[0].split("_")[-1])
        if success_index_cal == 0:
            print("Imaging is done successfully.\n")
        else:
            print("Imaging is unsuccessful.\n")
        return success_index_cal
    except Exception as e:
        traceback.print_exc()
        return 1


def run_ds_jobs(msname, workdir, outdir, target_scans=[], jobid=0, cpu_frac=0.8, mem_frac=0.8):
    """
    Make dynamic spectra of the target scans

    Parameters
    ----------
    msname : str
        Name of the measurement set
    workdir : str
        Name of the work directory
    outdir : str
        Output directory
    target_scans : list, optional
        Target scans
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use

    Returns
    -------
    int
        Success message
    """
    try:
        print("###########################")
        print("Making dynamic spectra of target scans .....")
        print("###########################\n")
        ds_basename = "ds_targets"
        target_scans = " ".join([str(s) for s in target_scans])
        ds_cmd = f"run_makeds {msname} --workdir {workdir} --outdir {outdir} --cpu_frac {cpu_frac} --mem_frac {mem_frac} --jobid {jobid} --target_scans {target_scans}"
        os.makedirs(workdir + "/logs",exist_ok=True)
        logfile = workdir + "/logs/" + ds_basename + ".log"
        ds_cmd += f" --logfile {logfile}"
        batch_file = create_batch_script_nonhpc(ds_cmd, workdir, ds_basename)
        print(ds_cmd + "\n")
        os.system("bash " + batch_file)
        print("Waiting to finish making of dynamic spectra...\n")
        while True:
            finished_file = glob.glob(workdir + "/.Finished_" + ds_basename + "*")
            if len(finished_file) > 0:
                break
            else:
                time.sleep(1)
        success_index_cal = int(finished_file[0].split("_")[-1])
        if success_index_cal == 0:
            print("Dynamic spectra are successfully made.\n")
        else:
            print("Dynamic spectra are not made successfully.\n")
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
    bool
        Finished or not
    """
    finished_file = glob.glob(workdir + "/.Finished_" + basename + "*")
    if len(finished_file) > 0:
        success_index_split_target = int(finished_file[0].split("_")[-1])
        if success_index_split_target == 0:
            return True
        else:
            return False
    else:
        return False


def start_ping_logger(jobid, remote_jobid, waittime, remote_link=""):
    """
    Run ping remote logger in background

    Parameters
    ----------
    jobid : int
        Job ID
    remote_jobid : str
        Remote log job ID
    waittime : float
        Wait time before process ends in hour
    remote_link : str, optional
        Remote logger link

    Returns
    -------
    int
        PID of the background job
    """
    if remote_link != "" and waittime > 0:
        stop_event = Event()
        proc = Process(
            target=ping_logger,
            args=(jobid, remote_jobid, stop_event, remote_link),
            daemon=False,
        )
        proc.start()

        def stop_after_delay():
            time.sleep(waittime * 3600)
            stop_event.set()

        from threading import Thread

        Thread(target=stop_after_delay, daemon=True).start()
        return int(proc.pid)
    else:
        return


def exit_job(start_time, mspath="", workdir=""):
    print(f"Total time taken: {round(time.time()-start_time,2)}s.")
    time.sleep(10)
    gc.collect()
    if mspath != "" and os.path.exists(mspath + "/dask-scratch-space"):
        os.system("rm -rf " + mspath + "/dask-scratch-space " + mspath + "/tmp")
    if workdir != "" and os.path.exists(workdir + "/dask-scratch-space"):
        os.system("rm -rf " + workdir + "/dask-scratch-space " + workdir + "/tmp")


def master_control(
    msname,
    workdir,
    outdir,
    solar_data=True,
    # Pre-calibration
    do_forcereset_weightflag=False,
    do_cal_partition=True,
    do_cal_flag=True,
    do_import_model=True,
    # Basic calibration
    do_basic_cal=True,
    do_noise_cal=True,
    do_applycal=True,
    # Target data preparation
    do_target_split=True,
    target_scans=[],
    freqrange="",
    timerange="",
    uvrange="",
    # Polarization calibration
    do_polcal=False,
    # Self-calibration
    do_selfcal=True,
    do_selfcal_split=True,
    do_apply_selfcal=True,
    do_ap_selfcal=True,
    solar_selfcal=True,
    solint="5min",
    # Sidereal correction
    do_sidereal_cor=False,
    # Dynamic spectra
    make_ds=True,
    # Imaging
    do_imaging=True,
    do_pbcor=True,
    weight="briggs",
    robust=0.0,
    minuv=0,
    image_freqres=-1,
    image_timeres=-1,
    pol="IQUV",
    apply_parang=True,
    clean_threshold=1.0,
    use_multiscale=True,
    use_solar_mask=True,
    cutout_rsun=2.5,
    make_overlay=True,
    # Resource settings
    cpu_frac=0.8,
    mem_frac=0.8,
    n_nodes=0,
    keep_backup=False,
    # Remote logging
    remote_logger=False,
    remote_logger_waittime=0,
):
    """
    Master controller of the entire pipeline

    Parameters
    ----------
    msname : str
        Measurement set name
    workdir : str
        Work directory path
    outdir : str
        Output directory
    solar_data : bool, optional
        Whether it is solar data or not

    do_forcereset_weightflag : bool, optional
        Reset weights and flags of the input ms
    do_cal_partition : bool, optional
        Make calibrator multi-MS
    do_cal_flag : bool, optional
        Perform flagging on calibrator
    do_import_model : bool, optional
        Import model visibilities of flux and polarization calibrators

    do_basic_cal : bool, optional
        Perform basic calibration
    do_noise_cal : bool, optional
        Peform calibration of solar attenuators using noise diode (only used if solar_data=True)
    do_applycal : bool, optional
        Apply basic calibration on target scans

    do_target_split : bool, optional
        Split target scans into chunks
    target_scans : list, optional
        Target scans to self-cal and image
    freqrange : str, optional
        Frequency range to image in MHz (xx1~xx2,xx3~xx4,)
    timerange : str, optional
        Time range to image in YYYY/MM/DD/hh:mm:ss format (tt1~tt2,tt3~tt4,...)
    uvrange : str, optional
        UV-range for calibration

    do_polcal : bool, optional
        Perform full-polarization calibration and imaging

    do_selfcal : bool, optional
        Perform self-calibration
    do_selfcal_split : bool, optional
        Split data after each round of self-calibration
    do_apply_selfcal : bool, optional
        Apply self-calibration solutions
    do_ap_selfcal : bool, optional
        Perform amplitude-phase self-cal or not
    solar_selfcal : bool, optional
        Whether self-calibration is performing on solar observation or not
    solint : str, optional
        Solution intervals in self-cal

    do_sidereal_cor : bool, optional
        Perform solar sidereal motion correction or not

    make_ds : bool, optional
        Make dynamic spectra

    do_imaging : bool, optional
        Perform final imaging
    do_pbcor : bool, optional
        Perform primary beam correction
    weight : str, optional
        Image weighting
    robust : float, optional
        Robust parameter for briggs weighting (-1 to 1)
    minuv : float, optional
        Minimum UV-lambda for final imaging
    image_freqres : float, optional
        Image frequency resolution in MHz (-1 means full bandwidth)
    image_timeres : float, optional
        Image temporal resolution in seconds (-1 means full scan duration)
    pol : str, optional
        Stokes parameters of final imaging
    apply_parang : bool, optional
        Apply parallactic angle correction
    clean_threshold : float, optional
        CLEAN threshold of final imaging
    use_multiscale : bool, optional
        Use multiscale scales or not
    use_solar_mask : bool, optional
        Use solar mask
    cutout_rsun : float, optional
        Cutout image size from center in solar radii (default : 2.5 solar radii)
    make_overlay : bool, optional
        Make SUVI MeerKAT overlay

    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    n_nodes: int, optional
        Number of nodes to use (Only for cluster architecture)
    keep_backup : bool, optional
        Keep backup of self-cal rounds and final models and residual images

    remote_logger : bool, optional
        Enable remote logging of the pipeline status
    remote_logger_waittime : int, optional
        Wait time (in seconds) before connecting to remote logger

    Returns
    -------
    int
        Success message
    """

    init_meersolar_data()
    msname = os.path.abspath(msname.rstrip("/"))
    if os.path.exists(msname) == False:
        print("Please provide a valid measurement set location.\n")
        exit_job(start_time)
        return 1
    mspath = os.path.dirname(msname)
    ###################################
    # Preparing working directories
    ###################################
    if workdir == "":
        workdir = os.path.dirname(os.path.abspath(msname)) + "/workdir"
    workdir=workdir.rstrip("/")
    if outdir=="":
        outdir=workdir
    outdir=outdir.rstrip("/")
    caldir=outdir+"/caltables"  
    caldir=caldir.rstrip("/")  
    os.makedirs(workdir,exist_ok=True)
    os.makedirs(outdir,exist_ok=True)
    os.makedirs(caldir,exist_ok=True)
    if mspath != "" and os.path.exists(mspath + "/dask-scratch-space"):
        os.system("rm -rf " + mspath + "/dask-scratch-space " + mspath + "/tmp")
    if workdir != "" and os.path.exists(workdir + "/dask-scratch-space"):
        os.system("rm -rf " + workdir + "/dask-scratch-space " + workdir + "/tmp")
    try:
        ####################################
        # Job and process IDs
        ####################################
        pid = os.getpid()
        jobid = get_jobid()
        main_job_file = save_main_process_info(
            pid,
            jobid,
            os.path.abspath(msname),
            os.path.abspath(workdir),
            os.path.abspath(outdir),
            cpu_frac,
            mem_frac,
        )
        start_time = time.time()
        print("###########################")
        print(f"MeerSOLAR Job ID: {jobid}")
        print(f"Work directory: {workdir}")
        print(f"Final product directory: {outdir}")
        print("###########################\n")
        #####################################
        # Moving into work directory
        #####################################
        os.chdir(workdir)
        cpu_frac_bkp = copy.deepcopy(cpu_frac)
        mem_frac_bkp = copy.deepcopy(mem_frac)
        if remote_logger:
            remote_link = get_remote_logger_link()
            if remote_link == "":
                print("Please provide a valid remote link.")
                remote_logger = False

        if remote_logger == False:
            emails = get_emails()
            timestamp = dt.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
            if emails != "":
                email_subject = f"MeerSOLAR Logger Details: {timestamp}"

                email_msg = (
                    f"MeerSOLAR user,\n\n"
                    f"MeerSOLAR Job ID: {jobid}\n\n"
                    f"Best,\n"
                    f"MeerSOLAR"
                )
                success_msg, error_msg = send_notification(emails, email_subject, email_msg)
        else:
            ####################################
            # Job name and logging password
            ####################################
            hostname = socket.gethostname()
            timestamp = dt.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
            job_name = f"{hostname} :: {timestamp} :: {os.path.basename(msname).split('.ms')[0]}"
            timestamp1 = dt.utcnow().strftime("%Y%m%dT%H%M%S")
            remote_job_id = (
                f"{hostname}_{timestamp1}_{os.path.basename(msname).split('.ms')[0]}"
            )
            password = generate_password()
            np.save(
                f"{workdir}/jobname_password.npy",
                np.array([job_name, password], dtype="object"),
            )
            print(
                "############################################################################"
            )
            print(remote_link)
            print(f"Job ID: {job_name}")
            print(f"Remote access password: {password}")
            print(
                "#############################################################################"
            )
            emails = get_emails()
            if emails != "":
                email_subject = f"MeerSOLAR Logger Details: {timestamp}"

                email_msg = (
                    f"MeerSOLAR user,\n\n"
                    f"MeerSOLAR Job ID: {jobid}\n\n"
                    f"Remote logger Job ID: {job_name}\n"
                    f"Remote access password: {password}\n\n"
                    f"Best,\n"
                    f"MeerSOLAR"
                )
                success_msg, error_msg = send_notification(emails, email_subject, email_msg)

        #####################################
        # Settings for solar data
        #####################################
        if solar_data:
            if use_solar_mask == False:
                print("Use solar mask during CLEANing.")
                use_solar_mask = True
            if solar_selfcal == False:
                solar_selfcal = True
            full_FoV=False
        else:
            if do_noise_cal:
                print(
                    "Turning off noise diode based calibration for non-solar observation."
                )
                do_noise_cal = False
            if use_solar_mask:
                print("Stop using solar mask during CLEANing.")
                use_solar_mask = False
            if solar_selfcal:
                solar_selfcal = False
            full_FoV=True

        ###################################################
        # Target spliting spectral and temporal chunks
        ##################################################
        if image_timeres > (2 * 3660):  # If more than 2 hours
            print(
                f"Image time integration is more than 2 hours, which may cause smearing due to solar differential rotation."
            )

        #################################################
        # Determining maximum allowed frequency averaging
        #################################################
        max_freqres = calc_bw_smearing_freqwidth(msname,full_FoV=full_FoV)
        if image_freqres>0:
            freqavg = round(min(image_freqres, max_freqres), 1)
        else:
            freqavg = round(max_freqres,1)

        ################################################
        # Determining maximum allowed temporal averaging
        ################################################
        if solar_data: # For solar data, it is assumed Sun is tracked. 
            max_timeres = calc_time_smearing_timewidth(msname)
        else:
            max_timeres = min(
                calc_time_smearing_timewidth(msname), max_time_solar_smearing(msname)
            )
        if image_timeres>0:
            timeavg = round(min(image_timeres, max_timeres), 1)
        else:
            timeavg = round(max_timeres,1)

        #########################################
        # Target ms frequency chunk based on band
        #########################################
        bad_spws = get_bad_chans(msname).split("0:")[-1].split(";")
        good_start = []
        good_end = []
        for i in range(len(bad_spws) - 1):
            start_chan = int(bad_spws[i].split("~")[-1]) + 1
            end_chan = int(bad_spws[i + 1].split("~")[0]) - 1
            good_start.append(start_chan)
            good_end.append(end_chan)
        start_chan = min(good_start)
        end_chan = max(good_end)
        spw = f"0:{start_chan}~{end_chan}"

        #############################################################
        # Determining numbers of spectral chunks for parallel imaging
        #############################################################
        if image_freqres < 0:
            target_freq_chunk = -1
            nchunk = 1
        else:
            msmd = msmetadata()
            msmd.open(msname)
            chanres = msmd.chanres(0, unit="MHz")[0]
            msmd.close()
            total_bw = chanres * (end_chan - start_chan)
            nchunk=int(total_bw/image_freqres)
            if nchunk>max(4,n_nodes): # Maximum 4 chunking or number of compute nodes
                nchunk=max(4,n_nodes)
                target_freq_chunk=total_bw/nchunk
            else:
                nchnuk=1
                target_freq_chunk=image_freqres
        
        #############################
        # Reset any previous weights
        ############################
        cpu_usage = psutil.cpu_percent(interval=1)  # Average over 1 second
        total_cpus = psutil.cpu_count(logical=True)
        available_cpus = int(total_cpus * (1 - cpu_usage / 100.0))
        available_cpus = max(1, available_cpus)  # Avoid zero workers
        reset_weights_and_flags(
            msname, n_threads=available_cpus, force_reset=do_forcereset_weightflag
        )

        #######################################
        # Run dynamic spectra making
        #######################################
        if make_ds:
            msg = run_ds_jobs(
                msname,
                workdir,
                outdir,
                jobid=jobid,
                target_scans=target_scans,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
            )
            if msg != 0:
                print("!!!! WARNING: Dynamic spectra could not be made. !!!!")

        ########################################
        # Run noise-diode based flux calibration
        ########################################
        if do_noise_cal:
            msg = run_noise_diode_cal(
                msname,
                workdir,
                caldir,
                jobid=jobid,
                keep_backup=keep_backup,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
            )  # Run noise diode based flux calibration
            if msg != 0:
                print(
                    "!!!! WARNING: Error in running noise-diode based flux calibration. Flux density calibration may not be correct. !!!!"
                )

        ##############################
        # Run partitioning jobs
        ##############################
        # If do partition or calibrator ms is not present in case of basic calibration is requested
        calibrator_msname = workdir + "/calibrator.ms"
        if do_basic_cal and (
            do_cal_partition or os.path.exists(calibrator_msname) == False
        ):
            msg = run_partion(
                msname,
                workdir,
                split_fullpol=do_polcal,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
            )
            if msg != 0:
                print("!!!! WARNING: Error in partitioning calibrator fields. !!!!")
                exit_job(start_time, mspath, workdir)
                if remote_logger:
                    pid = start_ping_logger(
                        jobid,
                        remote_job_id,
                        remote_logger_waittime,
                        remote_link=remote_link,
                    )
                    meersolar_cachedir = get_meersolar_cachedir()
                    if pid is not None:
                        save_pid(pid, f"{meersolar_cachedir}/pids/pids_{jobid}.txt")
                return 1

        ##################################
        # Run flagging jobs on calibrators
        ##################################
        # Only if basic calibration is requested
        if do_cal_flag and do_basic_cal:
            if os.path.exists(calibrator_msname) == False:
                print(f"Calibrator ms: {calibrator_ms} is not present.")
                exit_job(start_time, mspath, workdir)
                if remote_logger:
                    pid = start_ping_logger(
                        jobid,
                        remote_job_id,
                        remote_logger_waittime,
                        remote_link=remote_link,
                    )
                    meersolar_cachedir = get_meersolar_cachedir()
                    if pid is not None:
                        save_pid(pid, f"{meersolar_cachedir}/pids/pids_{jobid}.txt")
                return 1
            msg = run_flag(
                calibrator_msname,
                workdir,
                flag_calibrators=True,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
            )
            if msg != 0:
                print(
                    "!!!! WARNING: Flagging error. Examine calibration solutions with caution. !!!!"
                )

        #################################
        # Import model
        #################################
        # Only if basic calibration is requested
        if do_import_model and do_basic_cal:
            if os.path.exists(calibrator_msname) == False:
                print(f"Calibrator ms: {calibrator_ms} is not present.")
                exit_job(start_time, mspath, workdir)
                if remote_logger:
                    pid = start_ping_logger(
                        jobid,
                        remote_job_id,
                        remote_logger_waittime,
                        remote_link=remote_link,
                    )
                    meersolar_cachedir = get_meersolar_cachedir()
                    if pid is not None:
                        save_pid(pid, f"{meersolar_cachedir}/pids/pids_{jobid}.txt")
                return 1
            fluxcal_fields, fluxcal_scans = get_fluxcals(calibrator_msname)
            phasecal_fields, phasecal_scans, phasecal_fluxes = get_phasecals(
                calibrator_msname
            )
            calibrator_field = fluxcal_fields + phasecal_fields
            msg = run_import_model(
                calibrator_msname,
                workdir,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
            )  # Run model import
            if msg != 0:
                print(
                    "!!!! WARNING: Error in importing calibrator models. Not continuing calibration. !!!!"
                )
                exit_job(start_time, mspath, workdir)
                if remote_logger:
                    pid = start_ping_logger(
                        jobid,
                        remote_job_id,
                        remote_logger_waittime,
                        remote_link=remote_link,
                    )
                    meersolar_cachedir = get_meersolar_cachedir()
                    if pid is not None:
                        save_pid(pid, f"{meersolar_cachedir}/pids/pids_{jobid}.txt")
                return 1

        ###############################
        # Run basic calibration
        ###############################
        use_only_bandpass = False
        if do_basic_cal:
            if os.path.exists(calibrator_msname) == False:
                print(f"Calibrator ms: {calibrator_ms} is not present.")
                exit_job(start_time, mspath, workdir)
                if remote_logger:
                    pid = start_ping_logger(
                        jobid,
                        remote_job_id,
                        remote_logger_waittime,
                        remote_link=remote_link,
                    )
                    meersolar_cachedir = get_meersolar_cachedir()
                    if pid is not None:
                        save_pid(pid, f"{meersolar_cachedir}/pids/pids_{jobid}.txt")
                return 1
            msg = run_basic_cal_jobs(
                calibrator_msname,
                workdir,
                caldir,
                perform_polcal=do_polcal,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                keep_backup=keep_backup,
            )  # Run basic calibration
            if msg != 0:
                print(
                    "!!!! WARNING: Error in basic calibration. Not continuing further. !!!!"
                )
                exit_job(start_time, mspath, workdir)
                if remote_logger:
                    pid = start_ping_logger(
                        jobid,
                        remote_job_id,
                        remote_logger_waittime,
                        remote_link=remote_link,
                    )
                    meersolar_cachedir = get_meersolar_cachedir()
                    if pid is not None:
                        save_pid(pid, f"{meersolar_cachedir}/pids/pids_{jobid}.txt")
                return 1

        ##########################################
        # Checking presence of necessary caltables
        ##########################################
        if len(glob.glob(caldir + "/*.bcal")) == 0:
            print(f"No bandpass table is present in calibration directory : {caldir}.")
            exit_job(start_time, mspath, workdir)
            if remote_logger:
                pid = start_ping_logger(
                    jobid,
                    remote_job_id,
                    remote_logger_waittime,
                    remote_link=remote_link,
                )
                meersolar_cachedir = get_meersolar_cachedir()
                if pid is not None:
                    save_pid(pid, f"{meersolar_cachedir}/pids/pids_{jobid}.txt")
            return 1
        if len(glob.glob(caldir + "/*.gcal")) == 0:
            print(
                f"No time-dependent gaintable is present in calibration directory : {caldir}. Applying only bandpass solutions."
            )
            use_only_bandpass = True

        ############################################
        # Spliting for self-cals
        ############################################
        # Spliting only if self-cal is requested
        if do_selfcal:
            if do_selfcal_split == False:
                selfcal_target_mslist = glob.glob(workdir + "/selfcals_scan*.ms")
                if len(selfcal_target_mslist) == 0:
                    print(
                        "No measurement set is present for self-calibration. Spliting them.."
                    )
                    do_selfcal_split = True

            if do_selfcal_split:
                prefix = "selfcals"
                try:
                    time_interval = float(solint)
                except:
                    if "s" in solint:
                        time_interval = float(solint.split("s")[0])
                    elif "min" in solint:
                        time_interval = float(solint.split("min")[0]) * 60
                    elif solint == "int":
                        time_interval = image_timeres
                    else:
                        time_interval = -1
                msg = run_target_split_jobs(
                    msname,
                    workdir,
                    datacolumn="data",
                    freqres=freqavg,
                    timeres=timeavg,
                    target_freq_chunk=25,
                    n_spectral_chunk=nchunk,  # Number of target spectral chunk
                    target_scans=target_scans,
                    prefix=prefix,
                    merge_spws=True,
                    time_window=min(60, time_interval),
                    time_interval=time_interval,
                    split_fullpol=do_polcal,
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    max_cpu_frac=round(cpu_frac, 2),
                    max_mem_frac=round(mem_frac, 2),
                )
                if msg != 0:
                    print(
                        "!!!! WARNING: Error in running spliting target scans for selfcal. !!!!"
                    )
                    do_selfcal = False
                else:
                    print("Waiting to finish spliting of target scans for selfcal...\n")
                    split_basename = f"split_{prefix}"
                    while True:
                        finished_file = glob.glob(
                            workdir + "/.Finished_" + split_basename + "*"
                        )
                        if len(finished_file) > 0:
                            break
                        else:
                            time.sleep(1)
                    success_index_split_target = int(finished_file[0].split("_")[-1])
                    if success_index_split_target == 0:
                        print(
                            "Spliting target scans are done successfully for selfcal.\n"
                        )
                    else:
                        print(
                            "!!!! WARNING: Error in spliting target scans for selfcal. Not continuing further for selfcal. !!!!"
                        )
                        do_selfcal = False

        ####################################
        # Filtering any corrupted ms
        #####################################
        if do_selfcal:
            selfcal_target_mslist = glob.glob(workdir + "/selfcals_scan*.ms")
            filtered_mslist = []  # Filtering in case any ms is corrupted
            for ms in selfcal_target_mslist:
                checkcol = check_datacolumn_valid(ms)
                if checkcol:
                    filtered_mslist.append(ms)
                else:
                    print(f"Issue in : {ms}")
                    os.system(f"rm -rf {ms}")
            selfcal_mslist = filtered_mslist
            if len(selfcal_mslist) == 0:
                print(
                    "No splited target scan ms are available in work directory for selfcal. Not continuing further for selfcal."
                )
                do_selfcal = False
            print(f"Selfcal mslist : {[os.path.basename(i) for i in selfcal_mslist]}")

        #########################################################
        # Applying solutions on target scans for self-calibration
        #########################################################
        if do_selfcal:  # Applying solutions for selfcal
            msg = run_apply_basiccal_sol(
                selfcal_mslist,
                workdir,
                caldir,
                use_only_bandpass=use_only_bandpass,
                overwrite_datacolumn=False,
                applymode="calflag",
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
            )
            if msg != 0:
                print(
                    "!!!! WARNING: Error in applying basic calibration solutions on target scans. Not continuing further for selfcal.!!!!"
                )
                do_selfcal = False

        ########################################
        # Performing self-calibration
        ########################################
        if do_selfcal:
            os.system(
                "rm -rf " + workdir + "/*selfcal " + workdir + "/caltables/*selfcal*"
            )
            if do_sidereal_cor:
                msg = run_solar_siderealcor_jobs(
                    selfcal_mslist,
                    workdir,
                    prefix="selfcals",
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    max_cpu_frac=round(cpu_frac, 2),
                    max_mem_frac=round(mem_frac, 2),
                )
                if msg != 0:
                    print("Sidereal correction is not successful.")
            msg = run_selfcal_jobs(
                selfcal_mslist,
                workdir,
                caldir,
                solint=solint,
                do_apcal=do_ap_selfcal,
                solar_selfcal=solar_selfcal,
                keep_backup=keep_backup,
                uvrange=uvrange,
                weight="briggs",
                robust=0.0,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
            )
            if msg != 0:
                print(
                    "!!!! WARNING: Error in self-calibration on target scans. Not applying self-calibration. !!!!"
                )
                do_apply_selfcal = False

        ########################################
        # Checking self-cal caltables
        ########################################
        selfcal_tables = glob.glob(caldir + "/*selfcal*")
        if len(selfcal_tables) == 0:
            print(
                "Self-calibration is not performed and no self-calibration caltable is available."
            )
            do_apply_selfcal = False

        #############################################
        # Check spliting target scans finished or not
        #############################################
        # If corrected data is requested or imaging is requested
        prefix = "targets"
        if do_target_split and (do_applycal or do_imaging):
            msg = run_target_split_jobs(
                msname,
                workdir,
                datacolumn="data",
                spw=spw,
                target_freq_chunk=target_freq_chunk,
                freqres=freqavg,
                timeres=timeavg,
                n_spectral_chunk=-1,
                target_scans=target_scans,
                prefix=prefix,
                split_fullpol=do_polcal,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                max_cpu_frac=round(cpu_frac, 2),
                max_mem_frac=round(mem_frac, 2),
            )
            if msg != 0:
                print("!!!! WARNING: Error in running spliting target scans. !!!!")
                exit_job(start_time, mspath, workdir)
                if remote_logger:
                    pid = start_ping_logger(
                        jobid,
                        remote_job_id,
                        remote_logger_waittime,
                        remote_link=remote_link,
                    )
                    meersolar_cachedir = get_meersolar_cachedir()
                    if pid is not None:
                        save_pid(pid, f"{meersolar_cachedir}/pids/pids_{jobid}.txt")
                return 1
            ##################################
            # Waiting only spliting is started
            ##################################
            print("Waiting to finish spliting of target scans...\n")
            split_basename = f"split_{prefix}"
            while True:
                finished_file = glob.glob(
                    workdir + "/.Finished_" + split_basename + "*"
                )
                if len(finished_file) > 0:
                    break
                else:
                    time.sleep(1)
            success_index_split_target = int(finished_file[0].split("_")[-1])
            if success_index_split_target == 0:
                print("Spliting target scans are done successfully.\n")
            else:
                print(
                    "!!!! WARNING: Error in spliting target scans. Not continuing further. !!!!"
                )
                exit_job(start_time, mspath, workdir)
                if remote_logger:
                    pid = start_ping_logger(
                        jobid,
                        remote_job_id,
                        remote_logger_waittime,
                        remote_link=remote_link,
                    )
                    meersolar_cachedir = get_meersolar_cachedir()
                    if pid is not None:
                        save_pid(pid, f"{meersolar_cachedir}/pids/pids_{jobid}.txt")
                return 1

        cpu_frac = copy.deepcopy(cpu_frac_bkp)
        mem_frac = copy.deepcopy(mem_frac_bkp)
        target_mslist = glob.glob(workdir + "/targets_scan*.ms")

        ####################################
        # Filtering any corrupted ms
        #####################################
        filtered_mslist = []  # Filtering in case any ms is corrupted
        for ms in target_mslist:
            checkcol = check_datacolumn_valid(ms)
            if checkcol:
                filtered_mslist.append(ms)
            else:
                print(f"Issue in : {ms}")
                os.system("rm -rf {ms}")
        target_mslist = filtered_mslist
        if len(target_mslist) == 0:
            print("No splited target scan ms are available in work directory.")
            exit_job(start_time, mspath, workdir)
            if remote_logger:
                pid = start_ping_logger(
                    jobid,
                    remote_job_id,
                    remote_logger_waittime,
                    remote_link=remote_link,
                )
                meersolar_cachedir = get_meersolar_cachedir()
                if pid is not None:
                    save_pid(pid, f"{meersolar_cachedir}/pids/pids_{jobid}.txt")
            return 1

        if do_applycal or do_imaging:
            print(
                f"Target scan mslist : {[os.path.basename(i) for i in target_mslist]}"
            )

        #########################################################
        # Applying basic solutions on target scans
        #########################################################
        if do_applycal:
            if len(target_mslist) > 0:
                msg = run_apply_basiccal_sol(
                    target_mslist,
                    workdir,
                    caldir,
                    use_only_bandpass=use_only_bandpass,
                    overwrite_datacolumn=True,
                    applymode="calflag",
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                )
                if msg != 0:
                    print(
                        "!!!! WARNING: Error in applying basic calibration solutions on target scans. Not continuing further.!!!!"
                    )
                    exit_job(start_time, mspath, workdir)
                    if remote_logger:
                        pid = start_ping_logger(
                            jobid,
                            remote_job_id,
                            remote_logger_waittime,
                            remote_link=remote_link,
                        )
                        meersolar_cachedir = get_meersolar_cachedir()
                        if pid is not None:
                            save_pid(pid, f"{meersolar_cachedir}/pids/pids_{jobid}.txt")
                    return 1
            else:
                print(
                    "!!!! WARNING: No measurement set is present for basic calibration applying solutions. Not continuing further. !!!!"
                )
                exit_job(start_time, mspath, workdir)
                if remote_logger:
                    pid = start_ping_logger(
                        jobid,
                        remote_job_id,
                        remote_logger_waittime,
                        remote_link=remote_link,
                    )
                    meersolar_cachedir = get_meersolar_cachedir()
                    if pid is not None:
                        save_pid(pid, f"{meersolar_cachedir}/pids/pids_{jobid}.txt")
                return 1
            if do_sidereal_cor:
                msg = run_solar_siderealcor_jobs(
                    target_mslist,
                    workdir,
                    prefix="targets",
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    max_cpu_frac=round(cpu_frac, 2),
                    max_mem_frac=round(mem_frac, 2),
                )

        ########################################
        # Apply self-calibration
        ########################################
        if do_apply_selfcal:
            target_mslist = sorted(target_mslist)
            msg = run_apply_selfcal_sol(
                target_mslist,
                workdir,
                caldir,
                overwrite_datacolumn=False,
                applymode="calonly",
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
            )
            if msg != 0:
                print(
                    "!!!! WARNING: Error in applying self-calibration solutions on target scans. !!!!"
                )

        #####################################
        # Imaging
        ######################################
        if do_imaging:
            if do_polcal == False: # Only if do_polcal is False, overwrite to make only Stokes I
                pol = "I"
            band = get_band_name(target_mslist[0])
            msg = run_imaging_jobs(
                target_mslist,
                workdir,
                outdir,
                freqrange=freqrange,
                timerange=timerange,
                minuv=minuv,
                weight=weight,
                robust=float(robust),
                pol=pol,
                band=band,
                freqres=image_freqres,
                timeres=image_timeres,
                threshold=float(clean_threshold),
                use_multiscale=use_multiscale,
                use_solar_mask=use_solar_mask,
                cutout_rsun=cutout_rsun,
                make_overlay=make_overlay,
                savemodel=keep_backup,
                saveres=keep_backup,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
            )
            if msg != 0:
                print(
                    "!!!! WARNING: Final imaging on all measurement sets is not successful. Check the image directory. !!!!"
                )
                exit_job(start_time, mspath, workdir)
                if remote_logger:
                    pid = start_ping_logger(
                        jobid,
                        remote_job_id,
                        remote_logger_waittime,
                        remote_link=remote_link,
                    )
                    meersolar_cachedir = get_meersolar_cachedir()
                    if pid is not None:
                        save_pid(pid, f"{meersolar_cachedir}/pids/pids_{jobid}.txt")
                return 1

        ###########################
        # Primary beam correction
        ###########################
        if do_pbcor:
            if weight == "briggs":
                weight_str = f"{weight}_{robust}"
            else:
                weight_str = weight
            if image_freqres == -1 and image_timeres == -1:
                imagedir = outdir + f"/imagedir_f_all_t_all_w_{weight_str}"
            elif image_freqres != -1 and image_timeres == -1:
                imagedir = (
                    outdir
                    + f"/imagedir_f_{round(float(image_freqres),1)}_t_all_w_{weight_str}"
                )
            elif image_freqres == -1 and image_timeres != -1:
                imagedir = (
                    outdir
                    + f"/imagedir_f_all_t_{round(float(image_timeres),1)}_w_{weight_str}"
                )
            else:
                imagedir = (
                    outdir
                    + f"/imagedir_f_{round(float(image_freqres),1)}_t_{round(float(image_timeres),1)}_w_{weight_str}"
                )
            imagedir = imagedir + "/images"
            images = glob.glob(imagedir + "/*.fits")
            if len(images) == 0:
                print(f"No image is present in image directory: {imagedir}")
            else:
                msg = run_apply_pbcor(
                    imagedir,
                    workdir,
                    apply_parang=apply_parang,
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                )
                if msg != 0:
                    print(
                        "!!!! WARNING: Primary beam corrections of the final images are not successful. !!!!"
                    )
                    exit_job(start_time, mspath, workdir)
                    if remote_logger:
                        pid = start_ping_logger(
                            jobid,
                            remote_job_id,
                            remote_logger_waittime,
                            remote_link=remote_link,
                        )
                        meersolar_cachedir = get_meersolar_cachedir()
                        if pid is not None:
                            save_pid(pid, f"{meersolar_cachedir}/pids/pids_{jobid}.txt")
                    return 1
                print(f"Final image directory: {os.path.dirname(imagedir)}\n")
        ###########################################
        # Successful exit
        ###########################################
        print(
            f"Calibration and imaging pipeline is successfully run on measurement set : {msname}\n"
        )
        exit_job(start_time, mspath, workdir)
        if remote_logger:
            pid = start_ping_logger(
                jobid, remote_job_id, remote_logger_waittime, remote_link=remote_link
            )
            meersolar_cachedir = get_meersolar_cachedir()
            if pid is not None:
                save_pid(pid, f"{meersolar_cachedir}/pids/pids_{jobid}.txt")
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1
    finally:
        drop_cache(msname)
        drop_cache(workdir)
        drop_cache(outdir)


def main():
    parser = argparse.ArgumentParser(
        description="Run MeerSOLAR for calibration and imaging of solar observations.",
        formatter_class=SmartDefaultsHelpFormatter,
    )
    # === Essential parameters ===
    essential = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    essential.add_argument("msname", type=str, help="Measurement set name")
    essential.add_argument(
        "--workdir",
        type=str,
        dest="workdir",
        required=True,
        help="Working directory",
    )
    essential.add_argument(
        "--outdir",
        type=str,
        dest="outdir",
        required=True,
        help="Output products directory",
    )

    # === Advanced calibration parameters ===
    advanced_cal = parser.add_argument_group(
        "###################\nAdvanced calibration parameters\n###################"
    )
    advanced_cal.add_argument(
        "--solint",
        type=str,
        default="5min",
        help="Solution interval for calibration (e.g. 'int', '10s', '5min', 'inf')",
    )
    advanced_cal.add_argument(
        "--cal_uvrange",
        type=str,
        default="",
        help="UV range to filter data for calibration (e.g. '>100klambda', '100~10000lambda')",
    )
    advanced_cal.add_argument(
        "--no_polcal",
        action="store_false",
        dest="do_polcal",
        help="Disable polarization calibration",
    )

    # === Advanced imaging and calibration parameters ===
    advanced_image = parser.add_argument_group(
        "###################\nAdvanced imaging parameters\n###################"
    )
    advanced_image.add_argument(
        "--target_scans",
        nargs="*",
        type=str,
        default=[],
        help="List of target scans to process (space-separated, e.g. 3 5 7)",
    )
    advanced_image.add_argument(
        "--freqrange",
        type=str,
        default="",
        help="Frequency range in MHz to select during imaging (comma-seperate, e.g. '100~110,130~140')",
    )
    advanced_image.add_argument(
        "--timerange",
        type=str,
        default="",
        help="Time range to select during imaging (comma-seperated, e.g. '2014/09/06/09:30:00~2014/09/06/09:45:00,2014/09/06/10:30:00~2014/09/06/10:45:00')",
    )
    advanced_image.add_argument(
        "--image_freqres",
        type=int,
        default=-1,
        help="Output image frequency resolution in MHz (-1 = full)",
    )
    advanced_image.add_argument(
        "--image_timeres",
        type=int,
        default=-1,
        help="Output image time resolution in seconds (-1 = full)",
    )
    advanced_image.add_argument(
        "--pol",
        type=str,
        default="IQUV",
        help="Stokes parameter(s) to image (e.g. 'I', 'XX', 'RR', 'IQUV')",
    )
    advanced_image.add_argument(
        "--minuv",
        type=float,
        default=0,
        help="Minimum baseline length (in wavelengths) to include in imaging",
    )
    advanced_image.add_argument(
        "--weight",
        type=str,
        default="briggs",
        help="Imaging weighting scheme (e.g. 'briggs', 'natural', 'uniform')",
    )
    advanced_image.add_argument(
        "--robust",
        type=float,
        default=0.0,
        help="Robust parameter for Briggs weighting (-2 to +2)",
    )
    advanced_image.add_argument(
        "--no_multiscale",
        action="store_false",
        dest="use_multiscale",
        help="Disable multiscale CLEAN for extended structures",
    )
    advanced_image.add_argument(
        "--clean_threshold",
        type=float,
        default=1.0,
        help="Clean threshold in sigma for final deconvolution (Note this is not auto-mask)",
    )
    advanced_image.add_argument(
        "--do_pbcor",
        action="store_true",
        help="Apply primary beam correction after imaging",
    )
    advanced_image.add_argument(
        "--no_apply_parang",
        action="store_false",
        dest="apply_parang",
        help="Disable parallactic angle rotation during imaging",
    )
    advanced_image.add_argument(
        "--cutout_rsun",
        type=float,
        default=2.5,
        help="Field of view cutout radius in solar radii",
    )
    advanced_image.add_argument(
        "--no_solar_mask",
        action="store_false",
        dest="use_solar_mask",
        help="Disable use solar disk mask during deconvolution",
    )
    advanced_image.add_argument(
        "--no_overlay",
        action="store_false",
        dest="make_overlay",
        help="Disable overlay plot on GOES SUVI after imaging",
    )

    # === Advanced options ===
    advanced = parser.add_argument_group(
        "###################\nAdvanced pipeline parameters\n###################"
    )
    advanced.add_argument(
        "--non_solar_data",
        action="store_false",
        dest="solar_data",
        help="Disable solar data mode",
    )
    advanced.add_argument(
        "--non_ds",
        action="store_false",
        dest="make_ds",
        help="Disable making solar dynamic spectra",
    )
    advanced.add_argument(
        "--do_forcereset_weightflag",
        action="store_true",
        help="Force reset of weights and flags (disabled by default)",
    )
    advanced.add_argument(
        "--no_noise_cal",
        action="store_false",
        dest="do_noise_cal",
        help="Disable noise calibration",
    )
    advanced.add_argument(
        "--no_cal_partition",
        action="store_false",
        dest="do_cal_partition",
        help="Disable calibrator MS partitioning",
    )
    advanced.add_argument(
        "--no_cal_flag",
        action="store_false",
        dest="do_cal_flag",
        help="Disable initial flagging of calibrators",
    )
    advanced.add_argument(
        "--no_import_model",
        action="store_false",
        dest="do_import_model",
        help="Disable model import",
    )
    advanced.add_argument(
        "--no_basic_cal",
        action="store_false",
        dest="do_basic_cal",
        help="Disable basic gain calibration",
    )
    advanced.add_argument(
        "--do_sidereal_cor",
        action="store_true",
        dest="do_sidereal_cor",
        help="Sidereal motion correction for Sun (disabled by default)",
    )
    advanced.add_argument(
        "--no_selfcal_split",
        action="store_false",
        dest="do_selfcal_split",
        help="Disable split for self-calibration",
    )
    advanced.add_argument(
        "--no_selfcal",
        action="store_false",
        dest="do_selfcal",
        help="Disable self-calibration",
    )
    advanced.add_argument(
        "--no_ap_selfcal",
        action="store_false",
        dest="do_ap_selfcal",
        help="Disable amplitude-phase self-calibration",
    )
    advanced.add_argument(
        "--no_solar_selfcal",
        action="store_false",
        dest="solar_selfcal",
        help="Disable solar-specific self-calibration parameters",
    )
    advanced.add_argument(
        "--no_target_split",
        action="store_false",
        dest="do_target_split",
        help="Disable target data split",
    )
    advanced.add_argument(
        "--no_applycal",
        action="store_false",
        dest="do_applycal",
        help="Disable application of basic calibration solutions",
    )
    advanced.add_argument(
        "--no_apply_selfcal",
        action="store_false",
        dest="do_apply_selfcal",
        help="Disable application of self-calibration solutions",
    )
    advanced.add_argument(
        "--no_imaging",
        action="store_false",
        dest="do_imaging",
        help="Disable final imaging",
    )

    # === Advanced local system/ per node hardware resource parameters ===
    advanced_resource = parser.add_argument_group(
        "###################\nAdvanced hardware resource parameters for local system or per node on HPC cluster\n###################"
    )
    advanced_resource.add_argument(
        "--cpu_frac",
        type=float,
        default=0.8,
        help="Fraction of CPU usuage per node",
    )
    advanced_resource.add_argument(
        "--mem_frac",
        type=float,
        default=0.8,
        help="Fraction of memory usuage per node",
    )
    advanced_resource.add_argument(
        "--keep_backup",
        action="store_true",
        help="Keep backup of intermediate steps",
    )
    advanced_resource.add_argument(
        "--no_remote_logger",
        action="store_false",
        dest="remote_logger",
        help="Disable remote logger",
    )
    advanced_resource.add_argument(
        "--logger_alivetime",
        type=float,
        default=0,
        help="Keep remote logger alive for this many hours (Otherwise, logger will be removed after 15 minutes of inactivity)",
    )

    # === Advanced local system/ per node hardware resource parameters ===
    advanced_hpc = parser.add_argument_group(
        "###################\nAdvanced HPC settings\n###################"
    )
    advanced_hpc.add_argument(
        "--n_nodes",
        type=int,
        default=0,
        help="Number of compute nodes to use (0 means local cluster)",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    try:
        print("\n########################################")
        print("Starting MeerSOLAR Pipeline....")
        print("#########################################\n")
        # TODO: multi specification imaging parameters as json dic 
        msg = master_control(
            msname=args.msname,
            workdir=args.workdir,
            outdir=args.outdir,
            solar_data=args.solar_data,
            # Pre-calibration
            do_forcereset_weightflag=args.do_forcereset_weightflag,
            do_cal_partition=args.do_cal_partition,
            do_cal_flag=args.do_cal_flag,
            do_import_model=args.do_import_model,
            # Basic calibration
            do_basic_cal=args.do_basic_cal,
            do_noise_cal=args.do_noise_cal,
            do_applycal=args.do_applycal,
            # Target data preparation
            do_target_split=args.do_target_split,
            target_scans=args.target_scans,
            freqrange=args.freqrange,
            timerange=args.timerange,
            uvrange=args.cal_uvrange,
            # Polarization calibration
            do_polcal=args.do_polcal,
            # Self-calibration
            do_selfcal=args.do_selfcal,
            do_selfcal_split=args.do_selfcal_split,
            do_apply_selfcal=args.do_apply_selfcal,
            do_ap_selfcal=args.do_ap_selfcal,
            solar_selfcal=args.solar_selfcal,
            solint=args.solint,
            # Sidereal correction
            do_sidereal_cor=args.do_sidereal_cor,
            # Dynamic spectra
            make_ds=args.make_ds,
            # Imaging
            do_imaging=args.do_imaging,
            do_pbcor=args.do_pbcor,
            weight=args.weight,
            robust=args.robust,
            minuv=args.minuv,
            image_freqres=args.image_freqres,
            image_timeres=args.image_timeres,
            pol=args.pol,
            apply_parang=args.apply_parang,
            clean_threshold=args.clean_threshold,
            use_multiscale=args.use_multiscale,
            use_solar_mask=args.use_solar_mask,
            cutout_rsun=args.cutout_rsun,
            make_overlay=args.make_overlay,
            # Resource settings
            cpu_frac=args.cpu_frac,
            mem_frac=args.mem_frac,
            n_nodes=args.n_nodes,
            keep_backup=args.keep_backup,
            # Remote logging
            remote_logger=args.remote_logger,
            remote_logger_waittime=args.logger_alivetime,
        )
    except Exception as e:
        traceback.print_exc()
    finally:
        drop_cache(args.msname)
        drop_cache(args.workdir)
        drop_cache(args.outdir)


if __name__ == "__main__":
    main()
