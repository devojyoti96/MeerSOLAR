import os, glob, psutil, copy, time, traceback, argparse, socket, threading
from meersolar.pipeline.basic_func import *
from meersolar.pipeline.init_data import init_meersolar_data
from casatasks import casalog

logfile = casalog.logfile()
os.system("rm -rf " + logfile)


def run_flag(msname, workdir, flag_calibrators=True, jobid=0, cpu_frac=0.8, mem_frac=0.8):
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
            "run_flag --msname "
            + msname
            + " --datacolumn DATA --use_tfcrop True --flagbackup True"
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
            "run_flag --msname "
            + msname
            + " --datacolumn DATA --use_tfcrop True --flagdimension freq --flagbackup True"
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
            + " --workdir "
            + str(workdir)
            + " --jobid "
            + str(jobid)
        )
    flag_basename = f"flagging_{flagfield_type}_" + os.path.basename(msname).split(".ms")[0]
    logfile = workdir + "/logs/" + flag_basename + ".log"
    flagging_cmd+=f" --logfile {logfile}"
    if os.path.isdir(workdir + "/logs") == False:
        os.makedirs(workdir + "/logs")
    batch_file = create_batch_script_nonhpc(flagging_cmd, workdir, flag_basename, jobid)
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
        "import_model --msname "
        + msname
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
    import_model_cmd+=f" --logfile {logfile}"
    if os.path.isdir(workdir + "/logs") == False:
        os.makedirs(workdir + "/logs")
    batch_file = create_batch_script_nonhpc(import_model_cmd, workdir, model_basename, jobid)
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
    msname, workdir, perform_polcal=False, jobid=0, cpu_frac=0.8, mem_frac=0.8, keep_backup=False
):
    """
    Perform basic calibration

    Parameters
    ----------
    msname: str
        Name of the measurement set
    workdir : str
        Working directory
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
        "run_basic_cal --msname "
        + msname
        + " --workdir "
        + workdir
        + " --perform_polcal "
        + str(perform_polcal)
        + " --cpu_frac "
        + str(cpu_frac)
        + " --mem_frac "
        + str(mem_frac)
        + " --keep_backup "
        + str(keep_backup)
        + " --jobid "
        + str(jobid)
    )
    logfile = workdir + "/logs/" + cal_basename + ".log"
    basic_cal_cmd+=f" --logfile {logfile}"
    if os.path.isdir(workdir + "/logs") == False:
        os.makedirs(workdir + "/logs")
    batch_file = create_batch_script_nonhpc(basic_cal_cmd, workdir, cal_basename, jobid)
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


def run_noise_diode_cal(msname, workdir, jobid=0, cpu_frac=0.8, mem_frac=0.8):
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
            + " --jobid "
            + str(jobid)
        )
        logfile = workdir + "/logs/" + noisecal_basename + ".log"
        noise_cal_cmd+=f" --logfile {logfile}"
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        batch_file = create_batch_script_nonhpc(
            noise_cal_cmd, workdir, noisecal_basename, jobid
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


def run_partion(msname, workdir, split_fullpol=False, jobid=0, cpu_frac=0.8, mem_frac=0.8):
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
    split_cmd = f"run_partition --msname {msname} --outputms {calibrator_ms} --scans {cal_scans} --timebin {timebin} --width {width} --cpu_frac {cpu_frac} --mem_frac {mem_frac} --split_fullpol {split_fullpol} --workdir {workdir} --jobid {jobid}"
    ####################################
    # Partition fields
    ####################################
    print("\n########################################")
    partition_basename = f"partition_cal"
    logfile = workdir + "/logs/" + partition_basename + ".log"
    split_cmd+=f" --logfile {logfile}"
    if os.path.isdir(workdir + "/logs") == False:
        os.makedirs(workdir + "/logs")
    batch_file = create_batch_script_nonhpc(split_cmd, workdir, partition_basename, jobid)
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
            + " --split_fullpol "
            + str(split_fullpol)
            + " --time_window "
            + str(time_window)
            + " --time_interval "
            + str(time_interval)
            + " --merge_spws "
            + str(merge_spws)
            + " --jobid "
            + str(jobid)
        )
        if spw != "":
            split_cmd = split_cmd + " --spw " + str(spw)
        if len(target_scans) > 0:
            split_cmd = (
                split_cmd + " --scans " + ",".join([str(s) for s in target_scans])
            )
        logfile = workdir + "/logs/" + split_basename + ".log"
        split_cmd+=f" --logfile {logfile}"
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        batch_file = create_batch_script_nonhpc(split_cmd, workdir, split_basename, jobid)
        print(split_cmd + "\n")
        os.system("bash " + batch_file)
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1


def run_sidereal_cor_jobs(
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
    Apply sidereal motion correction

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
            "run_sidereal_cor --mslist "
            + mslist
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
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        logfile = workdir + "/logs/" + sidereal_basename + ".log"
        sidereal_cor_cmd+=f" --logfile {logfile}"
        batch_file = create_batch_script_nonhpc(
            sidereal_cor_cmd, workdir, sidereal_basename, jobid
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
            "run_meerpbcor --imagedir "
            + str(imagedir)
            + " --workdir "
            + str(workdir)
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
            + " --apply_parang "
            + str(apply_parang)
            + " --jobid "
            + str(jobid)
        )
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        logfile = workdir + "/logs/" + applypbcor_basename + ".log"
        applypbcor_cmd+=f" --logfile {logfile}"
        batch_file = create_batch_script_nonhpc(
            applypbcor_cmd, workdir, applypbcor_basename, jobid
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
        applycal_cmd = (
            "run_apply_basiccal --mslist "
            + ",".join(target_mslist)
            + " --workdir "
            + workdir
            + " --caldir "
            + caldir
            + " --use_only_bandpass "
            + str(use_only_bandpass)
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
            + " --overwrite_datacolumn "
            + str(overwrite_datacolumn)
            + " --applymode "
            + str(applymode)
            + " --do_post_flag False "
            + " --jobid "
            + str(jobid)
        )
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        logfile = workdir + "/logs/" + applycal_basename + ".log"
        applycal_cmd+=f" --logfile {logfile}"
        batch_file = create_batch_script_nonhpc(
            applycal_cmd, workdir, applycal_basename, jobid
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
        applycal_cmd = (
            "run_apply_selfcal --mslist "
            + ",".join(target_mslist)
            + " --workdir "
            + workdir
            + " --caldir "
            + caldir
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
            + " --overwrite_datacolumn "
            + str(overwrite_datacolumn)
            + " --applymode "
            + str(applymode)
            + " --jobid "
            + str(jobid)
        )
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        logfile = workdir + "/logs/" + applycal_basename + ".log"
        applycal_cmd+=f" --logfile {logfile}"
        batch_file = create_batch_script_nonhpc(
            applycal_cmd, workdir, applycal_basename, jobid
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
        selfcal_cmd = (
            "run_selfcal --mslist "
            + ",".join(mslist)
            + " --workdir "
            + workdir
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
        if do_apcal != "":
            selfcal_cmd += " --do_apcal " + str(do_apcal)
        if solar_selfcal != "":
            selfcal_cmd += " --solar_selfcal " + str(solar_selfcal)
        if keep_backup != "":
            selfcal_cmd += " --keep_backup " + str(keep_backup)
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
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        batch_file = create_batch_script_nonhpc(selfcal_cmd, workdir, selfcal_basename, jobid)
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
    freqrange : str, optional
        Frequency range to image
    timerange : str, optional
        Time range to image
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
        Stokes
    Parameters to image
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
        imaging_cmd = (
            "run_imaging --mslist "
            + ",".join(mslist)
            + " --workdir "
            + workdir
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
            + " --use_solar_mask "
            + str(use_solar_mask)
            + " --threshold "
            + str(threshold)
            + " --savemodel "
            + str(savemodel)
            + " --saveres "
            + str(saveres)
            + " --use_multiscale "
            + str(use_multiscale)
            + " --make_overlay "
            + str(make_overlay)
            + " --cutout_rsun "
            + str(cutout_rsun)
            + " --make_plots True "
            + " --jobid "
            + str(jobid)
        )
        if freqrange != "":
            imaging_cmd += " --freqrange " + str(freqrange)
        if timerange != "":
            imaging_cmd += " --timerange " + str(timerange)
        if band != "":
            imaging_cmd += " --band " + str(band)
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        batch_file = create_batch_script_nonhpc(imaging_cmd, workdir, imaging_basename, jobid)
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
    solar_data=True,
    do_forcereset_weightflag=False,
    do_cal_partition=True,
    do_cal_flag=True,
    do_import_model=True,
    do_basic_cal=True,
    do_noise_cal=True,
    do_applycal=True,
    do_target_split=True,
    do_polcal=False,
    do_selfcal=True,
    do_selfcal_split=True,
    do_apply_selfcal=True,
    do_ap_selfcal=True,
    solar_selfcal=True,
    target_scans=[],
    freqrange="",
    timerange="",
    uvrange="",
    solint="5min",
    do_imaging=True,
    do_pbcor=True,
    weight="briggs",
    robust=0.0,
    minuv=0,
    freqavg=-1,
    timeavg=-1,
    image_freqres=-1,
    image_timeres=-1,
    pol="I",
    apply_parang=True,
    clean_threshold=1.0,
    use_multiscale=True,
    use_solar_mask=True,
    cutout_rsun=2.5,
    make_overlay=True,
    cpu_frac=0.8,
    mem_frac=0.8,
    n_nodes=1,
    keep_backup=False,
    remote_logger=False,
    remote_logger_waittime=2,
):
    """
    Master controller of the entire pipeline

    Parameters
    ----------
    msname : str
        Measurement set name
    workdir : str
        Work directory path
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
    do_polcal : bool, optional
        Perform full-polarization calibration and imaging
    do_selfcal : bool, optional
        Perform self-calibration
    do_apply_selfcal : bool, optonal
        Apply self-calibration solutions
    do_ap_selfcal : bool, optional
        Perform amplitude-phase self-cal or not
    solar_selfcal : bool, optional
        Whether self-calibration is performing on solar observation or not
    target_scans : list, optional
        Target scans to self-cal and image
    freqrange : str, optional
        Frequency range to image in MHz (xx1~xx2,xx3~xx4,)
    timerange : str, optional
        Time range to image in YYYY/MM/DD/hh:mm:ss format (tt1~tt2,tt3~tt4,...)
    uvrange : str, optional
        UV-range for calibration
    solint : str, optional
        Solution intervals in self-cal
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
    freqavg : float, optional
        Frequency averaging in MHz
    timeavg : float, optional
        Time averaging in seconds
    image_freqres : float, optional
        Image frequency resolution in MHz
    image_timeres : float, optional
        Image temporal resolution
    pol : str, optional
        Stokes
    Parameters of final imaging
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
        Number of nodes to use (Only for cluster architechture)
    keep_backup : bool, optional
        Keep backup of self-cal rounds and final models and residual images

    Returns
    -------
    int
        Success message
    """
    init_meersolar_data()
    pid=os.getpid()
    jobid=get_jobid()
    main_job_file = save_main_process_info(pid, jobid, workdir, cpu_frac, mem_frac)
    start_time = time.time()
    print("###########################")
    print("Starting the pipeline .....")
    print (f"MeerSOLAR Job ID: {jobid}")
    print("###########################\n")
    msname = os.path.abspath(msname.rstrip("/"))
    if os.path.exists(msname) == False:
        print("Please provide a valid measurement set location.\n")
        exit_job(start_time)
        return 1
    mspath = os.path.dirname(msname)
    band = get_band_name(msname)
    ###################################
    # Preparing working directories
    ###################################
    if workdir == "":
        workdir = os.path.dirname(os.path.abspath(msname)) + "/workdir"
    if workdir[-1] == "/":
        workdir = workdir[:-1]
    if os.path.exists(workdir) == False:
        os.makedirs(workdir)
    if mspath != "" and os.path.exists(mspath + "/dask-scratch-space"):
        os.system("rm -rf " + mspath + "/dask-scratch-space " + mspath + "/tmp")
    if workdir != "" and os.path.exists(workdir + "/dask-scratch-space"):
        os.system("rm -rf " + workdir + "/dask-scratch-space " + workdir + "/tmp")
    os.chdir(workdir)
    caldir = workdir + "/caltables"
    cpu_frac_bkp = copy.deepcopy(cpu_frac)
    mem_frac_bkp = copy.deepcopy(mem_frac)
    if remote_logger:
        remote_link=get_remote_logger_link()    
        if remote_link=="":
            print ("Please provide a valid remote link.")
            remote_logger=False    

    if remote_logger:
        ####################################
        # Job name and logging password
        ####################################
        hostname = socket.gethostname()
        timestamp = dt.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        job_name = f"{hostname} :: {timestamp} :: {os.path.basename(msname).split('.ms')[0]}"
        timestamp1 = dt.utcnow().strftime("%Y%m%dT%H%M%S")
        job_id = f"{hostname}_{timestamp1}_{os.path.basename(msname).split('.ms')[0]}"
        password=generate_password()
        np.save(f"{workdir}/jobname_password.npy",np.array([job_name,password],dtype="object"))
        print ("############################################################################")
        print (remote_link)
        print (f"Job ID: {job_name}")
        print (f"Remote access password: {password}")
        print ("#############################################################################")
    
    #####################################
    # Settings for solar data
    #####################################
    if solar_data:
        if use_solar_mask == False:
            print("Use solar mask during CLEANing.")
            use_solar_mask = True
        if solar_selfcal == False:
            solar_selfcal = True
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

    ###################################################
    # Target spliting spectral and temporal chunks
    ##################################################
    if image_timeres > (25 * 60):  # If more than 25 minute
        print(
            f"Image time integration is more than 25 minute, which cause issue in primary beam effects."
        )
        print("Resetting to maximum 25 minutes averaging.\n")
        image_timeres = 25 * 60
    elif image_timeres < 0:
        image_timeres = 25 * 60

    #################################################
    # Determining maximum allowed frequency averaging
    #################################################
    max_freqres = calc_bw_smearing_freqwidth(msname)
    if freqavg < 0 or freqavg > max_freqres:
        freqavg = round(max_freqres, 2)
    if image_freqres > 0 and freqavg > image_freqres:
        freqavg = image_freqres

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

    msmd = msmetadata()
    msmd.open(msname)
    chanres = msmd.chanres(0, unit="MHz")[0]
    msmd.close()
    total_bw = chanres * (end_chan - start_chan)
    nchunk = 4
    target_freq_chunk = total_bw / nchunk
    if image_freqres < 0:
        band = get_band_name(msname)
        if band == "U":
            target_freq_chunk = -1
            nchunk = 1
    elif image_freqres > target_freq_chunk:
        nchunk = image_freqres // target_freq_chunk
        target_freq_chunk = image_freqres / nchunk

    ################################################
    # Determining maximum allowed temporal averaging
    ################################################
    max_timeres = min(
        calc_time_smearing_timewidth(msname), max_time_solar_smearing(msname)
    )
    if timeavg < 0 or timeavg > max_timeres:
        timeavg = round(max_timeres, 2)
    if image_timeres > 0 and timeavg > image_timeres:
        timeavg = image_timeres

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

    ########################################
    # Run noise-diode based flux calibration
    ########################################
    if do_noise_cal:
        msg = run_noise_diode_cal(
            msname,
            workdir,
            jobid=jobid,
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
    calibrator_msname = workdir + "/calibrator.ms"
    if do_cal_partition or ((do_cal_flag or do_import_model or do_basic_cal) and os.path.exists(calibrator_msname) == False):
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
                stop_event = threading.Event()
                threading.Thread(target=ping_logger, args=(job_id, stop_event), daemon=True).start()
                threading.Timer((remote_logger_waittime*3600), stop_event.set).start()
            return 1

    ##################################
    # Run flagging jobs on calibrators
    ##################################
    if do_cal_flag:
        if os.path.exists(calibrator_msname) == False:
            print(f"Calibrator ms: {calibrator_ms} is not present.")
            exit_job(start_time, mspath, workdir)
            if remote_logger:
                stop_event = threading.Event()
                threading.Thread(target=ping_logger, args=(job_id, stop_event), daemon=True).start()
                threading.Timer((remote_logger_waittime*3600), stop_event.set).start()
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
            print("!!!! WARNING: Flagging error. Examine calibration solutions with caution. !!!!")
            
    #################################
    # Import model
    #################################
    if do_import_model:
        if os.path.exists(calibrator_msname) == False:
            print(f"Calibrator ms: {calibrator_ms} is not present.")
            exit_job(start_time, mspath, workdir)
            if remote_logger:
                stop_event = threading.Event()
                threading.Thread(target=ping_logger, args=(job_id, stop_event), daemon=True).start()
                threading.Timer((remote_logger_waittime*3600), stop_event.set).start()
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
                stop_event = threading.Event()
                threading.Thread(target=ping_logger, args=(job_id, stop_event), daemon=True).start()
                threading.Timer((remote_logger_waittime*3600), stop_event.set).start()
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
                stop_event = threading.Event()
                threading.Thread(target=ping_logger, args=(job_id, stop_event), daemon=True).start()
                threading.Timer((remote_logger_waittime*3600), stop_event.set).start()
            return 1
        msg = run_basic_cal_jobs(
            calibrator_msname,
            workdir,
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
                stop_event = threading.Event()
                threading.Thread(target=ping_logger, args=(job_id, stop_event), daemon=True).start()
                threading.Timer((remote_logger_waittime*3600), stop_event.set).start()
            return 1

    ##########################################
    # Checking presence of necessary caltables
    ##########################################
    if len(glob.glob(caldir + "/*.bcal")) == 0:
        print(f"No bandpass table is present in calibration directory : {caldir}.")
        exit_job(start_time, mspath, workdir)
        if remote_logger:
            stop_event = threading.Event()
            threading.Thread(target=ping_logger, args=(job_id, stop_event), daemon=True).start()
            threading.Timer((remote_logger_waittime*3600), stop_event.set).start()
        return 1
    if len(glob.glob(caldir + "/*.gcal")) == 0:
        print(
            f"No time-dependent gaintable is present in calibration directory : {caldir}. Applying only bandpass solutions."
        )
        use_only_bandpass = True

    ############################################
    # Spliting for self-cals
    ############################################
    if do_selfcal_split==False:
        selfcal_target_mslist = glob.glob(workdir + "/selfcals_scan*.ms")
        if len(selfcal_target_mslist) == 0:
            print ("No measurement set is present for self-calibration. Spliting them..")
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
                time_interval = timeavg
            else:
                time_interval = -1
        msg = run_target_split_jobs(
            msname,
            workdir,
            datacolumn="data",
            timeres=timeavg,
            freqres=freqavg,
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
                print("Spliting target scans are done successfully for selfcal.\n")
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
        caldir = workdir + "/caltables"
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
        os.system("rm -rf " + workdir + "/*selfcal " + workdir + "/caltables/*selfcal*")
        msg = run_sidereal_cor_jobs(
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
    selfcaldir = glob.glob(workdir + "/caltables/*selfcal*")
    if len(selfcaldir) == 0:
        print(
            "Self-calibration is not performed and no self-calibration caltable is available."
        )
        do_apply_selfcal = False

    #############################################
    # Check spliting target scans finished or not
    #############################################
    prefix = "targets"
    if do_target_split:
        msg = run_target_split_jobs(
            msname,
            workdir,
            datacolumn="data",
            spw=spw,
            target_freq_chunk=target_freq_chunk,
            timeres=timeavg,
            freqres=freqavg,
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
                stop_event = threading.Event()
                threading.Thread(target=ping_logger, args=(job_id, stop_event), daemon=True).start()
                threading.Timer((remote_logger_waittime*3600), stop_event.set).start()
            return 1

    print("Waiting to finish spliting of target scans...\n")
    split_basename = f"split_{prefix}"
    while True:
        finished_file = glob.glob(workdir + "/.Finished_" + split_basename + "*")
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
            stop_event = threading.Event()
            threading.Thread(target=ping_logger, args=(job_id, stop_event), daemon=True).start()
            threading.Timer((remote_logger_waittime*3600), stop_event.set).start()
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
            stop_event = threading.Event()
            threading.Thread(target=ping_logger, args=(job_id, stop_event), daemon=True).start()
            threading.Timer((remote_logger_waittime*3600), stop_event.set).start()
        return 1

    if do_applycal or do_imaging:
        print(f"Target scan mslist : {[os.path.basename(i) for i in target_mslist]}")

    #########################################################
    # Applying basic solutions on target scans
    #########################################################
    if do_applycal:
        if len(target_mslist) > 0:
            caldir = workdir + "/caltables"
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
                    stop_event = threading.Event()
                    threading.Thread(target=ping_logger, args=(job_id, stop_event), daemon=True).start()
                    threading.Timer((remote_logger_waittime*3600), stop_event.set).start()
                return 1
        else:
            print(
                "!!!! WARNING: No measurement set is present for basic calibration applying solutions. Not continuing further. !!!!"
            )
            exit_job(start_time, mspath, workdir)
            if remote_logger:
                stop_event = threading.Event()
                threading.Thread(target=ping_logger, args=(job_id, stop_event), daemon=True).start()
                threading.Timer((remote_logger_waittime*3600), stop_event.set).start()
            return 1
        msg = run_sidereal_cor_jobs(
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
        caldir = workdir + "/caltables"
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
        if do_polcal == False:
            pol = "I"
        else:
            pol = "IQUV"
        msg = run_imaging_jobs(
            target_mslist,
            workdir,
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
                stop_event = threading.Event()
                threading.Thread(target=ping_logger, args=(job_id, stop_event), daemon=True).start()
                threading.Timer((remote_logger_waittime*3600), stop_event.set).start()
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
            imagedir = workdir + f"/imagedir_f_all_t_all_w_{weight_str}"
        elif image_freqres != -1 and image_timeres == -1:
            imagedir = (
                workdir
                + f"/imagedir_f_{round(float(image_freqres),1)}_t_all_w_{weight_str}"
            )
        elif image_freqres == -1 and image_timeres != -1:
            imagedir = (
                workdir
                + f"/imagedir_f_all_t_{round(float(image_timeres),1)}_w_{weight_str}"
            )
        else:
            imagedir = (
                workdir
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
                    stop_event = threading.Event()
                    threading.Thread(target=ping_logger, args=(job_id, stop_event), daemon=True).start()
                    threading.Timer((remote_logger_waittime*3600), stop_event.set).start()
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
        stop_event = threading.Event()
        threading.Thread(target=ping_logger, args=(job_id, stop_event), daemon=True).start()
        threading.Timer((remote_logger_waittime*3600), stop_event.set).start()
    return 0
    
def main():
    parser = argparse.ArgumentParser(
        description="Run MeerSOLAR for calibration and imaging."
    )
    
    # === Essential parameters ===
    essential = parser.add_argument_group("Essential parameters")
    essential.add_argument("msname", type=str, help="Measurement set name")
    essential.add_argument("workdir", type=str, help="Working directory")
    essential.add_argument("--solar_data", action="store_true", help="Enable solar data mode")
    
    # === Advanced options ===
    advanced = parser.add_argument_group("Advanced parameters")

    advanced.add_argument("--do_basic_cal", action="store_true", help="Perform basic gain calibration")
    advanced.add_argument("--do_imaging", action="store_true", help="Run final imaging")
    advanced.add_argument("--do_forcereset_weightflag", action="store_true")
    advanced.add_argument("--do_cal_partition", action="store_true")
    advanced.add_argument("--do_cal_flag", action="store_true")
    advanced.add_argument("--do_import_model", action="store_true")
    advanced.add_argument("--do_noise_cal", action="store_true")
    advanced.add_argument("--do_applycal", action="store_true")
    advanced.add_argument("--do_target_split", action="store_true")
    advanced.add_argument("--do_polcal", action="store_true")
    advanced.add_argument("--do_selfcal", action="store_true")
    advanced.add_argument("--do_selfcal_split", action="store_true")
    advanced.add_argument("--do_apply_selfcal", action="store_true")
    advanced.add_argument("--do_ap_selfcal", action="store_true")
    advanced.add_argument("--solar_selfcal", action="store_true")

    advanced.add_argument("--target_scans", nargs='*', type=str, default=[])
    advanced.add_argument("--freqrange", type=str, default="")
    advanced.add_argument("--timerange", type=str, default="")
    advanced.add_argument("--uvrange", type=str, default="")
    advanced.add_argument("--solint", type=str, default="5min")

    advanced.add_argument("--do_pbcor", action="store_true")
    advanced.add_argument("--weight", type=str, default="briggs")
    advanced.add_argument("--robust", type=float, default=0.0)
    advanced.add_argument("--minuv", type=float, default=0)
    advanced.add_argument("--freqavg", type=int, default=-1)
    advanced.add_argument("--timeavg", type=int, default=-1)
    advanced.add_argument("--image_freqres", type=int, default=-1)
    advanced.add_argument("--image_timeres", type=int, default=-1)
    advanced.add_argument("--pol", type=str, default="I")
    advanced.add_argument("--apply_parang", action="store_true")
    advanced.add_argument("--clean_threshold", type=float, default=1.0)
    advanced.add_argument("--use_multiscale", action="store_true")
    advanced.add_argument("--use_solar_mask", action="store_true")
    advanced.add_argument("--cutout_rsun", type=float, default=2.5)
    advanced.add_argument("--make_overlay", action="store_true")
    advanced.add_argument("--cpu_frac", type=float, default=0.8)
    advanced.add_argument("--mem_frac", type=float, default=0.8)
    advanced.add_argument("--n_nodes", type=int, default=1)
    advanced.add_argument("--keep_backup", action="store_true")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    args = parser.parse_args()
    
    master_control(
        msname=args.msname,
        workdir=args.workdir,
        solar_data=args.solar_data,
        do_forcereset_weightflag=args.do_forcereset_weightflag,
        do_cal_partition=args.do_cal_partition,
        do_cal_flag=args.do_cal_flag,
        do_import_model=args.do_import_model,
        do_basic_cal=args.do_basic_cal,
        do_noise_cal=args.do_noise_cal,
        do_applycal=args.do_applycal,
        do_target_split=args.do_target_split,
        do_polcal=args.do_polcal,
        do_selfcal=args.do_selfcal,
        do_selfcal_split=args.do_selfcal_split,
        do_apply_selfcal=args.do_apply_selfcal,
        do_ap_selfcal=args.do_ap_selfcal,
        solar_selfcal=args.solar_selfcal,
        target_scans=args.target_scans,
        freqrange=args.freqrange,
        timerange=args.timerange,
        uvrange=args.uvrange,
        solint=args.solint,
        do_imaging=args.do_imaging,
        do_pbcor=args.do_pbcor,
        weight=args.weight,
        robust=args.robust,
        minuv=args.minuv,
        freqavg=args.freqavg,
        timeavg=args.timeavg,
        image_freqres=args.image_freqres,
        image_timeres=args.image_timeres,
        pol=args.pol,
        apply_parang=args.apply_parang,
        clean_threshold=args.clean_threshold,
        use_multiscale=args.use_multiscale,
        use_solar_mask=args.use_solar_mask,
        cutout_rsun=args.cutout_rsun,
        make_overlay=args.make_overlay,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        n_nodes=args.n_nodes,
        keep_backup=args.keep_backup,
    )

if __name__ == "__main__":
    main()
    
    
