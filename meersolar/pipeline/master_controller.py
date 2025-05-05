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


def run_partion(msname, workdir, cpu_frac=0.8, mem_frac=0.8):
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
    split_cmd = f"run_partition --msname {msname} --outputms {calibrator_ms} --scans {cal_scans} --timebin {timebin} --width {width} --cpu_frac {cpu_frac} --mem_frac {mem_frac}"
    ####################################
    # Partition fields
    ####################################
    print("\n########################################")
    basename = f"partition_cal"
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
            + " --spectral_chunk "
            + str(target_freq_chunk)
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


def run_apply_basiccal_sol(
    target_mslist,
    workdir,
    caldir,
    use_only_bandpass=False,
    overwrite_datacolumn=True,
    applymode="calflag",
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
            +" --applymode "
            + str(applymode)
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
            +" --applymode "
            + str(applymode)
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
    cpu_frac=0.8,
    mem_frac=0.8,
    start_thresh=5.0,
    stop_thresh=3.0,
    max_iter=100,
    max_DR=1000,
    min_iter=2,
    conv_frac=0.3,
    solint="int",
    do_apcal=True,
    solar_selfcal=True,
    keep_backup=False,
    uvrange="",
    minuv=0,
    weight="briggs",
    robust=0.0,
    applymode="calonly",
    gaintype="G",
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
        selfcal_cmd = (
            "run_selfcal --mslist "
            + ",".join(mslist)
            + " --workdir "
            + workdir
            + " --cpu_frac "
            + str(cpu_frac)
            + " --mem_frac "
            + str(mem_frac)
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
        if weight != "":
            selfcal_cmd += " --weight " + str(weight)
        if robust != "":
            selfcal_cmd += " --robust " + str(robust)
        if applymode != "":
            selfcal_cmd += " --applymode " + str(applymode)
        if gaintype != "":
            selfcal_cmd += " --gaintype " + str(gaintype)
        if min_fractional_bw > 0:
            selfcal_cmd += " --min_fractional_bw " + str(min_fractional_bw)
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        batch_file = create_batch_script_nonhpc(selfcal_cmd, workdir, selfcal_basename, write_logfile = False)
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
    cpu_frac=0.8,
    mem_frac=0.8,
    minuv=-1,
    weight="briggs",
    robust=0.0,
    pol="IQUV",
    freqres=-1,
    timeres=-1,
    threshold=1.0,
    multiscale_scales=[],
    use_solar_mask=True,
    savemodel=False,
    saveres=False,
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
    threshold : float, optional
        CLEAN threshold
    multiscale_scales : list, optional
        Multiscale scales in pixel unit
    use_solar_mask : bool, optional
        Use solar mask or not
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
        )
        if len(multiscale_scales) > 0:
            multiscales = ",".join([str(i) for i in multiscale_scales])
            imaging_cmd += " --multiscale_scales " + multiscales
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        batch_file = create_batch_script_nonhpc(imaging_cmd, workdir, imaging_basename, write_logfile=False)
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

def exit_job(start_time,mspath="",workdir=""):
    print(f"Total time taken: {round(time.time()-start_time,2)}s.")
    if mspath!="" and os.path.exists(mspath + "/dask-scratch-space"):
        os.system("rm -rf " + mspath + "/dask-scratch-space")
    if workdir!="" and os.path.exists(workdir + "/dask-scratch-space"):
        os.system("rm -rf " + workdir + "/dask-scratch-space")

def master_control(
    msname,
    workdir,
    solar_data=True,
    do_reset_weight_flag=True,
    do_cal_partition=True,
    do_cal_flag=True,
    do_import_model=True,
    do_basic_cal=True,
    do_noise_cal=True,
    do_applycal=True,
    do_target_split=True,
    target_freq_chunk=-1,
    do_selfcal=True,
    do_apply_selfcal=True,
    do_ap_selfcal=True,
    solar_selfcal=True,
    uvrange="",
    solint="int",
    do_imaging=True,
    weight="briggs",
    robust=0.0,
    minuv=0,
    freqavg=-1,
    timeavg=-1,
    image_freqres=-1,
    image_timeres=-1,
    pol="IQUV",
    clean_threshold=1.0,
    multiscale_scales=[],
    use_solar_mask=True,
    cpu_frac=0.8,
    mem_frac=0.8,
    split_target_frac=0.2,
    n_nodes=1,
    keep_backup=False,
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
    do_reset_weight_flag : bool, optional
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
    target_freq_chunk : float, optional
        Spectral width of each target chunk in MHz.
        Note image frequency resolution can not be set more than this. If full-band imaging is required, do not use this parameter.
    do_selfcal : bool, optional
        Perform self-calibration
    do_apply_selfcal : bool, optonal
        Apply self-calibration solutions
    do_ap_selfcal : bool, optional
        Perform amplitude-phase self-cal or not
    solar_selfcal : bool, optional
        Whether self-calibration is performing on solar observation or not
    uvrange : str, optional
        UV-range for calibration
    solint : str, optional
        Solution intervals in self-cal
    do_imaging : bool, optional
        Perform final imaging
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
        Stokes parameters of final imaging
    clean_threshold : float, optional
        CLEAN threshold of final imaging
    multiscale_scales : list, optional
        Multiscale scales in pixel unit
    use_solar_mask : bool, optional
        Use solar mask
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    split_target_frac : float, optional
        Fraction of compute resource to use for target data spliting
    n_nodes: int, optional
        Number of nodes to use (Only for cluster architechture)
    keep_backup : bool, optional
        Keep backup of self-cal rounds and final models and residual images
    Returns
    -------
    int
        Success message
    """
    start_time = time.time()
    print("###########################")
    print("Starting the pipeline .....")
    print("###########################\n")
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
    if workdir[-1] == "/":
        workdir = workdir[:-1]
    if os.path.exists(workdir) == False:
        os.makedirs(workdir)
    os.chdir(workdir)
    caldir = workdir + "/caltables"
    cpu_frac_bkp=copy.deepcopy(cpu_frac)
    mem_frac_bkp=copy.deepcopy(mem_frac)
    
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

    ##########################################################
    # Determining maximum allowed time and frequency averaging
    ##########################################################
    if freqavg > 0:
        max_freqres = calc_bw_smearing_freqwidth(msname)
        if freqavg > max_freqres:
            freqavg = round(max_freqres, 2)
        if image_freqres > 0 and freqavg>image_freqres:
            freqavg = image_freqres
    if timeavg > 0:
        max_timeres = min(
            calc_time_smearing_timewidth(msname), max_time_solar_smearing(msname)
        )
        if timeavg > max_timeres:
            timeavg = round(max_timeres, 2)
        if image_timeres > 0 and timeavg>image_timeres:
            timeavg = image_timeres
            
    ###################################################
    # Target spliting spectral and temporal chunks
    ##################################################
    if target_freq_chunk>0 and image_freqres<0:
        target_freq_chunk=-1
    elif target_freq_chunk<image_freqres:
        print (f"Requested target ms spectral chunk is : {target_freq_chunk} MHz is smaller than image spectral chnuk: {image_freqres} MHz.")
        print ("Resetting target ms spectral chunk equal to image spectral chunk.\n")
        target_freq_chunk=image_freqres
    if image_timeres> (25*60): # If more than 25 minute
        print (f"Image time integration is more than 25 minute, which cause issue in primary beam effects.")
        print ("Resetting to maximum 25 minutes averaging.\n")
        image_timeres= 25*60
    elif image_timeres<0:
        image_timeres=25*60

    #############################
    # Reset any previous weights
    ############################
    if do_reset_weight_flag:
        cpu_usage = psutil.cpu_percent(interval=1)  # Average over 1 second
        total_cpus = psutil.cpu_count(logical=True)
        available_cpus = int(total_cpus * (1 - cpu_usage / 100.0))
        available_cpus = max(1, available_cpus)  # Avoid zero workers
        reset_weights_and_flags(msname, n_threads=available_cpus)

    ########################################
    # Run noise-diode based flux calibration
    ########################################
    if do_noise_cal:
        msg = run_noise_diode_cal(
            msname,
            workdir,
            cpu_frac=round(cpu_frac, 2),
            mem_frac=round(mem_frac, 2),
        )  # Run noise diode based flux calibration
        if msg != 0:
            print(
                "!!!! WARNING: Error in running noise-diode based flux calibration. Not continuing further. !!!!"
            )
            exit_job(start_time,mspath,workdir)
            return 1

    #########################################
    # Spliting target scans
    #########################################
    split_start=False
    if do_target_split and (cpu_frac>0.5 or mem_frac>0.5):
        if round((1-split_target_frac)*cpu_frac, 2)>=0.5 and round((1-split_target_frac)*mem_frac, 2)>=0.5:
            msg = run_target_split_jobs(
                msname,
                workdir,
                datacolumn="data",
                timeres=timeavg,
                freqres=freqavg,
                target_freq_chunk=target_freq_chunk,
                cpu_frac=round(split_target_frac*cpu_frac, 2),
                mem_frac=round(split_target_frac*mem_frac, 2),
            )
            cpu_frac=cpu_frac*(1-split_target_frac)
            mem_frac=mem_frac*(1-split_target_frac)
            if msg != 0:
                print("!!!! WARNING: Error in running spliting target scans. !!!!")
            split_start=True
                    
    ##############################
    # Run partitioning jobs
    ##############################
    if do_target_split and check_status(workdir, "split_targets") and split_start:
        cpu_frac=copy.deepcopy(cpu_frac_bkp)
        mem_frac=copy.deepcopy(mem_frac_bkp)
    calibrator_msname = workdir + "/calibrator.ms"    
    if do_cal_partition or os.path.exists(calibrator_msname) == False:
        msg = run_partion(
            msname,
            workdir,
            cpu_frac=round(cpu_frac, 2),
            mem_frac=round(mem_frac, 2),
        )
        if msg != 0:
            print("!!!! WARNING: Error in partitioning calibrator fields. !!!!")
            exit_job(start_time,mspath,workdir)
            return 1
    
    ##################################
    # Run flagging jobs on calibrators
    ##################################
    if do_target_split and check_status(workdir, "split_targets") and split_start:
        cpu_frac=copy.deepcopy(cpu_frac_bkp)
        mem_frac=copy.deepcopy(mem_frac_bkp)
    if do_cal_flag:
        if os.path.exists(calibrator_msname)==False:
            print (f"Calibrator ms: {calibrator_ms} is not present.")
            exit_job(start_time,mspath,workdir)
            return 1
        msg = run_flag(
            calibrator_msname,
            workdir,
            flag_calibrators=True,
            cpu_frac=round(cpu_frac, 2),
            mem_frac=round(mem_frac, 2),
        )
        if msg != 0:
            print("!!!! WARNING: Flagging error. !!!!")
            exit_job(start_time,mspath,workdir)
            return 1
                           
    #################################
    # Import model
    #################################
    if do_target_split and check_status(workdir, "split_targets") and split_start:
        cpu_frac=copy.deepcopy(cpu_frac_bkp)
        mem_frac=copy.deepcopy(mem_frac_bkp)
    if do_import_model:
        if os.path.exists(calibrator_msname)==False:
            print (f"Calibrator ms: {calibrator_ms} is not present.")
            exit_job(start_time,mspath,workdir)
            return 1
        fluxcal_fields, fluxcal_scans = get_fluxcals(calibrator_msname)
        phasecal_fields, phasecal_scans, phasecal_fluxes = get_phasecals(
            calibrator_msname
        )
        calibrator_field = fluxcal_fields + phasecal_fields
        msg = run_import_model(
            calibrator_msname,
            workdir,
            cpu_frac=round(cpu_frac, 2),
            mem_frac=round(mem_frac, 2),
        )  # Run model import
        if msg != 0:
            print(
                "!!!! WARNING: Error in importing calibrator models. Not continuing calibration. !!!!"
            )
            exit_job(start_time,mspath,workdir)
            return 1
      
    ###############################
    # Run basic calibration
    ###############################
    if do_target_split and check_status(workdir, "split_targets") and split_start:
        cpu_frac=copy.deepcopy(cpu_frac_bkp)
        mem_frac=copy.deepcopy(mem_frac_bkp)
    use_only_bandpass = False
    if do_basic_cal:
        if os.path.exists(calibrator_msname)==False:
            print (f"Calibrator ms: {calibrator_ms} is not present.")
            exit_job(start_time,mspath,workdir)
            return 1
        msg = run_basic_cal(
            calibrator_msname,
            workdir,
            cpu_frac=round(cpu_frac, 2),
            mem_frac=round(mem_frac, 2),
        )  # Run basic calibration
        if msg != 0:
            print(
                "!!!! WARNING: Error in basic calibration. Not continuing further. !!!!"
            )
            exit_job(start_time,mspath,workdir)
            return 1
            
    ##########################################
    # Checking presence of necessary caltables
    ##########################################
    if len(glob.glob(caldir + "/*.bcal")) == 0:
        print(f"No bandpass table is present in calibration directory : {caldir}.")
        exit_job(start_time,mspath,workdir)
        return 1
    if len(glob.glob(caldir + "/*.gcal")) == 0:
        print(
            f"No time-dependent gaintable is present in calibration directory : {caldir}. Applying only bandpass solutions."
        )
        use_only_bandpass = True    
        
    #############################################
    # Check spliting target scans finished or not
    #############################################
    if do_target_split:
        if split_start==False:
            msg = run_target_split_jobs(
                msname,
                workdir,
                datacolumn="data",
                timeres=timeavg,
                freqres=freqavg,
                target_freq_chunk=target_freq_chunk,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
            )
            if msg != 0:
                print("!!!! WARNING: Error in running spliting target scans. !!!!")
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
        else:
            print(
                "!!!! WARNING: Error in spliting target scans. Not continuing further. !!!!"
            )
            exit_job(start_time,mspath,workdir)
            return 1
    
    cpu_frac=copy.deepcopy(cpu_frac_bkp)
    mem_frac=copy.deepcopy(mem_frac_bkp)
    target_mslist = glob.glob(workdir + "/target_scan*.ms")
    if len(target_mslist) == 0:
        print("No splited target scan ms are available in work directory.")
        exit_job(start_time,mspath,workdir)
        return 1
    print(f"Target scan mslist : {[os.path.basename(i) for i in target_mslist]}")

    #################################################################################
    # Sorting only single frequency chunk ms for applycal if perform self-calibration
    #################################################################################
    target_mslist = sorted(target_mslist)
    if do_selfcal:
        selfcal_mslist = []
        spw_list = []
        for ms in target_mslist:
            ms_spw = os.path.basename(ms).rstrip(".ms").split("_")[-1]
            if ms_spw not in spw_list:
                spw_list.append(ms_spw)
        chosen_spw = spw_list[int(len(spw_list) / 2)]
        for ms in target_mslist:
            if chosen_spw in os.path.basename(ms):
                selfcal_mslist.append(ms)
        if len(selfcal_mslist)==0:
            print ("No measurement set for self-calibration.")
            do_selfcal=False
            
    #########################################################
    # Applying solutions on target scans for self-calibration
    #########################################################
    if (
        do_selfcal or do_applycal
    ):  # Applying solutions if do_selfcal is True even if do_applycal is False. This is for safety.
        if len(applycal_list) > 0:
            caldir = workdir + "/caltables"
            msg = run_apply_basiccal_sol(
                target_mslist,
                workdir,
                caldir,
                use_only_bandpass=use_only_bandpass,
                overwrite_datacolumn=True,
                applymode="calflag",
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
            )
            if msg != 0:
                print(
                    "!!!! WARNING: Error in applying basic calibration solutions on target scans. Not continuing further. !!!!"
                )
                exit_job(start_time,mspath,workdir)
                return 1
        else:
            print(
                "!!!! WARNING: No measurement set is present for basic calibration applying solutions. !!!!"
            )
            exit_job(start_time,mspath,workdir)
            return 1

    ########################################
    # Performing self-calibration
    ########################################
    if do_selfcal:
        os.system("rm -rf "+workdir+"/*selfcal "+workdir+"/caltables/*selfcal*")
        msg = run_selfcal_jobs(
            selfcal_mslist,
            workdir,
            cpu_frac=round(cpu_frac, 2),
            mem_frac=round(mem_frac, 2),
            solint=solint,
            do_apcal=do_ap_selfcal,
            solar_selfcal=solar_selfcal,
            keep_backup=keep_backup,
            uvrange=uvrange,
            minuv=50,
            weight=weight,
            robust=robust,
        )
        if msg != 0:
            print(
                "!!!! WARNING: Error in self-calibration on target scans. Not applying self-calibration. !!!!"
            )
            do_apply_selfcal = False

    ########################################
    # Checking self-cal caltables
    ########################################
    selfcaldir= workdir + "/caltables/*selfcal*"
    if len(selfcaldir)==0:
        print ("Self-calibration is not performed and no self-calibration caltable is available.")
        do_apply_selfcal=False
        
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
        msg = run_imaging_jobs(
            target_mslist,
            workdir,
            cpu_frac=round(cpu_frac, 2),
            mem_frac=round(mem_frac, 2),
            minuv=minuv,
            weight=weight,
            robust=float(robust),
            pol=pol,
            freqres=image_freqres,
            timeres=image_timeres,
            threshold=float(clean_threshold),
            multiscale_scales=multiscale_scales,
            use_solar_mask=use_solar_mask,
            savemodel=keep_backup,
            saveres=keep_backup,
        )
        if msg != 0:
            print(
                "!!!! WARNING: Final imaging on all measurement sets is not successful. Check the image directory. !!!!"
            )
            exit_job(start_time,mspath,workdir)
            return 1

    print(
        f"Calibration and imaging pipeline is successfully run on measurement set : {msname}\n"
    )
    exit_job(start_time,mspath,workdir)
    return 0
