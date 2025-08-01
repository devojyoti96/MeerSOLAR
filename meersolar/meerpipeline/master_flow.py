import os
import logging
import psutil
import numpy as np
import argparse
import traceback
import copy
import time
import glob
import sys
import os
import socket
import threading
from casatasks import casalog
from casatools import msmetadata
from datetime import datetime as dt
from multiprocessing import Process, Event
from meersolar.utils import *
from prefect import flow, task
from prefect.context import get_run_context
from prefect_dask.task_runners import DaskTaskRunner

logging.getLogger("distributed").setLevel(logging.WARNING)

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    pass

datadir = get_datadir()

   
@task(name="making_dynamic_spectra",retries=2,retry_delay_seconds=10,log_prints=True)
def run_ds_jobs(
    msname,
    workdir,
    outdir,
    target_scans=[],
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    dask_addr=None,
):
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
    remote_log: bool, optional
        Start remote logger
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message
    """
    ds_basename = "ds_targets"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{ds_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id=str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread = start_log_task_saver(task_id, task_name, logfile, poll_interval=3, stop_event=stop_event)
    try:
        ##################
        print("###########################")
        print("Making dynamic spectra of target scans .....")
        print("###########################")
        ##########################
        # Making dynamic spectrum
        ##########################
        from meersolar.meerpipeline import meer_make_ds
        msg = meer_make_ds.main(
            msname,
            workdir,
            outdir,
            target_scans=target_scans,
            cpu_frac=float(cpu_frac),
            mem_frac=float(mem_frac),
            logfile=logfile,
            jobid=jobid,
            start_remote_log=remote_log,
            dask_addr=dask_addr,
        )
    finally:    
        stop_event.set()
        log_thread.join(timeout=5)
    if msg!=0:
        raise RuntimeError("Dynamic spectrum making is failed.")
    else:
        return msg


@task(name="attenuation_calibration",retries=2,retry_delay_seconds=10,log_prints=True)
def run_noise_diode_cal(
    msname,
    workdir,
    caldir,
    keep_backup=False,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    dask_addr=None,
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
    remote_log: bool, optional
        Start remote logger
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message for noise diode based flux calibration
    """
    msname = msname.rstrip("/")
    noisecal_basename = "noise_cal"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{noisecal_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id=str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread = start_log_task_saver(task_id, task_name, logfile, poll_interval=3, stop_event=stop_event)
    try:
        #################
        print("###########################")
        print("Performing noise diode based flux calibration .....")
        print("###########################")
        #################
        # Attenuation calibration
        #################
        from meersolar.meerpipeline import do_fluxcal
        msg=do_fluxcal.main(
            msname,
            workdir,
            caldir,
            keep_backup=keep_backup,
            start_remote_log=remote_log,
            cpu_frac=float(cpu_frac),
            mem_frac=float(mem_frac),
            logfile=logfile,
            jobid=jobid,
            dask_addr=dask_addr,
        )
    finally:
        stop_event.set()
        log_thread.join(timeout=5)
    if msg!=0:
        raise RuntimeError("Attenuation calibration is failed.")
    else:
        return msg
    

@task(name="partitioning_calibrator",retries=2,retry_delay_seconds=10,log_prints=True)
def run_partition(
    msname,
    workdir,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    dask_addr=None,
):
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
    remote_log: bool, optional
        Start remote logger
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message
    """
    msname = msname.rstrip("/")
    partition_basename = f"partition_cal"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{partition_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id=str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread = start_log_task_saver(task_id, task_name, logfile, poll_interval=3, stop_event=stop_event)
    try:
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
        ##############
        print("###########################")
        print("Partitioning measurement set ...")
        print("###########################")
        ###################
        # Paritioning calibrator scans
        ###################
        from meersolar.meerpipeline import do_partition
        msg = do_partition.main(
            msname,
            outputms=calibrator_ms,
            workdir=workdir,
            scans=cal_scans,
            width=int(width),
            timebin=timebin,
            datacolumn="data",
            cpu_frac=float(cpu_frac),
            mem_frac=float(mem_frac),
            logfile=logfile,
            jobid=jobid,
            start_remote_log=remote_log,
            dask_addr=dask_addr,
        )
    finally:
        stop_event.set()
        log_thread.join(timeout=5)
    if msg!=0:
        raise RuntimeError("Partitioning calibrator scans is failed.")
    else:
        return msg


@task(name="spliting_target_scans",retries=2,retry_delay_seconds=10,log_prints=True)
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
    merge_spws=False,
    time_window=-1,
    time_interval=-1,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    max_cpu_frac=0.8,
    max_mem_frac=0.8,
    remote_log=False,
    dask_addr=None,
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
    remote_log: bool, optional
        Start remote logger
    dask_addr : str, optional
        Dask scheduler address
        
    Returns
    -------
    int
        Success message for spliting target scans
    """
    msname = msname.rstrip("/")
    split_basename = f"split_{prefix}"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{split_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id=str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread = start_log_task_saver(task_id, task_name, logfile, poll_interval=3, stop_event=stop_event)
    try:
        ############
        print("###########################")
        print(f"Spliting {prefix} scans .....")
        print("###########################")
        ##################
        # Spliting target scans
        ##################
        scans=",".join([str(s) for s in target_scans])
        from meersolar.meerpipeline import do_target_split
        msg = do_target_split.main(
            msname,
            workdir=workdir,
            datacolumn=datacolumn,
            spw=spw,
            scans=scans,
            time_window=time_window,
            time_interval=time_interval,
            spectral_chunk=target_freq_chunk,
            n_spectral_chunk=n_spectral_chunk,
            freqres=freqres,
            timeres=timeres,
            prefix=prefix,
            merge_spws=merge_spws,
            cpu_frac=float(cpu_frac),
            mem_frac=float(mem_frac),
            max_cpu_frac=float(max_cpu_frac),
            max_mem_frac=float(max_mem_frac),
            logfile=logfile,
            jobid=jobid,
            start_remote_log=remote_log,
            dask_addr=dask_addr,
        )
    finally:
        stop_event.set()
        log_thread.join(timeout=5)
    if msg!=0:
        raise RuntimeError("Spliting target scans is failed.")
    else:
        return msg

    
@task(name="flagging",retries=2,retry_delay_seconds=10,log_prints=True)
def run_flag(
    msname,
    workdir,
    flag_calibrators=True,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    dask_addr=None,
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
    remote_log: bool, optional
        Start remote logger
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message
    """
    msname = msname.rstrip("/")
    if flag_calibrators:
        flagdimension = "freqtime"
        flagfield_type = "cal"
    else:
        flagdimension = "freq"
        flagfield_type = "target"
    flag_basename = (
        f"flagging_{flagfield_type}_" + os.path.basename(msname).split(".ms")[0]
    )
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{flag_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id=str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread = start_log_task_saver(task_id, task_name, logfile, poll_interval=3, stop_event=stop_event)
    try:
        ##############
        print("###########################")
        print("Flagging ....")
        print("###########################")
        ########################
        # Calibrator ms flagging
        ########################
        from meersolar.meerpipeline import flagging
        msg = flagging.main(
            msname,
            workdir=workdir,
            datacolumn="DATA",
            flag_bad_ants=True,
            flag_bad_spw=True,
            use_tfcrop=True,
            use_rflag=False,
            flag_autocorr=True,
            flagbackup=True,
            flagdimension=flagdimension,
            cpu_frac=float(cpu_frac),
            mem_frac=float(cpu_frac),
            logfile=logfile,
            jobid=jobid,
            start_remote_log=remote_log,
            dask_addr=dask_addr,
        )
    finally:
        stop_event.set()
        log_thread.join(timeout=5)
    if msg!=0:
        raise RuntimeError("Calibrator flagging is failed.")
    else:
        return msg


@task(name="importing_model_visibilities",retries=2,retry_delay_seconds=10,log_prints=True)
def run_import_model(
    msname,
    workdir,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    dask_addr=None,
):
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
    remote_log: bool, optional
        Start remote logger
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message
    """
    msname = msname.rstrip("/")
    model_basename = "modeling_" + os.path.basename(msname).split(".ms")[0]
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{model_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id=str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread = start_log_task_saver(task_id, task_name, logfile, poll_interval=3, stop_event=stop_event)
    try:
        ##############
        print("###########################")
        print("Importing model visibilities ....")
        print("###########################")
        ########################
        # Calibrator ms flagging
        ########################
        from meersolar.meerpipeline import import_model
        msg = import_model.main(
            msname,
            workdir=workdir,
            cpu_frac=float(cpu_frac),
            mem_frac=float(mem_frac),
            logfile=logfile,
            jobid=jobid,
            start_remote_log=remote_log,
            dask_addr=dask_addr,
        )
    finally:
        stop_event.set()
        log_thread.join(timeout=5)
    if msg!=0:
        raise RuntimeError("Importing calibrator model is failed.")
    else:
        return msg
    
    
@task(name="basic_calibration",retries=2,retry_delay_seconds=10,log_prints=True)
def run_basic_cal_jobs(
    msname,
    workdir,
    caldir,
    perform_polcal=False,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    keep_backup=False,
    remote_log=False,
    dask_addr=None,
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
    remote_log: bool, optional
        Start remote logger
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message for basic calibration
    """
    msname = msname.rstrip("/")
    cal_basename = "basic_cal"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{cal_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id=str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread = start_log_task_saver(task_id, task_name, logfile, poll_interval=3, stop_event=stop_event)
    try:
        ##############
        print("###########################")
        print("Performing basic calibration .....")
        print("###########################")
        ########################
        # Basic calibration
        ########################
        from meersolar.meerpipeline import basic_cal
        msg = basic_cal.main(
            msname,
            workdir,
            caldir,
            perform_polcal=perform_polcal,
            keep_backup=keep_backup,
            start_remote_log=remote_log,
            cpu_frac=float(cpu_frac),
            mem_frac=float(mem_frac),
            logfile=logfile,
            jobid=jobid,
            dask_addr=dask_addr,
        )
    finally:
        stop_event.set()
        log_thread.join(timeout=5)
    if msg!=0:
        raise RuntimeError("Basic calibration is failed.")
    else:
        return msg
    

@task(name="applying_basic_calibration",retries=2,retry_delay_seconds=10,log_prints=True)
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
    remote_log=False,
    dask_addr=None,
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
    remote_log: bool, optional
        Start remote logger
    dask_addr: str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message for applying calibration solutions and spliting target scans
    """
    mslist = ",".join(target_mslist)
    applycal_basename = "apply_basiccal"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{applycal_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id=str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread = start_log_task_saver(task_id, task_name, logfile, poll_interval=3, stop_event=stop_event)
    try:
        ######################
        print("###########################")
        print("Applying basic calibration solutions on target scans .....")
        print("###########################")
        ######################
        # Applying basic calibration
        ######################
        from meersolar.meerpipeline import do_apply_basiccal
        msg = do_apply_basiccal.main(
            mslist,
            workdir,
            caldir,
            use_only_bandpass=use_only_bandpass,
            applymode=applymode,
            overwrite_datacolumn=overwrite_datacolumn,
            start_remote_log=remote_log,
            cpu_frac=float(cpu_frac),
            mem_frac=float(mem_frac),
            logfile=logfile,
            jobid=jobid,
            dask_addr=dask_addr,
        )
    finally:
        stop_event.set()
        log_thread.join(timeout=5)
    if msg!=0:
        raise RuntimeError("Applying basic calibration solutions is failed.")
    else:
        return msg
  
        
@task(name="solar_sidereal_correction",retries=2,retry_delay_seconds=10,log_prints=True)
def run_solar_siderealcor_jobs(
    mslist,
    workdir,
    prefix="targets",
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    max_cpu_frac=0.8,
    max_mem_frac=0.8,
    remote_log=False,
    dask_addr=None,
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
    remote_log: bool, optional
        Start remote logger
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message
    """
    mslist = ",".join(mslist)
    sidereal_basename = f"cor_sidereal_{prefix}"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{sidereal_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id=str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread = start_log_task_saver(task_id, task_name, logfile, poll_interval=3, stop_event=stop_event)
    try:
        #######################
        print("###########################")
        print("Correcting sidereal motion .....")
        print("###########################")
        #######################
        # Sidereal motion correction
        #######################
        from meersolar.meerpipeline import do_sidereal_cor
        msg = do_sidereal_cor.main(
            mslist,
            workdir=workdir,
            cpu_frac=float(cpu_frac),
            mem_frac=float(mem_frac),
            max_cpu_frac=float(max_cpu_frac),
            max_mem_frac=float(max_mem_frac),
            logfile=logfile,
            jobid=jobid,
            start_remote_log=remote_log,
            dask_addr=dask_addr,
        )
    finally:
        stop_event.set()
        log_thread.join(timeout=5)
    if msg!=0:
        raise RuntimeError("Solar sidereal motion correction is failed.")
    else:
        return msg
    
    
@task(name="selfcal",retries=2,retry_delay_seconds=10,log_prints=True)
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
    remote_log=False,
    dask_addr=None,
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
    remote_log: bool, optional
        Start remote logger
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message for self-calibration
    """
    mslist = ",".join(mslist)
    selfcal_basename = "selfcal_targets"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{selfcal_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id=str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread = start_log_task_saver(task_id, task_name, logfile, poll_interval=3, stop_event=stop_event)
    try:
        ########################
        print("###########################")
        print("Performing self-calibration of target scans .....")
        print("###########################")
        ########################
        # Selfcal jobs
        ########################
        from meersolar.meerpipeline import do_selfcal
        msg = do_selfcal.main(
            mslist,
            workdir,
            caldir,
            start_thresh=float(start_thresh),
            stop_thresh=float(stop_thresh),
            max_iter=float(max_iter),
            max_DR=float(max_DR),
            min_iter=float(min_iter),
            conv_frac=float(conv_frac),
            solint=solint,
            uvrange=uvrange,
            minuv=float(minuv),
            weight=weight,
            robust=float(robust),
            applymode=applymode,
            min_tol_factor=float(min_tol_factor),
            do_apcal=do_apcal,
            solar_selfcal=solar_selfcal,
            keep_backup=keep_backup,
            cpu_frac=float(cpu_frac),
            mem_frac=float(mem_frac),
            jobid=jobid,
            start_remote_log=remote_log,
            dask_addr=dask_addr,
        )
    finally:
        stop_event.set()
        log_thread.join(timeout=5)
    if msg!=0:
        raise RuntimeError("Self-calibration is failed.")
    else:
        return msg
 

@task(name="applying_self-calibration",retries=2,retry_delay_seconds=10,log_prints=True)
def run_apply_selfcal_sol(
    target_mslist,
    workdir,
    caldir,
    overwrite_datacolumn=True,
    applymode="calflag",
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    dask_addr=None,
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
    remote_log: bool, optional
        Start remote logger
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message for applying calibration solutions and spliting target scans
    """
    mslist = ",".join(target_mslist)
    applycal_basename = "apply_selfcal"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{applycal_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id=str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread = start_log_task_saver(task_id, task_name, logfile, poll_interval=3, stop_event=stop_event)
    try:
        ##################
        print("###########################")
        print("Applying self-calibration solutions on target scans .....")
        print("###########################")
        ########################
        # Applying self-calibration
        ########################
        from meersolar.meerpipeline import do_apply_selfcal
        msg = do_apply_selfcal.main(
            mslist,
            workdir,
            caldir,
            applymode=applymode,
            overwrite_datacolumn=overwrite_datacolumn,
            start_remote_log=remote_log,
            cpu_frac=float(cpu_frac),
            mem_frac=float(mem_frac),
            logfile=logfile,
            jobid=jobid,
            dask_addr=dask_addr,
        )
    finally:
        stop_event.set()
        log_thread.join(timeout=5)
    if msg!=0:
        raise RuntimeError("Applying self-calibration solutions is failed.")
    else:
        return msg
    
        
@task(name="imaging",retries=2,retry_delay_seconds=10,log_prints=True)
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
    remote_log=False,
    dask_addr=None,
):
    """
    Imaging on target scans

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
    remote_log: bool, optional
        Start remote logger
    dask_addr : str, optional
        Dask scheduler address 

    Returns
    -------
    int
        Success message for imaging
    """
    mslist = ",".join(mslist)
    imaging_basename = "imaging_targets"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{imaging_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id=str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread = start_log_task_saver(task_id, task_name, logfile, poll_interval=3, stop_event=stop_event)
    try:
        ######################
        print("###########################")
        print("Performing imaging of target scans .....")
        print("###########################")
        #######################
        # Performing imaging
        #######################
        from meersolar.meerpipeline import do_imaging
        msg = do_imaging.main(
            mslist,
            workdir,
            outdir,
            freqrange=freqrange,
            timerange=timerange,
            pol=pol,
            freqres=float(freqres),
            timeres=float(timeres),
            weight=weight,
            robust=float(robust),
            minuv=float(minuv),
            threshold=float(threshold),
            band=band,
            cutout_rsun=float(cutout_rsun),
            use_multiscale=use_multiscale,
            use_solar_mask=use_solar_mask,
            savemodel=savemodel,
            saveres=saveres,
            make_overlay=make_overlay,
            start_remote_log=remote_log,
            cpu_frac=float(cpu_frac),
            mem_frac=float(mem_frac),
            jobid=jobid,
            dask_addr=dask_addr,
        )
    finally:
        stop_event.set()
        log_thread.join(timeout=5)
    if msg!=0:
        raise RuntimeError("Imaging is failed.")
    else:
        return msg
        
  
@task(name="applying_primary_beam",retries=2,retry_delay_seconds=10,log_prints=True)
def run_apply_pbcor(
    imagedir,
    workdir,
    apply_parang=True,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    remote_log=False,
    dask_addr=None,
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
    remote_log: bool, optional
        Start remote logger
    dask_scheduler : str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message for applying primary beam correction on all images
    """
    applypbcor_basename = "apply_pbcor"
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{applypbcor_basename}.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    task_id=str(ctx.task_run.id)
    task_name = ctx.task_run.name
    stop_event = Event()
    log_thread = start_log_task_saver(task_id, task_name, logfile, poll_interval=3, stop_event=stop_event)
    try:
        ###################
        print("###########################")
        print("Applying primary beam corrections on all images .....")
        print("###########################")
        #####################
        # Applying primary beam correction
        #####################
        from meersolar.meerpipeline import meer_pbcor
        msg = meer_pbcor.main(
            imagedir,
            workdir=workdir,
            apply_parang=apply_parang,
            cpu_frac=float(cpu_frac),
            mem_frac=float(mem_frac),
            logfile=logfile,
            jobid=jobid,
            start_remote_log=remote_log,
            dask_addr=dask_addr,
        )
    finally:
        stop_event.set()
        log_thread.join(timeout=5)
    if msg!=0:
        raise RuntimeError("Primary beam correction is failed.")
    else:
        return msg
   

@flow(name="MeerSOLAR Master control",version="3.0",description="Calibration and Imaging Pipeline for MeerKAT Solar Observation",log_prints=True)
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
    dask_addr=None,
    jobid=None,
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
    dask_addr : str, optional
        Dask scheduler address
    jobid : str, optional
        Job ID

    Returns
    -------
    int
        Success message
    """
    ###################################
    # Preparing working directories
    ###################################
    print("Preparing working directories....")
    if workdir == "":
        workdir = os.path.dirname(os.path.abspath(msname)) + "/workdir"
    workdir = workdir.rstrip("/")
    
    #################################
    # Setup logger
    #################################
    logdir = f"{workdir}/logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/main.log"
    if os.path.exists(logfile):
        os.remove(logfile)
    ctx = get_run_context()
    flow_id=str(ctx.flow_run.id)
    flow_name = ctx.flow_run.name
    stop_event = Event()
    log_thread = start_flow_log_saver(flow_id, flow_name, logfile, poll_interval=3, stop_event=stop_event)
    
    #####################################
    # Initiating meersolar data
    #####################################
    from meersolar.meerpipeline.init_data import init_meersolar_data
    init_meersolar_data()
    
    ###################################################
    # Measurement set check and other working directory
    ###################################################
    msname = os.path.abspath(msname.rstrip("/"))
    if os.path.exists(msname) == False:
        print("Please provide a valid measurement set location.")
        return 1
    valid_ms=check_datacolumn_valid(msname)
    if valid_ms is not True:
        print (f"Measurement set : {msname} is corrupted.")
        return 1
    mspath = os.path.dirname(msname)
    if outdir == "":
        outdir = workdir
    outdir = outdir.rstrip("/")
    caldir = outdir + "/caltables"
    caldir = caldir.rstrip("/")
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(caldir, exist_ok=True)
    
    try:
        ####################################
        # Job and process IDs
        ####################################
        pid = os.getpid()
        if jobid is None:
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
        print("###########################")
        print(f"MeerSOLAR Job ID: {jobid}")
        print(f"Work directory: {workdir}")
        print(f"Final product directory: {outdir}")
        print("###########################")
        #####################################
        # Moving into work directory
        #####################################
        os.chdir(workdir)
        if remote_logger:
            trial=0
            while trial<=5:
                remote_link = get_remote_logger_link()
                if remote_link != "":
                    break
                else:
                    time.sleep(5)
                    trial+=1
            if remote_link=="":
                print("Please provide a valid remote link.")
                remote_logger = False

        if not remote_logger:
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
                from meersolar.data.sendmail import (
                    send_paircars_notification as send_notification,
                )

                success_msg, error_msg = send_notification(
                    emails, email_subject, email_msg
                )
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
                from meersolar.data.sendmail import (
                    send_paircars_notification as send_notification,
                )

                success_msg, error_msg = send_notification(
                    emails, email_subject, email_msg
                )

        #####################################
        # Settings for solar data
        #####################################
        if solar_data:
            if not use_solar_mask:
                print("Use solar mask during CLEANing.")
                use_solar_mask = True
            if not solar_selfcal:
                solar_selfcal = True
            full_FoV = False
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
            full_FoV = True

        ##################################################
        # Target spliting spectral and temporal chunks
        ##################################################
        if image_timeres > (2 * 3660):  # If more than 2 hours
            print(
                f"Image time integration is more than 2 hours, which may cause smearing due to solar differential rotation."
            )

        #####################################################################
        # Checking if ms is full pol for polarization calibration and imaging
        #####################################################################
        if do_polcal:
            print(
                "Checking measurement set suitability for polarization calibration...."
            )
            msmd = msmetadata()
            msmd.open(msname)
            npol = msmd.ncorrforpol()[0]
            msmd.close()
            if npol < 4:
                print(
                    "Measurement set is not full-polar. Do not performing polarization analysis."
                )
                do_polcal = False

        #################################################
        # Determining maximum allowed frequency averaging
        #################################################
        print("Estimating optimal frequency averaging....")
        max_freqres = calc_bw_smearing_freqwidth(msname, full_FoV=full_FoV)
        if image_freqres > 0:
            freqavg = round(min(image_freqres, max_freqres), 1)
        else:
            freqavg = round(max_freqres, 1)

        ################################################
        # Determining maximum allowed temporal averaging
        ################################################
        print("Estimating optimal temporal averaging....")
        if solar_data:  # For solar data, it is assumed Sun is tracked.
            max_timeres = calc_time_smearing_timewidth(msname)
        else:
            max_timeres = min(
                calc_time_smearing_timewidth(msname), max_time_solar_smearing(msname)
            )
        if image_timeres > 0:
            timeavg = round(min(image_timeres, max_timeres), 1)
        else:
            timeavg = round(max_timeres, 1)

        #########################################
        # Target ms frequency chunk based on band
        #########################################
        print("Determing bad spectral channels....")
        bad_spws = get_bad_chans(msname)
        if bad_spws != "":
            bad_spws = bad_spws.split("0:")[-1].split(";")
            good_start = []
            good_end = []
            for i in range(len(bad_spws) - 1):
                start_chan = int(bad_spws[i].split("~")[-1]) + 1
                end_chan = int(bad_spws[i + 1].split("~")[0]) - 1
                good_start.append(start_chan)
                good_end.append(end_chan)
            start_chan = min(good_start)
            end_chan = max(good_end)
        else:
            msmd = msmetadata()
            msmd.open(msname)
            nchan = msmd.nchan(0)
            msmd.close()
            start_chan = 0
            end_chan = nchan
        spw = f"0:{start_chan}~{end_chan}"

        #############################################################
        # Determining numbers of spectral chunks for parallel imaging
        #############################################################
        print("Determining spectral chunks for parallel imaging....")
        if image_freqres < 0:
            target_freq_chunk = -1
            nchunk = 1
        else:
            msmd = msmetadata()
            msmd.open(msname)
            chanres = msmd.chanres(0, unit="MHz")[0]
            msmd.close()
            total_bw = chanres * (end_chan - start_chan)
            nchunk = int(total_bw / image_freqres)
            if nchunk > max(
                4, n_nodes
            ):  # Maximum 4 chunking or number of compute nodes
                nchunk = max(4, n_nodes)
                target_freq_chunk = total_bw / nchunk
            else:
                nchnuk = 1
                target_freq_chunk = image_freqres

        #############################
        # Reset any previous weights
        ############################
        print("Resetting previous flags and weights....")
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
        future_maskms=None
        if make_ds:
            future_maskms = run_ds_jobs.with_options(task_run_name=f"making_dynamic_spectra_{jobid}").submit(
                msname,
                workdir,
                outdir,
                jobid=jobid,
                target_scans=target_scans,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
                dask_addr=dask_addr,
            )
            msg=future_maskms.result()
            try:
                msg = future_maskms.result()
            except Exception as e:
                print("!!! WARNING : Error in making dynamic spectra. !!!")
                traceback.print_exc()
                msg = None
               
        ########################################
        # Run noise-diode based flux calibration
        ########################################
        future_noisecal=None
        if do_noise_cal:
            future_noisecal = run_noise_diode_cal.with_options(task_run_name=f"attenuation_calibration_{jobid}").submit(
                msname,
                workdir,
                caldir,
                jobid=jobid,
                keep_backup=keep_backup,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
                dask_addr=dask_addr,
            )  
            try:
                msg = future_noisecal.result()
                if os.path.exists(f"{workdir}/.noattcal"):
                    os.system(f"rm -rf {workdir}/.noattcal")
                os.system(f"touch {workdir}/.attcal")
            except Exception as e:
                print(
                    "!!!! WARNING: Error in running noise-diode based flux calibration. Flux density calibration may not be correct. !!!!"
                )
                traceback.print_exc()
                msg = None
                if os.path.exists(f"{workdir}/.attcal"):
                    os.system(f"rm -rf {workdir}/.attcal")
                os.system(f"touch {workdir}/.noattcal")     
        else:
            if os.path.exists(f"{workdir}/.attcal") == False:
                os.system(f"touch {workdir}/.noattcal")
            
        ##############################
        # Run partitioning jobs
        ##############################
        # If do partition or calibrator ms is not present in case of basic
        # calibration is requested
        calibrator_msname = workdir + "/calibrator.ms"
        future_partition = None
        if do_basic_cal and (
            do_cal_partition or os.path.exists(calibrator_msname) == False
        ):
            future_partition = run_partition.with_options(task_run_name=f"partitioning_calibrator_{jobid}").submit(
                msname,
                workdir,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
                dask_addr=dask_addr,
            )
            try:
                msg = future_partition.result()
            except Exception as e:
                print("!!!! WARNING: Error in partitioning calibrator fields. !!!!")
                traceback.print_exc()
                return 1
                 
        ############################################
        # Spliting for self-cals
        ############################################
        # Spliting only if self-cal is requested
        if not do_selfcal_split and do_selfcal:
            selfcal_target_mslist = glob.glob(workdir + "/selfcals_scan*.ms")
            if len(selfcal_target_mslist) == 0:
                print(
                    "No measurement set is present for self-calibration. Spliting them.."
                )
                do_selfcal_split = True
        future_selfcal_split=None
        if do_selfcal and do_selfcal_split:
            prefix = "selfcals"
            try:
                time_interval = float(solint)
            except BaseException:
                if "s" in solint:
                    time_interval = float(solint.split("s")[0])
                elif "min" in solint:
                    time_interval = float(solint.split("min")[0]) * 60
                elif solint == "int":
                    time_interval = image_timeres
                else:
                    time_interval = -1
            future_selfcal_split = run_target_split_jobs.with_options(task_run_name=f"spliting_{prefix}_scans_{jobid}").submit(
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
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                max_cpu_frac=round(cpu_frac, 2),
                max_mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
                dask_addr=dask_addr,
            ) 
            
        ##################################
        # Run flagging jobs on calibrators
        ##################################
        # Only if basic calibration is requested
        future_flag=None
        if do_cal_flag and do_basic_cal:
            if os.path.exists(calibrator_msname) == False:
                print(f"Calibrator ms: {calibrator_ms} is not present.")
                return 1
            future_flag = run_flag.with_options(task_run_name=f"flagging_{jobid}").submit(
                calibrator_msname,
                workdir,
                flag_calibrators=True,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
                dask_addr=dask_addr,
            )
            try:
                msg = future_flag.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Flagging error. Examine calibration solutions with caution. !!!!"
                )
                traceback.print_exc()
                msg=1
          
        #################################
        # Import model
        #################################
        # Only if basic calibration is requested
        future_import_model=None
        if do_import_model and do_basic_cal:
            if os.path.exists(calibrator_msname) == False:
                print(f"Calibrator ms: {calibrator_ms} is not present.")
                return 1
            fluxcal_fields, fluxcal_scans = get_fluxcals(calibrator_msname)
            phasecal_fields, phasecal_scans, phasecal_fluxes = get_phasecals(
                calibrator_msname
            )
            calibrator_field = fluxcal_fields + phasecal_fields
            future_import_model = run_import_model.with_options(task_run_name=f"importing_model_visibilities_{jobid}").submit(
                calibrator_msname,
                workdir,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
                dask_addr=dask_addr,
            ) 
            try:
                msg = future_import_model.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Error in importing calibrator models. Not continuing calibration. !!!!"
                )
                traceback.print_exc()
                return 1

        ###############################
        # Run basic calibration
        ###############################
        use_only_bandpass = False
        future_basical=None
        if do_basic_cal:
            if os.path.exists(calibrator_msname) == False:
                print(f"Calibrator ms: {calibrator_ms} is not present.")
                return 1
            future_basical = run_basic_cal_jobs.with_options(task_run_name=f"basic_calibration_{jobid}").submit(
                calibrator_msname,
                workdir,
                caldir,
                perform_polcal=do_polcal,
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                keep_backup=keep_backup,
                remote_log=remote_logger,
                dask_addr=dask_addr,
            )  
            try:
                msg = future_basical.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Error in basic calibration. Not continuing further. !!!!"
                )
                traceback.print_exc()
                return 1
           
        ##########################################
        # Checking presence of necessary caltables
        ##########################################
        if len(glob.glob(caldir + "/*.bcal")) == 0:
            print(f"No bandpass table is present in calibration directory : {caldir}.")
            return 1
        if len(glob.glob(caldir + "/*.gcal")) == 0:
            print(
                f"No time-dependent gaintable is present in calibration directory : {caldir}. Applying only bandpass solutions."
            )
            use_only_bandpass = True
            
        ############################################
        # Check if spliting for selfcal is fininshed
        ############################################
        if future_selfcal_split is not None:
            print("Checking spliting of target scans for selfcal status...")
            try:
                msg = future_selfcal_split.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Error in running spliting target scans for selfcal. !!!!"
                )
                do_selfcal = False
                traceback.print_exc()
                msg=1
                
        #############################################
        # Spliting target scans 
        #############################################
        # If corrected data is requested or imaging is requested
        prefix = "targets"
        future_split=None
        if do_target_split and (do_applycal or do_imaging):
            future_split = run_target_split_jobs.with_options(task_run_name=f"spliting_{prefix}_scans_{jobid}").submit(
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
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                max_cpu_frac=round(cpu_frac, 2),
                max_mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
                dask_addr=dask_addr,
            )
            
        ####################################
        # Filtering any corrupted ms
        #####################################
        selfcal_target_mslist = glob.glob(workdir + "/selfcals_scan*.ms")
        if (selfcal_target_mslist)==0:
            print(
                "!!!! WARNING: Error in running spliting target scans for selfcal. !!!!"
            )
            do_selfcal=False 
        if do_selfcal:
            print("Checking measurement sets before spawning self-calibrations....")
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
        future_apply_basical_selfcal=None
        if do_selfcal:  
            future_apply_basical_selfcal = run_apply_basiccal_sol.with_options(task_run_name=f"applying_basiccal_selfcal_{jobid}").submit(
                selfcal_mslist,
                workdir,
                caldir,
                use_only_bandpass=use_only_bandpass,
                overwrite_datacolumn=False,
                applymode="calflag",
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
                dask_addr=dask_addr,
            )
            try:
                msg = future_apply_basical_selfcal.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Error in applying basic calibration solutions on target scans. Not continuing further for selfcal.!!!!"
                )
                do_selfcal = False
                traceback.print_exc()
                msg=1
          
        ########################################
        # Performing self-calibration
        ########################################
        future_selfcal=None
        if do_selfcal:
            os.system(
                "rm -rf " + workdir + "/*selfcal " + workdir + "/caltables/*selfcal*"
            )
            future_sidereal_cor_selfcal=None
            if do_sidereal_cor:
                future_sidereal_cor_selfcal = run_solar_siderealcor_jobs.with_options(task_run_name=f"solar_sidereal_correction_{jobid}").submit(
                    selfcal_mslist,
                    workdir,
                    prefix="selfcals",
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    max_cpu_frac=round(cpu_frac, 2),
                    max_mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                    dask_addr=dask_addr,
                )
                msg=future_sidereal_cor_selfcal.result()
                if msg != 0:
                    print("Sidereal correction is not successful.")
            future_selfcal = run_selfcal_jobs.with_options(task_run_name=f"selfcal_{jobid}").submit(
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
                remote_log=remote_logger,
                dask_addr=dask_addr,
            )
            msg=future_selfcal.result()
            try:
                msg = future_selfcal.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Error in self-calibration on target scans. Not applying self-calibration. !!!!"
                )
                do_apply_selfcal = False
                traceback.print_exc()
                msg=1
            
        ########################################
        # Checking self-cal caltables
        ########################################
        if do_imaging or do_apply_selfcal:
            selfcal_tables = glob.glob(caldir + "/*selfcal*")
            if len(selfcal_tables) == 0:
                print(
                    "Self-calibration is not performed and no self-calibration caltable is available."
                )
                do_apply_selfcal = False

        #############################################
        # Checking finishing of spliting target scans
        #############################################
        if future_split is not None:
            print("Checking spliting of target scans status...")
            try:
                msg = future_split.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Error in spliting target scans. !!!!"
                )
                traceback.print_exc()
                return 1
     
        if do_imaging or do_applycal or do_apply_selfcal:
            target_mslist = glob.glob(workdir + "/targets_scan*.ms")
            if len(target_mslist)==0:
                print(
                        "!!!! WARNING: No target scans are present. !!!!"
                    )    
                return 1

            ####################################
            # Filtering any corrupted ms
            #####################################
            print(
                "Checking final valid measurement sets before applying solutions and spawning imaging...."
            )
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
                print("No filtered target scan ms are available in work directory.")
                return 1

            if do_applycal or do_imaging:
                print(
                    f"Target scan mslist : {[os.path.basename(i) for i in target_mslist]}"
                )

        #########################################################
        # Applying basic solutions on target scans
        #########################################################
        future_apply_basical=None
        if do_applycal:
            future_apply_basical = run_apply_basiccal_sol.with_options(task_run_name=f"applying_basiccal_target_{jobid}").submit(
                target_mslist,
                workdir,
                caldir,
                use_only_bandpass=use_only_bandpass,
                overwrite_datacolumn=True,
                applymode="calflag",
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
                dask_addr=dask_addr,
            )
            try:
                msg = future_apply_basical.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Error in applying basic calibration solutions on target scans. Not continuing further.!!!!"
                )
                traceback.print_exc()
                return 1
            future_sidereal_cor=None
            if do_sidereal_cor:
                future_sidereal_cor = run_solar_siderealcor_jobs.with_options(task_run_name=f"solar_sidereal_correction_{jobid}").submit(
                    target_mslist,
                    workdir,
                    prefix="targets",
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    max_cpu_frac=round(cpu_frac, 2),
                    max_mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                    dask_addr=dask_addr,
                )
                try:
                    msg = future_sidereal_cor.result()
                except Exception as e:
                    print(
                        "!!!! WARNING: Error in applying sidereal correction.!!!!"
                    )
                    traceback.print_exc()
                    msg=1

        ########################################
        # Apply self-calibration
        ########################################
        future_apply_selfcal=None
        if do_apply_selfcal:
            target_mslist = sorted(target_mslist)
            future_apply_selfcal = run_apply_selfcal_sol.with_options(task_run_name=f"applying_selfcal_{jobid}").submit(
                target_mslist,
                workdir,
                caldir,
                overwrite_datacolumn=False,
                applymode="calonly",
                jobid=jobid,
                cpu_frac=round(cpu_frac, 2),
                mem_frac=round(mem_frac, 2),
                remote_log=remote_logger,
                dask_addr=dask_addr,
            )
            try:
                msg = future_apply_selfcal.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Error in applying self-calibration solutions on target scans. !!!!"
                )
                traceback.print_exc()
                msg=1
            
        #####################################
        # Imaging
        ######################################
        future_imaging=None
        if do_imaging:
            if (
                do_polcal == False
            ):  # Only if do_polcal is False, overwrite to make only Stokes I
                pol = "I"
            band = get_band_name(target_mslist[0])
            future_imaging = run_imaging_jobs.with_options(task_run_name=f"imaging_{jobid}").submit(
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
                remote_log=remote_logger,
                dask_addr=dask_addr,
            )
            try:
                msg = future_imaging.result()
            except Exception as e:
                print(
                    "!!!! WARNING: Final imaging on all measurement sets is not successful. Check the image directory. !!!!"
                )
                traceback.print_exc()
                return 1

        ###########################
        # Primary beam correction
        ###########################
        future_pbcor=None
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
                future_pbcor = run_apply_pbcor.with_options(task_run_name=f"applying_primary_beam_{jobid}").submit(
                    imagedir,
                    workdir,
                    apply_parang=apply_parang,
                    jobid=jobid,
                    cpu_frac=round(cpu_frac, 2),
                    mem_frac=round(mem_frac, 2),
                    remote_log=remote_logger,
                    dask_addr=dask_addr,
                )
                msg=future_pbcor.result()
                try:
                    msg = future_pbcor.result()
                except Exception as e:
                    print(
                        "!!!! WARNING: Primary beam corrections of the final images are not successful. !!!!"
                    )
                    traceback.print_exc()
                    return 1
                print(f"Final image directory: {os.path.dirname(imagedir)}")
                
        ###########################################
        # Successful exit
        ###########################################
        print(
            f"Calibration and imaging pipeline is successfully run on measurement set : {msname}"
        )
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1
    finally:
        drop_cache(msname)
        drop_cache(workdir)
        drop_cache(outdir)
        stop_event.set()
        log_thread.join(timeout=5)

def cli():
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
        "--no_ds",
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
    result=prefect_server_status()
    if result is not True:
        print ("Prefect server is not running. First start it using init-meersolar-prefect start")
        sys.exit(1)
        
    try:
        print("\n########################################")
        print("Starting MeerSOLAR Pipeline....")
        print("#########################################\n")
        
        dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = get_dask_client(
            1,
            dask_dir=args.workdir,
            cpu_frac=args.cpu_frac,
            mem_frac=args.mem_frac,
        )
        npcu=int(psutil.cpu_count()*args.cpu_frac)
        dask_cluster.adapt(minimum=1, maximum=npcu) # 1 worker will be required to run the flow
        dask_addr=dask_client.scheduler.address
            
        jobid = get_jobid()    
        
        msg = master_control.with_options(flow_run_name=f"meersolar_{jobid}",task_runner=DaskTaskRunner(address=dask_addr))(
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
            dask_addr=dask_addr,
            jobid=jobid,
        )
    except Exception as e:
        traceback.print_exc()
    finally:
        drop_cache(args.msname)
        drop_cache(args.workdir)
        drop_cache(args.outdir)


if __name__ == "__main__":
    cli()
