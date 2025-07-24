import logging
import psutil
import dask
import numpy as np
import argparse
import traceback
import copy
import time
import sys
import os
from casatasks import casalog
from casatools import msmetadata, table
from dask import delayed
from dask.distributed import Client
from meersolar.utils import *
from meersolar.meerpipeline.flagging import single_ms_flag
from meersolar.meerpipeline.import_model import import_fluxcal_models

logging.getLogger("distributed").setLevel(logging.WARNING)

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    pass

datadir = get_datadir()


def run_delaycal(
    msname="",
    field="",
    scan="",
    uvrange="",
    refant="",
    refantmode="flex",
    solint="inf",
    combine="",
    gaintable=[],
    gainfield=[],
    interp=[],
    n_threads=-1,
    dry_run=False,
):
    """
    Perform delay calibration
    """
    limit_threads(n_threads=n_threads)
    from meersolar.utils.calibration import delaycal
    from casatasks import gaincal

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    print(f"Performing delay calibration on : {msname}")
    caltable_prefix = os.path.basename(msname).split(".ms")[0]
    with suppress_casa_output():
        if uvrange == "":
            gaincal(
                vis=msname,
                caltable=caltable_prefix + ".kcal",
                field=str(field),
                scan=str(scan),
                uvrange="",
                refant=refant,
                refantmode=refantmode,
                solint=solint,
                gaintype="K",
                combine=combine,
                gaintable=gaintable,
                gainfield=gainfield,
                interp=interp,
            )
        else:
            delaycal(
                vis=msname,
                caltable=caltable_prefix + ".kcal",
                field=str(field),
                scan=str(scan),
                uvrange=uvrange,
                refant=refant,
                refantmode=refantmode,
                solint=solint,
                combine=combine,
                gaintable=gaintable,
                gainfield=gainfield,
                interp=interp,
            )
    return caltable_prefix + ".kcal"


def run_bandpass(
    msname="",
    field="",
    scan="",
    uvrange="",
    refant="",
    solint="inf",
    solnorm=False,
    combine="",
    gaintable=[],
    gainfield=[],
    interp=[],
    n_threads=-1,
    dry_run=False,
):
    """
    Perform bandpass calibration
    """
    limit_threads(n_threads=n_threads)
    from casatasks import bandpass, flagdata

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    print(f"Performing bandpass calibration on : {msname}")
    caltable_prefix = os.path.basename(msname).split(".ms")[0]
    with suppress_casa_output():
        bandpass(
            vis=msname,
            caltable=caltable_prefix + ".bcal",
            field=str(field),
            scan=str(scan),
            uvrange=uvrange,
            refant=refant,
            solint=solint,
            solnorm=solnorm,
            combine=combine,
            gaintable=gaintable,
            gainfield=gainfield,
            interp=interp,
        )
        flagdata(
            vis=caltable_prefix + ".bcal",
            mode="rflag",
            datacolumn="CPARAM",
            flagbackup=False,
        )
    return caltable_prefix + ".bcal"


def run_gaincal(
    msname="",
    field="",
    scan="",
    uvrange="",
    refant="",
    gaintype="G",
    solint="inf",
    calmode="ap",
    refantmode="flex",
    solmode="",
    smodel=[],
    rmsthresh=[],
    combine="",
    append=False,
    gaintable=[],
    gainfield=[],
    interp=[],
    n_threads=-1,
    dry_run=False,
):
    """
    Perform gain calibration
    """
    limit_threads(n_threads=n_threads)
    from casatasks import gaincal, flagdata

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    print(f"Performing gain calibration on : {msname}")
    caltable_prefix = os.path.basename(msname).split(".ms")[0]
    with suppress_casa_output():
        gaincal(
            vis=msname,
            caltable=caltable_prefix + ".gcal",
            field=str(field),
            scan=str(scan),
            uvrange=uvrange,
            refant=refant,
            refantmode=refantmode,
            solint=solint,
            combine=combine,
            gaintype=gaintype,
            calmode=calmode,
            solmode=solmode,
            rmsthresh=rmsthresh,
            smodel=smodel,
            append=append,
            gaintable=gaintable,
            gainfield=gainfield,
            interp=interp,
        )
    return caltable_prefix + ".gcal"


def run_leakagecal(
    msname="",
    field="",
    scan="",
    uvrange="",
    refant="",
    combine="",
    gaintable=[],
    gainfield=[],
    interp=[],
    n_threads=-1,
    dry_run=False,
):
    """
    Perform relative leakage calibration (pol-conversion calibration)
    """
    limit_threads(n_threads=n_threads)
    from casatasks import polcal, flagdata

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    print(f"Performing relative leakage calibration on : {msname}")
    caltable_prefix = os.path.basename(msname).split(".ms")[0]
    with suppress_casa_output():
        polcal(
            vis=msname,
            caltable=caltable_prefix + ".dcal",
            field=str(field),
            scan=str(scan),
            uvrange=uvrange,
            refant=refant,
            solint="inf,10MHz",
            combine=combine,
            poltype="Df",
            gaintable=gaintable,
            gainfield=gainfield,
            interp=interp,
        )
        flagdata(
            vis=caltable_prefix + ".dcal",
            mode="rflag",
            datacolumn="CPARAM",
            flagbackup=False,
        )
    return caltable_prefix + ".dcal"


def run_polcal(
    msname="",
    field="",
    scan="",
    uvrange="",
    refant="",
    combine="",
    gaintable=[],
    gainfield=[],
    interp=[],
    n_threads=-1,
    dry_run=False,
):
    """
    Perform cross-hand phase calibration
    """
    limit_threads(n_threads=n_threads)
    from casatasks import gaincal, polcal

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    print(f"Performing relative leakage calibration on : {msname}")
    caltable_prefix = os.path.basename(msname).split(".ms")[0]
    with suppress_casa_output():
        gaincal(
            vis=msname,
            caltable=caltable_prefix + ".kcrosscal",
            field=str(field),
            scan=str(scan),
            uvrange=uvrange,
            refant=refant,
            refantmode="flex",
            solint="inf",
            combine=combine,
            gaintype="KCROSS",
            gaintable=gaintable,
            gainfield=gainfield,
            interp=interp,
        )
    if os.path.exists(caltable_prefix + ".kcrosscal"):
        gaintable.append(caltable_prefix + ".kcrosscal")
        gainfield.append(str(field))
        interp.append("")
        with suppress_casa_output():
            polcal(
                vis=msname,
                caltable=caltable_prefix + ".xfcal",
                field=str(field),
                scan=str(scan),
                uvrange=uvrange,
                refant=refant,
                solint="inf,10MHz",
                combine=combine,
                poltype="Xf",
                gaintable=gaintable,
                gainfield=gainfield,
                interp=interp,
            )
        if os.path.exists(caltable_prefix + ".xfcal"):
            gaintable.append(caltable_prefix + ".xfcal")
            gainfield.append(str(field))
            interp.append("")
            with suppress_casa_output():
                polcal(
                    vis=msname,
                    caltable=caltable_prefix + ".panglecal",
                    field=str(field),
                    scan=str(scan),
                    uvrange=uvrange,
                    refant=refant,
                    refantmode="flex",
                    solint="inf,10MHz",
                    combine="obs,scan",
                    poltype="PosAng",
                    gaintable=gaintable,
                    gainfield=gainfield,
                    interp=interp,
                )
    return (
        caltable_prefix + ".kcrosscal",
        caltable_prefix + ".xfcal",
        caltable_prefix + ".panglecal",
    )


def run_applycal(
    msname="",
    field="",
    scan="",
    applymode="",
    flagbackup=True,
    gaintable=[],
    gainfield=[],
    interp=[],
    calwt=[],
    parang=False,
    n_threads=-1,
    dry_run=False,
):
    """
    Perform apply calibration
    """
    limit_threads(n_threads=n_threads)
    from casatasks import applycal

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    print(f"Applying calibration solutions on : {msname}")
    with suppress_casa_output():
        applycal(
            vis=msname,
            field=str(field),
            scan=str(scan),
            gaintable=gaintable,
            gainfield=gainfield,
            interp=interp,
            calwt=calwt,
            applymode=applymode,
            flagbackup=flagbackup,
            parang=parang,
        )
    return


def run_postcal_flag(
    msname="",
    datacolumn="corrected",
    uvrange="",
    mode="rflag",
    n_threads=-1,
    memory_limit=-1,
    dry_run=False,
):
    """
    Perform apply calibration
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    print(f"Post-calibration flagging on ms: {msname}")
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

    #################################################
    # Whether corrected data column is present or not
    #################################################
    if datacolumn == "corrected" or datacolumn == "CORRECTED_DATA":
        corcolumn_present = check_datacolumn_valid(msname, datacolumn="CORRECTED_DATA")
        if not corcolumn_present:
            print(
                "Corrected data column is chosen for flagging, but it is not present."
            )
            return
    try:
        nchunk = get_chunk_size(msname, memory_limit=memory_limit)
        if nchunk <= 1:
            ntime = "scan"
            print("Time chunk : full scan")
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
            print(f"Time chunk : {ntime}s")
        with suppress_casa_output():
            flagdata(
                vis=msname,
                mode=mode,
                uvrange=uvrange,
                datacolumn=datacolumn,
                flagbackup=False,
                ntime=ntime,
            )
    except Exception as e:
        traceback.print_exc()
    return


def single_round_cal_and_flag(
    msname,
    workdir,
    cal_round,
    refant,
    uvrange,
    fluxcal_scans,
    fluxcal_fields,
    phasecal_scans,
    phasecal_fields,
    phasecal_fluxes,
    polcal_scans=[],
    polcal_fields=[],
    do_phasecal=False,
    do_leakagecal=False,
    do_polcal=False,
    do_postcal_flag=True,
    cpu_frac=0.8,
    mem_frac=0.8,
    dask_addr=None,
):
    """
    Single round calibration and post-calibration flagging

    Parameters
    ----------
    msname : str
        Name of the measurement set
    workdir : str
        Work directory
    cal_round : int
        Calibration round number
    refant : str
        Reference antenna
    uvrange :str
        UV-range
    fluxcal_scans : dict
        Fluxcal scans for fluxcal fields
    fluxcal_fields : list
        Fluxcal field names
    phasecal_scans : dict
        Phasecal scans for phasecal fields
    phasecal_fields : list
        Phasecal field names
    phasecal_fluxes : dict
        Phasecal fluxes
    polcal_scans : list, optional
        Polarized calibrator scans
    polcal_fields : list, optional
        Polarized calibrator fields
    do_phasecal : bool, optional
        Perform calibration on phasecal fields
    do_leakagecal : bool, optional
        Perform leakage calibration or nor
    do_phasecal : bool, optional
        Perform calibration using polcal fields
        (Note: leakage calibration is always done using unpolarized fluxcal. This is for crossphase and polarization angle calibration)
    do_postcal_flag : bool, optional
        Peform post-calibration flagging
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
    str
        Caltables
    """
    try:
        if cpu_frac > 1:
            cpu_frac = 1
        if mem_frac > 1:
            mem_frac = 1
        caltable_prefix = msname.split(".ms")[0] + "_caltable"
        msmd = msmetadata()
        msmd.open(msname)
        npol = msmd.ncorrforpol()[0]
        msmd.close()
        parang = False
        ######################################
        # Removing previous rounds caltables
        ######################################
        delay_caltable = caltable_prefix + ".kcal"
        bpass_caltable = caltable_prefix + ".bcal"
        gain_caltable = caltable_prefix + ".gcal"
        fluxscale_caltable = caltable_prefix + ".fcal"
        leakage_caltable = caltable_prefix + ".dcal"
        kcross_caltable = caltable_prefix + ".kcrosscal"
        crossphase_caltable = caltable_prefix + ".xfcal"
        pangle_caltable = caltable_prefix + ".panglecal"

        if os.path.exists(delay_caltable):
            os.system("rm -rf " + delay_caltable)
        if os.path.exists(bpass_caltable):
            os.system("rm -rf " + bpass_caltable)
        if os.path.exists(gain_caltable):
            os.system("rm -rf " + gain_caltable)
        if os.path.exists(fluxscale_caltable):
            os.system("rm -rf " + fluxscale_caltable)
        if os.path.exists(leakage_caltable):
            os.system("rm -rf " + leakage_caltable)
        if os.path.exists(kcross_caltable):
            os.system("rm -rf " + kcross_caltable)
        if os.path.exists(crossphase_caltable):
            os.system("rm -rf " + crossphase_caltable)
        if os.path.exists(pangle_caltable):
            os.system("rm -rf " + pangle_caltable)

        result = get_submsname_scans(msname)
        if result is not None:  # If multi-ms
            mslist, scans = result
        else:
            print("Please provide a multi-MS with scans partitioned.")
            return 1, []
        fluxcal_mslist = []
        phasecal_mslist = []
        polcal_mslist = []

        for fluxcal_field in fluxcal_fields:
            scan_list = fluxcal_scans[fluxcal_field]
            for scan in scan_list:
                pos = scans.index(scan)
                fluxcal_mslist.append(str(mslist[pos]))

        for phasecal_field in phasecal_fields:
            scan_list = phasecal_scans[phasecal_field]
            for scan in scan_list:
                pos = scans.index(scan)
                phasecal_mslist.append(str(mslist[pos]))

        for polcal_field in polcal_fields:
            scan_list = polcal_scans[polcal_field]
            for scan in scan_list:
                pos = scans.index(scan)
                polcal_mslist.append(str(mslist[pos]))

        #######################################
        # Calibration on fluxcal fields
        #######################################
        print("\n##############################")
        print("Calibrating fluxcal fields ....")
        print("###############################\n")
        applycal_gaintable = []
        applycal_gainfield = []
        applycal_interp = []

        if len(fluxcal_mslist) == 0 and len(phasecal_mslist) == 0:
            print("No flux calibrator or phase calibrator is present.")
            return 1, []

        ##############################
        # Delay calibration
        ##############################
        if len(fluxcal_mslist) > 0:
            delaycal_mslist = fluxcal_mslist
            #############################################
            # Memory limit
            #############################################
            ms_size_list = [get_column_size(ms) for ms in fluxcal_mslist]
            mem_limit = max(ms_size_list)
            if dask_addr is None:
                dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = (
                    get_dask_client(
                        len(delaycal_mslist),
                        dask_dir=workdir,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                        min_mem_per_job=mem_limit,
                    )
                )
            else:
                _, _, n_jobs, n_threads, mem_limit, dask_dir = get_dask_client(
                    len(delaycal_mslist),
                    dask_dir=workdir,
                    cpu_frac=cpu_frac,
                    mem_frac=mem_frac,
                    min_mem_per_job=mem_limit,
                    only_cal=True,
                )
                os.system(f"rm -rf {dask_dir}")
                dask_client = Client(address=dask_addr)
            wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
            tasks = [
                delayed(run_delaycal)(
                    sub_msname,
                    uvrange=uvrange,
                    refant=refant,
                    solint="inf",
                    n_threads=n_threads,
                )
                for sub_msname in delaycal_mslist
            ]
            futures = dask_client.compute(tasks)
            delaycal_tables = list(dask_client.gather(futures))
            dask_client.close()
            if dask_addr is None:
                dask_cluster.close()
                os.system(f"rm -rf {dask_dir}")
            delay_caltable = merge_caltables(
                delaycal_tables, delay_caltable, keepcopy=False
            )
            if delay_caltable is not None and os.path.exists(delay_caltable):
                tb = table()
                tb.open(delay_caltable, nomodify=False)
                flag = tb.getcol("FLAG")
                flag *= False
                tb.putcol("FLAG", flag)
                tb.flush()
                tb.close()
                applycal_gaintable.append(delay_caltable)
                applycal_gainfield.append("")
                applycal_interp.append("nearest")

        ##############################
        # Bandpass calibration
        ##############################
        if len(fluxcal_mslist) > 0:
            ms_size_list = [get_column_size(ms) for ms in fluxcal_mslist]
            mem_limit = max(ms_size_list)
            if dask_addr is None:
                dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = (
                    get_dask_client(
                        len(fluxcal_mslist),
                        dask_dir=workdir,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                        min_mem_per_job=mem_limit,
                    )
                )
            else:
                _, _, n_jobs, n_threads, mem_limit, dask_dir = get_dask_client(
                    len(fluxcal_mslist),
                    dask_dir=workdir,
                    cpu_frac=cpu_frac,
                    mem_frac=mem_frac,
                    min_mem_per_job=mem_limit,
                    only_cal=True,
                )
                os.system(f"rm -rf {dask_dir}")
                dask_client = Client(address=dask_addr)
            wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
            tasks = [
                delayed(run_bandpass)(
                    sub_msname,
                    uvrange=uvrange,
                    refant=refant,
                    solint="inf",
                    gaintable=applycal_gaintable,
                    interp=applycal_interp,
                    n_threads=n_threads,
                )
                for sub_msname in fluxcal_mslist
            ]
            futures = dask_client.compute(tasks)
            bandpass_tables = list(dask_client.gather(futures))
            dask_client.close()
            if dask_addr is None:
                dask_cluster.close()
                os.system(f"rm -rf {dask_dir}")
            bpass_caltable = merge_caltables(
                bandpass_tables, bpass_caltable, keepcopy=False
            )
            if bpass_caltable is not None and os.path.exists(bpass_caltable):
                applycal_gaintable.append(bpass_caltable)
                applycal_gainfield.append("")
                applycal_interp.append("nearestflag")
            else:
                print("Bandpass calibration is not successful.")
                return 1, []
        else:
            print("No flux calibrator is present.")
            return 1, []

        ##########################################
        # Gain calibration on fluxcal (and polcal)
        ##########################################
        gaincal_mslist = fluxcal_mslist
        if do_polcal and len(polcal_mslist) > 0 and npol == 4:
            gaincal_mslist = fluxcal_mslist + polcal_mslist
        if len(gaincal_mslist) > 0:
            ms_size_list = [get_column_size(ms) for ms in gaincal_mslist]
            mem_limit = max(ms_size_list)
            if dask_addr is None:
                dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = (
                    get_dask_client(
                        len(gaincal_mslist),
                        dask_dir=workdir,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                        min_mem_per_job=mem_limit,
                    )
                )
            else:
                _, _, n_jobs, n_threads, mem_limit, dask_dir = get_dask_client(
                    len(gaincal_mslist),
                    dask_dir=workdir,
                    cpu_frac=cpu_frac,
                    mem_frac=mem_frac,
                    min_mem_per_job=mem_limit,
                    only_cal=True,
                )
                os.system(f"rm -rf {dask_dir}")
                dask_client = Client(address=dask_addr)
            wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
            tasks = [
                delayed(run_gaincal)(
                    sub_msname,
                    uvrange=uvrange,
                    refant=refant,
                    gaintype="T",
                    solint="1min",
                    calmode="ap",
                    gaintable=applycal_gaintable,
                    interp=applycal_interp,
                    n_threads=n_threads,
                )
                for sub_msname in gaincal_mslist
            ]
            futures = dask_client.compute(tasks)
            gain_tables = list(dask_client.gather(futures))
            dask_client.close()
            if dask_addr is None:
                dask_cluster.close()
                os.system(f"rm -rf {dask_dir}")
            gain_caltable = merge_caltables(gain_tables, gain_caltable, keepcopy=False)

        ######################################
        # Gain calibrations on phasecals
        ######################################
        if do_phasecal == False and (
            gain_caltable is not None and os.path.exists(gain_caltable)
        ):
            applycal_gaintable.append(gain_caltable)
            applycal_gainfield.append("")
            applycal_interp.append("nearest")
        else:
            print("\n##############################")
            print("Calibrating phasecal fields ....")
            print("###############################\n")
            ##############################
            # Gain calibration
            ##############################
            if len(phasecal_mslist) == 0 and (
                gain_caltable is not None and os.path.exists(gain_caltable)
            ):
                applycal_gaintable.append(gain_caltable)
                applycal_gainfield.append("")
                applycal_interp.append("nearest")
            else:
                ms_size_list = [
                    get_column_size(ms) for ms in phasecal_mslist
                ]
                mem_limit = max(ms_size_list)
                if dask_addr is None:
                    (
                        dask_client,
                        dask_cluster,
                        n_jobs,
                        n_threads,
                        mem_limit,
                        dask_dir,
                    ) = get_dask_client(
                        len(phasecal_mslist),
                        dask_dir=workdir,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                        min_mem_per_job=mem_limit,
                    )
                else:
                    _, _, n_jobs, n_threads, mem_limit, dask_dir = get_dask_client(
                        len(phasecal_mslist),
                        dask_dir=workdir,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                        min_mem_per_job=mem_limit,
                        only_cal=True,
                    )
                    os.system(f"rm -rf {dask_dir}")
                    dask_client = Client(address=dask_addr)
                wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
                tasks = [
                    delayed(run_gaincal)(
                        sub_msname,
                        uvrange=uvrange,
                        refant=refant,
                        gaintype="T",
                        solint="1min",
                        calmode="ap",
                        smodel=[1, 0, 0, 0],
                        gaintable=applycal_gaintable,
                        interp=applycal_interp,
                        n_threads=n_threads,
                    )
                    for sub_msname in phasecal_mslist
                ]
                futures = dask_client.compute(tasks)
                gain_tables = list(dask_client.gather(futures))
                dask_client.close()
                if dask_addr is None:
                    dask_cluster.close()
                    os.system(f"rm -rf {dask_dir}")
                gain_caltable = merge_caltables(
                    gain_tables, gain_caltable, append=True, keepcopy=False
                )
                #################################
                # Flux scaling
                #################################
                if os.path.exists(caltable_prefix + ".listfcal"):
                    os.system("rm -rf " + caltable_prefix + ".listfcal")
                from casatasks import fluxscale

                if gain_caltable is not None and os.path.exists(gain_caltable) == False:
                    print(
                        "Gain calibration was not successful and did not produce gain caltable."
                    )
                else:
                    with suppress_casa_output():
                        fluxscale_result = fluxscale(
                            vis=msname,
                            caltable=gain_caltable,
                            fluxtable=fluxscale_caltable,
                            reference=fluxcal_fields,
                            transfer=phasecal_fields,
                        )
                    if fluxscale_caltable is not None and os.path.exists(
                        fluxscale_caltable
                    ):
                        if os.path.exists(gain_caltable):
                            os.system("rm -rf " + gain_caltable)
                        os.system(f"mv {fluxscale_caltable} {gain_caltable}")
                        applycal_gaintable.append(gain_caltable)
                        applycal_gainfield.append("")
                        applycal_interp.append("nearest")
                    msmd.open(msname)
                    phasecal_field_ids = [
                        str(msmd.fieldsforname(f)[0]) for f in phasecal_fields
                    ]
                    msmd.close()
                    for pcal in phasecal_field_ids:
                        field_name = fluxscale_result[pcal]["fieldName"]
                        catalog_flux = phasecal_fluxes[field_name]
                        result = fluxscale_result[pcal]["0"]
                        flux = result["fluxd"][0]
                        fluxerr = result["fluxdErr"][0]
                        print(f"Catalog flux: {catalog_flux} Jy")
                        print(f"Estimated flux: {flux} +/- {fluxerr} Jy")
                        print(
                            f"Difference from catalog flux: {round(abs(flux-catalog_flux),2)} Jy"
                        )
                        print("###################################\n")

        ##############################
        # Leakage calibration
        ##############################
        if do_leakagecal:
            if npol != 4:
                print(
                    "Measurement set is not full-polar. Not performing leakage calibration."
                )
            elif len(fluxcal_mslist) > 0:
                ms_size_list = [
                    get_column_size(ms) for ms in fluxcal_mslist
                ]
                mem_limit = max(ms_size_list)
                if dask_addr is None:
                    (
                        dask_client,
                        dask_cluster,
                        n_jobs,
                        n_threads,
                        mem_limit,
                        dask_dir,
                    ) = get_dask_client(
                        len(fluxcal_mslist),
                        dask_dir=workdir,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                        min_mem_per_job=mem_limit,
                    )
                else:
                    _, _, n_jobs, n_threads, mem_limit, dask_dir = get_dask_client(
                        len(fluxcal_mslist),
                        dask_dir=workdir,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                        min_mem_per_job=mem_limit,
                        only_cal=True,
                    )
                    os.system(f"rm -rf {dask_dir}")
                    dask_client = Client(address=dask_addr)
                wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
                tasks = [
                    delayed(run_leakagecal)(
                        sub_msname,
                        uvrange=uvrange,
                        refant=refant,
                        gaintable=applycal_gaintable,
                        gainfield=["", "", ",".join(fluxcal_fields)],
                        interp=applycal_interp,
                        n_threads=n_threads,
                    )
                    for sub_msname in fluxcal_mslist
                ]
                futures = dask_client.compute(tasks)
                leakage_tables = list(dask_client.gather(futures))
                dask_client.close()
                if dask_addr is None:
                    dask_cluster.close()
                    os.system(f"rm -rf {dask_dir}")
                leakage_caltable = merge_caltables(
                    leakage_tables, leakage_caltable, keepcopy=False
                )
                if leakage_caltable is not None and os.path.exists(leakage_caltable):
                    applycal_gaintable.append(leakage_caltable)
                    applycal_gainfield.append("")
                    applycal_interp.append("nearest,nearestflag")
                    if not parang:
                        parang = True

        ########################################
        # Calibration using polarized calibrator
        ########################################
        if do_polcal and do_leakagecal:
            if len(polcal_mslist) == 0:
                print("No polarized calibrator fields are present.")
            elif npol != 4:
                print("Measurement set is not full-polar.")
            elif os.path.exists(leakage_caltable) == False:
                print("Leakage solutions are not present.")
            else:
                ms_size_list = [get_column_size(ms) for ms in polcal_mslist]
                mem_limit = max(ms_size_list)
                if dask_addr is None:
                    (
                        dask_client,
                        dask_cluster,
                        n_jobs,
                        n_threads,
                        mem_limit,
                        dask_dir,
                    ) = get_dask_client(
                        len(polcal_mslist),
                        dask_dir=workdir,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                        min_mem_per_job=mem_limit,
                    )
                else:
                    _, _, n_jobs, n_threads, mem_limit, dask_dir = get_dask_client(
                        len(polcal_mslist),
                        dask_dir=workdir,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                        min_mem_per_job=mem_limit,
                        only_cal=True,
                    )
                    os.system(f"rm -rf {dask_dir}")
                    dask_client = Client(address=dask_addr)
                wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
                tasks = [
                    delayed(run_polcal)(
                        sub_msname,
                        uvrange=uvrange,
                        refant=refant,
                        solint="inf",
                        gaintable=applycal_gaintable,
                        gainfield=[
                            "",
                            "",
                            ",".join(polcal_fields),
                            ",".join(fluxcal_fields),
                        ],
                        interp=applycal_interp,
                        n_threads=n_threads,
                    )
                    for sub_msname in polcal_mslist
                ]
                futures = dask_client.compute(tasks)
                results = list(dask_client.gather(futures))
                dask_client.close()
                if dask_addr is None:
                    dask_cluster.close()
                    os.system(f"rm -rf {dask_dir}")
                kcross_tables = []
                crossphase_tables = []
                pangle_tables = []
                for r in results:
                    kcross_tables.append(r[0])
                    if len(r) > 1:
                        crossphase_tables.append(r[1])
                    if len(r) > 2:
                        pangle_tables.append(r[2])
                kcross_caltable = merge_caltables(
                    kcross_tables, kcross_caltable, keepcopy=False
                )
                crossphase_caltable = merge_caltables(
                    crossphase_tables, crossphase_caltable, keepcopy=False
                )
                pangle_caltable = merge_caltables(
                    pangle_tables, pangle_caltable, keepcopy=False
                )

                if kcross_caltable is not None and os.path.exists(kcross_caltable):
                    applycal_gaintable.append(kcross_caltable)
                    applycal_gainfield.append("")
                    applycal_interp.append("nearest")
                    if not parang:
                        parang = True
                    if crossphase_caltable is not None and os.path.exists(
                        crossphase_caltable
                    ):
                        applycal_gaintable.append(crossphase_caltable)
                        applycal_gainfield.append("")
                        applycal_interp.append("nearest,nearestflag")
                        if pangle_caltable is not None and os.path.exists(
                            pangle_caltable
                        ):
                            applycal_gaintable.append(pangle_caltable)
                            applycal_gainfield.append("")
                            applycal_interp.append("nearest,nearestflag")
                        else:
                            print(
                                "Absolute polarization angle calibration could not be done."
                            )
                    else:
                        print(
                            "Crosshand phase and absolute polarization angle calibration could not be done."
                        )
                else:
                    print(
                        "Crosshand delay, crosshand phase and absolute polarization angle calibration could not be done."
                    )

        ##############################
        # Apply calibration
        ##############################
        all_mslist = copy.deepcopy(fluxcal_mslist)
        if len(phasecal_mslist) > 0:
            all_mslist += phasecal_mslist
        if len(polcal_mslist) > 0:
            all_mslist += polcal_mslist
        if len(all_mslist) > 0:
            do_flag_backup(msname, flagtype="applycal")
            ms_size_list = [get_column_size(ms) for ms in all_mslist]
            mem_limit = max(ms_size_list)
            if dask_addr is None:
                dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = (
                    get_dask_client(
                        len(all_mslist),
                        dask_dir=workdir,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                        min_mem_per_job=mem_limit,
                    )
                )
            else:
                _, _, n_jobs, n_threads, mem_limit, dask_dir = get_dask_client(
                    len(all_mslist),
                    dask_dir=workdir,
                    cpu_frac=cpu_frac,
                    mem_frac=mem_frac,
                    min_mem_per_job=mem_limit,
                    only_cal=True,
                )
                os.system(f"rm -rf {dask_dir}")
                dask_client = Client(address=dask_addr)
            wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
            tasks = [
                delayed(run_applycal)(
                    sub_msname,
                    flagbackup=False,
                    gaintable=applycal_gaintable,
                    gainfield=applycal_gainfield,
                    interp=applycal_interp,
                    calwt=[False] * len(applycal_gainfield),
                    parang=parang,
                    n_threads=n_threads,
                )
                for sub_msname in all_mslist
            ]
            futures = dask_client.compute(tasks)
            results = list(dask_client.gather(futures))
            dask_client.close()
            if dask_addr is None:
                dask_cluster.close()
                os.system(f"rm -rf {dask_dir}")

        ##############################
        # Post calibration flagging
        ##############################
        if do_postcal_flag and len(all_mslist) > 0:
            do_flag_backup(msname, flagtype="flagdata")
            ms_size_list = [get_column_size(ms) for ms in all_mslist]
            mem_limit = max(ms_size_list)
            if dask_addr is None:
                dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = (
                    get_dask_client(
                        len(all_mslist),
                        dask_dir=workdir,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                        min_mem_per_job=mem_limit,
                    )
                )
            else:
                _, _, n_jobs, n_threads, mem_limit, dask_dir = get_dask_client(
                    len(all_mslist),
                    dask_dir=workdir,
                    cpu_frac=cpu_frac,
                    mem_frac=mem_frac,
                    min_mem_per_job=mem_limit,
                    only_cal=True,
                )
                os.system(f"rm -rf {dask_dir}")
                dask_client = Client(address=dask_addr)
            wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
            tasks = []
            if len(all_mslist) > 0:
                tasks = []
                for sub_msname in all_mslist:
                    if sub_msname in fluxcal_mslist:
                        datacolumn = "residual"
                    else:
                        datacolumn = "corrected"
                    tasks.append(
                        delayed(run_postcal_flag)(
                            sub_msname,
                            datacolumn=datacolumn,
                            uvrange=uvrange,
                            mode="rflag",
                            n_threads=n_threads,
                            memory_limit=mem_limit,
                        )
                    )
            futures = dask_client.compute(tasks)
            results = list(dask_client.gather(futures))
            dask_client.close()
            if dask_addr is None:
                dask_cluster.close()
                os.system(f"rm -rf {dask_dir}")

        ###############################
        # Finished calibration round
        ###############################
        delay_caltable = (
            delay_caltable
            if (delay_caltable is not None and os.path.exists(delay_caltable))
            else None
        )
        bpass_caltable = (
            bpass_caltable
            if (bpass_caltable is not None and os.path.exists(bpass_caltable))
            else None
        )
        gain_caltable = (
            gain_caltable
            if (gain_caltable is not None and os.path.exists(gain_caltable))
            else None
        )
        leakage_caltable = (
            leakage_caltable
            if (leakage_caltable is not None and os.path.exists(leakage_caltable))
            else None
        )
        kcross_caltable = (
            kcross_caltable
            if (kcross_caltable is not None and os.path.exists(kcross_caltable))
            else None
        )
        crossphase_caltable = (
            crossphase_caltable
            if (crossphase_caltable is not None and os.path.exists(crossphase_caltable))
            else None
        )
        pangle_caltable = (
            pangle_caltable
            if (pangle_caltable is not None and os.path.exists(pangle_caltable))
            else None
        )
        return 0, [
            delay_caltable,
            bpass_caltable,
            gain_caltable,
            leakage_caltable,
            kcross_caltable,
            crossphase_caltable,
            pangle_caltable,
        ]
    except Exception as e:
        traceback.print_exc()
        return 1, []


def run_basic_cal_rounds(
    msname,
    workdir,
    refant="",
    uvrange="",
    keep_backup=False,
    perform_polcal=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    dask_addr=None,
):
    """
    Perform basic calibration rounds

    Parameters
    ----------
    msname : str
        Name of the measurement set
    workdir : str
        Warking directory
    refant : str, optional
        Reference antenna
    uvrange : str, optional
        UV-range
    perform_polcal : bool, optional
        Perform polarization calibration for fullpolar data
    keep_backup : bool, optional
        Keep backup of ms after each calibration rounds
    cpu_frac : float, options
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message
    list
        Caltables
    """
    start_time = time.time()
    try:
        os.chdir(workdir)
        print(f"Measurement set : {msname}")
        print("Extracting metadata from measurement set ....")
        correct_missing_col_subms(msname)
        fluxcal_fields, fluxcal_scans = get_fluxcals(msname)
        polcal_fields, polcal_scans = get_polcals(msname)
        phasecal_fields, phasecal_scans, phasecal_fluxes = get_phasecals(msname)
        msmd = msmetadata()
        msmd.open(msname)
        npol = msmd.ncorrforpol()[0]
        msmd.close()
        if npol == 4 or len(polcal_fields) > 0 or len(phasecal_fields) > 0:
            n_rounds = 3
        else:
            n_rounds = 2
        print(f"Total calibration rounds: {n_rounds}")
        do_phasecal = False
        do_polcal = False
        do_leakagecal = False
        do_postcal_flag = True
        ###################################################
        # Determining what calibrations will be done or not
        ###################################################
        if len(phasecal_fields) > 0:
            perform_phasecal = True
        else:
            perform_phasecal = False
        if perform_polcal and npol == 4:
            perform_leakagecal = True
            if len(polcal_fields) > 0:
                perform_polcal = (
                    True  # Leakage calibration is done using unpolarized fluxcal
                )
            else:
                perform_polcal = False
        else:  # If not a full polar ms
            print(
                "Measurement set is not full-polar. Do not performing any polarization calibration."
            )
            perform_leakagecal = False
            perform_polcal = False
        #####################################################
        if refant == "":
            refant = get_refant(msname)
        if uvrange == "":
            uvrange = ">200lambda"
        print("#########################################")
        print(f"Using UV-range for calibration: {uvrange}")
        print("#########################################")
        for cal_round in range(1, n_rounds + 1):
            print("\n#################################")
            print(f"Calibration round: {cal_round}")
            print("#################################\n")
            if cal_round == n_rounds:
                do_postcal_flag = False
            if cal_round > 1:
                if perform_phasecal:
                    do_phasecal = True
                if perform_polcal:
                    do_polcal = True
                if perform_leakagecal:
                    do_leakagecal = True
            msg, caltables = single_round_cal_and_flag(
                msname,
                workdir,
                cal_round,
                refant,
                uvrange,
                fluxcal_scans,
                fluxcal_fields,
                phasecal_scans,
                phasecal_fields,
                phasecal_fluxes,
                do_phasecal=do_phasecal,
                do_leakagecal=do_leakagecal,
                do_polcal=do_polcal,
                do_postcal_flag=do_postcal_flag,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                dask_addr=dask_addr,
            )
            if keep_backup:
                print(f"Backup directory: {workdir}/backup")
                os.makedirs(workdir + "/backup", exist_ok=True)
                for caltable in caltables:
                    if caltable is not None and os.path.exists(caltable):
                        cal_ext = os.path.basename(caltable).split(".")[-1]
                        outputname = (
                            workdir
                            + "/backup/"
                            + os.path.basename(caltable).split(f".{cal_ext}")[0]
                            + "_round_"
                            + str(cal_round)
                            + f".{cal_ext}"
                        )
                        os.system("cp -r " + caltable + " " + outputname)
            if msg == 1:
                print("##################")
                print("Basic calibration is not successful.")
                print(f"Total time taken : {time.time() - start_time}")
                print("##################\n")
                return 1, []
        print("##################")
        print("Basic calibration is done successfully.")
        print(f"Total time taken : {time.time() - start_time}")
        print("##################\n")
        return 0, caltables
    except Exception as e:
        traceback.print_exc()
        print("##################")
        print(f"Total time taken : {time.time() - start_time}")
        print("##################\n")
        return 1, []


def main(
    msname,
    workdir,
    caldir,
    refant="1",
    uvrange="",
    perform_polcal=False,
    keep_backup=False,
    start_remote_log=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    dask_addr=None,
):
    """
    Main function to perform basic calibration

    Parameters
    ----------
    msname : str
        Measurement set
    workdir : str
        Work directory
    caldir : str
        Caltables directory
    refant : str, optional
        Reference antenna
    uvrange : str, optional
        UV-range
    perform_polcal : bool, optional
        Perform polarization calibration
    start_remote_log : bool, optional
        Start logging to remote logger or not
    keep_backup : bool, optional
        Keep backup
    cpu_frac : float, optional
        CPU fraction
    mem_frac : float, optional
        Memory fraction
    logfile : str, optional
        Log file name
    jobid : str, optional
        Pipeline Job ID
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

    if caldir == "" or not os.path.exists(caldir):
        caldir = f"{workdir}/caltables"
    os.makedirs(caldir, exist_ok=True)

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
                "basic_cal", logfile, jobname=jobname, password=password
            )
    if observer == None:
        print("Remote link or jobname is blank. Not transmiting to remote logger.")

    try:
        if msname != "" and os.path.exists(msname):
            print("\n###################################")
            print("Starting initial calibration.")
            print("###################################\n")
            msg, caltables = run_basic_cal_rounds(
                msname,
                workdir,
                refant=refant,
                uvrange=uvrange,
                perform_polcal=perform_polcal,
                keep_backup=keep_backup,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                dask_addr=dask_addr,
            )

            for caltable in caltables:
                if caltable is not None and os.path.exists(caltable):
                    dest = caldir + "/" + os.path.basename(caltable)
                    if os.path.exists(dest):
                        os.system("rm -rf " + dest)
                    os.system("mv " + caltable + " " + caldir)
        else:
            print("Please provide a valid measurement set.\n")
            msg = 1
    except Exception as e:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        drop_cache(msname)
        drop_cache(workdir)
        drop_cache(caldir)
        clean_shutdown(observer)
    return msg


def cli():
    parser = argparse.ArgumentParser(
        description="Basic calibration using calibrators",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "msname",
        type=str,
        help="Name of measurement set (required positional argument)",
    )
    basic_args.add_argument(
        "--workdir",
        type=str,
        default="",
        required=True,
        help="Working directory for calibration outputs (default: auto-created next to MS)",
    )
    basic_args.add_argument(
        "--caldir",
        type=str,
        default="",
        required=True,
        help="Caltables directory (default: auto-created in the workdir MS)",
    )

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced calibration parameters\n###################"
    )
    adv_args.add_argument("--refant", type=str, default="1", help="Reference antenna")
    adv_args.add_argument(
        "--uvrange",
        type=str,
        default="",
        help="UV range for calibration (e.g. '>100lambda')",
    )
    adv_args.add_argument(
        "--perform_polcal", action="store_true", help="Enable polarization calibration"
    )
    adv_args.add_argument(
        "--keep_backup",
        action="store_true",
        help="Keep backup of measurement set after each calibration round",
    )
    adv_args.add_argument(
        "--start_remote_log", action="store_true", help="Start remote logging"
    )

    # Resource management parameters
    hard_args = parser.add_argument_group(
        "###################\nHardware resource management parameters\n###################"
    )
    hard_args.add_argument(
        "--cpu_frac", type=float, default=0.8, help="CPU fraction to use"
    )
    hard_args.add_argument(
        "--mem_frac", type=float, default=0.8, help="Memory fraction to use"
    )
    hard_args.add_argument(
        "--logfile", type=str, default=None, help="Optional path to log file"
    )
    hard_args.add_argument(
        "--jobid", type=str, default="0", help="Job ID for logging and PID tracking"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 0

    args = parser.parse_args()

    msg = main(
        args.msname,
        args.workdir,
        args.caldir,
        refant=args.refant,
        uvrange=args.uvrange,
        perform_polcal=args.perform_polcal,
        keep_backup=args.keep_backup,
        start_remote_log=args.start_remote_log,
        cpu_frac=float(args.cpu_frac),
        mem_frac=float(args.mem_frac),
        logfile=args.logfile,
        jobid=args.jobid,
    )
    return msg


if __name__ == "__main__":
    result = cli()
    print(
        "\n###################\nBasic calibration is finished.\n###################\n"
    )
    os._exit(result)
