import logging
import psutil
import dask
import numpy as np
import argparse
import traceback
import warnings
import copy
import time
import glob
import sys
import os
from casatasks import casalog
from casatools import table
from dask import delayed
from dask.distributed import Client
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from meersolar.utils import *
from meersolar.meerpipeline.flagging import single_ms_flag

logging.getLogger("distributed").setLevel(logging.WARNING)


try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    pass

datadir = get_datadir()


def interpolate_nans(data):
    """Linearly interpolate NaNs in 1D array."""
    nans = np.isnan(data)
    if np.all(nans):
        raise ValueError("All values are NaN.")
    x = np.arange(len(data))
    interp_func = interp1d(
        x[~nans],
        data[~nans],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return interp_func(x)


def filter_outliers(data, threshold=5, max_iter=3):
    """
    Filter outliers and perform cubic spline fitting

    Parameters
    ----------
    y : numpy.array
        Y values
    threshold : float
        Threshold of filtering
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    numpy.array
        Clean Y-values
    """
    for c in range(max_iter):
        data = np.asarray(data, dtype=np.float64)
        original_nan_mask = np.isnan(data)
        # Interpolate NaNs for smoothing
        interpolated_data = interpolate_nans(data)
        # Apply Gaussian smoothing
        smoothed = gaussian_filter1d(
            interpolated_data, sigma=threshold, truncate=3 * threshold
        )
        # Compute residuals and std only on valid original data
        residuals = data - smoothed
        valid_mask = ~original_nan_mask
        std_dev = np.std(residuals[valid_mask])
        # Detect outliers
        outlier_mask = np.abs(residuals) > threshold * std_dev
        combined_mask = valid_mask & ~outlier_mask
        # Replace outliers with NaN
        filtered_data = np.where(combined_mask, data, np.nan)
        data = copy.deepcopy(filtered_data)
    return filtered_data


def scale_bandpass(bandpass_table, att_tables):
    """
    Scale a bandpass calibration table using attenuation data.

    Parameters
    ----------
    bandpass_table : str
        Input bandpass calibration table.
    att_tables : list
        Attenuation NumPy .npy files containing attenuation frequency and values for different scans.

    Returns
    -------
    str
        Name of the output table.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if isinstance(att_tables, str):
        att_tables = [att_tables]
    if len(att_tables) == 0:
        print("No attenuation table is provided.")
        return
    scan_att_array = []
    for att_table in att_tables:
        results = np.load(att_table, allow_pickle=True)
        scan, freqs, att_values, flag_ants, att_array = results
        scan_att_array.append(att_values)
    scan_att_array = np.array(scan_att_array)
    att_values = np.nanmedian(scan_att_array, axis=0)  # Median across scans
    output_table = bandpass_table.split(".bcal")[0] + "_att.bcal"
    tb = table()
    tb.open(f"{bandpass_table}/SPECTRAL_WINDOW")
    caltable_freqs = tb.getcol("CHAN_FREQ").flatten()
    tb.close()
    # Prepare output table
    if os.path.exists(output_table):
        os.system(f"rm -rf {output_table}")
    os.system(f"cp -r {bandpass_table} {output_table}")
    tb.open(output_table, nomodify=False)
    gain = tb.getcol("CPARAM")
    flag = tb.getcol("FLAG")
    for i in range(att_values.shape[0]):
        att = filter_outliers(att_values[i])
        valid = ~np.isnan(att)
        best_fit, best_std = None, np.inf
        for deg in range(3, 9):
            coeffs = np.polyfit(freqs[valid], att[valid], deg)
            interp_func = np.poly1d(coeffs)
            interp_att = interp_func(caltable_freqs)
            residuals = att_values[i] - interp_att
            new_std = np.nanstd(residuals[~np.isnan(att_values[i])])
            if new_std >= best_std:
                break
            best_std = new_std
            best_fit = interp_att
        # Limit interpolation to valid frequency range
        nanpos = np.where(~np.isnan(att_values[i]))[0]
        minpos, maxpos = np.nanmin(nanpos), np.nanmax(nanpos)
        best_fit[:minpos] = best_fit[maxpos:] = np.nan
        # Broadcast to CPARAM shape
        interp_scaled = np.sqrt(best_fit)
        # Apply scaling
        gain[i, ...] *= interp_scaled[..., None]
    gain[flag] = 1.0
    tb.putcol("CPARAM", gain)
    tb.flush()
    tb.close()
    return output_table


def applysol(
    msname="",
    gaintable=[],
    gainfield=[],
    interp=[],
    parang=False,
    applymode="calflag",
    overwrite_datacolumn=False,
    n_threads=-1,
    memory_limit=-1,
    force_apply=False,
    soltype="basic",
    do_post_flag=False,
    dry_run=False,
):
    """
    Apply flux calibrated and attenuation calibrated solutions

    Parameters
    ----------
    msname : str
        Measurement set
    gaintable : list, optional
        Caltable list
    gainfield : list, optional
        Gain field list
    interp : list, optional
        Gain interpolation
    parang : bool, optional
        Parallactic angle apply or not
    applymode : str, optional
        Apply mode
    overwrite_datacolumn : bool, optional
        Overwrite data column with corrected solutions
    n_threads : int, optional
        Number of OpenMP threads
    memory_limit : float, optional
        Memory limit in GB
    force_apply : bool, optional
        Force to apply solutions if it is already applied
    soltype : str, optional
        Solution type
    do_post_flag : bool, optional
        Do post calibration flagging

    Returns
    -------
    int
        Success message
    """
    limit_threads(n_threads=n_threads)
    from casatasks import applycal, flagdata, split, clearcal

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    print(
        f"Applying solutions on ms: {os.path.basename(msname)} from caltables: {','.join([os.path.basename(i) for i in gaintable])}\n"
    )
    if soltype == "basic":
        check_file = "/.applied_sol"
    else:
        check_file = "/.applied_selfcalsol"
    try:
        if os.path.exists(msname + check_file) and force_apply == False:
            print("Solutions are already applied.")
            return 0
        else:
            if os.path.exists(msname + check_file) and force_apply:
                with suppress_casa_output():
                    clearcal(vis=msname)
                    flagdata(vis=msname, mode="unflag", spw="0", flagbackup=False)
                if os.path.exists(msname + ".flagversions"):
                    os.system("rm -rf " + msname + ".flagversions")
            with suppress_casa_output():
                applycal(
                    vis=msname,
                    gaintable=gaintable,
                    gainfield=gainfield,
                    applymode=applymode,
                    interp=interp,
                    calwt=[False] * len(gaintable),
                    parang=parang,
                    flagbackup=False,
                )
        if overwrite_datacolumn:
            print(f"Spliting corrected data for ms: {msname}.")
            outputvis = msname.split(".ms")[0] + "_cor.ms"
            if os.path.exists(outputvis):
                os.system(f"rm -rf {outputvis}")
            touch_file_names = glob.glob(f"{msname}/.*")
            if len(touch_file_names) > 0:
                touch_file_names = [os.path.basename(f) for f in touch_file_names]
            with suppress_casa_output():
                split(vis=msname, outputvis=outputvis, datacolumn="corrected")
            if os.path.exists(outputvis):
                os.system(f"rm -rf {msname} {msname}.flagversions")
                os.system(f"mv {outputvis} {msname}")
            for t in touch_file_names:
                os.system(f"touch {msname}/{t}")
        if do_post_flag:
            print(f"Post calibration flagging on: {msname}")
            if overwrite_datacolumn:
                datacolumn = "data"
            else:
                datacolumn = "corrected"
            single_ms_flag(
                msname=msname,
                datacolumn=datacolumn,
                use_tfcrop=True,
                use_rflag=False,
                flagdimension="freq",
                flag_autocorr=False,
                n_threads=n_threads,
                memory_limit=memory_limit,
            )
        os.system("touch " + msname + check_file)
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1


def run_all_applysol(
    mslist,
    workdir,
    caldir,
    use_only_bandpass=False,
    overwrite_datacolumn=False,
    applymode="calflag",
    force_apply=False,
    do_post_flag=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    dask_addr=None,
):
    """
    Apply basic-calibration solutions on all target scans

    Parameters
    ----------
    mslist : str
        Measurement set list
    workdir : str
        Working directory
    caldir : str
        Calibration directory
    use_only_bandpass : bool, optional
        Use only bandpass solutions
    overwrite_datacolumn : bool, optional
        Overwrite data column or not
    applymode : str, optional
        Apply mode
    force_apply : bool, optional
        Force to apply solutions even already applied
    do_post_flag : bool, optional
        Do post calibration flagging
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    dask_addr : str, optional
        Dask scheduler address

    Returns
    --------
    list
        Calibrated target scans
    """
    start_time = time.time()
    try:
        os.chdir(workdir)
        mslist = np.unique(mslist).tolist()
        parang = False
        os.system("rm -rf " + caldir + "/*scan*.bcal")
        att_caltables = glob.glob(caldir + "/*_attval_scan_*.npy")
        bandpass_table = glob.glob(caldir + "/calibrator_caltable.bcal")
        delay_table = glob.glob(caldir + "/calibrator_caltable.kcal")
        gain_table = glob.glob(caldir + "/calibrator_caltable.gcal")
        leakage_table = glob.glob(caldir + "/calibrator_caltable.dcal")
        if len(leakage_table) > 0:
            parang = True
            kcross_table = glob.glob(caldir + "/calibrator_caltable.kcrosscal")
            crossphase_table = glob.glob(caldir + "/calibrator_caltable.xfcal")
            pangle_table = glob.glob(caldir + "/calibrator_caltable.panglecal")
        else:
            print(f"No polarization leakage calibration table is present in : {caldir}")
            kcross_table = []
            crossphase_table = []
            pangle_table = []

        gaintable = []
        if len(bandpass_table) == 0:
            print(f"No bandpass table is present in calibration directory : {caldir}.")
            return []
        if len(gain_table) == 0:
            print(
                f"No time-dependent gaintable is present in calibration directory : {caldir}. Applying only bandpass solutions."
            )
            use_only_bandpass = True

        ################################
        # Scale bandpass for attenuators
        ################################
        if len(att_caltables) == 0:
            print(
                "No attenuation table is present. Bandpass is not scaled for attenuation."
            )
            scaled_bandpass_table = ""
        else:
            scaled_bandpass_table = scale_bandpass(bandpass_table[0], att_caltables)

        ###############################
        # Arranging applycal
        ###############################
        if len(delay_table) > 0:
            gaintable += delay_table
        if len(gain_table) > 0 and use_only_bandpass == False:
            gaintable += gain_table
        if len(leakage_table) > 0:
            gaintable += leakage_table
            if len(kcross_table) > 0:
                gaintable += kcross_table
            if len(crossphase_table) > 0:
                gaintable += crossphase_table
            if len(pangle_table) > 0:
                gaintable += pangle_table
        gaintable_bkp = copy.deepcopy(gaintable)
        for g in gaintable_bkp:
            if os.path.basename(g) == "full_selfcal.gcal":
                gaintable.remove(g)
        del gaintable_bkp

        ####################################
        # Filtering any corrupted ms
        #####################################
        filtered_mslist = []  # Filtering in case any ms is corrupted
        for ms in mslist:
            checkcol = check_datacolumn_valid(ms)
            if checkcol:
                filtered_mslist.append(ms)
            else:
                print(f"Issue in : {ms}")
                os.system("rm -rf {ms}")
        mslist = filtered_mslist
        if len(mslist) == 0:
            print("No valid measurement set.")
            print(f"Total time taken: {round(time.time()-start_time,2)}s")
            return 1

        ####################################
        # Applycal jobs
        ####################################
        print(f"Total ms list: {len(mslist)}")
        ms_size_list = [get_column_size(ms) for ms in mslist]
        mem_limit = max(ms_size_list)
        if dask_addr is None:
            dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = (
                get_dask_client(
                    len(mslist),
                    dask_dir=workdir,
                    cpu_frac=cpu_frac,
                    mem_frac=mem_frac,
                    min_mem_per_job=mem_limit,
                )
            )
        else:
            _, _, n_jobs, n_threads, mem_limit, dask_dir = get_dask_client(
                len(mslist),
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
        if scaled_bandpass_table != "" and os.path.exists(scaled_bandpass_table):
            bpass_table = scaled_bandpass_table
            if os.path.exists(f"{workdir}/.noattcal"):
                os.system(f"rm -rf {workdir}/.noattcal")
            os.system(f"touch {workdir}/.attcal")
        else:
            print("Bandpass tables with attenuation scaling are not found.")
            if os.path.exists(f"{workdir}/.attcal"):
                os.system(f"rm -rf {workdir}/.attcal")
            os.system(f"touch {workdir}/.noattcal")
            bpass_table = bandpass_table[0]
        for ms in mslist:
            interp = []
            final_gaintable = gaintable + [bpass_table]
            for g in final_gaintable:
                if ".gcal" in g:
                    interp.append("linear")
                elif ".kcal" in g:
                    interp.append("nearest")
                else:
                    interp.append("nearestflag")
            tasks.append(
                delayed(applysol)(
                    ms,
                    gaintable=final_gaintable,
                    overwrite_datacolumn=overwrite_datacolumn,
                    applymode=applymode,
                    interp=interp,
                    do_post_flag=do_post_flag,
                    n_threads=n_threads,
                    parang=parang,
                    memory_limit=mem_limit,
                    force_apply=force_apply,
                )
            )
        futures = dask_client.compute(tasks)
        results = list(dask_client.gather(futures))
        dask_client.close()
        if dask_addr is None:
            dask_cluster.close()
            os.system(f"rm -rf {dask_dir}")
        if np.nansum(results) == 0:
            print("##################")
            print(
                "Applying basic calibration solutions for target scans are done successfully."
            )
            print("Total time taken : ", time.time() - start_time)
            print("##################\n")
            return 0
        else:
            print("##################")
            print(
                "Applying basic calibration solutions for target scans are not done successfully."
            )
            print("Total time taken : ", time.time() - start_time)
            print("##################\n")
            return 1
    except Exception as e:
        traceback.print_exc()
        os.system("rm -rf casa*log")
        print("##################")
        print(
            "Applying basic calibration solutions for target scans are not done successfully."
        )
        print("Total time taken : ", time.time() - start_time)
        print("##################\n")
        return 1


def main(
    mslist,
    workdir,
    caldir,
    use_only_bandpass=False,
    applymode="calflag",
    overwrite_datacolumn=False,
    force_apply=False,
    do_post_flag=False,
    start_remote_log=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    dask_addr=None,
):
    """
    Apply calibration solutions to a list of measurement sets with optional post-flagging.

    Parameters
    ----------
    mslist : str
        Comma-separated list of measurement set paths to which calibration will be applied.
    workdir : str
        Directory for logs, PID files, and temporary data products.
    caldir : str
        Path to directory containing calibration tables (e.g., bandpass, gain, polarization).
    use_only_bandpass : bool, optional
        If True, applies only the bandpass calibration (no gain or polarization cal). Default is False.
    applymode : str, optional
        CASA calibration application mode (e.g., "calonly", "calflag", "flagonly"). Default is "calflag".
    overwrite_datacolumn : bool, optional
        If True, overwrites the CORRECTED column during calibration. Default is False.
    force_apply : bool, optional
        If True, forces re-application of calibration even if it appears already applied. Default is False.
    do_post_flag : bool, optional
        If True, performs a round of post-calibration flagging. Default is False.
    start_remote_log : bool, optional
        Whether to enable remote logging using job credentials in `workdir`. Default is False.
    cpu_frac : float, optional
        Fraction of CPU resources to allocate per worker. Default is 0.8.
    mem_frac : float, optional
        Fraction of system memory to allocate per worker. Default is 0.8.
    logfile : str or None, optional
        Path to the logfile. If None, logging to file is disabled. Default is None.
    jobid : int, optional
        Identifier for tracking the job and saving PID. Default is 0.
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
        workdir = os.path.dirname(os.path.abspath(mslist.split(",")[0])) + "/workdir"
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
                "apply_basiccal", logfile, jobname=jobname, password=password
            )
    if observer == None:
        print("Remote link or jobname is blank. Not transmiting to remote logger.")
    ############

    try:
        print("\n###################################")
        print("Starting applying solutions...")
        print("###################################\n")

        if caldir == "" or not os.path.exists(caldir):
            print("Provide existing caltable directory.")
            msg = 1
        else:
            mslist = mslist.split(",")
            msg = run_all_applysol(
                mslist,
                workdir,
                caldir,
                use_only_bandpass=use_only_bandpass,
                overwrite_datacolumn=overwrite_datacolumn,
                do_post_flag=do_post_flag,
                applymode=applymode,
                force_apply=force_apply,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                dask_addr=dask_addr,
            )
    except Exception:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        for ms in mslist:
            drop_cache(ms)
        drop_cache(workdir)
        clean_shutdown(observer)
    return msg


def cli():
    parser = argparse.ArgumentParser(
        description="Apply basic calibration solutions to target scans",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mslist",
        type=str,
        help="Comma-separated list of measurement sets (required)",
    )
    basic_args.add_argument(
        "--workdir",
        type=str,
        default="",
        required=True,
        help="Working directory for intermediate files",
    )
    basic_args.add_argument(
        "--caldir",
        type=str,
        default="",
        required=True,
        help="Directory containing calibration tables",
    )

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--use_only_bandpass",
        action="store_true",
        help="Use only bandpass calibration solutions",
    )
    adv_args.add_argument(
        "--applymode",
        type=str,
        default="calflag",
        help="Applycal mode (e.g. 'calonly', 'calflag')",
    )
    adv_args.add_argument(
        "--overwrite_datacolumn",
        action="store_true",
        help="Overwrite corrected data column in MS",
    )
    adv_args.add_argument(
        "--force_apply",
        action="store_true",
        help="Force apply calibration even if already applied",
    )
    adv_args.add_argument(
        "--do_post_flag", action="store_true", help="Perform post-calibration flagging"
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
        "--jobid", type=str, default="0", help="Job ID for logging and process tracking"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    msg = main(
        args.mslist,
        args.workdir,
        args.caldir,
        use_only_bandpass=args.use_only_bandpass,
        applymode=args.applymode,
        overwrite_datacolumn=args.overwrite_datacolumn,
        force_apply=args.force_apply,
        do_post_flag=args.do_post_flag,
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
        "\n###################\nApplying calibration solutions are done.\n###################\n"
    )
    os._exit(result)
