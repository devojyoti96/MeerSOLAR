import numpy as np, glob, os, copy, warnings, traceback, gc, argparse
from casatools import table
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from meersolar.pipeline.basic_func import *
from dask import delayed, compute
from casatasks import casalog

try:
    logfile = casalog.logfile()
    os.system("rm -rf " + logfile)
except:
    pass


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


def scale_bandpass(bandpass_table, att_table, freqavg=10):
    """
    Scale a bandpass calibration table using attenuation data.


    Parameters
    ----------
    bandpass_table : str
        Input bandpass calibration table.
    att_table : str
        NumPy .npy file containing attenuation frequency and values.
    freqavg : float, optional
        Frequency average in MHz for polynomial fitting (default is 10 MHz). Final table has same number of channels as input.

    Returns
    -------
    str
        Name of the output table.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if att_table == "":
        print(f"No attenuation caltable is provided for scan : {scan}")
        return
    print(f"Bandpass table: {bandpass_table}, Attenuation table: {att_table}")
    results = np.load(att_table, allow_pickle=True)
    scan, freqs, att_values, flag_ants, att_array = results
    freqres = abs(freqs[1] - freqs[0]) / 10**6  # In MHz
    n = int(freqavg / freqres)
    output_table = bandpass_table.split(".bcal")[0] + "_scan_" + str(scan) + ".bcal"
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
        num_blocks = att.shape[0] // n
        att_avg = np.nanmedian(att[: num_blocks * n].reshape(-1, n), axis=1)
        freq_avg = freqs[: num_blocks * n].reshape(-1, n).mean(axis=1)
        valid = ~np.isnan(att_avg)
        best_fit, best_std = None, np.inf
        for deg in range(3, 9):
            coeffs = np.polyfit(freq_avg[valid], att_avg[valid], deg)
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
    from meersolar.pipeline.flagging import single_ms_flag

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
            if os.path.exists(msname + check_file) and force_apply == True:
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
                    calwt=[False]*len(gaintable),
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
            gc.collect()
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
):
    """
    Apply self-calibrator solutions on all target scans

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
            scaled_bandpass_list = []
        else:
            dask_client, dask_cluster, n_jobs, n_threads, mem_limit = get_dask_client(
                len(att_caltables),
                dask_dir=workdir,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
            )
            tasks = []
            for att_table in att_caltables:
                tasks.append(delayed(scale_bandpass)(bandpass_table[0], att_table))
            scaled_bandpass_list = compute(*tasks)
            dask_client.close()
            dask_cluster.close()

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
        task = delayed(applysol)(dry_run=True)
        mem_limit = run_limited_memory_task(task, dask_dir=workdir)
        ms_size_list = [get_ms_size(ms) + mem_limit for ms in mslist]
        mem_limit = max(ms_size_list)
        dask_client, dask_cluster, n_jobs, n_threads, mem_limit = get_dask_client(
            len(mslist),
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit / 0.6,
        )
        tasks = []
        if len(scaled_bandpass_list) > 0:
            scaled_bandpass_scans = [
                int(a.split("scan_")[-1].split(".bcal")[0])
                for a in scaled_bandpass_list
            ]
        msmd = msmetadata()
        for ms in mslist:
            msmd.open(ms)
            scans = msmd.scannumbers()
            msmd.close()
            for scan in scans:
                if len(scaled_bandpass_list) > 0:
                    pos = scaled_bandpass_scans.index(scan)
                    bpass_table = scaled_bandpass_list[pos]
                else:
                    bpass_table = bandpass_table[0]
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
        results = compute(*tasks)
        dask_client.close()
        dask_cluster.close()
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
    finally:
        time.sleep(5)
        for ms in mslist:
            drop_cache(ms)
        drop_cache(workdir)


def main():
    parser = argparse.ArgumentParser(
        description="Apply basic calibration solutions to target scans",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    ## Essential parameters
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
        help="Working directory for intermediate files",
    )
    basic_args.add_argument(
        "--caldir", type=str, default="", help="Directory containing calibration tables"
    )

    ## Advanced parameters
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

    ## Resource management parameters
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

    pid = os.getpid()
    save_pid(pid, datadir + f"/pids/pids_{args.jobid}.txt")

    if args.workdir == "" or not os.path.exists(args.workdir):
        workdir = (
            os.path.dirname(os.path.abspath(args.mslist.split(",")[0])) + "/workdir"
        )
    else:
        workdir = args.workdir

    os.makedirs(workdir,exist_ok=True)
    
    logfile = args.logfile
    observer = None

    if os.path.exists(f"{workdir}/jobname_password.npy") and logfile is not None:
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            observer = init_logger(
                "apply_basiccal", logfile, jobname=jobname, password=password
            )

    try:
        print("\n###################################")
        print("Starting applying solutions...")
        print("###################################\n")

        if args.workdir == "" or not os.path.exists(args.workdir):
            print("Provide existing work directory name.")
            msg = 1
        elif args.caldir == "" or not os.path.exists(args.caldir):
            print("Provide existing caltable directory.")
            msg = 1
        else:
            mslist = args.mslist.split(",")
            msg = run_all_applysol(
                mslist,
                args.workdir,
                args.caldir,
                use_only_bandpass=args.use_only_bandpass,
                overwrite_datacolumn=args.overwrite_datacolumn,
                do_post_flag=args.do_post_flag,
                applymode=args.applymode,
                force_apply=args.force_apply,
                cpu_frac=args.cpu_frac,
                mem_frac=args.mem_frac,
            )
    except Exception:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        for ms in mslist:
            drop_cache(ms)
        drop_cache(args.workdir)
        clean_shutdown(observer)
    return msg


if __name__ == "__main__":
    result = main()
    print(
        "\n###################\nApplying calibration solutions are done.\n###################\n"
    )
    os._exit(result)
