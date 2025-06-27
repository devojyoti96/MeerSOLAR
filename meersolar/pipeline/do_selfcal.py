import os, numpy as np, copy, psutil, gc, traceback, resource, time, argparse
from astropy.io import fits
from meersolar.pipeline.basic_func import *
from dask import delayed, compute, config
from functools import partial
from casatasks import casalog

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except:
    pass


def single_selfcal_iteration(
    msname,
    logger,
    selfcaldir,
    cellsize,
    imsize,
    round_number=0,
    uvrange="",
    minuv=0,
    calmode="ap",
    solint="60s",
    refant="1",
    solmode="",
    gaintype="T",
    applymode="calonly",
    threshold=3,
    weight="briggs",
    robust=0.0,
    use_previous_model=False,
    use_solar_mask=True,
    mask_radius=20,
    min_tol_factor=-1,
    ncpu=-1,
    mem=-1,
):
    """
    A single self-calibration round

    Parameters
    ----------
    msname : str
        Name of the measurement set
    logger : logger
        Python logger
    selfcaldir : str
        Self-calibration directory
    cellsize : float
        Cellsize in arcsec
    imsize :  int
        Image pixel size
    round_number : int, optional
        Selfcal iteration number
    uvrange : float, optional
       UV range for calibration
    calmode : str, optional
        Calibration mode ('p' or 'ap')
    solint : str, optional
        Solution intervals
    refant : str, optional
        Reference antenna
    applymode : str, optional
        Solution apply mode (calonly or calflag)
    threshold : float, optional
        Imaging and auto-masking threshold
    weight : str, optional
        Image weighting
    robust : float, optional
        Robust parameter for briggs weighting
    use_previous_model : bool, optional
        Use previous model
    use_solar_mask : bool, optional
        Use solar disk mask or not
    mask_radius : float, optional
        Mask radius in arcminute
    min_tol_factor : float, optional
        Minimum tolerance factor
    ncpu : int, optional
        Number of CPUs to use in WSClean
    mem : float, optional
        Memory usage limit in WSClean

    Returns
    -------
    int
        Success message
    str
        Caltable name
    float
        RMS based dynamic range
    float
        RMS of the image
    str
        Image name
    str
        Model image name
    str
        Residual image name
    """
    limit_threads(n_threads=ncpu)
    from casatasks import gaincal, bandpass, applycal, flagdata, delmod, flagmanager
    from casatools import msmetadata, table

    try:
        ##################################
        # Setup wsclean params
        ##################################
        if ncpu < 1:
            ncpu = psutil.cpu_count()
        if mem < 0:
            mem = round(psutil.virtual_memory().available / (1024**3), 2)
        msname = msname.rstrip("/")
        if use_previous_model == False:
            delmod(vis=msname, otf=True, scr=True)
        prefix = (
            selfcaldir
            + "/"
            + os.path.basename(msname).split(".ms")[0]
            + "_selfcal_present"
        )
        ############################
        # Determining channel blocks
        ############################
        msmd = msmetadata()
        msmd.open(msname)
        times = msmd.timesforspws(0)
        timeres = np.diff(times)
        pos = np.where(timeres > 3 * np.nanmedian(timeres))[0]
        max_intervals = min(1, len(pos))
        freqs = msmd.chanfreqs(0, unit="MHz")
        freqres = freqs[1] - freqs[0]
        freq_width = calc_bw_smearing_freqwidth(msname)
        nchan = int(freq_width / freqres)
        total_nchan = len(freqs)
        freq = msmd.meanfreq(0, unit="MHz")
        total_time = max(times) - min(times)
        msmd.close()
        chanres = np.diff(freqs)
        chanres /= np.nanmin(chanres)
        pos = np.where(chanres > 1)[0]
        chanrange_list = []
        start_chan = 0
        for i in range(len(pos) + 1):
            if i > len(pos) - 1:
                end_chan = total_nchan
            else:
                end_chan = pos[i] + 1
            if end_chan - start_chan > 10 or len(chanrange_list) == 0:
                chanrange_list.append(f"{start_chan} {end_chan}")
            else:
                last_chanrange = chanrange_list[-1]
                chanrange_list.remove(last_chanrange)
                start_chan = last_chanrange.split(" ")[0]
                chanrange_list.append(f"{start_chan} {end_chan}")
            start_chan = end_chan + 1
        if len(chanrange_list) == 0:
            unflag_chans, flag_chans = get_chans_flag(msname)
            chanrange_list = [f"{min(unflag_chans)} {max(unflag_chans)}"]

        ########################################
        # Scale bias list and channel range list
        ########################################
        scale_bias_list = []
        for chanrange in chanrange_list:
            start_chan = int(chanrange.split(" ")[0])
            end_chan = int(chanrange.split(" ")[-1])
            mid_chan = int((start_chan + end_chan) / 2)
            mid_freq = freqs[mid_chan]
            scale_bias = round(get_multiscale_bias(mid_freq), 2)
            scale_bias_list.append(scale_bias)

        ############################################
        # Merge channel ranges with identical scale bias
        ############################################
        merged_channels = []
        merged_biases = []
        start, end = map(int, chanrange_list[0].split())
        current_bias = scale_bias_list[0]
        for i in range(1, len(chanrange_list)):
            next_start, next_end = map(int, chanrange_list[i].split())
            next_bias = scale_bias_list[i]
            if next_bias == current_bias:
                # Merge ranges (irrespective of contiguity)
                end = next_end
            else:
                # Finalize current group
                merged_channels.append(f"{start} {end}")
                merged_biases.append(current_bias)
                # Start new group
                start, end = next_start, next_end
                current_bias = next_bias
        # Final group
        merged_channels.append(f"{start} {end}")
        merged_biases.append(current_bias)
        chanrange_list = copy.deepcopy(merged_channels)
        scale_bias_list = copy.deepcopy(merged_biases)
        del merged_channels, merged_biases

        ###############################
        # Temporal chunking list
        ###############################
        nintervals = 1
        nchan_list = []
        nintervals_list = []
        for i in range(len(chanrange_list)):
            chanrange = chanrange_list[i]
            ###########################################################################
            # Spectral variation is kept fixed at 10% level.
            # Because selfcal is done along temporal axis, where variation matters most
            ###########################################################################
            if min_tol_factor <= 0:
                min_tol_factor = 1.0
            nintervals, _ = get_optimal_image_interval(
                msname,
                chan_range=f"{chanrange.replace(' ',',')}",
                temporal_tol_factor=float(min_tol_factor / 100.0),
                spectral_tol_factor=0.1,
                max_nchan=-1,
                max_ntime=max_intervals,
            )
            nchan_list.append(nchan)
            nintervals_list.append(nintervals)

        os.system(f"rm -rf {prefix}*image.fits {prefix}*residual.fits")

        if weight == "briggs":
            weight += " " + str(robust)

        wsclean_args = [
            "-quiet",
            "-scale " + str(cellsize) + "asec",
            "-size " + str(imsize) + " " + str(imsize),
            "-no-dirty",
            "-gridder tuned-wgridder",
            "-weight " + weight,
            "-niter 10000",
            "-mgain 0.85",
            "-nmiter 5",
            "-gain 0.1",
            "-minuv-l " + str(minuv),
            "-j " + str(ncpu),
            "-abs-mem " + str(mem),
            "-no-negative",
            "-auto-mask " + str(threshold + 0.1),
            "-auto-threshold " + str(threshold),
        ]

        ngrid = int(ncpu / 2)
        if ngrid > 1:
            wsclean_args.append("-parallel-gridding " + str(ngrid))

        ################################################
        # Creating and using solar mask
        ################################################
        if use_solar_mask:
            fits_mask = msname.split(".ms")[0] + "_solar-mask.fits"
            if os.path.exists(fits_mask) == False:
                logger.info(f"Creating solar mask of size: {mask_radius} arcmin.\n")
                fits_mask = create_circular_mask(
                    msname, cellsize, imsize, mask_radius=mask_radius
                )
            if fits_mask != None and os.path.exists(fits_mask):
                wsclean_args.append("-fits-mask " + fits_mask)

        ######################################
        # Running imaging per channel range
        ######################################
        final_image_list = []
        final_model_list = []
        final_residual_list = []

        for i in range(len(chanrange_list)):
            chanrange = chanrange_list[i]
            per_chanrange_wsclean_args = copy.deepcopy(wsclean_args)

            ######################################
            # Multiscale configuration
            ######################################
            start_chan = int(chanrange.split(" ")[0])
            end_chan = int(chanrange.split(" ")[-1])
            chan_number = int((start_chan + end_chan) / 2)
            multiscale_scales = calc_multiscale_scales(
                msname, 3, chan_number=chan_number
            )
            per_chanrange_wsclean_args.append("-multiscale")
            per_chanrange_wsclean_args.append("-multiscale-gain 0.1")
            per_chanrange_wsclean_args.append(
                "-multiscale-scales " + ",".join([str(s) for s in multiscale_scales])
            )
            per_chanrange_wsclean_args.append(
                f"-multiscale-scale-bias {scale_bias_list[i]}"
            )
            if imsize >= 2048 and 4 * max(multiscale_scales) < 1024:
                per_chanrange_wsclean_args.append("-parallel-deconvolution 1024")

            ###########################################################################
            # Spectral variation is kept fixed at 10% level.
            # Because selfcal is done along temporal axis, where variation matters most
            ###########################################################################
            if len(scale_bias_list) > 1:
                if min_tol_factor <= 0:
                    min_tol_factor = 1.0
                nintervals, _ = get_optimal_image_interval(
                    msname,
                    chan_range=f"{chanrange.replace(' ',',')}",
                    temporal_tol_factor=float(min_tol_factor / 100.0),
                    spectral_tol_factor=0.1,
                )

            #####################################
            # Spectral imaging configuration
            #####################################
            if nchan > 1:
                per_chanrange_wsclean_args.append(f"-channels-out {nchan}")
                per_chanrange_wsclean_args.append("-no-mf-weighting")
                per_chanrange_wsclean_args.append("-join-channels")

            #####################################
            # Temporal imaging configuration
            #####################################
            if nintervals > 1:
                per_chanrange_wsclean_args.append(f"-intervals-out {nintervals}")
            logger.info(f"Spectral chunks: {nchan}, temporal chunks: {nintervals}.")
            temp_prefix = f"{prefix}_chan_{chanrange.replace(' ','_')}"
            per_chanrange_wsclean_args.append(f"-name {temp_prefix}")
            per_chanrange_wsclean_args.append(f"-channel-range {chanrange}")

            if use_previous_model:
                previous_models = glob.glob(f"{temp_prefix}*model.fits")
                if nchan > 1:
                    total_models_expected = (nchan + 1) * nintervals
                else:
                    total_models_expected = (nchan) * nintervals
                if len(previous_models) == total_models_expected:
                    per_chanrange_wsclean_args.append("-continue")
                else:
                    os.system(f"rm -rf {temp_prefix}*")

            wsclean_cmd = (
                "wsclean " + " ".join(per_chanrange_wsclean_args) + " " + msname
            )
            logger.info(f"WSClean command: {wsclean_cmd}\n")
            msg = run_wsclean(wsclean_cmd, "meerwsclean", verbose=False)
            if msg != 0:
                gc.collect()
                logger.info(f"Imaging is not successful.\n")
            else:
                #####################################
                # Analyzing images
                #####################################
                wsclean_files = {}
                for suffix in ["image", "model", "residual"]:
                    files = glob.glob(temp_prefix + f"*MFS-{suffix}.fits")
                    if not files:
                        files = glob.glob(temp_prefix + f"*{suffix}.fits")
                    wsclean_files[suffix] = files

                wsclean_images = wsclean_files["image"]
                wsclean_models = wsclean_files["model"]
                wsclean_residuals = wsclean_files["residual"]

                final_image = (
                    temp_prefix.replace("present", f"{round_number}") + "_I_image.fits"
                )
                final_model = (
                    temp_prefix.replace("present", f"{round_number}") + "_I_model.fits"
                )
                final_residual = (
                    temp_prefix.replace("present", f"{round_number}")
                    + "_I_residual.fits"
                )

                if len(wsclean_images) == 0:
                    print("No image is made.")
                elif len(wsclean_images) == 1:
                    os.system(f"cp -r {wsclean_images[0]} {final_image}")
                else:
                    final_image = make_timeavg_image(
                        wsclean_images, final_image, keep_wsclean_images=True
                    )
                final_image_list.append(final_image)
                if len(wsclean_models) == 1:
                    os.system(f"cp -r {wsclean_models[0]} {final_model}")
                else:
                    final_model = make_timeavg_image(
                        wsclean_models, final_model, keep_wsclean_images=True
                    )
                final_model_list.append(final_model)
                if len(wsclean_residuals) == 1:
                    os.system(f"cp -r {wsclean_residuals[0]} {final_residual}")
                else:
                    final_residual = make_timeavg_image(
                        wsclean_residuals, final_residual, keep_wsclean_images=True
                    )
                final_residual_list.append(final_residual)

        #########################################
        # Restoring flags if applymode is calflag
        #########################################
        if applymode == "calflag":
            with suppress_casa_output():    
                flags = flagmanager(vis=msname, mode="list")
            keys = flags.keys()
            for k in keys:
                if k == "MS":
                    pass
                else:
                    version = flags[0]["name"]
                    if "applycal" in version:
                        try:
                            with suppress_casa_output():
                                flagmanager(vis=msname, mode="restore", versionname=version)
                                flagmanager(vis=msname, mode="delete", versionname=version)
                        except:
                            pass

        ##########################################################################
        # Final frequency averaged images for backup or calculating dynamic ranges
        ##########################################################################
        final_image = prefix.replace("present", f"{round_number}") + "_I_image.fits"
        final_model = prefix.replace("present", f"{round_number}") + "_I_model.fits"
        final_residual = (
            prefix.replace("present", f"{round_number}") + "_I_residual.fits"
        )
        if len(final_image_list) == 1:
            os.system(f"mv {final_image_list[0]} {final_image}")
        else:
            final_image = make_freqavg_image(
                final_image_list, final_image, keep_wsclean_images=False
            )
        if len(final_model_list) == 1:
            os.system(f"mv {final_model_list[0]} {final_model}")
        else:
            final_model = make_freqavg_image(
                final_model_list, final_model, keep_wsclean_images=False
            )
        if len(final_residual_list) == 1:
            os.system(f"mv {final_residual_list[0]} {final_residual}")
        else:
            final_residual = make_freqavg_image(
                final_residual_list, final_residual, keep_wsclean_images=False
            )
        os.system("rm -rf *psf.fits")
        #####################################
        # Calculating dynamic ranges
        ######################################
        model_flux, rms_DR, rms = calc_dyn_range(
            final_image,
            final_model,
            final_residual,
            fits_mask=fits_mask,
        )
        if model_flux == 0:
            gc.collect()
            logger.info(f"No model flux.\n")
            return 1, "", 0, 0, "", "", ""

        #####################
        # Perform calibration
        #####################
        bpass_caltable = prefix.replace("present", f"{round_number}") + ".gcal"
        if os.path.exists(bpass_caltable):
            os.system("rm -rf " + bpass_caltable)

        logger.info(
            f"bandpass(vis='{msname}',caltable='{bpass_caltable}',uvrange='{uvrange}',refant='{refant}',solint='{solint},10MHz',minsnr=1,solnorm=True)\n"
        )
        with suppress_casa_output():
            bandpass(
                vis=msname,
                caltable=bpass_caltable,
                uvrange=uvrange,
                refant=refant,
                minsnr=1,
                solint=f"{solint},10MHz",
                solnorm=True,
            )
        if os.path.exists(bpass_caltable) == False:
            logger.info(f"No gain solutions are found.\n")
            gc.collect()
            return 2, "", 0, 0, "", "", ""

        #########################################
        # Flagging bad gains
        #########################################
        with suppress_casa_output():
            flagdata(
                vis=bpass_caltable, mode="rflag", datacolumn="CPARAM", flagbackup=False
            )
        tb = table()
        tb.open(bpass_caltable, nomodify=False)
        gain = tb.getcol("CPARAM")
        if calmode == "p":
            gain /= np.abs(gain)
        flag = tb.getcol("FLAG")
        gain[flag] = 1.0
        pos = np.where(np.abs(gain) == 0.0)
        gain[pos] = 1.0
        flag *= False
        tb.putcol("CPARAM", gain)
        tb.putcol("FLAG", flag)
        tb.flush()
        tb.close()

        logger.info(
            f"applycal(vis={msname},gaintable=[{bpass_caltable}],interp=['linear,linearflag'],applymode='{applymode}',calwt=[False])\n"
        )
        with suppress_casa_output():
            applycal(
                vis=msname,
                gaintable=[bpass_caltable],
                interp=["linear,linearflag"],
                applymode=applymode,
                calwt=[False],
            )

        #####################################
        # Flag zeros
        #####################################
        with suppress_casa_output():
            flagdata(
                vis=msname,
                mode="clip",
                clipzeros=True,
                datacolumn="corrected",
                flagbackup=False,
            )
        gc.collect()
        return (
            0,
            bpass_caltable,
            rms_DR,
            rms,
            final_image,
            final_model,
            final_residual,
        )
    except Exception as e:
        traceback.print_exc()
        return 3, "", 0, 0, "", "", ""


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
    from casatools import msmetadata

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    sub_observer = None
    logger, logfile = create_logger(
        os.path.basename(logfile).split(".log")[0], logfile, verbose=False
    )
    if os.path.exists(f"{workdir}/jobname_password.npy") and logfile != None:
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
        os.makedirs(selfcaldir,exist_ok=True)
        
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
                except:
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
        fov = min(instrument_fov, 32 * 2 * 60)  # 2 solar radii
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
        msmd = msmetadata()
        msmd.open(msname)
        msmd.close()

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
                single_selfcal_iteration(
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
                    ) = single_selfcal_iteration(
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

            #####################
            # If DR is decreasing
            #####################
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
            elif (
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
            ###########################
            # Checking DR convergence
            ###########################
            # Condition 1
            ###########################
            if (
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
            ###############
            # Condition 2
            ###############
            else:
                if (
                    abs(DR1 - DR2) / DR2 < DR_convegerence_frac
                    and num_iter > min_iter
                    and threshold == end_threshold + 1
                ):
                    if do_apcal and calmode == "p":
                        logger.info(
                            f"Dynamic range converged. Changing calmode to 'ap'.\n"
                        )
                        calmode = "ap"
                        if num_iter_fixed_sigma > min_iter:
                            threshold -= 1
                            sigma_reduced_count += 1
                            num_iter_fixed_sigma = 0
                    elif (
                        do_apcal and num_iter_after_ap > min_iter
                    ) or do_apcal == False:
                        logger.info(f"Self-calibration has converged.\n")
                        os.system("rm -rf *_selfcal_present*")
                        time.sleep(5)
                        clean_shutdown(sub_observer)
                        return 0, gaintable
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
                elif (
                    (
                        (do_apcal and calmode == "ap" and num_iter_after_ap > min_iter)
                        or (do_apcal == False and num_iter > min_iter)
                    )
                    and abs(DR1 - DR2) / DR2 < DR_convegerence_frac
                    and threshold <= end_threshold
                ) or (
                    (do_apcal == False or (do_apcal and calmode == "ap"))
                    and num_iter == max_iter
                ):
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
    parser = argparse.ArgumentParser(description="Self-calibration",formatter_class=SmartDefaultsHelpFormatter)
    
    ## Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mslist",
        type=str,
        help="Comma-separated list of measurement sets (required positional argument)",
    )
    basic_args.add_argument(
        "--workdir", type=str, default="", required=True, help="Working directory", 
    )
    basic_args.add_argument(
        "--caldir", type=str, default="", required=True, help="Caltable directory",
    )
    
    ## Advanced parameters
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
    adv_args.add_argument(
        "--solint", type=str, default="60s", help="Solution interval"
    )
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
    adv_args.add_argument(
        "--weight", type=str, default="briggs", help="Imaging weight"
    )
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
    adv_args.add_argument("--no_apcal", action="store_false", dest="do_apcal", help="Do not perform ap-selfcal")
    adv_args.add_argument(
        "--no_solar_selfcal", action="store_false", dest="solar_selfcal", help="Do not perform solar self-calibration"
    )
    adv_args.add_argument(
        "--keep_backup",
        action="store_true",
        help="Keep backup of self-calibration rounds",
    )
    
    ## Resource management parameters
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
    hard_args.add_argument(
        "--jobid", type=int, default=0, help="Job ID"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    pid = os.getpid()
    save_pid(pid, datadir + f"/pids/pids_{args.jobid}.txt")
    if args.workdir == "" or os.path.exists(args.workdir) == False:
        workdir = os.path.dirname(os.path.abspath(args.msname)) + "/workdir"            
    else:
        workdir = args.workdir
    os.makedirs(workdir,exist_ok=True)
    
    if args.caldir=="" or not os.path.exists(args.caldir):
        caldir=f"{workdir}/caltables"
    else:
        caldir=args.caldir
    os.makedirs(caldir,exist_ok=True)
        
    os.makedirs(workdir + "/logs",exist_ok=True)
    mainlog_file = workdir + "/logs/selfcal_targets.mainlog"
    mainlogger, mainlog_file = create_logger(
        os.path.basename(mainlog_file).split(".mainlog")[0], mainlog_file, verbose=False
    )
    observer = None
    if os.path.exists(f"{workdir}/jobname_password.npy") and mainlog_file != None:
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
    if container_present == False:
        container_name = initialize_wsclean_container(name=container_name)
        if container_name == None:
            print(
                "Container {container_name} is not initiated. First initiate container and then run."
            )
            return 1
    mslist = str(args.mslist).split(",")
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
                resource.setrlimit(
                    resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit)
                )

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

            dask_client, dask_cluster, n_jobs, n_threads, mem_limit = (
                get_dask_client(
                    n_jobs,
                    dask_dir=args.workdir,
                    cpu_frac=float(args.cpu_frac),
                    mem_frac=float(args.mem_frac),
                    min_cpu_per_job=3,
                    min_mem_per_job=min_mem_per_job,
                )
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
            if args.keep_backup == False:
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
                mainlogger.info(
                    f"Total time taken: {round(time.time()-starttime,2)}s"
                )
                mainlogger.info("################################################")
                msg = 0
            else:
                mainlogger.info("No self-calibration is successful.")
                mainlogger.info("################################################")
                mainlogger.info(
                    f"Total time taken: {round(time.time()-starttime,2)}s"
                )
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
