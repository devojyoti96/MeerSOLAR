import os, glob, resource, traceback, psutil, time, copy, math, argparse
from meersolar.pipeline.basic_func import *
from dask import delayed, compute
from casatasks import casalog

try:
    logfile = casalog.logfile()
    os.system("rm -rf " + logfile)
except:
    pass


def rename_image(
    imagename,
    imagedir="",
    pol="",
    band="",
    cutout_rsun=2.5,
    make_overlay=True,
    make_plots=True,
):
    """
    Rename and move image to image directory

    Parameters
    ----------
    imagename : str
        Image name
    imagedir : str, optional
        Image directory (default given image directory)
    pol : str, optional
        Stokes parameters
    band : str, optional
        Observing band
    cutout_rsun : float, optional
        Cutout in solar radii from center (default: 2.5 solar radii)
    make_overlay : bool, optional
        Make overlay on SUVI
    make_plots : bool, optional
        Make radio map plot in helioprojective coordinates

    Returns
    -------
    str
        New imagename with full path
    """
    imagename = imagename.rstrip("/")
    imagename = cutout_image(
        imagename, imagename, x_deg=(cutout_rsun * 2 * 16.0) / 60.0
    )
    header = fits.getheader(imagename)
    time = header["DATE-OBS"]
    astro_time=Time(time,scale='utc')
    sun_jpl = Horizons(id='10', location='500', epochs=astro_time.jd)
    eph = sun_jpl.ephemerides()
    sun_coords = SkyCoord(ra=eph['RA'][0]*u.deg, dec=eph['DEC'][0]*u.deg, frame='icrs')
    maxval, minval, rms, total_val, mean_val, median_val, rms_dyn, minmax_dyn = calc_solar_image_stat(imagename,disc_size=18)
    with fits.open(imagename, mode="update") as hdul:
        hdr = hdul[0].header
        hdr["AUTHOR"] = "DevojyotiKansabanik,DeepanPatra"
        if band != "":
            hdr["BAND"] = band
        hdr["PIPELINE"]="MeerSOLAR"
        hdr["CRVAL1"] = sun_coords.ra.deg
        hdr["CRVAL2"] = sun_coords.dec.deg
        hdr["MAX"]=maxval
        hdr["MIN"]=minval
        hdr["RMS"]=rms
        hdr["SUM"]=total_val
        hdr["MEAN"]=mean_val
        hdr["MEDIAN"]=median_val
        hdr["RMSDYN"]=rms_dyn
        hdr["MIMADYN"]=minmax_dyn
    freq = round(header["CRVAL3"] / 10**6, 2)
    t_str = "".join(time.split("T")[0].split("-")) + (
        "".join(time.split("T")[-1].split(":"))
    )
    new_name = "time_" + t_str + "_freq_" + str(freq)
    if pol != "":
        new_name += "_pol_" + str(pol)
    if "MFS" in imagename:
        new_name += "_MFS"
    new_name = new_name + ".fits"
    if imagedir=="":
        imagedir=os.path.dirname(os.path.abspath(imagename))
    new_name = imagedir + "/" + new_name
    os.system("mv " + imagename + " " + new_name)
    hpcdir=f"{os.path.dirname(imagedir)}/images/hpcs"
    os.makedirs(hpcdir, exist_ok=True)
    save_in_hpc(new_name,outdir=hpcdir)
    if make_plots:
        try:
            pngdir = f"{os.path.dirname(imagedir)}/images/pngs"
            pdfdir = f"{os.path.dirname(imagedir)}/images/pdfs"
            os.makedirs(pngdir, exist_ok=True)
            os.makedirs(pdfdir, exist_ok=True)
            outimages,cropped_map=plot_in_hpc(
                new_name,
                draw_limb=True,
                extensions=["png","pdf"],
                outdirs=[pngdir,pdfdir],
            )
        except:
            pass
    if make_overlay:
        try:
            overlay_pngdir = f"{os.path.dirname(imagedir)}/overlays_pngs"
            overlay_pdfdir = f"{os.path.dirname(imagedir)}/overlays_pdfs"
            os.makedirs(overlay_pngdir, exist_ok=True)
            os.makedirs(overlay_pdfdir, exist_ok=True)
            outimages=make_meer_overlay(
                new_name,
                plot_file_prefix=os.path.basename(new_name).split(".fits")[0]
                + "_suvi_meerkat_overlay",
                extensions=["png","pdf"],
                outdirs=[overlay_pngdir,overlay_pdfdir]
            )
        except Exception as e:
            traceback.print_exc()
            pass
    return new_name


def perform_imaging(
    msname="",
    workdir="",
    datacolumn="CORRECTED_DATA",
    freqrange="",
    timerange="",
    imagedir="",
    imsize=1024,
    cellsize=2,
    nchan=1,
    ntime=1,
    pol="I",
    weight="briggs",
    robust=0.0,
    minuv=0,
    threshold=1.0,
    use_multiscale=True,
    use_solar_mask=True,
    mask_radius=25,
    savemodel=True,
    saveres=True,
    ncpu=-1,
    mem=-1,
    band="",
    cutout_rsun=2.5,
    make_overlay=True,
    make_plots=True,
    logfile="imaging.log",
    dry_run=False,
):
    """
    Perform spectropolarimetric snapshot imaging of a ms

    Parameters
    ----------
    msname : str
        Name of the measurement set
    workdir : str
        Work directory name
    datacolumn : str, optional
        Data column
    freqrange : str, optional
        Frequency range to image
    imagedir : str, optional
        Image directory name (default: workdir). Images, models, residuals will be saved in directories named images. models, residuals inside imagedir
    imsize : int, optional
        Image size in pixels
    cellsize : float, optional
        Cell size in arcseconds
    nchan : int, optional
        Number of spectral channels
    ntime : int, optional
        Number of temporal slices
    pol : str, optional
        Stokes parameters to image
    weight : str, optional
        Image weighting scheme
    robust : float, optional
        Briggs weighting robustness parameter
    minuv : float, optional
        Minimum UV-lambda to be used in imaging
    threshold : float, optional
        CLEAN threshold
    use_multiscale : bool, optional
        Use multiscale or not
    use_solar_mask : bool, optional
        Use solar mask
    mask_radius : float, optional
        Mask radius in arcminute
    savemodel : bool, optional
        Save model images or not
    saveres : bool, optional
        Save residual images or not
    band : str, optional
        Band name
    cutout_rsun : float, optional
        Cutout image size in solar radii from center (default: 2.5 solar radii)
    make_overlay : bool, optional
        Make SUVI MeerKAT overlay
    make_plots : bool, optional
        Make radio map helioprojective plots
    logfile : str, optional
        Log file name
    ncpu : int, optional
        Number of CPU threads to use
    mem : float, optional
        Memory in GB to use

    Returns
    -------
    int
        Success message
    list
        List of images [[images],[models],[residuals]]
    """
    if os.path.exists(logfile):
        os.system(f"rm -rf {logfile}")
    if dry_run:
        process = psutil.Process(os.getpid())
        usemem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return usemem
    logger, logfile = create_logger(
        os.path.basename(logfile).split(".log")[0], logfile, verbose=False
    )
    sub_observer = None
    if os.path.exists(f"{workdir}/jobname_password.npy") and logfile != None:
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            sub_observer = init_logger(
                "remotelogger_imaging_{os.path.basename(msname).split('.ms')[0]}",
                logfile,
                jobname=jobname,
                password=password,
            )
    try:
        msname = msname.rstrip("/")
        msname = os.path.abspath(msname)
        if band == "":
            band = get_band_name(msname)
        logger.info(f"{os.path.basename(msname)} --Perform imaging...\n")
        #########
        # Imaging
        #########
        msmd = msmetadata()
        msmd.open(msname)
        freq = msmd.meanfreq(0, unit="MHz")
        freqs = msmd.chanfreqs(0, unit="MHz")
        times = msmd.timesforspws(0)
        npol = msmd.ncorrforpol()[0]
        msmd.close()
        ###################################
        # Finding channel and time ranges
        ###################################
        if freqrange != "":
            start_chans = []
            end_chans = []
            freq_list = freqrange.split(",")
            for f in freq_list:
                start_freq = float(f.split("~")[0])
                end_freq = float(f.split("~")[-1])
                if start_freq >= np.nanmin(freqs) and end_freq <= np.nanmax(freqs):
                    start_chan = np.argmin(np.abs(start_freq - freqs))
                    end_chan = np.argmin(np.abs(end_freq - freqs))
                    start_chans.append(start_chan)
                    end_chans.append(end_chan)
        else:
            start_chans = [0]
            end_chans = [len(freqs)]
        if len(start_chans) == 0:
            print(f"Please provide valid channel range between 0 and {len(freqs)}")
            time.sleep(5)
            clean_shutdown(sub_observer)
            return 1, []
        if timerange != "":
            start_times = []
            end_times = []
            time_list = timerange.split(",")
            for timerange in time_list:
                start_time = timestamp_to_mjdsec(timerange.split("~")[0])
                end_time = timestamp_to_mjdsec(timerange.split("~")[-1])
                if start_time >= np.nanmin(times) and end_time <= np.nanmax(times):
                    start_times.append(np.argmin(np.abs(times - start_time)))
                    end_times.append(np.argmin(np.abs(times - end_time)))
        else:
            start_times = [0]
            end_times = [len(times)]
        if len(start_times) == 0:
            print(
                f"Please provide valid time range between {mjdsec_to_timestamp(times[0])} and {mjdsec_to_timestamp(times[-1])}"
            )
            time.sleep(5)
            clean_shutdown(sub_observer)
            return 1, []

        if npol < 4 and pol == "IQUV":
            pol = "I"
        if ncpu < 1:
            ncpu = psutil.cpu_count()
        if mem < 0:
            mem = psutil.virtual_memory().total / (1024**3)
        prefix = workdir + "/imaging_" + os.path.basename(msname).split(".ms")[0]
        if imagedir == "":
            imagedir = workdir
        os.makedirs(imagedir,exist_ok=True)
        if weight == "briggs":
            weight += " " + str(robust)
        if threshold <= 1:
            threshold = 1.1

        wsclean_args = [
            "-quiet",
            "-scale " + str(cellsize) + "asec",
            "-size " + str(imsize) + " " + str(imsize),
            "-no-dirty",
            "-gridder tuned-wgridder",
            "-weight " + weight,
            "-name " + prefix,
            "-pol " + str(pol),
            "-niter 10000",
            "-mgain 0.85",
            "-nmiter 5",
            "-gain 0.1",
            "-minuv-l " + str(minuv),
            "-j " + str(ncpu),
            "-abs-mem " + str(round(mem, 2)),
            "-auto-threshold 1 -auto-mask " + str(threshold),
            "-no-update-model-required",
        ]
        if datacolumn != "CORRECTED_DATA" and datacolumn != "corrected":
            wsclean_args.append("-data-column " + datacolumn)

        ngrid = int(ncpu / 2)
        if ngrid > 1:
            wsclean_args.append("-parallel-gridding " + str(ngrid))

        if pol == "I":
            wsclean_args.append("-no-negative")

        #####################################
        # Spectral imaging configuration
        #####################################
        if nchan > 1:
            wsclean_args.append(f"-channels-out {nchan}")
            wsclean_args.append("-no-mf-weighting")
            wsclean_args.append("-join-channels")

        #####################################
        # Temporal imaging configuration
        #####################################
        if ntime > 1:
            wsclean_args.append(f"-intervals-out {ntime}")

        ################################################
        # Creating and using a solar mask
        ################################################
        if use_solar_mask:
            fits_mask = prefix + "_solar-mask.fits"
            if os.path.exists(fits_mask) == False:
                logger.info(
                    f"{os.path.basename(msname)} -- Creating solar mask of size: {mask_radius} arcmin.\n",
                )
                fits_mask = create_circular_mask(
                    msname, cellsize, imsize, mask_radius=mask_radius
                )
            if fits_mask != None and os.path.exists(fits_mask):
                wsclean_args.append("-fits-mask " + fits_mask)
        final_list = []
        for i in range(len(start_chans)):
            for j in range(len(start_times)):
                temp_wsclean_args = copy.deepcopy(wsclean_args)
                temp_wsclean_args.append(
                    f"-channel-range {start_chans[i]} {end_chans[i]}"
                )
                temp_wsclean_args.append(f"-interval {start_times[j]} {end_times[j]}")

                ######################################
                # Multiscale configuration
                ######################################
                if use_multiscale:
                    num_pixel_in_psf = calc_npix_in_psf(weight, robust=robust)
                    multiscale_scales = calc_multiscale_scales(
                        msname,
                        num_pixel_in_psf,
                        chan_number=int((start_chans[i] + end_chans[i]) / 2),
                    )
                    temp_wsclean_args.append("-multiscale")
                    temp_wsclean_args.append("-multiscale-gain 0.1")
                    temp_wsclean_args.append(
                        "-multiscale-scales "
                        + ",".join([str(s) for s in multiscale_scales])
                    )
                    mid_freq = np.nanmean(
                        freqs[int(start_chans[i]) : int(end_chans[i])]
                    )
                    scale_bias = get_multiscale_bias(mid_freq)
                    temp_wsclean_args.append(f"-multiscale-scale-bias {scale_bias}")
                    if imsize >= 2048 and 4 * max(multiscale_scales) < 1024:
                        temp_wsclean_args.append("-parallel-deconvolution 1024")
                elif imsize >= 2048:
                    temp_wsclean_args.append("-parallel-deconvolution 1024")

                ######################################
                # Running imaging
                ######################################
                wsclean_cmd = "wsclean " + " ".join(temp_wsclean_args) + " " + msname
                logger.info(
                    f"{os.path.basename(msname)} -- WSClean command: {wsclean_cmd}\n",
                )
                msg = run_wsclean(wsclean_cmd, "meerwsclean", verbose=False)
                if msg == 0:
                    os.system("rm -rf " + prefix + "*psf.fits")
                    ######################
                    # Making stokes cubes
                    ######################
                    pollist = [i.upper() for i in list(pol)]
                    if len(pollist) == 1:
                        imagelist = sorted(glob.glob(prefix + "*image.fits"))
                        if savemodel == False:
                            os.system("rm -rf " + prefix + "*model.fits")
                        else:
                            modellist = sorted(glob.glob(prefix + "*model.fits"))
                        if saveres == False:
                            os.system("rm -rf " + prefix + "*residual.fits")
                        else:
                            reslist = sorted(glob.glob(prefix + "*residual.fits"))
                    else:
                        imagelist = []
                        stokeslist = []
                        for p in pollist:
                            stokeslist.append(
                                sorted(glob.glob(prefix + "*" + p + "-image.fits"))
                            )
                        for i in range(len(stokeslist[0])):
                            wsclean_images = sorted(
                                [stokeslist[k][i] for k in range(len(pollist))]
                            )
                            image_prefix = os.path.basename(wsclean_images[0]).split(
                                "-image"
                            )[0]
                            image_cube = make_stokes_wsclean_imagecube(
                                wsclean_images,
                                image_prefix + f"_{pol}_image.fits",
                                keep_wsclean_images=False,
                            )
                            imagelist.append(image_cube)
                        del stokeslist
                        if savemodel == False:
                            os.system("rm -rf " + prefix + "*model.fits")
                        else:
                            modellist = []
                            stokeslist = []
                            for p in pollist:
                                stokeslist.append(
                                    sorted(glob.glob(prefix + f"*{p}*model.fits"))
                                )
                            for i in range(len(stokeslist[0])):
                                wsclean_models = sorted(
                                    [stokeslist[k][i] for k in range(len(pollist))]
                                )
                                model_prefix = os.path.basename(
                                    wsclean_models[0]
                                ).split("-model")[0]
                                model_cube = make_stokes_wsclean_imagecube(
                                    wsclean_models,
                                    model_prefix + f"_{pol}_model.fits",
                                    keep_wsclean_images=False,
                                )
                                modellist.append(model_cube)
                            del stokeslist
                        if saveres == False:
                            os.system("rm -rf " + prefix + "*residual.fits")
                        else:
                            reslist = []
                            stokeslist = []
                            for p in pollist:
                                stokeslist.append(
                                    sorted(glob.glob(prefix + f"*{p}*residual.fits"))
                                )
                            for i in range(len(stokeslist[0])):
                                wsclean_residuals = sorted(
                                    [stokeslist[k][i] for k in range(len(pollist))]
                                )
                                res_prefix = os.path.basename(
                                    wsclean_residuals[0]
                                ).split("-residual")[0]
                                residual_cube = make_stokes_wsclean_imagecube(
                                    wsclean_residuals,
                                    res_prefix + f"_{pol}_residual.fits",
                                    keep_wsclean_images=False,
                                )
                                reslist.append(residual_cube)
                            del stokeslist

                    ######################
                    # Renaming images
                    ######################
                    if len(imagelist) > 0:
                        os.makedirs(imagedir + "/images",exist_ok=True)
                        final_image_list = []
                        for imagename in imagelist:
                            renamed_image = rename_image(
                                imagename,
                                imagedir=imagedir + "/images",
                                pol=pol,
                                band=band,
                                cutout_rsun=cutout_rsun,
                                make_overlay=make_overlay,
                                make_plots=make_plots,
                            )
                            final_image_list.append(renamed_image)
                        final_list.append(final_image_list)
                        if savemodel and len(modellist) > 0:
                            final_model_list = []
                            os.makedirs(imagedir + "/models",exist_ok=True)
                            for modelname in modellist:
                                renamed_model = rename_image(
                                    modelname,
                                    imagedir=imagedir + "/models",
                                    pol=pol,
                                    band=band,
                                    cutout_rsun=cutout_rsun,
                                    make_overlay=False,
                                    make_plots=False,
                                )
                                final_model_list.append(renamed_model)
                            final_list.append(final_model_list)
                        if saveres and len(reslist) > 0:
                            final_res_list = []
                            os.makedirs(imagedir + "/residuals",exist_ok=True)
                            for resname in reslist:
                                renamed_res = rename_image(
                                    resname,
                                    imagedir=imagedir + "/residuals",
                                    pol=pol,
                                    band=band,
                                    cutout_rsun=cutout_rsun,
                                    make_overlay=False,
                                    make_plots=False,
                                )
                                final_res_list.append(renamed_res)
                            final_list.append(final_res_list)
            if use_solar_mask and os.path.exists(fits_mask):
                os.system("rm -rf " + fits_mask)
            if len(final_list) == 0 or len(final_list[0]) == 0:
                logger.info(
                    f"{os.path.basename(msname)} -- No image is made.\n",
                )
                time.sleep(5)
                clean_shutdown(sub_observer)
                return 1, final_list
            else:
                logger.info(
                    f"{os.path.basename(msname)} -- Imaging is successfully done.\n",
                )
                time.sleep(5)
                clean_shutdown(sub_observer)
                return 0, final_list
        else:
            if use_solar_mask and os.path.exists(fits_mask):
                os.system("rm -rf " + fits_mask)
            logger.info(
                f"{os.path.basename(msname)} -- No image is made.\n",
            )
            time.sleep(5)
            clean_shutdown(sub_observer)
            return 1, []
    except Exception as e:
        traceback.print_exc()
        time.sleep(5)
        clean_shutdown(sub_observer)
        return 1, []
    finally:
        time.sleep(5)
        drop_cache(msname)


def run_all_imaging(
    mslist=[],
    mainlogger=None,
    workdir="",
    outdir="",
    freqrange="",
    timerange="",
    datacolumn="CORRECTED_DATA",
    freqres=-1,
    timeres=-1,
    weight="briggs",
    robust=0.0,
    minuv=0,
    pol="I",
    threshold=1.0,
    use_multiscale=True,
    use_solar_mask=True,
    imaging_params={}, #TODO
    savemodel=False,
    saveres=False,
    band="",
    cutout_rsun=2.5,
    make_overlay=True,
    make_plots=True,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile="imaging.log",
):
    """
    Run spectropolarimetric snapshot imaging on a list of measurement sets

    Parameters
    ----------
    mslist : list
        Measurement set list
    mainlogger : str
        Python logger
    workdir : str
        Work directory
    outdir : str
        Output directory
    freqrange : str, optional
        Frequency range to image
    timerange : str, optional
        Time range
    datacolumn : str, optional
        Data column
    freqres : float, optional
        Frequency resolution of spectral chunk in MHz
    timeres : float, optional
        Time resolution of temporal chunk in seconds
    weight : str, optional
        Image weighting
    robust : float, optional
        Briggs weighting robust parameter
    minuv : float, optional
        Minimum UV-lambda to use in imaging
    pol : str, optional
        Stokes parameters to image
    threshold : float, optional
        CLEAN threshold
    use_multiscale : bool, optional
        Use multiscale or not
    use_solar_mask : bool, optional
        Use solar mask
    savemodel : bool, optional
        Save model images or not
    saveres : bool, optional
        Save residual images or not
    band : str, optional
        Band name
    cutout_rsun : float, optional
        Cutout image size from center in solar radii (default : 2.5 solar radii)
    make_overlay : bool, optional
        Make SUVI MeerKAT overlay
    make_plots : bool, optional
        Make radio image helioprojective plots
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use

    Returns
    -------
    int
        Success message
    """
    start_time = time.time()
    mslist = sorted(mslist)
    observer = None
    if mainlogger == None:
        mainlog_file = workdir + "/logs/imaging_targets.mainlog"
        mainlogger, mainlog_file = create_logger(
            os.path.basename(mainlog_file).split(".mainlog")[0],
            mainlog_file,
            verbose=False,
        )
        observer = None
        if os.path.exists(f"{workdir}/jobname_password.npy") and mainlog_file != None:
            time.sleep(5)
            jobname, password = np.load(
                f"{workdir}/jobname_password.npy", allow_pickle=True
            )
            if os.path.exists(mainlog_file):
                print(f"Starting remote logger. Remote logger password: {password}")
                observer = init_logger(
                    "all_imaging", mainlog_file, jobname=jobname, password=password
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
    try:
        if len(mslist) == 0:
            mainlogger.error("Provide valid measurement set list.")
            time.sleep(5)
            clean_shutdown(observer)
            return 1
        if weight == "briggs":
            weight_str = f"{weight}_{robust}"
        else:
            weight_str = weight
        if freqres == -1 and timeres == -1:
            imagedir = outdir + f"/imagedir_f_all_t_all_w_{weight_str}"
        elif freqres != -1 and timeres == -1:
            imagedir = outdir + f"/imagedir_f_{freqres}_t_all_w_{weight_str}"
        elif freqres == -1 and timeres != -1:
            imagedir = outdir + f"/imagedir_f_all_t_{timeres}_w_{weight_str}"
        else:
            imagedir = outdir + f"/imagedir_f_{freqres}_t_{timeres}_w_{weight_str}"
        os.makedirs(imagedir, exist_ok=True)

        ####################################
        # Filtering any corrupted ms
        #####################################
        filtered_mslist = []  # Filtering in case any ms is corrupted
        for ms in mslist:
            checkcol = check_datacolumn_valid(ms)
            if checkcol:
                filtered_mslist.append(ms)
            else:
                mainlogger.warning(f"Issue in : {ms}")
                os.system(f"rm -rf {ms}")
        mslist = filtered_mslist

        #####################################
        # Determining spectro-temporal chunks
        #####################################
        if timeres < 0:
            ntime_list = [1] * len(mslist)
        else:
            ntime_list = []
            msmd = msmetadata()
            for ms in mslist:
                msmd.open(ms)
                times = msmd.timesforspws(0)
                msmd.close()
                tw = max(times) - min(times)
                ntime = max(1, int(tw / timeres))
                ntime_list.append(ntime)
        if freqres < 0:
            nchan_list = [1] * len(mslist)
        else:
            nchan_list = []
            msmd = msmetadata()
            for ms in mslist:
                msmd.open(ms)
                freqs = msmd.chanfreqs(0, unit="MHz")
                msmd.close()
                bw = max(freqs) - min(freqs)
                nchan = max(1, math.ceil(bw / freqres))
                nchan_list.append(nchan)

        ######################################
        # Resetting maximum file limit
        ######################################
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft_limit = max(soft_limit, int(0.8*hard_limit))
        if soft_limit < new_soft_limit:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))
        total_fd = 0
        npol = len(pol)
        for i in range(len(mslist)):
            ms = mslist[i]
            nchan = nchan_list[i]
            ntime = ntime_list[i]
            per_job_fd = (
                npol * (nchan + 1) * ntime * 4 * 2
            )  # 4 types of images, 2 is fudge factor
            total_fd+=per_job_fd
        n_jobs = max(1, int(new_soft_limit / total_fd))
        n_jobs = min(len(mslist), n_jobs)
        
        #################################
        # Dask client setup
        #################################
        task = delayed(perform_imaging)(dry_run=True)
        mem_limit = run_limited_memory_task(task, dask_dir=workdir)
        dask_client, dask_cluster, n_jobs, n_threads, mem_limit = get_dask_client(
            n_jobs,
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_cpu_per_job=3,
            min_mem_per_job=mem_limit / 0.6,
        )
        tasks = []
        for i in range(len(mslist)):
            ms = mslist[i]
            nchan = nchan_list[i]
            ntime = ntime_list[i]
            num_pixel_in_psf = calc_npix_in_psf(weight, robust=robust)
            cellsize = calc_cellsize(ms, num_pixel_in_psf)
            instrument_fov = calc_field_of_view(ms, FWHM=False)
            fov = min(instrument_fov, 32 * 3 * 60)  # 3 solar radii
            imsize = int(fov / cellsize)
            pow2 = np.ceil(np.log2(imsize)).astype("int")
            possible_sizes = []
            for p in range(pow2):
                for k in [3, 5]:
                    possible_sizes.append(k * 2**p)
            possible_sizes = np.sort(np.array(possible_sizes))
            possible_sizes = possible_sizes[possible_sizes >= imsize]
            imsize = max(1024, int(possible_sizes[0]))
            os.makedirs(workdir + "/logs",exist_ok=True)
            logfile = (
                workdir
                + "/logs/imaging_"
                + os.path.basename(ms).split(".ms")[0]
                + ".log"
            )
            mainlogger.info(
                f"Starting imaging for ms : {ms}, Log file : {logfile}\n",
            )
            tasks.append(
                delayed(perform_imaging)(
                    msname=ms,
                    workdir=workdir,
                    freqrange=freqrange,
                    timerange=timerange,
                    datacolumn=datacolumn,
                    imagedir=imagedir,
                    imsize=imsize,
                    cellsize=cellsize,
                    nchan=nchan,
                    ntime=ntime,
                    pol=pol,
                    weight=weight,
                    robust=robust,
                    minuv=minuv,
                    threshold=threshold,
                    use_multiscale=use_multiscale,
                    use_solar_mask=use_solar_mask,
                    savemodel=savemodel,
                    saveres=saveres,
                    band=band,
                    cutout_rsun=cutout_rsun,
                    make_overlay=make_overlay,
                    make_plots=make_plots,
                    ncpu=n_threads,
                    mem=mem_limit,
                    logfile=logfile,
                )
            )

        results = compute(*tasks)
        dask_client.close()
        dask_cluster.close()
        all_image_list = []
        all_imaged_ms_list = []
        for i in range(len(results)):
            r = results[i]
            if r[0] != 0:
                mainlogger.info(
                    f"Imaging failed for ms : {mslist[i]}",
                )
            else:
                all_imaged_ms_list.append(mslist[i])
                for image in r[1][0]:
                    all_image_list.append(image)
        mainlogger.info(
            f"Numbers of input measurement sets : {len(mslist)}.",
        )
        mainlogger.info(
            f"Imaging successfully done for: {len(all_imaged_ms_list)} measurement sets.",
        )
        mainlogger.info(f"Total images made: {len(all_image_list)}.")
        mainlogger.info(
            f"Total time taken: {round(time.time()-start_time,2)}s",
        )
        time.sleep(5)
        clean_shutdown(observer)
        return 0
    except Exception as e:
        traceback.print_exc()
        mainlogger.info(
            f"Total time taken: {round(time.time()-start_time,2)}s",
        )
        time.sleep(5)
        clean_shutdown(observer)
        return 1
    finally:
        time.sleep(5)
        for ms in mslist:
            drop_cache(ms)
        drop_cache(workdir)
            
def main():
    parser = argparse.ArgumentParser(
        description="Perform spectropolarimetric snapshot imaging",
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
        required=True,
        default="",
        help="Work directory for imaging",
    )
    basic_args.add_argument(
        "--outdir",
        type=str,
        required=True,
        default="",
        help="Output directory for imaging products",
    )

    ## Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced imaging parameters\n###################"
    )
    adv_args.add_argument(
        "--freqrange",
        type=str,
        default="",
        help="Frequency range to image",
    )
    adv_args.add_argument(
        "--timerange",
        type=str,
        default="",
        help="Time range to image",
    )
    adv_args.add_argument(
        "--datacolumn",
        type=str,
        default="CORRECTED_DATA",
        help="Data column to use for imaging",
    )
    adv_args.add_argument(
        "--pol",
        type=str,
        default="I",
        help="Stokes parameters to image",
    )
    adv_args.add_argument(
        "--freqres",
        type=float,
        default=-1,
        help="Frequency resolution per chunk in MHz (-1 for full)",
    )
    adv_args.add_argument(
        "--timeres",
        type=float,
        default=-1,
        help="Time resolution per chunk in seconds (-1 for full)",
    )
    adv_args.add_argument(
        "--weight",
        type=str,
        default="briggs",
        help="Imaging weighting scheme",
    )
    adv_args.add_argument(
        "--robust",
        type=float,
        default=0.0,
        help="Briggs robust parameter",
    )
    adv_args.add_argument(
        "--minuv_l",
        dest="minuv",
        type=float,
        default=0.0,
        help="Minimum UV distance in wavelengths",
    )
    adv_args.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="CLEAN threshold in Jy",
    )
    adv_args.add_argument(
        "--band",
        type=str,
        default="",
        help="Band name label for output",
    )
    adv_args.add_argument(
        "--cutout_rsun",
        type=float,
        default=2.5,
        help="Cutout radius for images (solar radii)",
    )
    adv_args.add_argument(
        "--no_multiscale",
        action="store_false",
        dest="use_multiscale",
        help="Do not use multiscale CLEAN",
    )
    adv_args.add_argument(
        "--no_solar_mask",
        action="store_false",
        dest="use_solar_mask",
        help="Do not use solar disk mask for CLEANing",
    )
    adv_args.add_argument(
        "--no_savemodel",
        action="store_false",
        dest="savemodel",
        help="Do no save model images",
    )
    adv_args.add_argument(
        "--no_saveres",
        action="store_false",
        dest="saveres",
        help="Do not save residual images",
    )
    adv_args.add_argument(
        "--no_make_overlay",
        action="store_false",
        dest="make_overlay",
        help="Do not generate overlay with SUVI images",
    )
    adv_args.add_argument(
        "--no_make_plots",
        action="store_false",
        dest="make_plots",
        help="Do not make generate helioprojective plots",
    )

    ## Resource management parameters
    hard_args = parser.add_argument_group(
        "###################\nHardware resource management parameters\n###################"
    )
    hard_args.add_argument(
        "--cpu_frac",
        type=float,
        default=0.8,
        help="Fraction of available CPU to use",
    )
    hard_args.add_argument(
        "--mem_frac",
        type=float,
        default=0.8,
        help="Fraction of available memory to use",
    )
    hard_args.add_argument(
        "--jobid",
        type=str,
        default="0",
        help="Job ID for process tracking and logging",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    pid = os.getpid()
    save_pid(pid, datadir + f"/pids/pids_{args.jobid}.txt")

    if args.workdir == "" or not os.path.exists(args.workdir):
        first_ms = args.mslist.split(",")[0]
        workdir = os.path.dirname(os.path.abspath(first_ms)) + "/workdir"     
    else:
        workdir = args.workdir
    os.makedirs(workdir,exist_ok=True)
        
    if args.outdir == "" or not os.path.exists(args.outdir):
        outdir=workdir
    else:
        outdir = args.outdir
    os.makedirs(outdir,exist_ok=True)

    os.makedirs(workdir + "/logs/",exist_ok=True)

    mainlog_file = workdir + "/logs/imaging_targets.mainlog"
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
                "all_imaging", mainlog_file, jobname=jobname, password=password
            )
    mslist = args.mslist.split(",")
    try:
        if len(mslist) == 0:
            mainlogger.info("Please provide a valid measurement set list.")
            msg = 1
        else:
            msg = run_all_imaging(
                mslist=mslist,
                mainlogger=mainlogger,
                workdir=workdir,
                outdir=outdir,
                freqrange=args.freqrange,
                timerange=args.timerange,
                datacolumn=args.datacolumn,
                freqres=args.freqres,
                timeres=args.timeres,
                weight=args.weight,
                robust=args.robust,
                minuv=args.minuv,
                threshold=args.threshold,
                use_multiscale=args.use_multiscale,
                use_solar_mask=args.use_solar_mask,
                pol=args.pol,
                band=args.band,
                make_plots=args.make_plots,
                cutout_rsun=args.cutout_rsun,
                make_overlay=args.make_overlay,
                savemodel=args.savemodel,
                saveres=args.saveres,
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
        drop_cache(workdir)
        drop_cache(outdir)
        clean_shutdown(observer)

    return msg


if __name__ == "__main__":
    result = main()
    if result > 0:
        result = 1
    print("\n###################\nImaging is done.\n###################\n")
    os._exit(result)
