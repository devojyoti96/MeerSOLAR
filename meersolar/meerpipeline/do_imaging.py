import resource
import logging
import psutil
import dask
import numpy as np
import argparse
import traceback
import copy
import time
import glob
import sys
import os
from casatasks import casalog
from casatools import msmetadata
from dask import delayed
from dask.distributed import Client
from meersolar.utils import *

logging.getLogger("distributed").setLevel(logging.WARNING)

try:
    logfile = casalog.logfile()
    os.system("rm -rf " + logfile)
except BaseException:
    pass

datadir = get_datadir()


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
        os.path.basename(logfile).split(".log")[0],
        logfile,
        verbose=False,
        get_print=True,
    )
    sub_observer = None
    if os.path.exists(f"{workdir}/jobname_password.npy") and logfile is not None:
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
    if os.path.exists(f"{workdir}/.attcal"):
        attcal = True
    elif os.path.exists(f"{workdir}/.noattcal"):
        attcal = False
    else:
        attcal = "NOINFO"
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
        os.makedirs(imagedir, exist_ok=True)
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
            wsclean_args.append(f"-channels-out {int(nchan)}")
            wsclean_args.append("-no-mf-weighting")
            wsclean_args.append("-join-channels")

        #####################################
        # Temporal imaging configuration
        #####################################
        if ntime > 1:
            wsclean_args.append(f"-intervals-out {int(ntime)}")

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
            if fits_mask is not None and os.path.exists(fits_mask):
                wsclean_args.append("-fits-mask " + fits_mask)
        final_list_dic = {"image": [], "model": [], "residual": []}
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
                    chan_number = int((start_chans[i] + end_chans[i]) / 2)
                    freqMHz = freqs[chan_number]
                    sun_dia = calc_sun_dia(freqMHz)  # Sun diameter in arcmin
                    sun_rad = sun_dia / 2
                    multiscale_scales = calc_multiscale_scales(
                        msname,
                        num_pixel_in_psf,
                        chan_number=chan_number,
                        max_scale=sun_rad,
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
                msg = run_wsclean(wsclean_cmd, "solarwsclean", verbose=False)
                if msg == 0:
                    os.system("rm -rf " + prefix + "*psf.fits")
                    ######################
                    # Making stokes cubes
                    ######################
                    pollist = [i.upper() for i in list(pol)]
                    if len(pollist) == 1:
                        imagelist = sorted(glob.glob(prefix + "*image.fits"))
                        if not savemodel:
                            os.system("rm -rf " + prefix + "*model.fits")
                        else:
                            modellist = sorted(glob.glob(prefix + "*model.fits"))
                        if not saveres:
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
                        if not savemodel:
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
                        if not saveres:
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
                        logger.info(f"Total {len(imagelist)} images are made.")
                        logger.info("Renaming and making plots...")
                        os.makedirs(imagedir + "/images", exist_ok=True)
                        final_image_list = []
                        for imagename in imagelist:
                            renamed_image = rename_meersolar_image(
                                imagename,
                                imagedir=imagedir + "/images",
                                pol=pol,
                                band=band,
                                attcal=attcal,
                                cutout_rsun=cutout_rsun,
                                make_overlay=make_overlay,
                                make_plots=make_plots,
                            )
                            final_image_list.append(renamed_image)
                        final_list_dic["image"] = final_image_list
                        if savemodel and len(modellist) > 0:
                            final_model_list = []
                            os.makedirs(imagedir + "/models", exist_ok=True)
                            for modelname in modellist:
                                renamed_model = rename_meersolar_image(
                                    modelname,
                                    imagedir=imagedir + "/models",
                                    pol=pol,
                                    band=band,
                                    attcal=attcal,
                                    cutout_rsun=cutout_rsun,
                                    make_overlay=False,
                                    make_plots=False,
                                )
                                final_model_list.append(renamed_model)
                            final_list_dic["model"] = final_model_list
                        if saveres and len(reslist) > 0:
                            final_res_list = []
                            os.makedirs(imagedir + "/residuals", exist_ok=True)
                            for resname in reslist:
                                renamed_res = rename_meersolar_image(
                                    resname,
                                    imagedir=imagedir + "/residuals",
                                    pol=pol,
                                    band=band,
                                    attcal=attcal,
                                    cutout_rsun=cutout_rsun,
                                    make_overlay=False,
                                    make_plots=False,
                                )
                                final_res_list.append(renamed_res)
                            final_list_dic["residual"] = final_res_list
            if os.path.exists(f"{imagedir}/images/dask-scratch-space"):
                os.system(f"rm -rf {imagedir}/images/dask-scratch-space")
            if use_solar_mask and os.path.exists(fits_mask):
                os.system("rm -rf " + fits_mask)
            if len(final_list_dic["image"]) == 0:
                logger.info(
                    f"{os.path.basename(msname)} -- No image is made.\n",
                )
                time.sleep(5)
                clean_shutdown(sub_observer)
                return 1, final_list_dic
            else:
                logger.info(
                    f"{os.path.basename(msname)} -- Imaging is successfully done.\n",
                )
                time.sleep(5)
                clean_shutdown(sub_observer)
                return 0, final_list_dic
        else:
            if use_solar_mask and os.path.exists(fits_mask):
                os.system("rm -rf " + fits_mask)
            logger.info(
                f"{os.path.basename(msname)} -- No image is made.\n",
            )
            time.sleep(5)
            clean_shutdown(sub_observer)
            return 1, {}
    except Exception as e:
        traceback.print_exc()
        logger.info(
            f"{os.path.basename(msname)} -- Error in imaging.\n",
        )
        time.sleep(5)
        clean_shutdown(sub_observer)
        return 1, {}
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
    imaging_params={},  # TODO
    savemodel=False,
    saveres=False,
    band="",
    cutout_rsun=-1,
    make_overlay=True,
    make_plots=True,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile="imaging.log",
    dask_addr=None,
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
        Cutout image size (width ans height is : 2 times cutout_rsun)
        Default value: 2 solar radii
        Note: default FoV is 2 solar solar radii. If cutout_rsun is chosen larger than 2 solar radii, FoV will be increased accordingly.
    make_overlay : bool, optional
        Make SUVI MeerKAT overlay
    make_plots : bool, optional
        Make radio image helioprojective plots
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
    """
    start_time = time.time()
    mslist = sorted(mslist)
    observer = None
    if mainlogger is None:
        mainlog_file = workdir + "/logs/imaging_targets.log"
        mainlogger, mainlog_file = create_logger(
            os.path.basename(mainlog_file).split(".log")[0],
            mainlog_file,
            verbose=False,
            get_print=True,
        )
        observer = None
        if (
            os.path.exists(f"{workdir}/jobname_password.npy")
            and mainlog_file is not None
        ):
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
    container_name = "solarwsclean"
    container_present = check_udocker_container(container_name)
    if not container_present:
        container_name = initialize_wsclean_container(name=container_name)
        if container_name is None:
            print(
                f"Container {container_name} is not initiated. First initiate container and then run."
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
                nchan = max(1, int(np.ceil(bw / freqres)))
                nchan_list.append(nchan)

        ######################################
        # Resetting maximum file limit
        ######################################
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft_limit = max(soft_limit, int(0.8 * hard_limit))
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
            total_fd += per_job_fd
        if total_fd <= 0:
            total_fd = 1
        n_jobs = max(1, int(new_soft_limit / total_fd))
        n_jobs = min(len(mslist), n_jobs)

        #################################
        # Dask client setup
        #################################
        mem_limit = perform_imaging(dry_run=True)
        if dask_addr is None:
            dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = (
                get_dask_client(
                    n_jobs,
                    dask_dir=workdir,
                    cpu_frac=cpu_frac,
                    mem_frac=mem_frac,
                    min_cpu_per_job=3,
                    min_mem_per_job=mem_limit,
                )
            )
        else:
            _, _, n_jobs, n_threads, mem_limit, dask_dir = (
                get_dask_client(
                    n_jobs,
                    dask_dir=workdir,
                    cpu_frac=cpu_frac,
                    mem_frac=mem_frac,
                    min_cpu_per_job=3,
                    min_mem_per_job=mem_limit,
                    only_cal=True,
                )
            )
            dask_client=Client(address=dask_addr)
            os.system(f"rm -rf {dask_dir}")
        wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
        tasks = []
        for i in range(len(mslist)):
            ms = mslist[i]
            nchan = nchan_list[i]
            ntime = ntime_list[i]
            num_pixel_in_psf = calc_npix_in_psf(weight, robust=robust)
            cellsize = calc_cellsize(ms, num_pixel_in_psf)
            instrument_fov = calc_field_of_view(ms, FWHM=False)
            msmd = msmetadata()
            msmd.open(ms)
            freqMHz = msmd.meanfreq(0, unit="MHz")
            msmd.close()
            sun_size = calc_sun_dia(freqMHz)
            fov = min(
                instrument_fov, 2 * sun_size * 60
            )  # 3 times sun size at that frequency
            if cutout_rsun == -1:
                cutout_rsun = 2 * round(
                    sun_size / 32, 2
                )  # Multiple of optical disk of the sun
            if fov < (2 * (cutout_rsun * 16) * 60):
                fov = 2 * (cutout_rsun * 16) * 60
            imsize = int(fov / cellsize)
            pow2 = np.ceil(np.log2(imsize)).astype("int")
            possible_sizes = []
            for p in range(pow2):
                for k in [3, 5]:
                    possible_sizes.append(k * 2**p)
            possible_sizes = np.sort(np.array(possible_sizes))
            possible_sizes = possible_sizes[possible_sizes >= imsize]
            imsize = max(1024, int(possible_sizes[0]))
            os.makedirs(workdir + "/logs", exist_ok=True)
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
        futures = dask_client.compute(tasks)
        results = list(dask_client.gather(futures))
        dask_client.close()
        if dask_addr is None:
            dask_cluster.close()
            os.system(f"rm -rf {dask_dir}")
        all_image_list = []
        all_imaged_ms_list = []
        for i in range(len(results)):
            r = results[i][1]
            if len(r) == 0:
                mainlogger.info(
                    f"Imaging failed for ms : {mslist[i]}",
                )
            else:
                image_list = r["image"]
                if len(image_list) == 0:
                    mainlogger.info(
                        f"No image is made for ms : {mslist[i]}",
                    )
                else:
                    all_imaged_ms_list.append(mslist[i])
                    for image in image_list:
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


def main(
    mslist,
    workdir,
    outdir,
    freqrange="",
    timerange="",
    datacolumn="CORRECTED_DATA",
    pol="I",
    freqres=-1,
    timeres=-1,
    weight="briggs",
    robust=0.0,
    minuv=0.0,
    threshold=1.0,
    band="",
    cutout_rsun=-1,
    use_multiscale=True,
    use_solar_mask=True,
    savemodel=True,
    saveres=True,
    make_overlay=True,
    make_plots=True,
    start_remote_log=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    jobid=0,
    dask_addr=None,
):
    """
    Perform distributed spectropolarimetric snapshot imaging on multiple measurement sets.

    Parameters
    ----------
    mslist : str
        Comma-separated list of measurement set paths to be imaged.
    workdir : str
        Directory for intermediate files, logs, and temporary outputs.
    outdir : str
        Directory where final images, models, and plots will be saved.
    freqrange : str, optional
        Frequency range to image (e.g., "500~1000MHz"). Default is "" (all frequencies).
    timerange : str, optional
        Time range to image (e.g., "09:30:00~09:40:00"). Default is "" (all times).
    datacolumn : str, optional
        Data column to image (e.g., "DATA", "CORRECTED_DATA"). Default is "CORRECTED_DATA".
    pol : str, optional
        Polarization product to image (e.g., "I", "XX", "RR", "QUV"). Default is "I".
    freqres : float, optional
        Frequency resolution in MHz for slicing the MS. Use -1 to disable. Default is -1.
    timeres : float, optional
        Time resolution in seconds for snapshot imaging. Use -1 to disable. Default is -1.
    weight : str, optional
        Weighting scheme for imaging ("natural", "uniform", "briggs"). Default is "briggs".
    robust : float, optional
        Robustness parameter for Briggs weighting. Default is 0.0.
    minuv : float, optional
        Minimum uv-distance (in wavelengths) to include in imaging. Default is 0.0.
    threshold : float, optional
        Cleaning threshold in Jy. Default is 1.0.
    band : str, optional
        Optional label for the frequency band used in filenames and plots. Default is "".
    cutout_rsun : float, optional
        Radius in solar radii to cut out around solar center. Set to -1 to disable. Default is -1.
    use_multiscale : bool, optional
        If True, enables multiscale CLEAN deconvolution. Default is True.
    use_solar_mask : bool, optional
        If True, applies a solar disk mask during CLEAN to reduce sidelobe artifacts. Default is True.
    savemodel : bool, optional
        If True, saves the CLEAN model images. Default is True.
    saveres : bool, optional
        If True, saves the residual images. Default is True.
    make_overlay : bool, optional
        If True, generates image overlays on solar maps. Default is True.
    make_plots : bool, optional
        If True, generates diagnostic plots for each image. Default is True.
    start_remote_log : bool, optional
        Whether to enable remote logging using credentials in the workdir. Default is False.
    cpu_frac : float, optional
        Fraction of total CPUs to use per task. Default is 0.8.
    mem_frac : float, optional
        Fraction of total system memory to use per task. Default is 0.8.
    jobid : int, optional
        Unique job identifier for logging and PID tracking. Default is 0.
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
    mslist = mslist.split(",")

    if workdir == "":
        first_ms = mslist[0]
        workdir = os.path.dirname(os.path.abspath(first_ms)) + "/workdir"
    os.makedirs(workdir, exist_ok=True)

    if outdir == "" or not os.path.exists(outdir):
        outdir = workdir
    os.makedirs(outdir, exist_ok=True)

    os.makedirs(workdir + "/logs/", exist_ok=True)

    mainlog_file = workdir + "/logs/imaging_targets.log"
    mainlogger, mainlog_file = create_logger(
        os.path.basename(mainlog_file).split(".log")[0],
        mainlog_file,
        get_print=True,
        verbose=False,
    )

    ############
    # Logger
    ############
    observer = None
    if (
        start_remote_log
        and os.path.exists(f"{workdir}/jobname_password.npy")
        and mainlog_file is not None
    ):
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(mainlog_file):
            observer = init_logger(
                "all_imaging", mainlog_file, jobname=jobname, password=password
            )
    if observer == None:
        print("Remote link or jobname is blank. Not transmiting to remote logger.")
    ##########

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
                freqrange=freqrange,
                timerange=timerange,
                datacolumn=datacolumn,
                freqres=freqres,
                timeres=timeres,
                weight=weight,
                robust=robust,
                minuv=minuv,
                threshold=threshold,
                use_multiscale=use_multiscale,
                use_solar_mask=use_solar_mask,
                pol=pol,
                band=band,
                make_plots=make_plots,
                cutout_rsun=cutout_rsun,
                make_overlay=make_overlay,
                savemodel=savemodel,
                saveres=saveres,
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
        drop_cache(outdir)
        clean_shutdown(observer)
    return msg


def cli():
    parser = argparse.ArgumentParser(
        description="Perform spectropolarimetric snapshot imaging",
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

    # Advanced parameters
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
        default=-1,
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
    adv_args.add_argument(
        "--start_remote_log", action="store_true", help="Start remote logging"
    )

    # Resource management parameters
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

    msg = main(
        mslist=args.mslist,
        workdir=args.workdir,
        outdir=args.outdir,
        freqrange=args.freqrange,
        timerange=args.timerange,
        datacolumn=args.datacolumn,
        pol=args.pol,
        freqres=args.freqres,
        timeres=args.timeres,
        weight=args.weight,
        robust=args.robust,
        minuv=args.minuv,
        threshold=args.threshold,
        band=args.band,
        cutout_rsun=args.cutout_rsun,
        use_multiscale=args.use_multiscale,
        use_solar_mask=args.use_solar_mask,
        savemodel=args.savemodel,
        saveres=args.saveres,
        make_overlay=args.make_overlay,
        make_plots=args.make_plots,
        start_remote_log=args.start_remote_log,
        cpu_frac=float(args.cpu_frac),
        mem_frac=float(args.mem_frac),
        jobid=args.jobid,
    )
    return msg


if __name__ == "__main__":
    result = cli()
    if result > 0:
        result = 1
    print("\n###################\nImaging is done.\n###################\n")
    os._exit(result)
