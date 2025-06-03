import os, glob, resource, traceback, psutil, time, copy, scipy, math
from astropy.io import fits
from casatools import msmetadata
from meersolar.pipeline.basic_func import *
from optparse import OptionParser
from dask import delayed, compute
from casatasks import casalog

logfile = casalog.logfile()
os.system("rm -rf " + logfile)


def rename_image(imagename, imagedir="", pol="", band=""):
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
    Returns
    -------
    str
        New imagename with full path
    """
    imagename = imagename.rstrip("/")
    with fits.open(imagename, mode="update") as hdul:
        hdr = hdul[0].header
        hdr["AUTHOR"] = "DevojyotiKansabanik,DeepanPatra"
        if band != "":
            hdr["BAND"] = band
    header = fits.getheader(imagename)
    time = header["DATE-OBS"]
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
    if imagedir != "":
        new_name = imagedir + "/" + new_name
    os.system("mv " + imagename + " " + new_name)
    return new_name


def perform_imaging(
    msname="",
    workdir="",
    imagedir="",
    imsize=1024,
    cellsize=2,
    nchan=-1,
    ntime=-1,
    pol="I",
    weight="briggs",
    robust=0.0,
    multiscale_scales=[],
    minuv=0,
    threshold=1.0,
    use_solar_mask=True,
    mask_radius=20,
    savemodel=True,
    saveres=True,
    ncpu=-1,
    mem=-1,
    band="",
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
    multiscale_scales : list, optional
        Multiscale scales in pixel units
    minuv : float, optional
        Minimum UV-lambda to be used in imaging
    threshold : float, optional
        CLEAN threshold
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
    logger, logfile = init_logger_console(
        os.path.basename(logfile).split(".log")[0], logfile, verbose=False
    )
    if dry_run:
        process = psutil.Process(os.getpid())
        usemem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return usemem
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
        npol = msmd.ncorrforpol()[0]
        msmd.close()
        if npol < 4 and pol == "IQUV":
            pol = "I"
        if ncpu < 1:
            ncpu = psutil.cpu_count()
        if mem < 0:
            mem = psutil.virtual_memory().total / (1024**3)
        prefix = workdir + "/imaging_" + os.path.basename(msname).split(".ms")[0]
        if imagedir == "":
            imagedir = workdir
        if os.path.exists(imagedir) == False:
            os.makedirs(imagedir)
        os.system("rm -rf " + prefix + "*")
        if weight == "briggs":
            weight += " " + str(robust)
        if threshold <= 1:
            threshold = 1.1

        wsclean_args = [
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
        
        ngrid=int(ncpu/2)
        if ngrid>1:
            wsclean_args.append("-parallel-gridding "+str(ngrid))
            
        if pol == "I":
            wsclean_args.append("-no-negative")

        ######################################
        # Multiscale configuration
        ######################################
        if len(multiscale_scales) > 0:
            wsclean_args.append("-multiscale")
            wsclean_args.append("-multiscale-gain 0.1")
            wsclean_args.append(
                "-multiscale-scales " + ",".join([str(s) for s in multiscale_scales])
            )
            scale_bias = get_multiscale_bias(freq)
            wsclean_args.append(f"-multiscale-scale-bias {scale_bias}")

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

        ######################################
        # Resetting maximum file limit
        ######################################
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))

        ######################################
        # Running imaging
        ######################################
        wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " " + msname
        logger.info(
            f"{os.path.basename(msname)} -- WSClean command: {wsclean_cmd}\n",
        )
        msg = run_wsclean(wsclean_cmd, "meerwsclean", verbose=False)
        if msg != 0:
            gc.collect()
            logger.info(
                f"{os.path.basename(msname)} -- Imaging is not successful.\n",
            )
            return 1, []

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
                stokeslist.append(sorted(glob.glob(prefix + "*" + p + "-image.fits")))
            for i in range(len(stokeslist[0])):
                wsclean_images = sorted([stokeslist[k][i] for k in range(len(pollist))])
                image_prefix = os.path.basename(wsclean_images[0]).split("-image")[0]
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
                    stokeslist.append(sorted(glob.glob(prefix + f"*{p}*model.fits")))
                for i in range(len(stokeslist[0])):
                    wsclean_models = sorted(
                        [stokeslist[k][i] for k in range(len(pollist))]
                    )
                    model_prefix = os.path.basename(wsclean_models[0]).split("-model")[
                        0
                    ]
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
                    stokeslist.append(sorted(glob.glob(prefix + f"*{p}*residual.fits")))
                for i in range(len(stokeslist[0])):
                    wsclean_residuals = sorted(
                        [stokeslist[k][i] for k in range(len(pollist))]
                    )
                    res_prefix = os.path.basename(wsclean_residuals[0]).split(
                        "-residual"
                    )[0]
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
            if os.path.exists(imagedir + "/images") == False:
                os.makedirs(imagedir + "/images")
            final_image_list = []
            for imagename in imagelist:
                renamed_image = rename_image(
                    imagename,
                    imagedir=imagedir + "/images",
                    pol=pol,
                    band=band,
                )
                final_image_list.append(renamed_image)
            final_list = [final_image_list]
            if savemodel and len(modellist) > 0:
                final_model_list = []
                if os.path.exists(imagedir + "/models") == False:
                    os.makedirs(imagedir + "/models")
                for modelname in modellist:
                    renamed_model = rename_image(
                        modelname,
                        imagedir=imagedir + "/models",
                        pol=pol,
                        band=band,
                    )
                    final_model_list.append(renamed_model)
                final_list.append(final_model_list)
            if saveres and len(reslist) > 0:
                final_res_list = []
                if os.path.exists(imagedir + "/residuals") == False:
                    os.makedirs(imagedir + "/residuals")
                for resname in reslist:
                    renamed_res = rename_image(
                        resname,
                        imagedir=imagedir + "/residuals",
                        pol=pol,
                        band=band,
                    )
                    final_res_list.append(renamed_res)
                final_list.append(final_res_list)
            if use_solar_mask and os.path.exists(fits_mask):
                os.system("rm -rf " + fits_mask)
            logger.info(
                f"{os.path.basename(msname)} -- Imaging is successfully done.\n",
            )
            return 0, final_list
        else:
            if use_solar_mask and os.path.exists(fits_mask):
                os.system("rm -rf " + fits_mask)
            logger.info(
                f"{os.path.basename(msname)} -- No image is made.\n",
            )
            return 1, []
    except Exception as e:
        traceback.print_exc()
        return 1, []


def run_all_imaging(
    mslist="",
    mainlog_file="",
    workdir="",
    freqres=-1,
    timeres=-1,
    weight="briggs",
    robust=0.0,
    minuv=0,
    pol="I",
    multiscale_scales=[],
    threshold=1.0,
    use_solar_mask=True,
    savemodel=False,
    saveres=False,
    band="",
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
    mainlog_file : str
        Main log file name
    workdir : str
        Work directory
    freqres : float, optional
        Frequency resolution of spectral chunk in MHz
    timeres : float, optional
        Time resolution of temporal chunk in MHz
    weight : str, optional
        Image weighting
    robust : float, optional
        Briggs weighting robust parameter
    minuv : float, optional
        Minimum UV-lambda to use in imaging
    pol : str, optional
        Stokes parameters to image
    multiscale_scales : list
        Multiscale scales in pixel unit (if not given, calculate automatically)
    threshold : float, optional
        CLEAN threshold
    use_solar_mask : bool, optional
        Use solar mask
    savemodel : bool, optional
        Save model images or not
    saveres : bool, optional
        Save residual images or not
    band : str, optional
        Band name
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
    if mainlog_file != "":
        mainlogger, mainlog_file = init_logger_console(
            os.path.basename(mainlog_file).split(".mainlog")[0],
            mainlog_file,
            verbose=False,
        )
    try:
        if len(mslist) == 0:
            mainlogger.error("Provide valid measurement set list.")
            return 1
        if weight == "briggs":
            weight_str = f"{weight}_{robust}"
        else:
            weight_str = weight
        if freqres == -1 and timeres == -1:
            imagedir = workdir + f"/imagedir_f_all_t_all_w_{weight_str}"
        elif freqres != -1 and timeres == -1:
            imagedir = workdir + f"/imagedir_f_{freqres}_t_all_w_{weight_str}"
        elif freqres == -1 and timeres != -1:
            imagedir = workdir + f"/imagedir_f_all_t_{timeres}_w_{weight_str}"
        else:
            imagedir = workdir + f"/imagedir_f_{freqres}_t_{timeres}_w_{weight_str}"
        os.makedirs(imagedir, exist_ok=True)
        os.system(f"rm -rf {imagedir}/*")

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
            ntime_list = [-1] * len(mslist)
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
            nchan_list = [-1] * len(mslist)
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

        #################################
        # Dask client setup
        #################################
        task = delayed(perform_imaging)(dry_run=True)
        mem_limit = run_limited_memory_task(task, dask_dir=workdir)
        dask_client, dask_cluster, n_jobs, n_threads, mem_limit = get_dask_client(
            len(mslist),
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_cpu_per_job=3,
            min_mem_per_job=mem_limit / 0.6,
        )
        mainlogger.info("\n#################################")
        mainlogger.info(
            f"Dask Dashboard: {dask_client.dashboard_link}",
        )
        mainlogger.info("\n#################################")
        tasks = []
        org_multiscale_scales = copy.deepcopy(multiscale_scales)
        for i in range(len(mslist)):
            ms = mslist[i]
            nchan = nchan_list[i]
            ntime = ntime_list[i]
            cellsize = calc_cellsize(ms, 5)
            fov = 32 * 3 * 60  # 3 solar radii
            imsize = int(fov / cellsize)
            pow2 = np.ceil(np.log2(imsize)).astype("int")
            possible_sizes = []
            for p in range(pow2):
                for k in [3, 5]:
                    possible_sizes.append(k * 2**p)
            possible_sizes = np.sort(np.array(possible_sizes))
            possible_sizes = possible_sizes[possible_sizes >= imsize]
            imsize = int(possible_sizes[0])
            if len(org_multiscale_scales) == 0:
                multiscale_scales = calc_multiscale_scales(ms, 5)
            else:
                multiscale_scales = copy.deepcopy(org_multiscale_scales)
            if os.path.exists(workdir + "/logs") == False:
                os.makedirs(workdir + "/logs")
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
                    imagedir=imagedir,
                    imsize=imsize,
                    cellsize=cellsize,
                    nchan=nchan,
                    ntime=ntime,
                    pol=pol,
                    weight=weight,
                    robust=robust,
                    multiscale_scales=multiscale_scales,
                    minuv=minuv,
                    threshold=threshold,
                    use_solar_mask=use_solar_mask,
                    savemodel=savemodel,
                    saveres=saveres,
                    band=band,
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
        return 0
    except Exception as e:
        traceback.print_exc()
        mainlogger.info(
            f"Total time taken: {round(time.time()-start_time,2)}s",
        )
        return 1


def main():
    usage = "Perform spectropolarimetric snapshot imaging"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--mslist",
        dest="mslist",
        default=None,
        help="Measurement set list",
        metavar="List",
    )
    parser.add_option(
        "--workdir",
        dest="workdir",
        default=None,
        help="Work directory",
        metavar="String",
    )
    parser.add_option(
        "--pol",
        dest="pol",
        default="I",
        help="Stokes parameters",
        metavar="String",
    )
    parser.add_option(
        "--freqres",
        dest="freqres",
        default=-1,
        help="Frequency resolution of spectral chunk in MHz",
        metavar="Float",
    )
    parser.add_option(
        "--timeres",
        dest="timeres",
        default=-1,
        help="Time resolution of temporal chunk in seconds",
        metavar="Float",
    )
    parser.add_option(
        "--weight",
        dest="weight",
        default="briggs",
        help="Image weighting",
        metavar="String",
    )
    parser.add_option(
        "--robust",
        dest="robust",
        default=0.0,
        help="Robust parameter for briggs weighting",
        metavar="Float",
    )
    parser.add_option(
        "--multiscale_scales",
        dest="multiscale_scales",
        default="",
        help="Multiscale scales",
        metavar="String",
    )
    parser.add_option(
        "--minuv_l",
        dest="minuv",
        default=0.0,
        help="Minimum UV in lambda",
        metavar="Float",
    )
    parser.add_option(
        "--threshold",
        dest="threshold",
        default=1.0,
        help="CLEAN threshold",
        metavar="Float",
    )
    parser.add_option(
        "--use_solar_mask",
        dest="use_solar_mask",
        default=True,
        help="Use solar mask or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--savemodel",
        dest="savemodel",
        default=False,
        help="Save model images or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--saveres",
        dest="saveres",
        default=False,
        help="Save residual images or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--band",
        dest="band",
        default="",
        help="Band name",
        metavar="String",
    )
    parser.add_option(
        "--cpu_frac",
        dest="cpu_frac",
        default=0.8,
        help="CPU fraction to use",
        metavar="Float",
    )
    parser.add_option(
        "--mem_frac",
        dest="mem_frac",
        default=0.8,
        help="Memory fraction to use",
        metavar="Float",
    )
    (options, args) = parser.parse_args()
    if options.mslist == None:
        print("Please provide correct measurement set list.")
        return 1
    elif options.workdir == None or os.path.exists(options.workdir) == False:
        print("Please provide correct workdiring directory.")
        return 1
    else:
        if os.path.exists(options.workdir + "/logs/") == False:
            os.makedirs(options.workdir + "/logs/")
        mainlog_file = options.workdir + "/logs/imaging_targets.mainlog"
        mslist = str(options.mslist).split(",")
        if len(mslist) == 0:
            print("Please provide correct measurement set list.")
            return 1
        if options.multiscale_scales != "":
            multiscale_scales = options.multiscale_scales.split(",")
        else:
            multiscale_scales = []
        msg = run_all_imaging(
            mslist=mslist,
            mainlog_file=mainlog_file,
            workdir=options.workdir,
            freqres=float(options.freqres),
            timeres=float(options.timeres),
            weight=options.weight,
            robust=float(options.robust),
            minuv=float(options.minuv),
            multiscale_scales=multiscale_scales,
            threshold=float(options.threshold),
            use_solar_mask=eval(str(options.use_solar_mask)),
            pol=options.pol,
            band=options.band,
            savemodel=eval(str(options.savemodel)),
            saveres=eval(str(options.saveres)),
            cpu_frac=float(options.cpu_frac),
            mem_frac=float(options.mem_frac),
        )
        return msg


if __name__ == "__main__":
    result = main()
    if result > 0:
        result = 1
    print("\n###################\nImaging is done.\n###################\n")
    os._exit(result)
