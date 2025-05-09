import os, glob, resource, traceback, psutil, time
from astropy.io import fits
from casatools import msmetadata
from meersolar.pipeline.basic_func import *
from optparse import OptionParser
from dask import delayed, compute
from casatasks import casalog

logfile = casalog.logfile()
os.system("rm -rf " + logfile)


def rename_image(imagename, imagedir="", pol=""):
    """
    Rename and move image to image directory
    Parameters
    ----------
    imagename : str
        Image name
    imagedir : str, optional
        Image directory (default given image directory)
    Returns
    -------
    str
        New imagename with full path
    """
    imagename = imagename.rstrip("/")
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
    minuv=50,
    threshold=1.0,
    use_solar_mask=True,
    savemodel=True,
    saveres=True,
    ncpu=-1,
    mem=-1,
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
    savemodel : bool, optional
        Save model images or not
    saveres : bool, optional
        Save residual images or not
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
    logf=open(logfile,"a")
    if dry_run:
        process = psutil.Process(os.getpid())
        usemem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return usemem
    try:
        msname = msname.rstrip("/")
        msname = os.path.abspath(msname)
        print(f"{os.path.basename(msname)} --Perform imaging...\n",file=logf,flush=True)
        #########
        # Imaging
        #########
        msmd = msmetadata()
        msmd.open(msname)
        npol = msmd.ncorrforpol()[0]
        msmd.close()
        if npol < 4 and pol == "IQUV":
            pol = "I"
        if ncpu < 1:
            ncpu = psutil.cpu_count()
        if mem < 0:
            mem = psutil.virtual_memory().total / (1024**3)
        ngrid = max(1, int(ncpu / 2))
        prefix = workdir + "/imaging_" + os.path.basename(msname).split(".ms")[0]
        if imagedir == "":
            imagedir = workdir
        if os.path.exists(imagedir) == False:
            os.makedirs(imagedir)
        os.system("rm -rf " + prefix + "*")
        if weight == "briggs":
            weight += " " + str(robust)
        if threshold < 1:
            threshold += 1
        elif threshold == 1:
            threshold += 0.1
        wsclean_args = [
            "-quiet",
            "-scale " + str(cellsize) + "asec",
            "-size " + str(imsize) + " " + str(imsize),
            "-no-dirty",
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
            "-parallel-reordering " + str(ncpu),
            "-parallel-gridding " + str(ngrid),
            "-auto-threshold 1 -auto-mask " + str(threshold),
            "-no-update-model-required",
            "-no-negative",
        ]
        
        ############################################
        # Parallel deconvolution within memory limit
        ############################################
        subimsize=1024
        if imsize > (2 * subimsize):
            subimage_memory=8*(subimsize**2)/(1024**3)
            n_subimage=int(imsize/subimsize)
            if ncpu*subimage_memory>mem:
                if n_subimage*subimage_memory>mem:
                    threads_to_use=int(mem/subimage_memory)
                    subimsize=int(imsize/threads_to_use)
                else:
                    threads_to_use=n_subimage
            else:
                threads_to_use=ncpu
            if threads_to_use>1 and subimsize>1024: 
                wsclean_args.append("-parallel-deconvolution 1024")
                wsclean_args.append("-deconvolution-threads " + str(threads_to_use))

        ######################################
        # Multiscale configuration
        ######################################
        if len(multiscale_scales) > 0:
            wsclean_args.append("-multiscale")
            wsclean_args.append("-multiscale-gain 0.1")
            wsclean_args.append(
                "-multiscale-scales " + ",".join([str(s) for s in multiscale_scales])
            )

        #####################################
        # Spectral imaging configuration
        #####################################
        if nchan > 1:
            wsclean_args.append(f"-channels-out {nchan}")
            wsclean_args.append(f"-join-channels")

        #####################################
        # Temporal imaging configuration
        #####################################
        if ntime > 1:
            wsclean_args.append(f"-intervals-out {ntime}")

        ################################################
        # Creating and using a 40arcmin diameter solar mask
        ################################################
        if use_solar_mask:
            fits_mask = prefix + "_solar-mask.fits"
            if os.path.exists(fits_mask) == False:
                mask_radius = 20
                print(
                    f"{os.path.basename(msname)} -- Creating solar mask of size: {mask_radius} arcmin.\n",file=logf,flush=True
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
        total_chunks = abs(nchan * ntime * npol)
        if total_chunks > soft_limit:
            resource.setrlimit(resource.RLIMIT_NOFILE, (total_chunks, hard_limit))

        ######################################
        # Running imaging
        ######################################
        wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " " + msname
        print(f"{os.path.basename(msname)} -- WSClean command: {wsclean_cmd}\n",file=logf,flush=True)
        msg = run_wsclean(wsclean_cmd, "meerwsclean", verbose=False)
        if msg != 0:
            gc.collect()
            print(f"{os.path.basename(msname)} -- Imaging is not successful.\n",file=logf,flush=True)
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
                    imagename, imagedir=imagedir + "/images", pol=pol
                )
                final_image_list.append(renamed_image)
            final_list = [final_image_list]
            if savemodel and len(modellist) > 0:
                final_model_list = []
                if os.path.exists(imagedir + "/models") == False:
                    os.makedirs(imagedir + "/models")
                for modelname in modellist:
                    renamed_model = rename_image(
                        modelname, imagedir=imagedir + "/models", pol=pol
                    )
                    final_model_list.append(renamed_model)
                final_list.append(final_model_list)
            if saveres and len(reslist) > 0:
                final_res_list = []
                if os.path.exists(imagedir + "/residuals") == False:
                    os.makedirs(imagedir + "/residuals")
                for resname in reslist:
                    renamed_res = rename_image(
                        resname, imagedir=imagedir + "/residuals", pol=pol
                    )
                    final_res_list.append(renamed_res)
                final_list.append(final_res_list)
            if use_solar_mask and os.path.exists(fits_mask):
                os.system("rm -rf " + fits_mask)
            print(f"{os.path.basename(msname)} -- Imaging is successfully done.\n",file=logf,flush=True)
            return 0, final_list
        else:
            if use_solar_mask and os.path.exists(fits_mask):
                os.system("rm -rf " + fits_mask)
            print(f"{os.path.basename(msname)} -- No image is made.\n",file=logf,flush=True)
            return 1, []
    except Exception as e:
        traceback.print_exc()
        return 1, []


def run_all_imaging(
    mslist="",
    workdir="",
    freqres=-1,
    timeres=-1,
    weight="briggs",
    robust=0.0,
    minuv=50,
    pol="I",
    multiscale_scales=[],
    threshold=1.0,
    use_solar_mask=True,
    savemodel=False,
    saveres=False,
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
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    Returns
    -------
    int
        Success message
    """
    if os.path.exists(workdir+"/logs/")==False:
        os.makedirs(workdir+"/logs/")
    mainlog_file=open(workdir+"/logs/imaging_targets.log","a")
    start_time = time.time()
    mslist=sorted(mslist)
    try:
        if len(mslist) == 0:
            print("Provide valid measurement set list.",file=mainlog_file,flush=True)
            return 1
        if freqres == -1 and timeres == -1:
            imagedir = workdir + "/imagedir_f_all_t_all"
        elif freqres != -1 and timeres == -1:
            imagedir = workdir + f"/imagedir_f_{freqres}_t_all"
        elif freqres == -1 and timeres != -1:
            imagedir = workdir + f"/imagedir_f_all_t_{timeres}"
        else:
            imagedir = workdir + f"/imagedir_f_{freqres}_t_{timeres}"
        if os.path.exists(imagedir) == False:
            os.makedirs(imagedir)
            
        ####################################
        # Filtering any corrupted ms
        #####################################    
        filtered_mslist=[] # Filtering in case any ms is corrupted
        for ms in mslist:
            checkcol=check_datacolumn_valid(ms)
            if checkcol:
                filtered_mslist.append(ms)
            else:
                print (f"Issue in : {ms}",file=mainlog_file,flush=True)
                os.system("rm -rf {ms}")
        mslist=filtered_mslist    

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
            for ms in mslist:
                msmd.open(ms)
                freqs = msmd.chanfreqs(0, unit="MHz")
                msmd.close()
                bw = max(freqs) - min(freqs)
                nchan = max(1, int(bw / freqres))
                nchan_list.append(nchan)
                
        #################################
        # Dask client setup
        #################################
        task = delayed(perform_imaging)(dry_run=True)
        mem_limit = run_limited_memory_task(task, dask_dir= workdir)
        dask_client, dask_cluster, n_jobs, n_threads, mem_limit = get_dask_client(
            len(mslist),
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit / 0.6,
        )
        print("\n#################################",file=mainlog_file,flush=True)
        print(f"Dask Dashboard: {dask_client.dashboard_link}",file=mainlog_file,flush=True)
        print("\n#################################",file=mainlog_file,flush=True)
        tasks = []
        for i in range(len(mslist)):
            ms = mslist[i]
            nchan = nchan_list[i]
            ntime = ntime_list[i]
            cellsize = calc_cellsize(ms, 5)
            fov = calc_field_of_view(ms)
            if fov < 60 * 60.0:  # Minimum 60 arcmin field of view
                fov = 60 * 60.0
            imsize = int(fov / cellsize)
            pow2 = round(np.log2(imsize / 10.0), 0)
            imsize = int((2**pow2) * 10)
            if len(multiscale_scales) == 0:
                multiscale_scales = calc_multiscale_scales(ms, 5)
            multiscale_scales = [str(i) for i in multiscale_scales]
            if os.path.exists(workdir+"/logs")==False:
                os.makedirs(workdir+"/logs")
            logfile=workdir+"/logs/imaging_"+os.path.basename(ms).split(".ms")[0]+".log"
            print (f"Starting imaging for ms : {ms}, Log file : {logfile}\n",file=mainlog_file,flush=True)
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
                print(f"Imaging failed for ms : {mslist[i]}",file=mainlog_file,flush=True)
            else:
                all_imaged_ms_list.append(mslist[i])
                for image in r[1][0]:
                    all_image_list.append(image)
        print(f"Numbers of input measurement sets : {len(mslist)}.",file=mainlog_file,flush=True)
        print(
            f"Imaging successfully done for: {len(all_imaged_ms_list)} measurement sets.",file=mainlog_file,flush=True
        )
        print(f"Total images made: {len(all_image_list)}.",file=mainlog_file,flush=True)
        print(f"Total time taken: {round(time.time()-start_time,2)}s",file=mainlog_file,flush=True)
        return 0
    except Exception as e:
        traceback.print_exc()
        print(f"Total time taken: {round(time.time()-start_time,2)}s",file=mainlog_file,flush=True)
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
