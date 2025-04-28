import os, numpy as np, copy, psutil, gc, traceback, resource
from casatools import msmetadata
from astropy.io import fits
from meersolar.pipeline.basic_func import *
from dask import delayed, compute, config
from optparse import OptionParser
from functools import partial
from casatasks import casalog

logfile = casalog.logfile()
os.system("rm -rf " + logfile)

"""
This code is written by Devojyoti Kansabanik, Apr 18, 2025
"""

def single_selfcal_iteration(
    msname,
    selfcaldir,
    cellsize,
    imsize,
    round_number=0,
    multiscale_scales=[],
    uvrange="",
    minuv=0,
    gaintype="G",
    calmode="ap",
    solmode="",
    solint="inf",
    refant="1",
    applymode="calonly",
    pol="I",
    threshold=5,
    weight="briggs",
    robust=0.0,
    use_solar_mask=True,
    min_fractional_bw=10,
    ncpu=-1,
    mem=-1,
):
    """
    A single self-calibration round
    Parameters
    ----------
    msname : str
        Name of the measurement set
    cellsize : float
        Cellsize in arcsec
    imsize :  int
        Image pixel size
    round_number : int, optional
        Selfcal iteration number
    multiscale_scales : list, optional
        Multiscale scales in pixel size
    uvrange : float, optional
       Uv range for calibration
    gaintype : str, optional
        Gaintype for gaincal ('G','T')
    calmode : str, optional
        Calibration mode ('p' or 'ap')
    solmode : str, optional
        Solver mode
    solint : str, optional
        Solution intervals
    refant : str, optional
        Reference antenna
    applymode : str, optional
        Solution apply mode (calonly or calflag)
    pol : str, optional
        Imaging polarization
    threshold : float, optional
        Imaging and auto-masking threshold
    weight : str, optional
        Image weighting
    robust : float, optional
        Robust parameter for briggs weighting
    use_solar_mask : bool, optional
        Use solar disk mask or not
    min_fractional_bw : float, optional
        Minimum fractional bandwidth in percentage
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
    float
        Negative based dynamic range
    str
        Image name
    str
        Model image name
    str
        Residual image name
    """
    limit_threads(n_threads=ncpu)
    from casatasks import gaincal, applycal, flagdata, delmod, flagmanager

    try:
        ##################################
        # Setup wsclean params
        ##################################
        if ncpu < 1:
            ncpu = psutil.cpu_count()
        if mem < 0:
            mem = psutil.virtual_memory().total / (1024**3)
        ngrid = max(1, int(ncpu / 2))
        msname = msname.rstrip("/")
        delmod(vis=msname, otf=True, scr=True)
        prefix = (
            selfcaldir
            + "/"
            + os.path.basename(msname).split(".ms")[0]
            + "_selfcal_"
            + str(round_number)
        )
        msmd = msmetadata()
        msmd.open(msname)
        npol = msmd.ncorrforpol()[0]
        msmd.close()
        if npol<4 and pol=="IQUV":
            pol="I"
        os.system("rm -rf " + prefix + "*")
        if weight == "briggs":
            weight += " " + str(robust)
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
            "-abs-mem " + str(mem),
            "-parallel-reordering " + str(ncpu),
            "-parallel-gridding " + str(ngrid),
            "-auto-threshold " + str(threshold) + " -auto-mask " + str(threshold + 0.1),
        ]
        if imsize > (2 * 1024):
            wsclean_args.append("-parallel-deconvolution 1024")
            wsclean_args.append("-deconvolution-threads " + str(ncpu))

        #####################################
        # Spectral imaging configuration
        #####################################
        frac_bw = calc_fractional_bandwidth(msname)
        if frac_bw > min_fractional_bw:
            msmd = msmetadata()
            msmd.open(msname)
            mean_freq = msmd.meanfreq(0)
            total_bw = np.abs(msmd.chanfreqs(0)[1] - msmd.chanfreqs(0)[0])
            msmd.close()
            bw = mean_freq * min_fractional_bw / 100.0
            nchan = int(total_bw / bw)
            if nchan > 1:
                print(
                    f"{os.path.basename(msname)} -- Spectral chunk: {nchan} for maintaining minimum fractional bandwidth: {min_fractional_bw}%"
                )
                wsclean_args.append(f"-channels-out {nchan}")
                wsclean_args.append(f"-join-channels")
        else:
            nchan=1

        ######################################
        # Multiscale configuration
        ######################################
        if len(multiscale_scales) > 0:
            wsclean_args.append("-multiscale")
            wsclean_args.append("-multiscale-gain 0.1")
            wsclean_args.append("-multiscale-scales " + ",".join([str(s) for s in multiscale_scales]))

        ################################################
        # Creating and using a 40arcmin diameter solar mask
        ################################################
        if use_solar_mask:
            fits_mask = msname.split(".ms")[0] + "_solar-mask.fits"
            if os.path.exists(fits_mask) == False:
                mask_radius = 25
                print(
                    f"{os.path.basename(msname)} -- Creating solar mask of size: {mask_radius} arcmin.\n"
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
        npol=len(list(pol))
        total_chunks = nchan * 1 *npol
        if total_chunks > soft_limit:
            resource.setrlimit(resource.RLIMIT_NOFILE, (total_chunks, hard_limit))
            
        ######################################
        # Running imaging
        ######################################
        wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " " + msname
        msg = run_wsclean(wsclean_cmd, "meerwsclean", verbose=False)
        if msg != 0:
            gc.collect()
            print(f"{os.path.basename(msname)} -- Imaging is not successful.")
            return 1, "", 0, 0, 0, "", "", ""

        #########################################
        # Restoring flags if applymode is calflag
        #########################################
        if applymode == "calflag":
            flags=flagmanager(vis=msname,mode='list')
            keys=flags.keys()
            for k in keys:
                if k=='MS':
                    pass
                else:
                    version=flags[0]['name']
                    try:
                        flagmanager(vis=msname,mode='restore',versionname=version)
                        flagmanager(vis=msname,mode='delete',versionname=version)
                    except:
                        pass

        #####################################
        # Analyzing images
        #####################################
        wsclean_files = {}
        for suffix in ["image", "model", "residual"]:
            files = glob.glob(prefix + f"*MFS-{suffix}.fits")
            if not files:
                files = glob.glob(prefix + f"*{suffix}.fits")
            else:
                for f in glob.glob(prefix + f"*{suffix}.fits"):
                    if "MFS" not in f:
                        os.system(f"rm -rf {f}")
            wsclean_files[suffix] = files
            os.system(f"rm -rf {prefix}*psf.fits")

        wsclean_images = wsclean_files["image"]
        wsclean_models = wsclean_files["model"]
        wsclean_residuals = wsclean_files["residual"]

        image_cube = make_stokes_wsclean_imagecube(
            wsclean_images, prefix + f"_{pol}_image.fits", keep_wsclean_images=False
        )
        model_cube = make_stokes_wsclean_imagecube(
            wsclean_models, prefix + f"_{pol}_model.fits", keep_wsclean_images=False
        )
        residual_cube = make_stokes_wsclean_imagecube(
            wsclean_residuals, prefix + f"_{pol}_residual.fits", keep_wsclean_images=False
        )

        model_flux, rms_DR, rms, neg_DR = calc_dyn_range(
            image_cube, model_cube, fits_mask
        )
        if model_flux == 0:
            gc.collect()
            print(f"{os.path.basename(msname)} -- No model flux.\n")
            return 1, "", 0, 0, 0, "", "", ""

        #####################
        # Perform calibration
        #####################
        if calmode == "p":
            minblperant = 3
        else:
            minblperant = 4
        gaincal(
            vis=msname,
            caltable=gain_caltable,
            uvrange=uvrange,
            refant=refant,
            calmode=calmode,
            solmode=solmode,
            gaintype=gaintype,
            minblperant=minblperant,
            minsnr=1,
            rmsthresh=[10, 7, 5, 3.5],
            solint=solint,
            solnorm=True,
        )
        if os.path.exists(gain_caltable) == False:
            print(f"{os.path.basename(msname)} -- No gain solutions are found.\n")
            gc.collect()
            return 2, "", 0, 0, 0, "", "", ""
        flagdata(vis=gain_caltable, mode="rflag", datacolumn="CPARAM", flagbackup=False)
        applycal(
            vis=msname,
            gaintable=[gain_caltable],
            interp=["nearest"],
            applymode=applymode,
            calwt=[False],
        )
        gc.collect()
        os.system("cp -r " + msname + " " + prefix + ".ms")
        return 0, gain_caltable, rms_DR, rms, neg_DR, image_cube, model_cube, residual_cube
    except Exception as e:
        traceback.print_exc()
        return 1, "" , 0, 0, 0, "", "", ""


def do_selfcal(
    msname="",
    selfcaldir="",
    start_threshold=5,
    end_threshold=3,
    max_iter=10,
    max_DR=1000,
    min_iter=2,
    DR_convegerence_frac=0.3,
    uvrange=">200lambda",
    minuv=0,
    solint="inf",
    weight="briggs",
    robust=0.0,
    do_apcal=True,
    min_fractional_bw=10,
    applymode="calonly",
    gaintype="G",
    solar_selfcal=True,
    ncpu=-1,
    mem=-1,
    dry_run=False,
):
    """
    Do selfcal iterations and use convergence rules to stop
    Parameters
    ----------
    msname : str
        Name of the measurement set
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
    min_fractional_bw : float, optional
        Minimum fractional bandwidth of spectral chunk in percentage
    applymode : str, optional
        Solution apply mode
    gaintype : str, optional
        Gaintype, G or T
    solar_selfcal : bool, optional
        Whether is is solar selfcal or not
    ncpu : int, optional
        Number of CPU threads to use
    mem : float, optional
        Memory in GB to use
    Returns
    -------
    int
        Success message
    str
        Final caltable
    """
    limit_threads(n_threads=ncpu)
    from casatasks import split, statwt, initweights

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    try:
        msname = os.path.abspath(msname.rstrip("/"))
        selfcaldir = selfcaldir.rstrip("/")
        if os.path.exists(selfcaldir) == False:
            os.makedirs(selfcaldir)
        else:
            os.system("rm -rf " + selfcaldir + "/*")
        os.chdir(selfcaldir)
        selfcalms = selfcaldir + "/selfcal_" + os.path.basename(msname)
        if os.path.exists(selfcalms):
            os.system("rm -rf " + selfcalms)
        if os.path.exists(selfcalms + ".flagversions"):
            os.system("rm -rf " + selfcalms + ".flagversions")
        hascor = check_datacolumn_valid(msname, datacolumn="CORRECTED_DATA")
        msmd = msmetadata()
        msmd.open(msname)
        scan = int(msmd.scannumbers()[0])
        field = int(msmd.fieldsforscan(scan)[0])
        msmd.close()
        if hascor:
            print(f"Spliting corrected data to ms : {selfcalms}")
            split(
                vis=msname,
                field=str(field),
                scan=str(scan),
                outputvis=selfcalms,
                datacolumn="corrected",
            )
        else:
            print(f"Spliting data to ms : {selfcalms}")
            split(
                vis=msname,
                field=str(field),
                scan=str(scan),
                outputvis=selfcalms,
                datacolumn="corrected",
            )
        msname = selfcalms

        ##########################################
        # Initiate proper weighting
        ##########################################
        initweights(vis=msname, wtmode="ones", dowtsp=True)
        statwt(vis=msname, datacolumn="data")

        ############################################
        # Imaging and calibration parameters
        ############################################
        print(f"{os.path.basename(msname)} -- Estimating imaging parameters ...")
        cellsize = calc_cellsize(msname, 5)
        fov = calc_field_of_view(msname)
        if fov < 60 * 60.0:  # Minimum 60 arcmin field of view
            fov = 60 * 60.0
        imsize = int(fov / cellsize)
        pow2 = round(np.log2(imsize / 10.0), 0)
        imsize = int((2**pow2) * 10)
        unflagged_antenna_names, flag_frac_list = get_unflagged_antennas(msname)
        refant = unflagged_antenna_names[0]
        msmd = msmetadata()
        msmd.open(msname)
        npol = msmd.ncorrforpol(0)
        msmd.close()

        ############################################
        # Initiating selfcal parameters
        ############################################
        print(
            f"{os.path.basename(msname)} -- Estimating self-calibration parameters..."
        )
        DR1 = 0.0
        DR2 = 0.0
        DR3 = 0.0
        DR4 = 0.0
        DR5 = 0.0
        DR6 = 0.0
        RMS1 = -1.0
        RMS2 = -1.0
        RMS3 = -1.0
        num_iter = 0
        num_iter_after_ap = 0
        num_iter_fixed_sigma = 0
        last_sigma_DR1 = 0
        last_sigma_DR2 = 0
        sigma_reduced_count = 0
        calmode = "p"
        if npol == 4:
            pol = "IQUV"
        else:
            pol = "I"
        threshold = start_threshold
        multiscale_scales = calc_multiscale_scales(msname, 5)
        multiscale_scales = [str(i) for i in multiscale_scales]

        ###########################################
        # Starting selfcal loops
        ##########################################
        while True:
            ##################################
            # Selfcal round parameters
            ##################################
            print("######################################")
            print(
                f"{os.path.basename(msname)} -- Selfcal iteration : "
                + str(num_iter)
                + ", Threshold: "
                + str(threshold)
                + ", Calibration mode: "
                + str(calmode)
            )
            msg, gaintable, dyn1, rms, dyn2, image_cube, model_cube, residual_cube = (
                single_selfcal_iteration(
                    msname,
                    selfcaldir,
                    cellsize,
                    imsize,
                    round_number=num_iter,
                    multiscale_scales=multiscale_scales,
                    uvrange=uvrange,
                    minuv=minuv,
                    gaintype=gaintype,
                    calmode=calmode,
                    solint=solint,
                    refant=str(refant),
                    applymode=applymode,
                    pol=pol,
                    min_fractional_bw=min_fractional_bw,
                    threshold=threshold,
                    weight=weight,
                    robust=robust,
                    use_solar_mask=solar_selfcal,
                    ncpu=ncpu,
                    mem=mem,
                )
            )
            if msg == 1:
                if num_iter == 0:
                    print(
                        f"{os.path.basename(msname)} -- No model flux is picked up in first round. Trying with lowest threshold.\n"
                    )
                    (
                        msg,
                        gaintable,
                        dyn1,
                        rms,
                        dyn2,
                        image_cube,
                        model_cube,
                        residual_cube,
                    ) = single_selfcal_iteration(
                        msname,
                        selfcaldir,
                        cellsize,
                        imsize,
                        round_number=num_iter,
                        multiscale_scales=multiscale_scales,
                        uvrange=uvrange,
                        minuv=minuv,
                        gaintype=gaintype,
                        calmode=calmode,
                        solint=solint,
                        refant=str(refant),
                        applymode=applymode,
                        pol=pol,
                        min_fractional_bw=min_fractional_bw,
                        threshold=end_threshold,
                        weight=weight,
                        robust=robust,
                        use_solar_mask=solar_selfcal,
                        ncpu=ncpu,
                        mem=mem,
                    )
                    if msg == 1:
                        return msg, []
                    else:
                        threshold = end_threshold
                else:
                    return msg, []
            if msg == 2:
                return msg, []
            if num_iter == 0:
                DR1 = DR5 = DR3 = dyn1
                DR2 = DR6 = DR4 = dyn2
                RMS1 = RMS2 = RMS3 = rms
            elif num_iter == 1:
                DR5 = dyn1
                DR6 = dyn2
                RMS2 = RMS1
                RMS1 = rms
            else:
                DR1 = DR3
                DR3 = DR5
                DR5 = dyn1
                DR2 = DR4
                DR4 = DR6
                DR6 = dyn2
                RMS3 = RMS2
                RMS2 = RMS1
                RMS1 = rms
            print(
                f"{os.path.basename(msname)} -- RMS based dynamic ranges: "
                + str(DR1)
                + ","
                + str(DR3)
                + ","
                + str(DR5)
            )
            print(
                f"{os.path.basename(msname)} -- RMS of the images: "
                + str(RMS1)
                + ","
                + str(RMS2)
                + ","
                + str(RMS3)
            )
            print(
                f"{os.path.basename(msname)} -- Negative based dynamic range: "
                + str(DR2)
                + ","
                + str(DR4)
                + ","
                + str(DR6)
            )
            print("######################################\n")
            #####################
            # If DR is decreasing
            #####################
            if (
                (
                    (DR5 < 0.85 * DR3 and DR5 < 0.9 * DR1 and DR3 > DR1)
                    or (DR6 < 0.75 * DR4 and DR6 < 0.8 * DR2 and DR4 > DR2)
                    or (RMS3 > 0.85 * RMS2 and RMS2 > 0.9 * RMS1 and RMS2 < RMS1)
                )
                and calmode == "p"
                and num_iter > min_iter
            ):
                print(
                    f"{os.path.basename(msname)} -- Dynamic range decreasing in phase-only self-cal."
                )
                if do_apcal:
                    print(f"{os.path.basename(msname)} -- Changed calmode to 'ap'.")
                    os.system("rm -rf " + msname)
                    os.system(
                        "cp -r "
                        + msname.split(".ms")[0]
                        + "_selfcal_"
                        + str(num_iter - 1)
                        + ".ms "
                        + msname
                    )
                    calmode = "ap"
                    solmode = "R"
                else:
                    return 0, gaintable
            elif (
                (
                    (DR5 < 0.9 * DR3 and DR3 > 1.5 * DR1)
                    or (DR6 < 0.85 * DR4 and DR4 > 2 * DR2)
                    or (RMS3 > 0.85 * RMS2 and RMS2 < 2 * RMS1)
                )
                and calmode == "ap"
                and num_iter_after_ap > min_iter
            ):
                print(
                    f"{os.path.basename(msname)} -- Dynamic range is decreasing after minimum numbers of 'ap' rounds.\n"
                )
                return 0, gaintable
            ###########################
            # If maximum DR has reached
            ###########################
            if DR5 >= max_DR and num_iter_after_ap > min_iter:
                print(
                    f"{os.path.basename(msname)} -- Maximum dynamic range is reached.\n"
                )
                return 0, gaintable
            ###########################
            # Checking DR convergence
            ###########################
            # Condition 1
            ###########################
            if (
                ((do_apcal and calmode == "ap") or do_apcal == False)
                and num_iter_fixed_sigma > min_iter
                and abs(round(np.nanmedian([DR1, DR3, DR5]), 0) - last_sigma_DR1)
                / last_sigma_DR1
                < DR_convegerence_frac
                and abs(last_sigma_DR2 - last_sigma_DR1) / last_sigma_DR2
                < DR_convegerence_frac
                and sigma_reduced_count > 1
            ):
                if threshold > end_threshold:
                    print(
                        f"{os.path.basename(msname)} -- DR does not increase over last two changes in threshold, but minimum threshold has not reached yet.\n"
                    )
                    print(
                        f"{os.path.basename(msname)} -- Starting final self-calibration rounds with threshold = "
                        + str(end_threshold)
                        + "sigma...\n"
                    )
                    threshold = end_threshold
                    sigma_reduced_count += 1
                    num_iter_fixed_sigma = 0
                    continue
                else:
                    print(
                        f"{os.path.basename(msname)} -- Selfcal converged. DR does not increase over last two changes in threshold.\n"
                    )
                    return 0, gaintable
            ###############
            # Condition 2
            ###############
            else:
                if (
                    abs(DR1 - DR3) / DR3 < DR_convegerence_frac
                    and abs(DR2 - DR4) / DR2 < DR_convegerence_frac
                    and num_iter > min_iter
                    and threshold == end_threshold + 1
                ):
                    if do_apcal and calmode == "p":
                        print(
                            f"{os.path.basename(msname)} -- Dynamic range converged. Changing calmode to 'ap'.\n"
                        )
                        calmode = "ap"
                        solmode = "R"
                    elif (
                        do_apcal and num_iter_after_ap > min_iter + 3
                    ) or do_apcal == False:
                        print(
                            f"{os.path.basename(msname)} -- Self-calibration has converged.\n"
                        )
                        return 0, gaintable
                elif (
                    abs(DR1 - DR3) / DR3 < DR_convegerence_frac
                    and abs(DR2 - DR4) / DR2 < DR_convegerence_frac
                    and threshold > end_threshold
                    and num_iter_fixed_sigma > min_iter
                ):
                    threshold -= 1
                    print(
                        f"{os.path.basename(msname)} -- Reducing threshold to : "
                        + str(threshold)
                    )
                    do_uvsub_flag = True
                    num_iter_fixed_sigma = 0
                    if last_sigma_DR1 > 0:
                        last_sigma_DR2 = round(last_sigma_DR1, 0)
                        last_sigma_DR1 = round(np.nanmean([DR1, DR3, DR5]), 0)
                    else:
                        last_sigma_DR1 = round(np.nanmean([DR1, DR3, DR5]), 0)
                        last_sigma_DR2 = round(last_sigma_DR1, 0)
                elif (
                    (
                        (do_apcal and calmode == "ap" and num_iter_after_ap > min_iter)
                        or (do_apcal == False and num_iter > min_iter)
                    )
                    and abs(DR1 - DR3) / DR3 < DR_convegerence_frac
                    and abs(DR2 - DR4) / DR2 < DR_convegerence_frac
                    and threshold <= end_threshold
                ) or (
                    (do_apcal == False or (do_apcal and calmode == "ap"))
                    and num_iter == max_iter
                ):
                    print(
                        f"{os.path.basename(msname)} -- Self-calibration has converged.\n"
                    )
                    return 0, gaintable
            os.system(
                "cp -r "
                + msname
                + " "
                + msname.split(".ms")[0]
                + "_selfcal_"
                + str(num_iter)
                + ".ms "
            )
            num_iter += 1
            if calmode == "ap":
                num_iter_after_ap += 1
            num_iter_fixed_sigma += 1
    except Exception as e:
        traceback.print_exc()
        return 1, []


def main():
    starttime = time.time()
    usage = "Self-calibration"
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
        default="",
        help="Working directory",
        metavar="String",
    )
    parser.add_option(
        "--start_thresh",
        dest="start_thresh",
        default=5,
        help="Starting CLEANing threshold ",
        metavar="Float",
    )
    parser.add_option(
        "--stop_thresh",
        dest="stop_thresh",
        default=3,
        help="Stop CLEANing threshold",
        metavar="Float",
    )
    parser.add_option(
        "--max_iter",
        dest="max_iter",
        default=10,
        help="Maximum numbers of selfcal iterations",
        metavar="Integer",
    )
    parser.add_option(
        "--max_DR",
        dest="max_DR",
        default=1000,
        help="Maximum dynamic range",
        metavar="Integer",
    )
    parser.add_option(
        "--min_iter",
        dest="min_iter",
        default=2,
        help="Minimum numbers of selfcal iterations",
        metavar="Integer",
    )
    parser.add_option(
        "--conv_frac",
        dest="conv_frac",
        default=0.3,
        help="Fractional change in DR to determine convergence",
        metavar="Float",
    )
    parser.add_option(
        "--solint",
        dest="solint",
        default="inf",
        help="Solution interval",
        metavar="String",
    )
    parser.add_option(
        "--do_apcal",
        dest="do_apcal",
        default=False,
        help="Peform ap-selfcal or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--solar_selfcal",
        dest="solar_selfcal",
        default=True,
        help="Peforming solar self-calibration or not",
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
    parser.add_option(
        "--keep_backup",
        dest="keep_backup",
        default=False,
        help="Keep backup of self-calibration rounds",
        metavar="Boolean",
    )
    parser.add_option(
        "--uvrange",
        dest="uvrange",
        default="",
        help="Calibration UV-range (CASA format)",
        metavar="String",
    )
    parser.add_option(
        "--minuv",
        dest="minuv",
        default=0,
        help="Minimum UV-lambda used for imaging",
        metavar="Float",
    )
    parser.add_option(
        "--weight",
        dest="weight",
        default="briggs",
        help="Imaging weight",
        metavar="String",
    )
    parser.add_option(
        "--robust",
        dest="robust",
        default=0.0,
        help="Robust parameter for briggs weight",
        metavar="Float",
    )
    parser.add_option(
        "--applymode",
        dest="applymode",
        default="calonly",
        help="Solution apply mode",
        metavar="String",
    )
    parser.add_option(
        "--gaintype",
        dest="gaintype",
        default="G",
        help="Gain solution type",
        metavar="String",
    )
    parser.add_option(
        "--min_fractional_bw",
        dest="min_fractional_bw",
        default=10.0,
        help="Minimum fractional bandwidth of spectral chunk in percentage",
        metavar="Float",
    )
    (options, args) = parser.parse_args()
    if options.mslist == None:
        print("Please provide a mslist.")
        return 1
    mslist = str(options.mslist).split(",")
    if len(mslist) == 0:
        print("Please provide at-least one measurement set.")
        return 1
    if options.workdir == "" or os.path.exists(options.workdir) == False:
        print("Please provide a valid working directory.")
        return 1
    caldir = options.workdir + "/caltables"
    if os.path.exists(caldir) == False:
        os.makedirs(caldir)
    task = delayed(do_selfcal)(dry_run=True)
    mem_limit = run_limited_memory_task(task)
    partial_do_selfcal = partial(
        do_selfcal,
        start_threshold=float(options.start_thresh),
        end_threshold=float(options.stop_thresh),
        max_iter=int(options.max_iter),
        max_DR=float(options.max_DR),
        min_iter=int(options.min_iter),
        DR_convegerence_frac=float(options.conv_frac),
        uvrange=str(options.uvrange),
        minuv=float(options.minuv),
        solint=str(options.solint),
        weight=str(options.weight),
        robust=float(options.robust),
        do_apcal=eval(str(options.do_apcal)),
        gaintype=options.gaintype,
        applymode=options.applymode,
        min_fractional_bw=float(options.min_fractional_bw),
        solar_selfcal=eval(str(options.solar_selfcal)),
    )

    dask_client, dask_cluster, n_jobs, n_threads = get_dask_client(
        len(mslist),
        float(options.cpu_frac),
        float(options.mem_frac),
        min_mem_per_job=mem_limit / 0.8,
    )
    workers = list(dask_client.scheduler_info()["workers"].items())
    addr, stats = workers[0]
    memory_limit = stats["memory_limit"] / 1024**3
    target_frac = config.get("distributed.worker.memory.target")
    memory_limit *= target_frac
    tasks = []
    for ms in mslist:
        tasks.append(
            delayed(partial_do_selfcal)(
                ms,
                options.workdir
                + "/"
                + os.path.basename(ms).split(".ms")[0]
                + "_selfcal",
                ncpu=n_threads,
                mem=memory_limit,
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
            print(f"Self-calibration was not successful for ms: {mslist[i]}.")
        else:
            gcal_list.append(r[1])
    final_gain_caltable = caldir + "/full_selfcal.gcal"
    if len(gcal_list) > 0:
        final_gain_caltable = merge_caltables(
            gcal_list,
            final_gain_caltable,
            append=False,
            keepcopy=eval(str(options.keep_backup)),
        )
        if eval(str(options.keep_backup)) == False:
            for ms in mslist:
                selfcaldir = (
                    options.workdir
                    + "/"
                    + os.path.basename(ms).split(".ms")[0]
                    + "_selfcal"
                )
                os.system("rm -rf " + selfcaldir)
        print("Final caltable:")
        if os.path.exists(final_gain_caltable):
            print(f"{final_gain_caltable}")
        print(f"Total time taken: {round(time.time()-starttime,2)}s")
        return 0
    else:
        print("No self-calibration is successful.")
        print(f"Total time taken: {round(time.time()-starttime,2)}s")
        return 1


if __name__ == "__main__":
    result = main()
    os.system("rm -rf casa*log")
    if result > 0:
        result = 1
    print("\n###################\nSelf-calibration is done.\n###################\n")
    os._exit(result)
