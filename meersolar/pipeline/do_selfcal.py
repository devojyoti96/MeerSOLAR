import os, numpy as np, copy
from casatasks import gaincal, applycal, flagdata, split, initweights, statwt
from casatools import msmetadata
from astropy.io import fits
from meersolar.pipeline.basic_func import *
from optparse import OptionParser
from casatasks import casalog

logfile = casalog.logfile()
os.system("rm -rf " + logfile)

"""
This code is written by Devojyoti Kansabanik, Apr 18, 2025
"""
    
def single_selfcal_iteration(
    msname,
    cellsize,
    imsize,
    round_number=0,
    multiscale_scales=[],
    uvrange="",
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
    ncpu : int, optional
        Number of CPUs to use in WSClean
    mem : float, optional
        Memory usage limit in WSClean
    Returns
    -------
    float
        RMS based dynamic range
    float
        Negative based dynamic range
    str
        Image name
    str
        Model image name
    str
        Residual image name
    """
    ##################################
    # Setup wsclean params
    ##################################
    prefix = msname.split(".ms")[0] + "_selfcal_" + str(round_number)
    os.system("rm -rf " + prefix + "*")
    if weight == "briggs":
        weight += " " + str(robust)
    wsclean_args = [
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
        "-auto-threshold " + str(threshold) + " -auto-mask " + str(threshold + 0.1),
    ]
    if len(multiscale_scales) > 0:
        wsclean_args.append("-multiscale")
        wsclean_args.append("-multiscale-gain 0.1")
        wsclean_args.append("-multiscale-scales " + ",".join(multiscale_scales))
    if ncpu > 0:
        wsclean_args.append("-j " + str(ncpu))
    if mem > 0:
        wsclean_args.append("-abs-mem " + str(mem))
    ################################################
    # Creating and using a 40arcmin solar mask
    ################################################
    if use_solar_mask:
        fits_mask = msname.split(".ms")[0] + "_solar-mask.fits"
        if os.path.exists(fits_mask) == False:
            fits_mask = create_circular_mask(msname, cellsize, imsize, mask_radius=20)
        if fits_mask!=None and os.path.exists(fits_mask):
            wsclean_args.append("-fits-mask " + fits_mask)
                
    ######################################
    # Running imaging
    ######################################
    wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " " + msname
    msg = run_wsclean(wsclean_cmd, container_name)
    if msg!=0:
        print ("Imaging is not successful.")
        return 1, 0, 0, "", "", ""
    
    #####################################
    # Analyzing images
    #####################################
    wsclean_images = glob.glob(prefix + "*image.fits")
    wsclean_models = glob.glob(prefix + "*model.fits")
    wsclean_residuals = glob.glob(prefix + "*residual.fits")
    model_flux, rms_DR, neg_DR = calc_dyn_range(
        wsclean_images, wsclean_models, fits_mask
    )
    if model_flux == 0:
        print("No model flux.\n")
        return 1, 0, 0, "", "", ""
    image_cube=make_stokes_wsclean_imagecube(wsclean_images, prefix+"_IQUV_image.fits", keep_wsclean_images=False)
    model_cube=make_stokes_wsclean_imagecube(wsclean_models, prefix+"_IQUV_model.fits", keep_wsclean_images=False)
    residual_cube=make_stokes_wsclean_imagecube(wsclean_residuals, prefix+"_IQUV_residual.fits", keep_wsclean_images=False)
    
    #####################
    # Perform calibration
    #####################
    delay_caltable=prefix+".kcal"
    gain_caltable=prefix+".gcal"
    delaycal(msname,delay_caltable,str(refant),solint=solint)
    if os.path.exists(delay_caltable)==False:
        gaintable=[]
    else:
        gaintable=[delay_caltable]
    gaincal(
        vis=msname,
        caltable=gain_caltable,
        uvrange=uvrange,
        refant=refant,
        calmode=calmode,
        solmode=solmode,
        rmsthresh=[10, 7, 5, 3.5],
        solint=solint,
        solnorm=True,
        gaintable=gaintable,
    )
    if os.path.exists(delay_caltable)==False and os.path.exists(gain_caltable)==False:
        print("No gain solutions are found.\n")
        return 2, 0, 0, "", "", ""
    gaintable.append(gain_caltable)
    applycal(
        vis=msname,
        gaintable=gaintable,
        interp=["nearest"]*len(gaintable),
        applymode=applymode,
        calwt=[False],
    )
    os.system("cp -r " + msname + " " + prefix + ".ms")
    return 0, rms_DR, neg_DR, image_cube, model_cube, residual_cube

def do_selfcal(
    msname,
    spw,
    timerange,
    scan_number,
    workdir="",
    start_threshold=10,
    end_threshold=5,
    max_iter=100,
    max_DR=1000,
    min_iter=3,
    DR_convegerence_frac=0.3,
    gaintable=[],
    solint="inf",
    do_apcal=False,
    freqavg=1,
    ncpu=-1,
    mem=-1,
    verbose=False,
):
    """
    Do selfcal iterations and use convergence rules to stop
    Parameters
    ----------
    msname : str
        Name of the measurement set
    spw : str
        Spectral window to use
    timerange : str
        Time range to use
    scan_number : int
        Scan number (provide correct scan number associated with the given time range)
    workdir : str
        Working directory
    start_threshold : int
        Start CLEAN threhold
    end_threshold : int
        End CLEAN threshold
    max_iter : int
        Maximum numbers of selfcal iterations
    max_DR : float
        Maximum dynamic range
    min_iter : int
        Minimum numbers of seflcal iterations at different stages
    DR_convegerence_frac : float
        Dynamic range fractional change to consider as converged
    gaintable : list
        Pre calibration table
    solint : str
        Solutions interval
    do_apcal : bool
        Perform ap-selfcal or not
    freqavg : float
        Frequency averaging in MHz
    ncpu : int
        Number of cpu threads to use
    mem : float
        Memory in GB to use
    verbose : bool
        Verbose output or not
    Returns
    -------
    """
    msname = os.path.abspath(msname)
    cwd = os.getcwd()
    msmd = msmetadata()
    msmd.open(msname)
    field = msmd.fieldsforscan(int(scan_number))[0]
    freqres = msmd.chanres(0)[0] / 10**6
    msmd.close()
    bw_smear_limit = calc_bw_smearing_freqwidth(msname)
    if (
        freqavg > bw_smear_limit
    ):  # If user given freqavg is more than bandwidth smearing frequency
        freqavg = bw_smear_limit
    if freqres < freqavg:  # Averaging at-least for user-given spectral averaging in MHz
        width = int(freqavg / freqres)
    else:
        width = 1
    ######################################
    # Spliting and phase shifting
    ######################################
    spw_str = (
        spw.split("0:")[-1].split("~")[0] + "_" + spw.split("0:")[-1].split("~")[-1]
    )
    time_str = "".join(
        timerange.split("~")[0].split("/")[-1].split(".")[0].split(":")
    ) + "".join(timerange.split("~")[-1].split("/")[-1].split(".")[0].split(":"))
    outputvis = (
        os.path.dirname(msname)
        + "/solar_"
        + spw_str
        + "_"
        + time_str
        + "_"
        + str(scan_number)
        + ".ms"
    )
    if os.path.exists(outputvis):
        os.system("rm -rf " + outputvis)
    print("Spliting data for self-calibration...\n")
    datacolumn_to_split = "data"
    if verbose:
        print(
            "split(vis='"
            + msname
            + "',outputvis='"
            + outputvis
            + "',field='"
            + str(field)
            + "',scan='"
            + str(scan_number)
            + "',datacolumn='"
            + datacolumn_to_split
            + "',spw='"
            + spw
            + "',timerange='"
            + timerange
            + "',width="
            + str(width)
            + ")\n"
        )
    split(
        vis=msname,
        outputvis=outputvis,
        field=str(field),
        scan=str(scan_number),
        datacolumn=datacolumn_to_split,
        spw=spw,
        timerange=timerange,
        width=width,
    )
    msname = outputvis  # msname is replaced with the splited ms
    if workdir == "":
        workdir = (
            os.path.dirname(msname)
            + "/"
            + os.path.basename(msname).split(".ms")[0]
            + "_selfcal_workdir"
        )
    else:
        workdir = (
            workdir
            + "/"
            + os.path.basename(msname).split(".ms")[0]
            + "_selfcal_workdir"
        )
    if os.path.exists(workdir):
        os.system("rm -rf " + workdir)
    os.makedirs(workdir)
    os.chdir(workdir)
    outputvis = split_move_sun(msname, int(scan_number))
    os.system("rm -rf " + msname)
    os.system("mv " + outputvis + " " + workdir)
    msname = workdir + "/" + os.path.basename(outputvis)
    #######################################
    # Applying any previous basic solutions
    #######################################
    if len(gaintable) > 0:
        applycaltables = []
        for gtable in gaintable:
            os.system("cp -r " + gtable + " " + workdir)
            applycaltables.append(workdir + "/" + os.path.basename(gtable))
        print("Applying solutions from : " + ",".join(applycaltables) + "\n")
        applycal(
            vis=msname,
            scan=str(scan_number),
            timerange=timerange,
            gaintable=applycaltables,
            interp=["nearest"] * len(applycaltables),
            calwt=[True],
            flagbackup=False,
        )
        applymode = "calonly"
    tb = table()
    tb.open(msname, nomodify=False)
    cor_data = tb.getcol("CORRECTED_DATA")
    tb.putcol("DATA", cor_data)
    tb.flush()
    tb.close()
    ###########################################
    # Flagging on corrected data
    ###########################################
    flag_uvbin(msname, uvbin=1000, datacolumn="data", mode="tfcrop", flagbackup=False)
    ############################################
    # Imaging and calibration parameters
    ############################################
    minuv = 200
    cellsize = calc_cellsize(msname, 3)
    imsize = int((2 * 3600) / cellsize)
    pow2 = round(np.log2(imsize / 10.0), 0)
    imsize = int((2**pow2) * 10)
    unflagged_antenna_names, flag_frac_list = get_unflagged_antennas(msname)
    refant = unflagged_antenna_names[0]
    ############################################
    # Initiating selfcal parameters
    ############################################
    DR1 = 0.0
    DR2 = 0.0
    DR3 = 0.0
    DR4 = 0.0
    DR5 = 0.0
    DR6 = 0.0
    num_iter = 0
    num_iter_after_ap = 0
    num_iter_fixed_sigma = 0
    last_sigma_DR1 = 0
    last_sigma_DR2 = 0
    sigma_reduced_count = 0
    calmode = "p"
    solmode = "R"
    pol = "I"
    threshold = start_threshold
    multiscale_scales = calc_multiscale_scales(msname, 3, max_scale=5)
    multiscale_scales = [str(i) for i in multiscale_scales]
    vis_weight = False
    do_uvsub_flag = False
    while True:
        if threshold <= start_threshold - 1:
            print("Using visibility weighting...\n")
            vis_weight = True
        if num_iter_fixed_sigma > 0 and do_uvsub_flag == True:
            do_uvsub_flag = False
        ##################################
        # Selfcal round parameters
        ##################################
        print("######################################")
        print(
            "Selfcal iteration : "
            + str(num_iter)
            + ", Threshold: "
            + str(threshold)
            + ", Cal mode: "
            + str(calmode)
        )
        msg, dyn1, dyn2, casa_image, casa_model = single_selfcal_iteration(
            msname,
            cellsize,
            imsize,
            round_number=num_iter,
            multiscale_scales=multiscale_scales,
            minuv=minuv,
            calmode=calmode,
            solmode=solmode,
            solint=solint,
            refant=refant,
            applymode=applymode,
            pol=pol,
            threshold=threshold,
            weight="briggs",
            robust=1.0,
            ncpu=ncpu,
            mem=mem,
            verbose=verbose,
            vis_weight=vis_weight,
            do_uvsub_flag=do_uvsub_flag,
        )
        if msg == 1:
            if num_iter == 0:
                print(
                    "No model flux is picked up in first round. Trying with lowest threshold.\n"
                )
                msg, dyn1, dyn2, casa_image, casa_model = single_selfcal_iteration(
                    msname,
                    cellsize,
                    imsize,
                    round_number=num_iter,
                    multiscale_scales=multiscale_scales,
                    minuv=minuv,
                    calmode=calmode,
                    solmode=solmode,
                    solint=solint,
                    refant=refant,
                    applymode=applymode,
                    pol=pol,
                    threshold=start_threshold - 1,
                    weight="briggs",
                    robust=1.0,
                    ncpu=ncpu,
                    mem=mem,
                    verbose=verbose,
                    vis_weight=vis_weight,
                    do_uvsub_flag=do_uvsub_flag,
                )
                if msg == 1:
                    os.chdir(cwd)
                    return msg, ""
                else:
                    threshold = start_threshold - 1
            else:
                os.chdir(cwd)
                return msg, ""
        if msg == 2:
            os.chdir(cwd)
            return msg, ""
        if num_iter == 0:
            DR1 = DR5 = DR3 = dyn1
            DR2 = DR6 = DR4 = dyn2
        elif num_iter == 1:
            DR5 = dyn1
            DR6 = dyn2
        else:
            DR1 = DR3
            DR3 = DR5
            DR5 = dyn1
            DR2 = DR4
            DR4 = DR6
            DR6 = dyn2
        print("RMS based dynamic ranges: " + str(DR1) + "," + str(DR3) + "," + str(DR5))
        print(
            "Negative based dynamic range: "
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
            )
            and calmode == "p"
            and num_iter > min_iter
        ):
            print("Dynamic range decreasing in phase-only self-cal.")
            if do_apcal:
                print("Changed calmode to 'ap'.")
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
                final_caltable = (
                    msname.split(".ms")[0] + "_selfcal_" + str(num_iter - 1) + ".gcal"
                )
                os.chdir(cwd)
                return 0, final_caltable
        elif (
            (
                (DR5 < 0.9 * DR3 and DR3 > 1.5 * DR1)
                or (DR6 < 0.85 * DR4 and DR4 > 2 * DR2)
            )
            and calmode == "ap"
            and num_iter_after_ap > min_iter
        ):
            print("Dynamic range is decreasing after minimum numbers of 'ap' rounds.\n")
            final_caltable = (
                msname.split(".ms")[0] + "_selfcal_" + str(num_iter - 1) + ".gcal"
            )
            os.chdir(cwd)
            return 0, final_caltable
        ###########################
        # If maximum DR has reached
        ###########################
        if DR5 >= max_DR and num_iter_after_ap > min_iter:
            print("Maximum dynamic range is reached.\n")
            final_caltable = (
                msname.split(".ms")[0] + "_selfcal_" + str(num_iter) + ".gcal"
            )
            os.chdir(cwd)
            return 0, final_caltable
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
                    "DR does not increase over last two changes in threshold, but minimum threshold has not reached yet.\n"
                )
                print(
                    "Starting final self-calibration rounds with threshold = "
                    + str(end_threshold)
                    + "sigma...\n"
                )
                threshold = end_threshold
                sigma_reduced_count += 1
                num_iter_fixed_sigma = 0
                continue
            else:
                print(
                    "Selfcal converged. DR does not increase over last two changes in threshold.\n"
                )
                final_caltable = (
                    msname.split(".ms")[0] + "_selfcal_" + str(num_iter) + ".gcal"
                )
                os.chdir(cwd)
                return 0, final_caltable
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
                    print("Dynamic range converged. Changing calmode to 'ap'.\n")
                    calmode = "ap"
                    solmode = "R"
                elif (
                    do_apcal and num_iter_after_ap > min_iter + 3
                ) or do_apcal == False:
                    print("Self-calibration has converged.\n")
                    final_caltable = (
                        msname.split(".ms")[0] + "_selfcal_" + str(num_iter) + ".gcal"
                    )
                    os.chdir(cwd)
                    return 0, final_caltable
            elif (
                abs(DR1 - DR3) / DR3 < DR_convegerence_frac
                and abs(DR2 - DR4) / DR2 < DR_convegerence_frac
                and threshold > end_threshold
                and num_iter_fixed_sigma > min_iter
            ):
                threshold -= 1
                print("Reducing threshold to : " + str(threshold))
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
                print("Self-calibration has converged.\n")
                final_caltable = (
                    msname.split(".ms")[0] + "_selfcal_" + str(num_iter) + ".gcal"
                )
                os.chdir(cwd)
                return 0, final_caltable
        num_iter += 1
        if calmode == "ap":
            num_iter_after_ap += 1
        num_iter_fixed_sigma += 1


def main():
    usage = "Self-calibration of solar scans"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--msname",
        dest="msname",
        default=None,
        help="Name of measurement set",
        metavar="Measurement Set",
    )
    parser.add_option(
        "--spw",
        dest="spw",
        default="",
        help="Spectral window (start_chan~end_chan)",
        metavar="String",
    )
    parser.add_option(
        "--timerange",
        dest="timerange",
        default="",
        help="Time range to be used for self-calibration",
        metavar="String",
    )
    parser.add_option(
        "--scan",
        dest="scan_number",
        default="",
        help="Scan number",
        metavar="Integer",
    )
    parser.add_option(
        "--start_thresh",
        dest="start_threshold",
        default=10,
        help="Starting CLEANing threshold ",
        metavar="Float",
    )
    parser.add_option(
        "--end_thresh",
        dest="end_threshold",
        default=5,
        help="End threshold",
        metavar="Float",
    )
    parser.add_option(
        "--max_iter",
        dest="max_iter",
        default=100,
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
        default=1,
        help="Minimum numbers of selfcal iterations",
        metavar="Integer",
    )
    parser.add_option(
        "--DR_frac",
        dest="DR_frac",
        default=0.3,
        help="Fractional change in DR to determine convergence",
        metavar="Float",
    )
    parser.add_option(
        "--workdir",
        dest="workdir",
        default="",
        help="Working directory",
        metavar="String",
    )
    parser.add_option(
        "--gaintable",
        dest="gaintable",
        default="",
        help="Pre-calibration table (comma seperated)",
        metavar="String",
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
        "--freqavg",
        dest="freqavg",
        default=1.0,
        help="Frequency averaging of the data in MHz",
        metavar="Float",
    )
    parser.add_option(
        "--ncpu",
        dest="ncpu",
        default=-1,
        help="Numbers of CPU threads to use by WSClean (default all)",
        metavar="Integer",
    )
    parser.add_option(
        "--mem",
        dest="mem",
        default=-1,
        help="Memory in GB to be used by WSClean",
        metavar="Integer",
    )
    parser.add_option(
        "--verbose",
        dest="verbose",
        default=False,
        help="Verbose output or not",
        metavar="Boolean",
    )
    (options, args) = parser.parse_args()
    if options.scan_number == "":
        print("Please provide correct scan number.")
        return 1
    if options.msname != None and os.path.exists(options.msname):
        if options.workdir == "" or os.path.exists(options.workdir) == False:
            workdir = os.path.dirname(os.path.abspath(options.msname)) + "/workdir"
            if os.path.exists(workdir) == False:
                os.makedirs(workdir)
        else:
            workdir = options.workdir
        caldir = workdir + "/selfcaltables"
        if os.path.exists(caldir) == False:
            os.makedirs(caldir)
        gaintable = (options.gaintable).split(",")
        msg, final_caltable = do_solar_selfcal(
            options.msname,
            "0:" + str(options.spw),
            options.timerange,
            str(options.scan_number),
            workdir=workdir,
            start_threshold=int(options.start_threshold),
            end_threshold=int(options.end_threshold),
            max_iter=int(options.max_iter),
            max_DR=int(options.max_DR),
            min_iter=int(options.min_iter),
            DR_convegerence_frac=float(options.DR_frac),
            gaintable=gaintable,
            solint=options.solint,
            do_apcal=eval(str(options.do_apcal)),
            freqavg=float(options.freqavg),
            ncpu=int(options.ncpu),
            mem=int(options.mem),
            verbose=eval(str(options.verbose)),
        )
        if msg == 0:
            if os.path.exists(
                caldir
                + "/"
                + os.path.basename(final_caltable).split("_selfcal")[0]
                + ".gcal.junk"
            ):
                os.system(
                    "rm -rf "
                    + caldir
                    + "/"
                    + os.path.basename(final_caltable).split("_selfcal")[0]
                    + ".gcal.junk"
                )
            os.system(
                "cp -r "
                + final_caltable
                + " "
                + caldir
                + "/"
                + os.path.basename(final_caltable).split("_selfcal")[0]
                + ".gcal"
            )
        os.system("rm -rf casa*log")
        return msg
    else:
        print("Please provide correct measurement set.")
        os.system("rm -rf casa*log")
        return 1


if __name__ == "__main__":
    result = main()
    os.system("rm -rf casa*log")
    if result > 0:
        result = 1
    print("\n###################\nSelf-calibration is done.\n###################\n")
    os._exit(result)
