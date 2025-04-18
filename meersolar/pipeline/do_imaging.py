import os, glob, resource
from astropy.io import fits
from casatasks import split, applycal, casalog
from casatools import table, msmetadata
from meersolar.pipeline.basic_func import *
from optparse import OptionParser

# from sendmail import *


def applycal_and_perform_imaging(
    msname,
    imsize,
    cellsize,
    spw,
    timerange,
    imagedir="",
    freqres=-1,
    timeres=-1,
    calibrator_gaintables=[],
    selfcaltables=[],
    pol="I",
    weight="briggs",
    robust=0.0,
    multiscale_scales=[],
    minuv_l=0,
    maxuv_l=-1,
    ncpu=-1,
    mem=-1,
):
    """
    msname : str
        Name of the measurement set
    imsize : int
        Image size
    cellsize : float
        Cell size in arcsecond
    spw : str
        Spectral window
    timerange : str
        Time range
    imagedir : str
        Image directory
    freqres : float
        Frequency resolution of images in MHz
    timeres : float
        Time resolution of images in seconds
    calibrator_gaintables : list
        Calibrator gaintables
    selfcaltables : list
        Selfcal gaintables
    pol : str
        Stokes parameters to image
    weight : str
        Image weighting parameter
    robust : float
        Robust parameter for briggs weighting
    multiscale_scales : list
        Multiscale scales
    minuv_l : float
        Minimum UV cutoff in lambda
    maxuv_l : float
        Maximum UV cutoff in lambda
    ncpu : int
        Number of CPU threads to use
    mem : float
        Memory in GB to use
    """
    ########################
    # Spliting and arranging
    ########################
    msname = os.path.abspath(msname)
    if imagedir == "":
        imagedir = os.path.dirname(msname)
    spw_str = (
        spw.split("0:")[-1].split("~")[0] + "_" + spw.split("0:")[-1].split("~")[-1]
    )
    time_str = "".join(
        timerange.split("~")[0].split("/")[-1].split(".")[0].split(":")
    ) + "".join(timerange.split("~")[-1].split("/")[-1].split(".")[0].split(":"))
    outputvis = os.path.dirname(msname) + "/solar_" + spw_str + "_" + time_str + ".ms"
    os.system("rm -rf " + outputvis + "*")
    print("Spliting dataset ....\n")
    split(
        vis=msname, outputvis=outputvis, spw=spw, timerange=timerange, datacolumn="data"
    )
    msname = outputvis
    msmd = msmetadata()
    msmd.open(msname)
    scan_number = msmd.scannumbers()[0]
    times = msmd.timesforscan(scan_number)
    total_chans = msmd.nchan(0)
    total_time = times[-1] - times[0]  # In second
    total_bw = msmd.bandwidths(0) / 10**6  # In MHz
    ms_freqres = msmd.chanres(0)[0] / 10**6  # In MHz
    ms_timeres = msmd.exposuretime(scan=scan_number)["value"]  # In second
    msmd.close()
    if freqres > 0:
        if freqres < ms_freqres:
            print(
                "Intended image spectral resolution is smaller than spectral resolution of the data. Hence, setting it to spectral resolution of the data.\n"
            )
            freqres = ms_freqres
        nchan = int(total_bw / freqres)
        if nchan > total_chans:
            nchan = total_chans
        if nchan < 1:
            nchan = 1
    if timeres > 0:
        if timeres < ms_timeres:
            print(
                "Intended image temporal resolution is smaller than temporal resolution of the data. Hence, setting it to temporal resolution of the data.\n"
            )
            timeres = ms_timeres
        ntime = int(total_time / timeres)
        if ntime > len(times):
            ntime = len(times)
        if ntime < 1:
            ntime = 1
    ################################
    # Applying calibration solutions
    ################################
    if len(calibrator_gaintables) > 0:
        print(
            "Applying calibrator solutions from: "
            + ",".join(calibrator_gaintables)
            + "\n"
        )
        applycal(
            vis=msname,
            gaintable=calibrator_gaintables,
            interp=["nearest"] * len(calibrator_gaintables),
            applymode="calflag",
            calwt=[True],
        )
        tb = table()
        tb.open(msname, nomodify=False)
        cor_data = tb.getcol("CORRECTED_DATA")
        tb.putcol("DATA", cor_data)
        tb.flush()
        tb.close()
    outputvis = split_move_sun(msname, scan_number)
    os.system("rm -rf " + msname)
    os.system("mv " + outputvis + " " + msname)
    if len(selfcaltables) > 0:
        print(
            "Applying self-calibration solutions from: "
            + ",".join(selfcaltables)
            + "\n"
        )
        applycal(
            vis=msname,
            gaintable=selfcaltables,
            interp=["nearest"] * len(selfcaltables),
            applymode="calonly",
            calwt=[True],
        )
    #########
    # Imaging
    #########
    prefix = msname.split(".ms")[0] + "_imaging"
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
        "-auto-threshold 5 -auto-mask 5.1",
        "-minuv-l " + str(minuv_l),
        "-nwlayers 1",
    ]
    if maxuv_l > minuv_l:
        wsclean_args.append("-maxuv-l " + str(maxuv_l))
    if freqres > 0:
        wsclean_args.append("-channels-out " + str(nchan))
    if timeres > 0:
        wsclean_args.append("-intervals-out " + str(ntime))
    if len(multiscale_scales) > 0:
        multiscale_scales = [str(i) for i in multiscale_scales]
        wsclean_args.append("-multiscale")
        wsclean_args.append("-multiscale-gain 0.1")
        wsclean_args.append("-multiscale-scales " + ",".join(multiscale_scales))
    if ncpu > 0:
        wsclean_args.append("-j " + str(ncpu))
    if mem > 0:
        wsclean_args.append("-abs-mem " + str(mem))
    fits_mask = msname.split(".ms")[0] + "_solar-mask.fits"
    if os.path.exists(fits_mask) == False:
        fits_mask = create_circular_mask(msname, cellsize, imsize, mask_radius=20)
    wsclean_args.append("-fits-mask " + fits_mask)
    wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " " + msname
    print("Start imaging ....\n")
    print(wsclean_cmd + "\n")
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    total_chunks = nchan * ntime
    if total_chunks > soft_limit:
        resource.setrlimit(resource.RLIMIT_NOFILE, (total_chunks, hard_limit))
    os.system(wsclean_cmd + " > tmp_wsclean")
    os.system(
        "rm -rf tmp_wsclean "
        + prefix
        + "*model.fits "
        + prefix
        + "*residual.fits "
        + prefix
        + "*psf.fits"
    )
    images = glob.glob(prefix + "*image.fits")
    final_image_list = []
    for image in images:
        header = fits.getheader(image)
        time = header["DATE-OBS"]
        freq = round(header["CRVAL3"] / 10**6, 2)
        t_str = "".join(time.split("T")[0].split("-")) + (
            "".join(time.split("T")[-1].split(":"))
        )
        new_name = "time_" + t_str + "_freq_" + str(freq)
        if "MFS" in image:
            new_name += "_MFS"
        new_name = new_name + ".fits"
        if os.path.exists(imagedir) == False:
            os.makedirs(imagedir)
        new_name = imagedir + "/" + new_name
        print("mv " + image + " " + new_name)
        os.system("mv " + image + " " + new_name)
        final_image_list.append(new_name)
    os.system("rm -rf " + msname + " " + msname + ".flagversions")
    if os.path.exists(fits_mask):
        os.system("rm -rf " + fits_mask)
    return final_image_list


def main():
    usage = "Perform solar imaging"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--msname",
        dest="msname",
        default=None,
        help="Name of measurement set",
        metavar="Measurement Set",
    )
    parser.add_option(
        "--imsize",
        dest="imsize",
        default=1280,
        help="Image size",
        metavar="Integer",
    )
    parser.add_option(
        "--cellsize",
        dest="cellsize",
        default=2.0,
        help="Cell size",
        metavar="Float",
    )
    parser.add_option(
        "--pol",
        dest="pol",
        default="I",
        help="Stokes parameters",
        metavar="String",
    )
    parser.add_option(
        "--spw",
        dest="spw",
        default="",
        help="Spectral window",
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
        "--imagedir",
        dest="imagedir",
        default="",
        help="Imaging directory",
        metavar="String",
    )
    parser.add_option(
        "--calibrator_gaintables",
        dest="calibrator_gaintables",
        default="",
        help="Calibrator gaintables",
        metavar="String",
    )
    parser.add_option(
        "--selfcaltables",
        dest="selfcaltables",
        default="",
        help="Calibrator gaintables",
        metavar="String",
    )
    parser.add_option(
        "--freqres",
        dest="freqres",
        default=-1.0,
        help="Frequency resolution of spectral in MHz",
        metavar="Float",
    )
    parser.add_option(
        "--timeres",
        dest="timeres",
        default=-1.0,
        help="Time resolution of spectral in MHz",
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
        "--multiscale_scales",
        dest="multiscale_scales",
        default="",
        help="Multiscale scales",
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
        "--minuv_l",
        dest="minuv_l",
        default=0.0,
        help="Minimum UV in lambda",
        metavar="Float",
    )
    parser.add_option(
        "--maxuv_l",
        dest="maxuv_l",
        default=-1,
        help="Maximum UV in lambda",
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
    (options, args) = parser.parse_args()
    if options.msname != None and os.path.exists(options.msname):
        if options.imagedir == "" or os.path.exists(options.imagedir) == False:
            imagedir = os.path.dirname(os.path.abspath(options.msname)) + "/imagedir"
            if os.path.exists(imagedir) == False:
                os.makedirs(imagedir)
        else:
            imagedir = options.imagedir
        if options.calibrator_gaintables != "":
            calibrator_gaintables = options.calibrator_gaintables.split(",")
        else:
            calibrator_gaintables = ""
        if options.selfcaltables != "":
            selfcaltables = options.selfcaltables.split(",")
        else:
            selfcaltables = ""
        if options.multiscale_scales != "":
            scales = options.multiscale_scales.split(",")
            scales = [int(i) for i in scales]
        else:
            scales = []
        images = applycal_and_perform_imaging(
            options.msname,
            int(options.imsize),
            float(options.cellsize),
            "0:" + str(options.spw),
            str(options.timerange),
            imagedir=imagedir,
            freqres=float(options.freqres),
            timeres=float(options.timeres),
            calibrator_gaintables=calibrator_gaintables,
            selfcaltables=selfcaltables,
            pol=str(options.pol),
            weight=str(options.weight),
            robust=float(options.robust),
            multiscale_scales=scales,
            minuv_l=float(options.minuv_l),
            maxuv_l=float(options.maxuv_l),
            ncpu=int(options.ncpu),
            mem=float(options.mem),
        )
        print("Total numbers of images made: " + str(len(images)))
        sub = "Imaging for ms: " + str(options.msname)
        msg = (
            "Imaging is done: SPW "
            + str(options.spw)
            + ", timerange: "
            + str(options.timerange)
            + "\nTotal images made: "
            + str(len(images))
        )
        """send_paircars_notification(
		'devojyoti96@gmail.com',
		sub,
		msg,
		attachments=[],
		)"""
        return 0
    else:
        print("Please provide correct measurement set.")
        os.system("rm -rf casa*log")
        return 1


if __name__ == "__main__":
    result = main()
    os.system("rm -rf casa*log")
    if result > 0:
        result = 1
    print("\n###################\nImaging is done.\n###################\n")
    os._exit(result)
