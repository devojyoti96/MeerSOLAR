import psutil
import numpy as np
import traceback
import copy
import glob
import os
from casatools import msmetadata, table
from .basic_utils import *
from .resource_utils import *
from .proc_manage_utils import *
from .ms_metadata import *
from .casatasks import *
from .flagging import *
from .calibration import *
from .imaging import *
from .image_utils import *
from .udocker_utils import *

def intensity_selfcal(
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

    try:
        ##################################
        # Setup wsclean params
        ##################################
        if ncpu < 1:
            ncpu = psutil.cpu_count()
        if mem < 0:
            mem = round(psutil.virtual_memory().available / (1024**3), 2)
        msname = msname.rstrip("/")
        if not use_previous_model:
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
        freqMHz = np.nanmean(freqs)
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
            ###################################################################
            # Spectral variation is kept fixed at 10% level.
            # Because selfcal is done along temporal axis, where variation matters most
            ###################################################################
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
            if fits_mask is not None and os.path.exists(fits_mask):
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
            sun_dia = calc_sun_dia(freqMHz)  # Sun diameter in arcmin
            sun_rad = sun_dia / 2
            multiscale_scales = calc_multiscale_scales(
                msname, 3, chan_number=chan_number, max_scale=sun_rad
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

            ###################################################################
            # Spectral variation is kept fixed at 10% level.
            # Because selfcal is done along temporal axis, where variation matters most
            ###################################################################
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
            msg = run_wsclean(wsclean_cmd, "solarwsclean", verbose=False)
            if msg != 0:
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
                                flagmanager(
                                    vis=msname, mode="restore", versionname=version
                                )
                                flagmanager(
                                    vis=msname, mode="delete", versionname=version
                                )
                        except BaseException:
                            pass

        #######################################################################
        # Final frequency averaged images for backup or calculating dynamic ranges
        #######################################################################
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
        print(e)
        traceback.print_exc()
        return 3, "", 0, 0, "", "", ""
