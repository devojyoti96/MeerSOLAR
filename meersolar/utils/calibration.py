from .all_depend import *
from .basic_utils import *
from .ms_metadata import *
from .imaging import *

#####################################
# Calibration related
#####################################


def merge_caltables(caltables, merged_caltable, append=False, keepcopy=False):
    """
    Merge multiple same type of caltables

    Parameters
    ----------
    caltables : list
        Caltable list
    merged_caltable : str
        Merged caltable name
    append : bool, optional
        Append with exisiting caltable
    keepcopy : bool, opitonal
        Keep input caltables or not

    Returns
    -------
    str
        Merged caltable
    """
    if not isinstance(caltables, list) or len(caltables) == 0:
        print("Please provide a list of caltable.")
        return
    if os.path.exists(merged_caltable) and append:
        pass
    else:
        if os.path.exists(merged_caltable):
            os.system("rm -rf " + merged_caltable)
        if keepcopy:
            os.system("cp -r " + caltables[0] + " " + merged_caltable)
        else:
            os.system("mv " + caltables[0] + " " + merged_caltable)
        caltables.remove(caltables[0])
    if len(caltables) > 0:
        tb = table()
        for caltable in caltables:
            if os.path.exists(caltable):
                tb.open(caltable)
                tb.copyrows(merged_caltable)
                tb.close()
                if not keepcopy:
                    os.system("rm -rf " + caltable)
    return merged_caltable


def determine_noise_diode_cal_scan(msname, scan):
    """
    Determine whether a calibrator scan is a noise-diode cal scan or not

    Parameters
    ----------
    msname : str
        Name of the measurement set
    scan : int
        Scan number

    Returns
    -------
    bool
        Whether it is noise-diode cal scan or not
    """

    def is_noisescan(msname, chan, scan):
        mstool = casamstool()
        mstool.open(msname)
        mstool.select({"antenna1": 1, "antenna2": 1, "scan_number": scan})
        mstool.selectchannel(nchan=1, width=1, start=chan)
        data = mstool.getdata("DATA", ifraxis=True)["data"][:, 0, 0, :]
        mstool.close()
        xx = np.abs(data[0, ...])
        yy = np.abs(data[-1, ...])
        even_xx = xx[1::2]
        odd_xx = xx[::2]
        minlen = min(len(even_xx), len(odd_xx))
        d_xx = even_xx[:minlen] - odd_xx[:minlen]
        even_yy = yy[1::2]
        odd_yy = yy[::2]
        d_yy = even_yy[:minlen] - odd_yy[:minlen]
        mean_d_xx = np.abs(np.nanmedian(d_xx))
        mean_d_yy = np.abs(np.nanmedian(d_yy))
        if mean_d_xx > 10 and mean_d_yy > 10:
            return True
        else:
            return False

    print(f"Check noise-diode cal for scan : {scan}")
    good_spw = get_good_chans(msname)
    chan = int(good_spw.split(";")[0].split(":")[-1].split("~")[0])
    return is_noisescan(msname, chan, scan)


def get_psf_size(msname, chan_number=-1):
    """
    Function to calculate PSF size in arcsec

    Parameters
    ----------
    msname : str
        Name of the measurement set
    chan_number : int, optional
        Channel number

    Returns
    -------
    float
            PSF size in arcsec
    """
    maxuv_m, maxuv_l = calc_maxuv(msname, chan_number=chan_number)
    psf = np.rad2deg(1.2 / maxuv_l) * 3600.0  # In arcsec
    return round(psf, 2)


def calc_bw_smearing_freqwidth(msname, full_FoV=False, FWHM=True):
    """
    Function to calculate spectral width to procude bandwidth smearing

    Parameters
    ----------
    msname : str
        Name of the measurement set
    full_FoV : bool, optional
        Consider smearing within solar disc or full FoV
    FWHM : bool, optional
        If using full FoV, consider upto FWHM or first null

    Returns
    -------
    float
        Spectral width in MHz
    """
    R = 0.9
    if full_FoV:
        fov = calc_field_of_view(msname, FWHM=FWHM)  # In arcsec
    else:
        fov = 35 * 60  # Size of the Sun, slightly larger is taken for U-band
    psf = get_psf_size(msname)
    tb = table()
    tb.open(f"{msname}/SPECTRAL_WINDOW")
    freq = float(tb.getcol("REF_FREQUENCY")[0]) / 10**6
    freqres = float(tb.getcol("CHAN_WIDTH")[0]) / 10**6
    tb.close()
    delta_nu = np.sqrt((1 / R**2) - 1) * (psf / fov) * freq
    delta_nu = ceil_to_multiple(delta_nu, freqres)
    return round(delta_nu, 2)


def calc_time_smearing_timewidth(msname, full_FoV=False, FWHM=True):
    """
    Calculate maximum time averaging to avoid time smearing over full FoV.

    Parameters
    ----------
    msname : str
        Measurement set name
    full_FoV : bool, optional
        Consider smearing within solar disc or full FoV
    FWHM : bool, optional
        If using full FoV, consider upto FWHM or first null

    Returns
    -------
    delta_t_max : float
        Maximum allowable time averaging in seconds.
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq_Hz = msmd.chanfreqs(0)[0]
    times = msmd.timesforspws(0)
    msmd.close()
    timeres = times[1] - times[0]
    c = 299792458.0  # speed of light in m/s
    omega_E = 7.2921159e-5  # Earth rotation rate in rad/s
    lam = c / freq_Hz  # wavelength in meters
    if full_FoV:
        fov = calc_field_of_view(msname, FWHM=FWHM)  # In arcsec
    else:
        fov = 35 * 60  # Size of the Sun, slightly larger is taken for U-band
    fov_deg = fov / 3600.0
    fov_rad = np.deg2rad(fov_deg)
    uv, uvlambda = calc_maxuv(msname)
    # Approximate maximum allowable time to avoid >10% amplitude loss
    delta_t_max = lam / (2 * np.pi * uv * omega_E * fov_rad)
    delta_t_max = ceil_to_multiple(delta_t_max, timeres)
    return round(delta_t_max, 2)


def max_time_solar_smearing(msname):
    """
    Max allowable time averaging to avoid solar motion smearing.

    Parameters
    ----------
    msname : str
        Measurement set name

    Returns
    -------
    t_max : float
        Maximum time averaging in seconds.
    """
    omega_sun = 2.5 / (60.0)  # solar apparent motion (2.5 arcsec/min to arcsec/sec)
    psf = get_psf_size(msname)
    t_max = 0.5 * (psf / omega_sun)  # seconds
    return t_max


def delaycal(msname="", caltable="", refant="", solint="inf", dry_run=False):
    """
    General delay calibration using CASA, not assuming any point source

    Parameters
    ----------
    msname : str, optional
        Measurement set
    caltable : str, optional
        Caltable name
    refant : str, optional
        Reference antenna
    solint : str, optional
        Solution interval

    Returns
    -------
    str
        Caltable name
    """
    from casatasks import bandpass, gaincal

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    try:
        warnings.filterwarnings("ignore")
        msname = msname.rstrip("/")
        mspath = os.path.dirname(os.path.abspath(msname))
        os.chdir(mspath)
        os.system("rm -rf " + caltable + "*")
        gaincal(
            vis=msname,
            caltable=caltable,
            refant=refant,
            gaintype="K",
            solint=solint,
            minsnr=1,
        )
        bandpass(
            vis=msname,
            caltable=caltable + ".tempbcal",
            refant=refant,
            solint=solint,
            minsnr=1,
        )
        tb = table()
        tb.open(caltable + ".tempbcal/SPECTRAL_WINDOW")
        freq = tb.getcol("CHAN_FREQ").flatten()
        tb.close()
        tb.open(caltable + ".tempbcal")
        gain = tb.getcol("CPARAM")
        flag = tb.getcol("FLAG")
        gain[flag] = np.nan
        tb.close()
        tb.open(caltable, nomodify=False)
        delay_gain = tb.getcol("FPARAM") * 0.0
        delay_flag = tb.getcol("FLAG")
        gain = np.nanmean(gain, axis=0)
        phase = np.angle(gain)
        for i in range(delay_gain.shape[0]):
            for j in range(delay_gain.shape[2]):
                try:
                    delay = np.polyfit(2 * np.pi * freq, phase[:, j], deg=1)[0] / (
                        10**-9
                    )  # Delay in nanosecond
                    if np.isnan(delay):
                        delay = 0.0
                    delay_gain[i, :, j] = delay
                except BaseException:
                    delay_gain[i, :, j] = 0.0
        tb.putcol("FPARAM", delay_gain)
        tb.putcol("FLAG", delay_flag)
        tb.flush()
        tb.close()
        os.system("rm -rf " + caltable + ".tempbcal")
        return caltable
    except Exception as e:
        traceback.print_exc()
        return
