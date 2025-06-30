# General imports
##################
import os, sys, glob, time, gc, tempfile, copy, warnings
import subprocess, contextlib, ctypes, platform
import traceback, resource, requests, threading, socket, argparse
os.environ["QT_OPENGL"] = "software"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import matplotlib, matplotlib.ticker as ticker, matplotlib.pyplot as plt 
import numpy as np, dask, psutil, logging, sunpy, tempfile, shutil
import astropy.units as u, string, secrets
from contextlib import contextmanager
from datetime import datetime as dt, timezone, timedelta
from scipy.ndimage import gaussian_filter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

#General,CASA,dask imports
##########################
from casatools import msmetadata, ms as casamstool, table, agentflagger
from casatasks import (
    casalog,
    importfits,
    listpartition,
)
from casatasks import casalog
from dask.distributed import Client, LocalCluster
from dask import delayed, compute, config

# Astropy imports
##################
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_sun
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, PowerStretch, LogStretch
from astroquery.jplhorizons import Horizons

# Sunpy imports
################
from sunpy.map import Map
from sunpy.coordinates import frames, sun
from sunpy import timeseries as ts
from sunpy.net import Fido, attrs as a
from sunpy.coordinates import SphericalScreen
from sunpy.map.maputils import all_coordinates_from_map
from sunpy.time import parse_time

# Matplotlib imports
#####################
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.colors import ListedColormap
from matplotlib import cm

try:
    logfile = casalog.logfile()
    os.system("rm -rf " + logfile)
except:
    pass


def get_datadir():
    """
    Get package data directory
    """
    from importlib.resources import files

    datadir_path = str(files("meersolar").joinpath("data"))
    os.makedirs(datadir_path, exist_ok=True)
    os.makedirs(f"{datadir_path}/pids", exist_ok=True)
    return datadir_path
    
def get_meersolar_cachedir():
    homedir=os.environ.get("HOME")
    if homedir is None:
        homedir=os.path.expanduser("~")
    username = os.getlogin()
    meersolar_cachedir=f"{homedir}/.meersolar"
    os.makedirs(meersolar_cachedir,exist_ok=True)
    os.makedirs(f"{meersolar_cachedir}/pids",exist_ok=True)
    return meersolar_cachedir


datadir = get_datadir()
udocker_dir = datadir + "/udocker"
os.environ["UDOCKER_DIR"] = udocker_dir
os.environ["UDOCKER_TARBALL"] = datadir + "/udocker-englib-1.2.11.tar.gz"
POSIX_FADV_DONTNEED = 4
libc = ctypes.CDLL("libc.so.6")


class SmartDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        # Don't show default for boolean flags
        if isinstance(action, argparse._StoreTrueAction) or isinstance(
            action, argparse._StoreFalseAction
        ):
            return action.help
        return super()._get_help_string(action)


def init_udocker():
    os.system("udocker install")


@contextmanager
def suppress_casa_output():
    with open(os.devnull, "w") as fnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(fnull.fileno(), 1)
        os.dup2(fnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)


def clean_shutdown(observer):
    if observer:
        observer.stop()
        observer.join(timeout=5)


#####################################
# Resource management
#####################################
def drop_file_cache(filepath, verbose=False):
    """
    Advise the OS to drop the given file from the page cache.
    Safe, per-file, no sudo required.
    """
    if platform.system() != "Linux":
        raise NotImplementedError("drop_file_cache is only supported on Linux")
    try:
        if not os.path.isfile(filepath):
            return
        fd = os.open(filepath, os.O_RDONLY)
        result = libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
        os.close(fd)
        if verbose:
            if result == 0:
                print(f"[cache drop] Released: {filepath}")
            else:
                print(f"[cache drop] Failed ({result}) for: {filepath}")
    except Exception as e:
        if verbose:
            print(f"[cache drop] Error for {filepath}: {e}")
            traceback.print_exc()


def drop_cache(path, verbose=False):
    """
    Drop file cache for a file or all files under a directory.

    Parameters
    ----------
    path : str
        File or directory path
    """
    if platform.system() != "Linux":
        raise NotImplementedError("drop_file_cache is only supported on Linux")
    if os.path.isfile(path):
        drop_file_cache(path, verbose=verbose)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                full_path = os.path.join(root, f)
                drop_file_cache(full_path, verbose=verbose)
    else:
        if verbose:
            print(f"[cache drop] Path does not exist or is not valid: {path}")


def has_space(path, required_gb):
    try:
        stat = shutil.disk_usage(path)
        return (stat.free / 1e9) >= required_gb
    except:
        return False


@contextmanager
def shm_or_tmp(required_gb, workdir, prefix="meersolar_", verbose=False):
    """
    Create a temporary working directory:
    1. Try /dev/shm if it has required space
    2. Else TMPDIR if set and has space
    4. Else work directory
    Temporarily sets TMPDIR to the selected path during the context.
    Cleans up after use.

    Parameters
    ----------
    required_gb : float
        Required disk space in GB
    workdir : str
        Fall back work directory
    prefix : str, optional
        Temp directory prefix
    verbose : bool, optional
        Verbose
    """

    def has_space(path, required_gb):
        try:
            stat = shutil.disk_usage(path)
            return (
                stat.free / 1e9
            ) >= 2 * required_gb  # At-least two times of required disk space
        except:
            return False

    candidates = []
    if has_space("/dev/shm", required_gb):
        candidates.append("/dev/shm")
    tmpdir_env = os.environ.get("TMPDIR")
    if tmpdir_env is not None and has_space(tmpdir_env, required_gb):
        candidates.append(tmpdir_env)
    candidates.append(os.getcwd())
    for i in range(len(candidates)):
        base_dir = candidates[i]
        try:
            temp_dir = tempfile.mkdtemp(dir=base_dir, prefix=prefix)
            if verbose:
                if i == 0:
                    print("Using RAM")
                elif i == 1:
                    print("Using {os.environ.get('TMPDIR')}")
                else:
                    print("Using {os.getcwd()}")
            break
        except Exception as e:
            print(f"[shm_or_tmp] Failed to create temp dir in {base_dir}: {e}")
    else:
        raise RuntimeError(
            "Could not create a temporary directory in any fallback location."
        )
    # Override TMPDIR
    old_tmpdir = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = temp_dir
    try:
        yield temp_dir
    finally:
        # Restore TMPDIR
        if old_tmpdir is not None:
            os.environ["TMPDIR"] = old_tmpdir
        else:
            os.environ.pop("TMPDIR", None)
        # Clean up the temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"[cleanup] Warning: could not delete {temp_dir}: {e}")


@contextmanager
def tmp_with_cache_rel(required_gb, workdir, prefix="meersolar_", verbose=False):
    """
    Combined context manager:
    - Uses shm_or_tmp() for workspace
    - Drops kernel page cache for all files in that directory on exit
    
    Parameters
    ----------
    required_gb : float
        Required disk space in GB
    workdir : str
        Fall back work directory
    prefix : str, optional
        Temp directory prefix
    verbose : bool, optional
        Verbose
    
    """
    with shm_or_tmp(required_gb, workdir, prefix=prefix, verbose=verbose) as tempdir:
        try:
            yield tempdir
        finally:
            if platform.system() == "Linux":
                drop_cache(tempdir)


def limit_threads(n_threads=-1):
    """
    Limit number of threads usuage

    Parameters
    ----------
    n_threads : int, optional
        Number of threads
    """
    if n_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)


##################################


def generate_password(length=6):
    """
    Generate secure 6-character password with letters, digits, and symbols
    """
    chars = string.ascii_letters + string.digits + "@#$&*"
    return "".join(secrets.choice(chars) for _ in range(length))


def get_remote_logger_link():
    meersolar_cachedir = get_meersolar_cachedir()
    username = os.getlogin()
    link_file = os.path.join(meersolar_cachedir, f"remotelink_{username}.txt")
    for _ in range(5):
        try:
            if os.path.isfile(link_file):
                with open(link_file, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                if lines:
                    remote_link = lines[0]
                    if remote_link:
                        try:
                            res = requests.get(remote_link, timeout=2)
                            if res.status_code == 200:
                                return remote_link
                        except Exception:
                            pass
        except Exception:
            pass
        time.sleep(2)
    return ""

def get_emails():
    meersolar_cachedir = get_meersolar_cachedir()
    username = os.getlogin()
    email_file = os.path.join(meersolar_cachedir, f"emails_{username}.txt")
    try:
        with open(email_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return ""
    if not lines:
        return ""
    else:
        return lines[0]


class RemoteLogger(logging.Handler):
    """
    Remote logging handler for posting log messages to a web endpoint.
    """

    def __init__(
        self, job_id="default", log_id="run_default", remote_link="", password=""
    ):
        super().__init__()
        self.job_id = job_id
        self.log_id = log_id
        self.password = password
        self.remote_link = remote_link

    def emit(self, record):
        msg = self.format(record)
        try:
            requests.post(
                f"{self.remote_link}/api/log",
                json={
                    "job_id": self.job_id,
                    "log_id": self.log_id,
                    "message": msg,
                    "password": self.password,
                    "first": False,
                },
                timeout=2,
            )
        except Exception as e:
            pass  # Fail silently to avoid interrupting the main app


class LogTailHandler(FileSystemEventHandler):
    """
    Continuous logging
    """

    def __init__(self, logfile, logger):
        self.logfile = logfile
        self.logger = logger
        self._position = os.path.getsize(logfile) if os.path.exists(logfile) else 0

    def on_modified(self, event):
        if event.src_path == self.logfile:
            try:
                with open(self.logfile, "r") as f:
                    f.seek(self._position)
                    lines = f.readlines()
                    self._position = f.tell()
                for line in lines:
                    if line != "" and line != " " and line != "\n":
                        self.logger.info(line.strip())
            except Exception:
                pass


def ping_logger(jobid, remote_jobid, stop_event, remote_link=""):
    """Ping a job-specific keep-alive endpoint periodically until stop_event is set."""
    pid = os.getpid()
    meersolar_cachedir = get_meersolar_cachedir()
    save_pid(pid, f"{meersolar_cachedir}/pids/pids_{jobid}.txt")
    interval = 10  # 10 min interval
    if remote_link != "":
        url = f"{remote_link}/api/ping/{remote_jobid}"
        while not stop_event.is_set():
            try:
                print(
                    f"[ping_logger] Ping sent for job {remote_jobid} at {dt.now().isoformat()}"
                )
                res = requests.post(url, timeout=2)
            except Exception as e:
                pass
            stop_event.wait(interval)


def create_logger(logname, logfile, verbose=False):
    """
    Create logger.

    Parameters
    ----------
    logname : str
        Name of the log
    workdir : str, optional
        Name of the working directory
    verbose : bool, optional
        Verbose output or not
    logfile : str, optional
        Log file name

    Returns
    -------
    logger
        Python logging object
    str
        Log file name
    """
    if os.path.exists(logfile):
        os.system("rm -rf " + logfile)
    formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    if verbose == True:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)
    filehandle = logging.FileHandler(logfile)
    filehandle.setFormatter(formatter)
    logger.addHandler(filehandle)
    logger.propagate = False
    logger.info("Log file : " + logfile + "\n")
    return logger, logfile


def get_logid(logfile):
    """
    Get log id for remote logger from logfile name
    """
    name = os.path.basename(logfile)
    logmap = {
        "apply_basiccal.log": "Applying basic calibration solutions",
        "apply_pbcor.log": "Applying primary beam corrections",
        "apply_selfcal.log": "Applying self-calibration solutions",
        "basic_cal.log": "Basic calibration",
        "cor_sidereal_selfcals.log": "Correction of sidereal motion before self-calibration",
        "cor_sidereal_targets.log": "Correction of sidereal motion for target scans",
        "flagging_cal_calibrator.log": "Basic flagging",
        "modeling_calibrator.log": "Simulating visibilities of calibrators",
        "split_targets.log": "Spliting target scans",
        "split_selfcals.log": "Spliting for self-calibration",
        "selfcal_targets.mainlog": "All self-calibrations main log",
        "imaging_targets.mainlog": "All imaging main log",
        "selfcal_targets.log": "All self-calibrations",
        "imaging_targets.log": "All imaging",
        "noise_cal.log": "Flux calibration using noise-diode",
        "partition_cal.log": "Partioning for basic calibration",
        "ds_targets.log": "Making dynamic spectra",
    }

    if name in logmap:
        return logmap[name]
    elif "selfcals_scan_" in name:
        name = name.rstrip("_selfcal.log")
        scan = name.split("scan_")[-1].split("_spw")[0]
        spw = name.split("spw_")[-1].split("_selfcal")[0]
        return f"Self-calibration for: Scan : {scan}, Spectral window: {spw}"
    elif "imaging_targets_scan_" in name:
        name = name.rstrip(".log")
        scan = name.split("scan_")[-1].split("_spw")[0]
        spw = name.split("spw_")[-1].split("_selfcal")[0]
        return f"Imaging for: Scan : {scan}, Spectral window: {spw}"
    else:
        return name


def init_logger(logname, logfile, jobname="", password=""):
    """
    Initialize a local + optional remote logger with watchdog-based tailing.

    Parameters
    ----------
    logname : str
        Logger name.
    logfile : str
        Path to the local logfile to also monitor.
    jobname : str, optional
        Remote logger job ID.
    password : str
        Password used for remote authentication.

    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
    """
    timeout = 30
    waited = 0
    while True:
        if os.path.exists(logfile) == False:
            time.sleep(1)
            waited += 1
        elif waited >= timeout:
            return
        else:
            break
    logger = logging.getLogger(logname)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s-%(message)s", "%Y-%m-%dT%H:%M:%S"
    )
    remote_link = get_remote_logger_link()
    if remote_link != "":
        if jobname:
            job_id = jobname
            log_id = get_logid(logfile)
            remote_handler = RemoteLogger(
                job_id=job_id, log_id=log_id, remote_link=remote_link, password=password
            )
            remote_handler.setFormatter(formatter)
            logger.addHandler(remote_handler)

            try:
                requests.post(
                    f"{remote_link}/api/log",
                    json={
                        "job_id": job_id,
                        "log_id": log_id,
                        "message": "Job starting...",
                        "password": password,
                        "first": True,
                    },
                    timeout=2,
                )
            except Exception:
                pass
        if os.path.exists(logfile):
            event_handler = LogTailHandler(logfile, logger)
            observer = Observer()
            observer.schedule(
                event_handler, path=os.path.dirname(logfile), recursive=False
            )
            observer.start()
            return observer
        else:
            return
    else:
        return


def flag_outside_uvrange(vis, uvrange, n_threads=-1, flagbackup=True):
    """
    Flag outside the given uv range

    Parameters
    ----------
    vis : str
        Measurement set name
    uvrange : str
        UV-range
    n_threads : int, optional
        Number of OpenMP threads to use
    flagbackup : bool, optional
        Flag backup
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    if "lambda" in uvrange:
        islambda = True
        uvrange = uvrange.replace("lambda", "")
    else:
        islambda = False
    if "~" in uvrange:
        low, high = uvrange.split("~")
        if islambda:
            low = f"{low}lambda"
            high = f"{high}lambda"
        cmds = [
            {"mode": "manual", "uvrange": f"<{low}", "flagbackup": flagbackup},
            {"mode": "manual", "uvrange": f">{high}", "flagbackup": flagbackup},
        ]
    elif ">" in uvrange:
        low = uvrange.split(">")[-1]
        if islambda:
            low = f"{low}lambda"
        cmds = [
            {"mode": "manual", "uvrange": f"<{low}", "flagbackup": flagbackup},
        ]
    elif "<" in uvrange:
        if islambda:
            high = f"{high}lambda"
        cmds = [
            {"mode": "manual", "uvrange": f">{high}", "flagbackup": flagbackup},
        ]
    else:
        cmds = []
    if len(cmds) > 0:
        for cmd in cmds:
            print(f"Flagging command: {cmd}")
            flagdata(vis=vis, **cmd)
    return


def make_ds_plot(dsfiles, plot_file=None, showgui=False):
    """
    Make dynamic spectrum plot

    Parameters
    ----------
    dsfile : list
        DS files list
    plot_file : str, optional
        Plot file name to save the plot
    showgui : bool, optional
        Show GUI

    Returns
    -------
    str
        Plot name
    """
    if showgui:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 18})
    for i, dsfile in enumerate(dsfiles):
        freqs_i, times_i, timestamps_i, data_i = np.load(dsfile, allow_pickle=True)
        if i == 0:
            freqs = freqs_i
            times = times_i
            timestamps = timestamps_i
            data = data_i
        else:
            gapsize = int(
                (np.nanmin(times_i) - np.nanmax(times)) / (times[1] - times[0])
            )
            if gapsize < 10:
                last_time_median = np.nanmedian(data[:, -1], axis=0)
                new_time_median = np.nanmedian(data_i[:, 0], axis=0)
                data_i = (data_i / new_time_median) * last_time_median
            # Insert vertical NaN gap (1 column wide)
            gap = np.full((data.shape[0], gapsize), np.nan)
            data = np.concatenate([data, gap, data_i], axis=1)
            # Insert dummy time and timestamp
            times = np.append(times, np.nan)
            timestamps = np.append(timestamps, "GAP")
            # Append new values
            times = np.append(times, times_i)
            timestamps = np.append(timestamps, timestamps_i)
            # (Optional) Check or merge freqs if needed — assuming same across files
    # Normalize by median bandshape
    median_bandshape = np.nanmedian(data, axis=-1)
    pos = np.where(np.isnan(median_bandshape) == False)[0]
    data /= median_bandshape[:, None]
    data = data[min(pos) : max(pos), :]
    freqs = freqs[min(pos) : max(pos)]
    temp_times = times[np.isnan(times) == False]
    maxtimepos = np.argmax(temp_times)
    mintimepos = np.argmin(temp_times)
    datestamp = f"{timestamps[mintimepos].split('T')[0]}"
    tstart = f"{timestamps[mintimepos].split('T')[0]} {':'.join(timestamps[mintimepos].split('T')[-1].split(':')[:2])}"
    tend = f"{timestamps[maxtimepos].split('T')[0]} {':'.join(timestamps[maxtimepos].split('T')[-1].split(':')[:2])}"
    print(f"Time range : {tstart}~{tend}")
    results = Fido.search(
        a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("avg1m")
    )
    files = Fido.fetch(results, path=os.path.dirname(dsfiles[0]), overwrite=False)
    goes_tseries = ts.TimeSeries(files, concatenate=True)
    goes_tseries = goes_tseries.truncate(tstart, tend)
    timeseries = np.nanmean(data, axis=0)
    # Normalization
    data_std = np.nanstd(data)
    data_median = np.nanmedian(data)
    norm = ImageNormalize(
        data,
        stretch=LogStretch(1),
        vmin=0.99 * np.nanmin(data),
        vmax=0.99 * np.nanmax(data),
    )
    # Create figure and GridSpec layout
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(nrows=3, ncols=2, width_ratios=[1, 0.03], height_ratios=[4, 1.5, 2])
    # Axes
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_ts = fig.add_subplot(gs[1, 0])
    ax_goes = fig.add_subplot(gs[2, 0])
    cax = fig.add_subplot(gs[:, 1])  # colorbar spans both rows
    # Plot dynamic spectrum
    im = ax_spec.imshow(data, aspect="auto", origin="lower", norm=norm, cmap="magma")
    ax_spec.set_ylabel("Frequency (MHz)")
    ax_spec.set_xticklabels([])  # Remove x-axis labels from top plot
    # Y-ticks
    yticks = ax_spec.get_yticks()
    yticks = yticks[(yticks >= 0) & (yticks < len(freqs))]
    ax_spec.set_yticks(yticks)
    ax_spec.set_yticklabels([f"{freqs[int(i)]:.1f}" for i in yticks])
    # Plot time series
    ax_ts.plot(timeseries)
    ax_ts.set_xlim(0, len(timeseries) - 1)
    ax_ts.set_ylabel("Mean \n flux density")
    goes_tseries.plot(axes=ax_goes)
    goes_times = goes_tseries.time
    times_dt = goes_times.to_datetime()
    ax_goes.set_xlim(times_dt[0], times_dt[-1])
    ax_goes.set_ylabel(r"Flux ($\frac{W}{m^2}$)")
    ax_goes.legend(ncol=2, loc="upper right")
    ax_goes.set_title("GOES light curve", fontsize=14)
    ax_ts.set_title("MeerKAT light curve", fontsize=14)
    ax_spec.set_title("MeerKAT dynamic spectrum", fontsize=14)
    ax_goes.set_xlabel("Time (UTC)")
    # Format x-ticks
    ax_ts.set_xticks([])
    ax_ts.set_xticklabels([])
    # Colorbar
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Flux density (arb. unit)")
    plt.tight_layout()
    # Save or show
    if plot_file:
        plt.savefig(plot_file, bbox_inches="tight")
        print(f"Plot saved: {plot_file}")
    if showgui:
        plt.show()
        plt.close(fig)
        plt.close("all")
    else:
        plt.close(fig)
    return plot_file


def make_solar_DS(
    msname,
    workdir,
    ds_file_name="",
    extension="png",
    target_scans=[],
    scans=[],
    merge_scan=False,
    showgui=False,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Make solar dynamic spectrum and plots

    Parameters
    ----------
    msname : str
        Measurement set name'
    workdir : str
        Work directory
    ds_file_name : str, optional
        DS file name prefix
    extension : str, optional
        Image file extension
    target_scans : list, optional
        Target scans
    scans : list, optional
        Scan list
    merge_scan : bool, optional
        Merge scans in one plot or not
    showgui : bool, optional
        Show GUI
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    """
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(f"{workdir}/dynamic_spectra", exist_ok=True)
    print("##############################################")
    print(f"Start making dynamic spectra for ms: {msname}")
    print("##############################################")
    if len(target_scans) > 0:
        temp_target_scans = []
        for s in target_scans:
            temp_target_scans.append(int(s))
        target_scans = temp_target_scans

    ##############################
    # Extract dynamic spectrum
    ##############################
    def make_ds_file_per_scan(msname, save_file, scan, datacolumn):
        if os.path.exists(f"{save_file}.npy") == False:
            mstool = casamstool()
            try:
                all_data = []
                for ant in range(5):
                    print(f"Extracting data for antenna :{ant}, scan: {scan}")
                    mstool.open(msname)
                    mstool.selectpolarization(["I"])
                    mstool.select(
                        {"antenna1": ant, "antenna2": ant, "scan_number": int(scan)}
                    )
                    data_dic = mstool.getdata(datacolumn)
                    mstool.close()
                    if datacolumn == "CORRECTED_DATA":
                        data = np.abs(data_dic["corrected_data"][0, ...])
                    else:
                        data = np.abs(data_dic["data"][0, ...])
                    del data_dic
                    m = np.nanmedian(data, axis=1)
                    data = data / m[:, None]
                    all_data.append(data)
                    del data
            except Exception as e:
                print("Auto-corrrelations are not present. Using short baselines.")
                count = 0
                all_data = []
                while count <= 5:
                    for i in range(5):
                        for j in range(5):
                            if i != j:
                                print(
                                    f"Extracting data for antennas :{i} and {j}, scan: {scan}"
                                )
                                mstool.open(msname)
                                mstool.selectpolarization(["I"])
                                mstool.select(
                                    {
                                        "antenna1": i,
                                        "antenna2": j,
                                        "scan_number": int(scan),
                                    }
                                )
                                data_dic = mstool.getdata(datacolumn)
                                mstool.close()
                                if datacolumn == "CORRECTED_DATA":
                                    data = np.abs(data_dic["corrected_data"][0, ...])
                                else:
                                    data = np.abs(data_dic["data"][0, ...])
                                del data_dic
                                m = np.nanmedian(data, axis=1)
                                data = data / m[:, None]
                                all_data.append(data)
                                del data
                                count += 1
            all_data = np.array(all_data)
            data = np.nanmedian(all_data, axis=0)
            bad_chans = get_bad_chans(msname)
            bad_chans = bad_chans.replace("0:", "").split(";")
            for bad_chan in bad_chans:
                s = int(bad_chan.split("~")[0])
                e = int(bad_chan.split("~")[-1]) + 1
                data[s:e, :] = np.nan
            msmd = msmetadata()
            msmd.open(msname)
            freqs = msmd.chanfreqs(0, unit="MHz")
            times = msmd.timesforscans(int(scan))
            timestamps = [mjdsec_to_timestamp(mjdsec, str_format=0) for mjdsec in times]
            msmd.close()
            np.save(
                f"{save_file}.npy",
                np.array([freqs, times, timestamps, data], dtype="object"),
            )
            del msmd, mstool, data
        return f"{save_file}.npy"

    ##################################
    # Making and ploting
    ##################################
    if len(scans) == 0:
        scans, cal_scans, f_scans, g_scans, p_scans = get_cal_target_scans(msname)
    valid_scans = get_valid_scans(msname)
    final_scans = []
    scan_size_list = []
    msmd = msmetadata()
    mstool = casamstool()
    for scan in scans:
        if scan in valid_scans:
            if len(target_scans) == 0 or (
                len(target_scans) > 0 and int(scan) in target_scans
            ):
                final_scans.append(int(scan))
                msmd.open(msname)
                nchan = msmd.nchan(0)
                nant = msmd.nantennas()
                msmd.close()
                mstool.open(msname)
                mstool.select({"scan_number": int(scan)})
                nrow = mstool.nrow(True)
                mstool.close()
                nbaselines = int(nant + (nant * (nant - 1) / 2))
                scan_size = (5 * (nrow / nbaselines) * 16) / (1024**3)
                scan_size_list.append(scan_size)
    if len(final_scans) == 0:
        print("No scans to make dynamic spectra.")
        return
    del scans
    scans = sorted(final_scans)
    print(f"Scans: {scans}")
    msname = msname.rstrip("/")
    if ds_file_name == "":
        ds_file_name = os.path.basename(msname).split(".ms")[0] + "_DS"
    hascor = check_datacolumn_valid(msname, datacolumn="CORRECTED_DATA")
    if hascor:
        datacolumn = "CORRECTED_DATA"
    else:
        datacolumn = "DATA"
    mspath = os.path.dirname(msname)
    mem_limit = max(scan_size_list)
    dask_client, dask_cluster, n_jobs, n_threads, mem_limit = get_dask_client(
        len(scans),
        dask_dir=workdir,
        cpu_frac=cpu_frac,
        mem_frac=mem_frac,
        min_mem_per_job=mem_limit / 0.6,
    )
    tasks = []
    for scan in scans:
        tasks.append(
            delayed(make_ds_file_per_scan)(
                msname,
                f"{workdir}/dynamic_spectra/{ds_file_name}_scan_{scan}",
                scan,
                datacolumn,
            )
        )
    compute(*tasks)
    dask_client.close()
    dask_cluster.close()
    ds_files = [
        f"{workdir}/dynamic_spectra/{ds_file_name}_scan_{scan}.npy" for scan in scans
    ]
    print(f"DS files: {ds_files}")
    if merge_scan == False:
        plots = []
        for dsfile in ds_files:
            plot_file = make_ds_plot(
                [dsfile],
                plot_file=dsfile.replace(".npy", f".{extension}"),
                showgui=showgui,
            )
            plots.append(plot_file)
    else:
        plot_file = make_ds_plot(
            ds_files,
            plot_file=f"{workdir}/dynamic_spectra/{ds_file_name}.{extension}",
            showgui=showgui,
        )
    gc.collect()
    goes_files = glob.glob(f"{workdir}/dynamic_spectra/sci*.nc")
    for f in goes_files:
        os.system(f"rm -rf {f}")
    os.system(f"rm -rf {workdir}/dask-scratch-space {workdir}/tmp")
    return


def plot_goes_full_timeseries(
    msname, workdir, plot_file_prefix=None, extension="png", showgui=False
):
    """
    Plot GOES full time series on the day of observation

    Parameters
    ----------
    msname : str
        Measurement set
    workdir : str
        Work directory
    plot_file_prefix : str, optional
        Plot file name prefix
    extension : str, optional
        Save file extension
    showgui : bool, optional
        Show GUI

    Returns
    -------
    str
        Plot file name
    """
    os.makedirs(workdir, exist_ok=True)
    if showgui:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 14})
    scans, cal_scans, f_scans, g_scans, p_scans = get_cal_target_scans(msname)
    valid_scans = get_valid_scans(msname)
    filtered_scans = []
    for scan in scans:
        if scan in valid_scans:
            filtered_scans.append(scan)
    msmd = msmetadata()
    msmd.open(msname)
    tstart_mjd = min(msmd.timesforscan(int(min(filtered_scans))))
    tend_mjd = max(msmd.timesforscan(int(max(filtered_scans))))
    msmd.close()
    tstart = mjdsec_to_timestamp(tstart_mjd, str_format=2)
    tend = mjdsec_to_timestamp(tend_mjd, str_format=2)
    print(f"Time range: {tstart}~{tend}")
    results = Fido.search(
        a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("avg1m")
    )
    files = Fido.fetch(results, path=workdir, overwrite=False)
    goes_tseries = ts.TimeSeries(files, concatenate=True)
    for f in files:
        os.system(f"rm -rf {f}")
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    goes_tseries.plot(axes=ax)
    times = goes_tseries.time
    times_dt = times.to_datetime()
    ax.axvspan(tstart, tend, alpha=0.2)
    ax.set_xlim(times_dt[0], times_dt[-1])
    plt.tight_layout()
    # Save or show
    if plot_file_prefix:
        plot_file = f"{workdir}/{plot_file_prefix}.{extension}"
        plt.savefig(plot_file, bbox_inches="tight")
        print(f"Plot saved: {plot_file}")
    else:
        plot_file = None
    if showgui:
        plt.show()
        plt.close(fig)
        plt.close("all")
    else:
        plt.close(fig)
    return plot_file


def get_suvi_map(obs_date, obs_time, workdir, wavelength=195):
    """
    Get GOES SUVI map

    Parameters
    ----------
    obs_date : str
        Observation date in yyyy-mm-dd format
    obs_time : str
        Observation time in hh:mm format
    workdir : str
        Work directory
    wavelength : float, optional
        Wavelength, options: 94, 131, 171, 195, 284, 304 Å

    Returns
    -------
    sunpy.map
        Sunpy SUVIMap
    """
    warnings.filterwarnings("ignore", message="This download has been started in a thread which is not the main thread")
    logging.getLogger('sunpy').setLevel(logging.ERROR)
    os.makedirs(workdir, exist_ok=True)
    start_time = dt.fromisoformat(f"{obs_date}T{obs_time}")
    t_start = (start_time - timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M")
    t_end = (start_time + timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M")
    time = a.Time(t_start, t_end)
    instrument = a.Instrument("suvi")
    wavelength = a.Wavelength(wavelength * u.angstrom)
    results = Fido.search(time, instrument, wavelength, a.Level(2))
    downloaded_files = Fido.fetch(results, path=workdir, progress=False)
    obs_times = []
    for image in downloaded_files:
        suvimap = Map(image)
        dateobs = suvimap.meta["date-obs"].split(".")[0]
        obs_times.append(dateobs)
    times_dt = [dt.strptime(t, "%Y-%m-%dT%H:%M:%S") for t in obs_times]
    closest_time = min(times_dt, key=lambda t: abs(t - start_time))
    pos = times_dt.index(closest_time)
    closest_time_str = closest_time.strftime("%Y-%m-%dT%H:%M")
    final_image = downloaded_files[pos]
    suvi_map = Map(final_image)
    for f in downloaded_files:
        os.system(f"rm -rf {f}")
    return suvi_map


def enhance_offlimb(sunpy_map, do_sharpen=True):
    """
    Enhance off-disk emission

    Parameters
    ----------
    sunpy_map : sunpy.map
        Sunpy map
    do_sharpen : bool, optional
        Sharpen images

    Returns
    -------
    sunpy.map
        Off-disk enhanced emission
    """
    logging.getLogger('sunpy').setLevel(logging.ERROR)
    hpc_coords = all_coordinates_from_map(sunpy_map)
    r = np.sqrt(hpc_coords.Tx**2 + hpc_coords.Ty**2) / sunpy_map.rsun_obs
    rsun_step_size = 0.01
    rsun_array = np.arange(1, r.max(), rsun_step_size)
    y = np.array(
        [
            sunpy_map.data[(r > this_r) * (r < this_r + rsun_step_size)].mean()
            for this_r in rsun_array
        ]
    )
    pos = np.where(y < 10e-3)[0][0]
    r_lim = round(rsun_array[pos], 2)
    params = np.polyfit(
        rsun_array[rsun_array < r_lim], np.log(y[rsun_array < r_lim]), 1
    )
    scale_factor = np.exp((r - 1) * -params[0])
    scale_factor[r < 1] = 1
    if do_sharpen:
        blurred = gaussian_filter(sunpy_map.data, sigma=3)
        data = sunpy_map.data + (sunpy_map.data - blurred)
    else:
        data = sunpy_map.data
    scaled_map = sunpy.map.Map(data * scale_factor, sunpy_map.meta)
    scaled_map.plot_settings["norm"] = ImageNormalize(stretch=LogStretch(10))
    return scaled_map


def make_meer_overlay(
    meerkat_image,
    suvi_wavelength=195,
    plot_file_prefix=None,
    plot_meer_colormap=True,
    enhance_offdisk=True,
    contour_levels=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
    do_sharpen_suvi=True,
    xlim=[-1600, 1600],
    ylim=[-1600, 1600],
    extensions=["png"],
    outdirs=[],
    showgui=False,
):
    """
    Make overlay of MeerKAT image on GOES SUVI image

    Parameters
    ----------
    meerkat_image : str
        MeerKAT image
    suvi_wavelength : float, optional
        GOES SUVI wavelength, options: 94, 131, 171, 195, 284, 304 Å
    plot_file_prefix : str, optional
        Plot file prefix name
    plot_meer_colormap : bool, optional
        Plot MeerKAT map colormap
    enhance_offdisk : bool, optional
        Enhance off-disk emission
    contour_levels : list, optional
        Contour levels in fraction of peak
    do_sharpen_suvi : bool, optional
        Do sharpen SUVI images
    xlim : list, optional
        X-axis limit in arcsec
    tlim : list, optional
        Y-axis limit in arcsec
    extensions : list, optional
        Image file extensions
    outdirs : list, optional
        Output directories for each extensions
    showgui : bool, optional
        Show GUI

    Returns
    -------
    list
        Plot file names
    """
    @delayed
    def reproject_map(smap, target_header):
        with SphericalScreen(smap.observer_coordinate):
            return smap.reproject_to(target_header)
    logging.getLogger('sunpy').setLevel(logging.ERROR)
    logging.getLogger('reproject.common').setLevel(logging.ERROR)
    if showgui:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")
    workdir=os.path.dirname(os.path.abspath(meerkat_image))
    meermap = get_meermap(meerkat_image)
    obs_datetime = fits.getheader(meerkat_image)["DATE-OBS"]
    obs_date = obs_datetime.split("T")[0]
    obs_time = ":".join(obs_datetime.split("T")[-1].split(":")[:2])
    suvi_map = get_suvi_map(obs_date, obs_time, workdir, wavelength=suvi_wavelength)
    if enhance_offdisk:
        suvi_map = enhance_offlimb(suvi_map, do_sharpen=do_sharpen_suvi)
    projected_coord = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        obstime=suvi_map.observer_coordinate.obstime,
        frame="helioprojective",
        observer=suvi_map.observer_coordinate,
        rsun=suvi_map.coordinate_frame.rsun,
    )
    projected_header = sunpy.map.make_fitswcs_header(
        suvi_map.data.shape,
        projected_coord,
        scale=u.Quantity(suvi_map.scale),
        instrument=suvi_map.instrument,
        wavelength=suvi_map.wavelength,
    )
    reprojected = [reproject_map(m, projected_header) for m in [meermap,suvi_map]]
    meer_reprojected,suvi_reprojected = compute(*reprojected)
    meertime = meermap.meta["date-obs"].split(".")[0]
    suvitime = suvi_map.meta["date-obs"].split(".")[0]
    if plot_meer_colormap and len(contour_levels) > 0:
        matplotlib.rcParams.update({"font.size": 18})
        fig = plt.figure(figsize=(16, 8))
        ax_colormap = fig.add_subplot(1, 2, 1, projection=suvi_reprojected)
        ax_contour = fig.add_subplot(1, 2, 2, projection=suvi_reprojected)
    elif plot_meer_colormap:
        matplotlib.rcParams.update({"font.size": 14})
        fig = plt.figure(figsize=(10, 8))
        ax_colormap = fig.add_subplot(projection=suvi_reprojected)
    elif len(contour_levels) > 0:
        matplotlib.rcParams.update({"font.size": 14})
        fig = plt.figure(figsize=(10, 8))
        ax_contour = fig.add_subplot(projection=suvi_reprojected)
    else:
        print("No overlay is plotting.")
        return

    title = f"SUVI time: {suvitime}\n MeerKAT time: {meertime}"
    if "transparent_inferno" not in plt.colormaps():
        cmap = cm.get_cmap("inferno", 256)
        colors = cmap(np.linspace(0, 1, 256))
        x = np.linspace(0, 1, 256)
        alpha = 0.8 * (1 - np.exp(-3 * x))
        colors[:, -1] = alpha  # Update the alpha channel
        transparent_inferno = ListedColormap(colors)
        plt.colormaps.register(name="transparent_inferno", cmap=transparent_inferno)
    if plot_meer_colormap and len(contour_levels) > 0:
        suptitle = title.replace("\n", ",")
        title = ""
        fig.suptitle(suptitle)
    if plot_meer_colormap:
        z = 0
        suvi_reprojected.plot(
            axes=ax_colormap,
            title=title,
            autoalign=True,
            clip_interval=(3, 99.9) * u.percent,
            zorder=z,
        )
        z += 1
        meer_reprojected.plot(
            axes=ax_colormap,
            title=title,
            clip_interval=(3, 99.9) * u.percent,
            cmap="transparent_inferno",
            zorder=z,
        )
    if len(contour_levels) > 0:
        z = 0
        suvi_reprojected.plot(
            axes=ax_contour,
            title=title,
            autoalign=True,
            clip_interval=(3, 99.9) * u.percent,
            zorder=z,
        )
        z += 1
        contour_levels = np.array(contour_levels) * np.nanmax(meer_reprojected.data)
        meer_reprojected.draw_contours(
            contour_levels, axes=ax_contour, cmap="YlGnBu", zorder=z
        )
        ax_contour.set_facecolor("black")

    if len(xlim) > 0:
        x_pix_limits = []
        for x in xlim:
            sky = SkyCoord(
                x * u.arcsec, 0 * u.arcsec, frame=suvi_reprojected.coordinate_frame
            )
            x_pix = suvi_reprojected.world_to_pixel(sky)[0].value
            x_pix_limits.append(x_pix)
        if plot_meer_colormap and len(contour_levels) > 0:
            ax_colormap.set_xlim(x_pix_limits)
            ax_contour.set_xlim(x_pix_limits)
        elif plot_meer_colormap:
            ax_colormap.set_xlim(x_pix_limits)
        elif len(contour_levels) > 0:
            ax_contour.set_xlim(x_pix_limits)
    if len(ylim) > 0:
        y_pix_limits = []
        for y in ylim:
            sky = SkyCoord(
                0 * u.arcsec, y * u.arcsec, frame=suvi_reprojected.coordinate_frame
            )
            y_pix = suvi_reprojected.world_to_pixel(sky)[1].value
            y_pix_limits.append(y_pix)
        if plot_meer_colormap and len(contour_levels) > 0:
            ax_colormap.set_ylim(y_pix_limits)
            ax_contour.set_ylim(y_pix_limits)
        elif plot_meer_colormap:
            ax_colormap.set_ylim(y_pix_limits)
        elif len(contour_levels) > 0:
            ax_contour.set_ylim(y_pix_limits)
    if plot_meer_colormap and len(contour_levels) > 0:
        ax_colormap.coords.grid(False)
        ax_contour.coords.grid(False)
    elif plot_meer_colormap:
        ax_colormap.coords.grid(False)
    elif len(contour_levels) > 0:
        ax_contour.coords.grid(False)
    fig.tight_layout()
    plot_file_list=[]
    print("#######################")
    if plot_file_prefix:
        for i in range(len(extensions)):
            ext=extensions[i]
            try:
                savedir=outdirs[i]
            except:
                savedir=workdir
            plot_file = f"{savedir}/{plot_file_prefix}.{ext}"
            plt.savefig(plot_file, bbox_inches="tight")
            print(f"Plot saved: {plot_file}")
            plot_file_list.append(plot_file)
        print("#######################\n")
    else:
        plot_file = None
    if showgui:
        plt.show()
        plt.close(fig)
        plt.close("all")
    else:
        plt.close(fig)
    return plot_file_list


def split_noise_diode_scans(
    msname="",
    noise_on_ms="",
    noise_off_ms="",
    field="",
    scan="",
    datacolumn="data",
    n_threads=-1,
    dry_run=True,
):
    """
    Split noise diode on and off timestamps into two seperate measurement sets

    Parameters
    ----------
    msname : str
        Measurement set
    noise_on_ms : str, optional
        Noise diode on ms
    noise_off_ms : str, optional
        Noise diode off ms
    field : str, optional
        Field name or id
    scan : str, optional
        Scan number
    datacolumn : str, optional
        Data column to split
    n_threads : int, optional
        Number of OpenMP threads

    Returns
    -------
    tuple
        splited ms names
    """
    limit_threads(n_threads=n_threads)
    from casatasks import split

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    print(f"Spliting ms: {msname} into noise diode on and off measurement sets.")
    if noise_on_ms == "":
        noise_on_ms = msname.split(".ms")[0] + "_noise_on.ms"
    if noise_off_ms == "":
        noise_off_ms = msname.split(".ms")[0] + "_noise_off.ms"
    if os.path.exists(noise_on_ms):
        os.system("rm -rf " + noise_on_ms)
    if os.path.exists(noise_on_ms + ".flagversions"):
        os.system("rm -rf " + noise_on_ms + ".flagversions")
    if os.path.exists(noise_off_ms):
        os.system("rm -rf " + noise_off_ms)
    if os.path.exists(noise_off_ms + ".flagversions"):
        os.system("rm -rf " + noise_off_ms + ".flagversions")
    tb = table()
    tb.open(msname)
    times = tb.getcol("TIME")
    tb.close()
    unique_times = np.unique(times)
    even_times = unique_times[::2]  # Even-indexed timestamps
    odd_times = unique_times[1::2]  # Odd-indexed timestamps
    even_timerange = ",".join(
        [mjdsec_to_timestamp(t, str_format=1) for t in even_times]
    )
    odd_timerange = ",".join([mjdsec_to_timestamp(t, str_format=1) for t in odd_times])
    even_ms = msname.split(".ms")[0] + "_even.ms"
    odd_ms = msname.split(".ms")[0] + "_odd.ms"
    split(
        vis=msname,
        outputvis=even_ms,
        timerange=even_timerange,
        field=field,
        scan=scan,
        datacolumn=datacolumn,
    )
    split(
        vis=msname,
        outputvis=odd_ms,
        timerange=odd_timerange,
        field=field,
        scan=scan,
        datacolumn=datacolumn,
    )
    mstool = casamstool()
    mstool.open(even_ms)
    mstool.select({"antenna1": 1, "antenna2": 1})
    even_data = np.nanmean(np.abs(mstool.getdata("DATA")["data"]))
    mstool.close()
    mstool.open(odd_ms)
    mstool.select({"antenna1": 1, "antenna2": 1})
    odd_data = np.nanmean(np.abs(mstool.getdata("DATA")["data"]))
    mstool.close()
    if even_data > odd_data:
        os.system("mv " + even_ms + " " + noise_on_ms)
        os.system("mv " + odd_ms + " " + noise_off_ms)
    else:
        os.system("mv " + odd_ms + " " + noise_on_ms)
        os.system("mv " + even_ms + " " + noise_off_ms)
    return noise_on_ms, noise_off_ms


def get_band_name(msname):
    """
    Get band name

    Parameters
    ----------
    msname : str
        Name of the ms

    Returns
    -------
    str
        Band name ('U','L','S')
    """
    msmd = msmetadata()
    msmd.open(msname)
    meanfreq = msmd.meanfreq(0) / 10**6
    msmd.close()
    msmd.done()
    if meanfreq >= 544 and meanfreq <= 1088:
        return "U"
    elif meanfreq >= 856 and meanfreq <= 1712:
        return "L"
    else:
        return "S"


def get_bad_chans(msname):
    """
    Get bad channels to flag

    Parameters
    ----------
    msname : str
        Name of the ms

    Returns
    -------
    str
        SPW string of bad channels
    """
    msmd = msmetadata()
    msmd.open(msname)
    chanfreqs = msmd.chanfreqs(0) / 10**6
    msmd.close()
    msmd.done()
    bandname = get_band_name(msname)
    if bandname == "U":
        bad_freqs = [
            (-1, 580),
            (925, 960),
            (1010, -1),
        ]
    elif bandname == "L":
        bad_freqs = [
            (-1, 879),
            (925, 960),
            (1166, 1186),
            (1217, 1237),
            (1242, 1249),
            (1375, 1387),
            (1526, 1626),
            (1681, -1),
        ]
    else:
        print("Data is not in UHF or L-band.")
        bad_freqs = []
    spw = "0:"
    for freq_range in bad_freqs:
        start_freq = freq_range[0]
        end_freq = freq_range[1]
        if start_freq == -1:
            start_chan = 0
        else:
            start_chan = np.argmin(np.abs(start_freq - chanfreqs))
        if end_freq == -1:
            end_chan = len(chanfreqs) - 1
        else:
            end_chan = np.argmin(np.abs(end_freq - chanfreqs))
        spw += str(start_chan) + "~" + str(end_chan) + ";"
    spw = spw[:-1]
    return spw


def get_good_chans(msname):
    """
    Get good channel range to perform gaincal

    Parameters
    ----------
    msname : str
        Name of the ms

    Returns
    -------
    str
        SPW string
    """
    msmd = msmetadata()
    msmd.open(msname)
    chanfreqs = msmd.chanfreqs(0) / 10**6
    meanfreq = msmd.meanfreq(0) / 10**6
    msmd.close()
    msmd.done()
    bandname = get_band_name(msname)
    if bandname == "U":
        good_freqs = [(580, 620)]  # For UHF band
    elif bandname == "L":
        good_freqs = [(890, 920)]  # For L band
    else:
        good_freqs = []  # For S band #TODO: fill it
    spw = "0:"
    for freq_range in good_freqs:
        start_freq = freq_range[0]
        end_freq = freq_range[1]
        start_chan = np.argmin(np.abs(start_freq - chanfreqs))
        end_chan = np.argmin(np.abs(end_freq - chanfreqs))
        spw += str(start_chan) + "~" + str(end_chan) + ";"
    spw = spw[:-1]
    return spw


def get_bad_ants(msname="", fieldnames=[], n_threads=-1, dry_run=False):
    """
    Get bad antennas

    Parameters
    ----------
    msname : str
        Name of the ms
    fieldnames : list, optional
        Fluxcal field names
    n_threads : int, optional
        Number of OpenMP threads

    Returns
    -------
    list
        Bad antenna list
    str
        Bad antenna string
    """
    limit_threads(n_threads=n_threads)
    from casatasks import visstat

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    msmd = msmetadata()
    good_chan = get_good_chans(msname)
    all_field_bad_ants = []
    msmd.open(msname)
    nant = msmd.nantennas()
    msmd.close()
    msmd.done()
    for field in fieldnames:
        ant_means = []
        bad_ants = []
        for ant in range(nant):
            stat_mean = visstat(
                vis=msname,
                field=str(field),
                uvrange="0lambda",
                spw=good_chan,
                antenna=str(ant) + "&&" + str(ant),
                useflags=False,
            )["DATA_DESC_ID=0"]["mean"]
            ant_means.append(stat_mean)
        ant_means = np.array(ant_means)
        all_ant_mean = np.nanmean(ant_means)
        all_ant_std = np.nanstd(ant_means)
        pos = np.where(ant_means < all_ant_mean - (5 * all_ant_std))[0]
        if len(pos) > 0:
            for b_ant in pos:
                bad_ants.append(b_ant)
        all_field_bad_ants.append(bad_ants)
    bad_ants = [set(sublist) for sublist in all_field_bad_ants]
    common_elements = set.intersection(*bad_ants)
    bad_ants = list(common_elements)
    if len(bad_ants) > 0:
        bad_ants_str = ",".join([str(i) for i in bad_ants])
    else:
        bad_ants_str = ""
    return bad_ants, bad_ants_str


def get_common_spw(spw1, spw2):
    """
    Return common spectral windows in merged CASA string format.

    Parameters
    ----------
    spw1 : str
        First spectral window
    spw2 : str
        Second spectral window

    Returns
    -------
    str
        Merged spectral window
    """
    from itertools import groupby
    from collections import defaultdict

    def to_set(s):
        out, cur = set(), None
        for part in s.split(";"):
            if ":" in part:
                cur, rng = part.split(":")
            else:
                rng = part
            cur = int(cur)
            a, *b = map(int, rng.split("~"))
            out.update((cur, i) for i in range(a, (b[0] if b else a) + 1))
        return out

    def to_str(pairs):
        spw_dict = defaultdict(list)
        for spw, ch in sorted(pairs):
            spw_dict[spw].append(ch)
        result = []
        for spw, chans in spw_dict.items():
            chans.sort()
            for _, g in groupby(enumerate(chans), lambda x: x[1] - x[0]):
                grp = list(g)
                a, b = grp[0][1], grp[-1][1]
                result.append(f"{a}" if a == b else f"{a}~{b}")
        return "0:" + ";".join(result)

    return to_str(to_set(spw1) & to_set(spw2))


def scans_in_timerange(msname="", timerange="", dry_run=False):
    """
    Get scans in the given timerange

    Parameters
    ----------
    msname : str
        Measurement set
    timerange : str
        Time range with date and time

    Returns
    -------
    dict
        Scan dict for timerange
    """
    from casatools import ms, quanta

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    qa = quanta()
    ms_tool = ms()
    ms_tool.open(msname)
    # Get scan summary
    scan_summary = ms_tool.getscansummary()
    # Convert input timerange to MJD seconds
    timerange_list = timerange.split(",")
    valid_scans = {}
    for timerange in timerange_list:
        tr_start_str, tr_end_str = timerange.split("~")
        tr_start = timestamp_to_mjdsec(tr_start_str)  # Try parsing as date string
        tr_end = timestamp_to_mjdsec(tr_end_str)
        for scan_id, scan_info in scan_summary.items():
            t0_str = scan_info["0"]["BeginTime"]
            t1_str = scan_info["0"]["EndTime"]
            scan_start = qa.convert(qa.quantity(t0_str, "d"), "s")["value"]
            scan_end = qa.convert(qa.quantity(t1_str, "d"), "s")["value"]
            # Check overlap
            if scan_end >= tr_start and scan_start <= tr_end:
                if tr_end >= scan_end:
                    e = scan_end
                else:
                    e = tr_end
                if tr_start <= scan_start:
                    s = scan_start
                else:
                    s = tr_start
                if scan_id in valid_scans.keys():
                    old_t = valid_scans[scan_id].split("~")
                    old_s = timestamp_to_mjdsec(old_t[0])
                    old_e = timestamp_to_mjdsec(old_t[-1])
                    if s > old_s:
                        s = old_s
                    if e < old_e:
                        e = old_e
                valid_scans[int(scan_id)] = (
                    mjdsec_to_timestamp(s, str_format=1)
                    + "~"
                    + mjdsec_to_timestamp(e, str_format=1)
                )
    ms_tool.close()
    return valid_scans


def get_refant(msname="", n_threads=-1, dry_run=False):
    """
    Get reference antenna

    Parameters
    ----------
    msname : str
        Name of the measurement set
    n_threads : int, optional
        Number of OpenMP threads

    Returns
    -------
    str
        Reference antenna
    """
    limit_threads(n_threads=n_threads)
    from casatasks import visstat

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    casalog.filter("SEVERE")
    fluxcal_fields, fluxcal_scans = get_fluxcals(msname)
    msmd = msmetadata()
    msmd.open(msname)
    nant = int(msmd.nantennas() / 2)
    msmd.close()
    msmd.done()
    antamp = []
    antrms = []
    for ant in range(nant):
        ant = str(ant)
        t = visstat(
            vis=msname,
            field=fluxcal_fields[0],
            antenna=ant,
            timeaverage=True,
            timebin="500min",
            timespan="state,scan",
            reportingaxes="field",
        )
        item = str(list(t.keys())[0])
        amp = float(t[item]["median"])
        rms = float(t[item]["rms"])
        antamp.append(amp)
        antrms.append(rms)
    antamp = np.array(antamp)
    antrms = np.array(antrms)
    medamp = np.median(antamp)
    medrms = np.median(antrms)
    goodrms = []
    goodamp = []
    goodant = []
    for i in range(len(antamp)):
        if antamp[i] > medamp:
            goodant.append(i)
            goodamp.append(antamp[i])
            goodrms.append(antrms[i])
    goodrms = np.array(goodrms)
    referenceant = np.argmin(goodrms)
    return str(referenceant)


def get_ms_scans(msname):
    """
    Get scans of the measurement set

    Parameters
    ----------
    msname : str
        Measurement set

    Returns
    -------
    list
        Scan list
    """
    msmd = msmetadata()
    msmd.open(msname)
    scans = msmd.scannumbers().tolist()
    msmd.close()
    return scans


def get_submsname_scans(msname):
    """
    Get sub-MS names for each scans of an multi-MS

    Parameters
    ----------
    msname : str
        Name of the measurement set

    Returns
    -------
    list
        msname list
    list
        scan list
    """
    if os.path.exists(msname + "/SUBMSS") == False:
        print("Input measurement set is not a multi-MS")
        return
    partitionlist = listpartition(vis=msname, createdict=True)
    scans = []
    mslist = []
    for i in range(len(partitionlist)):
        subms = partitionlist[i]
        subms_name = msname + "/SUBMSS/" + subms["MS"]
        mslist.append(subms_name)
        os.system(f"rm -rf {subms_name}/.flagversions")
        scan_number = list(subms["scanId"].keys())[0]
        scans.append(scan_number)
    return mslist, scans


def get_chans_flag(msname="", field="", n_threads=-1, dry_run=False):
    """
    Get flag/unflag channel list

    Parameters
    ----------
    msname : str
        Measurement set name
    field : str, optional
        Field name or ID
    n_threads : int, optional
        Number of OpenMP threads

    Returns
    -------
    list
        Unflag channel list
    list
        Flag channel list
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    casalog.filter("SEVERE")
    summary = flagdata(vis=msname, field=field, mode="summary", spwchan=True)
    unflag_chans = []
    flag_chans = []
    for chan in summary["spw:channel"]:
        r = summary["spw:channel"][chan]
        chan_number = int(chan.split("0:")[-1])
        flag_frac = r["flagged"] / r["total"]
        if flag_frac == 1:
            flag_chans.append(chan_number)
        else:
            unflag_chans.append(chan_number)
    return unflag_chans, flag_chans


def get_optimal_image_interval(
    msname,
    temporal_tol_factor=0.1,
    spectral_tol_factor=0.1,
    chan_range="",
    timestamp_range="",
    max_nchan=-1,
    max_ntime=-1,
):
    """
    Get optimal image spectral temporal interval such that total flux max-median in each chunk is within tolerance limit

    Parameters
    ----------
    msname : str
        Name of the measurement set
    temporal_tol_factor : float, optional
        Tolerance factor for temporal variation (default : 0.1, 10%)
    spectral_tol_factor : float, optional
        Tolerance factor for spectral variation (default : 0.1, 10%)
    chan_range : str, optional
        Channel range
    timestamp_range : str, optional
        Timestamp range
    max_nchan : int, optional
        Maxmium number of spectral chunk
    max_ntime : int, optional
        Maximum number of temporal chunk

    Returns
    -------
    int
        Number of time intervals to average
    int
        Number of channels to averages
    """

    def is_valid_chunk(chunk, tolerance):
        mean_flux = np.nanmedian(chunk)
        if mean_flux == 0:
            return False
        return (np.nanmax(chunk) - np.nanmin(chunk)) / mean_flux <= tolerance

    def find_max_valid_chunk_length(fluxes, tolerance):
        n = len(fluxes)
        for window in range(n, 1, -1):  # Try from largest to smallest
            valid = True
            for start in range(0, n, window):
                end = min(start + window, n)
                chunk = fluxes[start:end]
                if len(chunk) < window:  # Optionally require full window
                    valid = False
                    break
                if not is_valid_chunk(chunk, tolerance):
                    valid = False
                    break
            if valid:
                return window  # Return the largest valid window
        return 1  # Minimum chunk size is 1 if nothing else is valid

    tb = table()
    mstool = casamstool()
    msmd = msmetadata()
    msmd.open(msname)
    nchan = msmd.nchan(0)
    times = msmd.timesforspws(0)
    ntime = len(times)
    del times
    msmd.close()
    tb.open(msname)
    u, v, w = tb.getcol("UVW")
    tb.close()
    uvdist = np.sort(np.unique(np.sqrt(u**2 + v**2)))
    mstool.open(msname)
    if uvdist[0] == 0.0:
        mstool.select({"uvdist": [0.0, 0.0]})
    else:
        mstool.select({"antenna1": 0, "antenna2": 1})
    data_and_flag = mstool.getdata(["DATA", "FLAG"], ifraxis=True)
    data = data_and_flag["data"]
    flag = data_and_flag["flag"]
    data[flag] = np.nan
    mstool.close()
    if chan_range != "":
        start_chan = int(chan_range.split(",")[0])
        end_chan = int(chan_range.split(",")[-1])
        spectra = np.nanmedian(data[:, start_chan:end_chan, ...], axis=(0, 2, 3))
    else:
        spectra = np.nanmedian(data, axis=(0, 2, 3))
    if timestamp_range != "":
        t_start = int(timestamp_range.split(",")[0])
        t_end = int(timestamp_range.split(",")[-1])
        t_series = np.nanmedian(data[..., t_start:t_end], axis=(0, 1, 2))
    else:
        t_series = np.nanmedian(data, axis=(0, 1, 2))
    t_series = t_series[t_series != 0]
    spectra = spectra[spectra != 0]
    t_chunksize = find_max_valid_chunk_length(t_series, temporal_tol_factor)
    f_chunksize = find_max_valid_chunk_length(spectra, spectral_tol_factor)
    n_time_interval = int(len(t_series) / t_chunksize)
    n_spectral_interval = int(len(spectra) / f_chunksize)
    if max_nchan > 0 and n_spectral_interval > max_nchan:
        n_spectral_interval = max_nchan
    if max_ntime > 0 and n_time_interval > max_ntime:
        n_time_interval = max_ntime
    return n_time_interval, n_spectral_interval


def reset_weights_and_flags(
    msname="", restore_flag=True, n_threads=-1, force_reset=False, dry_run=False
):
    """
    Reset weights and flags for the ms

    Parameters
    ----------
    msname : str
        Measurement set
    restore_flag : bool, optional
        Restore flags or not
    n_threads : int, optional
        Number of OpenMP threads
    force_reset : bool, optional
        Force reset
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    if os.path.exists(f"{msname}/.reset") == False or force_reset == True:
        mspath = os.path.dirname(os.path.abspath(msname))
        os.chdir(mspath)
        if restore_flag:
            print(f"Restoring flags of measurement set : {msname}")
            if os.path.exists(msname + ".flagversions"):
                os.system("rm -rf " + msname + ".flagversions")
            flagdata(vis=msname, mode="unflag", flagbackup=False)
        print(f"Resetting previous weights of the measurement set: {msname}")
        msmd = msmetadata()
        msmd.open(msname)
        npol = msmd.ncorrforpol()[0]
        msmd.close()
        tb = table()
        tb.open(msname, nomodify=False)
        colnames = tb.colnames()
        nrows = tb.nrows()
        if "WEIGHT" in colnames:
            print(f"Resetting weight column to ones of measurement set : {msname}.")
            weight = np.ones((npol, nrows))
            tb.putcol("WEIGHT", weight)
        if "SIGMA" in colnames:
            print(f"Resetting sigma column to ones of measurement set: {msname}.")
            sigma = np.ones((npol, nrows))
            tb.putcol("SIGMA", sigma)
        if "WEIGHT_SPECTRUM" in colnames:
            print(f"Removing weight spectrum of measurement set: {msname}.")
            tb.removecols("WEIGHT_SPECTRUM")
        if "SIGMA_SPECTRUM" in colnames:
            print(f"Removing sigma spectrum of measurement set: {msname}.")
            tb.removecols("SIGMA_SPECTRUM")
        tb.flush()
        tb.close()
        os.system(f"touch {msname}/.reset")
    return


def correct_missing_col_subms(msname):
    """
    Correct for missing colurmns in sub-MSs

    Parameters
    ----------
    msname : str
        Name of the measurement set
    """
    tb = table()
    colname_list = []
    sub_mslist = glob.glob(msname + "/SUBMSS/*.ms")
    for ms in sub_mslist:
        tb.open(ms)
        colname_list.append(tb.colnames())
        tb.close()
    sets = [set(sublist) for sublist in colname_list]
    if len(sets) > 0:
        common_elements = set.intersection(*sets)
        unique_elements = set.union(*sets) - common_elements
        for ms in sub_mslist:
            tb.open(ms, nomodify=False)
            colnames = tb.colnames()
            for colname in unique_elements:
                if colname in colnames:
                    print(f"Removing column: {colname} from sub-MS: {ms}")
                    tb.removecols(colname)
            tb.flush()
            tb.close()
    return


def get_unflagged_antennas(msname="", scan="", n_threads=-1, dry_run=False):
    """
    Get unflagged antennas of a scan

    Parameters
    ----------
    msname : str
        Name of the measurement set
    scan : str
        Scans
    n_threads : int, optional
        Number of OpenMP threads

    Returns
    -------
    numpy.array
        Unflagged antenna names
    numpy.array
        Flag fraction list
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    flag_summary = flagdata(vis=msname, scan=str(scan), mode="summary")
    antenna_flags = flag_summary["antenna"]
    unflagged_antenna_names = []
    flag_frac_list = []
    for ant in antenna_flags.keys():
        flag_frac = antenna_flags[ant]["flagged"] / antenna_flags[ant]["total"]
        if flag_frac < 1.0:
            unflagged_antenna_names.append(ant)
            flag_frac_list.append(flag_frac)
    unflagged_antenna_names = np.array(unflagged_antenna_names)
    flag_frac_list = np.array(flag_frac_list)
    return unflagged_antenna_names, flag_frac_list


def calc_flag_fraction(msname="", field="", scan="", n_threads=-1, dry_run=False):
    """
    Function to calculate the fraction of total data flagged.


    Parameters
    ----------
    msname : str
        Name of the measurement set
    field : str, optional
        Field names
    scan : str, optional
        Scan names
    n_threads : int, optional
        Number of OpenMP threads

    Returns
    -------
    float
        Fraction of the total data flagged
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    summary = flagdata(vis=msname, field=field, scan=scan, mode="summary")
    flagged_fraction = summary["flagged"] / summary["total"]
    return flagged_fraction


def get_fluxcals(msname):
    """
    Get fluxcal field names and scans

    Parameters
    ----------
    msname : str
        Name of the ms

    Returns
    -------
    list
        Fluxcal field names
    dict
        Fluxcal scans
    """
    msmd = msmetadata()
    if os.path.exists(msname + "/SUBMSS"):
        mslist = glob.glob(msname + "/SUBMSS/*.ms")
    else:
        mslist = [msname]
    fluxcal_fields = []
    fluxcal_scans = {}
    for msname in mslist:
        msmd.open(msname)
        field_names = msmd.fieldnames()
        for field in field_names:
            if field == "J1939-6342" or field == "J0408-6545":
                if field not in fluxcal_fields:
                    fluxcal_fields.append(field)
                scans = msmd.scansforfield(field).tolist()
                if field in fluxcal_scans:
                    for scan in scans:
                        fluxcal_scans[field].append(scan)
                else:
                    fluxcal_scans[field] = scans
    msmd.close()
    msmd.done()
    return fluxcal_fields, fluxcal_scans


def get_polcals(msname):
    """
    Get polarization calibrator field names and scans

    Parameters
    ----------
    msname : str
        Name of the ms

    Returns
    -------
    list
        Polcal field names
    dict
        Polcal scans
    """
    msmd = msmetadata()
    if os.path.exists(msname + "/SUBMSS"):
        mslist = glob.glob(msname + "/SUBMSS/*.ms")
    else:
        mslist = [msname]
    polcal_fields = []
    polcal_scans = {}
    for msname in mslist:
        msmd.open(msname)
        field_names = msmd.fieldnames()
        for field in field_names:
            if field in ["3C286", "1328+307", "1331+305", "J1331+3030"] or field in [
                "3C138",
                "0518+165",
                "0521+166",
                "J0521+1638",
            ]:
                if field not in polcal_fields:
                    polcal_fields.append(field)
                scans = msmd.scansforfield(field).tolist()
                if field in polcal_scans:
                    for scan in scans:
                        polcal_scans[field].append(scan)
                else:
                    polcal_scans[field] = scans
    msmd.close()
    msmd.done()
    del msmd
    return polcal_fields, polcal_scans


def get_phasecals(msname):
    """
    Get phasecal field names and scans

    Parameters
    ----------
    msname : str
        Name of the ms

    Returns
    -------
    list
        Phasecal field names
    dict
        Phasecal scans
    dict
        Phasecal flux
    """
    msmd = msmetadata()
    if os.path.exists(msname + "/SUBMSS"):
        mslist = glob.glob(msname + "/SUBMSS/*.ms")
    else:
        mslist = [msname]
    phasecal_fields = []
    phasecal_scans = {}
    phasecal_flux_list = {}
    datadir = get_datadir()
    for msname in mslist:
        msmd.open(msname)
        field_names = msmd.fieldnames()
        bandname = get_band_name(msname)
        if bandname == "U":
            phasecals, phasecal_flux = np.load(
                datadir + "/UHF_band_cal.npy", allow_pickle=True
            ).tolist()
        elif bandname == "L":
            phasecals, phasecal_flux = np.load(
                datadir + "/L_band_cal.npy", allow_pickle=True
            ).tolist()
        for field in field_names:
            if field in phasecals and (field != "J1939-6342" and field != "J0408-6545"):
                if field not in phasecal_fields:
                    phasecal_fields.append(field)
                scans = msmd.scansforfield(field).tolist()
                if field in phasecal_scans:
                    for scan in scans:
                        phasecal_scans[field].append(scan)
                else:
                    phasecal_scans[field] = scans
                flux = phasecal_flux[phasecals.index(field)]
                phasecal_flux_list[field] = flux
    msmd.close()
    msmd.done()
    del msmd
    return phasecal_fields, phasecal_scans, phasecal_flux_list


def get_target_fields(msname):
    """
    Get target fields

    Parameters
    ----------
    msname : str
        Name of the measurement set

    Returns
    -------
    list
        Target field names
    dict
        Target field scans
    """
    fluxcal_field, fluxcal_scans = get_fluxcals(msname)
    phasecal_field, phasecal_scans, phasecal_fluxs = get_phasecals(msname)
    calibrator_field = fluxcal_field + phasecal_field
    msmd = msmetadata()
    msmd.open(msname)
    field_names = msmd.fieldnames()
    field_names = np.unique(field_names)
    target_fields = []
    target_scans = {}
    for f in field_names:
        if f not in calibrator_field:
            target_fields.append(f)
    for field in target_fields:
        scans = msmd.scansforfield(field).tolist()
        target_scans[field] = scans
    msmd.close()
    msmd.done()
    del msmd
    return target_fields, target_scans


def get_caltable_fields(caltable):
    """
    Get caltable field names

    Parameters
    ----------
    caltable : str
        Caltable name

    Returns
    -------
    list
        Field names
    """
    tb = table()
    tb.open(caltable + "/FIELD")
    field_names = tb.getcol("NAME")
    field_ids = tb.getcol("SOURCE_ID")
    tb.close()
    tb.open(caltable)
    fields = np.unique(tb.getcol("FIELD_ID"))
    tb.close()
    field_name_list = []
    for f in fields:
        pos = np.where(field_ids == f)[0][0]
        field_name_list.append(str(field_names[pos]))
    return field_name_list


def get_cal_target_scans(msname):
    """
    Get calibrator and target scans

    Parameters
    ----------
    msname : str
        Name of the measurement set

    Returns
    -------
    list
        Target scan numbers
    list
        Calibrator scan numbers
    """
    f_scans = []
    p_scans = []
    g_scans = []
    fluxcal_fields, fluxcal_scans = get_fluxcals(msname)
    phasecal_fields, phasecal_scans, phasecal_flux_list = get_phasecals(msname)
    polcal_fields, polcal_scans = get_polcals(msname)
    for fluxcal_scan in fluxcal_scans.values():
        for s in fluxcal_scan:
            f_scans.append(s)
    for polcal_scan in polcal_scans.values():
        for s in polcal_scan:
            p_scans.append(s)
    for phasecal_scan in phasecal_scans.values():
        for s in phasecal_scan:
            g_scans.append(s)
    cal_scans = f_scans + p_scans + g_scans
    msmd = msmetadata()
    msmd.open(msname)
    all_scans = msmd.scannumbers()
    msmd.close()
    msmd.done()
    target_scans = []
    for scan in all_scans:
        if scan not in cal_scans:
            target_scans.append(scan)
    return target_scans, cal_scans, f_scans, g_scans, p_scans


def get_solar_elevation_MeerKAT(date_time=""):
    """
    Get solar elevation at MeerKAT at a time

    Parameters
    ----------
    date_time : str
        Date and time in 'yyyy-mm-ddTHH:MM:SS' format (default: current time)

    Returns
    -------
    float
        Solar elevation in degree
    """
    lat = -30.7130
    lon = 21.4430
    elev = 1038
    latitude = lat * u.deg  # In degree
    longitude = lon * u.deg  # In degree
    elevation = elev * u.m  # In meter
    if date_time == "":
        time = Time.now()
    else:
        time = Time(date_time)
    location = EarthLocation(lat=latitude, lon=longitude, height=elevation)
    sun_coords = get_sun(time)
    altaz_frame = AltAz(obstime=time, location=location)
    sun_altaz = sun_coords.transform_to(altaz_frame)
    solar_elevation = sun_altaz.alt.deg
    return solar_elevation


def timestamp_to_mjdsec(timestamp, date_format=0):
    """
    Convert timestamp to mjd second.


    Parameters
    ----------
    timestamp : str
        Time stamp to convert
    date_format : int, optional
        Datetime string format
            0: 'YYYY/MM/DD/hh:mm:ss'

            1: 'YYYY-MM-DDThh:mm:ss'

            2: 'YYYY-MM-DD hh:mm:ss'

            3: 'YYYY_MM_DD_hh_mm_ss'

    Returns
    -------
    float
        Return correspondong MJD second of the day
    """
    import julian

    if date_format == 0:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y/%m/%d/%H:%M:%S.%f")
        except:
            timestamp_datetime = dt.strptime(timestamp, "%Y/%m/%d/%H:%M:%S")
    elif date_format == 1:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        except:
            timestamp_datetime = dt.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
    elif date_format == 2:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y-%m-%d%H:%M:%S.%f")
        except:
            timestamp_datetime = dt.strptime(timestamp, "%Y-%m-%d%H:%M:%S")
    elif date_format == 3:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y_%m_%d_%H_%M_%S.%f")
        except:
            timestamp_datetime = dt.strptime(timestamp, "%Y_%m_%d_%H_%M_%S")
    else:
        print("No proper format of timestamp.\n")
        return
    mjd = float(
        "{: .2f}".format(
            (julian.to_jd(timestamp_datetime) - 2400000.5) * (24.0 * 3600.0)
        )
    )
    return mjd


def mjdsec_to_timestamp(mjdsec, str_format=0):
    """
    Convert CASA MJD seceonds to CASA timestamp

    Parameters
    ----------
    mjdsec : float
            CASA MJD seconds
    str_format : int
        Time stamp format (0: yyyy-mm-ddTHH:MM:SS.ff, 1: yyyy/mm/dd/HH:MM:SS.ff, 2: yyyy-mm-dd HH:MM:SS)

    Returns
    -------
    str
            CASA time stamp in UTC at ISOT format
    """
    from casatools import measures, quanta

    me = measures()
    qa = quanta()
    today = me.epoch("utc", "today")
    mjd = np.array(mjdsec) / 86400.0
    today["m0"]["value"] = mjd
    hhmmss = qa.time(today["m0"], prec=8)[0]
    date = qa.splitdate(today["m0"])
    qa.done()
    if str_format == 0:
        utcstring = "%s-%02d-%02dT%s" % (
            date["year"],
            date["month"],
            date["monthday"],
            hhmmss,
        )
    elif str_format == 1:
        utcstring = "%s/%02d/%02d/%s" % (
            date["year"],
            date["month"],
            date["monthday"],
            hhmmss,
        )
    else:
        utcstring = "%s-%02d-%02d %s" % (
            date["year"],
            date["month"],
            date["monthday"],
            hhmmss,
        )
    return utcstring


def get_timeranges_for_scan(
    msname, scan, time_interval, time_window, quack_timestamps=-1
):
    """
    Get time ranges for a scan with certain time intervals

    Parameters
    ----------
    msname : str
        Name of the measurement set
    scan : int
        Scan number
    time_interval : float
        Time interval in seconds
    time_window : float
        Time window in seconds
    quack_timestamps : int, optional
        Number of timestamps ignored at the start and end of each scan

    Returns
    -------
    list
        List of time ranges
    """
    msmd = msmetadata()
    msmd.open(msname)
    try:
        times = msmd.timesforscan(int(scan))
    except:
        times = msmd.timesforspws(0)
    msmd.close()
    msmd.done()
    time_ranges = []
    if quack_timestamps > 0:
        times = times[quack_timestamps:-quack_timestamps]
    else:
        times = times[1:-1]
    start_time = times[0]
    end_time = times[-1]
    if time_interval < 0 or time_window < 0:
        t = (
            mjdsec_to_timestamp(start_time, str_format=1)
            + "~"
            + mjdsec_to_timestamp(end_time, str_format=1)
        )
        time_ranges.append(t)
        return time_ranges
    total_time = end_time - start_time
    timeres = total_time / len(times)
    ntime_chunk = int(total_time / time_interval)
    ntime = int(time_window / timeres)
    start_time = times[:-ntime]
    indices = np.linspace(0, len(start_time) - 1, num=ntime_chunk, dtype=int)
    timelist = [start_time[i] for i in indices]
    for t in timelist:
        time_ranges.append(
            f"{mjdsec_to_timestamp(t, str_format=1)}~{mjdsec_to_timestamp(t+time_window, str_format=1)}"
        )
    return time_ranges


def radec_sun(msname):
    """
    RA DEC of the Sun at the start of the scan

    Parameters
    ----------
    msname : str
        Name of the measurement set

    Returns
    -------
    str
        RA DEC of the Sun in J2000
    str
        RA string
    str
        DEC string
    float
        RA in degree
    float
        DEC in degree
    """
    msmd = msmetadata()
    msmd.open(msname)
    times = msmd.timesforspws(0)
    msmd.close()
    msmd.done()
    mid_time = times[int(len(times) / 2)]
    mid_timestamp = mjdsec_to_timestamp(mid_time)
    astro_time=Time(mid_timestamp,scale='utc')
    sun_jpl = Horizons(id='10', location='500', epochs=astro_time.jd)
    eph = sun_jpl.ephemerides()
    sun_coord = SkyCoord(ra=eph['RA'][0]*u.deg, dec=eph['DEC'][0]*u.deg, frame='icrs')
    sun_ra = (
        str(int(sun_coord.ra.hms.h))
        + "h"
        + str(int(sun_coord.ra.hms.m))
        + "m"
        + str(round(sun_coord.ra.hms.s, 2))
        + "s"
    )
    sun_dec = (
        str(int(sun_coord.dec.dms.d))
        + "d"
        + str(int(sun_coord.dec.dms.m))
        + "m"
        + str(round(sun_coord.dec.dms.s, 2))
        + "s"
    )
    sun_radec_string = "J2000 " + str(sun_ra) + " " + str(sun_dec)
    radeg = sun_coord.ra.deg
    radeg = radeg % 360
    decdeg = sun_coord.dec.deg
    decdeg = decdeg % 360
    return sun_radec_string, sun_ra, sun_dec, radeg, decdeg


def get_phasecenter(msname, field):
    """
    Get phasecenter of the measurement set

    Parameters
    ----------
    msname : str
        Name of the measurement set

    Returns
    -------
    float
            RA in degree
    float
            DEC in degree
    """
    msmd = msmetadata()
    msmd.open(msname)
    phasecenter = msmd.phasecenter()
    msmd.close()
    msmd.done()
    radeg = np.rad2deg(phasecenter["m0"]["value"])
    radeg = radeg % 360
    decdeg = np.rad2deg(phasecenter["m1"]["value"])
    decdeg = decdeg % 360
    return radeg, decdeg


def angular_separation_equatorial(ra1, dec1, ra2, dec2):
    """
    Calculate angular seperation between two equatorial coordinates

    Parameters
    ----------
    ra1 : float
        RA of the first coordinate in degree
    dec1 : float
        DEC of the first coordinate in degree
    ra2 : float
        RA of the second coordinate in degree
    dec2 : float
        DEC of the second coordinate in degree

    Returns
    -------
    float
        Angular distance in degree
    """
    # Convert RA and Dec from degrees to radians
    ra1 = np.radians(ra1)
    ra2 = np.radians(ra2)
    dec1 = np.radians(dec1)
    dec2 = np.radians(dec2)
    # Apply the spherical distance formula using NumPy functions
    cos_theta = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(
        ra1 - ra2
    )
    # Calculate the angular separation in radians
    theta_rad = np.arccos(cos_theta)
    # Convert the angular separation from radians to degrees
    theta_deg = np.degrees(theta_rad)
    return theta_deg


def move_to_sun(msname, only_uvw=False):
    """
    Move the phasecenter of the measurement set at the center of the Sun (Assuming ms has one scan)

    Parameters
    ----------
    msname : str
        Name of the measurement set
    only_uvw : bool, optional
        Note: This is required when visibilities are properly phase rotated in correlator to track the Sun,
        but while creating the MS, UVW values are estimated using a wrong phase center at the start of solar center at the start.

    Returns
    -------
    int
        Success message
    """
    sun_radec_string, sunra, sundec, sunra_deg, sundec_deg = radec_sun(msname)
    msg = run_chgcenter(
        msname, sunra, sundec, only_uvw=only_uvw, container_name="meerwsclean"
    )
    if msg != 0:
        print("Phasecenter could not be shifted.")
    return msg


def correct_solar_sidereal_motion(msname="", verbose=False, dry_run=False):
    """
    Correct sodereal motion of the Sun

    Parameters
    ----------
    msname : str
        Name of the measurement set

    Returns
    -------
    int
        Success message
    """
    if dry_run:
        mem = run_solar_sidereal_cor(dry_run=True)
        return mem
    print(f"Correcting sidereal motion for ms: {msname}\n")
    if os.path.exists(msname + "/.sidereal_cor") == False:
        msg = run_solar_sidereal_cor(
            msname=msname, container_name="meerwsclean", verbose=verbose
        )
        if msg != 0:
            print("Sidereal motion correction is not successful.")
        else:
            os.system("touch " + msname + "/.sidereal_cor")
        return msg
    else:
        print(f"Sidereal motion correction is already done for ms: {msname}")
        return 0


def check_scan_in_caltable(caltable, scan):
    """
    Check scan number available in caltable or not

    Parameters
    ----------
    caltable : str
        Name of the caltable
    scan : int
        Scan number

    Returns
    -------
    bool
        Whether scan is present in the caltable or not
    """
    tb = table()
    tb.open(caltable)
    scans = tb.getcol("SCAN_NUMBER")
    tb.close()
    if int(scan) in scans:
        return True
    else:
        return False


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


def get_valid_scans(msname, field="", min_scan_time=1, n_threads=-1):
    """
    Get valid list of scans

    Parameters
    ----------
    msname : str
        Measurement set name
    min_scan_time : float
        Minimum valid scan time in minute

    Returns
    -------
    list
        Valid scan list
    """
    limit_threads(n_threads=n_threads)
    from casatools import ms as casamstool

    mstool = casamstool()
    mstool.open(msname)
    scan_summary = mstool.getscansummary()
    mstool.close()
    scans = np.sort(np.array([int(i) for i in scan_summary.keys()]))
    target_scans, cal_scans, f_scans, g_scans, p_scans = get_cal_target_scans(msname)
    selected_field = []
    valid_scans = []
    if field != "":
        field = field.split(",")
        msmd = msmetadata()
        msmd.open(msname)
        for f in field:
            try:
                field_id = msmd.fieldsforname(f)[0]
            except:
                field_id = int(f)
            selected_field.append(field_id)
        msmd.close()
        msmd.done()
        del msmd
    for scan in scans:
        scan_field = scan_summary[str(scan)]["0"]["FieldId"]
        if len(selected_field) == 0 or scan_field in selected_field:
            duration = (
                scan_summary[str(scan)]["0"]["EndTime"]
                - scan_summary[str(scan)]["0"]["BeginTime"]
            ) * 86400.0
            duration = round(duration / 60.0, 1)
            if duration >= min_scan_time:
                valid_scans.append(scan)
    return valid_scans


def split_into_chunks(lst, target_chunk_size):
    """
    Split a list into equal number of elements

    Parameters
    ----------
    lst : list
        List of numbers
    target_chunk_size: int
        Number of elements per chunk

    Returns
    -------
    list
        Chunked list
    """
    n = len(lst)
    num_chunks = max(1, round(n / target_chunk_size))
    avg_chunk_size = n // num_chunks
    remainder = n % num_chunks

    chunks = []
    start = 0
    for i in range(num_chunks):
        extra = 1 if i < remainder else 0  # Distribute remainder
        end = start + avg_chunk_size + extra
        chunks.append(lst[start:end])
        start = end
    return chunks


def calc_maxuv(msname, chan_number=-1):
    """
    Calculate maximum UV

    Parameters
    ----------
    msname : str
        Name of the measurement set
    chan_number : int, optional
        Channel number

    Returns
    -------
    float
        Maximum UV in meter
    float
        Maximum UV in wavelength
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.chanfreqs(0)[chan_number]
    wavelength = 299792458.0 / (freq)
    msmd.close()
    msmd.done()
    tb = table()
    tb.open(msname)
    uvw = tb.getcol("UVW")
    tb.close()
    u, v, w = [uvw[i, :] for i in range(3)]
    uv = np.sqrt(u**2 + v**2)
    uv[uv == 0] = np.nan
    maxuv = np.nanmax(uv)
    return maxuv, maxuv / wavelength


def calc_minuv(msname, chan_number=-1):
    """
    Calculate minimum UV

    Parameters
    ----------
    msname : str
        Name of the measurement set
    chan_number : int, optional
        Channel number

    Returns
    -------
    float
        Minimum UV in meter
    float
        Minimum UV in wavelength
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.chanfreqs(0)[chan_number]
    wavelength = 299792458.0 / (freq)
    msmd.close()
    msmd.done()
    tb = table()
    tb.open(msname)
    uvw = tb.getcol("UVW")
    tb.close()
    u, v, w = [uvw[i, :] for i in range(3)]
    uv = np.sqrt(u**2 + v**2)
    uv[uv == 0] = np.nan
    minuv = np.nanmin(uv)
    return minuv, minuv / wavelength


def calc_field_of_view(msname, FWHM=True):
    """
    Calculate optimum field of view in arcsec.

    Parameters
    ----------
    msname : str
        Measurement set name
    FWHM : bool, optional
        Upto FWHM, otherwise upto first null

    Returns
    -------
    float
        Field of view in arcsec
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.chanfreqs(0)[0]
    msmd.close()
    tb = table()
    tb.open(msname + "/ANTENNA")
    dish_dia = np.nanmin(tb.getcol("DISH_DIAMETER"))
    tb.close()
    wavelength = 299792458.0 / freq
    if FWHM == True:
        FOV = 1.22 * wavelength / dish_dia
    else:
        FOV = 2.04 * wavelength / dish_dia
    fov_arcsec = np.rad2deg(FOV) * 3600  ### In arcsecs
    return fov_arcsec


def ceil_to_multiple(n, base):
    """
    Round up to the next multiple

    Parameters
    ----------
    n : float
        The number
    base : float
        Whose multiple will be

    Returns
    -------
    float
        The modified number
    """
    return ((n // base) + 1) * base


def calc_bw_smearing_freqwidth(msname,full_FoV=False,FWHM=True):
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
        fov=calc_field_of_view(msname,FWHM=FWHM) # In arcsec
    else:
        fov = 35*60  # Size of the Sun, slightly larger is taken for U-band
    psf = calc_psf(msname)
    tb=table()
    tb.open(f"{msname}/SPECTRAL_WINDOW")
    freq=float(tb.getcol("REF_FREQUENCY")[0])/10**6
    freqres=float(tb.getcol("CHAN_WIDTH")[0])/10**6
    tb.close()
    delta_nu = np.sqrt((1 / R**2) - 1) * (psf / fov) * freq
    delta_nu = ceil_to_multiple(delta_nu, freqres)
    return round(delta_nu, 2)


def calc_time_smearing_timewidth(msname,full_FoV=False,FWHM=True):
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
        fov=calc_field_of_view(msname,FWHM=FWHM) # In arcsec
    else:
        fov = 35*60  # Size of the Sun, slightly larger is taken for U-band
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
    psf = calc_psf(msname)
    t_max = 0.5 * (psf / omega_sun)  # seconds
    return t_max


def calc_psf(msname, chan_number=-1):
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
    return psf


def calc_npix_in_psf(weight, robust=0.0):
    """
    Calculate number of pixels in a PSF (could be in fraction)

    Parameters
    ----------
    weight : str
        Image weighting scheme
    robust : float, optional
        Briggs weighting robust parameter (-1,1)

    Returns
    -------
    float
        Number of pixels in a PSF
    """
    if weight.upper() == "NATURAL":
        npix = 3
    elif weight.upper() == "UNIFORM":
        npix = 5
    else:  # -1 to +1, uniform to natural
        robust = np.clip(robust, -1.0, 1.0)
        npix = 5.0 - ((robust + 1.0) / 2.0) * (5.0 - 3.0)
    return round(npix, 1)


def calc_cellsize(msname, num_pixel_in_psf):
    """
    Calculate pixel size in arcsec

    Parameters
    ----------
    msname : str
        Name of the measurement set
    num_pixel_in_psf : float
            Number of pixels in one PSF

    Returns
    -------
    int
            Pixel size in arcsec
    """
    psf = calc_psf(msname)
    pixel = round(psf / num_pixel_in_psf, 1)
    return pixel


def calc_multiscale_scales(msname, num_pixel_in_psf, chan_number=-1, max_scale=16):
    """
    Calculate multiscale scales

    Parameters
    ----------
    msname : str
        Name of the measurement set
    num_pixel_in_psf : float
            Number of pixels in one PSF
    max_scale : float, optional
        Maximum scale in arcmin

    Returns
    -------
    list
            Multiscale scales in pixel units
    """
    psf = calc_psf(msname, chan_number=chan_number)
    minuv, minuv_l = calc_minuv(msname, chan_number=chan_number)
    max_interferometric_scale = (
        0.5 * np.rad2deg(1.0 / minuv_l) * 60.0
    )  # In arcmin, half of maximum scale
    max_interferometric_scale = min(max_scale, max_interferometric_scale)
    max_scale_pixel = int((max_interferometric_scale * 60.0) / (psf / num_pixel_in_psf))
    multiscale_scales = [0]
    current_scale = num_pixel_in_psf
    while True:
        current_scale = current_scale * 2
        if current_scale >= max_scale_pixel:
            break
        multiscale_scales.append(current_scale)
    return multiscale_scales


def get_multiscale_bias(freq, bias_min=0.6, bias_max=0.9):
    """
    Get frequency dependent multiscale bias

    Parameters
    ----------
    freq : float
        Frequency in MHz
    bias_min : float, optional
        Minimum bias at minimum L-band frequency
    bias_max : float, optional
        Maximum bias at maximum L-band frequency

    Returns
    -------
    float
        Multiscale bias patrameter
    """
    if freq <= 1015:
        return bias_min
    elif freq >= 1670:
        return bias_max
    else:
        freq_min = 1015
        freq_max = 1670
        logf = np.log10(freq)
        logf_min = np.log10(freq_min)
        logf_max = np.log10(freq_max)
        frac = (logf - logf_min) / (logf_max - logf_min)
        return round(
            np.clip(bias_min + frac * (bias_max - bias_min), bias_min, bias_max), 3
        )


def cutout_image(fits_file, output_file, x_deg=2):
    """
    Cutout central part of the image

    Parameters
    ----------
    fits_file : str
        Input fits file
    output_file : str
        Output fits file name (If same as input, input image will be overwritten)
    x_deg : float, optional
        Size of the output image in degree

    Returns
    -------
    str
        Output image name
    """
    hdu = fits.open(fits_file)[0]
    data = hdu.data  # shape: (nfreq, nstokes, ny, nx)
    header = hdu.header
    wcs = WCS(header)
    _, _, ny, nx = data.shape
    center_x, center_y = nx // 2, ny // 2
    # Get pixel scale (deg/pixel)
    pix_scale_deg = np.abs(header["CDELT1"])
    x_pix = int((x_deg / pix_scale_deg) / 2)
    # Adjust if cutout size exceeds image size
    max_half_x = nx // 2
    max_half_y = ny // 2
    x_pix = min(x_pix, max_half_x)
    y_pix = min(x_pix, max_half_y)  # Assume square pixels
    # Define slice indices
    x0 = center_x - x_pix
    x1 = center_x + x_pix
    y0 = center_y - y_pix
    y1 = center_y + y_pix
    # Slice data
    cutout_data = data[:, :, y0:y1, x0:x1]
    # Update header
    new_header = header.copy()
    new_header["NAXIS1"] = x1 - x0
    new_header["NAXIS2"] = y1 - y0
    new_header["CRPIX1"] -= x0
    new_header["CRPIX2"] -= y0
    # Save
    fits.writeto(output_file, cutout_data, header=new_header, overwrite=True)
    return output_file


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
    import warnings

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
            except:
                delay_gain[i, :, j] = 0.0
    tb.putcol("FPARAM", delay_gain)
    tb.putcol("FLAG", delay_flag)
    tb.flush()
    tb.close()
    os.system("rm -rf " + caltable + ".tempbcal")
    return caltable


def average_timestamp(timestamps):
    """
    Compute the average timestamp using astropy from a list of ISO 8601 strings.

    Parameters
    ----------
    timestamps : list
        timestamps (list of str): List of timestamp strings in 'YYYY-MM-DDTHH:MM:SS' format.

    Returns
    --------
    str
        Average timestamp in 'YYYY-MM-DDTHH:MM:SS' format.
    """
    times = Time(timestamps, format="isot", scale="utc")
    avg_time = Time(np.mean(times.jd), format="jd", scale="utc")
    return avg_time.isot.split(".")[0]  # Strip milliseconds for clean output


def make_timeavg_image(wsclean_images, outfile_name, keep_wsclean_images=True):
    """
    Convert WSClean images into a time averaged image

    Parameters
    ----------
    wsclean_images : list
        List of WSClean images.
    outfile_name : str
        Name of the output file.
    keep_wsclean_images : bool, optional
        Whether to retain the original WSClean images (default: True).

    Returns
    -------
    str
        Output image name.
    """
    timestamps = []
    for i in range(len(wsclean_images)):
        image = wsclean_images[i]
        if i == 0:
            data = fits.getdata(image)
        else:
            data += fits.getdata(image)
        timestamps.append(fits.getheader(image)["DATE-OBS"])
    data /= len(wsclean_images)
    avg_timestamp = average_timestamp(timestamps)
    header = fits.getheader(wsclean_images[0])
    header["DATE-OBS"] = avg_timestamp
    fits.writeto(outfile_name, data=data, header=header, overwrite=True)
    if not keep_wsclean_images:
        for img in wsclean_images:
            os.system(f"rm -rf {img}")
    return outfile_name


def make_freqavg_image(wsclean_images, outfile_name, keep_wsclean_images=True):
    """
    Convert WSClean images into a frequency averaged image

    Parameters
    ----------
    wsclean_images : list
        List of WSClean images.
    outfile_name : str
        Name of the output file.
    keep_wsclean_images : bool, optional
        Whether to retain the original WSClean images (default: True).

    Returns
    -------
    str
        Output image name.
    """
    freqs = []
    for i in range(len(wsclean_images)):
        image = wsclean_images[i]
        if i == 0:
            data = fits.getdata(image)
        else:
            data += fits.getdata(image)
        header = fits.getheader(image)
        if header["CTYPE3"] == "FREQ":
            freqs.append(float(header["CRVAL3"]))
            freqaxis = 3
        elif header["CTYPE4"] == "FREQ":
            freqs.append(float(header["CRVAL4"]))
            freqaxis = 4
    data /= len(wsclean_images)
    if len(freqs) > 0:
        mean_freq = np.nanmean(freqs)
        width = max(freqs) - min(freqs)
        header = fits.getheader(wsclean_images[0])
        if freqaxis == 3:
            header["CRAVL3"] = mean_freq
            header["CDELT3"] = width
        elif freqaxis == 4:
            header["CRAVL4"] = mean_freq
            header["CDELT4"] = width
    fits.writeto(outfile_name, data=data, header=header, overwrite=True)
    if not keep_wsclean_images:
        for img in wsclean_images:
            os.system(f"rm -rf {img}")
    return outfile_name


def get_meermap(fits_image, band="", do_sharpen=False):
    """
    Make MeerKAT sunpy map

    Parameters
    ----------
    fits_image : str
        MeerKAT fits image
    band : str, optional
        Band name
    do_sharpen : bool, optional
        Sharpen the image

    Returns
    -------
    sunpy.map
        Sunpy map
    """
    logging.getLogger('sunpy').setLevel(logging.ERROR)
    MEERLAT = -30.7133
    MEERLON = 21.4429
    MEERALT = 1086.6
    meer_hdu = fits.open(fits_image)  # Opening MeerKAT fits file
    meer_header = meer_hdu[0].header  # meer header
    meer_data = meer_hdu[0].data
    if len(meer_data.shape) > 2:
        meer_data = meer_data[0, 0, :, :]  # meer data
    if meer_header["CTYPE3"] == "FREQ":
        frequency = meer_header["CRVAL3"] * u.Hz
    elif meer_header["CTYPE4"] == "FREQ":
        frequency = meer_header["CRVAL4"] * u.Hz
    else:
        frequency = ""
    if band == "":
        try:
            band = meer_header["BAND"]
        except:
            band = ""
    try:
        pixel_unit = meer_header["BUNIT"]
    except:
        pixel_nuit = ""
    obstime = Time(meer_header["date-obs"])
    meerpos = EarthLocation(
        lat=MEERLAT * u.deg, lon=MEERLON * u.deg, height=MEERALT * u.m
    )
    meer_gcrs = SkyCoord(meerpos.get_gcrs(obstime))  # Converting into GCRS coordinate
    reference_coord = SkyCoord(
        meer_header["crval1"] * u.Unit(meer_header["cunit1"]),
        meer_header["crval2"] * u.Unit(meer_header["cunit2"]),
        frame="gcrs",
        obstime=obstime,
        obsgeoloc=meer_gcrs.cartesian,
        obsgeovel=meer_gcrs.velocity.to_cartesian(),
        distance=meer_gcrs.hcrs.distance,
    )
    reference_coord_arcsec = reference_coord.transform_to(
        frames.Helioprojective(observer=meer_gcrs)
    )
    cdelt1 = (np.abs(meer_header["cdelt1"]) * u.deg).to(u.arcsec)
    cdelt2 = (np.abs(meer_header["cdelt2"]) * u.deg).to(u.arcsec)
    P1 = sun.P(obstime)  # Relative rotation angle
    new_meer_header = sunpy.map.make_fitswcs_header(
        meer_data,
        reference_coord_arcsec,
        reference_pixel=u.Quantity(
            [meer_header["crpix1"] - 1, meer_header["crpix2"] - 1] * u.pixel
        ),
        scale=u.Quantity([cdelt1, cdelt2] * u.arcsec / u.pix),
        rotation_angle=-P1,
        wavelength=frequency.to(u.MHz).round(2),
        observatory="MeerKAT",
    )
    if do_sharpen:
        blurred = gaussian_filter(meer_data, sigma=10)
        meer_data = meer_data + (meer_data - blurred)
    meer_map = Map(meer_data, new_meer_header)
    meer_map_rotate = meer_map.rotate()
    return meer_map_rotate


def save_in_hpc(fits_image,outdir="",xlim=[-1600, 1600],ylim=[-1600, 1600]):
    """
    Save solar image in helioprojective coordinates
    
    Parameters
    ----------
    fits_image : str
        FITS image name
    outdir : str, optional
        Output directory
    xlim : list
        X axis limit in arcsecond
    ylim : list
        Y axis limit in arcsecond
    
    Returns
    -------
    str
        FITS image in helioprojective coordinate
    """
    fits_header=fits.getheader(fits_image)
    meermap=get_meermap(fits_image)
    if len(xlim)==2 and len(ylim)==2:
        top_right = SkyCoord(
            xlim[1] * u.arcsec, ylim[1] * u.arcsec, frame=meermap.coordinate_frame
        )
        bottom_left = SkyCoord(
            xlim[0] * u.arcsec, ylim[0] * u.arcsec, frame=meermap.coordinate_frame
        )
        meermap = meermap.submap(bottom_left, top_right=top_right)
    if outdir=="":
        outdir=os.path.dirname(os.path.abspath(fits_image))
    outfile=f"{outdir}/{os.path.basename(fits_image).split('.fits')[0]}_HPC.fits"
    if os.path.exists(outfile):
        os.system(f"rm -rf {outfile}")
    meermap.save(outfile,filetype="fits")
    data=fits.getdata(outfile)
    data=data[np.newaxis,np.newaxis,...]
    hpc_header=fits.getheader(outfile)
    for key in [
        "NAXIS", "NAXIS3", "NAXIS4", "BUNIT", "CTYPE3", "CRPIX3", "CRVAL3", "CDELT3", "CUNIT3",
        "CTYPE4", "CRPIX4", "CRVAL4", "CDELT4", "CUNIT4", "AUTHOR", "PIPELINE", "BAND",
        "MAX", "MIN", "RMS", "SUM", "MEAN", "MEDIAN", "RMSDYN", "MIMADYN"
    ]:
        if key in fits_header:
            hpc_header[key] = fits_header[key]
    fits.writeto(outfile,data=data,header=hpc_header,overwrite=True) 
    return outfile

def plot_in_hpc(
    fits_image,
    draw_limb=False,
    extensions=["png"],
    outdirs=[],
    plot_range=[],
    power=0.5,
    xlim=[-1600, 1600],
    ylim=[-1600, 1600],
    contour_levels=[],
    band="",
    showgui=False,
):
    """
    Function to convert MeerKAT image into Helioprojective co-ordinate

    Parameters
    ----------
    fits_image : str
        Name of the fits image
    draw_limb : bool, optional
        Draw solar limb or not
    extensions : list, optional
        Output file extensions
    outdirs : list, optional
        Output directories for each extensions
    plot_range : list, optional
        Plot range
    power : float, optional
        Power stretch
    xlim : list
        X axis limit in arcsecond
    ylim : list
        Y axis limit in arcsecond
    contour_levels : list, optional
        Contour levels in fraction of peak, both positive and negative values allowed
    band : str, optional
        Band name
    showgui : bool, optional
        Show GUI

    Returns
    -------
    outfiles
        Saved plot file names
    sunpy.Map
        MeerKAT image in helioprojective co-ordinate
    """
    if showgui == False:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")
    matplotlib.rcParams.update({"font.size": 12})
    fits_image = fits_image.rstrip("/")
    meer_header = fits.getheader(fits_image)  # Opening MeerKAT fits file
    if meer_header["CTYPE3"] == "FREQ":
        frequency = meer_header["CRVAL3"] * u.Hz
    elif meer_header["CTYPE4"] == "FREQ":
        frequency = meer_header["CRVAL4"] * u.Hz
    else:
        frequency = ""
    if band == "":
        try:
            band = meer_header["BAND"]
        except:
            band = ""
    try:
        pixel_unit = meer_header["BUNIT"]
    except:
        pixel_nuit = ""
    obstime = Time(meer_header["date-obs"])
    meer_map_rotate = get_meermap(fits_image, band=band)
    top_right = SkyCoord(
        xlim[1] * u.arcsec, ylim[1] * u.arcsec, frame=meer_map_rotate.coordinate_frame
    )
    bottom_left = SkyCoord(
        xlim[0] * u.arcsec, ylim[0] * u.arcsec, frame=meer_map_rotate.coordinate_frame
    )
    cropped_map = meer_map_rotate.submap(bottom_left, top_right=top_right)
    meer_data = cropped_map.data
    if len(plot_range) < 2:
        norm = ImageNormalize(
            meer_data,
            vmin=0.03 * np.nanmax(meer_data),
            vmax=0.99 * np.nanmax(meer_data),
            stretch=PowerStretch(power),
        )
    else:
        norm = ImageNormalize(
            meer_data,
            vmin=np.nanmin(plot_range),
            vmax=np.nanmax(plot_range),
            stretch=PowerStretch(power),
        )
    if band == "U":
        cmap = "inferno"
        pos_color = "white"
        neg_color = "cyan"
    elif band == "L":
        pos_color = "hotpink"
        neg_color = "yellow"
        if "YlGnBu_inferno" not in plt.colormaps():
            # Sample YlGnBu_r colormap with 256 colors
            cmap_ylgnbu = cm.get_cmap("YlGnBu_r", 256)
            colors = cmap_ylgnbu(np.linspace(0, 1, 256))
            # Create perceptually linear spacing using inferno luminance
            cmap_inferno = cm.get_cmap("inferno", 256)
            # Sort YlGnBu colors by the inferred brightness from inferno
            luminance_ranks = np.argsort(
                np.mean(cmap_inferno(np.linspace(0, 1, 256))[:, :3], axis=1)
            )
            colors_uniform = colors[luminance_ranks]
            # New perceptual-YlGnBu-inspired colormap
            YlGnBu_inferno = ListedColormap(colors_uniform, name="YlGnBu_inferno")
            plt.colormaps.register(name="YlGnBu_inferno", cmap=YlGnBu_inferno)
        cmap = "YlGnBu_inferno"
    else:
        cmap = "cubehelix"
        pos_color = "cyan"
        neg_color = "gold"
    fig = plt.figure()
    ax = plt.subplot(projection=cropped_map)
    cropped_map.plot(norm=norm, cmap=cmap, axes=ax)
    if len(contour_levels) > 0:
        contour_levels = np.array(contour_levels)
        pos_cont = contour_levels[contour_levels >= 0]
        neg_cont = contour_levels[contour_levels < 0]
        if len(pos_cont) > 0:
            cropped_map.draw_contours(
                np.sort(pos_cont) * np.nanmax(meer_data), colors=pos_color
            )
        if len(neg_cont) > 0:
            cropped_map.draw_contours(
                np.sort(neg_cont) * np.nanmax(meer_data), colors=neg_color
            )
    ax.coords.grid(False)
    rgba_vmin = plt.get_cmap(cmap)(norm(norm.vmin))
    ax.set_facecolor(rgba_vmin)
    # Read synthesized beam from header
    try:
        bmaj = meer_header["BMAJ"] * u.deg.to(u.arcsec)  # in arcsec
        bmin = meer_header["BMIN"] * u.deg.to(u.arcsec)
        bpa = meer_header["BPA"] - sun.P(obstime).deg  # in degrees
    except KeyError:
        bmaj = bmin = bpa = None
    # Plot PSF ellipse in bottom-left if all values are present
    if bmaj and bmin and bpa is not None:
        # Coordinates where to place the beam (e.g., 5% above bottom-left corner)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        beam_center = SkyCoord(
            x0 + 0.08 * (x1 - x0),
            y0 + 0.08 * (y1 - y0),
            unit=u.arcsec,
            frame=cropped_map.coordinate_frame,
        )

        # Add ellipse patch
        beam_ellipse = Ellipse(
            (beam_center.Tx.value, beam_center.Ty.value),  # center in arcsec
            width=bmin,
            height=bmaj,
            angle=bpa,
            edgecolor="white",
            facecolor="white",
            lw=1,
        )
        ax.add_patch(beam_ellipse)
        # Draw square box around the ellipse
        box_size = 100  # slightly bigger than beam
        rect = Rectangle(
            (beam_center.Tx.value - box_size / 2, beam_center.Ty.value - box_size / 2),
            width=box_size,
            height=box_size,
            edgecolor="white",
            facecolor="none",
            lw=1.2,
            linestyle="solid",
        )
        ax.add_patch(rect)
    if draw_limb:
        cropped_map.draw_limb()
    formatter = ticker.FuncFormatter(lambda x, _: f"{int(x):.0e}")
    cbar = plt.colorbar(format=formatter)
    # Optional: set max 5 ticks to prevent clutter
    cbar.locator = ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()
    if pixel_unit == "K":
        cbar.set_label("Brightness temperature (K)")
    elif pixel_unit == "JY/BEAM":
        cbar.set_label("Flux density (Jy/beam)")
    fig.tight_layout()
    output_image_list=[]
    for i in range(len(extensions)):    
        ext=extensions[i]
        try:
            outdir=outdirs[i]
        except:
            outdir=os.path.dirname(os.path.abspath(fits_image))
        if len(contour_levels) > 0:
            output_image = (
                outdir
                + "/"
                + os.path.basename(fits_image).split(".fits")[0]
                + f"_contour.{ext}"
            )
        else:
            output_image = (
                outdir
                + "/"
                + os.path.basename(fits_image).split(".fits")[0]
                + f".{ext}"
            )
        output_image_list.append(output_image)
    for output_image in output_image_list:
        fig.savefig(output_image)
    if showgui:
        plt.show()
    plt.close(fig)
    plt.close("all")
    return output_image_list, cropped_map


def make_stokes_wsclean_imagecube(
    wsclean_images, outfile_name, keep_wsclean_images=True
):
    """
    Convert WSClean images into a Stokes cube image.


    Parameters
    ----------
    wsclean_images : list
        List of WSClean images.
    outfile_name : str
        Name of the output file.
    keep_wsclean_images : bool, optional
        Whether to retain the original WSClean images (default: True).

    Returns
    -------
    str
        Output image name.
    """
    stokes = sorted(
        set(
            (
                os.path.basename(i).split(".fits")[0].split(" - ")[-2]
                if " - " in i
                else "I"
            )
            for i in wsclean_images
        )
    )
    valid_stokes = [
        {"I"},
        {"I", "V"},
        {"I", "Q", "U", "V"},
        {"XX", "YY"},
        {"LL", "RR"},
        {"Q", "U"},
        {"I", "Q"},
    ]
    if set(stokes) not in valid_stokes:
        print("Invalid Stokes combination.")
        return
    imagename_prefix = "temp_" + os.path.basename(wsclean_images[0]).split(" - I")[0]
    imagename = imagename_prefix + ".image"
    data, header = fits.getdata(wsclean_images[0]), fits.getheader(wsclean_images[0])
    for img in wsclean_images[1:]:
        data = np.append(data, fits.getdata(img), axis=0)
    header.update(
        {"NAXIS4": len(stokes), "CRVAL4": 1 if "I" in stokes else -5, "CDELT4": 1}
    )
    temp_fits = imagename_prefix + ".fits"
    fits.writeto(outfile_name, data=data, header=header, overwrite=True)
    if not keep_wsclean_images:
        for img in wsclean_images:
            os.system(f"rm -rf {img}")
    return outfile_name


def do_flag_backup(msname, flagtype="flagdata"):
    """
    Take a flag backup

    Parameters
    ----------
    msname : str
        Measurement set name
    flagtype : str, optional
        Flag type
    """
    af = agentflagger()
    af.open(msname)
    versionlist = af.getflagversionlist()
    if len(versionlist) != 0:
        for version_name in versionlist:
            if flagtype in version_name:
                try:
                    version_num = (
                        int(version_name.split(":")[0].split(" ")[0].split("_")[-1]) + 1
                    )
                except:
                    version_num = 1
            else:
                version_num = 1
    else:
        version_num = 1
    dt_string = dt.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    af.saveflagversion(
        flagtype + "_" + str(version_num), "Flags autosave on " + dt_string
    )
    af.done()


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
    if type(caltables) != list or len(caltables) == 0:
        print("Please provide a list of caltable.")
        return
    if os.path.exists(merged_caltable) and append == True:
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
                if keepcopy == False:
                    os.system("rm -rf " + caltable)
    return merged_caltable


def get_nprocess_meersolar(jobid):
    """
    Get numbers of MeerSOLAR processes currently running

    Parameters
    ----------
    workdir : str
        Work directory name
    jobid : int
        MeerSOLAR Job ID

    Returns
    -------
    int
        Number of running processes
    """
    meersolar_cachedir = get_meersolar_cachedir()
    pid_file = f"{meersolar_cachedir}/pids/pids_{jobid}.txt"
    pids = np.loadtxt(pid_file, unpack=True)
    n_process = 0
    for pid in pids:
        if psutil.pid_exists(int(pid)):
            n_process += 1
    return n_process


def save_pid(pid, pid_file):
    """
    Save PID

    Parameters
    ----------
    pid : int
        Process ID
    pid_file : str
        File to save
    """
    if os.path.exists(pid_file):
        pids = np.loadtxt(pid_file, unpack=True, dtype="int")
        pids = np.append(pids, pid)
    else:
        pids = np.array([int(pid)])
    np.savetxt(pid_file, pids, fmt="%d")


def get_jobid():
    """
    Get MeerSOLAR Job ID with millisecond-level uniqueness.

    Returns
    -------
    int
        Job ID in the format YYYYMMDDHHMMSSmmm (milliseconds)
    """
    meersolar_cachedir = get_meersolar_cachedir()
    jobid_file = os.path.join(meersolar_cachedir, "jobids.txt")
    if os.path.exists(jobid_file):
        prev_jobids = np.loadtxt(jobid_file, unpack=True, dtype="int64")
        if prev_jobids.size == 0:
            prev_jobids = []
        elif prev_jobids.size==1:
            prev_jobids = [str(prev_jobids)]
        else:
            prev_jobids = [str(jid) for jid in prev_jobids]
    else:
        prev_jobids = []

    if len(prev_jobids) > 0:
        FORMAT = "%Y%m%d%H%M%S%f"
        CUTOFF = dt.utcnow() - timedelta(days=15)
        filtered_prev_jobids = []
        for job_id in prev_jobids:
            job_time = dt.strptime(job_id.ljust(20, "0"), FORMAT)  # pad if truncated
            if job_time >= CUTOFF or job_id == 0:  # Job ID 0 is always kept
                filtered_prev_jobids.append(job_id)
        prev_jobids = filtered_prev_jobids

    now = dt.utcnow()
    cur_jobid = (
        now.strftime("%Y%m%d%H%M%S") + f"{int(now.microsecond/1000):03d}"
    )  # ms = first 3 digits of microseconds
    prev_jobids.append(cur_jobid)

    job_ids_int = np.array(prev_jobids, dtype=np.int64)
    np.savetxt(jobid_file, job_ids_int, fmt="%d")

    return int(cur_jobid)


def save_main_process_info(pid, jobid, msname, workdir, outdir, cpu_frac, mem_frac):
    """
    Save MeerSOLAR main processes info

    Parameters
    ----------
    pid : int
        Main job process id
    jobid : int
        MeerSOLAR Job ID
    msname : str
        Main measurement set
    workdir : str
        Work directory
    outdir : str
        Output directory
    cpu_frac : float
        CPU fraction of the job
    mem_frac : float
        Mempry fraction of the job

    Returns
    -------
    str
        Job info file name
    """
    meersolar_cachedir = get_meersolar_cachedir()
    prev_main_pids = glob.glob(f"{meersolar_cachedir}/main_pids_*.txt")
    prev_jobids = [
        str(os.path.basename(i).rstrip(".txt").split("main_pids_")[-1])
        for i in prev_main_pids
    ]
    if len(prev_jobids) > 0:
        FORMAT = "%Y%m%d%H%M%S%f"
        CUTOFF = dt.utcnow() - timedelta(days=15)
        filtered_prev_jobids = []
        for i in range(len(prev_jobids)):
            job_id = prev_jobids[i]
            job_time = dt.strptime(job_id.ljust(20, "0"), FORMAT)  # pad if truncated
            if job_time < CUTOFF or job_id == 0:  # Job ID 0 is always kept
                filtered_prev_jobids.append(job_id)
            else:
                os.system(f"rm -rf {prev_main_pids[i]}")
                if os.path.exists(f"{meersolar_cachedir}/pids/pids_{job_id}.txt"):
                    os.system(f"rm -rf {meersolar_cachedir}/pids/pids_{job_id}.txt")
    main_job_file = f"{meersolar_cachedir}/main_pids_{jobid}.txt"
    main_str = f"{jobid} {pid} {msname} {workdir} {outdir} {cpu_frac} {mem_frac}"
    with open(main_job_file, "w") as f:
        f.write(main_str)
    return main_job_file


def create_batch_script_nonhpc(cmd, workdir, basename, write_logfile=True):
    """
    Function to make a batch script not non-HPC environment

    Parameters
    ----------
    cmd : str
        Command to run
    workdir : str
        Work directory of the measurement set
    basename : str
        Base name of the batch files
    write_logfile : bool, optional
        Write log file or not

    Returns
    -------
    str
        Batch file name
    str
        Log file name
    """
    batch_file = workdir + "/" + basename + ".batch"
    cmd_batch = workdir + "/" + basename + "_cmd.batch"
    finished_touch_file = workdir + "/.Finished_" + basename
    os.system("rm -rf " + finished_touch_file + "*")
    finished_touch_file_error = finished_touch_file + "_1"
    finished_touch_file_success = finished_touch_file + "_0"
    cmd_file_content = f"{cmd}; exit_code=$?; if [ $exit_code -ne 0 ]; then touch {finished_touch_file_error}; else touch {finished_touch_file_success}; fi"
    if write_logfile:
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        outputfile = workdir + "/logs/" + basename + ".log"
    else:
        outputfile = "/dev/null"
    batch_file_content = f"""export PYTHONUNBUFFERED=1\nnohup sh {cmd_batch}> {outputfile} 2>&1 &\nsleep 2\n rm -rf {batch_file}\n rm -rf {cmd_batch}"""
    if os.path.exists(cmd_batch):
        os.system("rm -rf " + cmd_batch)
    if os.path.exists(batch_file):
        os.system("rm -rf " + batch_file)
    with open(cmd_batch, "w") as cmd_batch_file:
        cmd_batch_file.write(cmd_file_content)
    with open(batch_file, "w") as b_file:
        b_file.write(batch_file_content)
    os.system("chmod a+rwx " + batch_file)
    os.system("chmod a+rwx " + cmd_batch)
    del cmd
    return workdir + "/" + basename + ".batch"


def get_dask_client(
    n_jobs,
    dask_dir,
    cpu_frac=0.8,
    mem_frac=0.8,
    spill_frac=0.6,
    min_mem_per_job=-1,
    min_cpu_per_job=1,
    only_cal=False,
):
    """
    Create a Dask client optimized for one-task-per-worker execution,
    where each worker is a separate process that can use multiple threads internally.


    Parameters
    ----------
    n_jobs : int
        Number of MS tasks (ideally = number of MS files)
    dask_dir : str
        Dask temporary directory
    cpu_frac : float
        Fraction of total CPUs to use
    mem_frac : float
        Fraction of total memory to use
    spill_frac : float, optional
        Spill to disk at this fraction
    min_mem_per_job : float, optional
        Minimum memory per job
    min_cpu_per_job : int, optional
        Minimum CPU threads per job
    only_cal : bool, optional
        Only calculate number of workers

    Returns
    -------
    client : dask.distributed.Client
        Dask clinet
    cluster : dask.distributed.LocalCluster
        Dask cluster
    n_workers : int
        Number of workers
    threads_per_worker : int
        Threads per worker to use
    """
    # Create the Dask temporary working directory if it does not already exist
    os.makedirs(dask_dir, exist_ok=True)
    dask_dir_tmp = dask_dir + "/tmp"
    os.makedirs(dask_dir_tmp, exist_ok=True)

    # Detect total system resources
    total_cpus = psutil.cpu_count(logical=True)  # Total logical CPU cores
    total_mem = psutil.virtual_memory().total  # Total system memory (bytes)
    if cpu_frac > 0.8:
        print(
            "Given CPU fraction is more than 80%. Resetting to 80% to avoid system crash."
        )
        cpu_frac = 0.8
    if mem_frac > 0.8:
        print(
            "Given memory fraction is more than 80%. Resetting to 80% to avoid system crash."
        )
        mem_frac = 0.8

    ############################################
    # Wait until enough free CPU is available
    ############################################
    count = 0
    while True:
        available_cpu_pct = 100 - psutil.cpu_percent(
            interval=1
        )  # Percent CPUs currently free
        available_cpus = int(
            total_cpus * available_cpu_pct / 100.0
        )  # Number of free CPU cores
        usable_cpus = max(
            1, int(total_cpus * cpu_frac)
        )  # Target number of CPU cores we want available based on cpu_frac
        if available_cpus >= int(
            0.5 * usable_cpus
        ):  # Enough free CPUs (at-least more than 50%), exit loop
            usable_cpus = min(usable_cpus, available_cpus)
            break
        else:
            if count == 0:
                print("Waiting for available free CPUs...")
            time.sleep(5)  # Wait a bit and retry
        count += 1
    ############################################
    # Wait until enough free memory is available
    ############################################
    count = 0
    while True:
        available_mem = (
            psutil.virtual_memory().available
        )  # Current available system memory (bytes)
        usable_mem = total_mem * mem_frac  # Target usable memory based on mem_frac
        if (
            available_mem >= 0.5 * usable_mem
        ):  # Enough free memory, (at-least more than 50%) exit loop
            usable_mem = min(usable_mem, available_mem)
            break
        else:
            if count == 0:
                print("Waiting for available free memory...")
            time.sleep(5)  # Wait and retry
        count += 1

    ############################################
    # Calculate memory per worker
    ############################################
    mem_per_worker = usable_mem / n_jobs  # Assume initially one job per worker
    # Apply minimum memory per worker constraint
    min_mem_per_job = round(
        min_mem_per_job, 2
    )  # Ensure min_mem_per_job is a clean float
    if min_mem_per_job > 0 and mem_per_worker < (min_mem_per_job * 1024**3):
        # If calculated memory per worker is smaller than user-requested minimum, adjust number of workers
        print(
            f"Total memory per job is smaller than {min_mem_per_job} GB. Adjusting total number of workers to meet this."
        )
        mem_per_worker = (
            min_mem_per_job * 1024**3
        )  # Reset memory per worker to minimum allowed
        n_workers = min(
            n_jobs, int(usable_mem / mem_per_worker)
        )  # Reduce number of workers accordingly
    else:
        # Otherwise, just keep n_jobs workers
        n_workers = n_jobs

    #########################################
    # Cap number of workers to available CPUs
    n_workers = max(
        1, min(n_workers, int(usable_cpus / min_cpu_per_job))
    )  # Prevent CPU oversubscription
    # Recalculate final memory per worker based on capped n_workers
    mem_per_worker = usable_mem / n_workers
    # Calculate threads per worker
    threads_per_worker = max(
        1, usable_cpus // max(1, n_workers)
    )  # Each worker gets min_cpu_per_job or more threads

    ##########################################
    if only_cal == False:
        print("#################################")
        print(
            f"Dask workers: {n_workers}, Threads per worker: {threads_per_worker}, Mem/worker: {round(mem_per_worker/(1024.0**3),2)} GB"
        )
        print("#################################")
    # Memory control settings
    swap = psutil.swap_memory()
    swap_gb = swap.total / 1024.0**3
    if swap_gb > 16:
        pass
    elif swap_gb > 4:
        spill_frac = 0.6
    else:
        spill_frac = 0.5

    if spill_frac > 0.7:
        spill_frac = 0.7
    if only_cal:
        final_mem_per_worker = round((mem_per_worker * spill_frac) / (1024.0**3), 2)
        return None, None, n_workers, threads_per_worker, final_mem_per_worker

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(int(hard * 0.8), hard)  # safe cap
    if soft < new_soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    dask.config.set({"temporary-directory": dask_dir})
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,  # one python-thread per worker, in workers OpenMP threads can be used
        memory_limit=f"{round(mem_per_worker/(1024.0**3),2)}GB",
        local_directory=dask_dir,
        processes=True,  # one process per worker
        dashboard_address=None,
        env={
            "TMPDIR": dask_dir_tmp,
            "TMP": dask_dir_tmp,
            "TEMP": dask_dir_tmp,
            "DASK_TEMPORARY_DIRECTORY": dask_dir,
            "MALLOC_TRIM_THRESHOLD_": "0",
        },  # Explicitly set for workers
    )
    client = Client(cluster, timeout="60s", heartbeat_interval="5s")
    dask.config.set(
        {
            "distributed.worker.memory.target": spill_frac,
            "distributed.worker.memory.spill": spill_frac + 0.1,
            "distributed.worker.memory.pause": spill_frac + 0.2,
            "distributed.worker.memory.terminate": spill_frac + 0.25,
        }
    )

    client.run_on_scheduler(gc.collect)
    final_mem_per_worker = round((mem_per_worker * spill_frac) / (1024.0**3), 2)
    return client, cluster, n_workers, threads_per_worker, final_mem_per_worker


def run_limited_memory_task(task, dask_dir="/tmp", timeout=30):
    """
    Run a task for a limited time, then kill and return memory usage.

    Parameters
    ----------
    task : dask.delayed
        Dask delayed task object
    timeout : int
        Time in seconds to let the task run

    Returns
    -------
    float
        Memory used by task (in GB)
    """
    import warnings

    dask.config.set({"temporary-directory": dask_dir})
    warnings.filterwarnings("ignore")
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,  # one python-thread per worker, in workers OpenMP threads can be used
        local_directory=dask_dir,
        processes=True,  # one process per worker
        dashboard_address=":0",
    )
    client = Client(cluster)
    future = client.compute(task)
    start = time.time()

    def get_worker_memory_info():
        proc = psutil.Process()
        return {
            "rss_GB": proc.memory_info().rss / 1024**3,
            "vms_GB": proc.memory_info().vms / 1024**3,
        }

    while not future.done():
        if time.time() - start > timeout:
            try:
                mem_info = client.run(get_worker_memory_info)
                total_rss = sum(v["rss_GB"] for v in mem_info.values())
                per_worker_mem = total_rss
            except Exception as e:
                per_worker_mem = None
            future.cancel()
            client.close()
            cluster.close()
            return per_worker_mem
        time.sleep(1)
    mem_info = client.run(get_worker_memory_info)
    total_rss = sum(v["rss_GB"] for v in mem_info.values())
    per_worker_mem = total_rss
    client.close()
    cluster.close()
    return round(per_worker_mem, 2)


def baseline_names(msname):
    """
    Get baseline names

    Parameters
    ----------
    msname : str
        Measurement set name

    Returns
    -------
    list
        Baseline names list
    """
    mstool = casamstool()
    mstool.open(msname)
    ants = mstool.getdata(["antenna1", "antenna2"])
    mstool.close()
    baseline_ids = set(zip(ants["antenna1"], ants["antenna2"]))
    baseline_names = []
    for ant1, ant2 in sorted(baseline_ids):
        baseline_names.append(str(ant1) + "&&" + str(ant2))
    return baseline_names


def get_ms_size(msname, only_autocorr=False):
    """
    Get measurement set total size

    Parameters
    ----------
    msname : str
        Measurement set name
    only_autocorr : bool, optional
        Only auto-correlation

    Returns
    -------
    float
        Size in GB
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(msname):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    if only_autocorr:
        msmd = msmetadata()
        msmd.open(msname)
        nant = msmd.nantennas()
        msmd.close()
        all_baselines = (nant * nant) / 2
        total_size /= all_baselines
        total_size *= nant
    return round(total_size / (1024**3), 2)  # in GB


def get_column_size(msname, only_autocorr=False):
    """
    Get time chunk size for a memory limit

    Parameters
    ----------
    msname : str
        Measurement set
    only_autocorr : bool, optional
        Only auto-correlations

    Returns
    -------
    float
        A single datacolumn data size in GB
    """
    msmd = msmetadata()
    msmd.open(msname)
    nrow = int(msmd.nrows())
    nchan = msmd.nchan(0)
    npol = msmd.ncorrforpol()[0]
    nant = msmd.nantennas()
    msmd.close()
    datasize = nrow * nchan * npol * 16 / (1024.0**3)
    if only_autocorr:
        all_baselines = (nant * nant) / 2
        datasize /= all_baselines
        datasize *= nant
    return round(datasize, 2)


def get_ms_scan_size(msname, scan, only_autocorr=False):
    """
    Get measurement set scan size

    Parameters
    ----------
    msname : str
        Measurement set
    scan : int
        Scan number
    only_autocorr : bool, optional
        Only for auto-correlations

    Returns
    -------
    float
        Size in GB
    """
    tb = table()
    tb.open(msname)
    nrow = tb.nrows()
    tb.close()
    mstool = casamstool()
    mstool.open(msname)
    mstool.select({"scan_number": int(scan)})
    scan_nrow = mstool.nrow(True)
    mstool.close()
    ms_size = get_ms_size(msname, only_autocorr=only_autocorr)
    scan_size = scan_nrow * (ms_size / nrow)
    return round(scan_size, 2)


def get_chunk_size(msname, memory_limit=-1, only_autocorr=False):
    """
    Get time chunk size for a memory limit

    Parameters
    ----------
    msname : str
        Measurement set
    memory_limit : int, optional
        Memory limit
    only_autocorr : bool, optional
        Only aut-correlation

    Returns
    -------
    int
        Number of chunks
    """
    if memory_limit == -1:
        memory_limit = psutil.virtual_memory().available / 1024**3  # In GB
    col_size = get_column_size(msname, only_autocorr=only_autocorr)
    nchunk = int(col_size / memory_limit)
    if nchunk < 1:
        nchunk = 1
    return nchunk


def check_datacolumn_valid(msname, datacolumn="DATA"):
    """
    Check whether a data column exists and valid

    Parameters
    ----------
    msname : str
        Measurement set
    datacolumn : str, optional
        Data column string in table (e.g.,DATA, CORRECTED_DATA', MODEL_DATA, FLAG, WEIGHT, WEIGHT_SPECTRUM, SIGMA, SIGMA_SPECTRUM)

    Returns
    -------
    bool
        Whether valid data column is present or not
    """
    tb = table()
    msname = msname.rstrip("/")
    msname = os.path.abspath(msname)
    try:
        tb.open(msname)
        colnames = tb.colnames()
        if datacolumn not in colnames:
            tb.close()
            return False
        try:
            model_data = tb.getcol(datacolumn, startrow=0, nrow=1)
            tb.close()
            if model_data is None or model_data.size == 0:
                return False
            elif (model_data == 0).all():
                return False
            else:
                return True
        except:
            tb.close()
            return False
    except:
        return False


def create_circular_mask(msname, cellsize, imsize, mask_radius=20):
    """
    Create fits solar mask

    Parameters
    ----------
    msname : str
        Name of the measurement set
    cellsize : float
        Cell size in arcsec
    imsize : int
        Imsize in number of pixels
    mask_radius : float
        Mask radius in arcmin

    Returns
    -------
    str
        Fits mask file name
    """
    try:
        msname = msname.rstrip("/")
        imagename_prefix = msname.split(".ms")[0] + "_solar"
        wsclean_args = [
            "-quiet",
            "-scale " + str(cellsize) + "asec",
            "-size " + str(imsize) + " " + str(imsize),
            "-nwlayers 1",
            "-niter 0 -name " + imagename_prefix,
            "-channel-range 0 1",
            "-interval 0 1",
        ]
        wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " " + msname
        msg = run_wsclean(wsclean_cmd, "meerwsclean", verbose=False)
        if msg == 0:
            center = (int(imsize / 2), int(imsize / 2))
            radius = mask_radius * 60 / cellsize
            Y, X = np.ogrid[:imsize, :imsize]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            mask = dist_from_center <= radius
            os.system(
                "cp -r "
                + imagename_prefix
                + "-image.fits mask-"
                + os.path.basename(imagename_prefix)
                + ".fits"
            )
            os.system("rm -rf " + imagename_prefix + "*")
            data = fits.getdata("mask-" + os.path.basename(imagename_prefix) + ".fits")
            header = fits.getheader(
                "mask-" + os.path.basename(imagename_prefix) + ".fits"
            )
            data[0, 0, ...][mask] = 1.0
            data[0, 0, ...][~mask] = 0.0
            fits.writeto(
                imagename_prefix + "-mask.fits",
                data=data,
                header=header,
                overwrite=True,
            )
            os.system("rm -rf mask-" + os.path.basename(imagename_prefix) + ".fits")
            if os.path.exists(imagename_prefix + "-mask.fits"):
                return imagename_prefix + "-mask.fits"
            else:
                print("Circular mask could not be created.")
                return
        else:
            print("Circular mask could not be created.")
            return
    except Exception as e:
        traceback.print_exc()
        return


def calc_fractional_bandwidth(msname):
    """
    Calculate fractional bandwidh

    Parameters
    ----------
    msname : str
        Name of measurement set

    Returns
    -------
    float
        Fraction bandwidth in percentage
    """
    msmd = msmetadata()
    msmd.open(msname)
    freqs = msmd.chanfreqs(0)
    bw = max(freqs) - min(freqs)
    frac_bandwidth = bw / msmd.meanfreq(0)
    msmd.close()
    return round(frac_bandwidth * 100.0, 2)

def create_circular_mask_array(data,radius):
    """
    Creating circular mask of a Numpy array
    
    Parameters
    ----------
    data : numpy.array
        2D numpy array
    radius : int
        Radius in pixels
    
    Returns
    -------
    numpy.array
        Mask array
    """
    shape = data.shape
    center = (shape[0] // 2, shape[1] // 2)
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = (X - center[1])**2 + (Y - center[0])**2
    mask = dist_from_center <= radius**2
    return mask

def calc_solar_image_stat(imagename,disc_size=18):
    """
    Calculate solar image dynamic range
    
    Parameters
    ----------
    imagename : str
        Fits image name
    disc_size : float, optional
        Solar disc size in arcmin (default : 18)
    
    Returns
    -------
    float
        Maximum value
    float
        Minimum value
    float
        RMS values
    float
        Total value
    float
        Mean value
    float
        Median value
    float
        RMS dynamic range
    float
        Min-max dynamic range
    """
    data=fits.getdata(imagename)
    header=fits.getheader(imagename)
    pix_size=abs(header["CDELT1"])*3600.0 # In arcsec
    radius=int((disc_size*60)/pix_size)
    if len(data.shape)>2:
        data=data[0,0,...]
    mask=create_circular_mask_array(data,radius)
    masked_data=copy.deepcopy(data)
    masked_data[mask]=np.nan
    unmasked_data=copy.deepcopy(data)
    unmasked_data[~mask]=np.nan
    maxval=np.nanmax(unmasked_data)
    minval=np.nanmin(data)
    rms=np.nanstd(masked_data)
    total_val=np.nansum(unmasked_data)
    rms_dyn=maxval/rms
    minmax_dyn=maxval/abs(minval)
    mean_val=np.nanmean(unmasked_data)
    median_val=np.nanmedian(unmasked_data)
    del data, mask, unmasked_data, masked_data
    return maxval, minval, rms, total_val, mean_val, median_val, rms_dyn, minmax_dyn

def calc_dyn_range(imagename, modelname, residualname, fits_mask=""):
    """
    Calculate dynamic ranges.


    Parameters
    ----------
    imagename : list or str
        Image FITS file(s)
    modelname : list or str
        Model FITS file(s)
    residualname : list ot str
        Residual FITS file(s)
    fits_mask : str, optional
        FITS file mask

    Returns
    -------
    model_flux : float
        Total model flux.
    dyn_range_rms : float
        Max/RMS dynamic range.
    rms : float
        RMS of the image
    """

    def load_data(name):
        return fits.getdata(name)

    def to_list(x):
        return [x] if isinstance(x, str) else x

    imagename = to_list(imagename)
    modelname = to_list(modelname)
    residualname = to_list(residualname)

    use_mask = bool(fits_mask and os.path.exists(fits_mask))
    mask_data = fits.getdata(fits_mask).astype(bool) if use_mask else None

    model_flux, dr1, rmsvalue = 0, 0, 0

    for i in range(len(imagename)):
        img = imagename[i]
        res = residualname[i]
        image = load_data(img)
        residual = load_data(res)
        rms = np.nanstd(residual)
        if use_mask:
            maxval = np.nanmax(image[mask_data])
        else:
            maxval = np.nanmax(image)
        dr1 += maxval / rms if rms else 0
        rmsvalue += rms

    for mod in modelname:
        model = load_data(mod)
        model_flux += np.nansum(model[mask_data] if use_mask else model)

    rmsvalue = rmsvalue / np.sqrt(len(residualname))
    return model_flux, round(dr1, 2), round(rmsvalue, 2)


def generate_tb_map(imagename, outfile=""):
    """
    Function to generate brightness temperature map

    Parameters
    ----------
    imagename : str
        Name of the flux calibrated image
    outfile : str, optional
        Output brightess temperature image name

    Returns
    -------
    str
        Output image name
    """
    print(f"Generating brightness temperature map for image: {imagename}")
    if outfile == "":
        outfile = imagename.split(".fits")[0] + "_TB.fits"
    image_header = fits.getheader(imagename)
    image_data = fits.getdata(imagename)
    major = float(image_header["BMAJ"]) * 3600.0  # In arcsec
    minor = float(image_header["BMIN"]) * 3600.0  # In arcsec
    if image_header["CTYPE3"] == "FREQ":
        freq = image_header["CRVAL3"] / 10**9  # In GHz
    elif image_header["CTYPE4"] == "FREQ":
        freq = image_header["CRVAL4"] / 10**9  # In GHz
    else:
        print("No frequency information is present in header.")
        return
    TB_conv_factor = (1.222e6) / ((freq**2) * major * minor)
    TB_data = image_data * TB_conv_factor
    image_header["BUNIT"] = "K"
    fits.writeto(outfile, data=TB_data, header=image_header, overwrite=True)
    return outfile


####################
# uDOCKER related
####################
def check_udocker_container(name):
    """
    Check whether a docker container is present or not

    Parameters
    ----------
    name : str
        Container name

    Returns
    -------
    bool
        Whether present or not
    """
    pid = os.getpid()
    timestamp = int(time.time() * 1000)
    tmp1 = f"tmp1_{pid}_{timestamp}.txt"
    tmp2 = f"tmp2_{pid}_{timestamp}.txt"
    b = os.system(
        f"udocker --insecure --quiet inspect " + name + f" >> {tmp1} >> {tmp2}"
    )
    os.system(f"rm -rf {tmp1} {tmp2}")
    if b != 0:
        return False
    else:
        return True


def initialize_wsclean_container(name="meerwsclean"):
    """
    Initialize WSClean container

    Parameters
    ----------
    name : str
        Name of the container

    Returns
    -------
    bool
        Whether initialized successfully or not
    """
    image_name = "devojyoti96/wsclean-solar:latest"
    check_cmd = f"udocker images | grep -q '{image_name}'"
    image_exists = os.system(check_cmd) == 0
    if not image_exists:
        a = os.system(f"udocker pull {image_name}")
    else:
        print(f"Image '{image_name}' already present.")
        a = 0
    if a == 0:
        a = os.system(f"udocker create --name={name} {image_name}")
        print(f"Container started with name : {name}")
        return name
    else:
        print(f"Container could not be created with name : {name}")
        return


def run_wsclean(
    wsclean_cmd,
    container_name="meerwsclean",
    check_container=False,
    verbose=False,
    dry_run=False,
):
    """
    Run WSClean inside a udocker container (no root permission required).

    Parameters
    ----------
    wsclean_cmd : str
        Full WSClean command as a string.
    container_name : str, optional
        Container name
    check_container : bool, optional
        Check container presence or not
    verbose : bool, optional
        Verbose output or not

    Returns
    -------
    int
        Success message
    """
    pid = os.getpid()
    timestamp = int(time.time() * 1000)
    tmp1 = f"tmp1_{pid}_{timestamp}.txt"
    tmp2 = f"tmp2_{pid}_{timestamp}.txt"

    def show_file(path):
        try:
            print(open(path).read())
        except Exception as e:
            print(f"Error: {e}")

    if check_container:
        container_present = check_udocker_container(container_name)
        if container_present == False:
            container_name = initialize_wsclean_container(name=container_name)
            if container_name == None:
                print(
                    "Container {container_name} is not initiated. First initiate container and then run."
                )
                return 1

    if dry_run:
        cmd = f"chgenter >> {tmp1} >> {tmp2}"
        cwd = os.getcwd()
        full_command = (
            f"udocker --quiet run --nobanner --volume={cwd}:{cwd} meerwsclean {cmd}"
        )
        os.system(full_command)
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        os.system(f"rm -rf {tmp1} {tmp2}")
        return mem

    msname = wsclean_cmd.split(" ")[-1]
    msname = os.path.abspath(msname)
    mspath = os.path.dirname(msname)
    temp_docker_path = tempfile.mkdtemp(prefix="wsclean_udocker_", dir=mspath)
    wsclean_cmd_args = wsclean_cmd.split(" ")[:-1]
    if "-fits-mask" in wsclean_cmd_args:
        index = wsclean_cmd_args.index("-fits-mask")
        name = wsclean_cmd_args[index + 1]
        namedir = os.path.dirname(os.path.abspath(name))
        basename = os.path.basename(os.path.abspath(name))
        wsclean_cmd_args.remove(name)
        wsclean_cmd_args.insert(index + 1, temp_docker_path + "/" + basename)
    if "-name" not in wsclean_cmd_args:
        wsclean_cmd_args.append(
            "-name " + temp_docker_path + "/" + os.path.basename(msname).split(".ms")[0]
        )
    else:
        index = wsclean_cmd_args.index("-name")
        name = wsclean_cmd_args[index + 1]
        namedir = os.path.dirname(os.path.abspath(name))
        basename = os.path.basename(os.path.abspath(name))
        wsclean_cmd_args.remove(name)
        wsclean_cmd_args.insert(index + 1, temp_docker_path + "/" + basename)
    if "-temp-dir" not in wsclean_cmd_args:
        wsclean_cmd_args.append("-temp-dir " + temp_docker_path)
    else:
        index = wsclean_cmd_args.index("-temp-dir")
        name = os.path.abspath(wsclean_cmd_args[index + 1])
        wsclean_cmd_args.remove(name)
        wsclean_cmd_args.insert(index + 1, temp_docker_path)
    wsclean_cmd = (
        " ".join(wsclean_cmd_args)
        + " "
        + temp_docker_path
        + "/"
        + os.path.basename(msname)
    )
    try:
        full_command = f"udocker run --nobanner --volume={mspath}:{temp_docker_path} --workdir {temp_docker_path} meerwsclean {wsclean_cmd}"
        if verbose == False:
            full_command += f" >> {mspath}/{tmp1} "
        else:
            print(wsclean_cmd + "\n")
        exit_code = os.system(full_command)
        if exit_code != 0:
            print("##########################")
            print(os.path.basename(msname))
            print("##########################")
            show_file(f"{mspath}/{tmp1}")
        os.system(f"rm -rf {temp_docker_path} {mspath}/{tmp1}")
        return 0 if exit_code == 0 else 1
    except Exception as e:
        os.system(f"rm -rf {temp_docker_path}")
        traceback.print_exc()
        return 1


def run_solar_sidereal_cor(
    msname="",
    only_uvw=False,
    container_name="meerwsclean",
    verbose=False,
    dry_run=False,
):
    """
    Run chgcenter inside a udocker container to correct solar sidereal motion (no root permission required).

    Parameters
    ----------
    msname : str
        Name of the measurement set
    only_uvw : bool, optional
        Update only UVW values
        Note: This is required when visibilities are properly phase rotated in correlator to track the Sun,
        but while creating the MS, UVW values are estimated using the first phasecenter of the Sun.
    container_name : str, optional
        Container name
    verbose : bool, optional
        Verbose output or not

    Returns
    -------
    int
        Success message
    """
    pid = os.getpid()
    timestamp = int(time.time() * 1000)
    tmp1 = f"tmp1_{pid}_{timestamp}.txt"
    tmp2 = f"tmp2_{pid}_{timestamp}.txt"
    container_present = check_udocker_container(container_name)
    if container_present == False:
        container_name = initialize_wsclean_container(name=container_name)
        if container_name == None:
            print(
                "Container {container_name} is not initiated. First initiate container and then run."
            )
            return 1

    if dry_run:
        cmd = f"chgcentre >> {tmp1} >> {tmp2}"
        cwd = os.getcwd()
        full_command = (
            f"udocker --quiet run --nobanner --volume={cwd}:{cwd} meerwsclean {cmd}"
        )
        os.system(full_command)
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        os.system(f"rm -rf {tmp1} {tmp2}")
        return mem

    msname = os.path.abspath(msname)
    mspath = os.path.dirname(msname)
    temp_docker_path = tempfile.mkdtemp(prefix="chgcenter_udocker_", dir=mspath)
    if only_uvw:
        cmd = (
            "chgcentre -only-uvw -solarcenter "
            + temp_docker_path
            + "/"
            + os.path.basename(msname)
        )
    else:
        cmd = (
            "chgcentre -solarcenter "
            + temp_docker_path
            + "/"
            + os.path.basename(msname)
        )
    try:
        full_command = f"udocker --quiet run --nobanner --volume={mspath}:{temp_docker_path} --workdir {temp_docker_path} meerwsclean {cmd}"
        if verbose == False:
            full_command += f" >> {tmp1} >> {tmp2}"
        else:
            print(cmd)
        with suppress_casa_output():
            exit_code = os.system(full_command)
        os.system(f"rm -rf {temp_docker_path} {tmp1} {tmp2}")
        return 0 if exit_code == 0 else 1
    except Exception as e:
        os.system(f"rm -rf {temp_docker_path} {tmp1} {tmp2}")
        traceback.print_exc()
        return 1


def run_chgcenter(
    msname,
    ra,
    dec,
    only_uvw=False,
    container_name="meerwsclean",
    verbose=False,
    dry_run=False,
):
    """
    Run chgcenter inside a udocker container (no root permission required).

    Parameters
    ----------
    msname : str
        Name of the measurement set
    ra : str
        RA can either be 00h00m00.0s or 00:00:00.0
    dec : str
        Dec can either be 00d00m00.0s or 00.00.00.0
    only_uvw : bool, optional
        Update only UVW values
        Note: This is required when visibilities are properly phase rotated in correlator,
        but while creating the MS, UVW values are estimated using a wrong phase center.
    container_name : str, optional
        Container name
    verbose : bool, optional
        Verbose output

    Returns
    -------
    int
        Success message
    """
    pid = os.getpid()
    timestamp = int(time.time() * 1000)
    tmp1 = f"tmp1_{pid}_{timestamp}.txt"
    tmp2 = f"tmp2_{pid}_{timestamp}.txt"
    container_present = check_udocker_container(container_name)
    if container_present == False:
        container_name = initialize_wsclean_container(name=container_name)
        if container_name == None:
            print(
                "Container {container_name} is not initiated. First initiate container and then run."
            )
            return 1
    if dry_run:
        cmd = f"chgenter >> {tmp1} >> {tmp2}"
        cwd = os.getcwd()
        full_command = (
            f"udocker --quiet run --nobanner --volume={cwd}:{cwd} meerwsclean {cmd}"
        )
        os.system(full_command)
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        os.system(f"rm -rf {tmp1} {tmp2}")
        return mem
    msname = os.path.abspath(msname)
    mspath = os.path.dirname(msname)
    temp_docker_path = tempfile.mkdtemp(prefix="chgcenter_udocker_", dir=mspath)
    if only_uvw:
        cmd = (
            "chgcentre -only-uvw "
            + temp_docker_path
            + "/"
            + os.path.basename(msname)
            + " "
            + ra
            + " "
            + dec
        )
    else:
        cmd = (
            "chgcentre "
            + temp_docker_path
            + "/"
            + os.path.basename(msname)
            + " "
            + ra
            + " "
            + dec
        )
    try:
        full_command = f"udocker --quiet run --nobanner --volume={mspath}:{temp_docker_path} --workdir {temp_docker_path} meerwsclean {cmd}"
        if verbose == False:
            full_command += f" >> {tmp1} >> {tmp2}"
        else:
            print(cmd)
        exit_code = os.system(full_command)
        os.system(f"rm -rf {temp_docker_path} {tmp1} {tmp2}")
        return 0 if exit_code == 0 else 1
    except Exception as e:
        os.system(f"rm -rf {temp_docker_path} {tmp1} {tmp2}")
        traceback.print_exc()
        return 1
    return
