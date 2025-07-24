import types
import secrets
import string
import logging
import argparse
import requests
import time
import glob
import sys
import os
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from datetime import datetime as dt
from .basic_utils import *
from .proc_manage_utils import *


##################################
# Logger related functions
##################################
class SmartDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        # Don't show default for boolean flags
        if isinstance(action, argparse._StoreTrueAction) or isinstance(
            action, argparse._StoreFalseAction
        ):
            return action.help
        return super()._get_help_string(action)


def clean_shutdown(observer):
    if observer:
        observer.stop()
        observer.join(timeout=5)


def generate_password(length=6):
    """
    Generate secure 6-character password with letters, digits, and symbols
    """
    chars = string.ascii_letters + string.digits + "@#$&*"
    return "".join(secrets.choice(chars) for _ in range(length))


def get_remote_logger_link():
    cachedir = get_cachedir()
    username = os.getlogin()
    link_file = os.path.join(cachedir, f"remotelink_{username}.txt")
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
    cachedir = get_cachedir()
    username = os.getlogin()
    email_file = os.path.join(cachedir, f"emails_{username}.txt")
    try:
        with open(email_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return ""
    if not lines:
        return ""
    else:
        return lines[0]


class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self._buffer = ""

    def write(self, message):
        # Remove trailing newlines and skip empty messages
        message = message.rstrip()
        if message:
            self.logger.log(self.log_level, message)

    def flush(self):
        pass  # Required for compatibility


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
    cachedir = get_cachedir()
    save_pid(pid, f"{cachedir}/pids/pids_{jobid}.txt")
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


def create_logger(logname, logfile, get_print=False, verbose=False):
    """
    Create logger.

    Parameters
    ----------
    logname : str
        Name of the log
    logfile : str, optional
        Log file name
    get_print : bool, optional
        Get print output to log
    verbose : bool, optional
        Verbose output or not

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
    if verbose:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)
    filehandle = logging.FileHandler(logfile)
    filehandle.setFormatter(formatter)
    logger.addHandler(filehandle)
    logger.propagate = False
    if get_print:
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
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
        "main.log":"Main pipeline log",
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
    Initialize a remote logger with watchdog-based tailing.

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
    observer 
        Observer object
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
    formatter = logging.Formatter("%(message)s")
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


# Exposing only functions
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
