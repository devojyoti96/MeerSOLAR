from sunpy.net import Fido, attrs as a
from sunpy.map import Map
from astroquery.jplhorizons import Horizons
from astropy.visualization import ImageNormalize, PowerStretch, LogStretch
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.wcs import FITSFixedWarning
from casatasks import casalog
from casatools import msmetadata, ms as casamstool, table
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
from datetime import datetime as dt, timezone, timedelta
from contextlib import contextmanager
import astropy.units as u
import secrets
import string
import julian
import resource
import shutil
import logging
import psutil
import dask
import numpy as np
import argparse
import requests
import traceback
import platform
import ctypes
import warnings
import tempfile
import gc
import copy
import time
import glob
import sys
import os

warnings.simplefilter("ignore", category=FITSFixedWarning)

try:
    logfile = casalog.logfile()
    os.system("rm -rf " + logfile)
except BaseException:
    pass
