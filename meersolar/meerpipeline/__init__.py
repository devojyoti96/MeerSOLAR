import os
from casatasks import casalog
from astropy.utils import iers

try:
    logfile = casalog.logfile()
    os.remove(logfile)
except BaseException:
    pass

iers.conf.auto_download = False
iers.conf.auto_max_age = None
