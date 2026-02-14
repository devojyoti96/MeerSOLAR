import os
from casatasks import casalog
from astropy.utils import iers
from astropy.coordinates import solar_system_ephemeris

try:
    logfile = casalog.logfile()
    os.remove(logfile)
except BaseException:
    pass

iers.conf.auto_download = False
iers.conf.auto_max_age = None
datadir = get_datadir()
try:
    solar_system_ephemeris.set(f"{datadir}/de440s")
except:
    solar_system_ephemeris.set("builtin")
