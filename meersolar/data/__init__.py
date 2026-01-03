import os
from casatasks import casalog

try:
    logfile = casalog.logfile()
    os.remove(logfile)
except BaseException:
    pass
