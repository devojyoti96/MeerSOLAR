from .basic_utils import *
from .calibration import *
from .casatasks import *
from .flagging import *
from .image_utils import *
from .imaging import *
from .logger_utils import *
from .meer_ploting_utils import *
from .meer_utils import *
from .ms_metadata import *
from .proc_manage_utils import *
from .resource_utils import *
from .selfcal_utils import *
from .sunpos_utils import *
from .udocker_utils import *
from .prefect_logger_utils import *
from .prefect_setup_utils import *
import os
from casatasks import casalog
from astropy.utils import iers
from astropy.coordinates import solar_system_ephemeris


create_datadir()
set_udocker_env()
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
