import os
from dotenv import load_dotenv

homedir = os.environ.get("HOME")
if homedir is None:
    homedir = os.path.expanduser("~")
username = os.getlogin()
cachedir = f"{homedir}/.solarpipe"
ENV_FILE=f"{cachedir}/meersolar_prefect.env"
load_dotenv(dotenv_path=ENV_FILE, override=False)

from casatasks import casalog
import os
try:
    logfile = casalog.logfile()
    os.remove(logfile)
except BaseException:
    pass
