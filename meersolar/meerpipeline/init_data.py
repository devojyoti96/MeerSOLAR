import logging
import psutil
import argparse
import requests
import sys
import os
from casatasks import casalog
from datetime import datetime as dt
from parfive import Downloader
from meersolar.utils import (
    get_datadir,
    get_cachedir,
    create_datadir,
    init_udocker,
    initialize_wsclean_container,
    SmartDefaultsHelpFormatter,
    start_server
    
)

logging.getLogger("distributed").setLevel(logging.WARNING)

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    pass

all_filenames = [
    "3C138_pol_model.txt",
    "udocker-englib-1.2.11.tar.gz",
    "J1939-6342_U_model.txt",
    "UHF_band_cal.npy",
    "L_band_cal.npy",
    "3C286_pol_model.txt",
    "J1939-6342_L_model.txt",
    "J0408-6545_U_model.txt",
    "MeerKAT_antavg_Uband.npz",
    "J0408-6545_L_model.txt",
    "MeerKAT_antavg_Lband.npz",
]


def get_zenodo_file_urls(record_id):
    url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return [(f["links"]["self"], f["key"]) for f in data.get("files", [])]


def download_with_parfive(record_id, update=False, output_dir="zenodo_download"):
    print("####################################")
    print("Downloading MeerSOLAR data files ...")
    print("####################################")
    urls = get_zenodo_file_urls(record_id)
    os.makedirs(output_dir, exist_ok=True)
    total_cpu = psutil.cpu_count()
    dl = Downloader(max_conn=min(total_cpu, len(all_filenames)))
    for file_url, filename in urls:
        if filename in all_filenames:
            if os.path.exists(f"{output_dir}/{filename}") == False or update:
                if os.path.exists(f"{output_dir}/{filename}"):
                    os.system(f"rm -rf {output_dir}/{filename}")
                dl.enqueue_file(file_url, path=output_dir, filename=filename)
    results = dl.download()


def init_meersolar_data(update=False, remote_link=None, emails=None):
    """
    Initiate MeerSOLAR data

    Parameters
    ----------
    update : bool, optional
        Update data, if already exists
    remote_link : str, optional
        Remote logger link to save in database
    emails : str, optional
        Email addresses to send remote logger JobID and password
    """
    datadir = get_datadir()
    os.makedirs(datadir, exist_ok=True)
    cachedir = get_cachedir()
    username = os.getlogin()
    linkfile = f"{cachedir}/remotelink_{username}.txt"
    emailfile = f"{cachedir}/emails_{username}.txt"
    if not os.path.exists(linkfile):
        with open(linkfile, "w") as f:
            f.write("")

    if remote_link is not None:
        with open(linkfile, "w") as f:
            f.write(str(remote_link))

    if emails is not None:
        with open(emailfile, "w") as f:
            f.write(str(emails))

    unavailable_files = [
        f for f in all_filenames if not os.path.exists(f"{datadir}/{f}")
    ]

    if unavailable_files or update:
        record_id = "15691548"
        download_with_parfive(record_id, update=update, output_dir=datadir)
        timestr = dt.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"MeeSOLAR data are updated in: {datadir} at time: {timestr}")


def main(
    init=False,
    prefect_server=True,
    datadir="",
    update=False,
    link=None,
    emails=None,
):
    """
    Initiate MeerSOLAR setup

    Parameters
    ----------
    init : bool, optional
        Initiate setup
    prefect_server : bool, optional
        Initiate prefect server
    datadir : str, optional
        User provided custom data directory
    update : bool, optional
        Update existing data (if corrupted by somehow)
    link : str
        Remote link
    emails : str
        E-mails for notifications
    """
    if init:
        create_datadir(datadir=datadir)
        datadir = get_datadir()
        print(f"MeerSOLAR data directory: {datadir}")
        init_meersolar_data(update=update, remote_link=link, emails=emails)
        print(f"MeerSOLAR data are initiated.")
        init_udocker()
        initialize_wsclean_container()
        print("uDOCKER inititalized")
        if prefect_server:
            start_server()
        return 0
    else:
        return 1


def cli():
    usage = "Initiate MeerSOLAR data"
    parser = argparse.ArgumentParser(
        description=usage, formatter_class=SmartDefaultsHelpFormatter
    )
    parser.add_argument("--init", action="store_true", help="Initiate data")
    parser.add_argument(
        "--datadir", type=str, default="", help="User provided data directory"
    )
    parser.add_argument(
        "--no_prefect_server", action="store_false", dest="init_prefect_server", help="Do not inititate prefect server"
    )
    parser.add_argument("--update", action="store_true", help="Update existing data")
    parser.add_argument(
        "--remotelink", dest="link", default=None, help="Set remote log link"
    )
    parser.add_argument(
        "--emails",
        dest="emails",
        default=None,
        help="Email addresses (comma seperated) to send Job ID and password for remote logger",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    msg = main(
        init=args.init,
        datadir=args.datadir,
        prefect_server=args.init_prefect_server,
        update=args.update,
        link=args.link,
        emails=args.emails,
    )
    return msg


if __name__ == "__main__":
    msg = cli()
    os._exit(msg)
