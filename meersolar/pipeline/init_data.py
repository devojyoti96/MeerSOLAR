from meersolar.pipeline.basic_func import *
import requests, os, psutil, argparse, sys
from parfive import Downloader

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
            if os.path.exists(f"{output_dir}/{filename}") == False or update == True:
                if os.path.exists(f"{output_dir}/{filename}"):
                    os.system(f"rm -rf {output_dir}/{filename}")
                dl.enqueue_file(file_url, path=output_dir, filename=filename)
    results = dl.download()


def init_meersolar_data(update=False,remote_link=None):
    """
    Initiate MeerSOLAR data
    
    Parameters
    ----------
    update : bool, optional
        Update data, if already exists
    remote_link : str, optional
        Remote logger link to save in database
    """
    datadir = get_datadir()
    os.makedirs(datadir, exist_ok=True)
    linkfile = f"{datadir}/remotelink.txt"
    if not os.path.exists(linkfile):
        with open(linkfile, "w") as f:
            f.write("")

    if remote_link is not None:
        with open(linkfile, "w") as f:
            f.write(str(remote_link))

    unavailable_files = [
        f for f in all_filenames if not os.path.exists(f"{datadir}/{f}")
    ]

    if unavailable_files or update:
        record_id = "15691548"
        download_with_parfive(record_id, update=update, output_dir=datadir)
        timestr = dt.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"MeeSOLAR data are updated in: {datadir} at time: {timestr}")
   

def main():
    usage = "Initiate MeerSOLAR data"
    parser = argparse.ArgumentParser(description=usage,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--init", action="store_true", help="Initiate data")
    parser.add_argument("--update", action="store_true", help="Update existing data")
    parser.add_argument(
        "--remotelink", dest="link", default=None, help="Set remote log link"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    if args.init:
        init_meersolar_data(update=args.update,remote_link=args.link)
        print (f"MeerSOLAR data are initiated.")

if __name__ == "__main__":
    main()
