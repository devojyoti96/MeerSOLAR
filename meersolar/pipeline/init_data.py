from meersolar.pipeline.basic_func import *
import requests, os, psutil
from parfive import Downloader

all_filenames=['3C138_pol_model.txt',
 'udocker-englib-1.2.11.tar.gz',
 'J1939-6342_U_model.txt',
 'MeerLogger.AppImage',
 'UHF_band_cal.npy',
 'L_band_cal.npy',
 '3C286_pol_model.txt',
 'J1939-6342_L_model.txt',
 'J0408-6545_U_model.txt',
 'MeerKAT_antavg_Uband.npz',
 'J0408-6545_L_model.txt',
 'MeerKAT_antavg_Lband.npz']

def get_zenodo_file_urls(record_id):
    url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return [(f["links"]["self"], f["key"]) for f in data.get("files", [])]

def download_with_parfive(record_id, output_dir="zenodo_download"):
    urls = get_zenodo_file_urls(record_id)
    os.makedirs(output_dir, exist_ok=True)
    total_cpu=psutil.cpu_count()
    dl = Downloader(max_conn=min(total_cpu,len(all_filenames)))
    for file_url, filename in urls:
        if filename in all_filenames:
            dl.enqueue_file(file_url, path=output_dir, filename=filename)
    results = dl.download()

def download_data():
    datadir=get_datadir()
    record_id = "15686175"
    download_with_parfive(record_id, output_dir=datadir)
    timestr = dt.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print (f"MeeSOLAR data are updated in: {datadir} at time: {timestr}")

