from meersolar.pipeline.basic_func import *
from optparse import OptionParser
import requests, os, psutil
from parfive import Downloader

all_filenames=['3C138_pol_model.txt',
 'udocker-englib-1.2.11.tar.gz',
 'J1939-6342_U_model.txt',
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

def download_with_parfive(record_id, update=False, output_dir="zenodo_download"):
    print ("####################################")
    print ("Downloading MeerSOLAR data files ...")
    print ("####################################")
    urls = get_zenodo_file_urls(record_id)
    os.makedirs(output_dir, exist_ok=True)
    total_cpu=psutil.cpu_count()
    dl = Downloader(max_conn=min(total_cpu,len(all_filenames)))
    for file_url, filename in urls:
        if filename in all_filenames:
            if os.path.exists(f"{output_dir}/{filename}")==False or update==True:
                if os.path.exists(f"{output_dir}/{filename}"):
                    os.system(f"rm -rf {output_dir}/{filename}")
                dl.enqueue_file(file_url, path=output_dir, filename=filename)
    results = dl.download()

def init_meersolar_data():
    usage = "Initiate MeerSOLAR data"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--update",
        dest="update",
        default=False,
        help="Update existing data",
        metavar="Boolean",
    )
    parser.add_option(
        "--remote_link",
        dest="link",
        default=None,
        help="Set remote log link",
        metavar="String",
    )
    (options, args) = parser.parse_args()
    datadir=get_datadir()
    if os.path.exists(f"{datadir}/remotelink.txt")==False:
        os.system(f"touch {datadir}/remotelink.txt")
    if options.link!=None:
        with open(f"{datadir}/remotelink.txt", "w") as f:
            f.write(str(options.link))
    unavailable_files=[]
    for f in all_filenames:
        if os.path.exists(f"{datadir}/{f}")==False:
            unavailable_files.append(f)
    if len(unavailable_files)>0 or eval(str(options.update))==True:    
        record_id = "15691548"
        download_with_parfive(record_id, update=eval(str(options.update)), output_dir=datadir)
        timestr = dt.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print (f"MeeSOLAR data are updated in: {datadir} at time: {timestr}")
    else:
        print (f"All MeerSOLAR data are available in : {datadir}")

