import shutil
import os
import requests
import psutil
from parfive import Downloader


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
    dl = Downloader(max_conn=1)
    for file_url, filename in urls:
        if os.path.exists(f"{output_dir}/{filename}") == False or update:
            if os.path.exists(f"{output_dir}/{filename}"):
                os.system(f"rm -rf {output_dir}/{filename}")
            print(f"Final path: {output_dir}/{filename}")
            dl.enqueue_file(file_url, path=output_dir, filename=filename)
    results = dl.download()


def check_test_data(path):
    if os.path.exists(path + "/testdata/.testdata") == False:
        os.makedirs(path + "/testdata/", exist_ok=True)
        download_with_parfive(15999983, output_dir=path + "/testdata/")
        shutil.unpack_archive(
            path + "/testdata/meersolar_test_data.tar.gz",
            extract_dir=path,
        )
        os.system("rm -rf " + path + "/testdata/meersolar_test_data.tar.gz")
        os.system("touch " + path + "/testdata/.testdata")


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    check_test_data(path)
