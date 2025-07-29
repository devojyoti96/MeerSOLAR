import logging
import dask
import numpy as np
import argparse
import warnings
import traceback
import time
import glob
import sys
import os
import subprocess
from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from casatasks import casalog
from dask import delayed
from dask.distributed import Client
from meersolar.meerpipeline.single_image_meerpbcor import get_pbcor_image
from meersolar.utils import *

logging.getLogger("distributed").setLevel(logging.WARNING)

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    pass

datadir = get_datadir()
warnings.simplefilter("ignore", FITSFixedWarning)


def get_fits_freq(image_file):
    hdr = fits.getheader(image_file)
    keys = hdr.keys()
    if "CTYPE3" in keys and hdr["CTYPE3"] == "FREQ":
        freq = hdr["CRVAL3"]
        return freq
    elif "CTYPE4" in keys and hdr["CTYPE4"] == "FREQ":
        freq = hdr["CRVAL4"]
        return freq
    else:
        print(f"No frequency axis in image: {image_file}.")
        return


def run_pbcor(
    imagename, pbdir, pbcor_dir, apply_parang, jobid=0, ncpu=8, verbose=False
):
    """
    Run single image orimary beam correction

    Parameters
    ----------
    imagename : str
        Imagename
    pbdir : str
        Primary beam directory
    pbcor_dir : str
        Primary beam corrected image directory
    apply_parang : bool
        Apply parallactic angle correction or not
    jobid : int, optional
        Job ID
    ncpu : int, optional
        Number of CPU threads to use
    verbose: bool, optional
        Verbose output

    Returns
    -------
    int
        Success message
    """
    cmd = [
        "run-meer-singlepbcor",
        imagename,
        "--pbdir",
        pbdir,
        "--pbcor_dir",
        pbcor_dir,
        "--ncpu",
        str(ncpu),
        "--jobid",
        str(jobid),
    ]
    if not apply_parang:
        cmd.append("--no_apply_parang")

    if verbose:
        print("Executing:", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,  # Set to True if you want to raise on error
        )
        if verbose or result.returncode != 0:
            print(result.stdout)
        return result.returncode
    except Exception as e:
        print(f"Exception during primary beam correction: {e}")
        return 1


def pbcor_all_images(
    imagedir,
    make_TB=True,
    make_plots=True,
    apply_parang=True,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
    dask_addr=None,
):
    """
    Correct primary beam of MeerKAT for images in a directory

    Parameters
    ----------
    imagedir : str
        Name of the image directory
    make_TB : bool, optional
        Make brightness temperature map
    make_plots : bool, optional
        Make plots
    apply_parang : bool, optional
        Apply parallactic angle correction
    jobid : int, optional
        Job ID
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message
    """
    imagedir = imagedir.rstrip("/")
    pbdir = f"{os.path.dirname(imagedir)}/pbdir"
    pbcor_dir = f"{os.path.dirname(imagedir)}/pbcor_images"
    os.makedirs(pbdir, exist_ok=True)
    os.makedirs(pbcor_dir, exist_ok=True)
    successful_pbcor = 0
    try:
        images = glob.glob(f"{imagedir}/*.fits")
        if make_TB:
            tb_dir = f"{os.path.dirname(imagedir)}/tb_images"
            os.makedirs(tb_dir, exist_ok=True)
        if len(images) == 0:
            print(f"No image is present in image directory: {imagedir}")
            return 1
        first_set = []
        remaining_set = []
        freqs = []
        for image in images:
            freq = get_fits_freq(image)
            if freq in freqs:
                remaining_set.append(image)
            else:
                freqs.append(freq)
                first_set.append(image)
        mem_limit = (
            16 * max([os.path.getsize(image) for image in images]) / 1024**3
        )  # In GB
        if len(first_set) > 0:
            if dask_addr is None:
                dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = (
                    get_dask_client(
                        len(first_set),
                        dask_dir=pbdir,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                        min_mem_per_job=mem_limit,
                    )
                )
            else:
                _, _, n_jobs, n_threads, mem_limit, dask_dir = (
                    get_dask_client(
                        len(first_set),
                        dask_dir=pbdir,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                        min_mem_per_job=mem_limit,
                        only_cal=True,
                    )
                )
                dask_client=Client(address=dask_addr)
                os.system(f"rm -rf {dask_dir}")
            wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
            tasks = []
            for image in first_set:
                task = delayed(run_pbcor)(
                    image, pbdir, pbcor_dir, apply_parang, jobid=jobid, ncpu=n_threads
                )
                tasks.append(task)
            futures = dask_client.compute(tasks)
            results = list(dask_client.gather(futures))
            dask_client.close()
            if dask_addr is None:
                dask_cluster.close()
                os.system(f"rm -rf {dask_dir}")
            for r in results:
                if r == 0:
                    successful_pbcor += 1
        if len(remaining_set) > 0:
            print(f"Correcting remaining images of different timestamps.")
            if dask_addr is None:
                dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = (
                    get_dask_client(
                        len(remaining_set),
                        dask_dir=pbdir,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                        min_mem_per_job=mem_limit,
                    )
                )
            else:
                _, _, n_jobs, n_threads, mem_limit, dask_dir = (
                    get_dask_client(
                        len(remaining_set),
                        dask_dir=pbdir,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                        min_mem_per_job=mem_limit,
                        only_cal=True,
                    )
                )
                dask_client=Client(address=dask_addr)
                os.system(f"rm -rf {dask_dir}")
            wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
            tasks = []
            for image in remaining_set:
                task = delayed(run_pbcor)(
                    image, pbdir, pbcor_dir, apply_parang, jobid=jobid, ncpu=n_threads
                )
                tasks.append(task)
            futures = dask_client.compute(tasks)
            results = list(dask_client.gather(futures))
            dask_client.close()
            if dask_addr is None:
                dask_cluster.close()
                os.system(f"rm -rf {dask_dir}")
            for r in results:
                if r == 0:
                    successful_pbcor += 1

        ############################################
        # Saving fits in helioprojective coordinates
        ############################################
        if successful_pbcor > 0:
            hpcdir = f"{pbcor_dir}/hpcs"
            pbcor_images = glob.glob(f"{pbcor_dir}/*.fits")
            os.makedirs(hpcdir, exist_ok=True)
            print("Saving primary beam corrected images helioprojective coordinates...")
            for image in pbcor_images:
                save_in_hpc(image, outdir=hpcdir)
            if make_plots:
                print("Making plots of primary beam corrected images ...")
                pngdir = f"{pbcor_dir}/pngs"
                os.makedirs(pngdir, exist_ok=True)
                for image in pbcor_images:
                    try:
                        outimages = plot_in_hpc(
                            image,
                            draw_limb=True,
                            extensions=["png"],
                            outdirs=[pngdir],
                        )
                    except BaseException:
                        junkpng = f"{pngdir}/{os.path.basename(image).split('.fits')[0]}.png.junk"
                        os.system(f"touch {junkpng}")

        ####################################
        # Making brightness temperature maps
        ####################################
        if successful_pbcor > 0 and make_TB:
            for pbcor_image in pbcor_images:
                tb_image = (
                    tb_dir
                    + "/"
                    + os.path.basename(pbcor_image).split(".fits")[0]
                    + "_TB.fits"
                )
                generate_tb_map(pbcor_image, outfile=tb_image)

            ############################################
            # Saving fits in helioprojective coordinates
            ###########################################
            hpcdir = f"{tb_dir}/hpcs"
            tb_images = glob.glob(f"{tb_dir}/*.fits")
            os.makedirs(hpcdir, exist_ok=True)
            print("Saving brightness temperature maps helioprojective coordinates...")
            for image in tb_images:
                save_in_hpc(image, outdir=hpcdir)

            if make_plots:
                print("Making plots of brightness temperature maps..")
                pngdir = f"{tb_dir}/pngs"
                os.makedirs(pngdir, exist_ok=True)
                for image in tb_images:
                    outimages = plot_in_hpc(
                        image,
                        draw_limb=True,
                        extensions=["png"],
                        outdirs=[pngdir],
                    )

        #########################################
        # Final calculations
        #########################################
        print(f"Total input images: {len(images)}")
        if successful_pbcor > 0:
            print(f"Total corrected images: {len(pbcor_images)}")
            if make_TB:
                print(f"Total brightness temperatures maps: {len(tb_images)}")
        else:
            print(f"Total corrected images: 0")
        os.system(f"rm -rf {pbdir}/dask-scratch-space")
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1
    finally:
        os.system(f"rm -rf {pbdir}")


def main(
    imagedir,
    workdir="",
    make_TB=True,
    make_plots=True,
    apply_parang=True,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    start_remote_log=False,
    dask_addr=None,
):
    """
    Primary beam correction of MeerKAT for a sets of images in a directory

    Parameters
    ----------
    imagedir : str
        Image directory
    workdir : str, optional
        Work directory
    make_TB : bool, optional
        Make brightness temperature map or not
    make_plots : bool, optional
        Make png plots
    apply_parang : bool, optional
        Apply parallactic correction or not
    cpu_frac : float,optional
        CPU fraction
    mem_frac : float
        Memory fraction
    logfile : str, optional
        Log file
    jobid : str, optional
        Job ID
    start_remote_log : bool, optional
        Start remote logger
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message
    """
    if workdir == "":
        workdir = imagedir + "/workdir"
    os.makedirs(workdir, exist_ok=True)

    pid = os.getpid()
    cachedir = get_cachedir()
    save_pid(pid, f"{cachedir}/pids/pids_{jobid}.txt")

    ############
    # Logger
    ############
    observer = None
    if (
        start_remote_log
        and os.path.exists(f"{workdir}/jobname_password.npy")
        and logfile is not None
    ):
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            observer = init_logger(
                "all_pbcor", logfile, jobname=jobname, password=password
            )
    ###########

    try:
        if os.path.exists(imagedir):
            msg = pbcor_all_images(
                imagedir,
                make_TB=make_TB,
                make_plots=make_plots,
                apply_parang=apply_parang,
                jobid=jobid,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                dask_addr=dask_addr,
            )
        else:
            print("Please provide correct image directory path.")
            msg = 1
    except Exception:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        drop_cache(imagedir)
        drop_cache(workdir)
        clean_shutdown(observer)
    return msg


def cli():
    parser = argparse.ArgumentParser(
        description="Correct all images for MeerKAT full-pol averaged primary beam",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument("imagedir", help="Path to image directory")
    basic_args.add_argument("--workdir", default="", help="Path to work directory")

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--no_make_TB",
        action="store_false",
        dest="make_TB",
        help="Do not generate brightness temperature map",
    )
    adv_args.add_argument(
        "--no_make_plots",
        action="store_false",
        dest="make_plots",
        help="Do not make png plots",
    )
    adv_args.add_argument(
        "--no_apply_parang",
        action="store_false",
        dest="apply_parang",
        help="Do not apply parallactic angle correction",
    )
    adv_args.add_argument(
        "--start_remote_log",
        action="store_false",
        dest="start_remote_log",
        help="Start remote logger",
    )

    # Resource management parameters
    hard_args = parser.add_argument_group(
        "###################\nHardware resource management parameters\n###################"
    )
    hard_args.add_argument(
        "--cpu_frac", type=float, default=0.8, help="CPU usage fraction"
    )
    hard_args.add_argument(
        "--mem_frac", type=float, default=0.8, help="Memory usage fraction"
    )
    hard_args.add_argument("--logfile", default=None, help="Path to log file")
    hard_args.add_argument(
        "--jobid", type=int, default=0, help="Job ID for logging and PID tracking"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    msg = main(
        imagedir=args.imagedir,
        workdir=args.workdir,
        make_TB=args.make_TB,
        make_plots=args.make_plots,
        apply_parang=args.apply_parang,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        logfile=args.logfile,
        jobid=args.jobid,
        start_remote_log=args.start_remote_log,
    )
    return msg


if __name__ == "__main__":
    result = cli()
    print(
        "\n###################\nPrimary beam corrections are done.\n###################\n"
    )
    os._exit(result)
