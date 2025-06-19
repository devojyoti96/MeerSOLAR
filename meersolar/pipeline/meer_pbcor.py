import os, glob, traceback
from optparse import OptionParser
from meersolar.pipeline.single_image_meerpbcor import get_pbcor_image
from meersolar.pipeline.basic_func import *
from dask import delayed, compute


def get_fits_freq(image_file):
    hdr = fits.getheader(image_file)
    if hdr["CTYPE3"] == "FREQ":
        freq = hdr["CRVAL3"]
        return freq
    elif hdr["CTYPE4"] == "FREQ":
        freq = hdr["CRVAL4"]
        return freq
    else:
        print("No frequency axis in image.")
        return


def run_pbcor(imagename, pbdir, pbcor_dir, apply_parang, jobid=0, ncpu=8):
    cmd = f"run_single_meerpbcor --imagename {imagename} --pbdir {pbdir} --pbcor_dir {pbcor_dir} --ncpu {ncpu} --apply_parang {apply_parang} --jobid {jobid}"
    a = os.system(f"{cmd} > {imagename}.tmp")
    os.system(f"rm -rf {imagename}.tmp")
    return a


def pbcor_all_images(
    imagedir,
    make_TB=True,
    make_plots=True,
    apply_parang=True,
    jobid=0,
    cpu_frac=0.8,
    mem_frac=0.8,
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

    Returns
    -------
    int
        Success message
    """
    try:
        images = glob.glob(imagedir + "/*.fits")
        pbdir = os.path.dirname(imagedir.rstrip("/")) + "/pbdir"
        pbcor_dir = os.path.dirname(imagedir.rstrip("/")) + "/pbcor_images"
        os.makedirs(pbdir, exist_ok=True)
        os.makedirs(pbcor_dir, exist_ok=True)
        if make_TB:
            tb_dir = os.path.dirname(imagedir.rstrip("/")) + "/tb_images"
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
            dask_client, dask_cluster, n_jobs, n_threads, mem_limit = get_dask_client(
                len(first_set),
                dask_dir=pbdir,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                min_mem_per_job=mem_limit / 0.6,
            )
            tasks = []
            for image in first_set:
                task = delayed(run_pbcor)(
                    image, pbdir, pbcor_dir, apply_parang, jobid=jobid, ncpu=n_threads
                )
                tasks.append(task)
            results = compute(*tasks)
            dask_client.close()
            dask_cluster.close()
            successful_pbcor = 0
            for r in results:
                if r == 0:
                    successful_pbcor += 1
        if len(remaining_set) > 0:
            print(f"Correcting remaining images of different timestamps.")
            dask_client, dask_cluster, n_jobs, n_threads, mem_limit = get_dask_client(
                len(remaining_set),
                dask_dir=pbdir,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                min_mem_per_job=mem_limit / 0.6,
            )
            tasks = []
            for image in remaining_set:
                task = delayed(run_pbcor)(image, pbdir, pbcor_dir, apply_parang, jobid=jobid, ncpu=n_threads)
                tasks.append(task)
            results = compute(*tasks)
            dask_client.close()
            dask_cluster.close()
            for r in results:
                if r == 0:
                    successful_pbcor += 1
        if make_plots:
            print("Making plots of primary beam corrected images ...")
            images = glob.glob(f"{pbcor_dir}/*.fits")
            pngdir = f"{pbcor_dir}/pngs"
            pdfdir = f"{pbcor_dir}/pdfs"
            os.makedirs(pngdir, exist_ok=True)
            os.makedirs(pdfdir, exist_ok=True)
            for image in images:
                plot_in_hpc(
                    image,
                    draw_limb=True,
                    extension="png",
                    outdir=pngdir,
                )
                plot_in_hpc(
                    image,
                    draw_limb=True,
                    extension="pdf",
                    outdir=pdfdir,
                )
        if successful_pbcor > 0 and make_TB:
            successful_images = glob.glob(pbcor_dir + "/*.fits")
            for pbcor_image in successful_images:
                tb_image = (
                    tb_dir
                    + "/"
                    + os.path.basename(pbcor_image).split(".fits")[0]
                    + "_TB.fits"
                )
                generate_tb_map(pbcor_image, outfile=tb_image)
        if make_plots:
            print("Making plots of brightness temperature maps..")
            images = glob.glob(f"{tb_dir}/*.fits")
            pngdir = f"{tb_dir}/pngs"
            pdfdir = f"{tb_dir}/pdfs"
            os.makedirs(pngdir, exist_ok=True)
            os.makedirs(pdfdir, exist_ok=True)
            for image in images:
                plot_in_hpc(
                    image,
                    draw_limb=True,
                    extension="png",
                    outdir=pngdir,
                )
                plot_in_hpc(
                    image,
                    draw_limb=True,
                    extension="pdf",
                    outdir=pdfdir,
                )
        print(f"Total input images: {len(images)}")
        print(f"Total corrected images: {successful_pbcor}")
        os.system(f"rm -rf {pbdir}/dask-scratch-space")
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1


def main():
    usage = "Correct all images in a directory for full-polar antenna averaged MeerKAT primary beam"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--imagedir",
        dest="imagedir",
        default="",
        help="Name of image directory",
        metavar="String",
    )
    parser.add_option(
        "--workdir",
        dest="workdir",
        default="",
        help="Name of work directory",
        metavar="String",
    )
    parser.add_option(
        "--make_TB",
        dest="make_TB",
        default=True,
        help="Make brightness temperature map",
        metavar="Boolean",
    )
    parser.add_option(
        "--make_plots",
        dest="make_plots",
        default=True,
        help="Make plots",
        metavar="Boolean",
    )
    parser.add_option(
        "--apply_parang",
        dest="apply_parang",
        default=True,
        help="Apply parallactic angle correction on beam",
        metavar="Boolean",
    )
    parser.add_option(
        "--cpu_frac",
        dest="cpu_frac",
        default=0.8,
        help="CPU fraction to use",
        metavar="Float",
    )
    parser.add_option(
        "--mem_frac",
        dest="mem_frac",
        default=0.8,
        help="Memory fraction to use",
        metavar="Float",
    )
    parser.add_option(
        "--logfile",
        dest="logfile",
        default=None,
        help="Log file",
        metavar="String",
    )
    parser.add_option(
        "--jobid",
        dest="jobid",
        default=0,
        help="Job ID",
        metavar="Integer",
    )
    (options, args) = parser.parse_args()
    pid=os.getpid()
    save_pid(pid,datadir + f"/pids/pids_{options.jobid}.txt")
    if options.workdir == "":
        print ("Please provide a valid work directory name.")
        return 1
    elif os.path.exists(options.workdir) == False:
        workdir=options.workdir
        os.makedirs(workdir,exist_ok=True)
    else:
        workdir = options.workdir
    logfile=options.logfile
    observer=None
    if os.path.exists(f"{workdir}/jobname_password.npy") and logfile!=None: 
        time.sleep(5)
        jobname,password=np.load(f"{workdir}/jobname_password.npy",allow_pickle=True)
        if os.path.exists(logfile):
            print (f"Starting remote logger. Remote logger password: {password}")
            observer=init_logger("all_pbcor",logfile,jobname=jobname,password=password)
    try:
        if options.imagedir != "" and os.path.exists(options.imagedir):
            msg = pbcor_all_images(
                options.imagedir,
                make_TB=eval(str(options.make_TB)),
                make_plots=eval(str(options.make_plots)),
                apply_parang=eval(str(options.apply_parang)),
                jobid=int(options.jobid),
                cpu_frac=float(options.cpu_frac),
                mem_frac=float(options.mem_frac),
            )
        else:
            print("Please provide correct image directory name.\n")
            msg=1
    except Exception as e:
        traceback.print_exc()
        msg=1
    finally:
        time.sleep(5)
        clean_shutdown(observer)
    return msg


if __name__ == "__main__":
    result = main()
    print(
        "\n###################\nPrimary beam corrections are done.\n###################\n"
    )
    os._exit(result)
