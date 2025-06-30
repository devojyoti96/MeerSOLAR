import os, glob, traceback, argparse, time
from meersolar.pipeline.single_image_meerpbcor import get_pbcor_image
from meersolar.pipeline.basic_func import *
from dask import delayed, compute
from casatasks import casalog

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except:
    pass


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
    cmd = f"run_single_meerpbcor {imagename} --pbdir {pbdir} --pbcor_dir {pbcor_dir} --ncpu {ncpu} --jobid {jobid}"
    print (cmd)
    if apply_parang == False:
        cmd += " --no_apply_parang"
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
    imagedir=imagedir.rstrip("/")
    pbdir = f"{os.path.dirname(imagedir)}/pbdir"
    pbcor_dir = f"{os.path.dirname(imagedir)}/pbcor_images"
    os.makedirs(pbdir, exist_ok=True)
    os.makedirs(pbcor_dir, exist_ok=True)
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
                task = delayed(run_pbcor)(
                    image, pbdir, pbcor_dir, apply_parang, jobid=jobid, ncpu=n_threads
                )
                tasks.append(task)
            results = compute(*tasks)
            dask_client.close()
            dask_cluster.close()
            for r in results:
                if r == 0:
                    successful_pbcor += 1
                    
        ############################################
        # Saving fits in helioprojective coordinates
        ############################################
        if successful_pbcor>0:
            hpcdir=f"{pbcor_dir}/hpcs"
            pbcor_images = glob.glob(f"{pbcor_dir}/*.fits")
            os.makedirs(hpcdir, exist_ok=True)
            print("Saving primary beam corrected images helioprojective coordinates...")
            for image in pbcor_images:
                save_in_hpc(image,outdir=hpcdir)
            if make_plots:
                print("Making plots of primary beam corrected images ...")
                pngdir = f"{pbcor_dir}/pngs"
                pdfdir = f"{pbcor_dir}/pdfs"
                os.makedirs(pngdir, exist_ok=True)
                os.makedirs(pdfdir, exist_ok=True)
                for image in pbcor_images:
                    try:
                        outimages=plot_in_hpc(
                            image,
                            draw_limb=True,
                            extensions=["png","pdf"],
                            outdirs=[pngdir,pdfdir]
                        )
                    except:
                        junkpng=f"{pngdir}/{os.path.basename(image).split('.fits')[0]}.png.junk"
                        junkpdf=f"{pngdir}/{os.path.basename(image).split('.fits')[0]}.pdf.junk"
                        os.system(f"touch {junkpng}")
                        os.system(f"touch {junkpdf}")
                                  
        ####################################
        # Making brightness temperature maps
        ####################################
        if successful_pbcor> 0 and make_TB:
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
            hpcdir=f"{tb_dir}/hpcs"
            tb_images = glob.glob(f"{tb_dir}/*.fits")
            os.makedirs(hpcdir, exist_ok=True)
            print("Saving brightness temperature maps helioprojective coordinates...")
            for image in tb_images:
                save_in_hpc(image,outdir=hpcdir)
                
            if make_plots:
                print("Making plots of brightness temperature maps..")
                pngdir = f"{tb_dir}/pngs"
                pdfdir = f"{tb_dir}/pdfs"
                os.makedirs(pngdir, exist_ok=True)
                os.makedirs(pdfdir, exist_ok=True)
                for image in tb_images:
                    outimages=plot_in_hpc(
                        image,
                        draw_limb=True,
                        extensions=["png","pdf"],
                        outdirs=[pngdir,pdfdir]
                    )
            
        #########################################
        # Final calculations
        #########################################
        print(f"Total input images: {len(images)}")
        print(f"Total corrected images: {len(pbcor_images)}")
        if make_TB:
            print(f"Total brightness temperatures maps: {len(tb_images)}") 
        os.system(f"rm -rf {pbdir}/dask-scratch-space")
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1
    finally:
        time.sleep(5)
        drop_cache(imagedir)
        drop_cache(pbcor_dir)
        os.system(f"rm -rf {pbdir}")


def main():
    parser = argparse.ArgumentParser(
        description="Correct all images for MeerKAT full-pol averaged primary beam",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    ## Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument("imagedir", help="Path to image directory")
    basic_args.add_argument("--workdir", default="", help="Path to work directory")

    ## Advanced parameters
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
        help="Do not make png and pdf plots",
    )
    adv_args.add_argument(
        "--no_apply_parang",
        action="store_false",
        dest="apply_parang",
        help="Do not apply parallactic angle correction",
    )

    ## Resource management parameters
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
    
    if args.workdir == "" or not os.path.exists(args.workdir):
        workdir = os.path.dirname(os.path.abspath(args.msname)) + "/workdir"
    else:
        workdir = args.workdir
    os.makedirs(workdir,exist_ok=True)

    pid = os.getpid()
    meersolar_cachedir = get_meersolar_cachedir()
    save_pid(pid, f"{meersolar_cachedir}/pids/pids_{args.jobid}.txt")

    logfile = args.logfile
    observer = None

    if os.path.exists(f"{workdir}/jobname_password.npy") and logfile is not None:
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            observer = init_logger(
                "all_pbcor", logfile, jobname=jobname, password=password
            )

    try:
        if os.path.exists(args.imagedir):
            msg = pbcor_all_images(
                args.imagedir,
                make_TB=args.make_TB,
                make_plots=args.make_plots,
                apply_parang=args.apply_parang,
                jobid=args.jobid,
                cpu_frac=args.cpu_frac,
                mem_frac=args.mem_frac,
            )
        else:
            print("Please provide correct image directory path.")
            msg = 1
    except Exception:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        drop_cache(args.imagedir)
        drop_cache(workdir)
        clean_shutdown(observer)

    return msg


if __name__ == "__main__":
    result = main()
    print(
        "\n###################\nPrimary beam corrections are done.\n###################\n"
    )
    os._exit(result)
