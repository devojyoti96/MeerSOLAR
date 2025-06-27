from meersolar.pipeline.basic_func import *
import argparse, os, glob, sys, traceback, time


def make_ds(
    msname,
    workdir,
    outdir,
    extension="png",
    target_scans=[],
    merge_scans=True,
    seperate_scans=True,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Make all dynamic spectra of the solar scans

    Parameters
    ----------
    msname : str
        Measurement set name
    workdir : str
        Work directory
    outdir : str
        Output directory
    extension : str, optional
        Plot file extension
    target_scans : list, optional
        Target scans
    merge_scans: bool, optional
        Merge scans
    seperate_scans : bool, optional
        Seperate scans
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use

    Returns
    -------
    list
        Dynamic spectra files
    """
    msname = msname.rstrip("/")
    workdir = workdir.rstrip("/")
    if seperate_scans == False and merge_scans == False:
        return
    try:
        if seperate_scans:
            make_solar_DS(
                msname,
                workdir,
                extension=extension,
                target_scans=target_scans,
                merge_scan=False,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
            )
        if merge_scans:
            make_solar_DS(
                msname,
                workdir,
                extension=extension,
                target_scans=target_scans,
                merge_scan=True,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
            )
        if os.path.samefile(outdir,workdir)==False:
            os.system(f"mv {workdir}/dynamic_spectra {outdir}")
        ds_file_name = os.path.basename(msname).split(".ms")[0] + "_DS"
        ds_files = glob.glob(f"{outdir}/dynamic_spectra/{ds_file_name}*.{extension}")
        return ds_files
    except Exception as e:
        traceback.print_exc()
        return []
    finally:
        time.sleep(5)
        drop_cache(msname)
        drop_cache(workdir)


def main():
    parser = argparse.ArgumentParser(
        description="Make dynamic spectra of solar scans",
        formatter_class=SmartDefaultsHelpFormatter,
    )
    # === Essential parameters ===
    essential = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    essential.add_argument("msname", type=str, help="Measurement set name")
    essential.add_argument(
        "--workdir",
        type=str,
        dest="workdir",
        required=True,
        help="Working directory",
    )
    essential.add_argument(
        "--outdir",
        type=str,
        dest="outdir",
        required=True,
        help="Output directory",
    )
    

    # === Advanced parameters ===
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--extension",
        type=str,
        default="png",
        help="Save file extension",
    )
    adv_args.add_argument(
        "--target_scans",
        nargs="*",
        type=str,
        default=[],
        help="List of target scans to process (space-separated, e.g. 3 5 7)",
    )
    adv_args.add_argument(
        "--no_merge",
        action="store_false",
        dest="merge",
        help="Do not merge scans",
    )
    adv_args.add_argument(
        "--no_seperate",
        action="store_false",
        dest="seperate",
        help="Do not seperate scans",
    )

    # === Advanced local system/ per node hardware resource parameters ===
    hard_args = parser.add_argument_group(
        "###################\nHardware resource management parameters\n###################"
    )
    hard_args.add_argument(
        "--cpu_frac",
        type=float,
        default=0.8,
        help="Fraction of CPU usuage per node",
    )
    hard_args.add_argument(
        "--mem_frac",
        type=float,
        default=0.8,
        help="Fraction of memory usuage per node",
    )
    hard_args.add_argument(
        "--logfile", type=str, default=None, help="Optional path to log file"
    )
    hard_args.add_argument(
        "--jobid", type=str, default="0", help="Job ID for logging and PID tracking"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    pid = os.getpid()
    save_pid(pid, datadir + f"/pids/pids_{args.jobid}.txt")

    if args.workdir == "" or not os.path.exists(args.workdir):
        first_ms = args.mslist.split(",")[0]
        workdir = os.path.dirname(os.path.abspath(first_ms)) + "/workdir"
    else:
        workdir = args.workdir
    os.makedirs(workdir,exist_ok=True)

    os.makedirs(workdir + "/logs/",exist_ok=True)

    logfile = args.logfile
    observer = None

    if os.path.exists(f"{workdir}/jobname_password.npy") and logfile is not None:
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            observer = init_logger(
                "ds_plot", logfile, jobname=jobname, password=password
            )

    try:
        if args.msname != "" and os.path.exists(args.msname):
            ds_files = make_ds(
                args.msname,
                args.workdir,
                args.outdir,
                extension=args.extension,
                target_scans=args.target_scans,
                merge_scans=args.merge,
                seperate_scans=args.seperate,
                cpu_frac=float(args.cpu_frac),
                mem_frac=float(args.mem_frac),
            )
            msg = 0
        else:
            print("Please provide a valid measurement set.\n")
            msg = 1
    except Exception as e:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        clean_shutdown(observer)
    return msg


if __name__ == "__main__":
    result = main()
    print(
        "\n###################\nDynamic spectra are produced successfully.\n###################\n"
    )
    os._exit(result)
