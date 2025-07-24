import logging
import dask
import numpy as np
import argparse
import traceback
import warnings
import time
import glob
import sys
import os
from casatasks import casalog
from casatools import msmetadata, ms as casamstool
from dask import delayed
from dask.distributed import Client
from meersolar.utils import *

logging.getLogger("distributed").setLevel(logging.WARNING)

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    pass

datadir = get_datadir()


def make_solar_DS(
    msname,
    workdir,
    ds_file_name="",
    extension="png",
    target_scans=[],
    scans=[],
    merge_scan=False,
    showgui=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    dask_addr=None,
):
    """
    Make solar dynamic spectrum and plots

    Parameters
    ----------
    msname : str
        Measurement set name'
    workdir : str
        Work directory
    ds_file_name : str, optional
        DS file name prefix
    extension : str, optional
        Image file extension
    target_scans : list, optional
        Target scans
    scans : list, optional
        Scan list
    merge_scan : bool, optional
        Merge scans in one plot or not
    showgui : bool, optional
        Show GUI
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    dask_addr : str, optional
        Dask scheduler address
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(f"{workdir}/dynamic_spectra", exist_ok=True)
    print("##############################################")
    print(f"Start making dynamic spectra for ms: {msname}")
    print("##############################################")
    if len(target_scans) > 0:
        temp_target_scans = []
        for s in target_scans:
            temp_target_scans.append(int(s))
        target_scans = temp_target_scans

    ##################################
    # Making and ploting
    ##################################
    if len(scans) == 0:
        scans, cal_scans, f_scans, g_scans, p_scans = get_cal_target_scans(msname)
    valid_scans = get_valid_scans(msname)
    final_scans = []
    scan_size_list = []
    msmd = msmetadata()
    mstool = casamstool()
    for scan in scans:
        if scan in valid_scans:
            if len(target_scans) == 0 or (
                len(target_scans) > 0 and int(scan) in target_scans
            ):
                final_scans.append(int(scan))
                msmd.open(msname)
                nchan = msmd.nchan(0)
                nant = msmd.nantennas()
                msmd.close()
                mstool.open(msname)
                mstool.select({"scan_number": int(scan)})
                nrow = mstool.nrow(True)
                mstool.close()
                nbaselines = int(nant + (nant * (nant - 1) / 2))
                scan_size = (5 * (nrow / nbaselines) * 16) / (1024**3)
                scan_size_list.append(scan_size)
    if len(final_scans) == 0:
        print("No scans to make dynamic spectra.")
        return
    del scans
    scans = sorted(final_scans)
    print(f"Scans: {scans}")
    msname = msname.rstrip("/")
    if ds_file_name == "":
        ds_file_name = os.path.basename(msname).split(".ms")[0] + "_DS"
    hascor = check_datacolumn_valid(msname, datacolumn="CORRECTED_DATA")
    if hascor:
        datacolumn = "CORRECTED_DATA"
    else:
        datacolumn = "DATA"
    mspath = os.path.dirname(msname)
    mem_limit = max(scan_size_list)
    if dask_addr is None:
        dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = get_dask_client(
            len(scans),
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit,
        )
    else:
        _, _, n_jobs, n_threads, mem_limit, dask_dir = get_dask_client(
            len(scans),
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit,
            only_cal=True,
        )
        dask_client=Client(address=dask_addr)
        os.system(f"rm -rf {dask_dir}")
    wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
    tasks = []
    for scan in scans:
        tasks.append(
            delayed(make_ds_file_per_scan)(
                msname,
                f"{workdir}/dynamic_spectra/{ds_file_name}_scan_{scan}",
                scan,
                datacolumn,
            )
        )
    futures = dask_client.compute(tasks)
    results = list(dask_client.gather(futures))
    dask_client.close()
    if dask_addr is None:
        dask_cluster.close()
        os.system(f"rm -rf {dask_dir}")
    ds_files = [
        f"{workdir}/dynamic_spectra/{ds_file_name}_scan_{scan}.npy" for scan in scans
    ]
    print(f"DS files: {ds_files}")
    if not merge_scan:
        plots = []
        for dsfile in ds_files:
            plot_file = make_ds_plot(
                [dsfile],
                plot_file=dsfile.replace(".npy", f".{extension}"),
                showgui=showgui,
            )
            plots.append(plot_file)
    else:
        plot_file = make_ds_plot(
            ds_files,
            plot_file=f"{workdir}/dynamic_spectra/{ds_file_name}.{extension}",
            showgui=showgui,
        )
    goes_files = glob.glob(f"{workdir}/dynamic_spectra/sci*.nc")
    for f in goes_files:
        os.system(f"rm -rf {f}")
    return


def make_dsfiles(
    msname,
    workdir,
    outdir,
    extension="png",
    target_scans=[],
    merge_scans=True,
    seperate_scans=True,
    cpu_frac=0.8,
    mem_frac=0.8,
    dask_addr=None,
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
    dask_addr : str, optional
        Dask scheduler address

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
                dask_addr=dask_addr,
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
                dask_addr=dask_addr,
            )
        if os.path.samefile(outdir, workdir) == False:
            os.makedirs(f"{outdir}/dynamic_spectra",exist_ok=True)
            os.system(f"mv {workdir}/dynamic_spectra/* {outdir}/dynamic_spectra/")
            os.system(f"rm -rf {workdir}/dynamic_spectra")
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


def main(
    msname,
    workdir,
    outdir,
    extension="png",
    target_scans=[],
    merge=True,
    seperate=True,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid="0",
    start_remote_log=False,
    dask_addr=None,
):
    """
    Make dynamic spectra

    Parameters
    ----------
    msname : str
        Measurement set
    workdir : str
        Work directory
    outdir : str
        Output directory
    extension : str, optional
        Plot extension
    target_scans : list, optional
        Target scans
    merge : bool, optional
        Merge scans plot
    seperate : bool, optional
        Seperate scan plots
    cpu_frac : float, optional
        CPU fraction
    mem_frac : float, optional
        Memory fraction
    logfile : str, optional
        Log file
    jobid : str, optional
        Job ID
    start_remote_log : bool, optional
        Start remote log
    dask_addr: str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success messsage
    """
    pid = os.getpid()
    cachedir = get_cachedir()
    save_pid(pid, f"{cachedir}/pids/pids_{jobid}.txt")

    if workdir == "":
        workdir = os.path.dirname(os.path.abspath(msname)) + "/workdir"
    os.makedirs(workdir, exist_ok=True)

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
                "ds_plot", logfile, jobname=jobname, password=password
            )
    ###########

    try:
        if msname != "" and os.path.exists(msname):
            ds_files = make_dsfiles(
                msname,
                workdir,
                outdir,
                extension=extension,
                target_scans=target_scans,
                merge_scans=merge,
                seperate_scans=seperate,
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                dask_addr=dask_addr,
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


def cli():
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
    adv_args.add_argument(
        "--start_remote_log", action="store_true", help="Start remote logging"
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

    msg = main(
        msname=args.msname,
        workdir=args.workdir,
        outdir=args.outdir,
        extension=args.extension,
        target_scans=args.target_scans,
        merge=args.merge,
        seperate=args.seperate,
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
        "\n###################\nDynamic spectra are produced successfully.\n###################\n"
    )
    os._exit(result)
