import numpy as np, glob, os, copy, warnings, traceback, gc
from casatools import table
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from meersolar.pipeline.basic_func import *
from meersolar.pipeline.do_apply_basiccal import applysol
from dask import delayed, compute
from optparse import OptionParser
from casatasks import casalog

logfile = casalog.logfile()
os.system("rm -rf " + logfile)

def run_all_applysol(
    mslist,
    workdir,
    caldir,
    overwrite_datacolumn=False,
    applymode="calonly",
    force_apply=False,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Apply self-calibrator solutions on all target scans
    Parameters
    ----------
    mslist : str
        Measurement set list
    workdir : str
        Working directory
    caldir : str
        Calibration directory
    overwrite_datacolumn : bool, optional
        Overwrite data column or not
    applymode : str, optional
        Apply mode
    force_apply : bool, optional
        Force to apply solutions even already applied 
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    Returns
    --------
    list
        Calibrated target scans
    """
    start_time = time.time()
    try:
        os.chdir(workdir)
        mslist = np.unique(mslist).tolist()
        parang = False
        gaintable=glob.glob(caldir+"/*cal")
        gaintable_bkp = copy.deepcopy(gaintable)
        for g in gaintable_bkp:
            if "selfcal" not in g:
                gaintable.remove(g)
        del gaintable_bkp
        
        ####################################
        # Filtering any corrupted ms
        #####################################    
        filtered_mslist=[] # Filtering in case any ms is corrupted
        for ms in mslist:
            checkcol=check_datacolumn_valid(ms)
            if checkcol:
                filtered_mslist.append(ms)
            else:
                print (f"Issue in : {ms}")
                os.system("rm -rf {ms}")
        mslist=filtered_mslist  
        if len(mslist)==0:
            print ("No valid measurement set.")
            print(f"Total time taken: {round(time.time()-start_time,2)}s")
            return 1  
               
        ####################################
        # Applycal jobs
        ####################################
        print(f"Total ms list: {len(mslist)}")
        task = delayed(applysol)(dry_run=True)
        mem_limit = run_limited_memory_task(task, dask_dir = workdir)
        ms_size_list=[get_ms_size(ms)+mem_limit for ms in mslist]
        mem_limit=max(ms_size_list)
        dask_client, dask_cluster, n_jobs, n_threads, mem_limit = get_dask_client(
            len(mslist),
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit / 0.6,
        )
        tasks = []
        msmd = msmetadata()
        for ms in mslist:
            tasks.append(
                delayed(applysol)(
                    msname=ms,
                    gaintable=gaintable,
                    overwrite_datacolumn=overwrite_datacolumn,
                    applymode=applymode,
                    interp=["linear"],
                    n_threads=n_threads,
                    parang=parang,
                    memory_limit=mem_limit,
                    force_apply=force_apply,
                    soltype="selfcal",
                )
            )
        results = compute(*tasks)
        dask_client.close()
        dask_cluster.close()
        if np.nansum(results) == 0:
            print("##################")
            print(
                "Applying self-calibration solutions for target scans are done successfully."
            )
            print("Total time taken : ", time.time() - start_time)
            print("##################\n")
            return 0
        else:
            print("##################")
            print(
                "Applying self-calibration solutions for target scans are not done successfully."
            )
            print("Total time taken : ", time.time() - start_time)
            print("##################\n")
            return 1
    except Exception as e:
        traceback.print_exc()
        os.system("rm -rf casa*log")
        print("##################")
        print(
            "Applying self-calibration solutions for target scans are not done successfully."
        )
        print("Total time taken : ", time.time() - start_time)
        print("##################\n")
        return 1


def main():
    usage = "Apply self-calibration solutions of target scans"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--mslist",
        dest="mslist",
        default="",
        help="Comma seperated list of measurement sets",
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
        "--caldir",
        dest="caldir",
        default="",
        help="Caltable directory",
        metavar="String",
    )
    parser.add_option(
        "--applymode",
        dest="applymode",
        default="calflag",
        help="Applycal mode",
        metavar="String",
    )
    parser.add_option(
        "--overwrite_datacolumn",
        dest="overwrite_datacolumn",
        default=False,
        help="Overwrite data column or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--force_apply",
        dest="force_apply",
        default=False,
        help="Force to apply solutions even it is already applied",
        metavar="Boolean",
    )
    parser.add_option(
        "--print_casalog",
        dest="print_casalog",
        default=False,
        help="Print CASA log",
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
    (options, args) = parser.parse_args()
    if eval(str(options.print_casalog)) == True:
        casalog.showconsole(True)
    if options.mslist != "":
        print("\n###################################")
        print("Starting applying solutions...")
        print("###################################\n")
        try:
            if options.workdir == "" or os.path.exists(options.workdir) == False:
                print("Provide existing work directory name.")
                return 1
            if options.caldir == "" or os.path.exists(options.caldir) == False:
                print("Provide existing caltable directory.")
                return 1
            msg = run_all_applysol(
                options.mslist.split(","),
                options.workdir,
                options.caldir,
                overwrite_datacolumn=eval(str(options.overwrite_datacolumn)),
                applymode=options.applymode,
                force_apply=eval(str(options.force_apply)),
                cpu_frac=float(options.cpu_frac),
                mem_frac=float(options.mem_frac),
            )
            return msg
        except Exception as e:
            traceback.print_exc()
            return 1
    else:
        print("Please provide valid measurement set list.\n")
        return 1


if __name__ == "__main__":
    result = main()
    print(
        "\n###################\nApplying self-calibration solutions are done.\n###################\n"
    )
    os._exit(result)
