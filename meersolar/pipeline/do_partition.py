from optparse import OptionParser
import os, time, copy, traceback, gc
from meersolar.pipeline.basic_func import *
from dask import delayed, compute
from casatasks import casalog

logfile = casalog.logfile()
os.system("rm -rf " + logfile)


def single_mstransform(
    msname="",
    outputms="",
    field="",
    scan="",
    width=1,
    timebin="",
    datacolumn="DATA",
    n_threads=-1,
    dry_run=False,
):
    """
    Perform mstransform of a single scan
    Parameters
    ----------
    msname : str
        Name of the measurement set
    outputms : str
        Output ms name
    field : str, optional
        Field name
    scan : str, optional
        Scans to split
    width : int, optional
        Number of channels to average
    timebin : str, optional
        Time to average
    datacolumn : str, optional
        Data column to split
    n_threads : int, optional
        Number of CPU threads
    Returns
    -------
    str
        Output measurement set name
    """
    limit_threads(n_threads=n_threads)
    from casatasks import mstransform

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    print(
        f"Transforming scan : {scan}, channel averaging: {width}, time averaging: {timebin}\n"
    )
    if timebin == "":
        timeaverage = False
    else:
        timeaverage = True
    if width > 1:
        chanaverage = True
    else:
        chanaverage = False
    outputms = outputms.rstrip("/")
    if os.path.exists(outputms):
        os.system("rm -rf " + outputms)
    if os.path.exists(outputms + ".flagversions"):
        os.system("rm -rf " + outputms + ".flagversions")
    try:
        mstransform(
            vis=msname,
            outputvis=outputms,
            field=field,
            scan=scan,
            datacolumn=datacolumn,
            createmms=True,
            timeaverage=timeaverage,
            timebin=timebin,
            chanaverage=chanaverage,
            width=width,
            nthreads=2,
            separationaxis="scan",
            numsubms=1,
        )
        gc.collect()
        return outputms
    except Exception as e:
        if os.path.exists(outputms):
            os.system("rm -rf " + outputms)
        return


def partion_ms(
    msname,
    outputms,
    fields="",
    scans="",
    width=1,
    timebin="",
    datacolumn="DATA",
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Perform mstransform of a single scan
    Parameters
    ----------
    msname : str
        Name of the measurement set
    outputms : str
        Output ms name
    field : str, optional
        Fields to be splited
    scans : str, optional
        Scans to split
    width : int, optional
        Number of channels to average
    timebin : str, optional
        Time to average
    datacolumn : str, optional
        Data column to split
    ncpu : int, optional
        Number of CPU threads to use
    Returns
    -------
    str
        Output multi-measurement set name
    """
    print("##################")
    print("Paritioning measurement set: " + msname)
    print("##################\n")
    print("Determining valid scan list ....")
    from casatools import msmetadata

    start_time = time.time()
    valid_scans = get_valid_scans(msname, min_scan_time=1)
    msmd = msmetadata()
    msname = os.path.abspath(msname.rstrip("/"))
    mspath = os.path.dirname(msname)
    os.chdir(mspath)
    msmd.open(msname)
    if scans != "":
        scan_list = scans.split(",")
    else:
        scan_list = msmd.scannumbers()
    scan_list = [int(i) for i in scan_list]
    if fields != "":  # Filtering scans only in the given fields
        scan_list_field = []
        field_list = []
        for i in fields.split(","):
            try:
                i = int(i)
            except:
                pass
            field_list.append(i)
        for field in field_list:
            a = msmd.scansforfield(field).tolist()
            scan_list_field = scan_list_field + a
        backup_scan_list = copy.deepcopy(scan_list)
        for s in scan_list:
            if s not in scan_list_field or s not in valid_scans:
                backup_scan_list.remove(s)
        scan_list = copy.deepcopy(backup_scan_list)
    else:
        backup_scan_list = copy.deepcopy(scan_list)
        for s in scan_list:
            if s not in valid_scans:
                backup_scan_list.remove(s)
        scan_list = copy.deepcopy(backup_scan_list)
    msmd.close()
    if len(scan_list) == 0:
        print("Please provide at-least one valid scan to split.")
        return

    field_list = []
    msmd = msmetadata()
    msmd.open(msname)
    field_names = msmd.fieldnames()
    for scan in scan_list:
        field = msmd.fieldsforscan(scan)[0]
        field_list.append(str(field_names[field]))
    msmd.close()
    msmd.done()
    field = ",".join(field_list)

    ###########################
    # Dask local cluster setup
    ###########################
    task = delayed(single_mstransform)(dry_run=True)
    mem_limit = run_limited_memory_task(task)
    dask_client, dask_cluster, n_jobs, n_threads = get_dask_client(
        len(scan_list),
        dask_dir = mspath,
        cpu_frac=cpu_frac,
        mem_frac=mem_frac,
        min_mem_per_job=mem_limit / 0.6,
    )
    tasks = []
    for i in range(len(scan_list)):
        scan = scan_list[i]
        outputvis = os.path.dirname(msname) + "/scan_" + str(scan) + ".ms"
        task = delayed(single_mstransform)(
            msname,
            outputvis,
            scan=str(scan),
            field="",
            width=width,
            timebin=timebin,
            n_threads=n_threads,
        )
        tasks.append(task)
    splited_ms_list = compute(*tasks)
    dask_client.close()
    dask_cluster.close()
    splited_ms_list_copy = copy.deepcopy(splited_ms_list)
    for ms in splited_ms_list:
        if ms == None:
            splited_ms_list_copy.remove(ms)
    splited_ms_list = copy.deepcopy(splited_ms_list_copy)
    outputms = outputms.rstrip("/")
    if os.path.exists(outputms):
        os.system("rm -rf " + outputms)
    if os.path.exists(outputms + ".flagversions"):
        os.system("rm -rf " + outputms + ".flagversions")
    if len(splited_ms_list) == 0:
        print("No splited ms to concat.")
    else:
        print("Making multi-MS ....")
        from casatasks import virtualconcat

        virtualconcat(vis=splited_ms_list, concatvis=outputms)

    print("##################")
    print("Total time taken : " + str(time.time() - start_time) + "s")
    print("##################\n")
    gc.collect()
    return outputms


def main():
    usage = "Partition measurement set in multi-MS format"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--msname",
        dest="msname",
        default=None,
        help="Name of measurement set",
        metavar="Measurement Set",
    )
    parser.add_option(
        "--outputms",
        dest="outputms",
        default="multi.ms",
        help="Name of output measurement set",
        metavar="Measurement Set",
    )
    parser.add_option(
        "--fields",
        dest="fields",
        default="",
        help="Field IDs to split",
        metavar="Comma seperated string",
    )
    parser.add_option(
        "--scans",
        dest="scans",
        default="",
        help="Scans to split",
        metavar="Comma seperated string",
    )
    parser.add_option(
        "--width",
        dest="width",
        default=1,
        help="Number of spectral channels to average",
        metavar="Integer",
    )
    parser.add_option(
        "--timebin",
        dest="timebin",
        default="",
        help="Time to average",
        metavar="String",
    )
    parser.add_option(
        "--datacolumn",
        dest="datacolumn",
        default="data",
        help="Datacolumn to split",
        metavar="String",
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
    if options.msname != None and os.path.exists(options.msname):
        try:
            outputms = partion_ms(
                options.msname,
                options.outputms,
                fields=options.fields,
                scans=options.scans,
                width=int(options.width),
                timebin=options.timebin,
                datacolumn=options.datacolumn,
                cpu_frac=float(options.cpu_frac),
                mem_frac=float(options.mem_frac),
            )
        except Exception as e:
            traceback.print_exc()
            return 1
        if outputms == None or os.path.exists(outputms) == False:
            print("Error in partitioning measurement set.")
            return 1
        else:
            print("Partitioned multi-MS is created at: ", outputms)
            return 0
    else:
        print("Please provide correct measurement set.\n")
        return 1


if __name__ == "__main__":
    result = main()
    print(
        "\n###################\nMeasurement set partitioning is finished.\n###################\n"
    )
    os._exit(result)
