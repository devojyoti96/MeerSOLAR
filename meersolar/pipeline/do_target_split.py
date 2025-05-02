import numpy as np, os, time, traceback, gc
from meersolar.pipeline.basic_func import *
from optparse import OptionParser
from casatasks import casalog
from dask import delayed, compute

logfile = casalog.logfile()
os.system("rm -rf " + logfile)


def split_scan(
    msname="",
    outputvis="",
    scan="",
    width="",
    timebin="",
    datacolumn="",
    spw="",
    timerange="",
    n_threads=-1,
    dry_run=False,
):
    """
    Split a single target scan
    Parameters
    ----------
    msname : str
        Measurement set
    outputvis : str
        Output measurement set
    scan : int
        Scan number
    width : int
        Channel width
    timebin : str
        Timebin width
    datacolumn : str
        Datacolumn to split
    spw : str, optional
        Spectral window to split
    timerange : str, optional
        Time range to split
    n_threads : int, optional
        Number of OpenMP threads
    Returns
    -------
    str
        Splited measurement set
    """
    limit_threads(n_threads=n_threads)
    from casatasks import split, clearcal, initweights
    from casatools import msmetadata

    msmd = msmetadata()
    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msmd.open(msname)
    fields = msmd.fieldsforscan(int(scan))
    msmd.close()
    del msmd
    fields_str = ""
    for f in fields:
        fields_str += str(f) + ","
    fields_str = fields_str[:-1]

    if os.path.exists(outputvis):
        os.system("rm -rf " + outputvis)
    if os.path.exists(outputvis + ".flagversions"):
        os.system("rm -rf " + outputvis + ".flagversions")
    print(f"Spliting scan : {scan} of ms: {msname}\n")
    print(
        f"split(vis='{msname}',outputvis='{outputvis}',field='{fields_str}',scan={scan},spw='{spw}',timerange='{timerange}',width={width},timebin='{timebin}',datacolumn='{datacolumn}')\n"
    )
    split(
        vis=msname,
        outputvis=outputvis,
        field=fields_str,
        scan=scan,
        spw=spw,
        timerange=timerange,
        width=width,
        timebin=timebin,
        datacolumn=datacolumn,
    )
    clearcal(vis=outputvis, addmodel=True)
    initweights(vis=outputvis, wtmode="ones", dowtsp=True)
    return outputvis


def split_target_scans(
    msname,
    workdir,
    timeres,
    freqres,
    datacolumn,
    spw="",
    chanchunk=-1,
    timerange="",
    scans=[],
    do_only_sidereal_cor=False,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Split target scans
    Parameters
    ----------
    msname : str
        Measurement set
    workdir : str
        Work directory
    timeres : float
        Time resolution in seconds
    freqres : float
        Frequency resolution in MHz
    datacolumn : str
        Data column to split
    spw : str, optional
        Spectral window
    chanchunk : int, optional
        Number of spectral chunks
    timerange : str, optional
        Time range
    scans : list
        Scan list to split
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    Returns
    -------
    list
        Splited ms list
    """
    start_time = time.time()
    try:
        os.chdir(workdir)
        print(f"Spliting ms : {msname}")
        target_scans, cal_scans, f_scans, g_scans, p_scans = get_cal_target_scans(
            msname
        )
        valid_scans = get_valid_scans(msname)
        filtered_scan_list = []
        for scan in target_scans:
            if scan in valid_scans:
                if len(scans) == 0 or (len(scans) > 0 and scan in scans):
                    filtered_scan_list.append(scan)

        #######################################
        # Extracting time frequency information
        #######################################
        from casatools import msmetadata

        msmd = msmetadata()
        msmd.open(msname)
        chanres = msmd.chanres(0, unit="MHz")[0]
        nchan = msmd.nchan(0)
        msmd.close()
        if freqres > 0:  # Image resolution is in MHz
            chanwidth = int(freqres / chanres)
            if chanwidth < 1:
                chanwidth = 1
        else:
            chanwidth = 1
        if timeres > 0:  # Image resolution is in seconds
            timebin = str(timeres) + "s"
        else:
            timebin = ""

        #############################
        # Making spectral chunks
        #############################
        bad_spws = get_bad_chans(msname).split("0:")[-1].split(";")
        good_spws = []
        for i in range(len(bad_spws) - 1):
            start_chan = int(bad_spws[i].split("~")[-1]) + 1
            end_chan = int(bad_spws[i + 1].split("~")[0]) - 1
            good_spws.append(f"{start_chan}~{end_chan}")
        if spw != "":
            good_spws = "0:" + ";".join(good_spws)
            common_spws = get_common_spw(good_spws, spws)
            good_spws = common_spws.split("0:")[-1].split(";")
        chanlist = []
        if chanchunk > 0:
            nchan_per_chunk = int(nchan / chanchunk)
            for good_spw in good_spws:
                start_chan = int(good_spw.split("~")[0])
                end_chan = int(good_spw.split("~")[-1])
                for s in range(start_chan, end_chan, nchan_per_chunk):
                    e = s + nchan_per_chunk - 1
                    if e > end_chan:
                        e = end_chan
                chanlist.append(f"{s}~{e}")
        else:
            chanlist = good_spws
        ##################################
        # Time range
        ##################################
        if timerange != "":
            scan_timerange_dic = scans_in_timerange(msname, timerange)
            scan_list = list(scan_timerange_dic.keys())
            filtered_scan_list_bkp = copy.deepcopy(filtered_scan_list)
            for s in filtered_scan_list_bkp:
                if s not in scan_list:
                    filtered_scan_list.remove(s)
            del filtered_scan_list_bkp
        else:
            scan_timerange_dic = {}

        ##################################
        # Parallel spliting
        ##################################
        if len(chanlist) > 0:
            total_chunks = len(chanlist) * len(filtered_scan_list)
        else:
            total_chunks = len(filtered_scan_list)
        if do_only_sidereal_cor == False:
            #############################################
            # Memory limit
            #############################################
            task = delayed(split_scan)(dry_run=True)
            mem_limit = run_limited_memory_task(task)
            #######################
            dask_client, dask_cluster, n_jobs, n_threads = get_dask_client(
                total_chunks,
                dask_dir=workdir,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                min_mem_per_job=mem_limit / 0.6,
            )
            tasks = []
            for scan in filtered_scan_list:
                if timerange != "":
                    time_range = scan_timerange_dic[scan]
                else:
                    time_range = ""
                for chanrange in chanlist:
                    outputvis = f"{workdir}/target_scan_{scan}_spw_{chanrange}.ms"
                    task = delayed(split_scan)(
                        msname,
                        outputvis,
                        scan,
                        chanwidth,
                        timebin,
                        datacolumn,
                        spw="0:" + chanrange,
                        timerange=time_range,
                        n_threads=n_threads,
                    )
                    tasks.append(task)
            splited_ms_list = compute(*tasks)
            dask_client.close()
            dask_cluster.close()
        else:
            ########################################
            # Correcting solar differential rotation
            ########################################
            splited_ms_list = []
            for scan in filtered_scan_list:
                if timerange != "":
                    time_range = scan_timerange_dic[scan]
                else:
                    time_range = ""
                for chanrange in chanlist:
                    outputvis = f"{workdir}/target_scan_{scan}_spw_{chanrange}.ms"
                    if os.path.exists(outputvis):
                        data_valid = check_datacolumn_valid(
                            outputvis, datacolumn="DATA"
                        )
                        if data_valid:
                            splited_ms_list.append(outputvis)
                        else:
                            os.system("rm -rf " + outputvis)
                if len(splited_ms_list) == 0:
                    print("No good splited ms is present. Starting with spliting.")
                    do_only_sidereal_cor = False

        #############################################
        # Memory limit
        #############################################
        task = delayed(correct_solar_sidereal_motion)(dry_run=True)
        mem_limit = run_limited_memory_task(task)
        #######################
        splited_ms_list_phaserotated = []
        dask_client, dask_cluster, n_jobs, n_threads = get_dask_client(
            len(splited_ms_list),
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit / 0.6,
        )
        tasks = []
        for ms in splited_ms_list:
            tasks.append(delayed(correct_solar_sidereal_motion)(ms))
        results = compute(*tasks)
        dask_client.close()
        dask_cluster.close()
        for i in range(len(results)):
            msg = results[i]
            if msg == 0:
                splited_ms_list_phaserotated.append(splited_ms_list[i])
        if len(splited_ms_list_phaserotated) == 0:
            print(
                "Sidereal motion correction is not successful for any measurement set."
            )
            print("##################")
            print("Spliting of target scans are done successfully.")
            print("Total time taken : ", time.time() - start_time)
            print("##################\n")
            return 0, splited_ms_list
        else:
            print("##################")
            print("Spliting of target scans are done successfully.")
            print("Total time taken : ", time.time() - start_time)
            print("##################\n")
            return 0, splited_ms_list_phaserotated
    except Exception as e:
        traceback.print_exc()
        print("##################")
        print("Spliting of target scans are unsuccessful.")
        print("Total time taken : ", time.time() - start_time)
        print("##################\n")
        return 1, []


def main():
    usage = "Split target scans"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--msname",
        dest="msname",
        default="",
        help="Name of measurement set",
        metavar="Measurement Set",
    )
    parser.add_option(
        "--workdir",
        dest="workdir",
        default="",
        help="Name of work directory",
        metavar="String",
    )
    parser.add_option(
        "--datacolumn",
        dest="datacolumn",
        default="data",
        help="Data column to split",
        metavar="String",
    )
    parser.add_option(
        "--spw",
        dest="spw",
        default="",
        help="Spectral window to split",
        metavar="String",
    )
    parser.add_option(
        "--timerange",
        dest="timerange",
        default="",
        help="Timerange to split",
        metavar="String",
    )
    parser.add_option(
        "--scans",
        dest="scans",
        default="",
        help="Target scan list (default: all)",
        metavar="String",
    )
    parser.add_option(
        "--nchan_chunck",
        dest="nchan_chunck",
        default=-1,
        help="Number of spectral chunk (default : unflag chan blocks)",
        metavar="Integer",
    )
    parser.add_option(
        "--print_casalog",
        dest="print_casalog",
        default=False,
        help="Print CASA log",
        metavar="Boolean",
    )
    parser.add_option(
        "--freqres",
        dest="freqres",
        default=-1,
        help="Frequency to average in MHz",
        metavar="Float",
    )
    parser.add_option(
        "--timeres",
        dest="timeres",
        default=-1,
        help="Time bin to average in seconds",
        metavar="Float",
    )
    parser.add_option(
        "--only_sidereal_cor",
        dest="only_sidereal_cor",
        default=False,
        help="Correct only sidereal motion",
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
    if options.msname != "" and os.path.exists(options.msname):
        print("\n###################################")
        print("Start spliting target scans.")
        print("###################################\n")
        try:
            if options.workdir == "" or os.path.exists(options.workdir) == False:
                print("Provide existing work directory name.")
                return 1
            if options.scans != "":
                scans = [int(i) for i in options.scans.split(",")]
            else:
                scans = []
            msg, final_target_mslist = split_target_scans(
                options.msname,
                options.workdir,
                float(options.timeres),
                float(options.freqres),
                options.datacolumn,
                spw=options.spw,
                timerange=options.timerange,
                scans=scans,
                do_only_sidereal_cor=eval(str(options.only_sidereal_cor)),
                chanchunk=int(options.nchan_chunck),
                cpu_frac=float(options.cpu_frac),
                mem_frac=float(options.mem_frac),
            )
            return msg
        except Exception as e:
            traceback.print_exc()
            return 1
    else:
        print("Please provide correct measurement set.\n")
        return 1


if __name__ == "__main__":
    result = main()
    print(
        "\n###################\nSpliting target scans are done.\n###################\n"
    )
    os._exit(result)
