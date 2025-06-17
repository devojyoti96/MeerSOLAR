import sys, psutil, traceback
import os, numpy as np, time, glob
from optparse import OptionParser
from meersolar.pipeline.basic_func import *
from casatasks import casalog
from casatools import msmetadata
from dask import delayed, compute

logfile = casalog.logfile()
os.system("rm -rf " + logfile)

datadir = get_datadir()


def get_polmodel_coeff(model_file):
    """
    Get polarization model coefficients

    Parameters
    ----------
    model_file : str
        Model ascii file name

    Returns
    -------
    float
        Reference frequency in GHz
    float
        Stokes I flux density in Jy at reference frequency
    list
        Stokes I spectrum log-polynomial coefficients
    list
        List of linear polarization fraction polynomial coefficients
    list
        List of linear polarization angle (in radians) coefficients
    """
    freq, I, pfrac, pangle = np.loadtxt(model_file, unpack=True, skiprows=5)
    ref_freq = freq[0]
    x = (freq - ref_freq) / ref_freq
    polyI = np.polyfit(np.log10(x[1:]), np.log10(I[1:]), deg=5)[::-1]
    poly_pfrac = np.polyfit(x, pfrac, deg=5)[::-1]
    poly_pangle = np.polyfit(x, pangle, deg=5)[::-1]
    return ref_freq, I[0], polyI[1:], poly_pfrac, poly_pangle


def polcal_setjy(msname="", field_name="", n_threads=-1, ismms=True, dry_run=False):
    """
    Setjy polcal fields (3C286 or 3C138)

    Parameters
    ----------
    msname : str
        Name of measurement set
    field : str, optional
        Field name
    n_threads : int, optional
        Number of OpenMP threads
    """
    limit_threads(n_threads=n_threads)
    from casatasks import setjy

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    try:
        print(f"Polarization calibrator field name : {field_name}")
        if field_name in ["3C286", "1328+307", "1331+305", "J1331+3030"]:
            ref_freq, I, polyI, poly_pfrac, poly_pangle = get_polmodel_coeff(
                datadir + "/3C286_pol_model.txt"
            )
        elif field_name in ["3C138", "0518+165", "0521+166", "J0521+1638"]:
            ref_freq, I, polyI, poly_pfrac, poly_pangle = get_polmodel_coeff(
                datadir + "/3C138_pol_model.txt"
            )
        else:
            print("Field name is not either of the polcal, 3C286 or 3C138.")
            return 1
        print(f"Reference frequency: {ref_freq} GHz.")
        setjy(
            vis=msname,
            field=field_name,
            scalebychan=True,
            standard="manual",
            fluxdensity=[I, 0.0, 0.0, 0.0],
            spix=polyI,
            reffreq=f"{ref_freq}GHz",
            polindex=poly_pfrac,
            polangle=poly_pangle,
            rotmeas=0,
            usescratch=True,
        )
    except Exception as e:
        traceback.print_exc()
    return


def phasecal_setjy(msname="", field="", ismms=False, n_threads=-1, dry_run=False):
    """
    Setjy phasecal fields

    Parameters
    ----------
    msname : str
        Measurement set
    field : str, optional
        Phasecal fields
    ismms : bool, optional
        Is a multi-ms ot not
    n_threads : int, optional
        Number of OpenMP threads
    """
    limit_threads(n_threads=n_threads)
    from casatasks import setjy

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    try:
        setjy(
            vis=msname,
            field=field,
            standard="manual",
            fluxdensity=[1.0, 0, 0, 0],
            usescratch=True,
            ismms=ismms,
        )
    except Exception as e:
        traceback.print_exc()
    return


def import_fluxcal_models(
    mslist, scans, fluxcal_fields, fluxcal_scans, ncpus=1, mem_frac=0.8
):
    """
    Import model visibilities using crystalball

    Parameters
    ----------
    mslist : list
        List of sub-MS
    scans : list
        Scans of each sub-ms
    fluxcal_fields : list
        Fluxcal fields
    fluxcal_scans : dict
        Fluxcal scans dict
    ncpus : int
        Number of CPU threads to use
    mem_frac : float
        Memory fraction to use

    Returns
    -------
    int
        Success message
    """
    try:
        if len(fluxcal_fields) == 0:
            print("No flux calibrator scan is present.\n")
            return 1
        print("##############################")
        print("Import fluxcal models")
        print("##############################")
        bandname = get_band_name(mslist[0])
        cpu_count = psutil.cpu_count()
        ncpus = max(1, ncpus)
        ncpus = min(ncpus, cpu_count - 1)
        mem_frac = max(mem_frac, 0.8)
        for fluxcal in fluxcal_fields:
            f_scan = fluxcal_scans[fluxcal]
            for s in f_scan:
                pos = np.argmin(np.abs(scans - s))
                sub_msname = mslist[pos]
                modelname = datadir + "/" + fluxcal + "_" + bandname + "_model.txt"
                crys_cmd_args = [
                    "-sm " + modelname,
                    "-f " + fluxcal,
                    "-ns 1000",
                    "-j " + str(ncpus),
                    "-mf " + str(mem_frac),
                ]
                crys_cmd = "crystalball " + " ".join(crys_cmd_args) + " " + sub_msname
                print(crys_cmd)
                tmpfile = f"tmp_{os.path.basename(sub_msname).split('.ms')[0]}"
                msg = os.system(crys_cmd + f" > {tmpfile}")
                os.system(f"rm -rf {tmpfile}")
                if msg == 0:
                    print(f"Fluxcal model is imported successfully for scan: {s}.")
                else:
                    print(f"Error in importing fluxcal model for scan: {s}.")
        return 0
    except Exception as e:
        print("Error in importing model.")
        traceback.print_exc()
        return 1


def import_phasecal_models(
    mslist, scans, phasecal_fields, phasecal_scans, workdir, cpu_frac=0.8, mem_frac=0.8
):
    """
    Import model visibilities for phasecal

    Parameters
    ----------
    mslist : list
        List of sub-MS
    scans : list
        Scans of each sub-ms
    phasecal_fields : list
        Phasecal fields
    phasecal_scans : list
        Phasecal scans
    workdir : str
        Work directory
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
        print("##########################")
        print("Import phasecal models")
        print("##########################")
        task = delayed(phasecal_setjy)(dry_run=True)
        mem_limit = run_limited_memory_task(task)
        dask_client, dask_cluster, n_jobs, n_threads, mem_limit = get_dask_client(
            len(mslist),
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit / 0.6,
        )
        tasks = []
        for phasecal in phasecal_fields:
            ph_scan = phasecal_scans[phasecal]
            for s in ph_scan:
                pos = np.argmin(np.abs(scans - s))
                sub_msname = mslist[pos]
                tasks.append(
                    delayed(phasecal_setjy)(
                        sub_msname,
                        field=phasecal,
                        ismms=True,
                        n_threads=n_threads,
                    )
                )
        results = compute(*tasks)
        dask_client.close()
        dask_cluster.close()
        print("Phasecal models are initiated.\n")
        return 0
    except Exception as e:
        print("Error in initiaing phasecal models.")
        traceback.print_exc()
        return 1


def import_polcal_model(
    mslist, scans, polcal_fields, polcal_scans, workdir, cpu_frac=0.8, mem_frac=0.8
):
    """
    Import model for polarization calibrators (3C286 or 3C138)

    Parameters
    ----------
    mslist : list
        List of sub-MS
    scans : list
        Scans of each sub-ms
    polcal_fields : list
        Polcal fields
    polcal_scans : list
        Polcal scans
    workdir : str
        Work directory
    n_threads : int, optional
        Number of OpenMP threads

    Returns
    -------
    int
        Success message
    """
    try:
        if len(polcal_scans) == 0:
            print("No polarization calibrator scans is present.")
            return 1
        task = delayed(polcal_setjy)(dry_run=True)
        mem_limit = run_limited_memory_task(task)
        dask_client, dask_cluster, n_jobs, n_threads, mem_limit = get_dask_client(
            len(polcal_scans),
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit / 0.6,
        )
        tasks = []
        for polcal_field in polcal_fields:
            p_scan = polcal_scans[polcal_field]
            for s in p_scan:
                pos = np.argmin(np.abs(scans - s))
                sub_msname = mslist[pos]
                tasks.append(
                    delayed(polcal_setjy)(
                        sub_msname, polcal_field, ismms=True, n_threads=n_threads
                    )
                )
        results = compute(*tasks)
        dask_client.close()
        dask_cluster.close()
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1


def import_all_models(msname, workdir, cpu_frac=0.8, mem_frac=0.8):
    """
    Import all calibrator models

    Parameters
    ----------
    msname : str
        Measurement set name
    workdir : str
        Work directory
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use

    Returns
    -------
    int
        Success message
    """
    start_time = time.time()
    cpu_threads = psutil.cpu_count()
    ncpu = int(cpu_threads * cpu_frac)
    mem_frac = 1 - mem_frac
    try:
        msname = msname.rstrip("/")
        correct_missing_col_subms(msname)
        mspath = os.path.dirname(os.path.abspath(msname))
        os.chdir(mspath)
        result = get_submsname_scans(msname)
        if result != None:  # If multi-ms
            mslist, scans = result
            scans = np.array(scans)
        else:
            print("Please provide a multi-MS with scans partitioned.")
            return 1, []
        fluxcal_fields, fluxcal_scans = get_fluxcals(msname)
        phasecal_fields, phasecal_scans, phasecal_flux_list = get_phasecals(msname)
        polcal_fields, polcal_scans = get_polcals(msname)
        fluxcal_result = import_fluxcal_models(
            mslist, scans, fluxcal_fields, fluxcal_scans, ncpus=ncpu, mem_frac=mem_frac
        )
        if fluxcal_result != 0:
            print("##################")
            print("Total time taken : " + str(time.time() - start_time) + "s")
            print("##################\n")
            return fluxcal_result, 1, 1
        phasecal_result = import_phasecal_models(
            mslist,
            scans,
            phasecal_fields,
            phasecal_scans,
            workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
        )
        polcal_result = import_polcal_model(
            mslist,
            scans,
            polcal_fields,
            polcal_scans,
            workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
        )
        if phasecal_result != 0:
            print(
                "Phasecal model was not import, but fluxcal model import is successful."
            )
        if polcal_result != 0:
            print(
                "Polcal model was not imported, but fluxcal model import is successful."
            )
        print("##################")
        print("Total time taken : " + str(time.time() - start_time) + "s")
        print("##################\n")
        return fluxcal_result, phasecal_result, polcal_result
    except Exception as e:
        print("##################")
        print("Total time taken : " + str(time.time() - start_time) + "s")
        print("##################\n")
        traceback.print_exc()
        return 1, 1, 1


def main():
    usage = "Import calibrator models"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--msname",
        dest="msname",
        default=None,
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
    (options, args) = parser.parse_args()
    if options.workdir == "" or os.path.exists(options.workdir) == False:
        workdir = os.path.dirname(os.path.abspath(options.msname)) + "/workdir"
        if os.path.exists(workdir) == False:
            os.makedirs(workdir)
    else:
        workdir = options.workdir
    logfile=options.logfile
    observer=None
    if os.path.exists(f"{workdir}/jobname_password.npy") and logfile!=None: 
        time.sleep(5)
        jobname,password=np.load(f"{workdir}/jobname_password.npy",allow_pickle=True)
        if os.path.exists(logfile):
            print (f"Starting remote logger. Remote logger password: {password}")
            observer=init_logger("import_model",logfile,jobname=jobname,password=password)
    try:
        if options.msname != None and os.path.exists(options.msname):
            fluxcal_result, phasecal_result, polcal_result = import_all_models(
                options.msname,
                options.workdir,
                cpu_frac=float(options.cpu_frac),
                mem_frac=float(options.mem_frac),
            )
            os.system("touch " + workdir + "/.fluxcal_" + str(fluxcal_result))
            os.system("touch " + workdir + "/.phasecal_" + str(phasecal_result))
            os.system("touch " + workdir + "/.polcal_" + str(polcal_result))
            if fluxcal_result == 1 or (
                fluxcal_result == 1 and phasecal_result == 1 and polcal_result == 1
            ):
                msg=1
            else:
                msg=0
        else:
            print("Please provide correct measurement set.\n")
            msg=1
    except Exception as e:
        traceback.print_exc()
        os.system("touch " + workdir + "/.fluxcal_1")
        os.system("touch " + workdir + "/.phasecal_1")
        os.system("touch " + workdir + "/.polcal_1")
        msg=1
    finally:
        time.sleep(5)
        clean_shutdown(observer)
    return msg  
    
if __name__ == "__main__":
    result = main()
    print(
        "\n###################\nVisibility simulation is finished.\n###################\n"
    )
    os._exit(result)
