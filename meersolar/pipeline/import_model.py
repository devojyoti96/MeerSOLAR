import sys, psutil, traceback
import os, numpy as np, time, glob
from optparse import OptionParser
from meersolar.pipeline.basic_func import *
from casatasks import casalog
from casatools import msmetadata
from dask import delayed, compute

logfile = casalog.logfile()
os.system("rm -rf " + logfile)

"""
This code is written by Devojyoti Kansabanik, Jul 30, 2023
"""

#####################################################
# Polarization models needs to modified for final use
#####################################################

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


def polcal_setjy(msname="", scan="", n_threads=-1, dry_run=False):
    """
    Setjy polcal fields (3C286 or 3C138)
    Parameters
    ----------
    msname : str
        Name of measurement set
    scan : str
        Scan name
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
        msmd = msmetadata()
        msmd.open(msname)
        field_names = msmd.fieldnames()
        field = msmd.fieldsforscan(int(scan))[0]
        msmd.close()
        field_name = field_names[field]
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


def import_fluxcal_models(msname, ncpus=1, mem_frac=0.8):
    """
    Import model visibilities using crystalball
    Parameters
    ----------
    msname : str
        Name of the ms
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
        fluxcal_fields, fluxcal_scans = get_fluxcals(msname)
        if len(fluxcal_fields) == 0:
            print("No flux calibrator scan is present.\n")
            return 1
        print("##############################")
        print("Import fluxcal models")
        print("##############################")
        bandname = get_band_name(msname)
        cpu_count = psutil.cpu_count()
        ncpus = max(1, ncpus)
        ncpus = min(ncpus, cpu_count - 1)
        mem_frac = max(mem_frac, 0.8)
        for fluxcal in fluxcal_fields:
            modelname = datadir + "/" + fluxcal + "_" + bandname + "_model.txt"
            crys_cmd_args = [
                "-sm " + modelname,
                "-f " + fluxcal,
                "-ns 1000",
                "-j " + str(ncpus),
                "-mf " + str(mem_frac),
            ]
            crys_cmd = "crystalball " + " ".join(crys_cmd_args) + " " + msname
            print(crys_cmd)
            msg = os.system(crys_cmd)
            if msg == 0:
                print("Fluxcal model is imported successfully.")
                return 0
            else:
                print("Error in importing fluxcal model.")
                return 1
    except Exception as e:
        print("Error in importing model.")
        traceback.print_exc()
        return 1


def import_phasecal_models(msname, cpu_frac=0.8, mem_frac=0.8):
    """
    Import model visibilities for phasecal
    Parameters
    ----------
    msname : str
        Name of the ms
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
        phasecal_fields, phasecal_scans, phasecal_flux_list = get_phasecals(msname)
        target_scans, cal_scans, f_scans, g_scans, p_scans = get_cal_target_scans(
            msname
        )
        if os.path.exists(msname + "/SUBMSS"):
            mslist, scans = get_submsname_scans(msname)
            task = delayed(phasecal_setjy)(dry_run=True)
            mem_limit = run_limited_memory_task(task)
            dask_client, dask_cluster, n_jobs, n_threads = get_dask_client(
                len(mslist),
                cpu_frac,
                mem_frac,
                min_mem_per_job=mem_limit / 0.8,
            )
            tasks = []
            for i in range(len(mslist)):
                ms = mslist[i]
                scan = scans[i]
                if scan in g_scans:
                    tasks.append(
                        delayed(phasecal_setjy)(
                            ms,
                            field=",".join(phasecal_fields),
                            ismms=True,
                            n_threads=n_threads,
                        )
                    )
            results = compute(*tasks)
            dask_client.close()
            dask_cluster.close()
        else:
            if len(phasecal_fields) == 0:
                print("No phase calibrator fields.")
                return 1
            cpu_count = psutil.cpu_count()
            ncpus = max(1, ncpus)
            ncpus = min(ncpus, cpu_count - 1)
            phasecal_setjy(
                msname, field=",".join(phasecal_fields), ismms=False, n_threads=ncpus
            )
        print("Phasecal models are initiated.\n")
        return 0
    except Exception as e:
        print("Error in initiaing phasecal models.")
        traceback.print_exc()
        return 1


def import_polcal_model(msname, cpu_frac=0.8, mem_frac=0.8):
    """
    Import model for polarization calibrators (3C286 or 3C138)
    Parameters
    ----------
    msname : str
        Name of measurment set
    n_threads : int, optional
        Number of OpenMP threads
    Returns
    -------
    int
        Success message
    """
    try:
        target_scans, cal_scans, f_scans, g_scans, p_scans = get_cal_target_scans(
            msname
        )
        if len(p_scans) == 0:
            print("No polarization calibrator scans is present.")
            return 1
        if os.path.exists(msname + "/SUBMSS"):
            mslist, scans = get_submsname_scans(msname)
            task = delayed(polcal_setjy)(dry_run=True)
            mem_limit = run_limited_memory_task(task)
            dask_client, dask_cluster, n_jobs, n_threads = get_dask_client(
                len(p_scans), cpu_frac, mem_frac, min_mem_per_job=mem_limit / 0.8
            )
            tasks = []
            for i in range(len(mslist)):
                ms = mslist[i]
                scan = scans[i]
                if scan in p_scans:
                    tasks.append(
                        delayed(polcal_setjy)(ms, scan, ismms=True, n_threads=n_threads)
                    )
            results = compute(*tasks)
            dask_client.close()
            dask_cluster.close()
            return 0
        else:
            if len(p_scans) == 0:
                print("No polarization calibrator scans.")
                return 1
            cpu_count = psutil.cpu_count()
            ncpus = max(1, ncpus)
            ncpus = min(ncpus, cpu_count - 1)
            for scan in p_scans:
                polcal_setjy(msname, scan, ismms=False, n_threads=ncpus)
    except Exception as e:
        traceback.print_exc()
        return 1


def import_all_models(msname, cpu_frac=0.8, mem_frac=0.8):
    """
    Import all calibrator models
    Parameters
    ----------
    msname : str
        Measurement set name
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
        mspath = os.path.dirname(os.path.abspath(msname))
        os.chdir(mspath)
        fluxcal_result = import_fluxcal_models(msname, ncpus=ncpu, mem_frac=mem_frac)
        if fluxcal_result != 0:
            print("##################")
            print("Total time taken : " + str(time.time() - start_time) + "s")
            print("##################\n")
            return fluxcal_result, 1, 1
        phasecal_result = import_phasecal_models(
            msname, cpu_frac=cpu_frac, mem_frac=mem_frac
        )
        polcal_result = import_polcal_model(
            msname, cpu_frac=cpu_frac, mem_frac=mem_frac
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
        if options.workdir == "" or os.path.exists(options.workdir) == False:
            workdir = os.path.dirname(os.path.abspath(options.msname)) + "/workdir"
            if os.path.exists(workdir) == False:
                os.makedirs(workdir)
        else:
            workdir = options.workdir
        try:
            fluxcal_result, phasecal_result, polcal_result = import_all_models(
                options.msname,
                cpu_frac=float(options.cpu_frac),
                mem_frac=float(options.mem_frac),
            )
            os.system("touch " + workdir + "/.fluxcal_" + str(fluxcal_result))
            os.system("touch " + workdir + "/.phasecal_" + str(phasecal_result))
            os.system("touch " + workdir + "/.polcal_" + str(polcal_result))
            return 0
        except Exception as e:
            traceback.print_exc()
            os.system("touch " + workdir + "/.fluxcal_1")
            os.system("touch " + workdir + "/.phasecal_1")
            os.system("touch " + workdir + "/.polcal_1")
            return 1
    else:
        print("Please provide correct measurement set.\n")
        return 1


if __name__ == "__main__":
    result = main()
    print(
        "\n###################\nVisibility simulation is finished.\n###################\n"
    )
    os._exit(result)
