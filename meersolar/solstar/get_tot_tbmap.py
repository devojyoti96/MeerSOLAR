import numpy as np
import h5py, sys, os, gc
import matplotlib.pyplot as plt
from scipy.io import readsav
import sunpy.map as smap
from optparse import OptionParser

jy2cgs = 1e-23
vc = 29979245800  # speed of light in cm/s
kb = 1.38065e-16
rad2asec = 180 / np.pi * 3600
asec2Mm = 0.73


def make_TB_maps(coronal_tbfile, chrom_tbfile, outfile):
    """
    Make total brightness temperature maps
    Parameters
    ----------
    coronal_tbfile : str
        Coronal brightness temperature file
    chrom_tbfile : str
        Chromospheric brightness temperature file
    outfile : str
        Output file name
    Returns
    -------
    str
        Output file name
    """
    print("########################")
    print("Estimating total brightness temperature for free-free emission ......")
    print("########################")
    if os.path.dirname(outfile) == "":
        outfile = os.getcwd() + "/" + outfile
    coronal_data = h5py.File(coronal_tbfile)
    freqs = np.array(coronal_data["freqs"])
    coronal_tb = np.array(coronal_data["Tb"])
    coronal_data.close()
    hf1 = h5py.File(chrom_tbfile)
    chrom_tb = np.array(hf1["tb"])
    total_tb = coronal_tb + np.expand_dims(chrom_tb, axis=2)
    pos=np.where(total_tb<1000)
    total_tb[pos]=0.0
    meta = hf1.attrs
    keys = hf1.attrs.keys()
    hf2 = h5py.File(outfile, "w")
    metadata = {}
    for key in keys:
        hf2.attrs[key] = meta[key]
        metadata[key] = meta[key]
    ##### writing the full data ####
    hf2.create_dataset("frequency", data=freqs)
    hf2.create_dataset("tb", data=total_tb)
    sr = (meta["cdelt1"] / rad2asec) * (meta["cdelt2"] / rad2asec)
    flux_tot = (
        2 * kb * total_tb * (freqs * 1e9) ** 2 * sr / (vc**2 * jy2cgs)
    )  ### converting Tb to Jy
    flux_tot[pos]=0.0
    hf2.create_dataset("flux_jy", data=flux_tot)
    hf2.close()
    hf1.close()
    gc.collect()
    return outfile


def main():
    usage = "Make total brightness temperature radio map"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--coronal_tbfile",
        dest="coronal_tbfile",
        default=None,
        help="Coronal TB file name (.h5)",
        metavar="String",
    )
    parser.add_option(
        "--chromo_tbfile",
        dest="chromo_tbfile",
        default=None,
        help="Chromospheric TB file name (.h5)",
        metavar="String",
    )
    parser.add_option(
        "--outfile",
        dest="outfile",
        default="TotalTB.h5",
        help="Output file name (.h5)",
        metavar="String",
    )
    (options, args) = parser.parse_args()
    if (
        options.coronal_tbfile == None
        or os.path.exists(options.coronal_tbfile) == False
    ):
        print("Please provide correct cornal TB file.\n")
        return 1
    elif (
        options.chromo_tbfile == None or os.path.exists(options.chromo_tbfile) == False
    ):
        print("Please provide correct chromospheric TB file path.\n")
        return 1
    else:
        total_tb_file = make_TB_maps(
            options.coronal_tbfile, options.chromo_tbfile, options.outfile
        )
    if total_tb_file != None:
        print("Output file name: ", total_tb_file)
        gc.collect()
        return 0
    else:
        gc.collect()
        return 1
