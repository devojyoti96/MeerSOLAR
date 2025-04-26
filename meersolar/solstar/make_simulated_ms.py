import h5py
import numpy as np
from astropy.io import fits
from casatools import image, simulator, measures, quanta, table
import os
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


def makeimage_spectral(
    ra, dec, freq, freqres, cell, flux, imagename="sim_onepoint_true.im"
):
    """
    ra dec is center of image. Give in radians
    freq in GHz
    cell : e.g. '2arcsec'
    """
    shape = np.shape(flux)
    qa = quanta()
    ia = image()
    ## Make the image from a shape
    ia.close()
    ia.fromshape(imagename, shape, overwrite=True)
    ## Make a coordinate system
    cs = ia.coordsys()
    cs.setunits(["rad", "rad", "", "Hz"])
    cell_rad = qa.convert(qa.quantity(str(cell) + "arcsec"), "rad")["value"]
    cs.setincrement([-cell_rad, cell_rad], "direction")
    cs.setreferencevalue([ra, dec], type="direction")
    cs.setreferencevalue(str(freq) + "GHz", "spectral")
    cs.setreferencepixel([0], "spectral")
    cs.setincrement(str(freqres) + "MHz", "spectral")
    ## Set the coordinate system in the image
    ia.setcoordsys(cs.torecord())
    ia.setbrightnessunit("Jy/pixel")
    ia.set(0.0)
    ia.close()
    ia.open(imagename)
    data = ia.getchunk()
    data = flux.T
    ia.putchunk(data)
    ia.close()


def makeimage(ra, dec, freq, cell, flux, imagename="sim_onepoint_true.im"):
    """
    ra dec is center of image. Give in radians
    freq in GHz
    cell : e.g. '2arcsec'
    """
    flux1 = flux
    shape = np.shape(flux1)
    qa = quanta()
    ia = image()
    ## Make the image from a shape
    ia.close()
    ia.fromshape(imagename, [shape[0], shape[1], 1, 1], overwrite=True)
    ## Make a coordinate system
    cs = ia.coordsys()
    cs.setunits(["rad", "rad", "", "Hz"])
    cell_rad = qa.convert(qa.quantity(str(cell) + "arcsec"), "rad")["value"]
    cs.setincrement([-cell_rad, cell_rad], "direction")
    cs.setreferencevalue([ra, dec], type="direction")
    cs.setreferencevalue(str(freq) + "GHz", "spectral")
    cs.setreferencepixel([0], "spectral")
    cs.setincrement("0.001GHz", "spectral")
    ## Set the coordinate system in the image
    ia.setcoordsys(cs.torecord())
    ia.setbrightnessunit("Jy/pixel")
    ia.set(0.0)
    ia.close()
    ia.open(imagename)
    data = ia.getchunk()
    data[:, :, 0, 0] = flux1.T
    ia.putchunk(data)
    ia.close()


def generate_ms(
    config_file,
    solar_model,
    source_ra,
    source_dec,
    reftime,
    integration_time=60,
    msname="fasr.ms",
    duration=None,
):
    """
    config_file: Antenna configuration file in standard format.
                 First column: x
                 Second COlumn:y
                 Third Column: z
                 Fourth column: dish diameter
                 Fifth column: Antenna name
    spws:  Frequencies of the spws
    source_ra, source_dec: ra, dec of phasecenter in radians
    reftime: Reference time of observation in CASA format. in UTC
    integration_time: in seconds
    duration: in seconds
    """
    sm = simulator()
    me = measures()
    qa = quanta()
    sm.open(msname)
    antenna_params = np.genfromtxt(config_file, usecols=(0, 1, 2))
    antenna_params = np.loadtxt(config_file, dtype="str", unpack=True)
    x = antenna_params[0, :].astype("float")
    y = antenna_params[1, :].astype("float")
    z = antenna_params[2, :].astype("float")
    dish_dia = antenna_params[3, :].astype("float")
    ant_names = antenna_params[4, :]
    hf = h5py.File(solar_model)
    spws = np.array(hf["frequency"])
    flux = np.array(hf["flux_jy"])
    cell = hf.attrs["cdelt1"]
    hf.close()
    ref_location = [
        qa.quantity(-30.7215, "deg"),  # Latitude
        qa.quantity(21.4110, "deg"),  # Longitude
        qa.quantity(1000.0, "m"),  # Altitude
    ]
    sm.setconfig(
        telescopename="MeerKAT",
        x=x,
        y=y,
        z=z,
        dishdiameter=dish_dia,
        mount=["alt-az"],
        coordsystem="global",
        antname=ant_names,
        referencelocation=ref_location,
    )

    num_spw = np.size(spws)
    for i in range(0, num_spw):
        sm.setspwindow(
            spwname="Band" + str(i),
            freq=str(spws[i]) + "GHz",
            deltafreq="1MHz",
            freqresolution="1MHz",
            nchannels=1,
            stokes="RR LL",
        )

    sm.setfeed("perfect R L")
    sm.setfield(
        sourcename="Sun",
        sourcedirection=["J2000", str(source_ra) + "rad", str(source_dec) + "rad"],
    )
    sm.setauto(autocorrwt=0.0)
    sm.settimes(
        integrationtime=str(integration_time) + "s",
        referencetime=me.epoch("UTC", reftime),
        usehourangle=False,
    )

    if duration == None:
        duration = integration_time

    starttime = str(-duration / 2) + "s"
    endtime = str(duration / 2) + "s"
    for i in range(0, num_spw):
        sm.observe("Sun", "Band" + str(i), starttime=starttime, stoptime=endtime)

    # for i in range(0,num_spw):
    # sm.setdata(spwid=i)
    # makeimage(source_ra,source_dec,spws[i],cell,flux[:,:,i],imagename='solar_image.im')
    # sm.predict(imagename='solar_image.im')

    sm.close()
    tb = table()
    tb.open(msname + "/ANTENNA", nomodify=False)
    ants = tb.getcol("NAME")
    ants = ants.astype("<U13")
    for i in range(len(ant_names)):
        ants[i] = ant_names[i]
    print(ants)
    tb.putcol("NAME", ants)
    tb.flush()
    tb.close()


### taking solar coords from meerkat data of same day.
solar_ra = np.deg2rad(345.256)
solar_dec = np.deg2rad(-6.29)

solar_model = "/media/devojyoti/Data1/TotalTB.h5"
config_file = "meerkat.config"


reftime = "2025/03/04/09:30:00"
duration = 60
msname = "meerkat_20250304_0930_uhfband.ms"
if os.path.exists(msname):
    os.system("rm -rf " + msname)
generate_ms(config_file, solar_model, solar_ra, solar_dec, reftime, msname=msname)
