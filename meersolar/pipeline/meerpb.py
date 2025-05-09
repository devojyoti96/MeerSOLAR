import numpy as np, copy, glob, psutil, datetime, astropy.units as u
from astropy.io import fits
from skimage.transform import resize
from scipy.ndimage import rotate
from astropy.coordinates import AltAz, EarthLocation, FK5, SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator



# Define MeerKAT location
meerpos = EarthLocation(lat=-30.7133 * u.deg, lon=21.4429 * u.deg, height=1086.6 * u.m)

def get_IQUV(filename,stokesaxis=4):
	"""
	Get IQUV from a fits
	Parameters
	----------
	filename : str
	    Fits image name
	stokesaxis : int, optional
	    Stokes axis
    Returns
    -------
    dict
        Stokes parameters
	"""
	data = fits.getdata(filename)
	shape=data.shape
	stokes = {}
	if shape[0]>1 or shape[1]>1:
	    if stokesaxis==3:
		    stokes['I'] = data[0, 0, :, :]
		    stokes['Q'] = data[0, 1, :, :]
		    stokes['U'] = data[0, 2, :, :]
		    stokes['V'] = data[0, 3, :, :]
	    else:
		    stokes['I'] = data[0, 0, :, :]
		    stokes['Q'] = data[1, 0, :, :]
		    stokes['U'] = data[2, 0, :, :]
		    stokes['V'] = data[3, 0, :, :]
    else:
        stokes['I']=data[0,0,:,:]
        stokes['Q']=copy.deepcopy(stokes['I'])*0
        stokes['U']=copy.deepcopy(stokes['I'])*0
        stokes['V']=copy.deepcopy(stokes['I'])*0
	return stokes
	
def get_brightness(stokes):
    """
    Returns brightness matrix from stokes dictionary
    """
    I = stokes["I"]
    Q = stokes["Q"]
    U = stokes["U"]
    V = stokes["V"]
    XX = I + Q
    XY = U + 1.0j * V
    YX = U - 1.0j * U
    YY = I - Q
    return ((XX, XY), (YX, YY))


def make_stokes(b):
    """
    Makes stokes images from brightness matrix
    """
    XX = b[:, :, 0, 0]
    XY = b[:, :, 0, 1]
    YX = b[:, :, 1, 0]
    YY = b[:, :, 1, 1]
    I = np.real(XX + YY) / 2
    Q = np.real(YY - XX) / 2
    U = np.real(XY + YX) / 2
    V = np.real(YX - XY) / 2
    return I, Q, U, V

def create_jones_inv(beam):
    """
    Creates inverse of jones matrix for a given jones from beam file
    """
    a, b, c, d = beam
    det = a * d - b * c
    if det==0:
        raise np.linalg.LinAlgError("Jones matrix is singular and not invertible.")
    return np.array([[d, -b], [-c, a]]) / det
    
def altaz_to_parallactic_angle(alt,az,LAT):
	'''
	Function to calculate parallactic angle given Altitude, Azimuith of the source and Latitude on the observatory
	Parameters
	----------
	alt : float
		Altitude in degree
	az : float
		Azimuith in degree
	lat : float
		Latitude in degree
	Returns
	-------
	float
		Parallactic angle in radian
	''' 
	alt=np.deg2rad(alt)
	az=np.deg2rad(az)
	lat=np.deg2rad(LAT)
	p=-np.arctan2(np.sin(az)*np.cos(lat),np.cos(alt)*np.sin(lat) - np.sin(alt)*np.cos(lat)*np.cos(az))
	return p

def get_za_az_from_fits(fits_filename):
    """
    Extract zenith angle and azimuth arrays from a FITS file with WCS headers.
    Parameters
    ----------
    fits_filename : str
        Path to the FITS image file.
    Returns
    -------
    dict
        Dictionary containing:
        - 'za_rad': Zenith angle array (in radians)
        - 'astro_az_rad': Azimuth angle array (in radians)
    """
    try:
        with fits.open(fits_filename) as hdulist:
            header = hdulist[0].header
    except IOError as err:
        print(f"Unable to open {fits_filename} for reading:\n{err}")
        return None
    # Extract observation time
    obs_time_str = header.get('DATE-OBS', None)
    if obs_time_str is None:
        print("DATE-OBS header not found.")
        return None
    obs_time_str = obs_time_str.split('.')[0]  # remove fractional seconds
    obs_datetime = datetime.datetime.strptime(obs_time_str, '%Y-%m-%dT%H:%M:%S')
    obs_time = Time(obs_datetime)
    # Validate WCS axes
    if 'RA' not in header.get('CTYPE1', '') or 'DEC' not in header.get('CTYPE2', ''):
        print("WCS CTYPE headers are not RA/DEC.")
        return None
    # Setup WCS and pixel grid
    wcs = WCS(header)
    nx, ny = header['NAXIS1'], header['NAXIS2']
    x_pix = np.arange(nx)
    y_pix = np.arange(ny)
    X, Y = np.meshgrid(x_pix, y_pix)
    pixel_coords = np.stack([X, Y], axis=-1).reshape(-1, 2)
    # Convert pixels to sky coordinates
    world_coords = wcs.pixel_to_world(pixel_coords[:, 0], pixel_coords[:, 1])
    sky_coords = SkyCoord(ra=world_coords[0], dec=world_coords[1], frame='icrs')
    # Transform to AltAz
    altaz_frame = AltAz(obstime=obs_time, location=meerpos)
    sky_altaz = sky_coords.transform_to(altaz_frame)
    altitude_deg = sky_altaz.alt.degree
    azimuth_deg = sky_altaz.az.degree
    zenith_angle_rad = np.deg2rad(90.0 - altitude_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)
    # Reshape to image dimensions
    za_rad = zenith_angle_rad.reshape(ny, nx)  # shape: [Y, X]
    az_rad = azimuth_rad.reshape(ny, nx)
    return {'za_rad': za_rad, 'astro_az_rad': az_rad}
    
	

	
	
	
	
