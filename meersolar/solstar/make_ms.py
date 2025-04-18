from casatools import simulator, quanta, image
import numpy as np
import os

# === CONFIG ===
cfg_file = '/Data/my_array.cfg'
ms_output = 'blank.ms'
blank_image = 'blank.image'

# === Function to parse CASA .cfg files ===
def read_cfg(filepath):
    names, x, y, z, diameters = [], [], [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            names.append(parts[0])
            x.append(float(parts[1]))
            y.append(float(parts[2]))
            z.append(float(parts[3]))
            diameters.append(float(parts[4]))
    return names, np.array(x), np.array(y), np.array(z), np.array(diameters)

# === Step 1: Create a blank image ===
print("🖼️  Creating blank image...")
nx, ny = 256, 256

if os.path.exists(blank_image):
    os.system(f'rm -rf {blank_image}')

ia = image()
ia.fromshape(blank_image, [nx, ny, 1, 1])  # [x, y, chan, pol]
cs = ia.coordsys()

# Set coordinate system properly
cs.setunits(['rad', 'rad', 'Hz', ''])  # 4 world axes
cs.setincrement([-1e-4, 1e-4], 'direction')     # ~20 arcsec/pixel
cs.setreferencevalue([0.0, 0.0], 'direction')   # RA=0, Dec=0
cs.setreferencepixel([nx / 2, ny / 2], 'direction')

cs.setincrement([1e6], 'spectral')             # 1 MHz channel width
cs.setreferencevalue([1.4e9], 'spectral')      # Center freq
cs.setreferencepixel([0.0], 'spectral')

cs.setstokes('I')

ia.setcoordsys(csys=cs.torecord())
ia.setbrightnessunit('Jy/pixel')
ia.putchunk(np.zeros((nx, ny, 1, 1)))  # Zero-flux
ia.close()
print("✅ Blank image created: blank.image")

# === Step 2: Read array config ===
print("📡 Reading array config...")
ant_names, ant_x, ant_y, ant_z, ant_diam = read_cfg(cfg_file)

# === Step 3: Create blank MS ===
if os.path.exists(ms_output):
    os.system(f'rm -rf {ms_output}')

print("🛰️  Setting up simulator...")
sm = simulator()
qa = quanta()
sm.open(ms_output)

# Reference location — update for your site if needed
ref_location = [
    qa.quantity(-30.7215, 'deg'),  # Latitude
    qa.quantity(21.4110, 'deg'),   # Longitude
    qa.quantity(1000.0, 'm')       # Altitude
]

sm.setconfig(
    telescopename='SIM',
    x=ant_x.tolist(),
    y=ant_y.tolist(),
    z=ant_z.tolist(),
    dishdiameter=ant_diam.tolist(),
    mount=['alt-az'] * len(ant_names),
    antname=ant_names,
    padname=[''] * len(ant_names),
    coordsystem='global',
    referencelocation=ref_location
)

sm.setspwindow(
    name='spw0',
    freq='1.4GHz',
    deltafreq='1MHz',
    freqresolution='1MHz',
    nchannels=1,
    stokes='XX YY'
)

sm.setfeed(mode='perfect X Y')
sm.setfield(sourcename='target', sourcedirection='J2000 12h00m00.0s -30d00m00.0s')

sm.settimes(
    integrationtime='10s',
    usehourangle=False,
    referencetime='2023/01/01/00:00:00'
)

# === Step 4: Set blank sky model ===
sm.setsky(mode='image', skymodel=blank_image)

# === Step 5: Simulate observation ===
print("📡 Simulating observation with blank image...")
sm.observe(sourcename='target', spwname='spw0', starttime='0s', stoptime='60s')

sm.done()
print(f"✅ Simulation complete. MS written to: {ms_output}")

