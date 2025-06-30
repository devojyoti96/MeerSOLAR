MeerSOLAR Documentation
=======================

**MeerSOLAR** is an automated calibration and imaging pipeline designed for solar radio observations using **MeerKAT** radio telescope. It performs end-to-end calibration, flagging, and imaging with a focus on dynamic solar data, supporting both spectral and temporal flexibility in imaging products.

Introduction
------------

Solar radio data presents unique challenges due to the high variability and brightness of the Sun, as well as the need for high time-frequency resolution. The MeerSOLAR pipeline addresses these challenges by:

- Automating the calibration of interferometric data, including flux, phase, and polarization calibrations
- Supporting time-sliced and frequency-sliced imaging workflows
- Leveraging Dask for scalable parallel processing
- Providing hooks for integration with contextual data from other wavelegths for enhanced solar analysis

Features of MeerSOLAR
----------------------

MeerSOLAR serves as a reference pipeline for science-ready processing of **MeerKAT solar observations**. It is designed to:

- Calibrate and image non-trivial solar observations
- Enable reproducible science through consistent and documented data reduction steps
- Designed to use on personal computer, single-node workstations, as well as in high-performance computing multi-node cluster
- It is tested on IDIA-supercomputing facility (where most of the MeerKAT observations are analysed)
- A free-tier cloud-based remote logger to monitor pipeline over the internet

Contents
---------
.. toctree::
   :maxdepth: 2
   
   quickstart
   install
   idia_setup
   initial_setup
   tutorial
   output
   kill
   cli
   ack
   lic
   meersolar

Indices
=======

* :ref:`genindex`

