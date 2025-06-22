Directory Structure and Data Products
=====================================
Once user started a MeerSOLAR pipeline job, MeerSOLAR assign a unique JobID based on current time in millisecond precision in YYYYMMDDHHMMSSmmm format.
Note down the Job ID to view the logger.

The following output will appear in terminal:

.. code-block :: console

    ########################################
    Starting MeerSOLAR Pipeline....
    #########################################

    ###########################
    MeerSOLAR Job ID: <YYYYMMDDHHMMSSmmm>
    Work directory: <workdir>
    ###########################

Directory structure
-------------------

All data intermediate and final data products will be saved in <workdir>.

.. admonition:: Click here to see directory structure in work directory
   :class: dropdown
   
   .. mermaid::

       graph LR
           WD["Work directory:<br>{workdir}"] --> CAL["Calibrator ms:<br>calibrator.ms"]
           WD --> SCMS["`Self-cal ms(s):<br>selfcals_scan_*_spw_*.ms`"]
           WD --> SCDIR["`Self-cal directories:<br>selfcals_scan_*_spw_*_selfcal`"]
           WD --> TMS["`Target ms(s):<br>targets_scan_*_spw_*.ms`"]
           WD --> BACK["Backup directory:<br>backup"]
           WD --> DS["`Dynamic spectra:<br>dynamic_spectra`"]
           WD --> LOG["Log directory:<br>logs"]
           WD --> CALTABLE["Caltable directory:<br>caltables"]
           WD --> IMG["`Image directory:<br>imagedir_f_*_t_*_w_briggs_*`"]

           LOG --> LOGF["*.log"]
           CALTABLE --> ATT["`Attenuator values:<br>*_attval_scan_*.npy`"]
           CALTABLE --> CTBL["`Caltables:<br>calibrator_caltable.bcal/gcal/kcal`"]
           CALTABLE --> BPTBL["`Bandpass tables scaled:<br>calibrator_caltable_scan_*.bcal`"]
           CALTABLE --> SCTBL["`Self-cal tables:<br>selfcal_scan_*.gcal`"]
           
           IMG --> IMAGE["Fits image:<br>images"]
           IMG --> MODEL["Fits models:<br>models"]
           IMG --> RES["Firs residual:<br>residuals"]
           IMG --> PBDIR["Primary beams:<br>pbdir"]
           IMG --> PBCOR["`Primary beam<br>corrected images:<br>pbcor_images`"]
           IMG --> TBIMG["`Brightness temperature images:<br>tb_images`"]
           IMG --> OVRPDF["Overlays of EUV:<br>PDF format:<br>overlays_pdfs"]
           IMG --> OVRPNG["Overlays of EUV:<br>PNG format:<br>overlays_pngs"]
           IMG --> RADPNG["Radio images in HPC coordinate:<br>PNG format:<br>images_pngs"]
           IMG --> RADPDF["Radio images in HPC coordinate:<br>PDF format:<br>images_pdfs"]

Data products
-------------
Pipeline produces calibrated visibilities as well as several imaging products.

Dynamic spectrum
~~~~~~~~~~~~~~~~
Dynamic spectra for all (or the ones selected) target scans are available in ``dynamic_spectra`` directory inside the work directory.

Calibrated visibilities
~~~~~~~~~~~~~~~~~~~~~~~
Calibrated measurements sets for all (or the ones selected) target scans will be available in workdir with naming format, ``targets_scan_<scan_number>_spw_<channel_range>.ms``.

Imaging products 
~~~~~~~~~~~~~~~~
Imaging products are available in ``imagedir_f_<freqres>_t_<timeres>_w_<weight>_<robust>`` directory inside work directory. If imaging is pereformed with different time and frequency resolutions or different weighting schemes, seperate image directories with corresponding parameters will have the corresponding images. 

1. **Image fits in RA/DEC** - Fits images in RA/DEC coordinate are available in ``imagedir_f_<freqres>_t_<timeres>_w_<weight>_<robust>/images`` directory inside work directory. These are not primary beam corrected.
 
2. **Primary beam corrected image fits** - Primary beam corrected fits images are available in ``imagedir_f_<freqres>_t_<timeres>_w_<weight>_<robust>/pbcor_images`` directory inside work directory.

3. **Brightness temperature image fits** - Brightness temperature fits images are available in ``imagedir_f_<freqres>_t_<timeres>_w_<weight>_<robust>/tb_images`` directory inside work directory. 

4. **CLEAN model and residual fits** - CLEAN model and residual images are available in ``imagedir_f_<freqres>_t_<timeres>_w_<weight>_<robust>/models`` and ``imagedir_f_<freqres>_t_<timeres>_w_<weight>_<robust>/residuals`` directory inside work directory. These are not saved only if ``keep_backup`` option is switched on.

5. **Radio images in helioprojective coordinates** - Directories like, ``images, pbcor_images, tb_images`` inside the image directory will have the helioprojective maps of corresponding images in PNG and PDF formats in ``pngs`` and ``pdfs`` directories inside the parent ones.

6. **Overlays on GOES-SUVI EUV images** - Overlays on GOES SUVI EUV images are saved in PNG and PDF formats in ``imagedir_f_<freqres>_t_<timeres>_w_<weight>_<robust>/overlays_pngs`` and ``imagedir_f_<freqres>_t_<timeres>_w_<weight>_<robust>/overlays_pngs``, respectively.









