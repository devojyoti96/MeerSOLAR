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
    Final product directory: <outputdir>
    ###########################

Directory structure
-------------------

All data intermediate data products will be saved in ``<workdir>``.

All final data products will be saved in ``<outputdir>``.

.. note :: 

   In local workstations, it is okay to choose ``<workdir>`` and ``<outputdir>``. In HPC environment, generally, high-speed disks are used during data-processing, which may have limited storage life-time, and has seperate long-term storage disks. It is recommended to choose ``<workdir>`` path inside the high-speed disk and ``<outputdir>`` inside the long-term storage disk. Otherwise, there may be possiblity that final data-products will be removed after certain time. 

.. admonition:: Click here to see directory structure in work directory
   :class: dropdown
   
   .. mermaid::

       graph LR
           WD["Work directory:<br>{workdir}"] --> CAL["Calibrator ms:<br>calibrator.ms"]
           WD --> SCMS["`Self-cal ms(s):<br>selfcals_scan_*_spw_*.ms`"]
           WD --> SCDIR["`Self-cal directories:<br>selfcals_scan_*_spw_*_selfcal`"]
           WD --> TMS["`Target ms(s):<br>targets_scan_*_spw_*.ms`"]
           WD --> BACK["Backup directory:<br>backup"]
           WD --> LOG["Log directory:<br>logs"]
           LOG --> LOGF["*.log"]
                      
.. admonition:: Click here to see directory structure in output directory
   :class: dropdown
   
   .. mermaid::

       graph LR
           WD["Output directory:<br>{outdir}"] --> CALTABLE["Caltable directory:<br>caltables"]
           WD --> DS["`Dynamic spectra:<br>dynamic_spectra`"]
           DS --> DSNPY["`Dynamic spectrum numpy files:<br>*.npy`"]
           DS --> DSPNG["`Dynamic spectrum plots in PNG:<br>*.png`"]
           WD --> IMG["`Image directory:<br>imagedir_f_*_t_*_w_briggs_*`"]
           CALTABLE --> ATT["`Attenuator values:<br>*_attval_scan_*.npy`"]
           CALTABLE --> CTBL["`Caltables:<br>calibrator_caltable.bcal/gcal/kcal`"]
           CALTABLE --> BPTBL["`Bandpass tables scaled:<br>calibrator_caltable_scan_*.bcal`"]
           CALTABLE --> SCTBL["`Self-cal tables:<br>selfcal_scan_*.gcal`"]
           IMG --> IMAGE["Fits image:<br>images"]
           IMG --> MODEL["Fits models:<br>models"]
           IMG --> RES["Fits residual:<br>residuals"]
           IMG --> PBCOR["`Primary beam<br>corrected images:<br>pbcor_images`"]
           IMG --> TBIMG["`Brightness temperature images:<br>tb_images`"]
           IMG --> OVRPDF["Overlays of EUV:<br>PDF format:<br>overlays_pdfs"]
           IMG --> OVRPNG["Overlays of EUV:<br>PNG format:<br>overlays_pngs"]
           IMAGE --> IMAGEHPC["Radio images in HPC coordinate:<br>FITS format:<br>hpcs"]
           PBCOR --> PBCORHPC["Radio images in HPC coordinate:<br>FITS format:<br>hpcs"]
           TBIMG --> TBIMGHPC["Radio images in HPC coordinate:<br>FITS format:<br>hpcs"]
           IMAGE --> IMAGEPDF["Quicklook in HPC coordinate:<br>PDF format:<br>pdfs"]
           IMAGE --> IMAGEPNG["Quicklook in HPC coordinate:<br>PNG format:<br>pngs"]
           PBCOR --> PBCORPDF["Quicklook in HPC coordinate:<br>PDF format:<br>pdfs"]
           PBCOR --> PBCORPNG["Quicklook in HPC coordinate:<br>PNG format:<br>pngs"]
           TBIMG --> TBIMGPDF["Quicklook in HPC coordinate:<br>PDF format:<br>pdfs"]
           TBIMG --> TBIMGPNG["Quicklook in HPC coordinate:<br>PNG format:<br>pngs"]

Data products
-------------
Pipeline produces calibrated visibilities as well as several imaging products.

Dynamic spectrum
~~~~~~~~~~~~~~~~
Dynamic spectra for all (or the ones selected) target scans are available in ``dynamic_spectra`` directory inside the output directory ``<outputdir>``.

Calibrated visibilities
~~~~~~~~~~~~~~~~~~~~~~~
Calibrated measurements sets for all (or the ones selected) target scans will be available in work directory ``<workdir>`` with naming format, ``targets_scan_<scan_number>_spw_<channel_range>.ms``. Calibrated measurement sets will not be saved in output directory ``<outputdir>`` (unless same as ``<workdir>``) to save space.

Imaging products 
~~~~~~~~~~~~~~~~
Imaging products are available in ``imagedir_f_<freqres>_t_<timeres>_w_<weight>_<robust>`` directory inside output directory ``<outputdir>``. If imaging is performed with different time and frequency resolutions or different weighting schemes, seperate image directories with corresponding parameters will have the corresponding images. 

1. **Image fits in RA/DEC** - Fits images in RA/DEC coordinate are available in ``imagedir_f_<freqres>_t_<timeres>_w_<weight>_<robust>/images`` directory inside work directory. These are not primary beam corrected.

    .. note ::
        
       All fits images have some MeerSOLAR specific metadata in the header and some image statistics.
       
       .. admonition:: Click here to see details of these metadata
           :class: dropdown
                               
            PIPELINE= 'MeerSOLAR' # Pipeline name
            
            AUTHOR  = 'DevojyotiKansabanik,DeepanPatra' # Pipeline developer  
                                              
            BAND    = 'U' # MeerKAT band name, required for proper primary beam correction     
                                                                   
            MAX     =  ``<maxval>`` # Maximum value on the solar disc       
                                                       
            MIN     =  ``<minval>`` # Minimum value on the solar disc      
                                                       
            RMS     =  ``<rms>`` # RMS value outside solar disc       
                                                    
            SUM     =  ``<sum>`` # Total sum on the solar disc  
                                                    
            MEAN    =  ``<mean>`` # Mean value on the solar disc   
                                                          
            MEDIAN  =  ``<median>`` # Median value on the solar disc  
                                                         
            RMSDYN  =  ``<rmsdyn>`` # RMS based dynamic range, ``<maxval/rms>``
                                                            
            MIMADYN =  ``<minmaxdyn>`` # Min-max based dynamic range, ``<maxval/abs(minval)>``  
 
2. **Primary beam corrected image fits** - Primary beam corrected fits images are available in ``imagedir_f_<freqres>_t_<timeres>_w_<weight>_<robust>/pbcor_images`` directory inside work directory.

3. **Brightness temperature image fits** - Brightness temperature fits images are available in ``imagedir_f_<freqres>_t_<timeres>_w_<weight>_<robust>/tb_images`` directory inside work directory. 

4. **CLEAN model and residual fits** - CLEAN model and residual images are available in ``imagedir_f_<freqres>_t_<timeres>_w_<weight>_<robust>/models`` and ``imagedir_f_<freqres>_t_<timeres>_w_<weight>_<robust>/residuals`` directory inside work directory. These are not saved only if ``keep_backup`` option is switched on.

5. **Radio images in helioprojective coordinates** - Directory names ``hpcs`` inside directories like, ``images, pbcor_images, tb_images`` inside the image directory will have the FITS images in helioprojective coordinates. Images in PNG and PDF formats are also available in ``pngs`` and ``pdfs`` directories inside the parent directories.

    .. note ::
      
       Header of helioprojective maps have wavelength information in unit of ``centimeter`` or ``meter``.  

6. **Overlays on GOES-SUVI EUV images** - Overlays on GOES SUVI EUV (193 Å) images are saved in PNG and PDF formats in ``imagedir_f_<freqres>_t_<timeres>_w_<weight>_<robust>/overlays_pngs`` and ``imagedir_f_<freqres>_t_<timeres>_w_<weight>_<robust>/overlays_pngs``, respectively.









