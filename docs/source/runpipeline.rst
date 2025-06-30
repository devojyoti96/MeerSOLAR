Run MeerSOLAR Pipeline
=======================

Basic run
---------
To run MeerSOLAR pipeline, with default settings for full analysis, run the following command from terminal. Work directory needs not to be created before hand, but the path where it will be created should exist.

.. code-block :: bash

    run_meersolar </path/to/measurement_set> --workdir </path/to/work_directory> --outdir </path/to/output_product_directory>
        
Advanced run
------------
For advanced run, user is requested to first check the parameters of **run_meersolar**.

.. code-block :: bash
 
    run_meersolar -h
    
.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run_meersolar -h
   
Multiple options demonstrated below can be combined to have all of them together.
   
1. To view details of measurement set:

.. code-block :: bash

    run_meersolar_showms </path/to/measurement_set>
    
Runs with advanced calibration paramaters 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Do calibration with custom calibration parameters. There are two parameters: **cal_uvrange** and **solint** which can be changed. Example, run the following command to perform gain solutions at 10second interval and >200lambda data:

.. code-block :: bash
    
    run_meersolar </path/to/measurement_set> --solint "10s" --uvrange ">200lambda" --workdir </path/to/work_directory> --outdir </path/to/output_product_directory> 
    
2. By default for full-polar data, polarization calibration will be performed. To disable it:

.. code-block :: bash 

    run_meersolar </path/to/measurement_set> --no_polcal --workdir </path/to/work_directory> --outdir </path/to/output_product_directory> 
    
    
Runs with advanced imaging paramaters 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Run pipeline to image specific target scans. Default is to use all. Users are requested to check target scan number using **run_meersolar_showms** first. Example, run the following command to image only target scans 3,8,13 from all available target scans:

.. code-block :: bash

    run_meersolar </path/to/measurement_set> --target_scans 3 8 13 --workdir </path/to/work_directory> --outdir </path/to/output_product_directory> 
    
2. Run pipeline to image specific time and frequency range. Default is to use entire time and frequency range. Example for imaging two time ranges given in UTC and frequency ranges given in MHz: 

.. code-block :: bash

    run_meersolar </path/to/measurement_set> --timerange 2024/06/10/09:00:00~2024/06/10/09:30:00,2024/06/10/10:15:00~2024/06/10/10:45:00 --freqrange 600~650,700~800 --workdir </path/to/work_directory> --outdir </path/to/output_product_directory> 
    
3. Run imaging with custom time and frequency resolution. Default is to use entire observing band and maximum 2 hours (or maximum scan duration) of integration to avoid smearing due to differential rotation of the Sun. Example run for imaging at 2 minutes (120 seconds) time resolution and 10 MHz frequency resolution:

.. code-block :: bash 
    
    run_meersolar </path/to/measurement_set> --image_timeres 120 --image_freqres 10 --workdir </path/to/work_directory> --outdir </path/to/output_product_directory> 
    
4. Default is to make only Stokes I images if `do_polcal=False` and Stokes IQUV, if `do_polcal=True`. To run only Stokes I imaging, even if `do_polcal=True`, run:

.. code-block :: bash
    
    run_meersolar </path/to/measurement_set> --pol I --workdir </path/to/work_directory> --outdir </path/to/output_product_directory> 
    
Similarly, all other advanced imaging parameters can be used.

Switching off particular pipeline step(s)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
By default, all steps will be performed by pipeline. Even pipeline was run upto certain stages and then stopped, all steps from beginning will be performed to avoid any potential issue in failure in previous runs. If user is certain that previous run was successful upto certain stages, those stages can be switched.

.. caution :: 
    
    User should not modify any file and directory structure in the work directory. Switching off certain parameters will only allow to run the pipeline forward, if the expected output products from those steps are present with appropriate name in appropriate directory. Otherwise, it will fail.

Take a look at the **Advanced pipeline parameters** in the help page of **run_meersolar**. Each parameters are self explanatory. Some examples are given below:

1. To switch on noise-diode based flux calibration:

.. code-block :: bash
    
    run_meersolar </path/to/measurement_set> --no_noise_cal --workdir </path/to/work_directory> --outdir </path/to/output_product_directory> 
    
2. To switch off self-calibration:

.. code-block :: bash
    
    run_meersolar </path/to/measurement_set> --no_selfcal --workdir </path/to/work_directory> --outdir </path/to/output_product_directory> 
    
3. To stop final imaging:

.. code-block :: bash
    
   run_meersolar </path/to/measurement_set> --no_imaging --workdir </path/to/work_directory> --outdir </path/to/output_product_directory> 
   
4. To switch off self-calibration and final imaging

.. code-block :: bash

    run_meersolar </path/to/measurement_set> --no_selfcal --no_imaging --workdir </path/to/work_directory> --outdir </path/to/output_product_directory> 









