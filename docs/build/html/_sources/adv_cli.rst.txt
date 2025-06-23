Advanced CLI
=============

Calibration related CLI
-----------------------

1. To perform solar attenuation calibration using noise diode, use ``run_fluxcal`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run_fluxcal -h 
   
2. Parition calibrator scans from main measurement set, use ``run_partition`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run_partition -h  
   
3. Flagging of calibrators, use ``run_flag`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run_flag -h  
   
4. Simulate visibilities for calibrators, use ``import_model`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: import_model -h

5. Perform basic calibration, use ``run_basic_cal`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run_basic_cal -h
   
6. Apply basic calibration solutions, use ``run_apply_basiccal`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run_apply_basiccal -h
   
7. Split measurement set for self-calibration or final imaging, use ``run_target_split`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run_target_split -h
   
8. Perform self-calibration, use ``run_selfcal`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run_selfcal -h
   
9. Apply self-calibration solutions, use ``run_apply_selfcal`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run_apply_selfcal -h
   
Solar specific CLI
------------------

1. To correct sidereal motion of the Sun, if the Sun is not tracked by the correlator delay center, use ``run_solar_siderealcor`` . This is useful for observations where the Sun is in sidelobe of the telescope primary beam.

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run_solar_siderealcor -h
   
2. Make dynamic spectra of solar scans, use ``run_makeds`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run_makeds -h
   
Imaging related CLI
-------------------
   
1. Perform spectro-polarimetric snapshot imaging, use ``run_imaging`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run_imaging -h
   
2. Perform primary beam correction of MeerKAT primary beam, for a single image, use ``run_single_meerpbcor`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run_single_meerpbcor -h
   
2. Perform primary beam corrections of MeerKAT primary beam for all images in a directory, use ``run_meerpbcor`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run_meerpbcor -h
