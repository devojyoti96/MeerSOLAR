Advanced CLI
=============

Calibration related CLI
-----------------------

1. To perform solar attenuation calibration using noise diode, use ``run-meer-fluxcal`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run-meer-fluxcal -h 
   
2. Parition calibrator scans from main measurement set, use ``run-meer-partition`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run-meer-partition -h  
   
3. Flagging of calibrators, use ``run-meer-flag`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run-meer-flag -h  
   
4. Simulate visibilities for calibrators, use ``run-meer-import-model`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run-meer-import-model -h

5. Perform basic calibration, use ``run-meer-basic-cal`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run-meer-basic-cal -h
   
6. Apply basic calibration solutions, use ``run-meer-apply-basiccal`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run-meer-apply-basiccal -h
   
7. Split measurement set for self-calibration or final imaging, use ``run-meer-split`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run-meer-split -h
   
8. Perform self-calibration, use ``run-meer-selfcal`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run-meer-selfcal -h
   
9. Apply self-calibration solutions, use ``run-meer-apply-selfcal`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run-meer-apply-selfcal -h
   
Solar specific CLI
------------------

1. To correct sidereal motion of the Sun, if the Sun is not tracked by the correlator delay center, use ``run-meer-solar-siderealcor`` . This is useful for observations where the Sun is in sidelobe of the telescope primary beam.

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run-meer-solar-siderealcor -h
   
2. Make dynamic spectra of solar scans, use ``run-meer-makeds`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run-meer-makeds -h
   
Imaging related CLI
-------------------
   
1. Perform spectro-polarimetric snapshot imaging, use ``run-meer-imaging`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run-meer-imaging -h
   
2. Perform primary beam correction of MeerKAT primary beam, for a single image, use ``run-meer-singlepbcor`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run-meer-singlepbcor -h
   
2. Perform primary beam corrections of MeerKAT primary beam for all images in a directory, use ``run-meer-meerpbcor`` .

.. admonition:: Click here to see parameters
   :class: dropdown

   .. program-output:: run-meer-meerpbcor -h
