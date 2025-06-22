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

All data intermediate and final data products will be saved in <workdir>.

Work directory structure
-------------------------

.. graphviz::

   digraph G {
       rankdir=TB;
       "workdir" -> "logs"
       "workdir" -> "caltables"
       "workdir" -> "images"
       "logs" -> "job_001.log"
       "logs" -> "job_002.log"
       "caltables" -> "bandpass.cal"
       "caltables" -> "gain.cal"
       "images" -> "20250621"
       "images" -> "20250622"
       "20250621" -> "image_I.fits"
       "20250621" -> "image_V.fits"
       "20250622" -> "image_I.fits"
   }


