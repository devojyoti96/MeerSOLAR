Stop MeerSOLAR Job
==================

Since MeerSOLAR runs several seperate parallel processes in background, if user wants to stop the pipeline it is not straightforward. Use the following MeerSOLAR command line tool and ``<jobid>`` to stop a particular MeerSOLAR pipeline job. It will stop all child processes of that particular MeerSOLAR job.

.. code-block :: bash

    kill_meersolar_job --jobid <jobid>
    
    


