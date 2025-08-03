Monitoring MeerSOLAR Pipeline Jobs
==================================
MeerSOLAR pipeline jobs are run in background and in parallel as ``prefect`` tasks managed by the ``prefect`` workflow. If ``prefect`` is running in ephmeral mode (default), no ``prefect`` dashboard will be available. In that case, a MeerSOLAR custom logger will show the task logs.


1. If ``prefect`` server is running, progress of the pipeline can be monitored from ``prefect`` dashboard.

   .. admonition:: Advantage
      :class: tip
   
   ``prefect`` server mode keeps backup of all pipeline runs logs in SQL format and provide easy access through a dashboard.  To view local default ``prefect`` dashboard:

   .. code-block :: bash
        
      run-meer-meerlogger
    
2. If MeerSOLAR is not running with ``prefect`` server mode, use local logger GUI.

.. toctree::
   :maxdepth: 2
   
   locallogger

.. admonition:: Recommendation
   :class: tip

   We recommend using remote logger.
   
.. warning ::    
   If you are working on a headless machine, such as work station, remote server or high-performace computing node, we recommend not to use local GUI logger, as it may encounter issue in deploying the PyQT based GUI window.
   
.. toctree::
   :maxdepth: 2
   
   remotelogger

