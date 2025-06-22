Monitoring MeerSOLAR Pipeline Jobs
==================================
MeerSOLAR pipeline jobs are run in background and in parallel. Hence, it will not pring directly on terminal the outputs. Instead user needs to use MeerSOLAR logger to monitor the logs of the MeerSOLAR jobs.

MeerSOLAR provide a local logger GUI as well as a remote logger. 

.. admonition:: Recommendation
   :class: tip

   We recommend using remote logger.
   
.. warning ::    
   If you are working on a headless machine, such as work station, remote server or high-performace computing node, we recommend not to use local GUI logger, as it may encounter issue in deploying the PyQT based GUI window.
   
.. toctree::
   :maxdepth: 2
   
   locallogger
   remotelogger

