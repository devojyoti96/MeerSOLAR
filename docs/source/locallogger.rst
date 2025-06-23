Local GUI Logger
================

1. To start and view local GUI logger, run:

    .. code-block :: bash
        
        run_meerlogger --jobid <jobid>
        
    If user forget MeerSOLAR <jobid>, but knows the log directory, run:

    .. code-block :: bash
        
        run_meerlogger --logdir <logdir>
        
    .. note ::

        Log directory is generally at `<workdir>/logs`.

This will open the local GUI logger as shown below:

.. image ::  _static/ss_logger.png
    
2. Different logs are shown in the left panels. Once new jobs are started, this list will automatically updated. 

3. To view any of these logs, click on the name. It will display the live-log in the right panel and continuously updating it. To view another log, simply click on the other log name and the right panel will be updated with that log. A demonstration is shown below.

.. image :: _static/ss_logger1.png 
