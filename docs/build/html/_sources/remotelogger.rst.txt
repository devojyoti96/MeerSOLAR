Remote MeerSOLAR Logger
=======================

If you have setup remote logger with remote logger link, ``https://<remote-logger-name>.onrender.com``, open this in your browser from any machine. It does not need to be opened from the machine jobs are running. Remote logger looks like the following.

.. image :: _static/rl1.png

Getting access keys
--------------------
To access remote logger, user needs two information; Remote logger job name: ``<jobname>`` and remote logger password: ``<remote_access_password>``.

1. On start-up of the MeerSOLAR job and if remote logger is not disable by `--no_remote_logger` option, you will see:

.. code-block :: console
    
   ############################################################################
   https://<remote-logger-name>.onrender.com/
   Job ID: <jobname>
   Remote access password: <remote_access_password>
   #############################################################################
 
Remote ``<jobname>`` is of the format: ``<hostname> :: <YYYY-MM-DDThh:mm:ss> :: <ms_prefix>`` for easy identification.
     
3. If you setup you e-mail id, you should receive an e-mail on startup of MeerSOLAR job from ``MeerSOLAR Notification``. The subject of the e-mail should be : ``MeerSOLAR Logger Details: <start_time>``, where ``<start_time>`` will be in ``YYYY-MM-DDThh:mm:ss`` format. The e-mail will contain two information; ``Remote logger Job ID : <jobname>`` and ``Remote access password: <remote_access_password>``. Both of these are required to access logs in remote logger. A demo e-mail is shown below:

.. image :: _static/email.png 

Using remote logger
-------------------

1. The home page of remote logger will show ``<jobname>`` s of all jobs attached to this remote logger. These jobs can be run on different physical machines even in different parts of the world.

2. Click on the ``<jobname>`` user want to see the logs. It will then pop-up a window asking for password. Enter ``<remote_access_password>`` linked with that particular ``<jobname>``. If the ``<jobname>`` and ``<remote_access_password>`` are not matching, logs will be available to the user. This provides privacy and protection if multiple user are using the same remote logger. The password window will be as below. Enter ``<remote_access_password>`` and press the **Submit** button to enter into log page.

.. image :: _static/rl2.png

3. In the log page, different logs are shown. Logs names are self-explanatory. This list will be updating continuously as jobs are running in the host machine. 

.. image :: _static/rl3.png

4. Click on any of the logs will take you to the logger window, which will show the live-log and continuously updating as jobs are progressing in the host machine.

.. image :: _static/rl4.png

5. One can use, ``Back to log list`` and ``Back to Job ID list`` buttons to go back to the respective pages. To go back and forth between logs and logger page, user do not need to enter password everytime. However, if user go back to home page and again want to enter into the log page, user needs to enter the password again. This is for added safety.







