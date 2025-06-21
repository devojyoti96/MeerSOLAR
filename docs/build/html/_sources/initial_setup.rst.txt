Initial setup
=============

After installation of **MeerSOLAR**, before running the pipeline, some initial setup is needed. These include downloading some required metadata for the pipeline and setup of remote logger.


Download MeerSOLAR metadata
---------------------------
1. To download and save the required MeerSOLAR metadata in appropriate directory, run from command line:

.. code-block :: bash
    
    init_meersolar_data --init
    
2. If data files are present, but needs to updated, run:

.. code-block :: bash
    init_meersolar_data --init --update
    
Setup remote logger link
-------------------------
If remote logger is intended to be used, setup the remote link in MeerSOLAR metadata.

.. code-block :: bash
    
    init_meersolar_data --init --remotelink https://<remote-logger-name>.onrender.com
    
Before doing this, create your own remote logger on free-tier cloud platform, https://render.com. One can use, same **remotelink** in multiple machines and users. However, free-tier link has some limitations on bandwidth. If you want to use **remotelink** for your institution, we suggest to purchase suitable paid version or setup seperate **remotelink** for different users.

Tutorial to setup **remotelink**
--------------------------------- 
    
