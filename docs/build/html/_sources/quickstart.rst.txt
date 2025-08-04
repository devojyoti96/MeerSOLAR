Quickstart
==========
MeerSOLAR is distributed on
`PyPI <https://pypi.org/project/meersolar/>`__. To use it:

1. Create conda environment with python 3.10

   .. code-block:: bash

      conda create -n meersolar_env python=3.10
      conda activate meersolar_env

2. Install MeerSOLAR in conda environment

   .. code-block:: bash

      pip install meersolar

3. Initiate necessary post-installation setup for metadata and ``prefect`` server

   .. code-block:: bash

      init-meersolar-setup --init

4. Run MeerSOLAR pipeline

   .. code-block:: bash

      run-meer-meersolar <path of measurement set> --workdir <path of work directory> --outdir <path of output products directory>

That’s all. You started MeerSOLAR pipeline for analysing your MeerKAT solar observation 🎉. Read the ``Directory Structure and Data Products`` section to understand how to find final images.

5. To see all running MeerSOLAR jobs

   .. code-block :: bash
        
      show-meersolar-status --show
      
       
6. To see ``prefect`` dashboard, if ``prefect`` server is running:

   .. code-block :: bash
    
      run-meer-meerlogger

7. If ``prefect`` dashboard is not showing logs, use local log of any job using the <jobid>:

   .. code-block :: bash
    
      run-meer-meerlogger --jobid <jobid>
      
8. If ``prefect`` dashboard is running, to see local log of any job using the <jobid>:

   .. code-block :: bash
    
      run-meer-meerlogger --jobid <jobid> --no-prefect
      
9. Output products will be saved in : ``<path of output products directory>``.

