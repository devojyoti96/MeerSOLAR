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

3. Initiate necessary metadata

   .. code-block:: bash

      init_meersolar_data --init

4. Run MeerSOLAR pipeline

   .. code-block:: bash

      run_meersolar <path of measurement set> --workdir <path of work directory> --outdir <path of output products directory>

That’s all. You started MeerSOLAR pipeline for analysing your MeerKAT solar observation 🎉.

5. To see all running MeerSOLAR jobs

   .. code-block :: bash
        
      show_meersolar_status --show
       
6. To see local log of any job using the <jobid>

   .. code-block :: bash
    
      run_meersolar_log --jobid <jobid>
      
7. Output products will be saved in : ``<path of output products directory>``.

