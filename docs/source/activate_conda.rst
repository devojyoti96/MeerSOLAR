Create and Activate Conda Environment
=====================================
This guideline provide how to activate **conda** environment before installing and using MeerSOLAR. 

Create conda environment in default conda directory:
----------------------------------------------------
This will create the conda environment in the default conda directory where conda is installed. If a custom directory is chosen during installation, environment will be created in that directory.

Create conda environment
~~~~~~~~~~~~~~~~~~~~~~~~

1. **To create a minimal Python 3.10 environment:**

.. code-block:: bash

   conda create -n meersolar_env python=3.10
   
Activate and deactivate conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **To activate the environment:**

.. code-block:: bash

   conda activate meersolar_env

2. **To deactivate:**

.. code-block:: bash

   conda deactivate

Create conda environment in custom conda directory:
---------------------------------------------------
To create in a custom directory ``</path/to/env>``, follow the steps below.

.. tip ::

    It is recommended to install in custom path in HPC architechture or your default conda path has limited disk space or does not have global access.

Create conda environment
~~~~~~~~~~~~~~~~~~~~~~~~

1. **To create a minimal Python 3.10 environment:**

.. code-block:: bash

   conda create -p </path/to/env>/meersolar_env python=3.10
   
Activate and deactivate conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **To activate the environment:**

.. code-block:: bash

   conda activate </path/to/env>/meersolar_env

2. **To deactivate:**

.. code-block:: bash

   conda deactivate

