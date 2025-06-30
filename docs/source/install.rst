Installation 
============
This is the guideline to setup the pipeline in any workstation or HPC environment. The full python package can be installed using `pip`. 

.. important:: 
    Required python version 3.10. Other versions may work, but not tested. Hence, recommend to use python 3.10.
    
.. tip::
   To make life easier, both the Python and other dependencies can be installed using conda.

    First, ensure you have `anaconda <https://www.anaconda.com/products/individual/>`_ or `miniconda <https://docs.conda.io/en/latest/miniconda.html/>`_ installed with python version 3.10.
    

Load installed conda module in HPC cluster
------------------------------------------
In many HPC clusters, **conda** may be installed already. In that case, it is recommended to read HPC cluster document and check how to load **conda**. In general, it can be loaded using the **module** to function. Load it as:
  
  .. code-block:: bash
  
     module load anaconda3  
     
Otherwise, install it following the steps below.
    
Install Conda: Anaconda or Miniconda
------------------------------------
.. toctree::
   :maxdepth: 2
   
   install_conda
   
Create and Activate Conda
--------------------------
.. toctree::
   :maxdepth: 2
   
   activate_conda
   
Install MeerSOLAR
------------------

.. toctree::
   :maxdepth: 2
   
   install_meersolar

   

    

   


