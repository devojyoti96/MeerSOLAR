Developer Instruction
=====================
1. Developers are instructed to follow the similar format of the scripts.

2. Include new functions in proper modules. If suitable module is not found, include them in ``basic_utils``.

3. After including any function, add a ``pytest`` in appropriate testing scripts and test it. Without proper ``pytest`` new functions will not be included in the main branch and future release.

Instructions for testing
-------------------------
## Steps to test
Install ``pytest`` using ``pip install pytest`` before running the test.


1. Go to test directory
   
   .. code-block:: bash
   
      cd <repo_path>/meersolar/tests

2. Download test data

   .. code-block:: bash
   
      python3 download_test_data.py
   
    
3. Run utils module test:

   .. code-block:: bash
   
      pytest -s -v utils

    
4. Run meerpipeline module test:
    
   .. code-block:: bash
   
      pytest -s -v meerpipeline


