# Tests for MeerSOLAR

This directory contains the full testing suite for the **MeerSOLAR** pipeline using ``pytest``.

## Steps to test
Install ``pytest`` using ``pip install pytest`` before running the test.


1. Go to test directory
   ```
   text
   cd <repo_path>/meersolar/tests

2. Download test data

    ```text
    python3 download_test_data.py
    ```
    
3. Run utils module test:

    ```
    text
    pytest -s -v utils
    ```
    
4. Run meerpipeline module test:
    
   ```
   text
   pytest -s -v meerpipeline
   ```
    




