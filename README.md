<p align="center">
  <img src="https://raw.githubusercontent.com/devojyoti96/MeerSOLAR/refs/heads/master/dark_logo.png" alt="MeerSOLAR Logo" width="200"/>
</p>
<p align="center">
  <h1>MeerSOLAR</h1> An automated calibration and imaging pipeline designed for solar radio observations using <strong>MeerKAT</strong> radio telescope. It performs end-to-end calibration, flagging, and imaging with a focus on dynamic solar data, supporting both spectral and temporal flexibility in imaging products.
</p>

## Background

<!-- start elevator-pitch -->

Solar radio data presents unique challenges due to the high variability and brightness of the Sun, as well as the need for high time-frequency resolution. The **MeerSOLAR** pipeline addresses these challenges by:

- Automating the calibration of interferometric data, including flux, phase, and polarization calibrations
- Supporting time-sliced and frequency-sliced imaging workflows
- Leveraging Dask for scalable parallel processing
- Providing hooks for integration with contextual data from other wavelegths for enhanced solar analysis

<!-- end elevator-pitch -->

## Documentation

MeerSOLAR documentation is available at: [meersolar.readthedocs.io]

[meersolar.readthedocs.io]: https://meersolar.readthedocs.io 

## Quickstart

<!-- start quickstart -->

**MeerSOLAR** is distributed on [PyPI]. To use it:

1. Create conda environment with python 3.10

    ```text
    conda create -n meersolar_env python=3.10
    conda activate meersolar_env
    ```

2. Install MeerSOLAR in conda environment

   ```text
   pip install meersolar
   ```

3. Initiate necessary metadata

    ```text
    init-meersolar-setup --init
    ```
    
4. Run MeerSOLAR pipeline

    ```text
    run-meer-meersolar <path of measurement set> --workdir <path of work directory> --outdir <path of output products directory>
    ```    

That's all. You started MeerSOLAR pipeline for analysing your MeerKAT solar observation 🎉.

5. To see all running MeerSOLAR jobs

    ```text
    show-meersolar-status --show
    ```
    
6. To see prefect dashboard

   ```text
   run-meer-meerlogger
   ```
      
7. To see local log of any job using the <jobid>

   ```text
   run-meer-meerlogger --jobid <jobid>
   ```
   
7. Output products will be saved in : `<path of output products directory>`

[pypi]: https://pypi.org/project/meersolar/

<!-- end quickstart -->

## Sample dataset
User can download and test entire MeerSOLAR pipeline using the sample dataset available in Zenodo: https://doi.org/10.5281/zenodo.16068485. Do not use this sample dataset for any publication without permission from the developer.


## Acknowledgements

MeerSOLAR is developed by Devojyoti Kansabanik (CPAESS-UCAR, Boulder, USA) and Deepan Patra (NCRA-TIFR, Pune, India). If you use **MeerSOLAR** for analysing your MeerKAT solar observations, include the following statement in your paper

```text
This MeerKAT solar observations are analysed using MeerSOLAR pipeline.
```

1. MeerSOLAR software in zenodo: https://doi.org/10.5281/zenodo.16058861

<!-- will be updated one published.
and cite the following papers.


1. [First MeerSOLAR paper] [kansabanik2025]
[Kansabanik2025]: https://kansabanik-meersolar.org

2. [Second MeerSOLAR paper] [Patra2025]
[Patra2025]: https://patra-meersolar.org
-->

## License

This project is licensed under the MIT License.
