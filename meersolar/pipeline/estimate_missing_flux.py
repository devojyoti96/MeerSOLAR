import numpy as np
from meersolar.pipeline.basic_func import *


def cal_missing_flux(workdir, obs_date, obs_time, start_freq, end_freq, freqres):
    cmd = f"run_solstar --workdir {workdir} --obs_date {obs_date} --obs_time {obs_time} --observatory MeerKAT --start_freq {start_freq} --end_freq {end_freq} --freqres {freqres} --make_ms True"
    dt_string = (
        "".join(options.obs_date.split("-"))
        + "_"
        + "".join(options.obs_time.split(":"))
    )
    simulated_ms = workdir + f"/simulated_{dt_string}.ms"
    imagedir = (
        workdir
        + f"/images_simulated_{dt_string}_freq_"
        + str(int(start_freq))
        + "_"
        + str(int(end_freq))
    )
    print(f"Simulating measurement set for : {obs_date} {obs_time}")
    print(cmd)
    os.system(cmd)
    if os.path.exists(simulated_ms) == False:
        print(f"Simulation is failed for : {obs_date} {obs_time}")
        return
    else:
        return
