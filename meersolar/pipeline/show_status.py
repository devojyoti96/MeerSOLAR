import os, glob, psutil, argparse, sys, traceback
from meersolar.pipeline.basic_func import *


def show_job_status(clean_old_jobs=False):
    """
    Show MeerSOLAR jobs status

    Parameters
    ----------
    clean_old_jobs : bool, optional
        Clean old informations for stopped jobs
    """
    datadir = get_datadir()
    try:
        main_pid_files = glob.glob(f"{datadir}/main_pids_*.txt")
        if len(main_pid_files) == 0:
            print("No MeerSOLAR jobs is running.")
        else:
            print("####################")
            print("MeerSOLAR Job status")
            print("####################")
            for pid_file in main_pid_files:
                with open(pid_file, "r") as f:
                    line = f.read().split(" ")
                jobid = line[0]
                pid = line[1]
                workdir = line[2]
                if psutil.pid_exists(int(pid)):
                    running = "Running/Waiting"
                else:
                    running = "Done/Stopped"
                print(f"Job ID: {jobid}, Work direcory: {workdir}, Status: {running}")
                if clean_old_jobs and running == "Done/Stopped":
                    os.system(f"rm -rf {pid_file}")
                    if os.path.exists(f"rm -rf {datadir}/pids/pids_{jobid}.txt"):
                        os.system(f"rm -rf {datadir}/pids/pids_{jobid}.txt")
    except Exception as e:
        traceback.print_exc()
    finally:
        drop_cache(datadir)


def main():
    parser = argparse.ArgumentParser(
        description="Show MeerSOLAR jobs status.",
        formatter_class=SmartDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--show",
        action="store_true",
        dest="show",
        required=True,
        help="Show job status",
    )
    parser.add_argument(
        "--clean_old_jobs",
        action="store_true",
        dest="clean_old_jobs",
        default=False,
        help="Clean old jobs",
    )
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    try:
        args = parser.parse_args()
        if args.show:
            show_job_status(clean_old_jobs=args.clean_old_jobs)
    except Exception as e:
        traceback.print_exc()
        


if __name__ == "__main__":
    main()
