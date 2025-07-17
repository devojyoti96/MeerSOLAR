import argparse
import sys
import os
from casatasks import casalog
from casatasks import listobs
from meersolar.utils.resource_utils import drop_cache

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    pass


def show_listobs(msname):
    if not os.path.exists(msname):
        raise FileNotFoundError(f"Measurement Set not found: {msname}")
    listfile = msname.split(".ms")[0] + ".listobs"
    os.system(f"rm -rf {listfile}")
    listobs(vis=msname, listfile=listfile, verbose=True)
    with open(listfile, "r") as f:
        lines = f.readlines()
    filtered_lines = []
    for line in lines:
        if "Sources:" in line:
            break
        filtered_lines.append(line)
    os.system(f"rm -rf {listfile}")
    print("".join(filtered_lines))
    drop_cache(msname)


def cli():
    parser = argparse.ArgumentParser(description="Run listobs and show from saved file")
    parser.add_argument("msname", type=str, help="Path to the measurement set")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    show_listobs(args.msname)


if __name__ == "__main__":
    cli()
