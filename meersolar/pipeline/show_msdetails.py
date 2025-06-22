import argparse
import os
import tempfile
from casatasks import listobs

def show_listobs(msname):
    if not os.path.exists(msname):
        raise FileNotFoundError(f"Measurement Set not found: {msname}")
    listfile = msname.split(".ms")[0]+".listobs"
    os.system(f"rm -rf {listfile}")
    listobs(vis=msname, listfile=listfile, verbose=True)
    with open(listfile, 'r') as f:
        lines=f.readlines()
    filtered_lines=[]
    for line in lines:
        if "Sources:" in line:
            break
        filtered_lines.append(line)
    os.system(f"rm -rf {listfile}")
    print ("".join(filtered_lines))

def main():
    parser = argparse.ArgumentParser(description="Run listobs and show from saved file")
    parser.add_argument("msname", type=str, help="Path to the measurement set")
    args = parser.parse_args()
    show_listobs(args.msname)

if __name__ == "__main__":
    main()
