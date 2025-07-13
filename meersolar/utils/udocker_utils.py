import types
import psutil
import traceback
import tempfile
import time
import glob
import os
from .basic_utils import *

####################
# uDOCKER related
####################


def set_udocker_env():
    datadir = get_datadir()
    udocker_dir = datadir + "/udocker"
    os.makedirs(udocker_dir, exist_ok=True)
    os.environ["UDOCKER_DIR"] = udocker_dir
    os.environ["UDOCKER_TARBALL"] = datadir + "/udocker-englib-1.2.11.tar.gz"


def init_udocker():
    set_udocker_env()
    os.system("udocker install")


def check_udocker_container(name):
    """
    Check whether a docker container is present or not

    Parameters
    ----------
    name : str
        Container name

    Returns
    -------
    bool
        Whether present or not
    """
    set_udocker_env()
    pid = os.getpid()
    timestamp = int(time.time() * 1000)
    tmp1 = f"tmp1_{pid}_{timestamp}.txt"
    tmp2 = f"tmp2_{pid}_{timestamp}.txt"
    b = os.system(
        f"udocker --insecure --quiet inspect " + name + f" >> {tmp1} >> {tmp2}"
    )
    os.system(f"rm -rf {tmp1} {tmp2}")
    if b != 0:
        return False
    else:
        return True


def initialize_wsclean_container(name="solarwsclean"):
    """
    Initialize WSClean container

    Parameters
    ----------
    name : str
        Name of the container

    Returns
    -------
    bool
        Whether initialized successfully or not
    """
    set_udocker_env()
    image_name = "devojyoti96/wsclean-solar:latest"
    check_cmd = f"udocker images | grep -q '{image_name}'"
    image_exists = os.system(check_cmd) == 0
    if not image_exists:
        a = os.system(f"udocker pull {image_name}")
    else:
        print(f"Image '{image_name}' already present.")
        a = 0
    if a == 0:
        a = os.system(f"udocker create --name={name} {image_name}")
        print(f"Container started with name : {name}")
        return name
    else:
        print(f"Container could not be created with name : {name}")
        return


def run_wsclean(
    wsclean_cmd,
    container_name="solarwsclean",
    check_container=False,
    verbose=False,
    dry_run=False,
):
    """
    Run WSClean inside a udocker container (no root permission required).

    Parameters
    ----------
    wsclean_cmd : str
        Full WSClean command as a string.
    container_name : str, optional
        Container name
    check_container : bool, optional
        Check container presence or not
    verbose : bool, optional
        Verbose output or not

    Returns
    -------
    int
        Success message
    """
    set_udocker_env()
    pid = os.getpid()
    timestamp = int(time.time() * 1000)
    tmp1 = f"tmp1_{pid}_{timestamp}.txt"
    tmp2 = f"tmp2_{pid}_{timestamp}.txt"

    def show_file(path):
        try:
            print(open(path).read())
        except Exception as e:
            print(f"Error: {e}")

    if check_container:
        container_present = check_udocker_container(container_name)
        if not container_present:
            container_name = initialize_wsclean_container(name=container_name)
            if container_name is None:
                print(
                    f"Container {container_name} is not initiated. First initiate container and then run."
                )
                return 1

    if dry_run:
        cmd = f"chgenter >> {tmp1} >> {tmp2}"
        cwd = os.getcwd()
        full_command = (
            f"udocker --quiet run --nobanner --volume={cwd}:{cwd} solarwsclean {cmd}"
        )
        os.system(full_command)
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        os.system(f"rm -rf {tmp1} {tmp2}")
        return mem

    msname = wsclean_cmd.split(" ")[-1]
    msname = os.path.abspath(msname)
    mspath = os.path.dirname(msname)
    temp_docker_path = tempfile.mkdtemp(prefix="wsclean_udocker_", dir=mspath)
    wsclean_cmd_args = wsclean_cmd.split(" ")[:-1]
    if "-fits-mask" in wsclean_cmd_args:
        index = wsclean_cmd_args.index("-fits-mask")
        name = wsclean_cmd_args[index + 1]
        namedir = os.path.dirname(os.path.abspath(name))
        basename = os.path.basename(os.path.abspath(name))
        wsclean_cmd_args.remove(name)
        wsclean_cmd_args.insert(index + 1, temp_docker_path + "/" + basename)
    if "-name" not in wsclean_cmd_args:
        wsclean_cmd_args.append(
            "-name " + temp_docker_path + "/" + os.path.basename(msname).split(".ms")[0]
        )
    else:
        index = wsclean_cmd_args.index("-name")
        name = wsclean_cmd_args[index + 1]
        namedir = os.path.dirname(os.path.abspath(name))
        basename = os.path.basename(os.path.abspath(name))
        wsclean_cmd_args.remove(name)
        wsclean_cmd_args.insert(index + 1, temp_docker_path + "/" + basename)
    if "-temp-dir" not in wsclean_cmd_args:
        wsclean_cmd_args.append("-temp-dir " + temp_docker_path)
    else:
        index = wsclean_cmd_args.index("-temp-dir")
        name = os.path.abspath(wsclean_cmd_args[index + 1])
        wsclean_cmd_args.remove(name)
        wsclean_cmd_args.insert(index + 1, temp_docker_path)
    wsclean_cmd = (
        " ".join(wsclean_cmd_args)
        + " "
        + temp_docker_path
        + "/"
        + os.path.basename(msname)
    )
    try:
        full_command = f"udocker run --nobanner --volume={mspath}:{temp_docker_path} --workdir {temp_docker_path} solarwsclean {wsclean_cmd}"
        if not verbose:
            full_command += f" >> {mspath}/{tmp1} "
        else:
            print(wsclean_cmd + "\n")
        exit_code = os.system(full_command)
        if exit_code != 0:
            print("##########################")
            print(os.path.basename(msname))
            print("##########################")
            show_file(f"{mspath}/{tmp1}")
        os.system(f"rm -rf {temp_docker_path} {mspath}/{tmp1}")
        return 0 if exit_code == 0 else 1
    except Exception as e:
        os.system(f"rm -rf {temp_docker_path}")
        traceback.print_exc()
        return 1


def run_solar_sidereal_cor(
    msname="",
    only_uvw=False,
    container_name="solarwsclean",
    check_container=False,
    verbose=False,
    dry_run=False,
):
    """
    Run chgcenter inside a udocker container to correct solar sidereal motion (no root permission required).

    Parameters
    ----------
    msname : str
        Name of the measurement set
    only_uvw : bool, optional
        Update only UVW values
        Note: This is required when visibilities are properly phase rotated in correlator to track the Sun,
        but while creating the MS, UVW values are estimated using the first phasecenter of the Sun.
    check_container : bool, optional
        Check container
    container_name : str, optional
        Container name
    verbose : bool, optional
        Verbose output or not

    Returns
    -------
    int
        Success message
    """
    set_udocker_env()
    pid = os.getpid()
    timestamp = int(time.time() * 1000)
    tmp1 = f"tmp1_{pid}_{timestamp}.txt"
    tmp2 = f"tmp2_{pid}_{timestamp}.txt"
    if check_container:
        container_present = check_udocker_container(container_name)
        if not container_present:
            container_name = initialize_wsclean_container(name=container_name)
            if container_name is None:
                print(
                    f"Container {container_name} is not initiated. First initiate container and then run."
                )
                return 1

    if dry_run:
        cmd = f"chgcentre >> {tmp1} >> {tmp2}"
        cwd = os.getcwd()
        full_command = (
            f"udocker --quiet run --nobanner --volume={cwd}:{cwd} solarwsclean {cmd}"
        )
        os.system(full_command)
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        os.system(f"rm -rf {tmp1} {tmp2}")
        return mem

    msname = os.path.abspath(msname)
    mspath = os.path.dirname(msname)
    temp_docker_path = tempfile.mkdtemp(prefix="chgcenter_udocker_", dir=mspath)
    if only_uvw:
        cmd = (
            "chgcentre -only-uvw -solarcenter "
            + temp_docker_path
            + "/"
            + os.path.basename(msname)
        )
    else:
        cmd = (
            "chgcentre -solarcenter "
            + temp_docker_path
            + "/"
            + os.path.basename(msname)
        )
    try:
        full_command = f"udocker --quiet run --nobanner --volume={mspath}:{temp_docker_path} --workdir {temp_docker_path} solarwsclean {cmd}"
        if not verbose:
            full_command += f" >> {tmp1} >> {tmp2}"
        else:
            print(cmd)
        with suppress_casa_output():
            exit_code = os.system(full_command)
        os.system(f"rm -rf {temp_docker_path} {tmp1} {tmp2}")
        return 0 if exit_code == 0 else 1
    except Exception as e:
        os.system(f"rm -rf {temp_docker_path} {tmp1} {tmp2}")
        traceback.print_exc()
        return 1


def run_chgcenter(
    msname,
    ra,
    dec,
    only_uvw=False,
    container_name="solarwsclean",
    verbose=False,
    dry_run=False,
):
    """
    Run chgcenter inside a udocker container (no root permission required).

    Parameters
    ----------
    msname : str
        Name of the measurement set
    ra : str
        RA can either be 00h00m00.0s or 00:00:00.0
    dec : str
        Dec can either be 00d00m00.0s or 00.00.00.0
    only_uvw : bool, optional
        Update only UVW values
        Note: This is required when visibilities are properly phase rotated in correlator,
        but while creating the MS, UVW values are estimated using a wrong phase center.
    container_name : str, optional
        Container name
    verbose : bool, optional
        Verbose output

    Returns
    -------
    int
        Success message
    """
    set_udocker_env()
    pid = os.getpid()
    timestamp = int(time.time() * 1000)
    tmp1 = f"tmp1_{pid}_{timestamp}.txt"
    tmp2 = f"tmp2_{pid}_{timestamp}.txt"
    container_present = check_udocker_container(container_name)
    if not container_present:
        container_name = initialize_wsclean_container(name=container_name)
        if container_name is None:
            print(
                f"Container {container_name} is not initiated. First initiate container and then run."
            )
            return 1
    if dry_run:
        cmd = f"chgenter >> {tmp1} >> {tmp2}"
        cwd = os.getcwd()
        full_command = (
            f"udocker --quiet run --nobanner --volume={cwd}:{cwd} solarwsclean {cmd}"
        )
        os.system(full_command)
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        os.system(f"rm -rf {tmp1} {tmp2}")
        return mem
    msname = os.path.abspath(msname)
    mspath = os.path.dirname(msname)
    temp_docker_path = tempfile.mkdtemp(prefix="chgcenter_udocker_", dir=mspath)
    if only_uvw:
        cmd = (
            "chgcentre -only-uvw "
            + temp_docker_path
            + "/"
            + os.path.basename(msname)
            + " "
            + ra
            + " "
            + dec
        )
    else:
        cmd = (
            "chgcentre "
            + temp_docker_path
            + "/"
            + os.path.basename(msname)
            + " "
            + ra
            + " "
            + dec
        )
    try:
        full_command = f"udocker --quiet run --nobanner --volume={mspath}:{temp_docker_path} --workdir {temp_docker_path} solarwsclean {cmd}"
        if not verbose:
            full_command += f" >> {tmp1} >> {tmp2}"
        else:
            print(cmd)
        exit_code = os.system(full_command)
        os.system(f"rm -rf {temp_docker_path} {tmp1} {tmp2}")
        return 0 if exit_code == 0 else 1
    except Exception as e:
        os.system(f"rm -rf {temp_docker_path} {tmp1} {tmp2}")
        traceback.print_exc()
        return 1
    return


# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
