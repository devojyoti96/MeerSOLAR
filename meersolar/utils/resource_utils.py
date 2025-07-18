import types
import shutil
import traceback
import platform
import ctypes
import tempfile
import os
from contextlib import contextmanager


POSIX_FADV_DONTNEED = 4
libc = ctypes.CDLL("libc.so.6")


#####################################
# Resource management
#####################################
def drop_file_cache(filepath, verbose=False):
    """
    Advise the OS to drop the given file from the page cache.
    Safe, per-file, no sudo required.
    """
    if platform.system() != "Linux":
        raise NotImplementedError("drop_file_cache is only supported on Linux")
    try:
        if not os.path.isfile(filepath):
            return
        fd = os.open(filepath, os.O_RDONLY)
        result = libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
        os.close(fd)
        if verbose:
            if result == 0:
                print(f"[cache drop] Released: {filepath}")
            else:
                print(f"[cache drop] Failed ({result}) for: {filepath}")
    except Exception as e:
        if verbose:
            print(f"[cache drop] Error for {filepath}: {e}")
            traceback.print_exc()


def drop_cache(path, verbose=False):
    """
    Drop file cache for a file or all files under a directory.

    Parameters
    ----------
    path : str
        File or directory path
    """
    if platform.system() != "Linux":
        raise NotImplementedError("drop_file_cache is only supported on Linux")
    if os.path.isfile(path):
        drop_file_cache(path, verbose=verbose)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                full_path = os.path.join(root, f)
                drop_file_cache(full_path, verbose=verbose)
    else:
        if verbose:
            print(f"[cache drop] Path does not exist or is not valid: {path}")


def has_space(path, required_gb):
    try:
        stat = shutil.disk_usage(path)
        return (stat.free / 1e9) >= required_gb
    except BaseException:
        return False


@contextmanager
def shm_or_tmp(required_gb, workdir, prefix="solar_", verbose=False):
    """
    Create a temporary working directory:
    1. Try /dev/shm if it has required space
    2. Else TMPDIR if set and has space
    4. Else work directory
    Temporarily sets TMPDIR to the selected path during the context.
    Cleans up after use.

    Parameters
    ----------
    required_gb : float
        Required disk space in GB
    workdir : str
        Fall back work directory
    prefix : str, optional
        Temp directory prefix
    verbose : bool, optional
        Verbose
    """
    candidates = []
    if has_space("/dev/shm", required_gb):
        candidates.append("/dev/shm")
    tmpdir_env = os.environ.get("TMPDIR")
    if tmpdir_env is not None and has_space(tmpdir_env, required_gb):
        candidates.append(tmpdir_env)
    candidates.append(os.getcwd())
    for i in range(len(candidates)):
        base_dir = candidates[i]
        try:
            temp_dir = tempfile.mkdtemp(dir=base_dir, prefix=prefix)
            if verbose:
                if i == 0:
                    print("Using RAM")
                elif i == 1:
                    print("Using {os.environ.get('TMPDIR')}")
                else:
                    print("Using {os.getcwd()}")
            break
        except Exception as e:
            print(f"[shm_or_tmp] Failed to create temp dir in {base_dir}: {e}")
    else:
        raise RuntimeError(
            "Could not create a temporary directory in any fallback location."
        )
    # Override TMPDIR
    old_tmpdir = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = temp_dir
    try:
        yield temp_dir
    finally:
        # Restore TMPDIR
        if old_tmpdir is not None:
            os.environ["TMPDIR"] = old_tmpdir
        else:
            os.environ.pop("TMPDIR", None)
        # Clean up the temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"[cleanup] Warning: could not delete {temp_dir}: {e}")


@contextmanager
def tmp_with_cache_rel(required_gb, workdir, prefix="solar_", verbose=False):
    """
    Combined context manager:
    - Uses shm_or_tmp() for workspace
    - Drops kernel page cache for all files in that directory on exit

    Parameters
    ----------
    required_gb : float
        Required disk space in GB
    workdir : str
        Fall back work directory
    prefix : str, optional
        Temp directory prefix
    verbose : bool, optional
        Verbose

    """
    with shm_or_tmp(required_gb, workdir, prefix=prefix, verbose=verbose) as tempdir:
        try:
            yield tempdir
        finally:
            if platform.system() == "Linux":
                drop_cache(tempdir)


def limit_threads(n_threads=-1):
    """
    Limit number of threads usuage

    Parameters
    ----------
    n_threads : int, optional
        Number of threads
    """
    if n_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)


# Exposing only functions
__all__ = [
    name
    for name, obj in globals().items()
    if isinstance(obj, types.FunctionType) and obj.__module__ == __name__
]
