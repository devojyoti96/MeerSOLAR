import pytest
import shutil
import tempfile
import os
from unittest.mock import patch, MagicMock
from meersolar.utils.resource_utils import *


@patch("meersolar.utils.resource_utils.platform.system", return_value="Linux")
@patch("meersolar.utils.resource_utils.os.path.isfile", return_value=True)
@patch("meersolar.utils.resource_utils.os.open", return_value=42)
@patch("meersolar.utils.resource_utils.os.close")
@patch("meersolar.utils.resource_utils.libc.posix_fadvise", return_value=0)
def test_drop_file_cache(
    mock_advise, mock_close, mock_open, mock_isfile, mock_platform, capsys
):
    drop_file_cache("/dummy/file", verbose=True)
    mock_open.assert_called_once()
    mock_close.assert_called_once()
    mock_advise.assert_called_once_with(42, 0, 0, 4)
    out = capsys.readouterr().out
    assert "[cache drop] Released: /dummy/file" in out


@patch("meersolar.utils.resource_utils.platform.system", return_value="Linux")
@patch("meersolar.utils.resource_utils.drop_file_cache")
@patch("meersolar.utils.resource_utils.os.path.isfile", return_value=True)
@patch("meersolar.utils.resource_utils.os.path.isdir", return_value=False)
def test_drop_cache(mock_isdir, mock_isfile, mock_drop, mock_platform):
    drop_cache("/some/file", verbose=True)
    mock_drop.assert_called_once_with("/some/file", verbose=True)


def test_has_space():
    assert has_space("/tmp", 0.0) == True
    assert has_space("/tmp", 10**10) == False
    assert has_space("/tmp1", 0.0) == False


@patch("meersolar.utils.resource_utils.has_space")
@patch("meersolar.utils.resource_utils.tempfile.mkdtemp")
@patch("meersolar.utils.resource_utils.shutil.rmtree")
@patch("meersolar.utils.resource_utils.os.getcwd", return_value="/fallback")
def test_shm_or_tmp(mock_getcwd, mock_rmtree, mock_mkdtemp, mock_has_space):
    mock_has_space.side_effect = lambda path, gb: path == "/dev/shm"
    mock_mkdtemp.return_value = "/dev/shm/solar_temp123"
    original_tmpdir = os.environ.get("TMPDIR")
    with shm_or_tmp(required_gb=1.0, workdir="/fallback") as tempdir:
        assert tempdir == "/dev/shm/solar_temp123"
        assert os.environ["TMPDIR"] == tempdir
        mock_mkdtemp.assert_called_once_with(dir="/dev/shm", prefix="solar_")
    if original_tmpdir is not None:
        assert os.environ["TMPDIR"] == original_tmpdir
    else:
        assert "TMPDIR" not in os.environ
    mock_rmtree.assert_called_once_with("/dev/shm/solar_temp123")


@pytest.mark.parametrize(
    "platform_name, should_call_drop", [("Linux", True), ("Darwin", False)]
)
@patch("meersolar.utils.resource_utils.shm_or_tmp")
@patch("meersolar.utils.resource_utils.drop_cache")
@patch("meersolar.utils.resource_utils.platform.system")
def test_tmp_with_cache_rel(
    mock_platform, mock_drop_cache, mock_shm_or_tmp, platform_name, should_call_drop
):
    mock_platform.return_value = platform_name
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = "/fake/tempdir"
    mock_shm_or_tmp.return_value = mock_ctx
    with tmp_with_cache_rel(1.0, "/fallback", verbose=True) as tempdir:
        assert tempdir == "/fake/tempdir"
    mock_shm_or_tmp.assert_called_once()
    if should_call_drop:
        mock_drop_cache.assert_called_once_with("/fake/tempdir")
    else:
        mock_drop_cache.assert_not_called()


def test_limit_threads():
    n_threads = 2
    limit_threads(n_threads=n_threads)
    assert os.environ.get("OMP_NUM_THREADS") == str(n_threads)
    assert os.environ.get("OPENBLAS_NUM_THREADS") == str(n_threads)
    assert os.environ.get("MKL_NUM_THREADS") == str(n_threads)
    assert os.environ.get("VECLIB_MAXIMUM_THREADS") == str(n_threads)
