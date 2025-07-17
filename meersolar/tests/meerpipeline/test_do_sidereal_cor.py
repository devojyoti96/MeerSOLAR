import pytest
from unittest.mock import patch, MagicMock
from meersolar.meerpipeline.do_sidereal_cor import *


@pytest.mark.parametrize(
    "container_present, compute_result, sidereal_exists, expected_code, expected_mslist",
    [
        # Test case: both MS succeed, one has .sidereal_cor
        (
            True,
            [0, 1],
            [True, False],
            0,
            ["/data/test1.ms"],
        ),
        # Test case: container missing and fails to initialize
        (
            False,
            [],
            [],
            1,
            [],
        ),
        # Test case: no MSs succeeded
        (
            True,
            [1, 1],
            [False, False],
            1,
            [],
        ),
    ],
)
@patch("meersolar.meerpipeline.do_sidereal_cor.drop_cache")
@patch("meersolar.meerpipeline.do_sidereal_cor.time.sleep", return_value=None)
@patch("meersolar.meerpipeline.do_sidereal_cor.delayed")
@patch("meersolar.meerpipeline.do_sidereal_cor.get_dask_client")
@patch(
    "meersolar.meerpipeline.do_sidereal_cor.run_limited_memory_task", return_value=4.0
)
@patch(
    "meersolar.meerpipeline.do_sidereal_cor.initialize_wsclean_container",
    return_value=None,
)
@patch("meersolar.meerpipeline.do_sidereal_cor.check_udocker_container")
@patch("meersolar.meerpipeline.do_sidereal_cor.os.path.exists")
def test_cor_sidereal_motion(
    mock_exists,
    mock_check_container,
    mock_init_container,
    mock_run_memtask,
    mock_get_dask,
    mock_delayed,
    mock_sleep,
    mock_drop,
    container_present,
    compute_result,
    sidereal_exists,
    expected_code,
    expected_mslist,
):
    mslist = ["/data/test1.ms", "/data/test2.ms"]
    workdir = "/tmp/test"

    # === Setup Mock Behavior ===

    # Container check and init
    mock_check_container.return_value = container_present
    if not container_present:
        mock_init_container.return_value = None

    # Delayed returns MagicMock
    mock_delayed.side_effect = lambda *args, **kwargs: MagicMock()

    # Dask client mock
    mock_client = MagicMock()
    mock_cluster = MagicMock()
    mock_get_dask.return_value = (mock_client, mock_cluster, 2, 2, 4.0)

    # Compute mock return
    mock_client.compute.return_value = compute_result

    # Mock existence of .sidereal_cor
    def exists_side_effect(path):
        if ".sidereal_cor" in path:
            return sidereal_exists[mslist.index(path.replace("/.sidereal_cor", ""))]
        return True

    mock_exists.side_effect = exists_side_effect

    # === Run test ===
    code, corrected = cor_sidereal_motion(mslist, workdir)

    assert code == expected_code
    assert corrected == expected_mslist


@pytest.mark.parametrize(
    "mslist_str, ms_exists, cor_success, expected_msg",
    [
        ("ms1.ms,ms2.ms", True, True, 0),  # Success
        ("ms1.ms,ms2.ms", True, False, 1),  # cor_sidereal_motion fails, but handled
        ("", False, False, 1),  # No MS provided
    ],
)
@patch("meersolar.meerpipeline.do_sidereal_cor.cor_sidereal_motion")
@patch("meersolar.meerpipeline.do_sidereal_cor.save_pid")
@patch(
    "meersolar.meerpipeline.do_sidereal_cor.get_cachedir", return_value="/mock/cache"
)
@patch("os.makedirs")
@patch("os.path.exists")
@patch("os.getpid", return_value=9876)
@patch("meersolar.meerpipeline.do_sidereal_cor.drop_cache")
@patch("meersolar.meerpipeline.do_sidereal_cor.clean_shutdown")
@patch("time.sleep", return_value=None)
@patch("traceback.print_exc", return_value=None)
def test_main_sidereal(
    mock_trace,
    mock_sleep,
    mock_shutdown,
    mock_drop_cache,
    mock_getpid,
    mock_exists,
    mock_makedirs,
    mock_cachedir,
    mock_save_pid,
    mock_cor_sidereal,
    mslist_str,
    ms_exists,
    cor_success,
    expected_msg,
):
    from meersolar.meerpipeline.do_sidereal_cor import main

    mslist = mslist_str.split(",") if mslist_str else []

    def exists_side_effect(path):
        return ms_exists if path in mslist else False

    mock_exists.side_effect = exists_side_effect
    mock_cor_sidereal.return_value = (0 if cor_success else 1, ["corrected.ms"])

    result = main(
        mslist=mslist_str,
        workdir="/mock/workdir",
        cpu_frac=0.7,
        mem_frac=0.6,
        max_cpu_frac=0.8,
        max_mem_frac=0.8,
        logfile=None,
        jobid=5,
        start_remote_log=False,
    )
    assert result == expected_msg


@pytest.mark.parametrize(
    "argv, should_exit",
    [
        (["prog.py"], True),  # No args
        (
            ["prog.py", "ms1.ms,ms2.ms", "--workdir", "/mock/work"],
            False,
        ),  # Normal
    ],
)
@patch(
    "meersolar.meerpipeline.do_sidereal_cor.cor_sidereal_motion",
    return_value=(0, ["ms1.ms"]),
)
@patch("meersolar.meerpipeline.do_sidereal_cor.save_pid")
@patch(
    "meersolar.meerpipeline.do_sidereal_cor.get_cachedir", return_value="/mock/cache"
)
@patch("os.makedirs")
@patch("os.path.exists", return_value=True)
@patch("os.getpid", return_value=6789)
@patch("meersolar.meerpipeline.do_sidereal_cor.drop_cache")
@patch("meersolar.meerpipeline.do_sidereal_cor.clean_shutdown")
@patch("time.sleep", return_value=None)
def test_cli_sidereal(
    mock_sleep,
    mock_shutdown,
    mock_drop,
    mock_getpid,
    mock_exists,
    mock_makedirs,
    mock_cachedir,
    mock_save_pid,
    mock_cor_sidereal,
    argv,
    should_exit,
):
    import sys
    from unittest.mock import patch
    from meersolar.meerpipeline.do_sidereal_cor import cli

    with patch.object(sys, "argv", argv):
        if should_exit:
            import pytest

            with pytest.raises(SystemExit) as e:
                cli()
            assert e.value.code == 1
        else:
            result = cli()
            assert result == 0
