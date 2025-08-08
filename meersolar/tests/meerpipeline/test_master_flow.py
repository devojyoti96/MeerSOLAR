import pytest
from unittest.mock import patch, MagicMock
from meersolar.meerpipeline.master_flow import *


@pytest.mark.parametrize("mock_msg,raises", [(0, False), (1, True)])
@patch("meersolar.meerpipeline.master_flow.get_run_context")
@patch("meersolar.meerpipeline.master_flow.start_log_task_saver")
@patch("meersolar.meerpipeline.master_flow.os.makedirs")
@patch("meersolar.meerpipeline.master_flow.os.path.exists")
@patch("meersolar.meerpipeline.master_flow.os.remove")
@patch("meersolar.meerpipeline.meer_make_ds.main")
@patch("meersolar.meerpipeline.master_flow.get_dask_client")
def test_run_ds_jobs(
    mock_dask_client,
    mock_main,
    mock_remove,
    mock_exists,
    mock_makedirs,
    mock_log_task_saver,
    mock_get_ctx,
    mock_msg,
    raises,
):
    # Setup mocks
    mock_main.return_value = mock_msg
    mock_exists.return_value = True

    mock_ctx = MagicMock()
    mock_ctx.task_run.id = "abc123"
    mock_ctx.task_run.name = "mock_task"
    mock_get_ctx.return_value = mock_ctx

    mock_thread = MagicMock()
    mock_log_task_saver.return_value = mock_thread

    # Parameters
    kwargs = dict(
        msname="mock.ms",
        workdir="/mock/workdir",
        outdir="/mock/outdir",
        target_scans=["scan1", "scan2"],
        jobid=42,
        cpu_frac=0.5,
        mem_frac=0.5,
        remote_log=True,
    )

    if raises:
        with pytest.raises(RuntimeError, match="Dynamic spectrum making is failed."):
            run_ds_jobs.fn(**kwargs)
    else:
        result = run_ds_jobs.fn(**kwargs)
        assert result == 0

    # Log file should be removed if exists
    mock_remove.assert_called_once_with("/mock/workdir/logs/ds_targets.log")
    mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    mock_log_task_saver.assert_called_once()
    mock_thread.join.assert_called_once_with(timeout=5)


@pytest.mark.parametrize("mock_msg,raises", [(0, False), (99, True)])
@patch("meersolar.meerpipeline.master_flow.get_run_context")
@patch("meersolar.meerpipeline.master_flow.start_log_task_saver")
@patch("meersolar.meerpipeline.master_flow.os.makedirs")
@patch("meersolar.meerpipeline.master_flow.os.path.exists")
@patch("meersolar.meerpipeline.master_flow.os.remove")
@patch("meersolar.meerpipeline.do_fluxcal.main")
@patch("meersolar.meerpipeline.master_flow.get_dask_client")
def test_run_noise_diode_cal(
    mock_client,
    mock_fluxcal_main,
    mock_remove,
    mock_exists,
    mock_makedirs,
    mock_log_saver,
    mock_get_ctx,
    mock_msg,
    raises,
):
    # Mock returns
    mock_fluxcal_main.return_value = mock_msg
    mock_exists.return_value = True

    # Mock context
    mock_ctx = MagicMock()
    mock_ctx.task_run.id = "xyz987"
    mock_ctx.task_run.name = "noise_task"
    mock_get_ctx.return_value = mock_ctx

    # Mock logging thread
    mock_thread = MagicMock()
    mock_log_saver.return_value = mock_thread

    # Arguments
    kwargs = dict(
        msname="mock.ms/",
        workdir="/mock/workdir",
        caldir="/mock/caldir",
        keep_backup=True,
        jobid=1,
        cpu_frac=0.6,
        mem_frac=0.5,
        remote_log=True,
    )

    if raises:
        with pytest.raises(RuntimeError, match="Attenuation calibration is failed."):
            run_noise_diode_cal.fn(**kwargs)
    else:
        result = run_noise_diode_cal.fn(**kwargs)
        assert result == 0

    mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    mock_remove.assert_called_once_with("/mock/workdir/logs/noise_cal.log")
    mock_log_saver.assert_called_once()
    mock_thread.join.assert_called_once_with(timeout=5)


@pytest.mark.parametrize("mock_msg,raises", [(0, False), (42, True)])
@patch("meersolar.meerpipeline.master_flow.get_run_context")
@patch("meersolar.meerpipeline.master_flow.start_log_task_saver")
@patch("meersolar.meerpipeline.master_flow.os.makedirs")
@patch("meersolar.meerpipeline.master_flow.os.path.exists")
@patch("meersolar.meerpipeline.master_flow.os.remove")
@patch("meersolar.meerpipeline.master_flow.copy.deepcopy")
@patch("meersolar.meerpipeline.master_flow.determine_noise_diode_cal_scan")
@patch("meersolar.meerpipeline.master_flow.get_cal_target_scans")
@patch("meersolar.meerpipeline.master_flow.msmetadata")
@patch("meersolar.meerpipeline.do_partition.main")
@patch("meersolar.meerpipeline.master_flow.get_dask_client")
def test_run_partition(
    mock_client,
    mock_main,
    mock_msmd,
    mock_get_scans,
    mock_det_noise,
    mock_deepcopy,
    mock_remove,
    mock_exists,
    mock_makedirs,
    mock_log_saver,
    mock_get_ctx,
    mock_msg,
    raises,
):
    # Set up return values
    mock_main.return_value = mock_msg
    mock_exists.return_value = True
    mock_deepcopy.side_effect = lambda x: x[:]  # just shallow copy for test
    mock_get_scans.return_value = ([11], [21, 22, 23], [1], [2], [3])
    mock_det_noise.return_value = None

    # Mock msmetadata tool
    mock_tool = MagicMock()
    mock_tool.nchan.return_value = 2048
    mock_tool.timesforfield.return_value = [0, 10]
    mock_tool.exposuretime.return_value = {"value": 2.0}
    mock_msmd.return_value = mock_tool

    # Prefect task context
    mock_ctx = MagicMock()
    mock_ctx.task_run.id = "mock-task-id"
    mock_ctx.task_run.name = "partitioning_calibrator"
    mock_get_ctx.return_value = mock_ctx

    # Log thread mock
    mock_thread = MagicMock()
    mock_log_saver.return_value = mock_thread

    # Arguments
    kwargs = dict(
        msname="mock.ms/",
        workdir="/mock/workdir",
        jobid=3,
        cpu_frac=0.6,
        mem_frac=0.7,
        remote_log=True,
    )

    if raises:
        with pytest.raises(
            RuntimeError, match="Partitioning calibrator scans is failed."
        ):
            run_partition.fn(**kwargs)
    else:
        result = run_partition.fn(**kwargs)
        assert result == 0

    mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    mock_remove.assert_called_once_with("/mock/workdir/logs/partition_cal.log")
    mock_thread.join.assert_called_once_with(timeout=5)
    mock_main.assert_called_once()


@pytest.mark.parametrize("mock_msg,raises", [(0, False), (99, True)])
@patch("meersolar.meerpipeline.master_flow.get_run_context")
@patch("meersolar.meerpipeline.master_flow.start_log_task_saver")
@patch("meersolar.meerpipeline.master_flow.os.makedirs")
@patch("meersolar.meerpipeline.master_flow.os.path.exists")
@patch("meersolar.meerpipeline.master_flow.os.remove")
@patch("meersolar.meerpipeline.do_target_split.main")
@patch("meersolar.meerpipeline.master_flow.get_dask_client")
def test_run_target_split_jobs(
    mock_client,
    mock_main,
    mock_remove,
    mock_exists,
    mock_makedirs,
    mock_log_saver,
    mock_get_ctx,
    mock_msg,
    raises,
):
    # Set up mocks
    mock_main.return_value = mock_msg
    mock_exists.return_value = True

    # Prefect task context mock
    mock_ctx = MagicMock()
    mock_ctx.task_run.id = "task-abc"
    mock_ctx.task_run.name = "spliting_target_scans"
    mock_get_ctx.return_value = mock_ctx

    # Thread mock
    mock_thread = MagicMock()
    mock_log_saver.return_value = mock_thread

    # Arguments
    kwargs = dict(
        msname="mock.ms/",
        workdir="/mock/workdir",
        datacolumn="corrected",
        spw="0:10~100",
        timeres=5.0,
        freqres=1.0,
        target_freq_chunk=4.0,
        n_spectral_chunk=2,
        target_scans=[1, 2, 3],
        prefix="targets",
        merge_spws=True,
        time_window=10.0,
        time_interval=5.0,
        jobid=123,
        cpu_frac=0.5,
        mem_frac=0.6,
        remote_log=True,
    )

    if raises:
        with pytest.raises(RuntimeError, match="Spliting target scans is failed."):
            run_target_split_jobs.fn(**kwargs)
    else:
        result = run_target_split_jobs.fn(**kwargs)
        assert result == 0

    # Validate expected calls
    mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    mock_remove.assert_called_once_with("/mock/workdir/logs/split_targets.log")
    mock_log_saver.assert_called_once()
    mock_thread.join.assert_called_once_with(timeout=5)
    mock_main.assert_called_once()


@pytest.mark.parametrize("mock_msg,raises", [(0, False), (1, True)])
@patch("meersolar.meerpipeline.master_flow.get_run_context")
@patch("meersolar.meerpipeline.master_flow.start_log_task_saver")
@patch("meersolar.meerpipeline.master_flow.os.makedirs")
@patch("meersolar.meerpipeline.master_flow.os.path.exists")
@patch("meersolar.meerpipeline.master_flow.os.remove")
@patch("meersolar.meerpipeline.import_model.main")
@patch("meersolar.meerpipeline.master_flow.get_dask_client")
def test_run_import_model(
    mock_client,
    mock_main,
    mock_remove,
    mock_exists,
    mock_makedirs,
    mock_log_saver,
    mock_get_ctx,
    mock_msg,
    raises,
):
    # Configure mocks
    mock_main.return_value = mock_msg
    mock_exists.return_value = True

    # Mock task context
    mock_ctx = MagicMock()
    mock_ctx.task_run.id = "task-xyz"
    mock_ctx.task_run.name = "importing_model_visibilities"
    mock_get_ctx.return_value = mock_ctx

    # Mock thread
    mock_thread = MagicMock()
    mock_log_saver.return_value = mock_thread

    # Parameters
    kwargs = dict(
        msname="mock.ms/",
        workdir="/mock/workdir",
        jobid=7,
        cpu_frac=0.6,
        mem_frac=0.7,
        remote_log=True,
    )

    if raises:
        with pytest.raises(RuntimeError, match="Importing calibrator model is failed."):
            run_import_model.fn(**kwargs)
    else:
        result = run_import_model.fn(**kwargs)
        assert result == 0

    log_file_path = "/mock/workdir/logs/modeling_mock.log"
    mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    mock_remove.assert_called_once_with(log_file_path)
    mock_log_saver.assert_called_once()
    mock_thread.join.assert_called_once_with(timeout=5)
    mock_main.assert_called_once()


@pytest.mark.parametrize("mock_msg,raises", [(0, False), (99, True)])
@patch("meersolar.meerpipeline.master_flow.get_run_context")
@patch("meersolar.meerpipeline.master_flow.start_log_task_saver")
@patch("meersolar.meerpipeline.master_flow.os.makedirs")
@patch("meersolar.meerpipeline.master_flow.os.path.exists")
@patch("meersolar.meerpipeline.master_flow.os.remove")
@patch("meersolar.meerpipeline.basic_cal.main")
@patch("meersolar.meerpipeline.master_flow.get_dask_client")
def test_run_basic_cal_jobs(
    mock_client,
    mock_main,
    mock_remove,
    mock_exists,
    mock_makedirs,
    mock_log_saver,
    mock_get_ctx,
    mock_msg,
    raises,
):
    # Setup
    mock_main.return_value = mock_msg
    mock_exists.return_value = True

    # Prefect context mock
    mock_ctx = MagicMock()
    mock_ctx.task_run.id = "abc123"
    mock_ctx.task_run.name = "basic_calibration"
    mock_get_ctx.return_value = mock_ctx

    # Logging thread mock
    mock_thread = MagicMock()
    mock_log_saver.return_value = mock_thread

    # Parameters
    kwargs = dict(
        msname="mock.ms/",
        workdir="/mock/workdir",
        caldir="/mock/caldir",
        perform_polcal=True,
        jobid=99,
        cpu_frac=0.7,
        mem_frac=0.8,
        keep_backup=True,
        remote_log=True,
    )

    if raises:
        with pytest.raises(RuntimeError, match="Basic calibration is failed."):
            run_basic_cal_jobs.fn(**kwargs)
    else:
        result = run_basic_cal_jobs.fn(**kwargs)
        assert result == 0

    log_path = "/mock/workdir/logs/basic_cal.log"
    mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    mock_remove.assert_called_once_with(log_path)
    mock_log_saver.assert_called_once()
    mock_thread.join.assert_called_once_with(timeout=5)
    mock_main.assert_called_once()


@pytest.mark.parametrize("mock_msg,raises", [(0, False), (1, True)])
@patch("meersolar.meerpipeline.master_flow.get_run_context")
@patch("meersolar.meerpipeline.master_flow.start_log_task_saver")
@patch("meersolar.meerpipeline.master_flow.os.makedirs")
@patch("meersolar.meerpipeline.master_flow.os.path.exists")
@patch("meersolar.meerpipeline.master_flow.os.remove")
@patch("meersolar.meerpipeline.do_apply_basiccal.main")
@patch("meersolar.meerpipeline.master_flow.get_dask_client", return_value=MagicMock())
def test_run_apply_basiccal_sol(
    mock_client,
    mock_main,
    mock_remove,
    mock_exists,
    mock_makedirs,
    mock_log_saver,
    mock_get_ctx,
    mock_msg,
    raises,
):
    # Setup
    mock_main.return_value = mock_msg
    mock_exists.return_value = True

    mock_ctx = MagicMock()
    mock_ctx.task_run.id = "task-id"
    mock_ctx.task_run.name = "apply_basiccal_target"
    mock_get_ctx.return_value = mock_ctx

    mock_thread = MagicMock()
    mock_log_saver.return_value = mock_thread

    dask_client = MagicMock()
    # Test input arguments
    kwargs = dict(
        target_mslist=["ms1.ms", "ms2.ms"],
        workdir="/mock/workdir",
        caldir="/mock/caldir",
        use_only_bandpass=True,
        overwrite_datacolumn=False,
        applymode="calonly",
        jobid=3,
        cpu_frac=0.5,
        mem_frac=0.6,
        remote_log=True,
    )

    if raises:
        with pytest.raises(
            RuntimeError, match="Applying basic calibration solutions is failed."
        ):
            run_apply_basiccal_sol.fn(**kwargs)
    else:
        result = run_apply_basiccal_sol.fn(**kwargs)
        assert result == 0

    expected_log = "/mock/workdir/logs/apply_basiccal_target.log"
    mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    mock_remove.assert_called_once_with(expected_log)
    mock_log_saver.assert_called_once()
    mock_thread.join.assert_called_once_with(timeout=5)
    mock_main.assert_called_once_with(
        "ms1.ms,ms2.ms",
        "/mock/workdir",
        "/mock/caldir",
        use_only_bandpass=True,
        applymode="calonly",
        overwrite_datacolumn=False,
        start_remote_log=True,
        cpu_frac=0.5,
        mem_frac=0.6,
        logfile=expected_log,
        jobid=3,
        dask_client=mock_client.return_value.__enter__.return_value,
    )


@pytest.mark.parametrize("mock_msg,raises", [(0, False), (1, True)])
@patch("meersolar.meerpipeline.master_flow.get_run_context")
@patch("meersolar.meerpipeline.master_flow.start_log_task_saver")
@patch("meersolar.meerpipeline.master_flow.os.makedirs")
@patch("meersolar.meerpipeline.master_flow.os.path.exists")
@patch("meersolar.meerpipeline.master_flow.os.remove")
@patch("meersolar.meerpipeline.do_sidereal_cor.main")
@patch("meersolar.meerpipeline.master_flow.get_dask_client")
def test_run_solar_siderealcor_jobs(
    mock_client,
    mock_main,
    mock_remove,
    mock_exists,
    mock_makedirs,
    mock_log_saver,
    mock_get_ctx,
    mock_msg,
    raises,
):
    # Setup
    mock_main.return_value = mock_msg
    mock_exists.return_value = True

    mock_ctx = MagicMock()
    mock_ctx.task_run.id = "mock-id"
    mock_ctx.task_run.name = "solar_sidereal_correction"
    mock_get_ctx.return_value = mock_ctx

    mock_thread = MagicMock()
    mock_log_saver.return_value = mock_thread

    kwargs = dict(
        mslist=["ms1.ms", "ms2.ms"],
        workdir="/mock/workdir",
        prefix="targets",
        jobid=99,
        cpu_frac=0.6,
        mem_frac=0.5,
        remote_log=True,
    )

    if raises:
        with pytest.raises(
            RuntimeError, match="Solar sidereal motion correction is failed."
        ):
            run_solar_siderealcor_jobs.fn(**kwargs)
    else:
        result = run_solar_siderealcor_jobs.fn(**kwargs)
        assert result == 0

    expected_log = "/mock/workdir/logs/cor_sidereal_targets.log"
    mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    mock_remove.assert_called_once_with(expected_log)
    mock_log_saver.assert_called_once()
    mock_thread.join.assert_called_once_with(timeout=5)
    mock_main.assert_called_once_with(
        "ms1.ms,ms2.ms",
        workdir="/mock/workdir",
        cpu_frac=0.6,
        mem_frac=0.5,
        logfile=expected_log,
        jobid=99,
        start_remote_log=True,
        dask_client=mock_client.return_value.__enter__.return_value,
    )


@pytest.mark.parametrize("mock_msg,raises", [(0, False), (1, True)])
@patch("meersolar.meerpipeline.master_flow.get_run_context")
@patch("meersolar.meerpipeline.master_flow.start_log_task_saver")
@patch("meersolar.meerpipeline.master_flow.os.makedirs")
@patch("meersolar.meerpipeline.master_flow.os.path.exists")
@patch("meersolar.meerpipeline.master_flow.os.remove")
@patch("meersolar.meerpipeline.do_selfcal.main")
@patch("meersolar.meerpipeline.master_flow.get_dask_client")
def test_run_selfcal_jobs(
    mock_client,
    mock_main,
    mock_remove,
    mock_exists,
    mock_makedirs,
    mock_log_saver,
    mock_get_ctx,
    mock_msg,
    raises,
):
    # Mock returns
    mock_main.return_value = mock_msg
    mock_exists.return_value = True

    # Prefect task context
    mock_ctx = MagicMock()
    mock_ctx.task_run.id = "selfcal-id"
    mock_ctx.task_run.name = "selfcal"
    mock_get_ctx.return_value = mock_ctx

    # Thread
    mock_thread = MagicMock()
    mock_log_saver.return_value = mock_thread

    # Inputs
    kwargs = dict(
        mslist=["target1.ms", "target2.ms"],
        workdir="/mock/workdir",
        caldir="/mock/caldir",
        start_thresh=5.5,
        stop_thresh=3.2,
        max_iter=10,
        max_DR=500,
        min_iter=1,
        conv_frac=0.2,
        solint="30s",
        do_apcal=False,
        solar_selfcal=False,
        keep_backup=True,
        uvrange=">100lambda",
        minuv=50.0,
        weight="natural",
        robust=0.5,
        applymode="calflag",
        min_tol_factor=5.0,
        jobid=101,
        cpu_frac=0.6,
        mem_frac=0.7,
        remote_log=True,
    )

    if raises:
        with pytest.raises(RuntimeError, match="Self-calibration is failed."):
            run_selfcal_jobs.fn(**kwargs)
    else:
        result = run_selfcal_jobs.fn(**kwargs)
        assert result == 0

    expected_log = "/mock/workdir/logs/selfcal_targets.log"
    mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    mock_remove.assert_called_once_with(expected_log)
    mock_log_saver.assert_called_once()
    mock_thread.join.assert_called_once_with(timeout=5)

    mock_main.assert_called_once_with(
        "target1.ms,target2.ms",
        "/mock/workdir",
        "/mock/caldir",
        start_thresh=5.5,
        stop_thresh=3.2,
        max_iter=10.0,
        max_DR=500.0,
        min_iter=1.0,
        conv_frac=0.2,
        solint="30s",
        uvrange=">100lambda",
        minuv=50.0,
        weight="natural",
        robust=0.5,
        applymode="calflag",
        min_tol_factor=5.0,
        do_apcal=False,
        solar_selfcal=False,
        keep_backup=True,
        cpu_frac=0.6,
        mem_frac=0.7,
        jobid=101,
        start_remote_log=True,
        logfile="/mock/workdir/logs/selfcal_targets.log",
        dask_client=mock_client.return_value.__enter__.return_value,
    )


@pytest.mark.parametrize("mock_msg,raises", [(0, False), (1, True)])
@patch("meersolar.meerpipeline.master_flow.get_run_context")
@patch("meersolar.meerpipeline.master_flow.start_log_task_saver")
@patch("meersolar.meerpipeline.master_flow.os.makedirs")
@patch("meersolar.meerpipeline.master_flow.os.path.exists")
@patch("meersolar.meerpipeline.master_flow.os.remove")
@patch("meersolar.meerpipeline.do_apply_selfcal.main")
@patch("meersolar.meerpipeline.master_flow.get_dask_client")
def test_run_apply_selfcal_sol(
    mock_client,
    mock_main,
    mock_remove,
    mock_exists,
    mock_makedirs,
    mock_log_saver,
    mock_get_ctx,
    mock_msg,
    raises,
):
    # Mock return value for success/failure
    mock_main.return_value = mock_msg
    mock_exists.return_value = True

    # Mock Prefect context
    mock_ctx = MagicMock()
    mock_ctx.task_run.id = "mock-task-id"
    mock_ctx.task_run.name = "applying_self-calibration"
    mock_get_ctx.return_value = mock_ctx

    # Mock log thread
    mock_thread = MagicMock()
    mock_log_saver.return_value = mock_thread

    # Task arguments
    kwargs = dict(
        target_mslist=["t1.ms", "t2.ms"],
        workdir="/mock/workdir",
        caldir="/mock/caldir",
        overwrite_datacolumn=False,
        applymode="calflag",
        jobid=10,
        cpu_frac=0.6,
        mem_frac=0.7,
        remote_log=True,
    )

    if raises:
        with pytest.raises(
            RuntimeError, match="Applying self-calibration solutions is failed."
        ):
            run_apply_selfcal_sol.fn(**kwargs)
    else:
        result = run_apply_selfcal_sol.fn(**kwargs)
        assert result == 0

    expected_log = "/mock/workdir/logs/apply_selfcal.log"
    mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    mock_remove.assert_called_once_with(expected_log)
    mock_log_saver.assert_called_once()
    mock_thread.join.assert_called_once_with(timeout=5)

    # Check correct call to main
    mock_main.assert_called_once_with(
        "t1.ms,t2.ms",
        "/mock/workdir",
        "/mock/caldir",
        applymode="calflag",
        overwrite_datacolumn=False,
        start_remote_log=True,
        cpu_frac=0.6,
        mem_frac=0.7,
        logfile=expected_log,
        jobid=10,
        dask_client=mock_client.return_value.__enter__.return_value,
    )


@pytest.mark.parametrize("mock_msg,raises", [(0, False), (1, True)])
@patch("meersolar.meerpipeline.master_flow.get_run_context")
@patch("meersolar.meerpipeline.master_flow.start_log_task_saver")
@patch("meersolar.meerpipeline.master_flow.os.makedirs")
@patch("meersolar.meerpipeline.master_flow.os.path.exists")
@patch("meersolar.meerpipeline.master_flow.os.remove")
@patch("meersolar.meerpipeline.do_imaging.main")
@patch("meersolar.meerpipeline.master_flow.get_dask_client")
def test_run_imaging_jobs(
    mock_client,
    mock_main,
    mock_remove,
    mock_exists,
    mock_makedirs,
    mock_log_saver,
    mock_get_ctx,
    mock_msg,
    raises,
):
    # Setup mock behavior
    mock_main.return_value = mock_msg
    mock_exists.return_value = True

    # Prefect context mock
    mock_ctx = MagicMock()
    mock_ctx.task_run.id = "imaging-id"
    mock_ctx.task_run.name = "imaging"
    mock_get_ctx.return_value = mock_ctx

    # Log thread mock
    mock_thread = MagicMock()
    mock_log_saver.return_value = mock_thread

    # Arguments
    kwargs = dict(
        mslist=["target1.ms", "target2.ms"],
        workdir="/mock/workdir",
        outdir="/mock/outdir",
        freqrange="100~200",
        timerange="2023/01/01/00:00:00~2023/01/01/01:00:00",
        minuv=100.0,
        weight="uniform",
        robust=-0.5,
        pol="I",
        freqres=1.0,
        timeres=5.0,
        band="L",
        threshold=0.5,
        use_multiscale=False,
        use_solar_mask=False,
        cutout_rsun=2.0,
        make_overlay=False,
        savemodel=True,
        saveres=True,
        jobid=202,
        cpu_frac=0.6,
        mem_frac=0.7,
        remote_log=True,
    )

    if raises:
        with pytest.raises(RuntimeError, match="Imaging is failed."):
            run_imaging_jobs.fn(**kwargs)
    else:
        result = run_imaging_jobs.fn(**kwargs)
        assert result == 0

    expected_log = "/mock/workdir/logs/imaging_targets.log"
    mock_remove.assert_called_once_with(expected_log)
    mock_log_saver.assert_called_once()
    mock_thread.join.assert_called_once_with(timeout=5)

    mock_main.assert_called_once_with(
        "target1.ms,target2.ms",
        "/mock/workdir",
        "/mock/outdir",
        freqrange="100~200",
        timerange="2023/01/01/00:00:00~2023/01/01/01:00:00",
        pol="I",
        freqres=1.0,
        timeres=5.0,
        weight="uniform",
        robust=-0.5,
        minuv=100.0,
        threshold=0.5,
        band="L",
        cutout_rsun=2.0,
        use_multiscale=False,
        use_solar_mask=False,
        savemodel=True,
        saveres=True,
        make_overlay=False,
        start_remote_log=True,
        cpu_frac=0.6,
        mem_frac=0.7,
        jobid=202,
        logfile="/mock/workdir/logs/imaging_targets.log",
        dask_client=mock_client.return_value.__enter__.return_value,
    )


@pytest.mark.parametrize("mock_msg,raises", [(0, False), (1, True)])
@patch("meersolar.meerpipeline.master_flow.get_run_context")
@patch("meersolar.meerpipeline.master_flow.start_log_task_saver")
@patch("meersolar.meerpipeline.master_flow.os.makedirs")
@patch("meersolar.meerpipeline.master_flow.os.path.exists")
@patch("meersolar.meerpipeline.master_flow.os.remove")
@patch("meersolar.meerpipeline.meer_pbcor.main")
@patch("meersolar.meerpipeline.master_flow.get_dask_client")
def test_run_apply_pbcor(
    mock_client,
    mock_main,
    mock_remove,
    mock_exists,
    mock_makedirs,
    mock_log_saver,
    mock_get_ctx,
    mock_msg,
    raises,
):
    # Mock return value
    mock_main.return_value = mock_msg
    mock_exists.return_value = True

    # Prefect context mock
    mock_ctx = MagicMock()
    mock_ctx.task_run.id = "mock-task-id"
    mock_ctx.task_run.name = "applying_primary_beam"
    mock_get_ctx.return_value = mock_ctx

    # Log thread mock
    mock_thread = MagicMock()
    mock_log_saver.return_value = mock_thread

    # Task arguments
    kwargs = dict(
        imagedir="/mock/images",
        workdir="/mock/workdir",
        apply_parang=False,
        jobid=5,
        cpu_frac=0.6,
        mem_frac=0.7,
        remote_log=True,
    )

    if raises:
        with pytest.raises(RuntimeError, match="Primary beam correction is failed."):
            run_apply_pbcor.fn(**kwargs)
    else:
        result = run_apply_pbcor.fn(**kwargs)
        assert result == 0

    expected_log = "/mock/workdir/logs/apply_pbcor.log"
    mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    mock_remove.assert_called_once_with(expected_log)
    mock_log_saver.assert_called_once()
    mock_thread.join.assert_called_once_with(timeout=5)

    mock_main.assert_called_once_with(
        "/mock/images",
        workdir="/mock/workdir",
        apply_parang=False,
        cpu_frac=0.6,
        mem_frac=0.7,
        logfile=expected_log,
        jobid=5,
        start_remote_log=True,
        dask_client=mock_client.return_value.__enter__.return_value,
    )
