import pytest
from unittest.mock import patch, MagicMock
from meersolar.meerpipeline.master_controller import *


@pytest.mark.parametrize(
    "flag_calibrators, success_index, expected_return",
    [
        (True, 0, 0),  # Calibrator flagging, successful
        (True, 1, 1),  # Calibrator flagging, failed
        (False, 0, 0),  # Target flagging, successful
        (False, 1, 1),  # Target flagging, failed
    ],
)
@patch("meersolar.meerpipeline.master_controller.glob.glob")
@patch("meersolar.meerpipeline.master_controller.time.sleep")
@patch("meersolar.meerpipeline.master_controller.os.system")
@patch("meersolar.meerpipeline.master_controller.os.makedirs")
@patch("meersolar.meerpipeline.master_controller.create_batch_script_nonhpc")
def test_run_flag(
    mock_create_batch_script,
    mock_makedirs,
    mock_system,
    mock_sleep,
    mock_glob,
    flag_calibrators,
    success_index,
    expected_return,
):
    # Setup mock return values
    mock_create_batch_script.return_value = ("/mock/script.sh", "/mock/script.log")
    mock_glob.return_value = [
        f"/mock/workdir/.Finished_flagging_cal_test_{success_index}"
    ]
    result = run_flag(
        msname="test.ms",
        workdir="/mock/workdir",
        flag_calibrators=flag_calibrators,
        jobid=123,
        cpu_frac=0.5,
        mem_frac=0.5,
        remote_log=False,
    )

    assert result == expected_return
    mock_create_batch_script.assert_called_once()
    mock_system.assert_called_once_with("bash /mock/script.sh")
    mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    mock_glob.assert_called()


@pytest.mark.parametrize("success_index, expected_return", [(0, 0), (1, 1)])
@patch("meersolar.meerpipeline.master_controller.glob.glob")
@patch("meersolar.meerpipeline.master_controller.time.sleep")
@patch("meersolar.meerpipeline.master_controller.os.system")
@patch("meersolar.meerpipeline.master_controller.os.makedirs")
@patch("meersolar.meerpipeline.master_controller.create_batch_script_nonhpc")
def test_run_import_model(
    mock_create_batch_script,
    mock_makedirs,
    mock_system,
    mock_sleep,
    mock_glob,
    success_index,
    expected_return,
):
    # Mocked return values
    mock_create_batch_script.return_value = ("/mock/script.sh", "/mock/script.log")
    mock_glob.return_value = [f"/mock/workdir/.Finished_modeling_test_{success_index}"]
    result = run_import_model(
        msname="test.ms",
        workdir="/mock/workdir",
        jobid=42,
        cpu_frac=0.5,
        mem_frac=0.5,
        remote_log=False,
    )

    assert result == expected_return
    mock_create_batch_script.assert_called_once()
    mock_system.assert_called_once_with("bash /mock/script.sh")
    mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    mock_glob.assert_called()


@pytest.mark.parametrize(
    "success_index, perform_polcal, keep_backup, expected_return",
    [
        (0, False, False, 0),
        (1, False, False, 1),
        (0, True, False, 0),
        (0, False, True, 0),
        (1, True, True, 1),
    ],
)
@patch("meersolar.meerpipeline.master_controller.glob.glob")
@patch("meersolar.meerpipeline.master_controller.time.sleep")
@patch("meersolar.meerpipeline.master_controller.os.system")
@patch("meersolar.meerpipeline.master_controller.os.makedirs")
@patch("meersolar.meerpipeline.master_controller.create_batch_script_nonhpc")
def test_run_basic_cal_jobs(
    mock_create_batch_script,
    mock_makedirs,
    mock_system,
    mock_sleep,
    mock_glob,
    success_index,
    perform_polcal,
    keep_backup,
    expected_return,
):
    # Setup
    mock_create_batch_script.return_value = ("/mock/script.sh", "/mock/script.log")
    mock_glob.return_value = [f"/mock/workdir/.Finished_basic_cal_{success_index}"]
    result = run_basic_cal_jobs(
        msname="test.ms",
        workdir="/mock/workdir",
        caldir="/mock/caldir",
        perform_polcal=perform_polcal,
        keep_backup=keep_backup,
        jobid=99,
        cpu_frac=0.7,
        mem_frac=0.6,
        remote_log=False,
    )

    assert result == expected_return
    mock_create_batch_script.assert_called_once()
    mock_system.assert_called_once_with("bash /mock/script.sh")
    mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    mock_glob.assert_called()


@pytest.mark.parametrize(
    "success_index, should_raise, expected_return",
    [
        (0, False, 0),  # Success case
        (1, False, 1),  # Failure case
        (None, True, 1),  # Exception handling
    ],
)
@patch("meersolar.meerpipeline.master_controller.glob.glob")
@patch("meersolar.meerpipeline.master_controller.time.sleep")
@patch("meersolar.meerpipeline.master_controller.os.system")
@patch("meersolar.meerpipeline.master_controller.os.makedirs")
@patch("meersolar.meerpipeline.master_controller.create_batch_script_nonhpc")
@patch("meersolar.meerpipeline.master_controller.traceback.print_exc")
def test_run_noise_diode_cal(
    mock_print_exc,
    mock_create_batch_script,
    mock_makedirs,
    mock_system,
    mock_sleep,
    mock_glob,
    success_index,
    should_raise,
    expected_return,
):
    mock_create_batch_script.return_value = ("/mock/script.sh", "/mock/script.log")

    if should_raise:
        mock_glob.side_effect = Exception("Test exception")
    else:
        mock_glob.return_value = [
            f"/mock/workdir/.Finished_run_meersolar_noise_cal_{success_index}"
        ]
    result = run_noise_diode_cal(
        msname="test.ms",
        workdir="/mock/workdir",
        caldir="/mock/caldir",
        keep_backup=True,
        jobid=5,
        cpu_frac=0.9,
        mem_frac=0.9,
        remote_log=True,
    )

    assert result == expected_return
    mock_create_batch_script.assert_called_once()
    mock_system.assert_called_once_with("bash /mock/script.sh")
    mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    if should_raise:
        mock_print_exc.assert_called_once()


@pytest.mark.parametrize(
    "split_fullpol, ms_exists, expected_return",
    [
        (False, True, 0),
        (False, False, 1),
        (True, True, 0),
        (True, False, 1),
    ],
)
@patch("meersolar.meerpipeline.master_controller.glob.glob")
@patch("meersolar.meerpipeline.master_controller.time.sleep")
@patch("meersolar.meerpipeline.master_controller.os.system")
@patch("meersolar.meerpipeline.master_controller.os.makedirs")
@patch("meersolar.meerpipeline.master_controller.os.path.exists")
@patch("meersolar.meerpipeline.master_controller.create_batch_script_nonhpc")
@patch("meersolar.meerpipeline.master_controller.determine_noise_diode_cal_scan")
@patch("meersolar.meerpipeline.master_controller.get_cal_target_scans")
@patch("meersolar.meerpipeline.master_controller.msmetadata")
def test_run_partition(
    mock_msmetadata,
    mock_get_scans,
    mock_determine_noise,
    mock_create_script,
    mock_exists,
    mock_makedirs,
    mock_system,
    mock_sleep,
    mock_glob,
    split_fullpol,
    ms_exists,
    expected_return,
):
    # Setup mocks
    mock_msmd = MagicMock()
    mock_msmd.nchan.return_value = 2048
    mock_msmd.timesforfield.return_value = [0.0, 10.0]
    mock_msmd.exposuretime.return_value = {"value": 4.0}
    mock_msmetadata.return_value = mock_msmd

    mock_get_scans.return_value = ([10], [20, 21], [], [], [])
    mock_determine_noise.return_value = False
    mock_create_script.return_value = ("/mock/script.sh", "/mock/script.log")
    mock_exists.return_value = ms_exists
    mock_glob.return_value = ["/mock/.Finished_partition_cal_0"]

    result = run_partition(
        msname="test.ms",
        workdir="/mock/workdir",
        split_fullpol=split_fullpol,
        jobid=1,
        cpu_frac=0.9,
        mem_frac=0.9,
        remote_log=True,
    )

    assert result == expected_return
    mock_create_script.assert_called_once()
    mock_system.assert_called_once_with("bash /mock/script.sh")
    mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    mock_msmd.open.assert_called_once_with("test.ms")
    mock_msmd.close.assert_called_once()
    mock_get_scans.assert_called_once()


@pytest.mark.parametrize(
    "split_fullpol, merge_spws, spw, target_scans, should_raise, expected_return",
    [
        (False, False, "", [], False, 0),
        (True, False, "", [], False, 0),
        (False, True, "", [], False, 0),
        (False, False, "0:10~20", [], False, 0),
        (False, False, "", [5, 6], False, 0),
        (False, False, "", [], True, 1),
    ],
)
@patch("meersolar.meerpipeline.master_controller.traceback.print_exc")
@patch("meersolar.meerpipeline.master_controller.create_batch_script_nonhpc")
@patch("meersolar.meerpipeline.master_controller.os.makedirs")
@patch("meersolar.meerpipeline.master_controller.os.system")
def test_run_target_split_jobs(
    mock_system,
    mock_makedirs,
    mock_create_batch_script,
    mock_print_exc,
    split_fullpol,
    merge_spws,
    spw,
    target_scans,
    should_raise,
    expected_return,
):
    mock_create_batch_script.return_value = ("/mock/script.sh", "/mock/script.log")
    if should_raise:
        mock_os = mock_system.side_effect = Exception("Test error")

    result = run_target_split_jobs(
        msname="test.ms",
        workdir="/mock/workdir",
        datacolumn="CORRECTED_DATA",
        spw=spw,
        timeres=10.0,
        freqres=1.0,
        target_freq_chunk=5.0,
        n_spectral_chunk=2,
        target_scans=target_scans,
        prefix="targets",
        split_fullpol=split_fullpol,
        merge_spws=merge_spws,
        time_window=5,
        time_interval=2,
        cpu_frac=0.6,
        mem_frac=0.7,
        max_cpu_frac=0.9,
        max_mem_frac=0.95,
        jobid=123,
        remote_log=True,
    )

    assert result == expected_return
    if not should_raise:
        mock_create_batch_script.assert_called_once()
        mock_system.assert_called_once_with("bash /mock/script.sh")
        mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
    else:
        mock_print_exc.assert_called_once()


@pytest.mark.parametrize(
    "success_index, raise_exception, expected_return",
    [
        (0, False, 0),  # Success
        (1, False, 1),  # Failure
        (None, True, 1),  # Exception
    ],
)
@patch("meersolar.meerpipeline.master_controller.traceback.print_exc")
@patch("meersolar.meerpipeline.master_controller.create_batch_script_nonhpc")
@patch("meersolar.meerpipeline.master_controller.os.makedirs")
@patch("meersolar.meerpipeline.master_controller.os.system")
@patch("meersolar.meerpipeline.master_controller.glob.glob")
@patch("meersolar.meerpipeline.master_controller.time.sleep")
def test_run_solar_siderealcor_jobs(
    mock_sleep,
    mock_glob,
    mock_system,
    mock_makedirs,
    mock_create_script,
    mock_print_exc,
    success_index,
    raise_exception,
    expected_return,
):
    mock_create_script.return_value = ("/mock/script.sh", "/mock/script.log")
    if raise_exception:
        mock_glob.side_effect = Exception("mocked error")
    else:
        mock_glob.return_value = [
            f"/mock/.Finished_cor_sidereal_targets_{success_index}"
        ]

    result = run_solar_siderealcor_jobs(
        mslist=["ms1.ms", "ms2.ms"],
        workdir="/mock/workdir",
        prefix="targets",
        jobid=5,
        cpu_frac=0.8,
        mem_frac=0.8,
        max_cpu_frac=0.9,
        max_mem_frac=0.9,
        remote_log=True,
    )

    assert result == expected_return

    if raise_exception:
        mock_print_exc.assert_called_once()
    else:
        mock_create_script.assert_called_once()
        mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
        mock_system.assert_called_once_with("bash /mock/script.sh")
        mock_glob.assert_called()


@pytest.mark.parametrize(
    "apply_parang, success_index, raise_exception, expected_return",
    [
        (True, 0, False, 0),  # success, with parang
        (True, 1, False, 1),  # failure, with parang
        (False, 0, False, 0),  # success, no parang
        (False, 1, False, 1),  # failure, no parang
        (True, None, True, 1),  # exception raised
    ],
)
@patch("meersolar.meerpipeline.master_controller.traceback.print_exc")
@patch("meersolar.meerpipeline.master_controller.create_batch_script_nonhpc")
@patch("meersolar.meerpipeline.master_controller.os.makedirs")
@patch("meersolar.meerpipeline.master_controller.os.system")
@patch("meersolar.meerpipeline.master_controller.glob.glob")
@patch("meersolar.meerpipeline.master_controller.time.sleep")
def test_run_apply_pbcor(
    mock_sleep,
    mock_glob,
    mock_system,
    mock_makedirs,
    mock_create_batch_script,
    mock_print_exc,
    apply_parang,
    success_index,
    raise_exception,
    expected_return,
):
    mock_create_batch_script.return_value = ("/mock/script.sh", "/mock/script.log")
    if raise_exception:
        mock_glob.side_effect = Exception("Simulated failure")
    else:
        mock_glob.return_value = [
            f"/mock/workdir/.Finished_apply_pbcor_{success_index}"
        ]

    result = run_apply_pbcor(
        imagedir="/mock/images",
        workdir="/mock/workdir",
        apply_parang=apply_parang,
        jobid=77,
        cpu_frac=0.8,
        mem_frac=0.9,
        remote_log=True,
    )

    assert result == expected_return

    if raise_exception:
        mock_print_exc.assert_called_once()
    else:
        mock_create_batch_script.assert_called_once()
        mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
        mock_system.assert_called_once_with("bash /mock/script.sh")
        mock_glob.assert_called_once()


@pytest.mark.parametrize(
    "use_only_bandpass, overwrite_datacolumn, success_index, raise_exc, expected_return",
    [
        (False, True, 0, False, 0),
        (True, False, 1, False, 1),
        (True, True, 0, False, 0),
        (False, False, 1, False, 1),
        (True, True, None, True, 1),
    ],
)
@patch("meersolar.meerpipeline.master_controller.traceback.print_exc")
@patch("meersolar.meerpipeline.master_controller.create_batch_script_nonhpc")
@patch("meersolar.meerpipeline.master_controller.os.makedirs")
@patch("meersolar.meerpipeline.master_controller.os.system")
@patch("meersolar.meerpipeline.master_controller.glob.glob")
@patch("meersolar.meerpipeline.master_controller.time.sleep")
def test_run_apply_basiccal_sol(
    mock_sleep,
    mock_glob,
    mock_system,
    mock_makedirs,
    mock_create_script,
    mock_print_exc,
    use_only_bandpass,
    overwrite_datacolumn,
    success_index,
    raise_exc,
    expected_return,
):
    mock_create_script.return_value = ("/mock/script.sh", "/mock/script.log")
    if raise_exc:
        mock_glob.side_effect = Exception("simulated failure")
    else:
        mock_glob.return_value = [
            f"/mock/workdir/.Finished_apply_basiccal_{success_index}"
        ]

    result = run_apply_basiccal_sol(
        target_mslist=["target1.ms", "target2.ms"],
        workdir="/mock/workdir",
        caldir="/mock/caldir",
        use_only_bandpass=use_only_bandpass,
        overwrite_datacolumn=overwrite_datacolumn,
        applymode="calflag",
        jobid=101,
        cpu_frac=0.7,
        mem_frac=0.6,
        remote_log=True,
    )

    assert result == expected_return

    if raise_exc:
        mock_print_exc.assert_called_once()
    else:
        mock_create_script.assert_called_once()
        mock_system.assert_called_once_with("bash /mock/script.sh")
        mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
        mock_glob.assert_called_once()

        # Optional: check presence of flags in the constructed command
        cmd = mock_create_script.call_args[0][0]
        assert ("--use_only_bandpass" in cmd) == use_only_bandpass
        assert ("--overwrite_datacolumn" in cmd) == overwrite_datacolumn


@pytest.mark.parametrize(
    "overwrite_datacolumn, success_index, raise_exc, expected_return",
    [
        (True, 0, False, 0),  # success with overwrite
        (False, 1, False, 1),  # fail without overwrite
        (True, 1, False, 1),  # fail with overwrite
        (False, 0, False, 0),  # success without overwrite
        (True, None, True, 1),  # exception raised
    ],
)
@patch("meersolar.meerpipeline.master_controller.traceback.print_exc")
@patch("meersolar.meerpipeline.master_controller.create_batch_script_nonhpc")
@patch("meersolar.meerpipeline.master_controller.os.makedirs")
@patch("meersolar.meerpipeline.master_controller.os.system")
@patch("meersolar.meerpipeline.master_controller.glob.glob")
@patch("meersolar.meerpipeline.master_controller.time.sleep")
def test_run_apply_selfcal_sol(
    mock_sleep,
    mock_glob,
    mock_system,
    mock_makedirs,
    mock_create_script,
    mock_print_exc,
    overwrite_datacolumn,
    success_index,
    raise_exc,
    expected_return,
):
    mock_create_script.return_value = ("/mock/script.sh", "/mock/script.log")
    if raise_exc:
        mock_glob.side_effect = Exception("simulated failure")
    else:
        mock_glob.return_value = [
            f"/mock/workdir/.Finished_apply_selfcal_{success_index}"
        ]

    result = run_apply_selfcal_sol(
        target_mslist=["targetA.ms", "targetB.ms"],
        workdir="/mock/workdir",
        caldir="/mock/caldir",
        overwrite_datacolumn=overwrite_datacolumn,
        applymode="calflag",
        jobid=202,
        cpu_frac=0.9,
        mem_frac=0.95,
        remote_log=True,
    )

    assert result == expected_return

    if raise_exc:
        mock_print_exc.assert_called_once()
    else:
        mock_create_script.assert_called_once()
        mock_system.assert_called_once_with("bash /mock/script.sh")
        mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
        mock_glob.assert_called_once()

        # Optional: Validate presence/absence of flag in command
        cmd = mock_create_script.call_args[0][0]
        assert ("--overwrite_datacolumn" in cmd) == overwrite_datacolumn


@pytest.mark.parametrize(
    "do_apcal, solar_selfcal, keep_backup, success_index, raise_exc, expected_return",
    [
        (True, True, False, 0, False, 0),
        (False, True, False, 0, False, 0),
        (True, False, True, 0, False, 0),
        (False, False, False, 1, False, 1),
        (True, True, True, None, True, 1),
    ],
)
@patch("meersolar.meerpipeline.master_controller.traceback.print_exc")
@patch("meersolar.meerpipeline.master_controller.create_batch_script_nonhpc")
@patch("meersolar.meerpipeline.master_controller.os.makedirs")
@patch("meersolar.meerpipeline.master_controller.os.system")
@patch("meersolar.meerpipeline.master_controller.glob.glob")
@patch("meersolar.meerpipeline.master_controller.time.sleep")
def test_run_selfcal_jobs(
    mock_sleep,
    mock_glob,
    mock_system,
    mock_makedirs,
    mock_create_script,
    mock_print_exc,
    do_apcal,
    solar_selfcal,
    keep_backup,
    success_index,
    raise_exc,
    expected_return,
):
    mock_create_script.return_value = ("/mock/script.sh", "/mock/script")
    if raise_exc:
        mock_glob.side_effect = Exception("simulated failure")
    else:
        mock_glob.return_value = [
            f"/mock/workdir/.Finished_selfcal_targets_{success_index}"
        ]

    result = run_selfcal_jobs(
        mslist=["target1.ms", "target2.ms"],
        workdir="/mock/workdir",
        caldir="/mock/caldir",
        start_thresh=5.0,
        stop_thresh=3.0,
        max_iter=10,
        max_DR=200,
        min_iter=2,
        conv_frac=0.25,
        solint="30s",
        do_apcal=do_apcal,
        solar_selfcal=solar_selfcal,
        keep_backup=keep_backup,
        uvrange=">100",
        minuv=200.0,
        weight="briggs",
        robust=0.5,
        applymode="calonly",
        min_tol_factor=5.0,
        jobid=500,
        cpu_frac=0.8,
        mem_frac=0.85,
        remote_log=True,
    )

    assert result == expected_return

    if raise_exc:
        mock_print_exc.assert_called_once()
    else:
        mock_create_script.assert_called_once()
        mock_system.assert_called_once_with("bash /mock/script.sh")
        mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
        mock_glob.assert_called_once()

        cmd = mock_create_script.call_args[0][0]
        assert ("--no_apcal" in cmd) == (not do_apcal)
        assert ("--no_solar_selfcal" in cmd) == (not solar_selfcal)
        assert ("--keep_backup" in cmd) == keep_backup


@pytest.mark.parametrize(
    "use_multiscale, use_solar_mask, make_overlay, savemodel, saveres, success_index, raise_exc, expected_return",
    [
        (True, True, True, True, True, 0, False, 0),  # All True, success
        (False, False, False, False, False, 1, False, 1),  # All False, failure
        (True, False, True, False, True, 0, False, 0),  # Mixed, success
        (True, True, True, True, True, None, True, 1),  # Exception
    ],
)
@patch("meersolar.meerpipeline.master_controller.traceback.print_exc")
@patch("meersolar.meerpipeline.master_controller.create_batch_script_nonhpc")
@patch("meersolar.meerpipeline.master_controller.os.makedirs")
@patch("meersolar.meerpipeline.master_controller.os.system")
@patch("meersolar.meerpipeline.master_controller.glob.glob")
@patch("meersolar.meerpipeline.master_controller.time.sleep")
def test_run_imaging_jobs(
    mock_sleep,
    mock_glob,
    mock_system,
    mock_makedirs,
    mock_create_script,
    mock_print_exc,
    use_multiscale,
    use_solar_mask,
    make_overlay,
    savemodel,
    saveres,
    success_index,
    raise_exc,
    expected_return,
):
    mock_create_script.return_value = ("/mock/script.sh", "/mock/script")
    if raise_exc:
        mock_glob.side_effect = Exception("simulated failure")
    else:
        mock_glob.return_value = [
            f"/mock/workdir/.Finished_imaging_targets_{success_index}"
        ]

    result = run_imaging_jobs(
        mslist=["t1.ms", "t2.ms"],
        workdir="/mock/workdir",
        outdir="/mock/outdir",
        freqrange="1100~1200",
        timerange="2025/01/01/00:00:00~2025/01/01/01:00:00",
        minuv=200,
        weight="briggs",
        robust=0.5,
        pol="I",
        freqres=1.0,
        timeres=10.0,
        band="L",
        threshold=2.0,
        use_multiscale=use_multiscale,
        use_solar_mask=use_solar_mask,
        make_overlay=make_overlay,
        savemodel=savemodel,
        saveres=saveres,
        jobid=10,
        cpu_frac=0.75,
        mem_frac=0.85,
        remote_log=True,
    )

    assert result == expected_return

    if raise_exc:
        mock_print_exc.assert_called_once()
    else:
        mock_create_script.assert_called_once()
        mock_system.assert_called_once_with("bash /mock/script.sh")
        mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
        mock_glob.assert_called_once()

        # Optional: inspect flags in command string
        cmd = mock_create_script.call_args[0][0]
        assert ("--no_multiscale" in cmd) == (not use_multiscale)
        assert ("--no_solar_mask" in cmd) == (not use_solar_mask)
        assert ("--no_make_overlay" in cmd) == (not make_overlay)
        assert ("--no_savemodel" in cmd) == (not savemodel)
        assert ("--no_saveres" in cmd) == (not saveres)
        assert "--freqrange 1100~1200" in cmd
        assert "--timerange 2025/01/01/00:00:00~2025/01/01/01:00:00" in cmd
        assert "--band L" in cmd


@pytest.mark.parametrize(
    "target_scans, success_index, raise_exc, expected_return",
    [
        ([1, 2, 3], 0, False, 0),  # Success
        ([5, 6], 1, False, 1),  # Failure
        ([], 0, False, 0),  # Empty scan list, success
        ([7], None, True, 1),  # Exception
    ],
)
@patch("meersolar.meerpipeline.master_controller.traceback.print_exc")
@patch("meersolar.meerpipeline.master_controller.create_batch_script_nonhpc")
@patch("meersolar.meerpipeline.master_controller.os.makedirs")
@patch("meersolar.meerpipeline.master_controller.os.system")
@patch("meersolar.meerpipeline.master_controller.glob.glob")
@patch("meersolar.meerpipeline.master_controller.time.sleep")
def test_run_ds_jobs(
    mock_sleep,
    mock_glob,
    mock_system,
    mock_makedirs,
    mock_create_script,
    mock_print_exc,
    target_scans,
    success_index,
    raise_exc,
    expected_return,
):
    mock_create_script.return_value = ("/mock/script.sh", "/mock/script")
    if raise_exc:
        mock_glob.side_effect = Exception("simulated error")
    else:
        mock_glob.return_value = [f"/mock/workdir/.Finished_ds_targets_{success_index}"]

    result = run_ds_jobs(
        msname="test.ms",
        workdir="/mock/workdir",
        outdir="/mock/outdir",
        target_scans=target_scans,
        jobid=42,
        cpu_frac=0.6,
        mem_frac=0.75,
        remote_log=True,
    )

    assert result == expected_return

    if raise_exc:
        mock_print_exc.assert_called_once()
    else:
        mock_create_script.assert_called_once()
        mock_system.assert_called_once_with("bash /mock/script.sh")
        mock_makedirs.assert_called_once_with("/mock/workdir/logs", exist_ok=True)
        mock_glob.assert_called_once()

        # Optional: check if scan list is included in command
        cmd = mock_create_script.call_args[0][0]
        joined_scans = " ".join(str(s) for s in target_scans)
        assert f"--target_scans {joined_scans}" in cmd


@pytest.mark.parametrize(
    "mock_glob_result, expected_return",
    [
        (["/mock/workdir/.Finished_test_0"], True),  # Success
        (["/mock/workdir/.Finished_test_1"], False),  # Failure
        ([], False),  # Not finished
    ],
)
@patch("meersolar.meerpipeline.master_controller.glob.glob")
def test_check_status(mock_glob, mock_glob_result, expected_return):
    mock_glob.return_value = mock_glob_result

    result = check_status("/mock/workdir", "test")
    assert result == expected_return
    mock_glob.assert_called_once_with("/mock/workdir/.Finished_test*")


@pytest.mark.parametrize(
    "remote_link, waittime, should_spawn, expected_pid_type",
    [
        ("http://remote", 0.5, True, int),  # valid → returns PID
        ("", 0.5, False, type(None)),  # empty remote_link
        ("http://remote", 0.0, False, type(None)),  # zero waittime
    ],
)
@patch("meersolar.meerpipeline.master_controller.Process")
@patch("threading.Thread")
@patch("meersolar.meerpipeline.master_controller.Event")
@patch("meersolar.meerpipeline.master_controller.ping_logger")
def test_start_ping_logger(
    mock_ping_logger,
    mock_Event,
    mock_Thread,
    mock_Process,
    remote_link,
    waittime,
    should_spawn,
    expected_pid_type,
):
    mock_event = MagicMock()
    mock_Event.return_value = mock_event

    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_Process.return_value = mock_proc

    pid = start_ping_logger(
        jobid=111, remote_jobid="abc123", waittime=waittime, remote_link=remote_link
    )

    if should_spawn:
        mock_Process.assert_called_once()
        mock_proc.start.assert_called_once()
        mock_Thread.assert_called_once()
        assert isinstance(pid, expected_pid_type)
    else:
        mock_Process.assert_not_called()
        assert pid is None


@patch("meersolar.meerpipeline.master_controller.time.sleep")
@patch("meersolar.meerpipeline.master_controller.os.system")
@patch("meersolar.meerpipeline.master_controller.os.path.exists")
def test_exit_job(mock_exists, mock_system, mock_sleep, capsys):
    # Simulate start time 2 seconds ago
    start_time = time.time() - 2.0

    # Simulate existing scratch directories
    def exists_side_effect(path):
        return "dask-scratch-space" in path

    mock_exists.side_effect = exists_side_effect

    exit_job(start_time, mspath="/mock/ms", workdir="/mock/work")

    # Assert correct cleanup commands were issued
    mock_system.assert_any_call("rm -rf /mock/ms/dask-scratch-space /mock/ms/tmp")
    mock_system.assert_any_call("rm -rf /mock/work/dask-scratch-space /mock/work/tmp")
    assert mock_system.call_count == 2

    mock_sleep.assert_called_once_with(10)

    # Check printed time is close to 2 seconds
    captured = capsys.readouterr()
    assert "Total time taken:" in captured.out
    assert "s." in captured.out


@pytest.mark.parametrize(
    "kwargs, expected_return",
    [
        (
            {
                "msname": "/mock/ms/test.ms",
                "workdir": "/mock/workdir",
                "outdir": "/mock/outdir",
                "solar_data": True,
                "do_basic_cal": False,
                "do_noise_cal": False,
                "do_applycal": False,
                "do_selfcal": False,
                "do_imaging": False,
                "do_pbcor": False,
                "make_ds": False,
            },
            0,
        ),
    ],
)
@patch("casatools.table")
@patch("meersolar.meerpipeline.master_controller.run_flag", return_value=0)
@patch("meersolar.meerpipeline.master_controller.run_import_model", return_value=0)
@patch("meersolar.meerpipeline.master_controller.run_basic_cal_jobs", return_value=0)
@patch("meersolar.meerpipeline.master_controller.run_noise_diode_cal", return_value=0)
@patch("meersolar.meerpipeline.master_controller.run_partition", return_value=0)
@patch("meersolar.meerpipeline.master_controller.run_target_split_jobs", return_value=0)
@patch(
    "meersolar.meerpipeline.master_controller.run_solar_siderealcor_jobs",
    return_value=0,
)
@patch("meersolar.meerpipeline.master_controller.run_apply_pbcor", return_value=0)
@patch(
    "meersolar.meerpipeline.master_controller.run_apply_basiccal_sol", return_value=0
)
@patch("meersolar.meerpipeline.master_controller.run_apply_selfcal_sol", return_value=0)
@patch("meersolar.meerpipeline.master_controller.run_selfcal_jobs", return_value=0)
@patch("meersolar.meerpipeline.master_controller.run_imaging_jobs", return_value=0)
@patch("meersolar.meerpipeline.master_controller.run_ds_jobs", return_value=0)
@patch("meersolar.meerpipeline.master_controller.os.chdir")
@patch("meersolar.meerpipeline.master_controller.os.path.exists", return_value=True)
@patch("meersolar.meerpipeline.master_controller.os.makedirs")
@patch("meersolar.meerpipeline.master_controller.glob.glob")
@patch("meersolar.meerpipeline.master_controller.psutil.cpu_percent", return_value=10.0)
@patch("meersolar.meerpipeline.master_controller.psutil.cpu_count", return_value=4)
@patch("meersolar.meerpipeline.master_controller.init_meersolar_data")
@patch("meersolar.meerpipeline.master_controller.msmetadata")
@patch(
    "meersolar.meerpipeline.master_controller.calc_bw_smearing_freqwidth",
    return_value=1.0,
)
@patch(
    "meersolar.meerpipeline.master_controller.calc_time_smearing_timewidth",
    return_value=10.0,
)
@patch(
    "meersolar.meerpipeline.master_controller.max_time_solar_smearing", return_value=5.0
)
@patch("meersolar.meerpipeline.master_controller.get_bad_chans", return_value="")
@patch("meersolar.meerpipeline.master_controller.reset_weights_and_flags")
@patch("meersolar.meerpipeline.master_controller.get_jobid", return_value=1234)
@patch(
    "meersolar.meerpipeline.master_controller.save_main_process_info",
    return_value="/mock/job/file",
)
@patch("meersolar.meerpipeline.master_controller.get_emails", return_value="")
@patch(
    "meersolar.meerpipeline.master_controller.generate_password",
    return_value="mock_password",
)
@patch(
    "meersolar.meerpipeline.master_controller.get_remote_logger_link", return_value=""
)
@patch("meersolar.meerpipeline.master_controller.drop_cache")
@patch("meersolar.meerpipeline.master_controller.os.system")
@patch("meersolar.meerpipeline.master_controller.time.sleep")
@patch("meersolar.meerpipeline.master_controller.os.getpid", return_value=1111)
@patch(
    "meersolar.meerpipeline.master_controller.check_datacolumn_valid", return_value=True
)
def test_master_control(
    mock_valid_datacolumn,
    mock_getpid,
    mock_sleep,
    mock_system,
    mock_drop_cache,
    mock_get_remote_logger_link,
    mock_generate_password,
    mock_get_emails,
    mock_save_main_process_info,
    mock_get_jobid,
    mock_reset_weights_and_flags,
    mock_get_bad_chans,
    mock_max_time_solar_smearing,
    mock_calc_time_smearing,
    mock_calc_bw_smearing,
    mock_msmetadata,
    mock_init_data,
    mock_cpu_count,
    mock_cpu_percent,
    mock_glob,
    mock_makedirs,
    mock_exists,
    mock_chdir,
    mock_run_ds_jobs,
    mock_run_imaging_jobs,
    mock_run_selfcal_jobs,
    mock_run_apply_selfcal_sol,
    mock_run_apply_basiccal_sol,
    mock_run_apply_pbcor,
    mock_run_siderealcor,
    mock_run_target_split,
    mock_run_partition,
    mock_run_noise_cal,
    mock_run_basic_cal,
    mock_run_import_model,
    mock_run_flag,
    mock_table,
    kwargs,
    expected_return,
):
    def glob_side_effect(pattern):
        if "/*.bcal" in pattern:
            return ["/mock/outdir/caltables/test.bcal"]
        elif "/*.gcal" in pattern:
            return ["/mock/outdir/caltables/test.gcal"]
        elif "selfcals_scan*.ms" in pattern:
            return ["/mock/workdir/selfcals_scan0.ms"]
        elif "targets_scan*.ms" in pattern:
            return ["/mock/workdir/targets_scan0.ms"]
        return []

    mock_glob.side_effect = glob_side_effect

    mock_table_inst = MagicMock()
    mock_table.return_value = mock_table_inst
    mock_table_inst.open.return_value = True
    mock_table_inst.getcol.return_value = [0]
    mock_table_inst.close.return_value = None

    msmd_mock = MagicMock()
    msmd_mock.nchan.return_value = 1024
    msmd_mock.chanres.return_value = [0.1]
    msmd_mock.ncorrforpol.return_value = [4]
    mock_msmetadata.return_value = msmd_mock

    with patch("os.path.abspath", side_effect=lambda x: x.rstrip("/")):
        ret = master_control(**kwargs)
    assert ret == expected_return


@pytest.mark.parametrize(
    "argv_args, expect_main_called",
    [
        (["run_master_controller"], False),  # No args → help + exit(1)
        (
            [
                "run_master_controller",
                "test.ms",
                "--workdir",
                "/mock/work",
                "--outdir",
                "/mock/out",
            ],
            True,
        ),  # Valid minimal
    ],
)
@patch("meersolar.meerpipeline.master_controller.master_control", return_value=0)
@patch("meersolar.meerpipeline.master_controller.drop_cache")
@patch("meersolar.meerpipeline.master_controller.argparse.ArgumentParser.print_help")
def test_cli(
    mock_print_help,
    mock_drop_cache,
    mock_master_control,
    argv_args,
    expect_main_called,
):
    with patch("sys.argv", argv_args):
        if expect_main_called:
            from meersolar.meerpipeline import master_controller

            result = master_controller.cli()
            mock_master_control.assert_called_once()
            mock_print_help.assert_not_called()
            mock_drop_cache.assert_any_call("test.ms")
            mock_drop_cache.assert_any_call("/mock/work")
            mock_drop_cache.assert_any_call("/mock/out")
            assert result is None
        else:
            with patch(
                "meersolar.meerpipeline.master_controller.sys.exit",
                side_effect=SystemExit(1),
            ) as mock_exit:
                with pytest.raises(SystemExit) as excinfo:
                    from meersolar.meerpipeline import master_controller

                    master_controller.cli()
                assert excinfo.value.code == 1
                mock_print_help.assert_called_once()
                mock_master_control.assert_not_called()
