import pytest
from unittest.mock import patch, MagicMock, call
from meersolar.meerpipeline.basic_cal import *


@pytest.mark.parametrize(
    "uvrange, expected_uvrange",
    [
        ("", ""),  # No uvrange
        (">200lambda", ">200lambda"),  # With uvrange
    ],
)
@patch("meersolar.meerpipeline.basic_cal.limit_threads")
@patch("meersolar.meerpipeline.basic_cal.suppress_output")
@patch("casatasks.gaincal")
@patch("meersolar.utils.calibration.delaycal")
def test_run_delaycal(
    mock_delaycal,
    mock_gaincal,
    mock_suppress_output,
    mock_limit_threads,
    uvrange,
    expected_uvrange,
):
    msname = "/mock/path/test.ms"
    expected_caltable = "test.kcal"
    result = run_delaycal(
        msname=msname,
        field="0",
        scan="1",
        refant="m001",
        refantmode="flex",
        solint="inf",
        combine="",
        gaintable=["prev.bcal"],
        gainfield=["0"],
        interp=["linear"],
        n_threads=1,
        uvrange=uvrange,
    )

    expected_calls = [call(n_threads=1)]
    mock_limit_threads.assert_has_calls(expected_calls)
    # Determine which function to assert based on which was used
    if mock_gaincal.called:
        mock_gaincal.assert_called_once_with(
            vis="/mock/path/test.ms",
            caltable="test.kcal",
            field="0",
            scan="1",
            uvrange=expected_uvrange,
            refant="m001",
            refantmode="flex",
            solint="inf",
            combine="",
            gaintype="K",
            gaintable=["prev.bcal"],
            gainfield=["0"],
            interp=["linear"],
        )
    else:
        mock_delaycal.assert_called_once_with(
            vis="/mock/path/test.ms",
            caltable="test.kcal",
            field="0",
            scan="1",
            uvrange=expected_uvrange,
            refant="m001",
            refantmode="flex",
            solint="inf",
            combine="",
            gaintable=["prev.bcal"],
            gainfield=["0"],
            interp=["linear"],
        )

    assert result == expected_caltable


@patch("meersolar.meerpipeline.basic_cal.limit_threads")
@patch("meersolar.meerpipeline.basic_cal.suppress_output")
@patch("casatasks.flagdata")
@patch("casatasks.bandpass")
def test_run_bandpass(
    mock_bandpass,
    mock_flagdata,
    mock_suppress_output,
    mock_limit_threads,
):
    """Test full bandpass calibration with mocks"""
    msname = "/mock/path/test.ms"
    expected_caltable = "test.bcal"

    result = run_bandpass(
        msname=msname,
        field="0",
        scan="1",
        uvrange=">100lambda",
        refant="m001",
        solint="int",
        solnorm=True,
        combine="scan",
        gaintable=["test.kcal"],
        gainfield=["0"],
        interp=["linear"],
        n_threads=2,
    )

    expected_calls = [call(n_threads=2)]
    mock_limit_threads.assert_has_calls(expected_calls)
    mock_bandpass.assert_called_once_with(
        vis=msname,
        caltable=expected_caltable,
        field="0",
        scan="1",
        uvrange=">100lambda",
        refant="m001",
        solint="int",
        solnorm=True,
        combine="scan",
        gaintable=["test.kcal"],
        gainfield=["0"],
        interp=["linear"],
    )
    mock_flagdata.assert_called_once_with(
        vis=expected_caltable,
        mode="rflag",
        datacolumn="CPARAM",
        flagbackup=False,
    )
    assert result == expected_caltable


@patch("meersolar.meerpipeline.basic_cal.limit_threads")
@patch("meersolar.meerpipeline.basic_cal.suppress_output")
@patch("casatasks.gaincal")
def test_run_gaincal(mock_gaincal, mock_suppress_output, mock_limit_threads):
    msname = "/mock/data/flux.ms"
    expected_caltable = "flux.gcal"

    result = run_gaincal(
        msname=msname,
        field="0",
        scan="1",
        uvrange=">100lambda",
        refant="m000",
        gaintype="G",
        solint="int",
        calmode="ap",
        refantmode="strict",
        solmode="",
        smodel=[1.0],
        rmsthresh=[3.0],
        combine="scan",
        append=True,
        gaintable=["flux.kcal"],
        gainfield=["0"],
        interp=["linear"],
        n_threads=4,
    )

    expected_calls = [call(n_threads=4)]
    mock_limit_threads.assert_has_calls(expected_calls)
    mock_gaincal.assert_called_once_with(
        vis=msname,
        caltable=expected_caltable,
        field="0",
        scan="1",
        uvrange=">100lambda",
        refant="m000",
        refantmode="strict",
        solint="int",
        combine="scan",
        gaintype="G",
        calmode="ap",
        solmode="",
        smodel=[1.0],
        rmsthresh=[3.0],
        append=True,
        gaintable=["flux.kcal"],
        gainfield=["0"],
        interp=["linear"],
    )

    assert result == expected_caltable


@patch("meersolar.meerpipeline.basic_cal.limit_threads")
@patch("meersolar.meerpipeline.basic_cal.suppress_output")
@patch("casatasks.polcal")
@patch("casatasks.flagdata")
def test_run_leakagecal(
    mock_flagdata,
    mock_polcal,
    mock_suppress_output,
    mock_limit_threads,
):
    msname = "/mock/data/target.ms"
    expected_caltable = "target.dcal"

    result = run_leakagecal(
        msname=msname,
        field="1",
        scan="3",
        uvrange=">50lambda",
        refant="m001",
        combine="scan",
        gaintable=["target.gcal"],
        gainfield=["1"],
        interp=["linear"],
        n_threads=2,
    )

    expected_calls = [call(n_threads=2)]
    mock_limit_threads.assert_has_calls(expected_calls)
    mock_polcal.assert_called_once_with(
        vis=msname,
        caltable=expected_caltable,
        field="1",
        scan="3",
        uvrange=">50lambda",
        refant="m001",
        solint="inf,10MHz",
        combine="scan",
        poltype="Df",
        gaintable=["target.gcal"],
        gainfield=["1"],
        interp=["linear"],
    )
    mock_flagdata.assert_called_once_with(
        vis=expected_caltable,
        mode="rflag",
        datacolumn="CPARAM",
        flagbackup=False,
    )
    assert result == expected_caltable


@patch("meersolar.meerpipeline.basic_cal.os.path.exists")
@patch("casatasks.polcal")
@patch("casatasks.gaincal")
@patch("meersolar.meerpipeline.basic_cal.suppress_output")
@patch("meersolar.meerpipeline.basic_cal.limit_threads")
def test_run_polcal(
    mock_limit_threads,
    mock_suppress_output,
    mock_gaincal,
    mock_polcal,
    mock_path_exists,
):
    mock_path_exists.side_effect = [True, True]
    msname = "mock.ms"
    field = "1"
    scan = "2"
    refant = "m001"
    gaintable = []
    gainfield = []
    interp = []
    kcrosscal, xfcal, panglecal = run_polcal(
        msname=msname,
        field=field,
        scan=scan,
        uvrange=">100lambda",
        refant=refant,
        combine="scan",
        gaintable=gaintable,
        gainfield=gainfield,
        interp=interp,
    )
    assert kcrosscal.endswith(".kcrosscal")
    assert xfcal.endswith(".xfcal")
    assert panglecal.endswith(".panglecal")
    mock_gaincal.assert_called()
    mock_polcal.assert_called()
    assert mock_polcal.call_count == 2


@patch("casatasks.applycal")
@patch("meersolar.meerpipeline.basic_cal.suppress_output")
@patch("meersolar.meerpipeline.basic_cal.limit_threads")
def test_run_applycal(
    mock_limit_threads,
    mock_suppress_output,
    mock_applycal,
):
    msname = "mock.ms"
    field = "1"
    scan = "2"
    gaintable = ["mock.kcal", "mock.bcal"]
    gainfield = ["1", "1"]
    interp = ["", ""]
    calwt = [False, False]
    result = run_applycal(
        msname=msname,
        field=field,
        scan=scan,
        applymode="calonly",
        flagbackup=True,
        gaintable=gaintable,
        gainfield=gainfield,
        interp=interp,
        calwt=calwt,
        parang=False,
        n_threads=2,
    )
    assert result is None
    mock_applycal.assert_called_once()


@patch("meersolar.meerpipeline.basic_cal.traceback")
@patch("meersolar.meerpipeline.basic_cal.suppress_output")
@patch("meersolar.meerpipeline.basic_cal.msmetadata")
@patch("meersolar.meerpipeline.basic_cal.get_chunk_size", return_value=2)
@patch("meersolar.meerpipeline.basic_cal.check_datacolumn_valid", return_value=True)
@patch("meersolar.meerpipeline.basic_cal.psutil.Process")
@patch("casatasks.flagdata")
@patch("meersolar.meerpipeline.basic_cal.limit_threads")
def test_run_postcal_flag(
    mock_limit_threads,
    mock_flagdata,
    mock_psutil_process,
    mock_check_col_valid,
    mock_get_chunk_size,
    mock_msmetadata,
    mock_suppress_output,
    mock_traceback,
):
    msname = "mock.ms"
    # Set up mock memory stats
    mock_proc = MagicMock()
    mock_proc.memory_info.return_value.rss = 3 * 1024**3  # 3 GB
    mock_psutil_process.return_value = mock_proc
    # Set up mock metadata
    mock_msmd = MagicMock()
    mock_msmd.scannumbers.return_value = [1]
    mock_msmd.timesforspws.return_value = np.array([0.0, 1.0, 2.0, 3.0])
    mock_msmetadata.return_value = mock_msmd
    run_postcal_flag(
        msname=msname,
        datacolumn="corrected",
        uvrange="",
        mode="rflag",
        n_threads=2,
        memory_limit=4,
    )
    expected_calls = [call(n_threads=2)]
    mock_limit_threads.assert_has_calls(expected_calls)
    mock_flagdata.assert_called_once()
    mock_suppress_output.assert_called_once()
    mock_msmetadata.assert_called_once()
    mock_get_chunk_size.assert_called_once_with(msname, memory_limit=4)


@patch("meersolar.meerpipeline.basic_cal.drop_cache")
@patch("meersolar.meerpipeline.basic_cal.time.sleep")
@patch("meersolar.meerpipeline.basic_cal.get_submsname_scans")
@patch("meersolar.meerpipeline.basic_cal.msmetadata")
@patch("meersolar.meerpipeline.basic_cal.get_local_dask_cluster")
@patch(
    "meersolar.meerpipeline.basic_cal.delayed",
    side_effect=lambda f: lambda *args, **kwargs: f(*args, **kwargs),
)
@patch(
    "meersolar.meerpipeline.basic_cal.merge_caltables",
    side_effect=lambda x, y, **kwargs: y,
)
@patch("meersolar.meerpipeline.basic_cal.get_column_size", return_value=1.0)
@patch("meersolar.meerpipeline.basic_cal.do_flag_backup")
@patch("meersolar.meerpipeline.basic_cal.run_applycal", return_value=None)
@patch("meersolar.meerpipeline.basic_cal.run_postcal_flag", return_value=None)
@patch(
    "meersolar.meerpipeline.basic_cal.run_delaycal", return_value="test_caltable.kcal"
)
@patch(
    "meersolar.meerpipeline.basic_cal.run_bandpass", return_value="test_caltable.bcal"
)
@patch(
    "meersolar.meerpipeline.basic_cal.run_gaincal", return_value="test_caltable.gcal"
)
@patch(
    "meersolar.meerpipeline.basic_cal.run_leakagecal", return_value="test_caltable.dcal"
)
@patch(
    "meersolar.meerpipeline.basic_cal.run_polcal",
    return_value=("mocked.kcrosscal", "mocked.xfcal", "mocked.panglecal"),
)
@patch("meersolar.meerpipeline.basic_cal.suppress_output")
@patch(
    "casatasks.fluxscale",
    return_value={
        "0": {"fieldName": "field2", "0": {"fluxd": [1.0], "fluxdErr": [0.1]}}
    },
)
@patch("meersolar.meerpipeline.basic_cal.os.path.exists", return_value=True)
@patch("meersolar.meerpipeline.basic_cal.os.system")
@patch("meersolar.meerpipeline.basic_cal.os.makedirs")
@patch("meersolar.meerpipeline.basic_cal.table")
def test_single_round_cal_and_flag(
    mock_table,
    mock_makedirs,
    mock_system,
    mock_exists,
    mock_fluxscale,
    mock_suppress,
    mock_polcal,
    mock_leakagecal,
    mock_gaincal,
    mock_bandpass,
    mock_delaycal,
    mock_postcal_flag,
    mock_applycal,
    mock_flag_backup,
    mock_get_col_size,
    mock_merge,
    mock_delayed,
    mock_get_local_dask_cluster,
    mock_msmetadata,
    mock_get_submsname_scans,
    mock_sleep,
    mock_drop_cache,
):
    # Setup fake Dask client
    fake_client = MagicMock()
    fake_client.compute.side_effect = lambda x: x  # Return the argument directly
    fake_client.gather.return_value = "mocked.kcal"
    mock_get_local_dask_cluster.return_value = (
        fake_client,
        fake_client,
        "/mock_dask_dir",
    )
    # Simulate sub-MS names and scan IDs
    mock_get_submsname_scans.return_value = (["ms1", "ms2"], [1, 2])
    # Simulate msmetadata
    msmd_instance = MagicMock()
    msmd_instance.ncorrforpol.return_value = [4]
    msmd_instance.fieldsforname.return_value = [0]
    mock_msmetadata.return_value = msmd_instance
    # Simulate CASA table object
    mock_tb = MagicMock()
    mock_table.return_value = mock_tb
    mock_tb.getcol.return_value = np.zeros((1, 1, 1), dtype=bool)
    status, caltables = single_round_cal_and_flag(
        "test.ms",
        fake_client,  # triggers use of mocked get_local_dask_cluster
        workdir="/tmp",
        cal_round=1,
        refant="ant1",
        uvrange="",
        fluxcal_scans={"field1": [1]},
        fluxcal_fields=["field1"],
        phasecal_scans={"field2": [2]},
        phasecal_fields=["field2"],
        phasecal_fluxes={"field2": 1.0},
        polcal_scans={"field3": [1]},
        polcal_fields=["field3"],
        do_phasecal=True,
        do_leakagecal=True,
        do_polcal=True,
        do_postcal_flag=True,
        cpu_frac=0.8,
        mem_frac=0.8,
    )
    # ✅ Assertions
    assert status == 0
    assert len(caltables) == 7
    for cal in caltables:
        assert cal is not None and cal.endswith("cal")
    assert fake_client.gather.call_count == 8
    expected_calls = [
        call("test_caltable.kcal"),
        call("test_caltable.bcal"),
        call("test_caltable.gcal"),
        call("mocked.kcal"),
        call("test_caltable.dcal"),
        call([("mocked.kcrosscal", "mocked.xfcal", "mocked.panglecal")]),
        call([None, None, None]),
        call([None, None, None]),
    ]
    fake_client.gather.assert_has_calls(expected_calls, any_order=False)


@patch("meersolar.meerpipeline.basic_cal.drop_cache")
@patch("meersolar.meerpipeline.basic_cal.time.sleep")
@patch("meersolar.meerpipeline.basic_cal.time.time", side_effect=lambda: 0)
@patch("meersolar.meerpipeline.basic_cal.os.chdir")
@patch("meersolar.meerpipeline.basic_cal.os.path.exists", return_value=True)
@patch("meersolar.meerpipeline.basic_cal.os.makedirs")
@patch("meersolar.meerpipeline.basic_cal.os.system")
@patch("meersolar.meerpipeline.basic_cal.msmetadata")
@patch(
    "meersolar.meerpipeline.basic_cal.get_fluxcals",
    return_value=(["field1"], {"field1": [1]}),
)
@patch(
    "meersolar.meerpipeline.basic_cal.get_polcals",
    return_value=(["field3"], {"field3": [3]}),
)
@patch(
    "meersolar.meerpipeline.basic_cal.get_phasecals",
    return_value=(["field2"], {"field2": [2]}, {"field2": 1.0}),
)
@patch("meersolar.meerpipeline.basic_cal.correct_missing_col_subms")
@patch("meersolar.meerpipeline.basic_cal.get_refant", return_value="ant1")
@patch("meersolar.meerpipeline.basic_cal.single_round_cal_and_flag")
@patch("casatasks.flagdata")
def test_run_basic_cal_rounds(
    mock_flagdata,
    mock_single_round,
    mock_get_refant,
    mock_correct_subms,
    mock_get_phase,
    mock_get_pol,
    mock_get_flux,
    mock_msmd,
    mock_system,
    mock_makedirs,
    mock_exists,
    mock_chdir,
    mock_time,
    mock_sleep,
    mock_drop,
):
    mock_msmd_instance = MagicMock()
    mock_msmd_instance.ncorrforpol.return_value = [4]
    mock_msmd.return_value = mock_msmd_instance
    mock_single_round.return_value = (
        0,
        ["a.cal", "b.cal", "c.cal", "d.cal", "e.cal", "f.cal", "g.cal"],
    )
    dask_client = MagicMock()
    status, caltables = run_basic_cal_rounds(
        "test.ms",
        dask_client,
        workdir="/tmp",
        keep_backup=True,
        perform_polcal=True,
    )
    assert status == 0
    assert len(caltables) == 7
    mock_single_round.side_effect = [
        (0, ["a.cal", "b.cal", "c.cal", "d.cal", "e.cal", "f.cal", "g.cal"]),
        (0, ["a.cal", "b.cal", "c.cal", "d.cal", "e.cal", "f.cal", "g.cal"]),
        (1, []),
    ]
    dask_client = MagicMock()
    status_fail, caltables_fail = run_basic_cal_rounds(
        "test.ms",
        dask_client,
        workdir="/tmp",
        keep_backup=False,
        perform_polcal=True,
    )
    assert mock_flagdata.call_count >= 1
    assert status_fail == 1
    assert caltables_fail == []


@patch("meersolar.meerpipeline.basic_cal.clean_shutdown")
@patch("meersolar.meerpipeline.basic_cal.drop_cache")
@patch("meersolar.meerpipeline.basic_cal.init_logger")
@patch("meersolar.meerpipeline.basic_cal.np.load", return_value=("jobid", "password"))
@patch("meersolar.meerpipeline.basic_cal.os.path.exists", side_effect=lambda p: True)
@patch("meersolar.meerpipeline.basic_cal.os.makedirs")
@patch("meersolar.meerpipeline.basic_cal.os.getpid", return_value=12345)
@patch("meersolar.meerpipeline.basic_cal.get_cachedir", return_value="/mock/cache")
@patch("meersolar.meerpipeline.basic_cal.save_pid")
@patch(
    "meersolar.meerpipeline.basic_cal.run_basic_cal_rounds",
    return_value=(0, ["/mock/caltable1", "/mock/caltable2"]),
)
@patch("meersolar.meerpipeline.basic_cal.os.system")
def test_main(
    mock_system,
    mock_run_basic_cal_rounds,
    mock_save_pid,
    mock_get_cachedir,
    mock_getpid,
    mock_makedirs,
    mock_exists,
    mock_npload,
    mock_init_logger,
    mock_drop_cache,
    mock_clean_shutdown,
):
    # Setup return for logger
    mock_logger = MagicMock()
    mock_init_logger.return_value = mock_logger

    # Inputs
    msname = "/mock/data/test.ms"
    workdir = "/mock/data/workdir"
    caldir = "/mock/data/caltables"
    dask_client = MagicMock()

    # Call main
    result = main(
        msname=msname,
        workdir=workdir,
        caldir=caldir,
        refant="m001",
        uvrange=">100lambda",
        perform_polcal=True,
        keep_backup=True,
        cpu_frac=0.5,
        mem_frac=0.5,
        logfile="/mock/logfile.log",
        jobid="123",
        start_remote_log=True,
        dask_client=dask_client,
    )

    # Assertions
    assert result == 0
    mock_save_pid.assert_called_once()

    mock_run_basic_cal_rounds.assert_called_once_with(
        msname,
        dask_client,
        workdir,
        refant="m001",
        uvrange=">100lambda",
        perform_polcal=True,
        keep_backup=True,
        cpu_frac=0.5,
        mem_frac=0.5,
    )

    for caltable in ["/mock/caltable1", "/mock/caltable2"]:
        dest = os.path.join(caldir, os.path.basename(caltable))
        mock_system.assert_any_call(f"rm -rf {dest}")
        mock_system.assert_any_call(f"mv {caltable} {caldir}")


@pytest.mark.parametrize(
    "argv_args, expect_main_called, expected_exit",
    [
        (["run_basic_cal"], False, 1),  # No arguments → print help and exit(1)
        (
            [
                "run_basic_cal",
                "test.ms",
                "--workdir",
                "/mock/work",
                "--caldir",
                "/mock/cal",
            ],
            True,
            0,
        ),  # Minimal valid case
        (
            [
                "run_basic_cal",
                "test.ms",
                "--workdir",
                "/mock/work",
                "--caldir",
                "/mock/cal",
                "--perform_polcal",
                "--keep_backup",
                "--cpu_frac",
                "0.5",
                "--mem_frac",
                "0.6",
                "--jobid",
                "123",
                "--start_remote_log",
                "--logfile",
                "log.txt",
            ],
            True,
            0,
        ),  # Full case with all optional arguments
    ],
)
@patch("meersolar.meerpipeline.basic_cal.main", return_value=0)
@patch("meersolar.meerpipeline.basic_cal.sys.exit")
@patch("meersolar.meerpipeline.basic_cal.argparse.ArgumentParser.print_help")
def test_cli(
    mock_print_help,
    mock_exit,
    mock_main,
    argv_args,
    expect_main_called,
    expected_exit,
):
    with patch("sys.argv", argv_args):
        from meersolar.meerpipeline import basic_cal

        result = basic_cal.cli()
        assert result == expected_exit
