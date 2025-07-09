import pytest
from unittest.mock import patch, MagicMock
from meersolar.pipeline.basic_cal import *

@patch("meersolar.pipeline.basic_cal.limit_threads")
@patch("meersolar.pipeline.basic_cal.suppress_casa_output")
@patch("casatasks.gaincal")
def test_run_delaycal(
    mock_gaincal,
    mock_suppress_output,
    mock_limit_threads
):
    """Test full execution path with CASA gaincal mocked"""
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
        dry_run=False,
    )
    mock_limit_threads.assert_called_once_with(n_threads=1)
    mock_gaincal.assert_called_once_with(
        vis=msname,
        caltable=expected_caltable,
        field="0",
        scan="1",
        uvrange="",
        refant="m001",
        refantmode="flex",
        solint="inf",
        combine="",
        gaintype="K",
        gaintable=["prev.bcal"],
        gainfield=["0"],
        interp=["linear"],
    )
    assert result == expected_caltable
    
    
@patch("meersolar.pipeline.basic_cal.limit_threads")
@patch("meersolar.pipeline.basic_cal.suppress_casa_output")
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
        dry_run=False,
    )

    mock_limit_threads.assert_called_once_with(n_threads=2)
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
    
 
@patch("meersolar.pipeline.basic_cal.limit_threads")
@patch("meersolar.pipeline.basic_cal.suppress_casa_output")
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
        dry_run=False,
    )

    mock_limit_threads.assert_called_once_with(n_threads=4)
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
    

@patch("meersolar.pipeline.basic_cal.limit_threads")
@patch("meersolar.pipeline.basic_cal.suppress_casa_output")
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
        dry_run=False,
    )

    mock_limit_threads.assert_called_once_with(n_threads=2)
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


@patch("meersolar.pipeline.basic_cal.os.path.exists")
@patch("casatasks.polcal")
@patch("casatasks.gaincal")
@patch("meersolar.pipeline.basic_cal.suppress_casa_output")
@patch("meersolar.pipeline.basic_cal.limit_threads")
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
        dry_run=False,
    )
    assert kcrosscal.endswith(".kcrosscal")
    assert xfcal.endswith(".xfcal")
    assert panglecal.endswith(".panglecal")
    mock_gaincal.assert_called()
    mock_polcal.assert_called()
    assert mock_polcal.call_count == 2
    
    
@patch("casatasks.applycal")
@patch("meersolar.pipeline.basic_cal.suppress_casa_output")
@patch("meersolar.pipeline.basic_cal.limit_threads")
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
        dry_run=False,
    )
    assert result is None
    mock_applycal.assert_called_once()
    
    
@patch("meersolar.pipeline.basic_cal.traceback")
@patch("meersolar.pipeline.basic_cal.suppress_casa_output")
@patch("meersolar.pipeline.basic_cal.msmetadata")
@patch("meersolar.pipeline.basic_cal.get_chunk_size", return_value=2)
@patch("meersolar.pipeline.basic_cal.check_datacolumn_valid", return_value=True)
@patch("meersolar.pipeline.basic_cal.psutil.Process")
@patch("casatasks.flagdata")
@patch("meersolar.pipeline.basic_cal.limit_threads")
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
        dry_run=False,
    )
    mock_limit_threads.assert_called_once()
    mock_flagdata.assert_called_once()
    mock_suppress_output.assert_called_once()
    mock_msmetadata.assert_called_once()
    mock_get_chunk_size.assert_called_once_with(msname, memory_limit=4)
    

@patch("meersolar.pipeline.basic_cal.drop_cache")
@patch("meersolar.pipeline.basic_cal.time.sleep")
@patch("meersolar.pipeline.basic_cal.get_submsname_scans")
@patch("meersolar.pipeline.basic_cal.msmetadata")
@patch("meersolar.pipeline.basic_cal.get_dask_client")
@patch("meersolar.pipeline.basic_cal.run_limited_memory_task")
@patch("meersolar.pipeline.basic_cal.compute", side_effect=lambda *args: [[f"{i}.cal" for i in range(len(args))]])
@patch("meersolar.pipeline.basic_cal.delayed", side_effect=lambda f: f)
@patch("meersolar.pipeline.basic_cal.merge_caltables", side_effect=lambda x, y, **kwargs: y)
@patch("meersolar.pipeline.basic_cal.get_ms_size", return_value=1.0)
@patch("meersolar.pipeline.basic_cal.do_flag_backup")
@patch("meersolar.pipeline.basic_cal.run_applycal", return_value=None)
@patch("meersolar.pipeline.basic_cal.run_postcal_flag", return_value=None)
@patch("meersolar.pipeline.basic_cal.run_delaycal", return_value="test_caltable.kcal")
@patch("meersolar.pipeline.basic_cal.run_bandpass", return_value="bandpass.cal")
@patch("meersolar.pipeline.basic_cal.run_gaincal", return_value="gain.cal")
@patch("meersolar.pipeline.basic_cal.run_leakagecal", return_value="leakage.cal")
@patch("meersolar.pipeline.basic_cal.run_polcal", return_value=("kcross.cal", "crossphase.cal", "pangle.cal"))
@patch("meersolar.pipeline.basic_cal.suppress_casa_output")
@patch("casatasks.fluxscale", return_value={
    "0": {"fieldName": "field2", "0": {"fluxd": [1.0], "fluxdErr": [0.1]}},
})
@patch("meersolar.pipeline.basic_cal.os.path.exists", return_value=True)
@patch("meersolar.pipeline.basic_cal.os.system")
@patch("meersolar.pipeline.basic_cal.os.makedirs")
@patch("meersolar.pipeline.basic_cal.table")
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
    mock_ms_size,
    mock_merge,
    mock_delayed,
    mock_compute,
    mock_memtask,
    mock_dask,
    mock_msmd,
    mock_getscans,
    mock_sleep,
    mock_drop
):
    # Setup mocks
    mock_dask.return_value = (MagicMock(), MagicMock(), 1, 1, 1.0)
    mock_getscans.return_value = (["ms1", "ms2"], [1, 2])
    mock_memtask.return_value = 0.1
    # Mock msmetadata
    mock_msmd_instance = MagicMock()
    mock_msmd_instance.ncorrforpol.return_value = [4]
    mock_msmd_instance.fieldsforname.return_value = [0]
    mock_msmd.return_value = mock_msmd_instance
    # Mock CASA table behavior
    mock_tb = MagicMock()
    mock_table.return_value = mock_tb
    mock_tb.open.return_value = None
    mock_tb.getcol.return_value = np.zeros((1, 1, 1), dtype=bool)
    mock_tb.putcol.return_value = None
    mock_tb.flush.return_value = None
    mock_tb.close.return_value = None
    # Call the function
    status, caltables = single_round_cal_and_flag(
        msname="test.ms",
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
        do_delaycal=True,
        do_phasecal=True,
        do_leakagecal=True,
        do_polcal=True,
        do_postcal_flag=True,
        cpu_frac=0.8,
        mem_frac=0.8,
    )
    assert status == 0
    assert len(caltables) == 7
    for cal in caltables:
        assert cal is not None and cal.endswith("cal")
        
        
@patch("meersolar.pipeline.basic_cal.drop_cache")
@patch("meersolar.pipeline.basic_cal.time.sleep")
@patch("meersolar.pipeline.basic_cal.time.time", side_effect=lambda: 0)
@patch("meersolar.pipeline.basic_cal.os.chdir")
@patch("meersolar.pipeline.basic_cal.os.path.exists", return_value=True)
@patch("meersolar.pipeline.basic_cal.os.makedirs")
@patch("meersolar.pipeline.basic_cal.os.system")
@patch("meersolar.pipeline.basic_cal.msmetadata")
@patch("meersolar.pipeline.basic_cal.get_fluxcals", return_value=(["field1"], {"field1": [1]}))
@patch("meersolar.pipeline.basic_cal.get_polcals", return_value=(["field3"], {"field3": [3]}))
@patch("meersolar.pipeline.basic_cal.get_phasecals", return_value=(["field2"], {"field2": [2]}, {"field2": 1.0}))
@patch("meersolar.pipeline.basic_cal.correct_missing_col_subms")
@patch("meersolar.pipeline.basic_cal.get_refant", return_value="ant1")
@patch("meersolar.pipeline.basic_cal.single_round_cal_and_flag")
def test_run_basic_cal_rounds(
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
    mock_drop
):
    mock_msmd_instance = MagicMock()
    mock_msmd_instance.ncorrforpol.return_value = [4]
    mock_msmd.return_value = mock_msmd_instance
    mock_single_round.return_value = (0, ["a.cal", "b.cal", "c.cal", "d.cal", "e.cal", "f.cal", "g.cal"])
    status, caltables = run_basic_cal_rounds(
        msname="test.ms",
        workdir="/tmp",
        keep_backup=True,
        do_delaycal=True,
        perform_polcal=True,
    )
    assert status == 0
    assert len(caltables) == 7
    mock_single_round.side_effect = [
        (0, ["a.cal", "b.cal", "c.cal", "d.cal", "e.cal", "f.cal", "g.cal"]),
        (0, ["a.cal", "b.cal", "c.cal", "d.cal", "e.cal", "f.cal", "g.cal"]),
        (1, [])
    ]
    status_fail, caltables_fail = run_basic_cal_rounds(
        msname="test.ms",
        workdir="/tmp",
        keep_backup=False,
        do_delaycal=True,
        perform_polcal=True,
    )
    assert status_fail == 1
    assert caltables_fail == []
    
    
@pytest.mark.parametrize(
    "argv_args, expected_return",
    [
        (
            [
                "run_basic_cal",
                "test.ms",
                "--workdir", "/mock/workdir",
                "--caldir", "/mock/caldir",
                "--logfile", "/mock/logfile.log",
                "--jobid", "42"
            ],
            0,  
        )
    ]
)
@patch("meersolar.pipeline.basic_cal.np.load", return_value=("jobname", "password"))
@patch("meersolar.pipeline.basic_cal.sys")
@patch("meersolar.pipeline.basic_cal.init_logger")
@patch("meersolar.pipeline.basic_cal.time.sleep")
@patch("meersolar.pipeline.basic_cal.clean_shutdown")
@patch("meersolar.pipeline.basic_cal.drop_cache")
@patch("meersolar.pipeline.basic_cal.save_pid")
@patch("meersolar.pipeline.basic_cal.os.path.exists", return_value=True)
@patch("meersolar.pipeline.basic_cal.os.makedirs")
@patch("meersolar.pipeline.basic_cal.os.system")
@patch("meersolar.pipeline.basic_cal.run_basic_cal_rounds",return_value=(0,["a.cal","b.cal"]))
def test_main(
    mock_run_basic,
    mock_system,
    mock_makedirs,
    mock_exists,
    mock_save_pid,
    mock_drop_cache,
    mock_shutdown,
    mock_sleep,
    mock_init_logger,
    mock_sys,
    mock_npload,
    argv_args,
    expected_return,
):
    with patch("sys.argv", argv_args):
        from meersolar.pipeline import basic_cal
        result = basic_cal.main()
        assert result == expected_return
    
