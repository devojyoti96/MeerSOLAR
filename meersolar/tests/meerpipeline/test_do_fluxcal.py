import pytest
from unittest.mock import patch, MagicMock, call, ANY
from meersolar.meerpipeline.do_fluxcal import *


def test_split_casatask(dummy_msname):
    outputvis = os.getcwd() + "/scan1.ms"
    if os.path.exists(outputvis):
        os.remove(outputvis)
    result = split_casatask(msname=dummy_msname, outputvis=outputvis, scan="1")
    assert result == outputvis
    assert os.path.exists(result)
    os.system(f"rm -rf {result}")
    assert os.path.exists(result) == False


@patch("meersolar.meerpipeline.do_fluxcal.split_casatask")
@patch("meersolar.meerpipeline.do_fluxcal.get_local_dask_cluster")
@patch("meersolar.meerpipeline.do_fluxcal.os.path.exists", return_value=False)
@patch("meersolar.meerpipeline.do_fluxcal.get_ms_scan_size")
@patch(
    "meersolar.meerpipeline.do_fluxcal.delayed",
    side_effect=lambda f: lambda *args, **kwargs: f(*args, **kwargs),
)
def test_split_autocorr(
    mock_delayed,
    mock_get_column_size,
    mock_exists,
    mock_get_dask_client,
    mock_split_casatask,
):
    # Mock split_casatask return
    mock_split_casatask.side_effect = lambda ms, out, scan, tr, **kwargs: out

    # Create fake Dask client and cluster
    dummy_client = MagicMock()
    dummy_cluster = MagicMock()
    dummy_client.cluster = dummy_cluster
    dummy_client.compute.side_effect = lambda x: x  # pass-through
    dummy_client.gather.side_effect = lambda x: x  # pass-through

    mock_get_dask_client.return_value = (dummy_client, dummy_cluster, "/mock/dask_dir")
    mock_get_column_size.side_effect = [0.01, 0.02]  # GB for scans 1 and 2

    # Call function
    result = split_autocorr(
        "mock.ms",
        dask_client=dummy_client,
        workdir="/mock/workdir",
        scan_list=[1, 2],
        time_window=-1,
        cpu_frac=0.5,
        mem_frac=0.5,
    )

    # Assert results
    assert isinstance(result, list)
    assert result == [
        "/mock/workdir/autocorr_scan_1.ms",
        "/mock/workdir/autocorr_scan_2.ms",
    ]

    # Cluster adaptation and compute/gather checks
    assert dummy_client.compute.called
    assert dummy_client.gather.called


@patch("meersolar.meerpipeline.do_fluxcal.casamstool")
def test_get_on_off_power(mock_casamstool):
    mock_ms = MagicMock()
    mock_casamstool.return_value = mock_ms
    data = np.ones((1, 2, 4, 6), dtype=np.complex64)  # shape: pol, chan, ant, time
    flag = np.zeros_like(data, dtype=bool)
    data[..., 0::2] *= 2  # simulate higher power for ON
    mock_ms.getdata.return_value = {"data": data, "flag": flag}
    msname = "dummy.ms"
    ant_list = [0, 1, 2, 3]
    scale_factor = np.ones((1, 2))
    result = get_on_off_power(
        msname=msname,
        scale_factor=scale_factor,
        ant_list=ant_list,
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 2, 4)  # averaged over time
    np.testing.assert_allclose(result, 1.0, rtol=0.1)
    mock_ms.open.assert_called_once_with(msname)
    mock_ms.select.assert_called()
    mock_ms.selectpolarization.assert_called_with(["XX", "YY"])
    mock_ms.getdata.assert_called_once()
    mock_ms.close.assert_called_once()


@patch("meersolar.meerpipeline.do_fluxcal.msmetadata")
@patch("meersolar.meerpipeline.do_fluxcal.get_on_off_power")
def test_get_att_per_ant(mock_get_on_off_power, mock_msmetadata):
    mock_msmd_instance = MagicMock()
    mock_msmd_instance.nantennas.return_value = 4
    mock_msmetadata.return_value = mock_msmd_instance
    cal_diff = np.ones((2, 3, 4)) * 2
    source_diff = np.ones((2, 3, 4)) * 1
    mock_get_on_off_power.side_effect = [cal_diff, source_diff]
    cal_ms = "cal.ms"
    source_ms = "source.ms"
    scale_factor = np.ones((2, 3))  # shape npol x nchan
    att = get_att_per_ant(cal_ms, source_ms, scale_factor, ant_list=[])
    assert isinstance(att, np.ndarray)
    assert att.shape == (2, 3, 4)
    np.testing.assert_allclose(att, 0.5, rtol=1e-6)
    mock_msmd_instance.open.assert_called_once_with(cal_ms)
    mock_msmd_instance.nantennas.assert_called_once()
    mock_msmd_instance.close.assert_called_once()
    calls = mock_get_on_off_power.call_args_list
    assert len(calls) == 2
    call1_args, call1_kwargs = calls[0]
    assert call1_args[0] == cal_ms
    assert np.allclose(call1_args[1], scale_factor)
    assert call1_kwargs["ant_list"] == [0, 1, 2, 3]
    call2_args, call2_kwargs = calls[1]
    assert call2_args[0] == source_ms
    assert np.allclose(call2_args[1], np.ones_like(scale_factor))
    assert call2_kwargs["ant_list"] == [0, 1, 2, 3]


@patch("meersolar.meerpipeline.do_fluxcal.psutil.Process")
@patch("meersolar.meerpipeline.do_fluxcal.get_att_per_ant")
@patch("meersolar.meerpipeline.do_fluxcal.get_on_off_power")
@patch("meersolar.meerpipeline.do_fluxcal.get_column_size")
@patch("meersolar.meerpipeline.do_fluxcal.table")
@patch("meersolar.meerpipeline.do_fluxcal.msmetadata")
@patch("meersolar.meerpipeline.do_fluxcal.limit_threads")
def test_get_power_diff(
    mock_limit_threads,
    mock_msmetadata,
    mock_table,
    mock_get_column_size,
    mock_get_on_off_power,
    mock_get_att_per_ant,
    mock_psutil_process,
):
    mock_msmd_instance = MagicMock()
    mock_msmd_instance.nrows.return_value = 400
    mock_msmd_instance.nchan.return_value = 3
    mock_msmd_instance.ncorrforpol.return_value = 2
    mock_msmd_instance.nantennas.return_value = 4
    mock_msmd_instance.nbaselines.return_value = 6
    mock_msmetadata.return_value = mock_msmd_instance
    mock_table_instance = MagicMock()
    mock_table.return_value = mock_table_instance
    on_gain = np.ones((2, 3, 4), dtype=np.complex64)
    off_gain = np.full((2, 3, 4), 0.5, dtype=np.complex64)
    mock_table_instance.getcol.side_effect = [on_gain, off_gain]
    mock_get_column_size.side_effect = [0.01, 0.02]  # in GB
    mock_get_on_off_power.return_value = 0.01  # function memory cost in GB
    mock_att_per_ant = np.ones((2, 3, 4)) * 0.5
    mock_get_att_per_ant.return_value = mock_att_per_ant
    att, att_ant_array = get_power_diff(
        cal_msname="flux.ms",
        source_msname="source.ms",
        on_cal="on.tbl",
        off_cal="off.tbl",
        n_threads=1,
        memory_limit=1.0,
    )
    assert isinstance(att, np.ndarray)
    assert isinstance(att_ant_array, np.ndarray)
    assert att.shape == (2, 3)
    assert att_ant_array.shape == (2, 3, 4)
    np.testing.assert_allclose(att, 0.5)
    np.testing.assert_allclose(att_ant_array, 0.5)
    assert mock_get_att_per_ant.called
    mock_msmd_instance.open.assert_called_once()
    mock_table_instance.getcol.assert_any_call("CPARAM")


@patch(
    "meersolar.meerpipeline.do_fluxcal.get_power_diff",
    return_value=[
        np.array([[0.5, 0.6, 0.7], [0.52, 0.62, 0.72]]),
        np.array(
            [
                [
                    [0.48, 0.51, 0.49, 0.52],
                    [0.59, 0.61, 0.6, 0.58],
                    [0.7, 0.69, 0.71, 0.72],
                ],
                [
                    [0.5, 0.53, 0.51, 0.54],
                    [0.6, 0.63, 0.6, 0.64],
                    [0.7, 0.73, 0.7, 0.74],
                ],
            ]
        ),
    ],
)
@patch("meersolar.meerpipeline.do_fluxcal.delayed", side_effect=lambda f: f)
@patch(
    "meersolar.meerpipeline.do_fluxcal.get_fluxcals",
    return_value=(["J0408-6545"], {1: [5]}),
)
@patch("meersolar.meerpipeline.do_fluxcal.get_bad_chans", return_value=[1, 2])
@patch("meersolar.meerpipeline.do_fluxcal.get_bad_ants", return_value=([0, 1], "0,1"))
@patch("meersolar.meerpipeline.do_fluxcal.msmetadata")
@patch("meersolar.meerpipeline.do_fluxcal.get_ms_scan_size", return_value=0.01)
@patch("meersolar.meerpipeline.do_fluxcal.split_autocorr")
@patch("meersolar.meerpipeline.do_fluxcal.drop_cache")
@patch("meersolar.meerpipeline.do_fluxcal.single_ms_flag", return_value=0)
@patch("meersolar.meerpipeline.do_fluxcal.np.save")
@patch("meersolar.meerpipeline.do_fluxcal.os.path.exists", return_value=True)
@patch("meersolar.meerpipeline.do_fluxcal.os.makedirs")
@patch("meersolar.meerpipeline.do_fluxcal.get_column_size", return_value=0.01)
def test_estimate_att(
    mock_get_column_size,
    mock_makedirs,
    mock_exists,
    mock_save,
    mock_single_ms_flag,
    mock_drop_cache,
    mock_split_autocorr,
    mock_ms_size,
    mock_msmetadata,
    mock_get_bad_ants,
    mock_get_bad_chans,
    mock_get_fluxcals,
    mock_delayed,
    mock_get_power_diff,
):
    msname = "test.ms"
    workdir = "/mock/workdir"
    on_cal = "on.cal"
    off_cal = "off.cal"
    flux_scan = 5
    target_scans = [10]

    mock_dask_client = MagicMock()
    mock_dask_cluster = MagicMock()
    mock_dask_client.cluster = mock_dask_cluster
    mock_dask_client.compute.side_effect = lambda tasks: tasks
    mock_dask_client.gather.side_effect = lambda tasks: [
        mock_get_power_diff.return_value
    ]

    mock_msmd = MagicMock()
    mock_msmd.chanfreqs.return_value = np.array([100e6, 110e6, 120e6])
    mock_msmetadata.return_value = mock_msmd

    mock_split_autocorr.return_value = [
        f"{workdir}/autocorr_scan_{flux_scan}.ms",
        f"{workdir}/autocorr_scan_{target_scans[0]}.ms",
    ]

    status, att_level, att_files = estimate_att(
        msname,
        mock_dask_client,
        workdir,
        on_cal,
        off_cal,
        flux_scan,
        target_scans,
        time_window=300,
        cpu_frac=0.5,
        mem_frac=0.5,
    )

    assert status == 0
    assert isinstance(att_level, dict)
    assert isinstance(att_files, list)
    assert target_scans[0] in att_level
    assert att_level[target_scans[0]].shape == (2, 3)
    assert att_files[0].endswith(".npy")
    mock_dask_client.compute.assert_called()
    mock_dask_client.gather.assert_called()
    mock_save.assert_called()


@patch("meersolar.meerpipeline.do_fluxcal.drop_cache")
@patch("meersolar.meerpipeline.do_fluxcal.os.system")
@patch("meersolar.meerpipeline.do_fluxcal.os.makedirs")
@patch("casatasks.bandpass")
@patch("casatasks.split")
@patch("meersolar.meerpipeline.do_fluxcal.import_fluxcal_models")
@patch("meersolar.meerpipeline.do_fluxcal.single_ms_flag")
@patch("meersolar.meerpipeline.do_fluxcal.get_bad_ants", return_value=([0, 1], "0,1"))
@patch("meersolar.meerpipeline.do_fluxcal.get_bad_chans", return_value=[1, 2])
@patch(
    "meersolar.meerpipeline.do_fluxcal.get_fluxcals",
    return_value=(["J0408-6545"], {"1": [1]}),
)
@patch(
    "meersolar.meerpipeline.do_fluxcal.determine_noise_diode_cal_scan", return_value="1"
)
@patch(
    "meersolar.meerpipeline.do_fluxcal.get_cal_target_scans",
    return_value=(["2"], ["1"], ["1"], [], []),
)
@patch("meersolar.meerpipeline.do_fluxcal.get_valid_scans", return_value=["1", "2"])
@patch("meersolar.meerpipeline.do_fluxcal.msmetadata")
@patch("meersolar.meerpipeline.do_fluxcal.casamstool")
@patch("meersolar.meerpipeline.do_fluxcal.estimate_att")
@patch("meersolar.meerpipeline.do_fluxcal.os.chdir")
def test_run_noise_cal(
    mock_chdir,
    mock_estimate_att,
    mock_casamstool,
    mock_msmetadata,
    mock_get_valid_scans,
    mock_get_cal_target_scans,
    mock_determine_noise_diode_cal_scan,
    mock_get_fluxcals,
    mock_get_bad_chans,
    mock_get_bad_ants,
    mock_single_ms_flag,
    mock_import_fluxcal_models,
    mock_split,
    mock_bandpass,
    mock_makedirs,
    mock_system,
    mock_drop_cache,
):
    # Setup mocks
    mock_mstool = MagicMock()
    mock_casamstool.return_value = mock_mstool
    mock_mstool.getdata.return_value = {"data": np.ones((1, 1, 1, 3))}
    mock_msmd = MagicMock()
    mock_msmd.timesforscan.return_value = [1.0, 2.0, 3.0]
    mock_msmetadata.return_value = mock_msmd
    mock_estimate_att.return_value = (
        0,
        {"2": np.array([0.5, 0.6, 0.7])},
        ["attfile.npy"],
    )
    msname = "test.ms"
    workdir = "/mock/workdir"
    dask_client = MagicMock()
    status, att_level, att_files = run_noise_cal(msname, dask_client, workdir)
    assert status == 0
    assert isinstance(att_level, dict)
    assert isinstance(att_files, list)
    assert "2" in att_level
    assert att_files[0].endswith(".npy")
    mock_split.assert_called_once()
    mock_bandpass.assert_called()
    mock_estimate_att.assert_called_once()


@pytest.mark.parametrize(
    "argv, should_exit",
    [
        (["prog.py"], 1),
        (
            [
                "prog.py",
                "ms1.ms",
                "--workdir",
                "/mock/work",
                "--caldir",
                "/mock/caldir",
                "--cpu_frac",
                "0.6",
                "--mem_frac",
                "0.7",
                "--keep_backup",
                "--jobid",
                "123",
            ],
            0,
        ),
    ],
)
@patch("meersolar.meerpipeline.do_fluxcal.main", return_value=0)
def test_cli_fluxcal(mock_main, argv, should_exit):
    with patch("sys.argv", argv):
        from meersolar.meerpipeline import do_fluxcal

        result = do_fluxcal.cli()
        assert result == should_exit
