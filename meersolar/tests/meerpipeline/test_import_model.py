import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from meersolar.meerpipeline.import_model import *


@patch("meersolar.meerpipeline.import_model.np.loadtxt")
def test_get_polmodel_coeff(mock_loadtxt):
    freq = np.array(
        [1.022, 1.465, 1.865, 2.565, 3.565, 4.885, 6.680, 8.435, 11.320, 14.065, 16.564]
    )
    I = np.array([17.46, 14.64, 13.03, 10.85, 8.94, 7.33, 5.97, 5.12, 4.12, 3.54, 3.15])
    pfrac = np.array(
        [
            0.08618,
            0.09794,
            0.10122,
            0.10575,
            0.11153,
            0.11525,
            0.11858,
            0.12045,
            0.12261,
            0.12303,
            0.12544,
        ]
    )
    pangle = np.array(
        [
            0.57632,
            0.57261,
            0.57613,
            0.57580,
            0.57624,
            0.57503,
            0.57758,
            0.57827,
            0.58412,
            0.59534,
            0.60172,
        ]
    )
    mock_loadtxt.return_value = (freq, I, pfrac, pangle)
    ref_freq, I0, polyI, poly_pfrac, poly_pangle = get_polmodel_coeff("dummy.txt")
    assert ref_freq == freq[0]
    assert I0 == I[0]
    assert len(polyI) == 5
    assert len(poly_pfrac) == 6
    assert len(poly_pangle) == 6
    assert all(isinstance(x, float) for x in polyI)


@pytest.mark.parametrize(
    "field_name, expected",
    [
        ("3C286", None),  # full run calls setjy
        ("Unknown", 1),  # unknown field returns 1
    ],
)
@patch("meersolar.meerpipeline.import_model.traceback.print_exc")
@patch("meersolar.meerpipeline.import_model.suppress_output")
@patch("casatasks.setjy")
@patch("meersolar.meerpipeline.import_model.get_polmodel_coeff")
@patch("meersolar.meerpipeline.import_model.psutil.Process")
@patch("meersolar.meerpipeline.import_model.limit_threads")
def test_polcal_setjy(
    mock_limit_threads,
    mock_process,
    mock_getmodel,
    mock_setjy,
    mock_suppress,
    mock_traceback,
    field_name,
    expected,
):
    mock_process.return_value.memory_info.return_value.rss = 2.5 * 1024**3
    mock_getmodel.return_value = (
        1.0,
        10.0,
        [0.1, 0.01, -0.001],
        [0.05, 0.01],
        [0.3, -0.1],
    )
    result = polcal_setjy(
        msname="test.ms",
        field_name=field_name,
    )
    if field_name == "Unknown":
        assert result == expected
        mock_setjy.assert_not_called()
    else:
        assert result == expected
        mock_setjy.assert_called_once()


@pytest.mark.parametrize(
    "expected",
    [
        (None),  # normal run
    ],
)
@patch("meersolar.meerpipeline.import_model.traceback.print_exc")
@patch("meersolar.meerpipeline.import_model.suppress_output")
@patch("casatasks.setjy")
@patch("meersolar.meerpipeline.import_model.psutil.Process")
@patch("meersolar.meerpipeline.import_model.limit_threads")
def test_phasecal_setjy(
    mock_limit_threads,
    mock_process,
    mock_setjy,
    mock_suppress,
    mock_traceback,
    expected,
):
    mock_process.return_value.memory_info.return_value.rss = 2.5 * 1024**3
    result = phasecal_setjy(
        msname="test.ms",
        field="phasecal_field",
        ismms=True,
    )
    assert result == expected
    mock_setjy.assert_called_once()


@pytest.mark.parametrize(
    "fluxcal_fields, expected_return, system_status",
    [
        ([], 1, 0),  # No flux calibrators → return 1
        (["J0408-6545"], 0, 0),  # Success path → return 0
        (["J0408-6545"], 0, 1),  # Error path in crystalball (still continue) → return 0
    ],
)
@patch("meersolar.meerpipeline.import_model.os.system")
@patch("meersolar.meerpipeline.import_model.suppress_output")
@patch("meersolar.meerpipeline.import_model.get_band_name")
@patch("meersolar.meerpipeline.import_model.get_ms_scans")
@patch("meersolar.meerpipeline.import_model.psutil.cpu_count", return_value=8)
def test_import_fluxcal_models(
    mock_cpu,
    mock_get_scans,
    mock_get_band,
    mock_suppress,
    mock_system,
    fluxcal_fields,
    expected_return,
    system_status,
):
    mslist = ["test1.ms", "test2.ms"]
    fluxcal_scans = {"J0408-6545": ["1", "2"]}
    mock_get_band.return_value = "U"
    mock_get_scans.side_effect = [["1"], ["2"]]
    mock_system.return_value = system_status

    result = import_fluxcal_models(
        mslist=mslist,
        fluxcal_fields=fluxcal_fields,
        fluxcal_scans=fluxcal_scans,
        ncpus=2,
        mem_frac=0.9,
    )
    assert result == expected_return
    if fluxcal_fields:
        assert mock_system.call_count >= 1
        assert any(
            "crystalball" in str(call.args[0]) for call in mock_system.call_args_list
        )
    else:
        mock_system.assert_not_called()


def test_import_phasecal_models(dummy_submsname):
    mslist = [
        f"{dummy_submsname}/SUBMSS/test_subms.ms.0002.ms",
        f"{dummy_submsname}/SUBMSS/test_subms.ms.0003.ms",
    ]
    workdir = os.getcwd()
    phasecal_fields, phasecal_scans, phasecal_flux_list = get_phasecals(dummy_submsname)
    dask_client = MagicMock()
    result = import_phasecal_models(
        mslist,
        dask_client,
        phasecal_fields,
        phasecal_scans,
        workdir,
    )
    assert result == 0


def test_import_polcal_models(dummy_submsname):
    mslist = [
        f"{dummy_submsname}/SUBMSS/test_subms.ms.0002.ms",
        f"{dummy_submsname}/SUBMSS/test_subms.ms.0003.ms",
    ]
    workdir = os.getcwd()
    polcal_fields, polcal_scans = get_polcals(dummy_submsname)
    dask_client = MagicMock()
    result = import_polcal_models(
        mslist,
        dask_client,
        polcal_fields,
        polcal_scans,
        workdir,
    )
    assert result == 1


@patch("meersolar.meerpipeline.import_model.time.sleep")
@patch("meersolar.meerpipeline.import_model.drop_cache")
@patch("meersolar.meerpipeline.import_model.import_polcal_models")
@patch("meersolar.meerpipeline.import_model.import_phasecal_models")
@patch("meersolar.meerpipeline.import_model.import_fluxcal_models")
@patch("meersolar.meerpipeline.import_model.get_polcals")
@patch("meersolar.meerpipeline.import_model.get_phasecals")
@patch("meersolar.meerpipeline.import_model.get_fluxcals")
@patch("meersolar.meerpipeline.import_model.get_submsname_scans")
@patch("meersolar.meerpipeline.import_model.correct_missing_col_subms")
def test_import_all_models(
    mock_correct,
    mock_get_scans,
    mock_get_fluxcals,
    mock_get_phasecals,
    mock_get_polcals,
    mock_flux_import,
    mock_phase_import,
    mock_pol_import,
    mock_drop_cache,
    mock_sleep,
):
    msname = "test.ms"
    workdir = "/mock/work"
    mock_get_scans.return_value = (["ms1.ms", "ms2.ms"], ["1", "2"])
    mock_get_fluxcals.return_value = (["J0408-6545"], {"J0408-6545": ["1"]})
    mock_get_phasecals.return_value = (
        ["J1331+3030"],
        {"J1331+3030": ["2"]},
        {"J1331+3030": 1},
    )
    mock_get_polcals.return_value = (["3C286"], {"3C286": ["1"]})
    mock_flux_import.return_value = 0
    mock_phase_import.return_value = 0
    mock_pol_import.return_value = 0
    dask_client = MagicMock()
    flux, phase, pol = import_all_models(msname, dask_client, workdir)
    assert (flux, phase, pol) == (0, 0, 0)
    assert mock_correct.called
    assert mock_get_scans.called
    assert mock_flux_import.called
    assert mock_phase_import.called
    assert mock_pol_import.called


@pytest.mark.parametrize(
    "ms_exists, import_result, expected_msg",
    [
        (True, (0, 0, 0), 0),  # All good
        (True, (1, 0, 0), 1),  # Fluxcal failed
        (False, (1, 1, 1), 1),  # MS doesn't exist
    ],
)
@patch("meersolar.meerpipeline.import_model.import_all_models")
@patch("meersolar.meerpipeline.import_model.save_pid")
@patch("meersolar.meerpipeline.import_model.get_cachedir", return_value="/mock/cache")
@patch("os.makedirs")
@patch("os.path.exists")
@patch("os.getpid", return_value=4321)
@patch("os.system")
@patch("meersolar.meerpipeline.import_model.drop_cache")
@patch("meersolar.meerpipeline.import_model.clean_shutdown")
@patch("time.sleep", return_value=None)
@patch("traceback.print_exc", return_value=None)
def test_main_function(
    mock_traceback,
    mock_sleep,
    mock_shutdown,
    mock_drop,
    mock_system,
    mock_getpid,
    mock_exists,
    mock_makedirs,
    mock_cachedir,
    mock_save_pid,
    mock_import_all,
    ms_exists,
    import_result,
    expected_msg,
):
    msname = "mock.ms"
    workdir = "/mock/work"

    def exists_side_effect(path):
        return path == msname if ms_exists else False

    mock_exists.side_effect = exists_side_effect
    mock_import_all.return_value = import_result
    dask_client = MagicMock()

    msg = main(
        msname=msname,
        workdir=workdir,
        cpu_frac=0.7,
        mem_frac=0.6,
        logfile=None,
        jobid=101,
        start_remote_log=False,
        dask_client=dask_client,
    )
    assert msg == expected_msg


@pytest.mark.parametrize(
    "argv, should_exit",
    [
        (["prog.py"], True),  # Missing required args
        (["prog.py", "mock.ms", "--workdir", "/mock/work"], False),  # Valid
    ],
)
@patch("meersolar.meerpipeline.import_model.main", return_value=0)
@patch("meersolar.meerpipeline.import_model.sys.exit")
@patch("meersolar.meerpipeline.import_model.argparse.ArgumentParser.print_help")
def test_cli(
    mock_print_help,
    mock_exit,
    mock_main,
    argv,
    should_exit,
):
    with patch("sys.argv", argv):
        from meersolar.meerpipeline import import_model

        result = import_model.cli()
        assert result == should_exit
