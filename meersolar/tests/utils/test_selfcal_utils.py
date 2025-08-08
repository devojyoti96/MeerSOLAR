import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock
from meersolar.utils.selfcal_utils import *


@patch("meersolar.utils.selfcal_utils.traceback.print_exc")
@patch("meersolar.utils.selfcal_utils.os.system")
@patch("meersolar.utils.selfcal_utils.os.path.exists", return_value=True)
@patch("meersolar.utils.selfcal_utils.glob.glob", return_value=["mock.fits"])
@patch("meersolar.utils.selfcal_utils.make_timeavg_image", return_value="avg.fits")
@patch("meersolar.utils.selfcal_utils.make_freqavg_image", return_value="freqavg.fits")
@patch("meersolar.utils.selfcal_utils.calc_dyn_range", return_value=(1.0, 100.0, 0.01))
@patch("meersolar.utils.selfcal_utils.create_circular_mask", return_value="mask.fits")
@patch("meersolar.utils.selfcal_utils.get_optimal_image_interval", return_value=(1, 1))
@patch("meersolar.utils.selfcal_utils.get_multiscale_bias", return_value=0.6)
@patch("meersolar.utils.selfcal_utils.get_chans_flag", return_value=([10, 11], []))
@patch("meersolar.utils.selfcal_utils.calc_multiscale_scales", return_value=[0, 5, 15])
@patch("meersolar.utils.selfcal_utils.calc_bw_smearing_freqwidth", return_value=1.0)
@patch("meersolar.utils.selfcal_utils.run_wsclean", return_value=0)
@patch("meersolar.utils.selfcal_utils.limit_threads")
@patch("meersolar.utils.selfcal_utils.psutil.cpu_count", return_value=4)
@patch("meersolar.utils.selfcal_utils.psutil.virtual_memory")
@patch("meersolar.utils.selfcal_utils.suppress_output")
@patch("meersolar.utils.selfcal_utils.msmetadata")
@patch("meersolar.utils.selfcal_utils.table")
@patch("casatasks.bandpass")
@patch("casatasks.applycal")
@patch("casatasks.flagdata")
@patch("casatasks.delmod")
@patch("casatasks.flagmanager")
def test_intensity_selfcal(
    mock_flagmanager,
    mock_delmod,
    mock_flagdata,
    mock_applycal,
    mock_bandpass,
    mock_table,
    mock_msmetadata,
    mock_suppress_output,
    mock_virtual_memory,
    mock_cpu_count,
    mock_limit_threads,
    mock_run_wsclean,
    mock_bw_smearing,
    mock_multiscale_scales,
    mock_get_chans_flag,
    mock_get_multiscale_bias,
    mock_get_interval,
    mock_create_mask,
    mock_dyn_range,
    mock_freqavg,
    mock_timeavg,
    mock_glob,
    mock_path_exists,
    mock_os_system,
    mock_traceback,
):
    # Mock memory
    mock_virtual_memory.return_value.available = 4 * 1024**3  # 4 GB

    # Mock msmetadata
    mock_msmd = MagicMock()
    mock_msmd.open.return_value = None
    mock_msmd.timesforspws.return_value = [0, 1, 2, 10]
    mock_msmd.chanfreqs.return_value = np.linspace(100.0, 200.0, 10)
    mock_msmd.meanfreq.return_value = 150.0
    mock_msmetadata.return_value = mock_msmd

    # Mock CASA table object
    tb_mock = MagicMock()
    tb_mock.getcol.return_value = np.array([True])
    tb_mock.putcol.return_value = None
    tb_mock.close.return_value = None
    mock_table.return_value = tb_mock

    logger = MagicMock()
    status, caltable, dynrange, rms, image, model, residual = intensity_selfcal(
        msname="mock.ms",
        logger=logger,
        selfcaldir="selfcal",
        cellsize=1.0,
        imsize=512,
        round_number=1,
        threshold=3.0,
    )

    assert status == 0
    assert "gcal" in caltable
    assert dynrange > 0
    assert rms > 0
    assert image.endswith(".fits")
    assert model.endswith(".fits")
    assert residual.endswith(".fits")
