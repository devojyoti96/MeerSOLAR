import pytest
from meersolar.utils.all_depend import *


@pytest.fixture
def dummy_caltables(tmp_path):
    """
    Provide a list of dummy CASA caltables for testing.
    They are shallow copies of a real reference caltable.
    """
    # Path to a known valid CASA caltable
    ref_caltable = (
        "/media/devojyoti/Data1/meersolar/meersolar/tests/data/test_caltable.gcal"
    )
    if not os.path.exists(ref_caltable):
        pytest.skip("Reference caltable is not found")
    # Create two dummy copies
    cal1 = tmp_path / "cal1.K"
    cal2 = tmp_path / "cal2.K"
    shutil.copytree(ref_caltable, cal1)
    shutil.copytree(ref_caltable, cal2)
    return [str(cal1), str(cal2)]


@pytest.fixture
def dummy_caltable(tmp_path):
    ref_caltable = (
        "/media/devojyoti/Data1/meersolar/meersolar/tests/data/test_caltable.gcal"
    )
    if not os.path.exists(ref_caltable):
        pytest.skip("Caltable is not found")
    return ref_caltable


@pytest.fixture
def dummy_msname(tmp_path):
    ref_msname = "/media/devojyoti/Data1/meersolar/meersolar/tests/data/test_ms.ms"
    if not os.path.exists(ref_msname):
        pytest.skip("Reference ms is not found")
    return ref_msname


@pytest.fixture
def dummy_submsname(tmp_path):
    ref_submsname = (
        "/media/devojyoti/Data1/meersolar/meersolar/tests/data/test_subms.ms"
    )
    if not os.path.exists(ref_submsname):
        pytest.skip("Reference subms is not found")
    return ref_submsname


@pytest.fixture
def mock_npy_data():
    # Dummy phase calibrator name and flux
    phasecals = ["J0431+2037"]
    fluxes = [3.412]
    return np.array([phasecals, fluxes])
