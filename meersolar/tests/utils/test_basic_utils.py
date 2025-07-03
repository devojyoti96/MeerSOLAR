import pytest
from meersolar.utils.basic_utils import *


def test_suppress_casa_output_fd():
    with suppress_casa_output():
        os.write(1, b"This should not appear\n")
        os.write(2, b"This error should not appear\n")


def test_get_datadir(mocker):
    dummy_path = "/fake/datadir"

    class DummyFiles:
        def joinpath(self, sub):
            assert sub == "data"
            return dummy_path

    mocker.patch("importlib.resources.files", return_value=DummyFiles())
    makedirs_mock = mocker.patch("os.makedirs")
    result = get_datadir()
    assert result == dummy_path
    makedirs_mock.assert_called_once_with(dummy_path, exist_ok=True)


def test_get_meersolar_cachedir(mocker):
    dummy_home = "/dummy/home"
    dummy_user = "dummyuser"
    expected_cachedir = f"{dummy_home}/.meersolar"
    mocker.patch.dict("os.environ", {"HOME": dummy_home})
    mocker.patch("os.getlogin", return_value=dummy_user)
    makedirs_mock = mocker.patch("os.makedirs")
    cachedir = get_meersolar_cachedir()
    assert cachedir == expected_cachedir
    makedirs_mock.assert_any_call(expected_cachedir, exist_ok=True)
    makedirs_mock.assert_any_call(f"{expected_cachedir}/pids", exist_ok=True)


@pytest.mark.parametrize(
    "lst,target_chunk_size,result",
    [
        ([1, 2, 3, 4, 5], 2, [[1, 2, 3], [4, 5]]),
        ([1, 2, 3, 4, 5], 3, [[1, 2, 3], [4, 5]]),
        ([1, 2, 3], 1, [[1], [2], [3]]),
        ([1], 1, [[1]]),
        ([], 1, [[]]),
    ],
)
def test_split_into_chunks(lst, target_chunk_size, result):
    assert split_into_chunks(lst, target_chunk_size) == result


@pytest.mark.parametrize(
    "timestamps,expected",
    [
        (
            ["2014-05-01T00:00:00", "2014-05-01T01:00:00", "2014-05-01T02:00:00"],
            "2014-05-01T01:00:00",
        ),
        (
            ["2024-02-29T00:00:00", "2024-02-28T23:59:59", "2024-03-01T00:00:01"],
            "2024-02-29T08:00:00",
        ),
        ([], ""),
    ],
)
def test_average_timestamp(timestamps, expected):
    assert average_timestamp(timestamps) == expected


@pytest.mark.parametrize("n,base,result", [(10, 2, 12), (16, 3, 18), (19, 5, 20)])
def test_ceil_to_multiple(n, base, result):
    assert ceil_to_multiple(n, base) == result


@pytest.mark.parametrize(
    "ra1, dec1, ra2, dec2, expected",
    [
        (10.0, -30.0, 10.0, -30.0, 0.0),
        (0.0, 0.0, 90.0, 0.0, 90.0),
        (0.0, 0.0, 180.0, 0.0, 180.0),
        (10.0, 20.0, 30.0, 40.0, 26.33),
        (120.0, -45.0, 130.0, -44.0, 7.2),
    ],
)
def test_angular_separation_equatorial(ra1, dec1, ra2, dec2, expected):
    result = angular_separation_equatorial(ra1, dec1, ra2, dec2)
    assert result == expected


@pytest.mark.parametrize(
    "timestamp, date_format",
    [
        ("2024/06/30/12:34:56", 0),
        ("2024-06-30T12:34:56", 1),
        ("2024-06-30 12:34:56", 2),
        ("2024_06_30_12_34_56", 3),
    ],
)
def test_timestamp_to_mjdsec(timestamp, date_format):
    expected = 5226467696.0
    result = timestamp_to_mjdsec(timestamp, date_format)
    assert result == expected


def test_mjdsec_to_timestamp():
    assert mjdsec_to_timestamp(5226467696.0, 0) == "2024-06-30T12:34:56.00"
    assert mjdsec_to_timestamp(5226467696.0, 1) == "2024/06/30/12:34:56.00"
    assert mjdsec_to_timestamp(5226467696.0, 2) == "2024-06-30 12:34:56.00"
