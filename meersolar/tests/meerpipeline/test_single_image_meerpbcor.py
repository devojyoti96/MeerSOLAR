import pytest
from unittest.mock import patch, MagicMock
from meersolar.meerpipeline.single_image_meerpbcor import *


@pytest.fixture
def fake_stokes_cube():
    """Returns a fake 4D numpy array with 4 Stokes components"""
    data = np.zeros((4, 1, 128, 128), dtype=np.float32)
    data[0, 0] = 1.0  # I
    data[1, 0] = 0.1  # Q
    data[2, 0] = 0.2  # U
    data[3, 0] = 0.3  # V
    return data


@pytest.mark.parametrize(
    "ctype_key, stokesaxis, expected_index",
    [
        ("CTYPE3", 3, [0, 1, 2, 3]),  # STOKES along axis 3
        ("CTYPE4", 4, [0, 1, 2, 3]),  # STOKES along axis 4
    ],
)
@patch("meersolar.meerpipeline.single_image_meerpbcor.fits.getheader")
@patch("meersolar.meerpipeline.single_image_meerpbcor.fits.getdata")
def test_get_IQUV(
    mock_getdata,
    mock_getheader,
    fake_stokes_cube,
    ctype_key,
    stokesaxis,
    expected_index,
):
    # Setup mock FITS header
    header = {ctype_key: "STOKES"}
    mock_getheader.return_value = header

    # Rearrange cube to match dimensionality based on axis type
    if stokesaxis == 3:
        fake_data = np.zeros((1, 4, 128, 128), dtype=np.float32)
        for i in range(4):
            fake_data[0, i] = fake_stokes_cube[i, 0]
    else:
        fake_data = np.zeros((4, 1, 128, 128), dtype=np.float32)
        for i in range(4):
            fake_data[i, 0] = fake_stokes_cube[i, 0]

    mock_getdata.return_value = fake_data

    result = get_IQUV("fake_stokes.fits")

    assert set(result.keys()) == {"I", "Q", "U", "V"}
    np.testing.assert_array_equal(
        result["I"], fake_data[0, 0] if stokesaxis == 3 else fake_data[0, 0]
    )
    np.testing.assert_array_equal(
        result["Q"], fake_data[0, 1] if stokesaxis == 3 else fake_data[1, 0]
    )
    np.testing.assert_array_equal(
        result["U"], fake_data[0, 2] if stokesaxis == 3 else fake_data[2, 0]
    )
    np.testing.assert_array_equal(
        result["V"], fake_data[0, 3] if stokesaxis == 3 else fake_data[3, 0]
    )


@pytest.mark.parametrize(
    "ctype_key, stokesaxis, shape, expected_slices",
    [
        ("CTYPE3", 3, (1, 4, 128, 128), [(0, 0), (0, 1), (0, 2), (0, 3)]),
        ("CTYPE4", 4, (4, 1, 128, 128), [(0, 0), (1, 0), (2, 0), (3, 0)]),
        ("", 1, (1, 1, 128, 128), [(0, 0)]),
    ],
)
@patch("meersolar.meerpipeline.single_image_meerpbcor.fits.writeto")
def test_put_IQUV(mock_writeto, ctype_key, stokesaxis, shape, expected_slices):
    # Create a mock header
    header = {
        "NAXIS": len(shape),
        **{f"NAXIS{i+1}": s for i, s in enumerate(reversed(shape))},
    }
    if ctype_key:
        header[ctype_key] = "STOKES"

    # Create dummy Stokes data
    stokes = {
        "I": np.ones((128, 128), dtype=np.float32),
        "Q": np.ones((128, 128), dtype=np.float32) * 0.1,
        "U": np.ones((128, 128), dtype=np.float32) * 0.2,
        "V": np.ones((128, 128), dtype=np.float32) * 0.3,
    }

    filename = "dummy_output.fits"
    result = put_IQUV(filename, stokes, header)

    # Check writeto was called
    mock_writeto.assert_called_once()
    args, kwargs = mock_writeto.call_args
    assert args[0] == filename
    assert "data" in kwargs
    assert "header" in kwargs
    data = kwargs["data"]
    assert data.shape == shape

    if stokesaxis == 3:
        np.testing.assert_array_equal(data[0, 0], stokes["I"])
        np.testing.assert_array_equal(data[0, 1], stokes["Q"])
        np.testing.assert_array_equal(data[0, 2], stokes["U"])
        np.testing.assert_array_equal(data[0, 3], stokes["V"])
    elif stokesaxis == 4:
        np.testing.assert_array_equal(data[0, 0], stokes["I"])
        np.testing.assert_array_equal(data[1, 0], stokes["Q"])
        np.testing.assert_array_equal(data[2, 0], stokes["U"])
        np.testing.assert_array_equal(data[3, 0], stokes["V"])
    else:
        np.testing.assert_array_equal(data[0, 0], stokes["I"])

    assert result == filename


def test_get_brightness():
    # Create mock stokes maps (2x2 image)
    I = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    Q = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    U = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    V = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)

    stokes = {"I": I, "Q": Q, "U": U, "V": V}
    B = get_brightness(stokes)

    # Shape checks
    assert B.shape == (2, 2, 2, 2)

    # Value checks at (0,0)
    assert np.allclose(B[0, 0, 0, 0], I[0, 0] - Q[0, 0])  # XX
    assert np.allclose(B[0, 0, 0, 1], U[0, 0] - 1j * V[0, 0])  # XY
    assert np.allclose(B[0, 0, 1, 0], U[0, 0] + 1j * V[0, 0])  # YX
    assert np.allclose(B[0, 0, 1, 1], I[0, 0] + Q[0, 0])  # YY

    # Check dtype
    assert B.dtype == np.complex64


def test_make_stokes():
    # Create 2x2 Stokes images
    I = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    Q = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    U = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    V = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)

    # Construct brightness matrix
    XX = I - Q
    YY = I + Q
    XY = U - 1j * V
    YX = U + 1j * V

    B = np.array([[XX, XY], [YX, YY]])  # shape: (2,2,H,W)
    B = np.transpose(B, (0, 1, 2, 3)).astype("complex64")  # ensure shape: (2,2,H,W)

    # Run function
    stokes_out = make_stokes(B)

    # Assertions
    assert np.allclose(stokes_out["I"], I)
    assert np.allclose(stokes_out["Q"], Q)
    assert np.allclose(stokes_out["U"], U)
    assert np.allclose(stokes_out["V"], V)


@pytest.mark.parametrize(
    "ctype_key, crval, cdelt, expected_band, beam_file",
    [
        ("CTYPE3", 1.4e9, 1e6, "L", "MeerKAT_antavg_Lband.npz"),
        ("CTYPE4", 7e8, 1e6, "U", "MeerKAT_antavg_Uband.npz"),
    ],
)
@patch("meersolar.meerpipeline.single_image_meerpbcor.datadir", "/mockdata")
@patch("meersolar.meerpipeline.single_image_meerpbcor.fits.getheader")
@patch("meersolar.meerpipeline.single_image_meerpbcor.np.load")
def test_load_beam(
    mock_np_load,
    mock_getheader,
    ctype_key,
    crval,
    cdelt,
    expected_band,
    beam_file,
):
    # Mock FITS header
    mock_header = {
        ctype_key: "FREQ",
        "CRVAL3" if ctype_key == "CTYPE3" else "CRVAL4": crval,
        "CDELT3" if ctype_key == "CTYPE3" else "CDELT4": cdelt,
    }
    mock_getheader.return_value = mock_header

    # Mock beam data
    mock_data = {
        "freqs": np.linspace(700, 1700, 10),  # in MHz
        "coords": np.array([[0, 0], [1, 1]]),
        "beams": np.ones((2, 10, 64, 64), dtype=np.complex64),
    }
    mock_npz = MagicMock()
    mock_npz.__enter__.return_value = mock_data
    mock_np_load.return_value = mock_data

    coords, beam = load_beam("fake.fits")

    assert coords.shape[1] == 2
    assert beam.shape[0] == 2
    assert beam.dtype == np.complex64
    mock_np_load.assert_called_once_with(f"/mockdata/{beam_file}", mmap_mode="r")


@patch("meersolar.meerpipeline.single_image_meerpbcor.fits.getheader")
@patch("meersolar.meerpipeline.single_image_meerpbcor.WCS")
def test_get_radec_grid(mock_wcs, mock_getheader):
    # Mock header
    mock_hdr = {"NAXIS1": 64, "NAXIS2": 32}
    mock_getheader.return_value = mock_hdr

    # Mock WCS
    mock_world = MagicMock()
    mock_world.ra.deg = np.random.rand(32, 64)
    mock_world.dec.deg = np.random.rand(32, 64)
    mock_celestial = MagicMock()
    mock_celestial.pixel_to_world.return_value = mock_world
    mock_wcs.return_value.celestial = mock_celestial

    ra, dec = get_radec_grid("fake.fits")

    assert ra.shape == (32, 64)
    assert dec.shape == (32, 64)
    mock_getheader.assert_called_once_with("fake.fits")
    mock_wcs.assert_called_once_with(mock_hdr)
    mock_celestial.pixel_to_world.assert_called()


@patch("meersolar.meerpipeline.single_image_meerpbcor.fits.getheader")
def test_get_pointingcenter_radec(mock_getheader):
    # Mock header values
    mock_header = {
        "CRVAL1": 123.456,
        "CRVAL2": -45.678,
        "NAXIS1": 64,
        "NAXIS2": 32,
    }
    mock_getheader.return_value = mock_header

    ra, dec = get_pointingcenter_radec("mock_image.fits")

    assert ra == 123.456
    assert dec == -45.678
    mock_getheader.assert_called_once_with("mock_image.fits")


@pytest.mark.parametrize(
    "ra_deg, dec_deg, ra0_deg, dec0_deg, expected_l, expected_m",
    [
        # Centered at phase center → (l, m) = (0, 0)
        (np.array([[180.0]]), np.array([[45.0]]), 180.0, 45.0, 0.0, 0.0),
        # Offset in RA by 1° at dec=0 → l ≈ sin(1°), m ≈ 0
        (np.array([[1.0]]), np.array([[0.0]]), 0.0, 0.0, np.sin(np.radians(1.0)), 0.0),
        # Offset in Dec by 1° at ra=0 → m ≈ sin(1°), l ≈ 0
        (np.array([[0.0]]), np.array([[1.0]]), 0.0, 0.0, 0.0, np.sin(np.radians(1.0))),
    ],
)
def test_radec_to_lm(ra_deg, dec_deg, ra0_deg, dec0_deg, expected_l, expected_m):
    l, m = radec_to_lm(ra_deg, dec_deg, ra0_deg, dec0_deg)
    assert np.allclose(l, expected_l, atol=1e-8)
    assert np.allclose(m, expected_m, atol=1e-8)


@pytest.mark.parametrize(
    "obs_time, ra_deg, dec_deg, expected_deg",
    [
        # Zenith observation → parallactic angle ~ 0 (or undefined but returns finite)
        ("2024-01-01T12:00:00", 180.0, -MEERLAT, -105.99),
        # Source rising east → angle changes rapidly, here just checking finite result
        ("2024-01-01T18:00:00", 90.0, 0.0, None),
        # Source setting west
        ("2024-01-01T06:00:00", 270.0, 0.0, None),
    ],
)
def test_get_parallactic_angle(obs_time, ra_deg, dec_deg, expected_deg):
    angle = get_parallactic_angle(obs_time, ra_deg, dec_deg)
    assert np.isfinite(angle)
    if expected_deg is not None:
        assert angle == expected_deg


@patch("meersolar.meerpipeline.single_image_meerpbcor.RectBivariateSpline")
def test_get_beam_interpolator(mock_spline):
    # Setup dummy Jones matrix with 4 components (2x2), shape: (4, 64, 64)
    jones = np.random.randn(4, 64, 64) + 1j * np.random.randn(4, 64, 64)
    coords = np.linspace(-1, 1, 64)
    # Configure the spline mock to return a unique mock for each call
    spline_mocks = [MagicMock(name=f"spline_{i}") for i in range(8)]
    mock_spline.side_effect = spline_mocks
    result = get_beam_interpolator(jones, coords)
    # Check all 8 interpolators returned
    assert len(result) == 8
    assert all(r in spline_mocks for r in result)
    # Ensure RectBivariateSpline was called 8 times
    assert mock_spline.call_count == 8
    # Optional: check the arguments for the first call
    x, y, z = (
        mock_spline.call_args_list[0][1]["x"],
        mock_spline.call_args_list[0][1]["y"],
        mock_spline.call_args_list[0][1]["z"],
    )
    assert np.allclose(x, coords)
    assert np.allclose(y, coords)
    assert z.shape == (64, 64)


def test_apply_parallactic_rotation():
    # Input Jones matrix: shape (4, H, W), here (4, 2, 2) for simplicity
    j00 = np.ones((2, 2), dtype="complex64")
    j01 = np.zeros((2, 2), dtype="complex64")
    j10 = np.zeros((2, 2), dtype="complex64")
    j11 = np.ones((2, 2), dtype="complex64")
    jones = np.stack([j00, j01, j10, j11], axis=0)
    rotated = apply_parallactic_rotation(jones, 90)
    assert np.allclose(rotated[0], 0)
    assert np.allclose(rotated[1], -1)
    assert np.allclose(rotated[2], 1)
    assert np.allclose(rotated[3], 0)


@patch(
    "meersolar.meerpipeline.single_image_meerpbcor.get_parallactic_angle",
    return_value=30.0,
)
@patch(
    "meersolar.meerpipeline.single_image_meerpbcor.get_pointingcenter_radec",
    return_value=(100.0, 30.0),
)
@patch(
    "meersolar.meerpipeline.single_image_meerpbcor.get_radec_grid",
    return_value=(np.zeros((2, 2)), np.zeros((2, 2))),
)
@patch(
    "meersolar.meerpipeline.single_image_meerpbcor.radec_to_lm",
    return_value=(np.zeros((2, 2)), np.zeros((2, 2))),
)
@patch("meersolar.meerpipeline.single_image_meerpbcor.load_beam")
@patch("meersolar.meerpipeline.single_image_meerpbcor.get_beam_interpolator")
@patch("meersolar.meerpipeline.single_image_meerpbcor.Parallel")
@patch("meersolar.meerpipeline.single_image_meerpbcor.fits.getheader")
def test_get_image_beam(
    mock_getheader,
    mock_parallel,
    mock_get_beam_interpolator,
    mock_load_beam,
    mock_radec_to_lm,
    mock_get_radec_grid,
    mock_get_pointingcenter,
    mock_get_parallactic,
):
    # Mock header
    mock_getheader.return_value = {
        "CTYPE3": "FREQ",
        "CRVAL3": 1420000000.0,
        "DATE-OBS": "2021-01-01T00:00:00",
        "NAXIS1": 2,
        "NAXIS2": 2,
    }

    # Mock beam
    coords = np.linspace(-1, 1, 10)
    beam_array = np.random.rand(4, 10, 10).astype(np.complex64)
    mock_load_beam.return_value = (coords, beam_array)

    # Mock interpolators (each returns (2, 2))
    interpolators = [MagicMock(return_value=np.zeros((2, 2))) for _ in range(8)]
    mock_get_beam_interpolator.return_value = interpolators

    # Mock Parallel
    mock_parallel.return_value.__enter__.return_value = lambda x: [np.zeros((2, 2))] * 8

    # Run
    beam = get_image_beam(
        image_file="mock_image.fits",
        pbdir="/tmp/pb",
        save_beam=False,
        band="L",
        apply_parang=True,
        n_cpu=2,
        verbose=False,
    )

    assert beam.shape == (2, 2, 2, 2)
    assert beam.dtype == np.complex64


@pytest.mark.parametrize("apply_parang", [True, False])
@patch(
    "meersolar.meerpipeline.single_image_meerpbcor.fits.getheader",
    return_value={"CTYPE3": "FREQ", "CRVAL3": 1.4e9, "DATE-OBS": "2021-01-01T12:00:00"},
)
@patch(
    "meersolar.meerpipeline.single_image_meerpbcor.put_IQUV",
    return_value="/mock/path/mock_image_pbcor.fits",
)
@patch("meersolar.meerpipeline.single_image_meerpbcor.make_stokes")
@patch("meersolar.meerpipeline.single_image_meerpbcor.get_brightness")
@patch("meersolar.meerpipeline.single_image_meerpbcor.get_IQUV")
@patch("meersolar.meerpipeline.single_image_meerpbcor.get_image_beam")
@patch(
    "meersolar.meerpipeline.single_image_meerpbcor.os.path.basename",
    return_value="mock_image.fits",
)
def test_get_pbcor_image(
    mock_basename,
    mock_get_image_beam,
    mock_get_IQUV,
    mock_get_brightness,
    mock_make_stokes,
    mock_put_IQUV,
    mock_getheader,
    apply_parang,
):
    # --- Setup mock return values ---
    mock_beam = np.ones((2, 2, 2, 2), dtype=np.complex64)
    for i in range(2):
        for j in range(2):
            mock_beam[i, j] = np.array([[1 + 1j, 2], [3, 4 + 1j]], dtype=np.complex64)
    mock_get_image_beam.return_value = mock_beam
    mock_I = np.ones((2, 2), dtype=np.float32)
    mock_I[0, 0] = 2.0
    mock_I[0, 1] = 3.0
    stokes_dict = {"I": mock_I, "Q": mock_I, "U": mock_I, "V": mock_I}
    mock_get_IQUV.return_value = stokes_dict
    B_mock = np.ones((2, 2, 2, 2), dtype=np.complex64)
    mock_get_brightness.return_value = B_mock
    mock_make_stokes.return_value = stokes_dict
    # --- Run the function ---
    result = get_pbcor_image(
        image_file="mock_image.fits",
        pbdir="/tmp/pb",
        pbcor_dir="/mock/path",
        save_beam=False,
        band="L",
        apply_parang=apply_parang,
        n_cpu=2,
        verbose=True,
    )
    # --- Assertions ---
    assert result == "/mock/path/mock_image_pbcor.fits"
    mock_get_image_beam.assert_called_once()
    mock_get_IQUV.assert_called_once()
    mock_get_brightness.assert_called_once()
    mock_make_stokes.assert_called_once()
    mock_put_IQUV.assert_called_once()


@pytest.mark.parametrize(
    "imagename_exists, pbdir_given, pbcor_success, expected_msg",
    [
        (True, True, True, 0),  # Success case
        (True, True, False, 1),  # get_pbcor_image returns None
        (True, False, False, 1),  # pbdir not provided
        (False, True, False, 1),  # imagename doesn't exist
    ],
)
@patch("meersolar.meerpipeline.single_image_meerpbcor.get_pbcor_image")
@patch("meersolar.meerpipeline.single_image_meerpbcor.save_pid")
@patch(
    "meersolar.meerpipeline.single_image_meerpbcor.get_cachedir",
    return_value="/mock/cache",
)
@patch("os.makedirs")
@patch("os.path.exists")
@patch("os.getpid", return_value=1234)
def test_main_function(
    mock_getpid,
    mock_exists,
    mock_makedirs,
    mock_get_cachedir,
    mock_save_pid,
    mock_get_pbcor_image,
    imagename_exists,
    pbdir_given,
    pbcor_success,
    expected_msg,
):
    imagename = "mock_image.fits"
    pbdir = "pbdir" if pbdir_given else ""
    pbcor_dir = "pbcor"

    def exists_side_effect(path):
        if path == imagename:
            return imagename_exists
        elif pbcor_success and path.endswith(".pbcor.fits"):
            return True
        return False

    mock_exists.side_effect = exists_side_effect
    mock_get_pbcor_image.return_value = (
        "mock_image.pbcor.fits" if pbcor_success else None
    )

    msg = main(
        imagename=imagename,
        pbdir=pbdir,
        pbcor_dir=pbcor_dir,
        save_beam=True,
        band="L",
        apply_parang=True,
        verbose=True,
        ncpu=2,
        jobid=42,
    )

    assert msg == expected_msg


@pytest.mark.parametrize(
    "argv, expected_exit_code",
    [
        (["prog.py"], 1),  # No arguments, should print help and exit
        (
            ["prog.py", "mock_image.fits", "--pbdir", "pb", "--pbcor_dir", "pc"],
            0,
        ),  # Success case
    ],
)
@patch("meersolar.meerpipeline.single_image_meerpbcor.main", return_value=0)
@patch("meersolar.meerpipeline.single_image_meerpbcor.sys.exit")
@patch(
    "meersolar.meerpipeline.single_image_meerpbcor.argparse.ArgumentParser.print_help"
)
def test_cli_function(
    mock_print_help,
    mock_exit,
    mock_main,
    argv,
    expected_exit_code,
):
    with patch("sys.argv", argv):
        from meersolar.meerpipeline import single_image_meerpbcor

        result = single_image_meerpbcor.cli()
        assert result == expected_exit_code
