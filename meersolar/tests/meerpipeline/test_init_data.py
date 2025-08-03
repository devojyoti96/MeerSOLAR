import pytest
import builtins
from unittest.mock import patch, MagicMock, mock_open, Mock
from meersolar.meerpipeline.init_data import *
from requests.exceptions import HTTPError


@pytest.mark.parametrize(
    "record_id, mock_json, raise_exc, expected_result, expect_error",
    [
        (
            "123456",
            {
                "files": [
                    {
                        "links": {"self": "https://zenodo.org/api/files/abc"},
                        "key": "file1.txt",
                    },
                    {
                        "links": {"self": "https://zenodo.org/api/files/xyz"},
                        "key": "file2.txt",
                    },
                ]
            },
            None,
            [
                ("https://zenodo.org/api/files/abc", "file1.txt"),
                ("https://zenodo.org/api/files/xyz", "file2.txt"),
            ],
            False,
        ),
        (
            "123457",
            {"files": []},
            None,
            [],
            False,
        ),
        (
            "123458",
            {},
            None,
            [],
            False,
        ),
        (
            "123459",
            None,
            HTTPError("404 Not Found"),
            None,
            True,
        ),
    ],
)
@patch("meersolar.meerpipeline.init_data.requests.get")
def test_get_zenodo_file_urls(
    mock_get, record_id, mock_json, raise_exc, expected_result, expect_error
):
    mock_response = MagicMock()
    if raise_exc:
        mock_response.raise_for_status.side_effect = raise_exc
    else:
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = mock_json
    mock_get.return_value = mock_response
    if expect_error:
        with pytest.raises(HTTPError):
            get_zenodo_file_urls(record_id)
    else:
        result = get_zenodo_file_urls(record_id)
        assert result == expected_result
    mock_get.assert_called_once_with(f"https://zenodo.org/api/records/{record_id}")


@pytest.mark.parametrize(
    "update_flag, existing_file, expect_enqueue, expect_remove",
    [
        (False, False, True, False),  # file doesn't exist → enqueue
        (False, True, False, False),  # file exists and update=False → skip
        (True, True, True, True),  # file exists and update=True → remove and enqueue
    ],
)
@patch("meersolar.meerpipeline.init_data.os.makedirs")
@patch("meersolar.meerpipeline.init_data.os.system")
@patch("meersolar.meerpipeline.init_data.psutil.cpu_count", return_value=4)
@patch("meersolar.meerpipeline.init_data.Downloader")
@patch("meersolar.meerpipeline.init_data.get_zenodo_file_urls")
@patch("meersolar.meerpipeline.init_data.os.path.exists")
@patch("meersolar.meerpipeline.init_data.all_filenames", new=["mockfile.txt"])
def test_download_with_parfive(
    mock_exists,
    mock_get_urls,
    mock_downloader_cls,
    mock_cpu_count,
    mock_system,
    mock_makedirs,
    update_flag,
    existing_file,
    expect_enqueue,
    expect_remove,
):
    record_id = "12345"
    output_dir = "zenodo_download"
    test_file = "mockfile.txt"
    file_url = "https://zenodo.org/api/files/mockfile.txt"

    # Mocking side effects
    def side_effect(path):
        if test_file in path:
            return existing_file
        return False

    mock_exists.side_effect = side_effect
    mock_get_urls.return_value = [(file_url, test_file)]
    mock_dl = MagicMock()
    mock_downloader_cls.return_value = mock_dl
    # Run the function
    download_with_parfive(record_id, update=update_flag, output_dir=output_dir)
    # Assertions
    if expect_enqueue:
        mock_dl.enqueue_file.assert_called_once_with(
            file_url, path=output_dir, filename=test_file
        )
    else:
        mock_dl.enqueue_file.assert_not_called()
    if expect_remove:
        mock_system.assert_called_once_with(f"rm -rf {output_dir}/{test_file}")
    else:
        mock_system.assert_not_called()
    # Should always trigger download
    mock_dl.download.assert_called_once()


@pytest.mark.parametrize(
    "remote_link, emails, update, file_exists, expect_download",
    [
        (None, None, False, True, False),
        ("http://remote", None, False, True, False),
        (None, "test@example.com", False, True, False),
        ("http://remote", "test@example.com", False, True, False),
        (None, None, True, True, True),
        (None, None, False, False, True),
    ],
)
@patch("meersolar.meerpipeline.init_data.download_with_parfive")
@patch("meersolar.meerpipeline.init_data.os.path.exists")
@patch("meersolar.meerpipeline.init_data.open", new_callable=mock_open)
@patch("meersolar.meerpipeline.init_data.os.getlogin", return_value="mockuser")
@patch("meersolar.meerpipeline.init_data.get_cachedir", return_value="/mock/cache")
@patch("meersolar.meerpipeline.init_data.get_datadir", return_value="/mock/data")
@patch("meersolar.meerpipeline.init_data.os.makedirs")
def test_init_meersolar_data(
    mock_makedirs,
    mock_get_datadir,
    mock_get_cachedir,
    mock_getlogin,
    mock_open_func,
    mock_exists,
    mock_download,
    remote_link,
    emails,
    update,
    file_exists,
    expect_download,
):
    import meersolar.meerpipeline.init_data as init_data

    init_data.all_filenames = ["file1.txt", "file2.txt"]

    # Patch os.path.exists logic
    def side_effect(path):
        if "remotelink" in path or "emails" in path:
            return False
        if any(f in path for f in init_data.all_filenames):
            return file_exists
        return False

    mock_exists.side_effect = side_effect
    # Run function
    init_meersolar_data(update=update, remote_link=remote_link, emails=emails)
    # Download check
    if expect_download:
        mock_download.assert_called_once_with(
            "15691548", update=update, output_dir="/mock/data"
        )
    else:
        mock_download.assert_not_called()
    # Check remote link written
    if remote_link is not None:
        mock_open_func().write.assert_any_call(str(remote_link))
    # Check emails written
    if emails is not None:
        mock_open_func().write.assert_any_call(str(emails))


@pytest.mark.parametrize(
    "init_flag, expected_calls",
    [
        # Case 1: init=True, all functions should be called
        (
            True,
            {
                "create_datadir": 1,
                "get_datadir": 1,
                "init_meersolar_data": 1,
                "init_udocker": 1,
                "initialize_wsclean_container": 1,
            },
        ),
        # Case 2: init=False, nothing should be called
        (
            False,
            {
                "create_datadir": 0,
                "get_datadir": 0,
                "init_meersolar_data": 0,
                "init_udocker": 0,
                "initialize_wsclean_container": 0,
            },
        ),
    ],
)
def test_main(init_flag, expected_calls, monkeypatch):
    from meersolar.meerpipeline import init_data

    # Create mock functions
    mocks = {name: Mock(name=f"mock_{name}") for name in expected_calls}

    # Setup return values for the ones that return something
    mocks["get_datadir"].return_value = "/mockdir"

    # Patch all into the module
    for name, mock_func in mocks.items():
        monkeypatch.setattr(init_data, name, mock_func)

    # Call main
    init_data.main(
        init=init_flag,
        datadir="/mockdir",
        update=True,
        link="http://remote.url",
        emails="test@example.com",
        prefect_server=False,
    )

    # Assert calls based on expectation
    (
        mocks["create_datadir"].assert_called_once()
        if expected_calls["create_datadir"]
        else mocks["create_datadir"].assert_not_called()
    )
    (
        mocks["get_datadir"].assert_called_once()
        if expected_calls["get_datadir"]
        else mocks["get_datadir"].assert_not_called()
    )
    (
        mocks["init_meersolar_data"].assert_called_once_with(
            update=True, remote_link="http://remote.url", emails="test@example.com"
        )
        if expected_calls["init_meersolar_data"]
        else mocks["init_meersolar_data"].assert_not_called()
    )
    (
        mocks["init_udocker"].assert_called_once()
        if expected_calls["init_udocker"]
        else mocks["init_udocker"].assert_not_called()
    )
    (
        mocks["initialize_wsclean_container"].assert_called_once()
        if expected_calls["initialize_wsclean_container"]
        else mocks["initialize_wsclean_container"].assert_not_called()
    )


@pytest.mark.parametrize(
    "argv_args, expect_exit",
    [
        # Case 0: no arguments → help and exit
        (["prog"], 1),
        # Case 1: minimal valid call
        (
            [
                "prog",
                "--init",
                "--datadir",
                "/mockdir",
                "--update",
                "--remotelink",
                "http://example.com",
                "--emails",
                "a@b.com",
            ],
            0,
        ),
    ],
)
@patch("meersolar.meerpipeline.init_data.main", return_value=0)
@patch("meersolar.meerpipeline.init_data.sys.exit")
@patch("meersolar.meerpipeline.init_data.argparse.ArgumentParser.print_help")
def test_cli_init_data(mock_print_help, mock_exit, mock_main, argv_args, expect_exit):
    with patch("sys.argv", argv_args):
        from meersolar.meerpipeline import init_data

        result = init_data.cli()
        assert result == expect_exit
