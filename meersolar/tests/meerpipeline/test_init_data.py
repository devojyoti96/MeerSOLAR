import pytest
import builtins
from unittest.mock import patch, MagicMock, mock_open
from meersolar.meerpipeline.init_data import *
from requests.exceptions import HTTPError

@pytest.mark.parametrize(
    "record_id, mock_json, raise_exc, expected_result, expect_error",
    [
        (
            "123456",
            {
                "files": [
                    {"links": {"self": "https://zenodo.org/api/files/abc"}, "key": "file1.txt"},
                    {"links": {"self": "https://zenodo.org/api/files/xyz"}, "key": "file2.txt"},
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
        (True, True, True, True),     # file exists and update=True → remove and enqueue
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
        mock_dl.enqueue_file.assert_called_once_with(file_url, path=output_dir, filename=test_file)
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
        mock_download.assert_called_once_with("15691548", update=update, output_dir="/mock/data")
    else:
        mock_download.assert_not_called()
    # Check remote link written
    if remote_link is not None:
        mock_open_func().write.assert_any_call(str(remote_link))
    # Check emails written
    if emails is not None:
        mock_open_func().write.assert_any_call(str(emails))
    
  
@pytest.mark.parametrize(
    "argv, expect_exit, expect_create, expect_init",
    [
        (["script.py"], True, False, False),  # No args → should exit
        (["script.py", "--init"], True, True, True),  # Default init
        (["script.py", "--init", "--datadir", "/custom/data"], True, True, True),  # Custom datadir
        (["script.py", "--init", "--update"], True, True, True),  # Update flag
        (["script.py", "--init", "--remotelink", "http://remote", "--emails", "a@b.com"], True, True, True),
    ],
)
@patch("meersolar.meerpipeline.init_data.init_meersolar_data")
@patch("meersolar.meerpipeline.init_data.create_datadir")
@patch("meersolar.meerpipeline.init_data.sys.exit")
@patch("builtins.print")
def test_main(
    mock_print,
    mock_exit,
    mock_create_datadir,
    mock_init_meersolar_data,
    argv,
    expect_exit,
    expect_create,
    expect_init,
):
    # Patch sys.argv
    with patch.object(sys, "argv", argv):
        from meersolar.meerpipeline import init_data
        try:
            init_data.main()
        except SystemExit:
            pass

    # Check if sys.exit was called only for empty CLI
    if len(argv) == 1:
        mock_exit.assert_called_once_with(1)
    else:
        mock_exit.assert_not_called()

    if expect_create:
        mock_create_datadir.assert_called_once()
    if expect_init:
        mock_init_meersolar_data.assert_called_once()
