import pytest
from unittest.mock import patch, MagicMock, mock_open
from meersolar.meerpipeline.show_msdetails import *


@pytest.mark.parametrize(
    "msname, exists, file_content, expected_lines, expect_exception",
    [
        # Case 1: file does not exist, should raise FileNotFoundError
        ("missing.ms", False, "", [], True),
        # Case 2: file exists, normal content, stops at "Sources:"
        (
            "test.ms",
            True,
            "Line 1\nLine 2\nSources: Done\nLine 4\n",
            ["Line 1\n", "Line 2\n"],
            False,
        ),
    ],
)
@patch("meersolar.meerpipeline.show_msdetails.drop_cache")
@patch("meersolar.meerpipeline.show_msdetails.print")
@patch("meersolar.meerpipeline.show_msdetails.os.system")
@patch("meersolar.meerpipeline.show_msdetails.open", new_callable=mock_open)
@patch("meersolar.meerpipeline.show_msdetails.listobs")
@patch("meersolar.meerpipeline.show_msdetails.os.path.exists")
def test_show_listobs(
    mock_exists,
    mock_listobs,
    mock_open_func,
    mock_system,
    mock_print,
    mock_drop_cache,
    msname,
    exists,
    file_content,
    expected_lines,
    expect_exception,
):
    listfile = msname.split(".ms")[0] + ".listobs"
    mock_exists.return_value = exists
    mock_open_func.return_value.__enter__.return_value.readlines.return_value = (
        file_content.splitlines(True)
    )

    if expect_exception:
        with pytest.raises(
            FileNotFoundError, match=f"Measurement Set not found: {msname}"
        ):
            show_listobs(msname)
        mock_listobs.assert_not_called()
        mock_print.assert_not_called()
    else:
        show_listobs(msname)
        mock_listobs.assert_called_once_with(
            vis=msname, listfile=listfile, verbose=True
        )
        mock_print.assert_called_once_with("".join(expected_lines))
        mock_system.assert_any_call(f"rm -rf {listfile}")
        mock_drop_cache.assert_called_once_with(msname)


@pytest.mark.parametrize(
    "argv_args, expect_exit, expect_call, expected_msname",
    [
        (["script.py"], True, False, None),  # No args → should exit
        (
            ["script.py", "mock.ms"],
            False,
            True,
            "mock.ms",
        ),  # Valid input → call show_listobs
    ],
)
@patch("meersolar.meerpipeline.show_msdetails.show_listobs")
@patch("meersolar.meerpipeline.show_msdetails.argparse.ArgumentParser.print_help")
@patch("meersolar.meerpipeline.show_msdetails.sys.exit")
def test_cli(
    mock_exit,
    mock_print_help,
    mock_show_listobs,
    argv_args,
    expect_exit,
    expect_call,
    expected_msname,
):
    mock_exit.side_effect = SystemExit  # Allow normal test flow after sys.exit

    with patch.object(sys, "argv", argv_args):
        try:
            cli()
        except SystemExit:
            pass

    if expect_exit:
        mock_print_help.assert_called_once()
        assert any(call.args == (1,) for call in mock_exit.call_args_list)
        mock_show_listobs.assert_not_called()
    else:
        mock_show_listobs.assert_called_once_with(expected_msname)
