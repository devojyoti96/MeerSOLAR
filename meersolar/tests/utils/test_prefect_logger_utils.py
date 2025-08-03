import pytest
import asyncio
import threading
import time
from unittest.mock import patch, AsyncMock, MagicMock, mock_open

from meersolar.utils.prefect_logger_utils import (
    save_logs_by_task_id,
    save_logs_by_flow_id,
    start_log_task_saver,
    start_flow_log_saver,
)


@pytest.mark.asyncio
async def test_save_logs_by_task_id():
    # Mock a log entry
    mock_log = MagicMock()
    mock_log.id = "log1"
    mock_log.message = "Test message"
    mock_log.task_run_id = "task123"
    mock_log.timestamp.to_datetime_string.return_value = "2025-07-24T12:00:00"

    # Mock Prefect client
    mock_client = AsyncMock()
    mock_client.read_logs.return_value = [mock_log]

    with patch(
        "meersolar.utils.prefect_logger_utils.get_client",
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)),
    ):
        with patch("builtins.open", mock_open()) as m:
            stop_event = threading.Event()
            # Run one poll iteration then stop
            task = asyncio.create_task(
                save_logs_by_task_id(
                    "task123",
                    "test-task",
                    "logfile.log",
                    poll_interval=0.1,
                    stop_event=stop_event,
                )
            )
            await asyncio.sleep(0.15)
            stop_event.set()
            await task

            m().write.assert_any_call(
                "2025-07-24T12:00:00 | test-task | Test message\n"
            )


@pytest.mark.asyncio
async def test_save_logs_by_flow_id():
    # Mock a flow-level log
    mock_log = MagicMock()
    mock_log.id = "log2"
    mock_log.message = "Flow log message"
    mock_log.task_run_id = None
    mock_log.timestamp.to_datetime_string.return_value = "2025-07-24T12:00:01"

    mock_client = AsyncMock()
    mock_client.read_logs.return_value = [mock_log]

    with patch(
        "meersolar.utils.prefect_logger_utils.get_client",
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_client)),
    ):
        with patch("builtins.open", mock_open()) as m:
            stop_event = threading.Event()
            task = asyncio.create_task(
                save_logs_by_flow_id(
                    "flow456",
                    "test-flow",
                    "flowfile.log",
                    poll_interval=0.1,
                    stop_event=stop_event,
                )
            )
            await asyncio.sleep(0.15)
            stop_event.set()
            await task

            m().write.assert_any_call(
                "2025-07-24T12:00:01 | test-flow | Flow log message\n"
            )


def test_start_log_task_saver():
    stop_event = threading.Event()

    with patch(
        "meersolar.utils.prefect_logger_utils.save_logs_by_task_id",
        new_callable=AsyncMock,
    ) as mock_async:
        thread = start_log_task_saver(
            "taskid",
            "test-task",
            "logfile.log",
            poll_interval=0.1,
            stop_event=stop_event,
        )
        assert isinstance(thread, threading.Thread)
        time.sleep(0.2)  # Let thread spin up
        stop_event.set()
        thread.join(timeout=1)
        mock_async.assert_called_once()


def test_start_flow_log_saver():
    stop_event = threading.Event()

    with patch(
        "meersolar.utils.prefect_logger_utils.save_logs_by_flow_id",
        new_callable=AsyncMock,
    ) as mock_async:
        thread = start_flow_log_saver(
            "flowid",
            "test-flow",
            "flowfile.log",
            poll_interval=0.1,
            stop_event=stop_event,
        )
        assert isinstance(thread, threading.Thread)
        time.sleep(0.2)
        stop_event.set()
        thread.join(timeout=1)
        mock_async.assert_called_once()
