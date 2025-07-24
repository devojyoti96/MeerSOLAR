import os
import asyncio
import threading
import types
from pathlib import Path
from prefect.client.orchestration import get_client
from prefect.client.schemas.sorting import LogSort
from prefect.client.schemas.filters import LogFilter


async def save_logs_by_task_id(task_run_id, task_name, logfile, poll_interval=5, stop_event=None):
    """
    Fetch and save prefect task logs to a file
    
    Parameters
    ----------
    task_run_id : str
        The Prefect task run ID to monitor
    task_name : str
        Task name
    logfile : str
        Output log file
    poll_interval : int
        How often to check for new logs (in seconds)
    stop_event : threading.Event
        Optional external signal to stop logging
    """
    seen_ids = set()
    while not (stop_event and stop_event.is_set()):
        try:
            async with get_client() as client:
                log_filter = LogFilter(task_run={"any_": [task_run_id]})
                logs = await client.read_logs(log_filter=log_filter,sort=LogSort.TIMESTAMP_ASC)
                with open(logfile, "a") as f:
                    for log in logs:
                        if log.id not in seen_ids:
                            seen_ids.add(log.id)
                            ts = log.timestamp.to_datetime_string()
                            if str(log.task_run_id)==str(task_run_id):
                                f.write(f"{ts} | {task_name} | {log.message}\n")
        except Exception as e:
            with open(logfile, "a") as f:
                f.write(f"Error fetching logs: {e}\n")
        await asyncio.sleep(poll_interval)
 
        
async def save_logs_by_flow_id(flow_run_id, flow_name, logfile, poll_interval=5, stop_event=None):
    """
    Fetch and save prefect flow logs to a file
    
    Parameters
    ----------
    flow_run_id : str
        The Prefect flow run ID to monitor
    flow_name : str
        Flow name
    logfile : str
        Output log file
    poll_interval : int
        How often to check for new logs (in seconds)
    stop_event : threading.Event
        Optional external signal to stop logging
    """
    seen_ids = set()
    while not (stop_event and stop_event.is_set()):
        try:
            async with get_client() as client:
                log_filter = LogFilter(flow_run={"any_": [flow_run_id]})
                logs = await client.read_logs(log_filter=log_filter, sort=LogSort.TIMESTAMP_ASC)

                with open(logfile, "a") as f:
                    for log in logs:
                        if log.id not in seen_ids:
                            seen_ids.add(log.id)
                            # Only include logs without task_run_id = flow-level logs
                            if log.task_run_id is None:
                                ts = log.timestamp.to_datetime_string()
                                f.write(f"{ts} | {flow_name} | {log.message}\n")

        except Exception as e:
            with open(logfile, "a") as f:
                f.write(f"Error fetching flow logs: {e}\n")

        await asyncio.sleep(poll_interval)

def start_log_task_saver(task_run_id, task_name, logfile, poll_interval=5, stop_event=None):
    """
    Start a background thread that saves Prefect task logs to a file continuously.

    Parameters
    ----------
    task_run_id : str
        The Prefect task run ID to monitor
    task_name : str
        Task name
    logfile : str
        Output log file.
    poll_interval : int
        How often to check for new logs (in seconds)
    stop_event : threading.Event
        Optional external signal to stop logging
    """
    def run_loop():
        asyncio.run(save_logs_by_task_id(task_run_id, task_name, logfile, poll_interval, stop_event))

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    return thread


def start_flow_log_saver(flow_run_id, flow_name, logfile, poll_interval=5, stop_event=None):
    """
    Start a background thread that saves Prefect flow logs to a file continuously.
    
    Parameters
    ----------
    flow_run_id : str
        The Prefect flow run ID to monitor
    flow_name : str
        Flow name
    logfile : str
        Output log file
    poll_interval : int
        How often to check for new logs (in seconds)
    stop_event : threading.Event
        Optional external signal to stop logging
    """
    def run_loop():
        asyncio.run(save_logs_by_flow_id(flow_run_id, flow_name, logfile, poll_interval, stop_event))

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    return thread

  
# Exposing only functions
__all__ = [
    name
    for name, obj in globals().items()
    if isinstance(obj, types.FunctionType) and obj.__module__ == __name__
]

