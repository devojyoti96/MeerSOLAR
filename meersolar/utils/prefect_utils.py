import threading
import time
import pendulum
import asyncio
from prefect import flow, get_run_logger
from prefect.context import get_run_context
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import LogFilter
from prefect.client.schemas.sorting import LogSort


def log_task(task_run_id, logfile="task.log", poll_interval=3):
    """
    Write prefect task log to file
    
    Parameters
    ----------
    task_run_id : str
        Task tun id
    logfile : str
        Log file name
    poll_interval : int, optional
        Update rate (in seconds)
    """
    time.sleep(poll_interval)
    async def run():
        seen_ids = set()
        async with get_client() as client:
            while True:
                log_filter = LogFilter(task_run_id={"any_": [task_run_id]})
                logs = await client.read_logs(
                    log_filter=log_filter,
                    sort=LogSort.TIMESTAMP_ASC
                )
                with open(logfile, "a") as f:
                    for log in logs:
                        if log.id not in seen_ids:
                            ts = pendulum.instance(log.timestamp).to_datetime_string()
                            f.write(f"{ts}.{log.timestamp.microsecond // 1000:03d} | "
                                    f"{log.level:<7} | {log.message}\n")
                            seen_ids.add(log.id)
                # Stop polling if task has finished
                task_run = await client.read_task_run(task_run_id)
                if task_run.state.is_final():
                    break
                await asyncio.sleep(poll_interval)
    asyncio.run(run())

def log_flow(flow_run_id, logfile="flow.log", poll_interval=3):
    """
    Write prefect flow log to file
    
    Parameters
    ----------
    flow_run_id : str
        Flow tun id
    logfile : str
        Log file name
    poll_interval : int, optional
        Update rate (in seconds)
    """
    time.sleep(poll_interval)
    async def run():
        seen_ids = set()
        async with get_client() as client:
            while True:
                flow_run = await client.read_flow_run(flow_run_id)
                state_type = flow_run.state.type

                log_filter = LogFilter(flow_run_id={"any_": [flow_run_id]})
                logs = await client.read_logs(
                    log_filter=log_filter,
                    sort=LogSort.TIMESTAMP_ASC
                )
                with open(logfile, "a") as f:
                    for log in logs:
                        if log.id not in seen_ids:
                            ts = pendulum.instance(log.timestamp).to_datetime_string()
                            f.write(f"{ts}.{log.timestamp.microsecond//1000:03d} | {log.level:<7} | {log.message}\n")
                            seen_ids.add(log.id)
                if state_type in {"COMPLETED", "FAILED", "CANCELLED", "CRASHED"}:
                    break
                await asyncio.sleep(poll_interval)
    asyncio.run(run())
    
