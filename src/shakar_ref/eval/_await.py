from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Iterable

from ..runtime import ShakarRuntimeError

def _wrap_awaitable(value: Any) -> Awaitable[Any]:
    if inspect.isawaitable(value):
        return value

    async def _immediate() -> Any:
        return value

    return _immediate()

def _run_asyncio(coro: Awaitable[Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError: # no active event loop, ok to run
        return asyncio.run(coro)

    raise ShakarRuntimeError("await constructs cannot run inside an active event loop")

async def _await_any_async(entries: list[tuple[dict[str, Any], Any]]) -> tuple[dict[str, Any], Any]:
    tasks: dict[asyncio.Task[Any], dict[str, Any]] = {}

    for arm, value in entries:
        task = asyncio.create_task(_wrap_awaitable(value))
        tasks[task] = arm

    errors: list[Exception] = []

    pending: set[asyncio.Task[Any]] = set(tasks.keys())
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            arm = tasks.pop(task, None)

            try:
                result = task.result()
            except Exception as exc:
                errors.append(exc)
                continue

            for other in pending:
                other.cancel()

            await asyncio.gather(*pending, return_exceptions=True)
            return arm, result

    if errors:
        raise errors[-1]

    raise ShakarRuntimeError("await[any] arms did not produce a value")

async def _await_all_async(entries: list[tuple[dict[str, Any], Any]]) -> list[tuple[dict[str, Any], Any]]:
    coros = [_wrap_awaitable(value) for _, value in entries]
    results = await asyncio.gather(*coros)

    return [(entries[i][0], results[i]) for i in range(len(entries))]

def await_any_entries(entries: list[tuple[dict[str, Any], Any]]) -> tuple[dict[str, Any], Any]:
    return _run_asyncio(_await_any_async(entries))

def await_all_entries(entries: list[tuple[dict[str, Any], Any]]) -> list[tuple[dict[str, Any], Any]]:
    return _run_asyncio(_await_all_async(entries))

def resolve_await_result(value: Any) -> Any:
    if inspect.isawaitable(value):
        return _run_asyncio(_wrap_awaitable(value))

    return value
