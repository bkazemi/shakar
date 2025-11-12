"""Built-in stdlib functions (print, etc.) registered via shakar_runtime."""

from __future__ import annotations

import asyncio
from typing import List

try:
    from shakar_runtime import register_stdlib, ShkNull, ShkString, ShkNumber, ShkBool, ShakarTypeError
except ImportError:  # running as package
    from .shakar_runtime import register_stdlib, ShkNull, ShkString, ShkNumber, ShkBool, ShakarTypeError

def _render(value):
    if isinstance(value, ShkString):
        return value.value
    if isinstance(value, (ShkNumber, ShkBool)):
        return str(value.value)
    if isinstance(value, ShkNull):
        return "null"
    return str(value)

@register_stdlib("print")
def std_print(_env, args: List[object]) -> ShkNull:
    rendered = [_render(arg) for arg in args]
    print(*rendered)
    return ShkNull()

@register_stdlib("sleep")
def std_sleep(_env, args: List[object]):
    if len(args) != 1:
        raise ShakarTypeError("sleep expects exactly one argument")
    duration = args[0]
    if not isinstance(duration, ShkNumber):
        raise ShakarTypeError("sleep expects a numeric duration")
    milliseconds = max(0.0, float(duration.value))
    seconds = milliseconds / 1000.0

    async def _sleep_coro() -> ShkNull:
        await asyncio.sleep(seconds)
        return ShkNull()

    return _sleep_coro()
