"""Built-in stdlib functions (print, etc.) registered via shakar_runtime."""

from __future__ import annotations

import asyncio
from typing import List

from .runtime import register_stdlib, ShkNull, ShkString, ShkNumber, ShkBool, ShkValue, ShakarTypeError, ShkObject

def _render(value):
    if isinstance(value, ShkString):
        return value.value

    if isinstance(value, (ShkNumber, ShkBool)):
        return str(value.value)

    if isinstance(value, ShkNull):
        return "null"
    return str(value)

@register_stdlib("print")
def std_print(_frame, args: List[ShkValue]) -> ShkNull:
    rendered = [_render(arg) for arg in args]
    print(*rendered)
    return ShkNull()

@register_stdlib("sleep")
def std_sleep(_frame, args: List[ShkValue]):
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

@register_stdlib("error")
def std_error(_frame, args: List[ShkValue]) -> ShkObject:
    if len(args) < 2 or len(args) > 3:
        raise ShakarTypeError("error(type, message[, data]) expects 2 or 3 arguments")
    type_arg, message_arg, *rest = args

    if not isinstance(type_arg, ShkString) or not isinstance(message_arg, ShkString):
        raise ShakarTypeError("error expects string type and message")
    data = rest[0] if rest else ShkNull()
    slots: dict[str, ShkValue] = {
        '__error__': ShkBool(True),
        'type': type_arg,
        'message': message_arg,
        'data': data,
    }

    return ShkObject(slots)
