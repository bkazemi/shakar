"""Built-in stdlib functions (print, etc.) registered via shakar_runtime."""

from __future__ import annotations

from typing import List

try:
    from shakar_runtime import register_stdlib, ShkNull, ShkString, ShkNumber, ShkBool
except ImportError:  # running as package
    from .shakar_runtime import register_stdlib, ShkNull, ShkString, ShkNumber, ShkBool


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
