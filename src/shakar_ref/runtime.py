from __future__ import annotations

import importlib
import subprocess
from typing import Callable, Dict, List, Optional, Tuple
from .tree import Node
from .types import (
    ShkNull, ShkNumber, ShkString, ShkRegex, ShkBool, ShkArray, ShkObject, ShkCommand, ShkPath,
    ShkSelector, SelectorIndex, SelectorSlice, SelectorPart,
    ShkFn, ShkDecorator, DecoratorConfigured, DecoratorContinuation,
    BoundMethod, BuiltinMethod, Descriptor, StdlibFunction, DeferEntry,
    ShkValue, EvalResult, DotValue, Frame, ShkType, ShkOptional, ShkUnion,
    ShakarRuntimeError, ShakarTypeError, ShakarArityError, ShakarKeyError,
    ShakarIndexError, ShakarMethodNotFound, CommandError, ShakarAssertionError,
    ShakarReturnSignal, ShakarBreakSignal, ShakarContinueSignal,
    Method, MethodRegistry, Builtins,
    is_shk_value, _ensure_shk_value, StdlibFn
)

__all__ = [
    "Frame",
    "ShkNull",
    "ShkNumber",
    "ShkString",
    "ShkRegex",
    "ShkBool",
    "ShkArray",
    "ShkObject",
    "ShkCommand",
    "ShkPath",
    "ShkSelector",
    "SelectorIndex",
    "SelectorSlice",
    "SelectorPart",
    "ShkFn",
    "ShkDecorator",
    "DecoratorConfigured",
    "DecoratorContinuation",
    "BoundMethod",
    "BuiltinMethod",
    "Descriptor",
    "StdlibFunction",
    "DeferEntry",
    "ShkValue",
    "EvalResult",
    "DotValue",
    "ShkType",
    "ShkOptional",
    "ShkUnion",
    "ShakarRuntimeError",
    "ShakarTypeError",
    "ShakarArityError",
    "ShakarKeyError",
    "ShakarIndexError",
    "ShakarMethodNotFound",
    "CommandError",
    "ShakarAssertionError",
    "ShakarReturnSignal",
    "ShakarBreakSignal",
    "ShakarContinueSignal",
    "Method",
    "MethodRegistry",
    "Builtins",
    "is_shk_value",
    "_ensure_shk_value",
    "StdlibFn",
    "init_stdlib",
]

_STDLIB_INITIALIZED = False

def init_stdlib() -> None:
    """Load stdlib modules (idempotent) so register_stdlib hooks run."""
    global _STDLIB_INITIALIZED

    if _STDLIB_INITIALIZED:
        return

    for module_name in ("shakar_ref.stdlib",):
        try:
            importlib.import_module(module_name)
            _STDLIB_INITIALIZED = True
            return
        except ModuleNotFoundError:
            continue

    stdlib_expect_arity: Dict[str, int] = {}

    known_arity: Dict[Tuple[str, str], Tuple[str, int]] = {
        ("array","map"):     ("exact", 1),
        ("array","filter"):  ("exact", 1),
        ("array","zipWith"): ("exact", 2),
        ("string","len"):    ("exact", 0),
        ("string","trim"):   ("exact", 0),
        ("string","lower"):  ("exact", 0),
        ("string","upper"):  ("exact", 0),
        ("string","hasPrefix"): ("exact", 1),
        ("string","hasSuffix"): ("exact", 1),
        ("string","isAscii"): ("exact", 0),
        ("object","items"):  ("exact", 0),
    }

def register_method(registry: MethodRegistry, name: str):
    def dec(fn: Callable[..., ShkValue]):
        registry[name] = fn
        return fn

    return dec

def register_array(name: str):
    return register_method(Builtins.array_methods, name)

def register_string(name: str):
    return register_method(Builtins.string_methods, name)

def register_regex(name: str):
    return register_method(Builtins.regex_methods, name)

def register_object(name: str):
    return register_method(Builtins.object_methods, name)

def register_command(name: str):
    return register_method(Builtins.command_methods, name)

def register_path(name: str):
    return register_method(Builtins.path_methods, name)

def register_stdlib(name: str, *, arity: Optional[int] = None):
    def dec(fn: StdlibFn):
        Builtins.stdlib_functions[name] = StdlibFunction(fn=fn, arity=arity)
        return fn

    return dec

def _string_expect_arity(method: str, args: List[ShkValue], expected: int) -> None:
    if len(args) != expected:
        raise ShakarArityError(f"string.{method} expects {expected} argument(s); got {len(args)}")

def _string_arg(method: str, arg: ShkValue) -> str:
    if isinstance(arg, ShkString):
        return arg.value

    raise ShakarTypeError(f"string.{method} expects a string argument")

def _regex_expect_arity(method: str, args: List[ShkValue], expected: int) -> None:
    if len(args) != expected:
        raise ShakarArityError(f"regex.{method} expects {expected} argument(s); got {len(args)}")

def _regex_arg(method: str, arg: ShkValue) -> str:
    if isinstance(arg, ShkString):
        return arg.value

    raise ShakarTypeError(f"regex.{method} expects a string argument")

@register_string("join")
def _string_join(_frame: Frame, recv: ShkString, args: List[ShkValue]) -> ShkString:
    items: List[ShkValue]
    if len(args) == 1 and isinstance(args[0], ShkArray):
        items = args[0].items
    else:
        items = args

    from .eval.common import stringify
    strings = []

    for item in items:
        strings.append(stringify(item))

    return ShkString(recv.value.join(strings))

@register_string("trim")
def _string_trim(_frame: Frame, recv: ShkString, args: List[ShkValue]) -> ShkString:
    _string_expect_arity("trim", args, 0)

    return ShkString(recv.value.strip())

@register_string("lower")
def _string_lower(_frame: Frame, recv: ShkString, args: List[ShkValue]) -> ShkString:
    _string_expect_arity("lower", args, 0)

    return ShkString(recv.value.lower())

@register_string("upper")
def _string_upper(_frame: Frame, recv: ShkString, args: List[ShkValue]) -> ShkString:
    _string_expect_arity("upper", args, 0)

    return ShkString(recv.value.upper())

@register_string("hasPrefix")
def _string_has_prefix(_frame: Frame, recv: ShkString, args: List[ShkValue]) -> ShkBool:
    _string_expect_arity("hasPrefix", args, 1)
    prefix = _string_arg("hasPrefix", args[0])

    return ShkBool(recv.value.startswith(prefix))

@register_string("hasSuffix")
def _string_has_suffix(_frame: Frame, recv: ShkString, args: List[ShkValue]) -> ShkBool:
    _string_expect_arity("hasSuffix", args, 1)
    suffix = _string_arg("hasSuffix", args[0])

    return ShkBool(recv.value.endswith(suffix))

@register_string("isAscii")
def _string_is_ascii(_frame: Frame, recv: ShkString, args: List[ShkValue]) -> ShkBool:
    _string_expect_arity("isAscii", args, 0)

    return ShkBool(recv.value.isascii())

@register_command("run")
def _command_run(_frame: Frame, recv: ShkCommand, args: List[ShkValue]) -> ShkValue:
    if args:
        raise ShakarArityError(f"command.run expects 0 args; got {len(args)}")
    cmd = recv.render()

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    except OSError as exc:
        raise ShakarRuntimeError(f"Command execution failed: {exc}") from exc

    if result.returncode != 0:
        raise CommandError(cmd, result.returncode, result.stdout, result.stderr)

    return ShkString(result.stdout)

def regex_match_value(regex: ShkRegex, text: str) -> ShkValue:
    match = regex.compiled.search(text)
    if match is None:
        return ShkNull()

    groups = list(match.groups())
    if regex.include_full:
        values = [match.group(0)] + groups
    else:
        values = groups if groups else [match.group(0)]

    items: List[ShkValue] = [
        ShkString(val) if val is not None else ShkNull()
        for val in values
    ]
    return ShkArray(items)

@register_regex("test")
def _regex_test(_frame: Frame, recv: ShkRegex, args: List[ShkValue]) -> ShkBool:
    _regex_expect_arity("test", args, 1)
    text = _regex_arg("test", args[0])
    return ShkBool(recv.compiled.search(text) is not None)

@register_regex("match")
def _regex_match(_frame: Frame, recv: ShkRegex, args: List[ShkValue]) -> ShkValue:
    _regex_expect_arity("match", args, 1)
    text = _regex_arg("match", args[0])
    return regex_match_value(recv, text)

@register_regex("replace")
def _regex_replace(_frame: Frame, recv: ShkRegex, args: List[ShkValue]) -> ShkString:
    _regex_expect_arity("replace", args, 2)
    text = _regex_arg("replace", args[0])
    repl = _regex_arg("replace", args[1])
    return ShkString(recv.compiled.sub(repl, text))

def _path_expect_arity(method: str, args: List[ShkValue], expected: int) -> None:
    if len(args) != expected:
        raise ShakarArityError(f"path.{method} expects {expected} argument(s); got {len(args)}")

def _path_arg_number(method: str, arg: ShkValue) -> int:
    if isinstance(arg, ShkNumber):
        if float(arg.value).is_integer():
            return int(arg.value)
        raise ShakarTypeError(f"path.{method} expects an integer mode")
    raise ShakarTypeError(f"path.{method} expects a number mode")

def _path_arg_string(method: str, arg: ShkValue) -> str:
    if isinstance(arg, ShkString):
        return arg.value
    raise ShakarTypeError(f"path.{method} expects a string argument")

@register_path("read")
def _path_read(_frame: Frame, recv: ShkPath, args: List[ShkValue]) -> ShkString:
    _path_expect_arity("read", args, 0)
    path = recv.as_path()
    try:
        content = path.read_text()
    except OSError as exc:
        raise ShakarRuntimeError(f"Path read failed: {exc}") from exc
    return ShkString(content)

@register_path("write")
def _path_write(_frame: Frame, recv: ShkPath, args: List[ShkValue]) -> ShkNull:
    _path_expect_arity("write", args, 1)
    content = _path_arg_string("write", args[0])
    path = recv.as_path()
    try:
        path.write_text(content)
    except OSError as exc:
        raise ShakarRuntimeError(f"Path write failed: {exc}") from exc
    return ShkNull()

@register_path("chmod")
def _path_chmod(_frame: Frame, recv: ShkPath, args: List[ShkValue]) -> ShkNull:
    _path_expect_arity("chmod", args, 1)
    mode = _path_arg_number("chmod", args[0])
    path = recv.as_path()
    try:
        path.chmod(mode)
    except OSError as exc:
        raise ShakarRuntimeError(f"Path chmod failed: {exc}") from exc
    return ShkNull()

def call_builtin_method(recv: ShkValue, name: str, args: List[ShkValue], frame: 'Frame') -> ShkValue:
    registry_by_type: Dict[type, MethodRegistry] = {
        ShkArray: Builtins.array_methods,
        ShkString: Builtins.string_methods,
        ShkRegex: Builtins.regex_methods,
        ShkObject: Builtins.object_methods,
        ShkCommand: Builtins.command_methods,
        ShkPath: Builtins.path_methods,
    }

    registry = registry_by_type.get(type(recv))
    if registry:
        handler = registry.get(name)
        if handler is not None:
            return handler(frame, recv, args)

    raise ShakarMethodNotFound(recv, name)

def _validate_return_contract(fn: ShkFn, result: ShkValue, callee_frame: 'Frame') -> ShkValue:
    """Validate return value against function's return contract if present"""
    if fn.return_contract is None:
        return result

    from .evaluator import eval_node  # local import to avoid cycle
    from .eval.match import match_structure

    # Evaluate the contract expression in the function's closure scope
    contract_value = eval_node(fn.return_contract, callee_frame)

    # Check if result matches the contract
    if not match_structure(result, contract_value):
        raise ShakarTypeError(f"Return value does not match contract: expected {contract_value}, got {result}")

    return result

def call_shkfn(fn: ShkFn, positional: List[ShkValue], subject: Optional[ShkValue], caller_frame: 'Frame') -> ShkValue:
    """
    Subjectful call semantics:
    - subject is available to callee as frame.dot
    - If fn.params is None, treat as unary subject-lambda: ignore positional, evaluate body with dot bound.
    - Else: arity must match len(fn.params); bind params; dot=subject.
    """

    if fn.decorators:
        return _call_shkfn_with_decorators(fn, positional, subject, caller_frame)

    return _call_shkfn_raw(fn, positional, subject, caller_frame)

def _bind_params_with_spread(params: List[str], vararg_indices: List[int], positional: List[ShkValue], *, label: str) -> List[ShkValue]:
    spread_set = set(vararg_indices)
    non_spread_after = [0] * len(params)
    count = 0

    for idx in range(len(params) - 1, -1, -1):
        non_spread_after[idx] = count
        if idx not in spread_set:
            count += 1

    bound_values: List[ShkValue] = []
    current_pos = 0
    total_args = len(positional)

    for idx, _name in enumerate(params):
        if idx in spread_set:
            needed_after = non_spread_after[idx]
            available = total_args - current_pos
            take = max(0, available - needed_after)
            spread_vals = positional[current_pos:current_pos + take]
            bound_values.append(ShkArray(list(spread_vals)))
            current_pos += take
            continue

        if current_pos >= total_args:
            raise ShakarArityError(f"{label} expects {len(params)} args; got {len(positional)}")
        bound_values.append(positional[current_pos])
        current_pos += 1

    if current_pos < total_args:
        raise ShakarArityError(f"too many arguments for {label.lower()} with {len(params)} parameters")

    return bound_values

def _bind_decorator_params(params: List[str], vararg_indices: List[int], args: List[ShkValue]) -> List[ShkValue]:
    if not vararg_indices:
        if len(args) != len(params):
            raise ShakarArityError(f"Decorator expects {len(params)} args; got {len(args)}")
        return list(args)

    required = len(params) - len(vararg_indices)
    if len(args) < required:
        raise ShakarArityError(f"Decorator expects at least {required} args; got {len(args)}")
    return _bind_params_with_spread(params, vararg_indices, list(args), label="Decorator")

def _call_shkfn_raw(fn: ShkFn, positional: List[ShkValue], subject: Optional[ShkValue], caller_frame: 'Frame') -> ShkValue:
    _ = caller_frame
    from .evaluator import eval_node  # local import to avoid cycle

    if fn.params is None:
        if subject is None:
            if not positional:
                raise ShakarArityError("Subject-only amp_lambda expects a subject argument")
            subject, *positional = positional

        if positional:
            raise ShakarArityError(f"Subject-only amp_lambda does not take positional args; got {len(positional)} extra")

        callee_frame = Frame(parent=fn.frame, dot=subject)
        callee_frame.mark_function_frame()

        try:
            result = _ensure_shk_value(eval_node(fn.body, callee_frame))
        except ShakarReturnSignal as signal:
            result = signal.value

        return _validate_return_contract(fn, result, callee_frame)

    callee_frame = Frame(parent=fn.frame, dot=subject)

    varargs = fn.vararg_indices or []

    if not varargs:
        if len(positional) != len(fn.params):
            raise ShakarArityError(f"Function expects {len(fn.params)} args; got {len(positional)}")
        bound_values = positional
    else:
        required = len(fn.params) - len(varargs)
        if len(positional) < required:
            raise ShakarArityError(f"Function expects at least {required} args; got {len(positional)}")
        bound_values = _bind_params_with_spread(fn.params, varargs, positional, label="Function")

    for name, val in zip(fn.params, bound_values):
        callee_frame.define(name, val)

    callee_frame.mark_function_frame()

    try:
        result = _ensure_shk_value(eval_node(fn.body, callee_frame))
    except ShakarReturnSignal as signal:
        result = signal.value

    return _validate_return_contract(fn, result, callee_frame)

def _call_shkfn_with_decorators(fn: ShkFn, positional: List[ShkValue], subject: Optional[ShkValue], caller_frame: 'Frame') -> ShkValue:
    chain = fn.decorators or ()
    args = ShkArray(list(positional))

    return _run_decorator_chain(fn, chain, 0, args, subject, caller_frame)

def _run_decorator_chain(
    fn: ShkFn,
    chain: Tuple[DecoratorConfigured, ...],
    index: int,
    args_value: ShkArray,
    subject: Optional[ShkValue],
    caller_frame: 'Frame',
) -> ShkValue:
    if index >= len(chain):
        return _call_shkfn_raw(fn, list(args_value.items), subject, caller_frame)
    continuation = DecoratorContinuation(
        fn=fn,
        decorators=chain,
        index=index + 1,
        subject=subject,
        caller_frame=caller_frame,
    )
    inst = chain[index]

    return _execute_decorator_instance(inst, continuation, args_value, subject, caller_frame)

def _execute_decorator_instance(
    inst: DecoratorConfigured,
    continuation: DecoratorContinuation,
    args_value: ShkArray,
    subject: Optional[ShkValue],
    caller_frame: 'Frame',
) -> ShkValue:
    from .evaluator import eval_node  # defer to avoid cycle
    deco_frame = Frame(parent=inst.decorator.frame, dot=subject)
    deco_frame.mark_function_frame()
    params = inst.decorator.params or []

    for name, val in zip(params, inst.args):
        deco_frame.define(name, val)
    deco_frame.define('f', continuation)
    deco_frame.define('args', args_value)

    try:
        eval_node(inst.decorator.body, deco_frame)
    except ShakarReturnSignal as signal:
        return signal.value

    updated_args_value = deco_frame.get('args')

    return continuation.invoke(updated_args_value)

def _coerce_decorator_args(value: ShkValue) -> ShkArray:
    if isinstance(value, ShkArray):
        return value

    raise ShakarTypeError("Decorator args must be an array value")

# Runtime type constants for structural matching
TYPE_INT = ShkType("Int", ShkNumber)
TYPE_FLOAT = ShkType("Float", ShkNumber)
TYPE_STR = ShkType("Str", ShkString)
TYPE_BOOL = ShkType("Bool", ShkBool)
TYPE_ARRAY = ShkType("Array", ShkArray)
TYPE_OBJECT = ShkType("Object", ShkObject)
TYPE_NIL = ShkType("Nil", ShkNull)
TYPE_PATH = ShkType("Path", ShkPath)

Builtins.type_constants = {
    "Int": TYPE_INT,
    "Float": TYPE_FLOAT,
    "Str": TYPE_STR,
    "Bool": TYPE_BOOL,
    "Array": TYPE_ARRAY,
    "Object": TYPE_OBJECT,
    "Nil": TYPE_NIL,
    "Path": TYPE_PATH,
}
