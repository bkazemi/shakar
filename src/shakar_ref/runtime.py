from __future__ import annotations

import importlib
import pathlib
import re
import subprocess
from typing import Callable, Dict, List, Optional, Tuple, Union
from .tree import Node
from .types import (
    ShkNull,
    ShkNumber,
    ShkDuration,
    ShkSize,
    ShkString,
    ShkRegex,
    ShkBool,
    ShkArray,
    ShkFan,
    ShkObject,
    ShkModule,
    ShkCommand,
    ShkPath,
    ShkEnvVar,
    ShkSelector,
    SelectorIndex,
    SelectorSlice,
    SelectorPart,
    ShkFn,
    ShkDecorator,
    DecoratorConfigured,
    DecoratorContinuation,
    BoundMethod,
    BuiltinMethod,
    Descriptor,
    StdlibFunction,
    DeferEntry,
    ShkValue,
    EvalResult,
    DotValue,
    Frame,
    ShkType,
    ShkOptional,
    ShkUnion,
    ShakarImportError,
    ShakarRuntimeError,
    ShakarTypeError,
    ShakarArityError,
    ShakarKeyError,
    ShakarIndexError,
    ShakarMethodNotFound,
    CommandError,
    ShakarAssertionError,
    ShakarReturnSignal,
    ShakarBreakSignal,
    ShakarContinueSignal,
    Method,
    MethodRegistry,
    Builtins,
    is_shk_value,
    _ensure_shk_value,
    StdlibFn,
)
from .utils import envvar_value_by_name, stringify

__all__ = [
    "Frame",
    "ShkNull",
    "ShkNumber",
    "ShkDuration",
    "ShkSize",
    "ShkString",
    "ShkRegex",
    "ShkBool",
    "ShkArray",
    "ShkFan",
    "ShkObject",
    "ShkModule",
    "ShkCommand",
    "ShkPath",
    "ShkEnvVar",
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
    "ShakarImportError",
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
    "register_module_factory",
    "import_module",
    "init_stdlib",
]

_STDLIB_INITIALIZED = False
_MODULE_LOADING = object()

MODULE_REGISTRY: Dict[str, Union[ShkModule, object]] = {}
MODULE_FACTORIES: Dict[str, Callable[[], ShkModule]] = {}
MODULE_FACTORY_REUSABLE: Dict[str, bool] = {}


def register_module_factory(
    name: str, factory: Callable[[], ShkModule], *, reusable: bool = False
) -> None:
    MODULE_FACTORIES[name] = factory
    if reusable:
        MODULE_FACTORY_REUSABLE[name] = True
    else:
        MODULE_FACTORY_REUSABLE.pop(name, None)


def _looks_like_path(name: str) -> bool:
    if name.startswith(("./", "../", "/", ".\\", "..\\")):
        return True
    return re.match(r"^[A-Za-z]:[\\/]", name) is not None


def _resolve_import_path(name: str, frame: Optional[Frame]) -> pathlib.Path:
    path = pathlib.Path(name)

    if not path.is_absolute():
        base = frame.source_path if frame else None
        base_path = pathlib.Path(base).parent if base else pathlib.Path.cwd()
        path = base_path / path

    if path.suffix == "":
        candidate = path.with_suffix(".shk")
        if candidate.exists():
            path = candidate

    return path.resolve()


def _resolve_import_target(
    name: str, frame: Optional[Frame]
) -> tuple[str, Optional[pathlib.Path]]:
    if _looks_like_path(name):
        path = _resolve_import_path(name, frame)
        return str(path), path
    return name, None


def _load_module_from_factory(key: str, factory: Callable[[], ShkModule]) -> ShkModule:
    MODULE_REGISTRY[key] = _MODULE_LOADING

    try:
        module = factory()
    except Exception:
        MODULE_REGISTRY.pop(key, None)
        raise

    if not isinstance(module, ShkModule):
        MODULE_REGISTRY.pop(key, None)
        raise ShakarImportError(f"Module factory for '{key}' did not return a module")

    if module.name is None:
        module.name = key

    if not MODULE_FACTORY_REUSABLE.get(key, False):
        MODULE_FACTORIES.pop(key, None)
        MODULE_FACTORY_REUSABLE.pop(key, None)
    MODULE_REGISTRY[key] = module
    return module


def _load_module_from_file(
    key: str, path: pathlib.Path, *, source_name: str
) -> ShkModule:
    if not path.exists() or not path.is_file():
        raise ShakarImportError(f"Module file not found: '{source_name}'")

    MODULE_REGISTRY[key] = _MODULE_LOADING

    try:
        source = path.read_text(encoding="utf-8")

        from .parser_rd import parse_source, ParseError
        from .lexer_rd import LexError
        from .ast_transforms import Prune, looks_like_offside
        from .lower import lower
        from .evaluator import eval_expr

        preferred = looks_like_offside(source)
        attempts = [preferred, not preferred]
        last_error = None
        tree = None

        for flag in attempts:
            try:
                tree = parse_source(source, use_indenter=flag)
                break
            except (ParseError, LexError) as exc:
                last_error = exc

        if tree is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Parser failed without producing a parse tree")

        ast = Prune().transform(tree)
        ast2 = lower(ast)
        d = getattr(ast2, "data", None)
        if d in ("start_noindent", "start_indented"):
            children = getattr(ast2, "children", None)
            if children and len(children) == 1:
                ast2 = children[0]

        module_frame = Frame(source=source, source_path=str(path))
        builtin_snapshot = dict(module_frame.vars)
        eval_expr(ast2, module_frame, source=source)

        exports: Dict[str, ShkValue] = {}
        for name, value in module_frame.vars.items():
            if name not in builtin_snapshot or value is not builtin_snapshot[name]:
                exports[name] = value

        module = ShkModule(slots=exports, name=str(path))
        MODULE_REGISTRY[key] = module
        return module
    except Exception:
        MODULE_REGISTRY.pop(key, None)
        raise


def import_module(name: str, frame: Optional[Frame] = None) -> ShkModule:
    init_stdlib()

    if not _looks_like_path(name):
        if "\\" in name:
            raise ShakarImportError("Relative imports must start with './' or '../'")
        if name.endswith(".shk"):
            raise ShakarImportError(
                "Builtin modules cannot use '.shk' extension; use './' for file imports"
            )

    key, path = _resolve_import_target(name, frame)
    existing = MODULE_REGISTRY.get(key)
    if existing is _MODULE_LOADING:
        raise ShakarImportError(f"Circular import detected for '{name}'")
    if isinstance(existing, ShkModule):
        return existing

    factory = MODULE_FACTORIES.get(key)
    if factory is not None:
        return _load_module_from_factory(key, factory)

    if path is None:
        raise ShakarImportError(f"Module '{name}' not found")

    return _load_module_from_file(key, path, source_name=name)


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


def register_envvar(name: str):
    return register_method(Builtins.envvar_methods, name)


def register_stdlib(name: str, *, arity: Optional[int] = None):
    def dec(fn: StdlibFn):
        Builtins.stdlib_functions[name] = StdlibFunction(fn=fn, arity=arity)
        return fn

    return dec


def _string_expect_arity(method: str, args: List[ShkValue], expected: int) -> None:
    if len(args) != expected:
        raise ShakarArityError(
            f"string.{method} expects {expected} argument(s); got {len(args)}"
        )


def _string_arg(method: str, arg: ShkValue) -> str:
    if isinstance(arg, ShkString):
        return arg.value

    raise ShakarTypeError(f"string.{method} expects a string argument")


def _validate_repeat_count(method_name: str, arg: ShkValue) -> int:
    if not isinstance(arg, ShkNumber) or not float(arg.value).is_integer():
        raise ShakarTypeError(f"{method_name} expects an integer count")

    count = int(arg.value)
    if count < 0:
        raise ShakarRuntimeError(f"{method_name} count cannot be negative")

    return count


def _regex_expect_arity(method: str, args: List[ShkValue], expected: int) -> None:
    if len(args) != expected:
        raise ShakarArityError(
            f"regex.{method} expects {expected} argument(s); got {len(args)}"
        )


def _regex_arg(method: str, arg: ShkValue) -> str:
    if isinstance(arg, ShkString):
        return arg.value

    raise ShakarTypeError(f"regex.{method} expects a string argument")


@register_array("push")
def _array_push(_frame: Frame, recv: ShkArray, args: List[ShkValue]) -> ShkNull:
    if len(args) != 1:
        raise ShakarArityError(f"array.push expects 1 argument; got {len(args)}")
    recv.items.append(args[0])
    return ShkNull()


@register_array("append")
def _array_append(_frame: Frame, recv: ShkArray, args: List[ShkValue]) -> ShkNull:
    # Alias for push
    return _array_push(_frame, recv, args)


@register_array("pop")
def _array_pop(_frame: Frame, recv: ShkArray, args: List[ShkValue]) -> ShkValue:
    if len(args) != 0:
        raise ShakarArityError(f"array.pop expects 0 arguments; got {len(args)}")
    if not recv.items:
        raise ShakarRuntimeError("pop from empty array")
    return recv.items.pop()


@register_array("repeat")
def _array_repeat(_frame: Frame, recv: ShkArray, args: List[ShkValue]) -> ShkArray:
    if len(args) != 1:
        raise ShakarArityError(f"array.repeat expects 1 argument; got {len(args)}")

    count = _validate_repeat_count("array.repeat", args[0])

    return ShkArray(recv.items * count)


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


@register_string("repeat")
def _string_repeat(_frame: Frame, recv: ShkString, args: List[ShkValue]) -> ShkString:
    _string_expect_arity("repeat", args, 1)

    count = _validate_repeat_count("string.repeat", args[0])

    return ShkString(recv.value * count)


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


@register_string("split")
def _string_split(_frame: Frame, recv: ShkString, args: List[ShkValue]) -> ShkArray:
    _string_expect_arity("split", args, 1)
    sep_val = args[0]
    if not isinstance(sep_val, ShkString):
        raise ShakarTypeError("string.split separator must be a string")

    sep = sep_val.value
    if not sep:
        raise ShakarRuntimeError("empty separator")

    parts = recv.value.split(sep)
    return ShkArray([ShkString(p) for p in parts])


@register_object("keys")
def _object_keys(_frame: Frame, recv: ShkObject, args: List[ShkValue]) -> ShkArray:
    if len(args) != 0:
        raise ShakarArityError(f"object.keys expects 0 arguments; got {len(args)}")
    return ShkArray([ShkString(k) for k in recv.slots.keys()])


@register_object("values")
def _object_values(_frame: Frame, recv: ShkObject, args: List[ShkValue]) -> ShkArray:
    if len(args) != 0:
        raise ShakarArityError(f"object.values expects 0 arguments; got {len(args)}")
    return ShkArray(list(recv.slots.values()))


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
        ShkString(val) if val is not None else ShkNull() for val in values
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
        raise ShakarArityError(
            f"path.{method} expects {expected} argument(s); got {len(args)}"
        )


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


# ---------- EnvVar methods ----------

import os as _os


def _envvar_expect_arity(method: str, args: List[ShkValue], expected: int) -> None:
    if len(args) != expected:
        raise ShakarArityError(
            f"envvar.{method} expects {expected} args; got {len(args)}"
        )


@register_envvar("assign")
def _envvar_assign(_frame: Frame, recv: ShkEnvVar, args: List[ShkValue]) -> ShkEnvVar:
    """Set the environment variable to a new value."""
    _envvar_expect_arity("assign", args, 1)
    val = args[0]
    if isinstance(val, ShkString):
        str_val = val.value
    elif isinstance(val, ShkEnvVar):
        env_val = _os.environ.get(val.name)
        if env_val is None:
            _os.environ.pop(recv.name, None)
            return recv
        str_val = env_val
    elif isinstance(val, ShkNull):
        # Setting to nil is equivalent to unset
        _os.environ.pop(recv.name, None)
        return recv
    else:
        str_val = stringify(val)
    _os.environ[recv.name] = str_val
    return recv


@register_envvar("unset")
def _envvar_unset(_frame: Frame, recv: ShkEnvVar, args: List[ShkValue]) -> ShkNull:
    """Remove the environment variable."""
    _envvar_expect_arity("unset", args, 0)
    _os.environ.pop(recv.name, None)
    return ShkNull()


def call_builtin_method(
    recv: ShkValue, name: str, args: List[ShkValue], frame: "Frame"
) -> ShkValue:
    if isinstance(recv, ShkEnvVar):
        handler = Builtins.envvar_methods.get(name)
        if handler is not None:
            return handler(frame, recv, args)

        str_handler = Builtins.string_methods.get(name)
        if str_handler is not None:
            env_val = envvar_value_by_name(recv.name)
            if env_val is None:
                raise ShakarTypeError(f"Env var '{recv.name}' has no value")
            return str_handler(frame, ShkString(env_val), args)

    registry_by_type: Dict[type, MethodRegistry] = {
        ShkArray: Builtins.array_methods,
        ShkString: Builtins.string_methods,
        ShkRegex: Builtins.regex_methods,
        ShkObject: Builtins.object_methods,
        ShkCommand: Builtins.command_methods,
        ShkPath: Builtins.path_methods,
        ShkEnvVar: Builtins.envvar_methods,
    }

    registry = registry_by_type.get(type(recv))
    if registry:
        handler = registry.get(name)
        if handler is not None:
            return handler(frame, recv, args)

    raise ShakarMethodNotFound(recv, name)


def _validate_return_contract(
    fn: ShkFn, result: ShkValue, callee_frame: "Frame"
) -> ShkValue:
    """Validate return value against function's return contract if present"""
    if fn.return_contract is None:
        return result

    from .evaluator import eval_node  # local import to avoid cycle
    from .eval.match import match_structure

    # Evaluate the contract expression in the function's closure scope
    contract_value = eval_node(fn.return_contract, callee_frame)

    # Check if result matches the contract
    if not match_structure(result, contract_value):
        raise ShakarTypeError(
            f"Return value does not match contract: expected {contract_value}, got {result}"
        )

    return result


def call_shkfn(
    fn: ShkFn,
    positional: List[ShkValue],
    subject: Optional[ShkValue],
    caller_frame: "Frame",
) -> ShkValue:
    """
    Subjectful call semantics:
    - subject is available to callee as frame.dot
    - If fn.params is None, treat as unary subject-lambda: ignore positional, evaluate body with dot bound.
    - Else: arity must match len(fn.params); bind params; dot=subject.
    """

    if fn.decorators:
        return _call_shkfn_with_decorators(fn, positional, subject, caller_frame)

    return _call_shkfn_raw(fn, positional, subject, caller_frame)


def _bind_params_with_defaults(
    params: List[str],
    vararg_indices: List[int],
    positional: List[ShkValue],
    defaults: Optional[List[Optional[Node]]],
    *,
    label: str,
    eval_default: Optional[Callable[[Node], ShkValue]] = None,
    on_bind: Optional[Callable[[str, ShkValue], None]] = None,
) -> List[ShkValue]:
    defaults_list = defaults or []
    if len(defaults_list) < len(params):
        defaults_list = defaults_list + [None] * (len(params) - len(defaults_list))

    spread_index = None
    if vararg_indices:
        if len(vararg_indices) != 1 or vararg_indices[0] != len(params) - 1:
            raise ShakarRuntimeError("Spread parameter must be last")
        spread_index = vararg_indices[0]

    optional_present = False
    for idx in range(len(params)):
        if idx == spread_index:
            continue
        if defaults_list[idx] is not None:
            optional_present = True
            break

    if spread_index is None and not optional_present:
        if len(positional) != len(params):
            raise ShakarArityError(
                f"{label} expects {len(params)} args; got {len(positional)}"
            )
        if on_bind is not None:
            for name, val in zip(params, positional):
                on_bind(name, val)
        return list(positional)

    required = 0
    for idx in range(len(params)):
        if idx == spread_index:
            continue
        if defaults_list[idx] is None:
            required += 1

    if len(positional) < required:
        raise ShakarArityError(
            f"{label} expects at least {required} args; got {len(positional)}"
        )

    bound_values: List[ShkValue] = []
    current_pos = 0

    for idx, name in enumerate(params):
        if spread_index is not None and idx == spread_index:
            spread_vals = positional[current_pos:]
            bound = ShkArray(list(spread_vals))
            bound_values.append(bound)
            if on_bind is not None:
                on_bind(name, bound)
            current_pos = len(positional)
            continue

        if current_pos < len(positional):
            bound = positional[current_pos]
            current_pos += 1
        else:
            default_node = defaults_list[idx]
            if default_node is None:
                raise ShakarArityError(
                    f"{label} expects at least {required} args; got {len(positional)}"
                )
            if eval_default is None:
                raise ShakarRuntimeError("Missing default evaluator")
            bound = eval_default(default_node)

        bound_values.append(bound)
        if on_bind is not None:
            on_bind(name, bound)

    if current_pos < len(positional):
        raise ShakarArityError(
            f"too many arguments for {label.lower()} with {len(params)} parameters"
        )

    return bound_values


def _bind_decorator_params(
    decorator: ShkDecorator, args: List[ShkValue]
) -> List[ShkValue]:
    from .evaluator import eval_node  # local import to avoid cycle

    params = decorator.params or []
    defaults = decorator.param_defaults or []

    temp_frame = Frame(parent=decorator.frame)

    return _bind_params_with_defaults(
        params=params,
        vararg_indices=decorator.vararg_indices or [],
        positional=list(args),
        defaults=defaults,
        label="Decorator",
        eval_default=lambda node: _ensure_shk_value(eval_node(node, temp_frame)),
        on_bind=temp_frame.define,
    )


def _call_shkfn_raw(
    fn: ShkFn,
    positional: List[ShkValue],
    subject: Optional[ShkValue],
    caller_frame: "Frame",
) -> ShkValue:
    _ = caller_frame
    from .evaluator import eval_node  # local import to avoid cycle

    if fn.params is None:
        if subject is None:
            if not positional:
                raise ShakarArityError(
                    "Subject-only amp_lambda expects a subject argument"
                )
            subject, *positional = positional

        if positional:
            raise ShakarArityError(
                f"Subject-only amp_lambda does not take positional args; got {len(positional)} extra"
            )

        callee_frame = Frame(parent=fn.frame, dot=subject)
        callee_frame.mark_function_frame()

        try:
            result = _ensure_shk_value(eval_node(fn.body, callee_frame))
        except ShakarReturnSignal as signal:
            result = signal.value

        return _validate_return_contract(fn, result, callee_frame)

    callee_frame = Frame(parent=fn.frame, dot=subject)

    _bind_params_with_defaults(
        params=fn.params,
        vararg_indices=fn.vararg_indices or [],
        positional=positional,
        defaults=fn.param_defaults,
        label="Function",
        eval_default=lambda node: _ensure_shk_value(eval_node(node, callee_frame)),
        on_bind=callee_frame.define,
    )

    callee_frame.mark_function_frame()

    try:
        result = _ensure_shk_value(eval_node(fn.body, callee_frame))
    except ShakarReturnSignal as signal:
        result = signal.value

    return _validate_return_contract(fn, result, callee_frame)


def _call_shkfn_with_decorators(
    fn: ShkFn,
    positional: List[ShkValue],
    subject: Optional[ShkValue],
    caller_frame: "Frame",
) -> ShkValue:
    chain = fn.decorators or ()
    args = ShkArray(list(positional))

    return _run_decorator_chain(fn, chain, 0, args, subject, caller_frame)


def _run_decorator_chain(
    fn: ShkFn,
    chain: Tuple[DecoratorConfigured, ...],
    index: int,
    args_value: ShkArray,
    subject: Optional[ShkValue],
    caller_frame: "Frame",
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

    return _execute_decorator_instance(
        inst, continuation, args_value, subject, caller_frame
    )


def _execute_decorator_instance(
    inst: DecoratorConfigured,
    continuation: DecoratorContinuation,
    args_value: ShkArray,
    subject: Optional[ShkValue],
    caller_frame: "Frame",
) -> ShkValue:
    from .evaluator import eval_node  # defer to avoid cycle

    deco_frame = Frame(parent=inst.decorator.frame, dot=subject)
    deco_frame.mark_function_frame()
    params = inst.decorator.params or []

    for name, val in zip(params, inst.args):
        deco_frame.define(name, val)
    deco_frame.define("f", continuation)
    deco_frame.define("args", args_value)

    try:
        eval_node(inst.decorator.body, deco_frame)
    except ShakarReturnSignal as signal:
        return signal.value

    updated_args_value = deco_frame.get("args")

    return continuation.invoke(updated_args_value)


def _coerce_decorator_args(value: ShkValue) -> ShkArray:
    if isinstance(value, ShkArray):
        return value

    raise ShakarTypeError("Decorator args must be an array value")


# Runtime type constants for structural matching
TYPE_INT = ShkType("Int", ShkNumber)
TYPE_FLOAT = ShkType("Float", ShkNumber)
TYPE_DURATION = ShkType("Duration", ShkDuration)
TYPE_SIZE = ShkType("Size", ShkSize)
TYPE_STR = ShkType("Str", ShkString)
TYPE_BOOL = ShkType("Bool", ShkBool)
TYPE_ARRAY = ShkType("Array", ShkArray)
TYPE_OBJECT = ShkType("Object", ShkObject)
TYPE_NIL = ShkType("Nil", ShkNull)
TYPE_PATH = ShkType("Path", ShkPath)

Builtins.type_constants = {
    "Int": TYPE_INT,
    "Float": TYPE_FLOAT,
    "Duration": TYPE_DURATION,
    "Size": TYPE_SIZE,
    "Str": TYPE_STR,
    "Bool": TYPE_BOOL,
    "Array": TYPE_ARRAY,
    "Object": TYPE_OBJECT,
    "Nil": TYPE_NIL,
    "Path": TYPE_PATH,
}
