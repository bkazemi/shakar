from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union
from ..tree import Tok
from ..token_types import TT

from ..runtime import (
    BoundMethod,
    BoundCallable,
    BuiltinMethod,
    DecoratorConfigured,
    DecoratorContinuation,
    EvalResult,
    CallSite,
    Frame,
    ShkDecorator,
    ShkFn,
    ShkArray,
    ShkFan,
    ShkObject,
    ShkSelector,
    SelectorIndex,
    SelectorPart,
    ShkValue,
    ShakarArityError,
    ShakarKeyError,
    ShakarMethodNotFound,
    ShakarRuntimeError,
    ShakarTypeError,
    StdlibFunction,
    _bind_decorator_params,
    call_builtin_method,
    call_shkfn,
)
from ..tree import (
    Node,
    Tree,
    child_by_label,
    is_token,
    is_tree,
    tree_children,
    tree_label,
)
from .bind import FanContext, RebindContext, apply_fan_op
from .common import callsite_from_node, expect_ident_token as _expect_ident_token
from .helpers import eval_anchor_scoped
from .mutation import get_field_value, index_value, slice_value
from .selector import evaluate_selectorlist, apply_selectors_to_value
from .valuefan import eval_valuefan

EvalFunc = Callable[[Node, Frame], ShkValue]


def _call_name_for_value(cal: ShkValue) -> str:
    match cal:
        case BoundMethod(name=name) if name:
            return f"{name}()"
        case BoundMethod(fn=fn) if fn.name:
            return f"{fn.name}()"
        case BoundCallable(name=name) if name:
            return f"{name}()"
        case BoundCallable(target=target):
            return _call_name_for_value(target)
        case BuiltinMethod(name=name):
            return f"{name}()"
        case StdlibFunction(name=name) if name:
            return f"{name}()"
        case StdlibFunction(fn=fn):
            return f"{fn.__name__}()"
        case ShkFn(name=name) if name:
            return f"{name}()"
        case ShkFn():
            return "<fn>()"
        case DecoratorContinuation():
            return "decorator()"
        case ShkDecorator():
            return "decorator()"
        case _:
            return f"{type(cal).__name__}()"


def _with_call_site(
    frame: Frame, site: Optional["CallSite"], thunk: Callable[[], ShkValue]
) -> ShkValue:
    # Thunks are always called immediately; lambda captures are safe.
    if site is None:
        return thunk()
    frame.call_stack.append(site)
    try:
        return thunk()
    except ShakarRuntimeError as exc:
        if getattr(exc, "shk_call_stack", None) is None:
            exc.shk_call_stack = list(frame.call_stack)
        raise
    finally:
        frame.call_stack.pop()


def eval_args_node(
    args_node: Optional[Union[Tree, List[Tree]]],
    frame: Frame,
    eval_func: EvalFunc,
) -> List[ShkValue]:
    def label(node: Node) -> Optional[str]:
        return tree_label(node)

    def flatten(node: Node) -> List[Node]:
        if is_tree(node):
            tag = label(node)

            if tag in {"args", "arglist", "arglistnamedmixed"}:
                out: List[Node] = []

                for ch in node.children:
                    out.extend(flatten(ch))
                return out

            if tag in {"argitem", "arg"} and node.children:
                return flatten(node.children[0])

            if tag == "namedarg" and node.children:
                # Keep namedarg intact; handle spreading logic separately.
                return [node]

        return [node]

    if is_tree(args_node):
        return _eval_args(flatten(args_node), frame, eval_func)

    if isinstance(args_node, list):
        res: List[Tree] = []
        for n in args_node:
            res.extend(flatten(n))
        return _eval_args(res, frame, eval_func)

    return []


def eval_args_node_with_named(
    args_node: Optional[Union[Tree, List[Tree]]],
    frame: Frame,
    eval_func: EvalFunc,
) -> Tuple[List[ShkValue], Dict[str, ShkValue], bool]:
    def label(node: Node) -> Optional[str]:
        return tree_label(node)

    def flatten(node: Node) -> List[Node]:
        if is_tree(node):
            tag = label(node)

            if tag in {"args", "arglist", "arglistnamedmixed"}:
                out: List[Node] = []

                for ch in node.children:
                    out.extend(flatten(ch))
                return out

            if tag in {"argitem", "arg"} and node.children:
                return flatten(node.children[0])

            if tag == "namedarg" and node.children:
                return [node]

        return [node]

    if is_tree(args_node):
        return _eval_args_with_named(flatten(args_node), frame, eval_func)

    if isinstance(args_node, list):
        res: List[Tree] = []
        for n in args_node:
            res.extend(flatten(n))
        return _eval_args_with_named(res, frame, eval_func)

    return [], {}, False


def _eval_args(nodes: List[Node], frame: Frame, eval_func: EvalFunc) -> List[ShkValue]:
    positional, named, _interleaved = _eval_args_with_named(nodes, frame, eval_func)
    if named:
        raise ShakarTypeError(
            f"Unexpected named argument(s): {', '.join(named.keys())}"
        )
    return positional


def _eval_args_with_named(
    nodes: List[Node], frame: Frame, eval_func: EvalFunc
) -> Tuple[List[ShkValue], Dict[str, ShkValue], bool]:
    positional: List[ShkValue] = []
    named: Dict[str, ShkValue] = {}
    _positional_before_named = False
    _saw_named = False
    _positional_after_named = False

    for node in nodes:
        if _is_namedarg(node):
            _saw_named = True
            name_tok = node.children[0] if node.children else None
            value_expr = node.children[-1] if node.children else None
            if name_tok is None or value_expr is None:
                raise ShakarRuntimeError("Malformed named argument")
            key = name_tok.value if hasattr(name_tok, "value") else str(name_tok)
            if key in named:
                raise ShakarTypeError(f"Duplicate named argument: {key}")
            named[key] = eval_anchor_scoped(value_expr, frame, eval_func)
            continue

        if _saw_named:
            _positional_after_named = True
        else:
            _positional_before_named = True

        if _is_spread(node):
            spread_expr = node.children[0] if is_tree(node) and node.children else None
            if spread_expr is None:
                raise ShakarRuntimeError("Malformed spread argument")
            spread_val = eval_anchor_scoped(spread_expr, frame, eval_func)
            if isinstance(spread_val, (ShkArray, ShkFan)):
                positional.extend(spread_val.items)
                continue
            if isinstance(spread_val, ShkObject):
                positional.extend(spread_val.slots.values())
                continue
            raise ShakarRuntimeError("Spread argument must be an array or object value")

        if _is_raw_fieldfan(node):
            spread_val = eval_anchor_scoped(node, frame, eval_func)

            if not isinstance(spread_val, ShkArray):
                raise ShakarRuntimeError(
                    "Fanout argument did not produce an array value"
                )

            positional.extend(spread_val.items)
            continue

        positional.append(eval_anchor_scoped(node, frame, eval_func))

    interleaved = _positional_before_named and _positional_after_named
    return positional, named, interleaved


def _is_namedarg(node: Node) -> bool:
    return is_tree(node) and tree_label(node) == "namedarg"


def _is_spread(node: Node) -> bool:
    return is_tree(node) and tree_label(node) == "spread"


def _is_raw_fieldfan(node: Node) -> bool:
    """Detect `Base.{a,b}` with no trailing ops so we can auto-flatten in call args."""
    if not is_tree(node) or tree_label(node) != "explicit_chain":
        return False

    ops = node.children[1:]
    if not (bool(ops) and len(ops) == 1):
        return False
    return tree_label(ops[0]) in {"fieldfan", "valuefan"}


def evaluate_index_operand(
    index_node: Tree, frame: Frame, eval_func: EvalFunc
) -> ShkSelector | ShkValue:
    selectorlist = child_by_label(index_node, "selectorlist")

    if selectorlist is not None:
        selectors = evaluate_selectorlist(selectorlist, frame, eval_func)

        if len(selectors) == 1 and isinstance(selectors[0], SelectorIndex):
            return selectors[0].value

        return ShkSelector(selectors)

    expr_node = _index_expr_from_children(index_node.children)
    return eval_func(expr_node, frame)


def apply_slice(
    recv: ShkValue, arms: List[Node], frame: Frame, eval_func: EvalFunc
) -> ShkValue:
    def arm_to_py(node: Node) -> Optional[int]:
        if tree_label(node) == "emptyexpr":
            return None

        value = eval_func(node, frame)
        if hasattr(value, "value"):
            return int(value.value)
        return None

    start, stop, step = map(arm_to_py, arms)

    return slice_value(recv, start, stop, step)


def apply_index_operation(
    recv: ShkValue, op: Tree, frame: Frame, eval_func: EvalFunc
) -> ShkValue:
    selectorlist = child_by_label(op, "selectorlist")
    default_node = _default_arg(op)
    default_thunk = (
        (lambda: eval_func(default_node, frame)) if default_node is not None else None
    )

    if selectorlist is None:
        expr_node = _index_expr_from_children(op.children)
        idx_val = eval_func(expr_node, frame)
        return index_value(recv, idx_val, frame, default_thunk=default_thunk)

    selectors = evaluate_selectorlist(selectorlist, frame, eval_func)

    if default_thunk is not None and not _is_single_index_selector(selectors):
        raise ShakarTypeError("Default index requires a single key selector")

    if len(selectors) == 1 and isinstance(selectors[0], SelectorIndex):
        return index_value(recv, selectors[0].value, frame, default_thunk=default_thunk)

    if default_thunk is not None:
        raise ShakarTypeError("Default index expects an object receiver")

    return apply_selectors_to_value(recv, selectors)


def apply_op(
    recv: EvalResult, op: Tree, frame: Frame, eval_func: EvalFunc
) -> EvalResult:
    if isinstance(recv, FanContext):
        return apply_fan_op(recv, op, frame, apply_op=apply_op, eval_func=eval_func)

    context = None

    if isinstance(recv, RebindContext):
        context = recv
        recv = context.value

    # Handle noanchor wrapper: capture receiver and unwrap
    if op.data == "noanchor":
        frame.pending_anchor_override = recv
        op = op.children[0]

    op_name = op.data
    if op_name in {"field", "fieldsel"}:
        result = _get_field(recv, op, frame)
    elif op_name in {"index", "lv_index"}:
        result = apply_index_operation(recv, op, frame, eval_func)
    elif op_name == "slicesel":
        result = apply_slice(recv, op.children, frame, eval_func)
    elif op_name == "fieldfan":
        result = apply_fan_op(recv, op, frame, apply_op=apply_op, eval_func=eval_func)
    elif op_name == "valuefan":
        result = _valuefan(recv, op, frame, eval_func)
    elif op_name == "call":
        args_node = op.children[0] if op.children else None
        positional, named, interleaved = eval_args_node_with_named(
            args_node, frame, eval_func
        )
        result = call_value(
            recv,
            positional,
            frame,
            eval_func,
            named=named,
            interleaved=interleaved,
            call_node=op,
        )
    elif op_name == "method":
        result = _call_method(recv, op, frame, eval_func)
    else:
        raise ShakarRuntimeError(f"Unknown chain op: {op.data}")

    if context is not None:
        context.value = result

        if not isinstance(result, (BuiltinMethod, BoundMethod, BoundCallable, ShkFn)):
            context.setter(result)
        return context

    return result


def _get_field(recv: ShkValue, op: Tree, frame: Frame) -> ShkValue:
    # field nodes may carry a leading DOT token plus the identifier; pick the name token
    tokens = [ch for ch in op.children if isinstance(ch, Tok)]
    tok = tokens[-1] if tokens else op.children[0]
    try:
        field_name = _expect_ident_token(tok, "Field access")
    except ShakarRuntimeError as err:
        meta = getattr(tok, "meta", None)
        span = (
            (getattr(meta, "line", None), getattr(meta, "column", None))
            if meta
            else (None, None)
        )
        raise ShakarRuntimeError(f"{err.args[0]} at {span}") from None
    return get_field_value(recv, field_name, frame)


def _is_missing_field_error(err: ShakarTypeError) -> bool:
    """Heuristic to detect missing-field errors without masking real failures."""
    # TODO: Replace message matching with a dedicated field-not-found error type.
    msg = err.args[0] if err.args else ""
    return (
        "has no field" in msg
        or "has no fields" in msg
        or "Unsupported field access" in msg
    )


def _call_method(
    recv: ShkValue, op: Tree, frame: Frame, eval_func: EvalFunc
) -> ShkValue:
    method_name = _expect_ident_token(op.children[0], "Method call")
    positional, named, interleaved = eval_args_node_with_named(
        op.children[1] if len(op.children) > 1 else None, frame, eval_func
    )
    call_site = callsite_from_node(f"{method_name}()", op, frame)

    try:
        cal = get_field_value(recv, method_name, frame)
    except ShakarKeyError:
        cal = None
    except ShakarTypeError as err:
        if _is_missing_field_error(err):
            cal = None
        else:
            raise

    if cal is not None:
        if isinstance(cal, ShkFn):
            return call_shkfn(
                cal,
                positional,
                subject=recv,
                caller_frame=frame,
                named=named,
                call_site=call_site,
            )
        return call_value(
            cal,
            positional,
            frame,
            eval_func,
            named=named,
            interleaved=interleaved,
            call_node=op,
        )

    try:
        target = frame.get(method_name)
    except ShakarRuntimeError:
        raise ShakarMethodNotFound(recv, method_name) from None

    if not isinstance(
        target,
        (
            ShkFn,
            BoundMethod,
            BuiltinMethod,
            StdlibFunction,
            DecoratorContinuation,
            ShkDecorator,
            BoundCallable,
        ),
    ):
        raise ShakarTypeError(
            f"UFCS target '{method_name}' is not callable (got {type(target).__name__})"
        )

    style = "subject" if isinstance(target, StdlibFunction) else "prepend"
    ufcs_target = BoundCallable(
        target=target, subject=recv, style=style, name=method_name
    )

    return call_value(
        ufcs_target,
        positional,
        frame,
        eval_func,
        named=named,
        interleaved=interleaved,
        call_node=op,
    )


def _valuefan(base: ShkValue, op: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    """Evaluate value fanout `base.{...}` to an array of values."""
    return eval_valuefan(base, op, frame, eval_func, apply_op)


def _call_stdlib(
    cal: StdlibFunction,
    *,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    frame: Frame,
    call_site: Optional[CallSite],
    named: Optional[Dict[str, ShkValue]],
    interleaved: bool,
) -> ShkValue:
    call_arg_count = len(args) + (1 if subject is not None else 0)
    if cal.accepts_named:
        if interleaved:
            raise ShakarTypeError(
                "Positional arguments cannot appear on both sides of named arguments"
            )
        if cal.arity is not None and call_arg_count != cal.arity:
            raise ShakarArityError(
                f"Function expects {cal.arity} args; got {call_arg_count}"
            )
        named_args = named or {}
        return _with_call_site(
            frame, call_site, lambda: cal.fn(frame, subject, args, named_args)
        )

    if named:
        raise ShakarTypeError(
            f"Unexpected named argument(s): {', '.join(named.keys())}"
        )
    if cal.arity is not None and call_arg_count != cal.arity:
        raise ShakarArityError(
            f"Function expects {cal.arity} args; got {call_arg_count}"
        )
    return _with_call_site(frame, call_site, lambda: cal.fn(frame, subject, args, None))


def call_value(
    cal: ShkValue,
    args: List[ShkValue],
    frame: Frame,
    eval_func: EvalFunc,
    named: Optional[Dict[str, ShkValue]] = None,
    interleaved: bool = False,
    call_node: Optional[Node] = None,
) -> ShkValue:
    call_site = (
        callsite_from_node(_call_name_for_value(cal), call_node, frame)
        if call_node is not None
        else None
    )

    def _call_with_site(target: ShkValue, call_args: List[ShkValue]) -> ShkValue:
        match target:
            case BoundCallable(target=inner, subject=subject, style=style):
                if style == "prepend":
                    return _call_with_site(inner, [subject, *call_args])
                if style == "subject":
                    if not isinstance(inner, StdlibFunction):
                        raise ShakarTypeError(
                            "UFCS subject-style target is not a stdlib function"
                        )
                    return _call_stdlib(
                        inner,
                        subject=subject,
                        args=call_args,
                        frame=frame,
                        call_site=call_site,
                        named=named,
                        interleaved=interleaved,
                    )
                raise ShakarTypeError(f"Unknown UFCS binding style '{style}'")
            case BoundMethod(fn=fn, subject=subject):
                return call_shkfn(
                    fn,
                    call_args,
                    subject=subject,
                    caller_frame=frame,
                    named=named,
                    call_site=call_site,
                )
            case BuiltinMethod(name=name, subject=subject):
                if named:
                    raise ShakarTypeError(
                        "Builtin methods do not accept named arguments"
                    )
                return _with_call_site(
                    frame,
                    call_site,
                    lambda: call_builtin_method(subject, name, call_args, frame),
                )
            case StdlibFunction():
                return _call_stdlib(
                    target,
                    subject=None,
                    args=call_args,
                    frame=frame,
                    call_site=call_site,
                    named=named,
                    interleaved=interleaved,
                )
            case DecoratorContinuation():
                if named:
                    raise ShakarTypeError(
                        "Decorator continuations do not accept named arguments"
                    )
                if len(call_args) != 1:
                    raise ShakarArityError(
                        "Decorator continuation expects exactly 1 argument (the args array)"
                    )
                return _with_call_site(
                    frame, call_site, lambda: target.invoke(call_args[0])
                )
            case ShkDecorator():
                if named:
                    raise ShakarTypeError("Decorators do not accept named arguments")
                bound = _bind_decorator_params(target, call_args)
                return DecoratorConfigured(decorator=target, args=list(bound))
            case ShkFn():
                return call_shkfn(
                    target,
                    call_args,
                    subject=None,
                    caller_frame=frame,
                    named=named,
                    call_site=call_site,
                )
            case _:
                raise ShakarTypeError(
                    f"Cannot call value of type {type(target).__name__}"
                )

    return _call_with_site(cal, args)


def _index_expr_from_children(children: List[Node]) -> Tree:
    queue = list(children)
    idx = 0

    while idx < len(queue):
        node = queue[idx]
        idx += 1

        if is_token(node):
            continue

        if not is_tree(node):
            return node

        tag = tree_label(node)
        if tag in {"selectorlist", "selector", "indexsel"}:
            queue.extend(node.children)
            continue

        return node

    raise ShakarRuntimeError("Malformed index expression")


def _default_arg(node: Tree) -> Optional[Tree]:
    children = tree_children(node)
    selector_index = None

    for idx, child in enumerate(children):
        if tree_label(child) == "selectorlist":
            selector_index = idx
            break

    if selector_index is None:
        return None

    skip_tokens = {TT.RSQB, TT.COMMA, TT.COLON}

    for child in children[selector_index + 1 :]:
        if is_token(child) and child.type in skip_tokens:
            continue

        return child

    return None


def _is_single_index_selector(selectors: List[SelectorPart]) -> bool:
    return len(selectors) == 1 and isinstance(selectors[0], SelectorIndex)
