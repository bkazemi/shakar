from __future__ import annotations

from typing import Callable, List, Optional
from ..tree import Tok
from ..token_types import TT

from ..runtime import (
    BoundMethod,
    BuiltinMethod,
    DecoratorConfigured,
    DecoratorContinuation,
    EvalResult,
    Frame,
    ShkDecorator,
    ShkFn,
    ShkArray,
    ShkObject,
    ShkSelector,
    SelectorIndex,
    SelectorPart,
    ShkValue,
    ShakarArityError,
    ShakarMethodNotFound,
    ShakarRuntimeError,
    ShakarTypeError,
    StdlibFunction,
    _bind_decorator_params,
    call_builtin_method,
    call_shkfn,
)
from ..tree import Node, Tree, Tok, child_by_label, is_token, is_tree, tree_children, tree_label
from .bind import FanContext, RebindContext, apply_fan_op
from .common import expect_ident_token as _expect_ident_token
from .mutation import get_field_value, index_value, slice_value
from .selector import evaluate_selectorlist, apply_selectors_to_value
from .valuefan import eval_valuefan

EvalFunc = Callable[[Node, Frame], ShkValue]

def eval_args_node(args_node: Tree | list[Tree] | None, frame: Frame, eval_func: EvalFunc) -> List[ShkValue]:
    def label(node: Node) -> Optional[str]:
        return tree_label(node)

    def flatten(node: Node) -> List[Node]:
        if is_tree(node):
            tag = label(node)

            if tag in {'args', 'arglist', 'arglistnamedmixed'}:
                out: List[Node] = []

                for ch in node.children:
                    out.extend(flatten(ch))
                return out

            if tag in {'argitem', 'arg'} and node.children:
                return flatten(node.children[0])

            if tag == 'namedarg' and node.children:
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

def _eval_args(nodes: List[Node], frame: Frame, eval_func: EvalFunc) -> List[ShkValue]:
    values: List[ShkValue] = []

    for node in nodes:
        if _is_namedarg(node):
            # named args: evaluate value once; no auto-spread
            value_expr = node.children[-1] if node.children else None
            if value_expr is None:
                raise ShakarRuntimeError("Malformed named argument")
            values.append(eval_func(value_expr, frame))
            continue

        if _is_spread(node):
            spread_expr = node.children[0] if is_tree(node) and node.children else None
            if spread_expr is None:
                raise ShakarRuntimeError("Malformed spread argument")
            spread_val = eval_func(spread_expr, frame)
            if isinstance(spread_val, ShkArray):
                values.extend(spread_val.items)
                continue
            if isinstance(spread_val, ShkObject):
                values.extend(spread_val.slots.values())
                continue
            raise ShakarRuntimeError("Spread argument must be an array or object value")

        if _is_raw_fieldfan(node):
            spread_val = eval_func(node, frame)

            if not isinstance(spread_val, ShkArray):
                raise ShakarRuntimeError("Fanout argument did not produce an array value")

            values.extend(spread_val.items)
            continue

        values.append(eval_func(node, frame))

    return values

def _is_namedarg(node: Node) -> bool:
    return is_tree(node) and tree_label(node) == 'namedarg'

def _is_spread(node: Node) -> bool:
    return is_tree(node) and tree_label(node) == 'spread'

def _is_raw_fieldfan(node: Node) -> bool:
    """Detect `Base.{a,b}` with no trailing ops so we can auto-flatten in call args."""
    if not is_tree(node) or tree_label(node) != 'explicit_chain':
        return False

    ops = node.children[1:]
    if not (bool(ops) and len(ops) == 1):
        return False
    return tree_label(ops[0]) in {'fieldfan', 'valuefan'}

def evaluate_index_operand(index_node: Tree, frame: Frame, eval_func: EvalFunc) -> ShkSelector | ShkValue:
    selectorlist = child_by_label(index_node, 'selectorlist')

    if selectorlist is not None:
        selectors = evaluate_selectorlist(selectorlist, frame, eval_func)

        if len(selectors) == 1 and isinstance(selectors[0], SelectorIndex):
            return selectors[0].value

        return ShkSelector(selectors)

    expr_node = _index_expr_from_children(index_node.children)
    return eval_func(expr_node, frame)

def apply_slice(recv: ShkValue, arms: List[Node], frame: Frame, eval_func: EvalFunc) -> ShkValue:
    def arm_to_py(node: Node) -> Optional[int]:
        if tree_label(node) == 'emptyexpr':
            return None

        value = eval_func(node, frame)
        if hasattr(value, "value"):
            return int(value.value)
        return None

    start, stop, step = map(arm_to_py, arms)

    return slice_value(recv, start, stop, step)

def apply_index_operation(recv: ShkValue, op: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    selectorlist = child_by_label(op, 'selectorlist')
    default_node = _default_arg(op)
    default_thunk = (lambda: eval_func(default_node, frame)) if default_node is not None else None

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

def apply_op(recv: EvalResult, op: Tree, frame: Frame, eval_func: EvalFunc) -> EvalResult:
    if isinstance(recv, FanContext):
        return apply_fan_op(recv, op, frame, apply_op=apply_op, eval_func=eval_func)

    context = None

    if isinstance(recv, RebindContext):
        context = recv
        recv = context.value

    op_handlers: dict[str, Callable[[], EvalResult]] = {
        'field': lambda: _get_field(recv, op, frame),
        'fieldsel': lambda: _get_field(recv, op, frame),
        'index': lambda: apply_index_operation(recv, op, frame, eval_func),
        'lv_index': lambda: apply_index_operation(recv, op, frame, eval_func),
        'slicesel': lambda: apply_slice(recv, op.children, frame, eval_func),
        'fieldfan': lambda: apply_fan_op(recv, op, frame, apply_op=apply_op, eval_func=eval_func),
        'valuefan': lambda: _valuefan(recv, op, frame, eval_func),
        'call': lambda: call_value(recv, eval_args_node(op.children[0] if op.children else None, frame, eval_func), frame, eval_func),
        'method': lambda: _call_method(recv, op, frame, eval_func),
    }

    handler = op_handlers.get(op.data)
    if handler is None:
        raise ShakarRuntimeError(f"Unknown chain op: {op.data}")

    result = handler()

    if context is not None:
        context.value = result

        if not isinstance(result, (BuiltinMethod, BoundMethod, ShkFn)):
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
        meta = getattr(tok, 'meta', None)
        span = (getattr(meta, 'line', None), getattr(meta, 'column', None)) if meta else (None, None)
        raise ShakarRuntimeError(f"{err.args[0]} at {span}") from None
    return get_field_value(recv, field_name, frame)

def _call_method(recv: ShkValue, op: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    method_name = _expect_ident_token(op.children[0], "Method call")
    args = eval_args_node(op.children[1] if len(op.children) > 1 else None, frame, eval_func)

    try:
        return call_builtin_method(recv, method_name, args, frame)
    except ShakarMethodNotFound:
        cal = get_field_value(recv, method_name, frame)

        if isinstance(cal, BoundMethod):
            return call_shkfn(cal.fn, args, subject=cal.subject, caller_frame=frame)
        if isinstance(cal, ShkFn):
            return call_shkfn(cal, args, subject=recv, caller_frame=frame)
        raise

def _valuefan(base: ShkValue, op: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    """Evaluate value fanout `base.{...}` to an array of values."""
    return eval_valuefan(base, op, frame, eval_func, apply_op)

def call_value(cal: ShkValue, args: List[ShkValue], frame: Frame, eval_func: EvalFunc) -> ShkValue:
    match cal:
        case BoundMethod(fn=fn, subject=subject):
            return call_shkfn(fn, args, subject=subject, caller_frame=frame)
        case BuiltinMethod(name=name, subject=subject):
            return call_builtin_method(subject, name, args, frame)
        case StdlibFunction(fn=fn, arity=arity):
            if arity is not None and len(args) != arity:
                raise ShakarArityError(f"Function expects {arity} args; got {len(args)}")
            return fn(frame, args)
        case DecoratorContinuation():
            if len(args) != 1:
                raise ShakarArityError("Decorator continuation expects exactly 1 argument (the args array)")
            return cal.invoke(args[0])
        case ShkDecorator():
            params = cal.params or []
            varargs = cal.vararg_indices or []
            bound = _bind_decorator_params(params, varargs, args)
            return DecoratorConfigured(decorator=cal, args=list(bound))
        case ShkFn():
            return call_shkfn(cal, args, subject=None, caller_frame=frame)
        case _:
            raise ShakarTypeError(f"Cannot call value of type {type(cal).__name__}")

def _index_expr_from_children(children: List[Node]) -> Tree:
    queue = list(children)

    while queue:
        node = queue.pop(0)

        if is_token(node):
            continue

        if not is_tree(node):
            return node

        tag = tree_label(node)
        if tag in {'selectorlist', 'selector', 'indexsel'}:
            queue.extend(node.children)
            continue

        return node

    raise ShakarRuntimeError("Malformed index expression")

def _default_arg(node: Tree) -> Optional[Tree]:
    children = tree_children(node)
    selector_index = None

    for idx, child in enumerate(children):
        if tree_label(child) == 'selectorlist':
            selector_index = idx
            break

    if selector_index is None:
        return None

    skip_tokens = {TT.RSQB, TT.COMMA, TT.COLON}

    for child in children[selector_index + 1:]:
        if is_token(child) and child.type in skip_tokens:
            continue

        return child

    return None

def _is_single_index_selector(selectors: List[SelectorPart]) -> bool:
    return len(selectors) == 1 and isinstance(selectors[0], SelectorIndex)
