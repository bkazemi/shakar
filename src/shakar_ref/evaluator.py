from __future__ import annotations

from typing import Any, Callable, List, Optional
import math
from lark import Tree, Token

from .runtime import (
    BoundMethod,
    BuiltinMethod,
    DecoratorConfigured,
    DecoratorContinuation,
    Descriptor,
    Frame,
    ShkArray,
    ShkBool,
    ShkDecorator,
    ShkFn,
    ShkNull,
    ShkNumber,
    ShkObject,
    ShkSelector,
    ShkString,
    SelectorIndex,
    ShakarArityError,
    ShakarIndexError,
    ShakarMethodNotFound,
    ShakarReturnSignal,
    ShakarBreakSignal,
    ShakarContinueSignal,
    ShakarRuntimeError,
    ShakarTypeError,
    ShakarKeyError,
    StdlibFunction,
    call_builtin_method,
    call_shkfn,
    init_stdlib,
)

from .utils import shk_equals
from .tree import (
    child_by_label,
    is_token_node,
    is_tree_node,
    node_meta,
    tree_children,
    tree_label,
)

from .eval.selector import eval_selectorliteral, evaluate_selectorlist, apply_selectors_to_value, selector_iter_values

from .eval.mutation import (
    set_field_value,
    set_index_value,
    index_value,
    slice_value,
    get_field_value,
)
from .eval.control import (
    coerce_throw_value as _coerce_throw_value,
    eval_assert as _eval_assert,
    eval_return_if as _eval_return_if,
    eval_return_stmt as _eval_return_stmt,
    eval_throw_stmt as _eval_throw_stmt,
)
from .eval.helpers import (
    current_function_frame as _current_function_frame,
    is_truthy as _is_truthy,
    retargets_anchor as _retargets_anchor,
)

from .eval.blocks import (
    eval_program,
    eval_inline_body,
    eval_indent_block,
    eval_oneline_guard,
    eval_defer_stmt,
    get_subject,
    temporary_subject,
    temporary_bindings,
)

from .eval.postfix import (
    define_new_ident,
    eval_postfix_if as _postfix_eval_if,
    eval_postfix_unless as _postfix_eval_unless,
)

from .eval.loops import (
    eval_if_stmt,
    eval_for_in,
    eval_for_subject,
    eval_for_indexed,
    eval_for_map2,
    eval_listcomp,
    eval_setcomp,
    eval_setliteral,
    eval_dictcomp,
)
from .eval.destructure import eval_destructure

from .eval._await import eval_await_value, eval_await_stmt, eval_await_any_call, eval_await_all_call

from .eval.common import (
    token_kind as _token_kind,
    expect_ident_token as _expect_ident_token,
    ident_token_value as _ident_token_value,
    is_literal_node as _is_literal_node,
    get_source_segment as _get_source_segment,
    render_expr as _render_expr,
    node_source_span as _node_source_span,
    require_number as _require_number,
    token_number as _token_number,
    token_string as _token_string,
    stringify as _stringify,
    collect_free_identifiers as _collect_free_identifiers,
)

from .eval.bind import (
    FanContext,
    RebindContext,
    assign_ident,
    assign_lvalue,
    assign_pattern_value,
    apply_assign,
    apply_fan_op,
    apply_numeric_delta,
    build_fieldfan_context,
    read_lvalue,
    resolve_assignable_node,
    resolve_chain_assignment,
    resolve_rebind_lvalue,
)

# ---------------- Public API ----------------

def eval_expr(ast: Any, frame: Optional[Frame]=None, source: Optional[str]=None) -> Any:
    init_stdlib()

    if frame is None:
        frame = Frame(source=source)
    else:
        if source is not None:
            frame.source = source
        elif not hasattr(frame, 'source'):
            frame.source = None

    return eval_node(ast, frame)

# ---------------- Core evaluator ----------------

def eval_node(n: Any, frame: Frame) -> Any:
    if _is_literal_node(n):
        return n

    if is_token_node(n):
        return _eval_token(n, frame)

    d = n.data
    handler = _NODE_DISPATCH.get(d)
    if handler is not None:
        return handler(n, frame)

    match d:
        # common wrapper nodes (delegate to single child)
        case 'start_noindent' | 'start_indented' | 'stmtlist':
            return eval_program(n.children, frame, eval_node)
        case 'stmt':
            return eval_program(n.children, frame, eval_node)
        case 'literal' | 'primary' | 'expr' | 'expr_nc':
            if len(n.children) == 1:
                return eval_node(n.children[0], frame)

            if len(n.children) == 0 and d == 'literal':
                return _eval_keyword_literal(n)
            raise ShakarRuntimeError(f"Unsupported wrapper shape {d} with {len(n.children)} children")
        case 'array':
            return ShkArray([eval_node(c, frame) for c in n.children])
        case 'object':
            return _eval_object(n, frame)
        case 'unary' | 'unary_nc':
            op, rhs_node = n.children
            return _eval_unary(op, rhs_node, frame)
        case 'pow' | 'pow_nc':
            return _eval_infix(n.children, frame, right_assoc_ops={'**', 'POW'})
        case 'mul' | 'mul_nc' | 'add' | 'add_nc':
            return _eval_infix(n.children, frame)
        case 'explicit_chain':
            head, *ops = n.children

            if ops and tree_label(ops[-1]) in {'incr', 'decr'}:
                tail = ops[-1]
                # ++/-- mutate the final receiver; resolve assignable context first.
                context = resolve_chain_assignment(
                    head,
                    ops[:-1],
                    frame,
                    eval_func=eval_node,
                    apply_op=_apply_op,
                    evaluate_index_operand=_evaluate_index_operand,
                )

                delta = 1 if tree_label(tail) == 'incr' else -1

                if isinstance(context, FanContext):
                    raise ShakarRuntimeError("++/-- not supported on field fan assignments")
                old_val, _ = apply_numeric_delta(context, delta)
                return old_val

            val = eval_node(head, frame)
            head_label = tree_label(head) if is_tree_node(head) else None
            head_is_rebind = head_label in {'rebind_primary', 'rebind_primary_grouped'}
            head_is_grouped_rebind = head_label == 'rebind_primary_grouped'
            tail_has_effect = False

            for op in ops:
                label = tree_label(op)
                if label not in {'field', 'fieldsel', 'index'}:
                    tail_has_effect = True

                val = _apply_op(val, op, frame)

            if head_is_rebind:
                if not ops:
                    raise ShakarRuntimeError("Prefix rebind requires a tail expression")

                if not head_is_grouped_rebind and not tail_has_effect:
                    raise ShakarRuntimeError("Prefix rebind requires a tail expression")

            if isinstance(val, RebindContext):
                final = val.value
                val.setter(final)
                return final

            if isinstance(val, FanContext):
                return ShkArray(val.snapshot())
            return val
        case 'implicit_chain':
            return _eval_implicit_chain(n.children, frame)
        case 'call':
            args_node = n.children[0] if n.children else None
            args = _eval_args_node(args_node, frame)
            cal = frame.get('')  # unreachable in practice
            return _call_value(cal, args, frame)
        case 'and' | 'or' | 'and_nc' | 'or_nc':
            return _eval_logical(d, n.children, frame)
        case 'walrus' | 'walrus_nc':
            return _eval_walrus(n.children, frame)
        case 'returnstmt':
            return _eval_return_stmt(n.children, frame, eval_func=eval_node)
        case 'returnif':
            return _eval_return_if(n.children, frame, eval_func=eval_node)
        case 'throwstmt':
            return _eval_throw_stmt(n.children, frame, eval_func=eval_node)
        case 'breakstmt':
            return _eval_break_stmt(frame)
        case 'continuestmt':
            return _eval_continue_stmt(frame)
        case 'assignstmt':
            return _eval_assign_stmt(n.children, frame)
        case 'postfixif':
            return _postfix_eval_if(n.children, frame, eval_func=eval_node, truthy_fn=_is_truthy)
        case 'postfixunless':
            return _postfix_eval_unless(n.children, frame, eval_func=eval_node, truthy_fn=_is_truthy)
        case 'compound_assign':
            return _eval_compound_assign(n.children, frame)
        case 'fndef':
            return _eval_fn_def(n.children, frame)
        case 'decorator_def':
            return _eval_decorator_def(n.children, frame)
        case 'deferstmt':
            return eval_defer_stmt(n.children, frame, eval_node)
        case 'assert':
            return _eval_assert(n.children, frame, eval_func=eval_node)
        case 'bind' | 'bind_nc':
            return _eval_apply_assign(n.children, frame)
        case 'subject':
            return get_subject(frame)
        case 'keyexpr' | 'keyexpr_nc':
            return eval_node(n.children[0], frame) if n.children else ShkNull()
        case 'destructure':
            return eval_destructure(n, frame, eval_node, create=False, allow_broadcast=False)
        case 'destructure_walrus':
            return eval_destructure(n, frame, eval_node, create=True, allow_broadcast=True)
        case _:
            raise ShakarRuntimeError(f"Unknown node: {d}")

# ---------------- Tokens ----------------

def _eval_token(t: Token, frame: Frame) -> Any:
    handler = _TOKEN_DISPATCH.get(t.type)
    if handler:
        return handler(t, frame)

    if t.type == 'IDENT':
        return frame.get(t.value)

    raise ShakarRuntimeError(f"Unhandled token {t.type}:{t.value}")

def _eval_keyword_literal(node: Tree) -> Any:
    meta = node_meta(node)
    if meta is None:
        raise ShakarRuntimeError("Missing metadata for literal")

    start = getattr(meta, "start_pos", None)
    end = getattr(meta, "end_pos", None)
    if start is None or end is None:
        raise ShakarRuntimeError("Missing source span for literal")

    width = end - start
    match width:
        case 3:
            return ShkNull()
        case 4:
            return ShkBool(True)
        case 5:
            return ShkBool(False)

    raise ShakarRuntimeError("Unknown literal")

def _eval_implicit_chain(ops: List[Any], frame: Frame) -> Any:
    """Evaluate `.foo().bar` style chains using the current subject anchor."""
    val = get_subject(frame)

    for op in ops:
        val = _apply_op(val, op, frame)

    return val

# ---------------- Assignment ----------------

def _eval_walrus(children: List[Any], frame: Frame) -> Any:
    """Implements `name := expr` inline assignments."""

    if len(children) != 2:
        raise ShakarRuntimeError("Malformed walrus expression")

    name_node, value_node = children
    name = _expect_ident_token(name_node, "Walrus target")
    value = eval_node(value_node, frame)

    return define_new_ident(name, value, frame)

def _eval_assign_stmt(children: List[Any], frame: Frame) -> Any:
    """Handles simple `lhs = rhs` statements (no destructuring)."""

    if len(children) < 2:
        raise ShakarRuntimeError("Malformed assignment statement")

    lvalue_node = children[0]
    value_node = children[-1]
    value = eval_node(value_node, frame)
    assign_lvalue(
        lvalue_node,
        value,
        frame,
        eval_func=eval_node,
        apply_op=_apply_op,
        evaluate_index_operand=_evaluate_index_operand,
        create=False,
    )

    return ShkNull()

def _build_error_payload(exc: ShakarRuntimeError) -> ShkObject:
    """Expose exception metadata to catch handlers as a lightweight object."""
    payload = getattr(exc, 'shk_payload', None)

    if isinstance(payload, ShkObject):
        return payload

    slots = {
        "message": ShkString(str(exc)),
        "type": ShkString(getattr(exc, 'shk_type', type(exc).__name__)),
    }

    if isinstance(exc, ShakarKeyError):
        slots["key"] = ShkString(str(exc.key))

    if isinstance(exc, ShakarMethodNotFound):
        slots["method"] = ShkString(exc.name)

    data = getattr(exc, 'shk_data', None)
    if data is not None:
        slots["data"] = data

    return ShkObject(slots)

def _parse_catch_components(children: List[Any]) -> tuple[Any, Any | None, List[str], Any]:
    """Split canonical catch nodes into try expression, binder token, type list, and handler."""

    if not children:
        raise ShakarRuntimeError("Malformed catch node")

    idx = 0
    try_node = children[idx]
    idx += 1
    binder = None
    type_names: List[str] = []

    if idx < len(children) and is_token_node(children[idx]):
        binder = children[idx]
        idx += 1

    if idx < len(children) and is_tree_node(children[idx]) and tree_label(children[idx]) == 'catchtypes':
        type_names = [
            _expect_ident_token(tok, "Catch type")
            for tok in tree_children(children[idx]) if is_token_node(tok)
        ]
        idx += 1

    if idx >= len(children):
        raise ShakarRuntimeError("Missing catch handler")

    handler = children[idx]

    return try_node, binder, type_names, handler

def _run_catch_handler(
    handler: Any,
    frame: Frame,
    binder: Any | None,
    payload: ShkObject,
    original_exc: ShakarRuntimeError,
    allowed_types: List[str],
) -> Any:
    # type matching happens before we evaluate the handler body so unmatched errors bubble up
    payload_type = None

    if isinstance(payload, ShkObject):
        slot = payload.slots.get('type') if hasattr(payload, 'slots') else None

        if isinstance(slot, ShkString):
            payload_type = slot.value
        elif isinstance(slot, str):
            payload_type = slot

    if allowed_types:
        if not isinstance(payload_type, str) or payload_type not in allowed_types:
            raise original_exc

    binder_name = None
    if binder is not None:
        binder_name = _expect_ident_token(binder, "Catch binder")

    def _exec_handler() -> Any:
        label = tree_label(handler)

        if label == 'inlinebody':
            return eval_inline_body(handler, frame, eval_node)

        if label == 'indentblock':
            return eval_indent_block(handler, frame, eval_node)

        return eval_node(handler, frame)

    # track the currently handled exception so a bare `throw` can rethrow it later
    prev_error = getattr(frame, '_active_error', None)
    frame._active_error = original_exc

    try:
        with temporary_subject(frame, payload):
            if binder_name:
                with temporary_bindings(frame, {binder_name: payload}):
                    return _exec_handler()
            return _exec_handler()
    finally:
        frame._active_error = prev_error

def _eval_catch_expr(children: List[Any], frame: Frame) -> Any:
    try_node, binder, type_names, handler = _parse_catch_components(children)

    try:
        return eval_node(try_node, frame)
    except ShakarRuntimeError as exc:
        payload = _build_error_payload(exc)
        return _run_catch_handler(handler, frame, binder, payload, exc, type_names)

def _eval_catch_stmt(children: List[Any], frame: Frame) -> Any:
    try_node, binder, type_names, body = _parse_catch_components(children)

    try:
        eval_node(try_node, frame)
        return ShkNull()
    except ShakarRuntimeError as exc:
        payload = _build_error_payload(exc)
        _run_catch_handler(body, frame, binder, payload, exc, type_names)
        return ShkNull()

def _eval_break_stmt(frame: Frame) -> Any:
    raise ShakarBreakSignal()

def _eval_continue_stmt(frame: Frame) -> Any:
    raise ShakarContinueSignal()

def _eval_fn_def(children: List[Any], frame: Frame) -> Any:
    if not children:
        raise ShakarRuntimeError("Malformed function definition")

    name = _expect_ident_token(children[0], "Function name")
    params_node = None
    body_node = None
    decorators_node = None

    for node in children[1:]:
        if params_node is None and is_tree_node(node) and tree_label(node) == 'paramlist':
            params_node = node
        elif body_node is None and is_tree_node(node) and tree_label(node) in {'inlinebody', 'indentblock'}:
            body_node = node
        elif decorators_node is None and is_tree_node(node) and tree_label(node) == 'decorator_list':
            decorators_node = node

    if body_node is None:
        body_node = Tree('inlinebody', [])

    params = _extract_param_names(params_node, context="function definition")
    fn_value = ShkFn(params=params, body=body_node, frame=Frame(parent=frame, dot=None))

    if decorators_node is not None:
        instances = _evaluate_decorator_list(decorators_node, frame)
        if instances:
            fn_value.decorators = tuple(reversed(instances))

    assign_ident(name, fn_value, frame, create=True)

    return ShkNull()

def _eval_decorator_def(children: List[Any], frame: Frame) -> Any:
    if not children:
        raise ShakarRuntimeError("Malformed decorator definition")

    name = _expect_ident_token(children[0], "Decorator name")
    params_node = None
    body_node = None

    for node in children[1:]:
        if params_node is None and is_tree_node(node) and tree_label(node) == 'paramlist':
            params_node = node
        elif body_node is None and is_tree_node(node) and tree_label(node) in {'inlinebody', 'indentblock'}:
            body_node = node

    if body_node is None:
        body_node = Tree('inlinebody', [])

    params = _extract_param_names(params_node, context="decorator definition")
    decorator = ShkDecorator(params=params, body=body_node, frame=Frame(parent=frame, dot=None))
    assign_ident(name, decorator, frame, create=True)

    return ShkNull()

def _evaluate_decorator_list(node: Tree, frame: Frame) -> List[DecoratorConfigured]:
    configured: List[DecoratorConfigured] = []

    for entry in tree_children(node):
        kids = tree_children(entry)
        expr_node = kids[0] if kids else None

        if expr_node is None:
            continue

        value = eval_node(expr_node, frame)
        configured.append(_coerce_decorator_instance(value))

    return configured

def _coerce_decorator_instance(value: Any) -> DecoratorConfigured:
    match value:
        case DecoratorConfigured():
            return DecoratorConfigured(decorator=value.decorator, args=list(value.args))
        case ShkDecorator(params=params):
            arity = len(params) if params is not None else 0

            if arity:
                raise ShakarRuntimeError("Decorator requires arguments; call it with parentheses")
            return DecoratorConfigured(decorator=value, args=[])
        case _:
            raise ShakarTypeError("Decorator expression must evaluate to a decorator")

def _eval_anonymous_fn(children: List[Any], frame: Frame) -> ShkFn:
    params_node = None
    body_node = None

    for node in children:
        if is_tree_node(node) and tree_label(node) == 'paramlist':
            params_node = node
        elif is_tree_node(node) and tree_label(node) in {'inlinebody', 'indentblock'}:
            body_node = node

    if body_node is None:
        body_node = Tree('inlinebody', [])

    params = _extract_param_names(params_node, context="anonymous function")

    return ShkFn(params=params, body=body_node, frame=Frame(parent=frame, dot=None))

def _eval_compound_assign(children: List[Any], frame: Frame) -> Any:
    """Handle `x += y` and friends."""
    if len(children) < 3:
        raise ShakarRuntimeError("Malformed compound assignment")

    lvalue_node = children[0]
    rhs_node = children[-1]
    op_token = next((child for child in children[1:-1] if is_token_node(child)), None)

    if op_token is None:
        raise ShakarRuntimeError("Compound assignment missing operator")

    op_value = op_token.value
    op_symbol = _COMPOUND_ASSIGN_OPERATORS.get(op_value)

    if op_symbol is None:
        raise ShakarRuntimeError(f"Unsupported compound operator {op_value}")

    current_value = read_lvalue(
        lvalue_node,
        frame,
        eval_func=eval_node,
        apply_op=_apply_op,
        evaluate_index_operand=_evaluate_index_operand,
    )
    rhs_value = eval_node(rhs_node, frame)
    new_value = _apply_binary_operator(op_symbol, current_value, rhs_value)
    assign_lvalue(
        lvalue_node,
        new_value,
        frame,
        eval_func=eval_node,
        apply_op=_apply_op,
        evaluate_index_operand=_evaluate_index_operand,
        create=False,
    )

    return ShkNull()

def _eval_apply_assign(children: List[Any], frame: Frame) -> Any:
    """Evaluate the `.= expr` apply-assign form (subject-aware updates)."""
    lvalue_node = None
    rhs_node = None

    for child in children:
        if tree_label(child) == 'lvalue':
            lvalue_node = child
        elif is_tree_node(child):
            rhs_node = child

    if lvalue_node is None or rhs_node is None:
        raise ShakarRuntimeError("Malformed apply-assign expression")

    return apply_assign(
        lvalue_node,
        rhs_node,
        frame,
        eval_func=eval_node,
        apply_op=_apply_op,
        evaluate_index_operand=_evaluate_index_operand,
    )

def _evaluate_index_operand(index_node: Tree, frame: Frame) -> Any:
    selectorlist = child_by_label(index_node, 'selectorlist')

    if selectorlist is not None:
        selectors = evaluate_selectorlist(selectorlist, frame, eval_node)

        if len(selectors) == 1 and isinstance(selectors[0], SelectorIndex):
            return selectors[0].value

        return ShkSelector(selectors)

    expr_node = _index_expr_from_children(index_node.children)
    return eval_node(expr_node, frame)

# ---------------- Comparison ----------------

def _eval_compare(children: List[Any], frame: Frame) -> Any:
    if not children:
        return ShkNull()

    if len(children) == 1:
        return eval_node(children[0], frame)

    subject = eval_node(children[0], frame)
    idx = 1
    joiner = 'and'
    last_comp: Optional[str] = None
    agg: Optional[bool] = None

    while idx < len(children):
        node = children[idx]
        tok = _token_kind(node)

        if tok == 'AND':
            joiner = 'and'
            idx += 1
            continue
        if tok == 'OR':
            joiner = 'or'
            idx += 1
            continue

        label = tree_label(node)
        if label == 'cmpop':
            comp = _as_op(node)
            last_comp = comp
            idx += 1

            if idx >= len(children):
                raise ShakarRuntimeError("Missing right-hand side for comparison")
            rhs = eval_node(children[idx], frame)
            idx += 1
        else:
            if last_comp is None:
                raise ShakarRuntimeError("Comparator required in comparison chain")

            comp = last_comp
            rhs = eval_node(node, frame)
            idx += 1

        leg_val = _compare_values(comp, subject, rhs)

        if agg is None:
            agg = leg_val
        else:
            agg = (agg and leg_val) if joiner == 'and' else (agg or leg_val)

    return ShkBool(bool(agg)) if agg is not None else ShkBool(True)

def _compare_values(op: str, lhs: Any, rhs: Any) -> bool:
    if isinstance(rhs, ShkSelector):
        return _compare_with_selector(op, lhs, rhs)

    match op:
        case '==':
            return shk_equals(lhs, rhs)
        case '!=':
            return not shk_equals(lhs, rhs)
        case '<':
            _require_number(lhs); _require_number(rhs)
            return lhs.value < rhs.value
        case '<=':
            _require_number(lhs); _require_number(rhs)
            return lhs.value <= rhs.value
        case '>':
            _require_number(lhs); _require_number(rhs)
            return lhs.value > rhs.value
        case '>=':
            _require_number(lhs); _require_number(rhs)
            return lhs.value >= rhs.value
        case 'is':
            return shk_equals(lhs, rhs)
        case '!is' | 'is not':
            return not shk_equals(lhs, rhs)
        case 'in':
            return _contains(rhs, lhs)
        case 'not in' | '!in':
            return not _contains(rhs, lhs)
        case _:
            raise ShakarRuntimeError(f"Unknown comparator {op}")

def _selector_values(selector: ShkSelector) -> List[ShkNumber]:
    values = selector_iter_values(selector)
    if not values:
        raise ShakarRuntimeError("Selector literal produced no values")

    return values

def _coerce_number(value: Any) -> float:
    if isinstance(value, ShkNumber):
        return value.value

    raise ShakarTypeError("Expected number")

def _compare_with_selector(op: str, lhs: Any, selector: ShkSelector) -> bool:
    values = _selector_values(selector)

    if op == '==':
        return all(shk_equals(lhs, val) for val in values)

    if op == '!=':
        return any(not shk_equals(lhs, val) for val in values)

    lhs_num = _coerce_number(lhs)
    rhs_nums = [_coerce_number(val) for val in values]

    match op:
        case '<':
            return all(lhs_num < num for num in rhs_nums)
        case '<=':
            return all(lhs_num <= num for num in rhs_nums)
        case '>':
            return all(lhs_num > num for num in rhs_nums)
        case '>=':
            return all(lhs_num >= num for num in rhs_nums)
        case _:
            raise ShakarTypeError(f"Unsupported comparator '{op}' for selector literal")

def _contains(container: Any, item: Any) -> bool:
    match container:
        case ShkArray(items=items):
            return any(shk_equals(element, item) for element in items)
        case ShkString(value=text):
            if isinstance(item, ShkString):
                return item.value in text
            raise ShakarTypeError("String membership requires a string value")
        case ShkObject(slots=slots):
            if isinstance(item, ShkString):
                return item.value in slots
            raise ShakarTypeError("Object membership requires a string key")
        case _:
            raise ShakarTypeError(f"Unsupported container type for 'in': {type(container).__name__}")

def _eval_logical(kind: str, children: List[Any], frame: Frame) -> Any:
    if not children:
        return ShkNull()

    normalized = 'and' if 'and' in kind else 'or'
    prev_dot = frame.dot

    try:
        if normalized == 'and':
            last_val: Any = ShkBool(True)

            for child in children:
                if is_token_node(child) and _token_kind(child) in {'AND', 'OR'}:
                    continue

                val = eval_node(child, frame)
                frame.dot = val if _retargets_anchor(child) else frame.dot
                last_val = val

                if not _is_truthy(val):
                    return val
            return last_val

        last_val: Any = ShkBool(False)

        for child in children:
            if is_token_node(child) and _token_kind(child) in {'AND', 'OR'}:
                continue

            val = eval_node(child, frame)
            frame.dot = val if _retargets_anchor(child) else frame.dot
            last_val = val

            if _is_truthy(val):
                return val
        return last_val
    finally:
        frame.dot = prev_dot

def _eval_nullish(children: List[Any], frame: Frame) -> Any:
    exprs = [child for child in children if not (is_token_node(child) and _token_kind(child) == 'NULLISH')]

    if not exprs:
        return ShkNull()

    current = eval_node(exprs[0], frame)

    for expr in exprs[1:]:
        if not isinstance(current, ShkNull):
            return current
        current = eval_node(expr, frame)

    return current

def _eval_nullsafe(children: List[Any], frame: Frame) -> Any:
    if not children:
        return ShkNull()

    expr = children[0]
    saved_dot = frame.dot

    try:
        return _eval_chain_nullsafe(expr, frame)
    finally:
        frame.dot = saved_dot

def _eval_chain_nullsafe(node: Any, frame: Frame) -> Any:
    if not is_tree_node(node) or tree_label(node) != 'explicit_chain':
        return eval_node(node, frame)

    children = tree_children(node)

    if not children:
        return ShkNull()

    head = children[0]
    current = eval_node(head, frame)

    if isinstance(current, ShkNull):
        return ShkNull()

    for op in children[1:]:
        try:
            current = _apply_op(current, op, frame)
        except (ShakarRuntimeError, ShakarTypeError) as err:
            if _nullsafe_recovers(err, current):
                return ShkNull()
            raise

        if isinstance(current, RebindContext):
            current = current.value

        if isinstance(current, ShkNull):
            return ShkNull()
    return current

def _nullsafe_recovers(err: Exception, recv: Any) -> bool:
    if isinstance(recv, ShkNull):
        return True
    return isinstance(err, (ShakarKeyError, ShakarIndexError))

# ---------------- Arithmetic ----------------

def _normalize_unary_op(op_node: Any, frame: Frame) -> Any:
    if tree_label(op_node) == 'unaryprefixop':
        if op_node.children:
            return _normalize_unary_op(op_node.children[0], frame)
        src = getattr(frame, 'source', None)
        meta = node_meta(op_node)

        if src is not None and meta is not None:
            return src[meta.start_pos:meta.end_pos]
        return ''
    return op_node

def _eval_unary(op_node: Any, rhs_node: Any, frame: Frame) -> Any:
    op_norm = _normalize_unary_op(op_node, frame)
    op_value = op_norm.value if isinstance(op_norm, Token) else op_norm

    if op_value in ('++', '--'):
        context = resolve_assignable_node(
            rhs_node,
            frame,
            eval_func=eval_node,
            apply_op=_apply_op,
            evaluate_index_operand=_evaluate_index_operand,
        )
        _, new_val = apply_numeric_delta(context, 1 if op_value == '++' else -1)
        return new_val

    rhs = eval_node(rhs_node, frame)

    match op_norm:
        case Token(type='PLUS') | '+':
            raise ShakarRuntimeError("unary + not supported")
        case Token(type='MINUS') | '-':
            _require_number(rhs)
            return ShkNumber(-rhs.value)
        case Token(type='TILDE') | '~':
            raise ShakarRuntimeError("bitwise ~ not supported yet")
        case Token(type='NOT') | 'not':
            return ShkBool(not _is_truthy(rhs))
        case Token(type='NEG') | '!':
            return ShkBool(not _is_truthy(rhs))
        case _:
            raise ShakarRuntimeError("Unsupported unary op")

def _eval_infix(children: List[Any], frame: Frame, right_assoc_ops: set|None=None) -> Any:
    if not children:
        return ShkNull()

    if right_assoc_ops and _all_ops_in(children, right_assoc_ops):
        vals = [eval_node(children[i], frame) for i in range(0, len(children), 2)]
        acc = vals[-1]

        for i in range(len(vals)-2, -1, -1):
            lhs, rhs = vals[i], acc
            _require_number(lhs); _require_number(rhs)
            acc = ShkNumber(lhs.value ** rhs.value)
        return acc

    it = iter(children)
    acc = eval_node(next(it), frame)

    for x in it:
        op = _as_op(x)
        rhs = eval_node(next(it), frame)
        acc = _apply_binary_operator(op, acc, rhs)

    return acc

def _all_ops_in(children: List[Any], allowed: set) -> bool:
    return all(_as_op(children[i]) in allowed for i in range(1, len(children), 2))

def _as_op(x: Any) -> str:
    if is_token_node(x):
        return x.value

    label = tree_label(x)
    if label is not None:
        # e.g. Tree('addop', [Token('PLUS','+')]) or mulop/powop
        if label in ('addop', 'mulop', 'powop') and len(x.children) == 1 and is_token_node(x.children[0]):
            return x.children[0].value

        if label == 'cmpop':
            tokens = [tok.value for tok in x.children if is_token_node(tok)]
            if not tokens:
                raise ShakarRuntimeError("Empty comparison operator")

            if tokens[0] == '!' and len(tokens) > 1:
                return '!' + ''.join(tokens[1:])

            if len(tokens) > 1:
                return " ".join(tokens)
            return tokens[0]

    raise ShakarRuntimeError(f"Expected operator token, got {x!r}")

_COMPOUND_ASSIGN_OPERATORS: dict[str, str] = {
    '+=': '+',
    '-=': '-',
    '*=': '*',
    '/=': '/',
    '//=': '//',
    '%=': '%',
    '**=': '**',
}

def _apply_binary_operator(op: str, lhs: Any, rhs: Any) -> Any:
    match op:
        case '+':
            if isinstance(lhs, ShkString) or isinstance(rhs, ShkString):
                return ShkString(_stringify(lhs) + _stringify(rhs))
            _require_number(lhs); _require_number(rhs)
            return ShkNumber(lhs.value + rhs.value)
        case '-':
            _require_number(lhs); _require_number(rhs)
            return ShkNumber(lhs.value - rhs.value)
        case '*':
            _require_number(lhs); _require_number(rhs)
            return ShkNumber(lhs.value * rhs.value)
        case '/':
            _require_number(lhs); _require_number(rhs)
            return ShkNumber(lhs.value / rhs.value)
        case '//':
            _require_number(lhs); _require_number(rhs)
            return ShkNumber(math.floor(lhs.value / rhs.value))
        case '%':
            _require_number(lhs); _require_number(rhs)
            return ShkNumber(lhs.value % rhs.value)
        case '**':
            _require_number(lhs); _require_number(rhs)
            return ShkNumber(lhs.value ** rhs.value)
    raise ShakarRuntimeError(f"Unknown operator {op}")

def _token_number(t: Token, _: Frame) -> ShkNumber:
    return ShkNumber(float(t.value))

def _token_string(t: Token, _: Frame) -> ShkString:
    v = t.value

    if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
        v = v[1:-1]

    return ShkString(v)

def _eval_string_interp(node: Tree, frame: Frame) -> ShkString:
    parts: list[str] = []

    for part in tree_children(node):
        if is_token_node(part):
            parts.append(part.value)
            continue

        if tree_label(part) == 'string_interp_expr':
            expr_node = part.children[0] if tree_children(part) else None
            if expr_node is None:
                raise ShakarRuntimeError("Empty interpolation expression")

            value = eval_node(expr_node, frame)
            parts.append(_stringify(value))
            continue

        raise ShakarRuntimeError("Unexpected node in string interpolation literal")

    return ShkString("".join(parts))

def _extract_param_names(params_node: Any, context: str="parameter list") -> List[str]:
    if params_node is None:
        return []

    names: List[str] = []

    for p in tree_children(params_node):
        name = _ident_token_value(p)

        if name is not None:
            names.append(name)
            continue
        if _token_kind(p) == 'COMMA':
            continue

        raise ShakarRuntimeError(f"Unsupported parameter node in {context}: {p}")

    return names

# ---------------- Chains ----------------

def _apply_op(recv: Any, op: Tree, frame: Frame) -> Any:
    """Apply one explicit-chain operation (call/index/member) to `recv`."""

    if isinstance(recv, FanContext):
        return apply_fan_op(recv, op, frame, apply_op=_apply_op)

    context = None

    if isinstance(recv, RebindContext):
        context = recv
        recv = context.value

    d = op.data

    if d in {'field', 'fieldsel'}:
        field_name = _expect_ident_token(op.children[0], "Field access")
        result = get_field_value(recv, field_name, frame)
    elif d == 'index':
        result = _apply_index_operation(recv, op, frame)
    elif d == 'slicesel':
        result = _apply_slice(recv, op.children, frame)
    elif d == 'fieldfan':
        return build_fieldfan_context(recv, op, frame)
    elif d == 'call':
        args = _eval_args_node(op.children[0] if op.children else None, frame)
        result = _call_value(recv, args, frame)
    elif d == 'method':
        method_name = _expect_ident_token(op.children[0], "Method call")
        args = _eval_args_node(op.children[1] if len(op.children)>1 else None, frame)

        try:
            result = call_builtin_method(recv, method_name, args, frame)
        except ShakarMethodNotFound:
            cal = get_field_value(recv, method_name, frame)

            if isinstance(cal, BoundMethod):
                result = call_shkfn(cal.fn, args, subject=cal.subject, caller_frame=frame)
            elif isinstance(cal, ShkFn):
                result = call_shkfn(cal, args, subject=recv, caller_frame=frame)
            else:
                raise
    else:
        raise ShakarRuntimeError(f"Unknown chain op: {d}")

    if context is not None:
        context.value = result

        if not isinstance(result, (BuiltinMethod, BoundMethod, ShkFn)):
            context.setter(result)
        return context

    return result

def _eval_args_node(args_node: Any, frame: Frame) -> List[Any]:
    def label(node: Tree) -> str | None:
        return tree_label(node)

    def flatten(node: Any) -> List[Any]:
        if is_tree_node(node):
            tag = label(node)

            if tag in {'args', 'arglist', 'arglistnamedmixed'}:
                out: List[Any] = []

                for ch in node.children:
                    out.extend(flatten(ch))
                return out

            if tag in {'argitem', 'arg'} and node.children:
                return flatten(node.children[0])

            if tag == 'namedarg' and node.children:
                # ignore the name for now; evaluate the value node
                return flatten(node.children[-1])

        return [node]

    if is_tree_node(args_node):
        return [eval_node(n, frame) for n in flatten(args_node)]

    if isinstance(args_node, list):  # deliberate: args_node can be pure python list from visitors
        res: List[Any] = []
        for n in args_node:
            res.extend(flatten(n))
        return [eval_node(n, frame) for n in res]

    return []

def _index_expr_from_children(children: List[Any]) -> Any:
    queue = list(children)

    while queue:
        node = queue.pop(0)

        if is_token_node(node):
            continue

        if not is_tree_node(node):
            return node

        tag = tree_label(node)
        if tag in {'selectorlist', 'selector', 'indexsel'}:
            queue.extend(node.children)
            continue

        return node

    raise ShakarRuntimeError("Malformed index expression")

def _apply_slice(recv: Any, arms: List[Any], frame: Frame) -> Any:
    def arm_to_py(node: Any) -> int | None:
        if tree_label(node) == 'emptyexpr':
            return None

        value = eval_node(node, frame)
        if isinstance(value, ShkNumber):
            return int(value.value)
        return None

    start, stop, step = map(arm_to_py, arms)

    return slice_value(recv, start, stop, step)

# ---------------- Await primitives ----------------

# ---------------- Selectors ----------------

def _apply_index_operation(recv: Any, op: Tree, frame: Frame) -> Any:
    selectorlist = child_by_label(op, 'selectorlist')

    if selectorlist is None:
        expr_node = _index_expr_from_children(op.children)
        idx_val = eval_node(expr_node, frame)
        return index_value(recv, idx_val, frame)

    with temporary_subject(frame, recv):
        selectors = evaluate_selectorlist(selectorlist, frame, eval_node)

    if len(selectors) == 1 and isinstance(selectors[0], SelectorIndex):
        return index_value(recv, selectors[0].value, frame)

    return apply_selectors_to_value(recv, selectors)

def _eval_group(n: Tree, frame: Frame) -> Any:
    child = n.children[0] if n.children else None
    if child is None:
        return ShkNull()

    saved = frame.dot

    try:
        return eval_node(child, frame)
    finally:
        frame.dot = saved

def _eval_ternary(n: Tree, frame: Frame) -> Any:
    if len(n.children) != 3:
        raise ShakarRuntimeError("Malformed ternary expression")

    cond_node, true_node, false_node = n.children
    cond_val = eval_node(cond_node, frame)

    if _is_truthy(cond_val):
        return eval_node(true_node, frame)

    return eval_node(false_node, frame)

def _eval_rebind_primary(n: Tree, frame: Frame) -> Any:
    if not n.children:
        raise ShakarRuntimeError("Missing target for prefix rebind")

    target = n.children[0]
    ctx = resolve_rebind_lvalue(
        target,
        frame,
        eval_func=eval_node,
        apply_op=_apply_op,
        evaluate_index_operand=_evaluate_index_operand,
    )
    frame.dot = ctx.value

    return ctx

def _eval_optional_expr(node: Any, frame: Frame) -> Any:
    if node is None:
        return None

    return eval_node(node, frame)

def _call_value(cal: Any, args: List[Any], frame: Frame) -> Any:
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

            if len(args) != len(params):
                raise ShakarArityError(f"Decorator expects {len(params)} args; got {len(args)}")
            return DecoratorConfigured(decorator=cal, args=list(args))
        case ShkFn():
            return call_shkfn(cal, args, subject=None, caller_frame=frame)
        case _:
            raise ShakarTypeError(f"Cannot call value of type {type(cal).__name__}")

# ---------------- Objects ----------------

def _eval_object(n: Tree, frame: Frame) -> ShkObject:
    """Build an object literal, installing descriptors/getters as needed."""
    slots: dict[str, Any] = {}

    def _install_descriptor(name: str, getter: ShkFn|None=None, setter: ShkFn|None=None) -> None:
        existing = slots.get(name)

        if isinstance(existing, Descriptor):
            if getter is not None:
                existing.getter = getter
            if setter is not None:
                existing.setter = setter
            slots[name] = existing
        else:
            slots[name] = Descriptor(getter=getter, setter=setter)

    def _extract_params(params_node: Tree|None) -> List[str]:
        if params_node is None:
            return []

        names: List[str] = []
        queue = list(tree_children(params_node))
        while queue:
            node = queue.pop(0)
            ident = _unwrap_ident(node)

            if ident is not None:
                names.append(ident)
                continue

            if is_tree_node(node):
                queue.extend(tree_children(node))
        return names

    def _unwrap_ident(node: Any) -> str | None:
        cur = node
        seen = set()

        while is_tree_node(cur) and cur.children and id(cur) not in seen:
            seen.add(id(cur))

            if len(cur.children) != 1:
                break
            cur = cur.children[0]
        return _ident_token_value(cur)

    def _maybe_method_signature(key_node: Any) -> tuple[str, List[str]] | None:
        if tree_label(key_node) != 'key_expr':
            return None
        target = key_node.children[0] if key_node.children else None
        chain = None

        if is_tree_node(target):
            if tree_label(target) == 'explicit_chain':
                chain = target
            else:
                chain = next(
                    (ch for ch in tree_children(target) if tree_label(ch) == 'explicit_chain'),
                    None,
                )

        if chain is None or len(chain.children) != 2:
            return None

        head, call_node = chain.children
        name = _expect_ident_token(head, "Object method key")

        if tree_label(call_node) != 'call':
            return None

        args_node = call_node.children[0] if call_node.children else None
        params: List[str] = []

        if is_tree_node(args_node):
            queue = list(args_node.children)

            while queue:
                raw = queue.pop(0)

                raw_label = tree_label(raw)
                if raw_label in {'namedarg', 'kwarg'}:
                    return None

                raw_children = tree_children(raw) if is_tree_node(raw) else None
                if raw_children and raw_label not in {'args', 'arglist', 'arglistnamedmixed', 'argitem', 'arg'}:
                    # allow deeper structures by re-queueing children
                    queue.extend(raw_children)
                    continue

                ident = _unwrap_ident(raw)
                if ident is None:
                    if is_tree_node(raw):
                        queue.extend(tree_children(raw))
                    else:
                        return None
                else:
                    params.append(ident)
        elif args_node is not None:
            ident = _unwrap_ident(args_node)

            if ident is None:
                return None
            params.append(ident)
        return (name, params)

    def handle_item(item: Tree) -> None:
        match item.data:
            case 'obj_field':
                key_node, val_node = item.children
                method_sig = _maybe_method_signature(key_node)

                if method_sig:
                    name, params = method_sig
                    method_fn = ShkFn(params=params, body=val_node, frame=Frame(parent=frame))
                    slots[name] = method_fn
                    return
                key = _eval_key(key_node, frame)
                val = eval_node(val_node, frame)
                slots[str(key)] = val
            case 'obj_get':
                name_tok, body = item.children

                if name_tok is None:
                    raise ShakarRuntimeError("Getter missing name")

                key = name_tok.value
                getter_fn = ShkFn(params=None, body=body, frame=Frame(parent=frame))
                _install_descriptor(key, getter=getter_fn)
            case 'obj_set':
                name_tok, param_tok, body = item.children

                if name_tok is None or param_tok is None:
                    raise ShakarRuntimeError("Setter missing name or parameter")

                key = name_tok.value
                setter_fn = ShkFn(params=[param_tok.value], body=body, frame=Frame(parent=frame))
                _install_descriptor(key, setter=setter_fn)
            case 'obj_method':
                name_tok, params_node, body = item.children

                if name_tok is None:
                    raise ShakarRuntimeError("Method missing name")
                param_names = _extract_params(params_node)
                method_fn = ShkFn(params=param_names, body=body, frame=Frame(parent=frame))
                slots[name_tok.value] = method_fn
            case _:
                raise ShakarRuntimeError(f"Unknown object item {item.data}")

    for child in tree_children(n):
        if is_token_node(child):
            continue

        child_label = tree_label(child)
        if child_label == 'object_items':
            for item in tree_children(child):
                if not is_tree_node(item) or tree_label(item) == 'obj_sep':
                    continue
                handle_item(item)
        else:
            if child_label == 'obj_sep':
                continue

            if is_tree_node(child):
                handle_item(child)
    return ShkObject(slots)

def _eval_key(k: Any, frame: Frame) -> Any:
    label = tree_label(k)

    if label is not None:
        match label:
            case 'key_ident':
                t = k.children[0]; return t.value
            case 'key_string':
                t = k.children[0]; s = t.value

                if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
                    s = s[1:-1]
                return s
            case 'key_expr':
                v = eval_node(k.children[0], frame)
                return v.value if isinstance(v, ShkString) else v

    if is_token_node(k) and _token_kind(k) in ('IDENT','STRING'):
        return k.value.strip('"').strip("'")

    return eval_node(k, frame)

def _eval_amp_lambda(n: Tree, frame: Frame) -> ShkFn:
    if len(n.children) == 1:
        return ShkFn(params=None, body=n.children[0], frame=Frame(parent=frame, dot=None), kind="amp")

    if len(n.children) == 2:
        params_node, body = n.children
        params = _extract_param_names(params_node, context="amp_lambda")
        return ShkFn(params=params, body=body, frame=Frame(parent=frame, dot=None), kind="amp")

    raise ShakarRuntimeError("amp_lambda malformed")

_NODE_DISPATCH: dict[str, Callable[[Tree, Frame], Any]] = {
    'listcomp': lambda n, frame: eval_listcomp(n, frame, eval_node),
    'setcomp': lambda n, frame: eval_setcomp(n, frame, eval_node),
    'setliteral': lambda n, frame: eval_setliteral(n, frame, eval_node),
    'dictcomp': lambda n, frame: eval_dictcomp(n, frame, eval_node),
    'selectorliteral': lambda n, frame: eval_selectorliteral(n, frame, eval_node),
    'string_interp': _eval_string_interp,
    'group': _eval_group,
    'no_anchor': _eval_group,
    'ternary': _eval_ternary,
    'rebind_primary': _eval_rebind_primary,
    'rebind_primary_grouped': _eval_rebind_primary,
    'amp_lambda': _eval_amp_lambda,
    'anonfn': lambda n, frame: _eval_anonymous_fn(n.children, frame),
    'await_value': lambda n, frame: eval_await_value(n, frame, eval_node),
    'compare': lambda n, frame: _eval_compare(n.children, frame),
    'compare_nc': lambda n, frame: _eval_compare(n.children, frame),
    'nullish': lambda n, frame: _eval_nullish(n.children, frame),
    'nullsafe': lambda n, frame: _eval_nullsafe(n.children, frame),
    'breakstmt': lambda _, frame: _eval_break_stmt(frame),
    'continuestmt': lambda _, frame: _eval_continue_stmt(frame),
    'awaitstmt': lambda n, frame: eval_await_stmt(n, frame, eval_node),
    'awaitanycall': lambda n, frame: eval_await_any_call(n, frame, eval_node),
    'awaitallcall': lambda n, frame: eval_await_all_call(n, frame, eval_node),
    'ifstmt': lambda n, frame: eval_if_stmt(n, frame, eval_node),
    'catchexpr': lambda n, frame: _eval_catch_expr(n.children, frame),
    'catchstmt': lambda n, frame: _eval_catch_stmt(n.children, frame),
    'forin': lambda n, frame: eval_for_in(n, frame, eval_node),
    'forsubject': lambda n, frame: eval_for_subject(n, frame, eval_node),
    'forindexed': lambda n, frame: eval_for_indexed(n, frame, eval_node),
    'formap1': lambda n, frame: eval_for_indexed(n.children[0] if n.children else None, frame, eval_node),
    'formap2': lambda n, frame: eval_for_map2(n, frame, eval_node),
    'inlinebody': lambda n, frame: eval_inline_body(n, frame, eval_node),
    'indentblock': lambda n, frame: eval_indent_block(n, frame, eval_node),
    'onelineguard': lambda n, frame: eval_oneline_guard(n.children, frame, eval_node),
}

_TOKEN_DISPATCH: dict[str, Callable[[Token, Frame], Any]] = {
    'NUMBER': _token_number,
    'STRING': _token_string,
    'TRUE': lambda _, __: ShkBool(True),
    'FALSE': lambda _, __: ShkBool(False),
    'NIL': lambda _, __: ShkNull(),
}
