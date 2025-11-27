from __future__ import annotations

from typing import Callable, List, Optional
from lark import Token

from .runtime import (
    Frame,
    ShkArray,
    ShkBool,
    ShkNull,
    ShkValue,
    ShakarBreakSignal,
    ShakarContinueSignal,
    ShakarRuntimeError,
    ShakarTypeError,
    init_stdlib,
)

from .tree import Node, Tree, child_by_label, is_token, is_tree, tree_children, tree_label, node_meta

from .eval.selector import eval_selectorliteral
from .eval.mutation import set_field_value, set_index_value
from .eval.control import (
    coerce_throw_value,
    eval_assert,
    eval_return_if,
    eval_return_stmt,
    eval_throw_stmt,
    eval_catch_expr,
    eval_catch_stmt,
)
from .eval.helpers import (
    is_truthy,
)

from .eval.blocks import (
    eval_program,
    eval_inline_body,
    eval_indent_block,
    eval_guard,
    eval_defer_stmt,
    get_subject,
)

from .eval.postfix import define_new_ident, eval_postfix_if as _postfix_eval_if, eval_postfix_unless as _postfix_eval_unless

from .eval.loops import (
    eval_if_stmt,
    eval_while_stmt,
    eval_for_in,
    eval_for_subject,
    eval_for_indexed,
    eval_for_map2,
    eval_listcomp,
    eval_setcomp,
    eval_setliteral,
    eval_dictcomp,
)
from .eval.fanout import eval_fanout_block
from .eval.destructure import eval_destructure

from .eval._await import eval_await_value, eval_await_stmt, eval_await_any_call, eval_await_all_call
from .eval.chains import apply_op, evaluate_index_operand, eval_args_node, call_value
from .eval.valuefan import eval_valuefan
from .eval.expr import (
    eval_unary,
    eval_infix,
    eval_compare,
    eval_logical,
    eval_nullish,
    eval_nullsafe,
    eval_ternary,
)
from .eval.literals import eval_keyword_literal, eval_shell_string, eval_string_interp
from .eval.objects import eval_object, eval_key
from .eval.fn import eval_fn_def, eval_decorator_def, eval_anonymous_fn, eval_amp_lambda, evaluate_decorator_list
from .eval.using import eval_using_stmt

EvalFunc = Callable[[Node, Frame], ShkValue]

from .eval.common import is_literal_node, token_number, token_string, node_source_span
from types import SimpleNamespace

from .eval.bind import (
    FanContext,
    RebindContext,
    assign_ident,
    eval_walrus,
    eval_assign_stmt,
    eval_compound_assign,
    eval_apply_assign,
    eval_rebind_primary,
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


def _maybe_attach_location(exc: ShakarRuntimeError, node: Node, frame: Frame) -> None:
    if getattr(exc, "_augmented", False):
        return

    meta = node_meta(node)

    if meta is not None and getattr(meta, "line", None) is not None:
        exc.shk_meta = meta
        exc._augmented = True  # type: ignore[attr-defined]
        return

    start, _ = node_source_span(node)
    if start is None:
        return

    source = getattr(frame, "source", None)
    if source is None:
        return

    line = source.count("\n", 0, start) + 1
    last_nl = source.rfind("\n", 0, start)
    col = start + 1 if last_nl == -1 else start - last_nl
    exc.shk_meta = SimpleNamespace(line=line, column=col)
    exc._augmented = True  # type: ignore[attr-defined]

# ---------------- Public API ----------------

def eval_expr(ast: Node, frame: Optional[Frame]=None, source: Optional[str]=None) -> ShkValue:
    init_stdlib()

    if frame is None:
        frame = Frame(source=source)
    else:
        if source is not None:
            frame.source = source
        elif not hasattr(frame, 'source'):
            frame.source = None

    try:
        return eval_node(ast, frame)
    except ShakarRuntimeError as e:
        _maybe_attach_location(e, ast, frame)
        raise

# ---------------- Core evaluator ----------------

def eval_node(n: Node, frame: Frame) -> ShkValue:
    try:
        return _eval_node_inner(n, frame)
    except ShakarRuntimeError as e:
        _maybe_attach_location(e, n, frame)
        raise


def _eval_node_inner(n: Node, frame: Frame) -> ShkValue:
    if is_literal_node(n):
        return n

    if is_token(n):
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
                return eval_keyword_literal(n)
            raise ShakarRuntimeError(f"Unsupported wrapper shape {d} with {len(n.children)} children")
        case 'array':
            return ShkArray([eval_node(c, frame) for c in n.children])
        case 'object':
            return eval_object(n, frame, eval_node)
        case 'unary' | 'unary_nc':
            op, rhs_node = n.children
            return eval_unary(op, rhs_node, frame, eval_node)
        case 'pow' | 'pow_nc':
            return eval_infix(n.children, frame, eval_node, right_assoc_ops={'**', 'POW'})
        case 'mul' | 'mul_nc' | 'add' | 'add_nc':
            return eval_infix(n.children, frame, eval_node)
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
                    apply_op=apply_op,
                    evaluate_index_operand=evaluate_index_operand,
                )

                delta = 1 if tree_label(tail) == 'incr' else -1

                if isinstance(context, FanContext):
                    raise ShakarRuntimeError("++/-- not supported on field fan assignments")
                old_val, _ = apply_numeric_delta(context, delta)
                return old_val

            val = eval_node(head, frame)
            head_label = tree_label(head) if is_tree(head) else None
            head_is_rebind = head_label in {'rebind_primary', 'rebind_primary_grouped'}
            head_is_grouped_rebind = head_label == 'rebind_primary_grouped'
            tail_has_effect = False

            for op in ops:
                label = tree_label(op)
                if label not in {'field', 'fieldsel', 'index'}:
                    tail_has_effect = True

                val = apply_op(val, op, frame, eval_node)

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
            args = eval_args_node(args_node, frame, eval_node)
            cal = frame.get('')  # unreachable in practice
            return call_value(cal, args, frame, eval_node)
        case 'and' | 'or' | 'and_nc' | 'or_nc':
            return eval_logical(d, n.children, frame, eval_node)
        case 'walrus' | 'walrus_nc':
            return eval_walrus(n.children, frame, eval_node)
        case 'returnstmt':
            return eval_return_stmt(n.children, frame, eval_func=eval_node)
        case 'returnif':
            return eval_return_if(n.children, frame, eval_func=eval_node)
        case 'throwstmt':
            return eval_throw_stmt(n.children, frame, eval_func=eval_node)
        case 'breakstmt':
            return _eval_break_stmt(frame)
        case 'continuestmt':
            return _eval_continue_stmt(frame)
        case 'assignstmt':
            return eval_assign_stmt(n.children, frame, eval_node, apply_op, evaluate_index_operand)
        case 'postfixif':
            return _postfix_eval_if(n.children, frame, eval_func=eval_node, truthy_fn=is_truthy)
        case 'postfixunless':
            return _postfix_eval_unless(n.children, frame, eval_func=eval_node, truthy_fn=is_truthy)
        case 'compound_assign':
            return eval_compound_assign(n.children, frame, eval_node, apply_op, evaluate_index_operand)
        case 'fndef':
            return eval_fn_def(n.children, frame, eval_node)
        case 'decorator_def':
            return eval_decorator_def(n.children, frame)
        case 'deferstmt':
            return eval_defer_stmt(n.children, frame, eval_node)
        case 'assert':
            return eval_assert(n.children, frame, eval_func=eval_node)
        case 'bind' | 'bind_nc':
            return eval_apply_assign(n.children, frame, eval_node, apply_op, evaluate_index_operand)
        case 'subject':
            return get_subject(frame)
        case 'keyexpr' | 'keyexpr_nc':
            return eval_node(n.children[0], frame) if n.children else ShkNull()
        case 'destructure':
            # plain destructure must not create new names; walrus form does.
            return eval_destructure(n, frame, eval_node, create=False, allow_broadcast=False)
        case 'destructure_walrus':
            return eval_destructure(n, frame, eval_node, create=True, allow_broadcast=True)
        case _:
            raise ShakarRuntimeError(f"Unknown node: {d}")

# ---------------- Tokens ----------------

def _eval_token(t: Token, frame: Frame) -> ShkValue:
    handler = _TOKEN_DISPATCH.get(t.type)
    if handler:
        return handler(t, frame)

    if t.type in ('IDENT', 'ANY', 'ALL', 'OVER'):
        return frame.get(t.value)

    raise ShakarRuntimeError(f"Unhandled token {t.type}:{t.value}")

def _eval_implicit_chain(ops: List[Tree], frame: Frame) -> ShkValue:
    """Evaluate `.foo().bar` style chains using the current subject anchor."""
    val = get_subject(frame)

    for op in ops:
        val = apply_op(val, op, frame, eval_node)

    return val

def _eval_break_stmt(frame: Frame) -> ShkValue:
    raise ShakarBreakSignal()

def _eval_continue_stmt(frame: Frame) -> ShkValue:
    raise ShakarContinueSignal()

def _eval_formap1(n: Tree, frame: Frame) -> ShkValue:
    child = n.children[0] if n.children else None

    if not is_tree(child):
        raise ShakarRuntimeError("Malformed indexed loop")

    return eval_for_indexed(child, frame, eval_node)

# ---------------- Grouping / dispatch ----------------

def _eval_group(n: Tree, frame: Frame) -> ShkValue:
    child = n.children[0] if n.children else None
    if child is None:
        return ShkNull()

    saved = frame.dot

    try:
        return eval_node(child, frame)
    finally:
        frame.dot = saved

_NODE_DISPATCH: dict[str, Callable[[Tree, Frame], ShkValue]] = {
    'listcomp': lambda n, frame: eval_listcomp(n, frame, eval_node),
    'setcomp': lambda n, frame: eval_setcomp(n, frame, eval_node),
    'setliteral': lambda n, frame: eval_setliteral(n, frame, eval_node),
    'dictcomp': lambda n, frame: eval_dictcomp(n, frame, eval_node),
    'selectorliteral': lambda n, frame: eval_selectorliteral(n, frame, eval_node),
    'string_interp': lambda n, frame: eval_string_interp(n, frame, eval_node),
    'shell_string': lambda n, frame: eval_shell_string(n, frame, eval_node),
    'group': _eval_group,
    'no_anchor': _eval_group,
    'ternary': lambda n, frame: eval_ternary(n, frame, eval_node),
    'rebind_primary': lambda n, frame: eval_rebind_primary(n, frame, eval_node, apply_op, evaluate_index_operand),
    'rebind_primary_grouped': lambda n, frame: eval_rebind_primary(n, frame, eval_node, apply_op, evaluate_index_operand),
    'amp_lambda': eval_amp_lambda,
    'anonfn': lambda n, frame: eval_anonymous_fn(n.children, frame),
    'await_value': lambda n, frame: eval_await_value(n, frame, eval_node),
    'compare': lambda n, frame: eval_compare(n.children, frame, eval_node),
    'compare_nc': lambda n, frame: eval_compare(n.children, frame, eval_node),
    'nullish': lambda n, frame: eval_nullish(n.children, frame, eval_node),
    'nullsafe': lambda n, frame: eval_nullsafe(n, frame, eval_node),
    'breakstmt': lambda _, frame: _eval_break_stmt(frame),
    'continuestmt': lambda _, frame: _eval_continue_stmt(frame),
    'awaitstmt': lambda n, frame: eval_await_stmt(n, frame, eval_node),
    'awaitanycall': lambda n, frame: eval_await_any_call(n, frame, eval_node),
    'awaitallcall': lambda n, frame: eval_await_all_call(n, frame, eval_node),
    'usingstmt': lambda n, frame: eval_using_stmt(n, frame, eval_node),
    'ifstmt': lambda n, frame: eval_if_stmt(n, frame, eval_node),
    'whilestmt': lambda n, frame: eval_while_stmt(n, frame, eval_node),
    'fanoutblock': lambda n, frame: eval_fanout_block(n, frame, eval_node, apply_op, evaluate_index_operand),
    'valuefan': lambda n, frame: eval_valuefan(eval_node(n.children[0], frame), n, frame, eval_node, apply_op),
    'catchexpr': lambda n, frame: eval_catch_expr(n.children, frame, eval_node),
    'catchstmt': lambda n, frame: eval_catch_stmt(n.children, frame, eval_node),
    'forin': lambda n, frame: eval_for_in(n, frame, eval_node),
    'forsubject': lambda n, frame: eval_for_subject(n, frame, eval_node),
    'forindexed': lambda n, frame: eval_for_indexed(n, frame, eval_node),
    'formap1': _eval_formap1,
    'formap2': lambda n, frame: eval_for_map2(n, frame, eval_node),
    'inlinebody': lambda n, frame: eval_inline_body(n, frame, eval_node),
    'indentblock': lambda n, frame: eval_indent_block(n, frame, eval_node),
    'onelineguard': lambda n, frame: eval_guard(n.children, frame, eval_node),
    'pack': lambda n, frame: ShkArray([eval_node(ch, frame) for ch in n.children]),
}

_TOKEN_DISPATCH: dict[str, Callable[[Token, Frame], ShkValue]] = {
    'NUMBER': token_number,
    'STRING': token_string,
    'RAW_STRING': token_string,
    'RAW_HASH_STRING': token_string,
    'TRUE': lambda _, __: ShkBool(True),
    'FALSE': lambda _, __: ShkBool(False),
    'NIL': lambda _, __: ShkNull(),
}
