from __future__ import annotations

from typing import Callable, List, Optional
from types import SimpleNamespace

from .token_types import TT
from .tree import Node, Tree, Tok, is_token, is_tree, node_meta, tree_label
from .runtime import (
    Frame,
    ShkArray,
    ShkBool,
    ShkNull,
    ShkValue,
    ShakarBreakSignal,
    ShakarContinueSignal,
    ShakarRuntimeError,
    init_stdlib,
)

from .eval.common import (
    is_literal_node,
    token_number,
    token_path,
    token_regex,
    token_string,
    node_source_span,
)
from .eval.selector import eval_selectorliteral
from .eval.control import (
    eval_assert,
    eval_if_stmt,
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
    eval_call_stmt,
    eval_emit_expr,
    get_subject,
)

from .eval.postfix import (
    eval_postfix_if as _postfix_eval_if,
    eval_postfix_unless as _postfix_eval_unless,
)

from .eval.loops import (
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

from .eval._await import (
    eval_await_value,
    eval_await_stmt,
    eval_await_any_call,
    eval_await_all_call,
)
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
    eval_explicit_chain,
)
from .eval.literals import (
    eval_array_literal,
    eval_keyword_literal,
    eval_path_interp,
    eval_shell_string,
    eval_string_interp,
)
from .eval.objects import eval_object
from .eval.fn import eval_fn_def, eval_decorator_def, eval_anonymous_fn, eval_amp_lambda
from .eval.using import eval_using_stmt

from .eval.bind import (
    eval_walrus,
    eval_assign_stmt,
    eval_compound_assign,
    eval_apply_assign,
    eval_rebind_primary,
)

EvalFunc = Callable[[Node, Frame], ShkValue]


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


def eval_expr(
    ast: Node, frame: Optional[Frame] = None, source: Optional[str] = None
) -> ShkValue:
    init_stdlib()

    if frame is None:
        frame = Frame(source=source)
    else:
        if source is not None:
            frame.source = source
        elif not hasattr(frame, "source"):
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
        case "start_noindent" | "start_indented" | "stmtlist":
            return eval_program(n.children, frame, eval_node)
        case "stmt":
            # stmt is a wrapper for a single child - unwrap it to preserve loop control
            if len(n.children) == 1:
                return eval_node(n.children[0], frame)
            return eval_program(n.children, frame, eval_node)
        case "literal" | "primary" | "expr":
            if len(n.children) == 1:
                return eval_node(n.children[0], frame)

            if len(n.children) == 0 and d == "literal":
                return eval_keyword_literal(n)
            raise ShakarRuntimeError(
                f"Unsupported wrapper shape {d} with {len(n.children)} children"
            )
        case "array":
            return eval_array_literal(n, frame, eval_node)
        case "object":
            return eval_object(n, frame, eval_node)
        case "unary":
            op, rhs_node = n.children
            return eval_unary(op, rhs_node, frame, eval_node)
        case "pow":
            return eval_infix(
                n.children, frame, eval_node, right_assoc_ops={"**", "POW"}
            )
        case "mul" | "add":
            return eval_infix(n.children, frame, eval_node)
        case "explicit_chain":
            return eval_explicit_chain(n, frame, eval_node)
        case "implicit_chain":
            return _eval_implicit_chain(n.children, frame)
        case "spread":
            raise ShakarRuntimeError(
                "Spread operator is only valid in array/object literals and call arguments"
            )
        case "call":
            args_node = n.children[0] if n.children else None
            args = eval_args_node(args_node, frame, eval_node)
            cal = frame.get("")  # unreachable in practice
            return call_value(cal, args, frame, eval_node)
        case "and" | "or":
            return eval_logical(d, n.children, frame, eval_node)
        case "walrus":
            return eval_walrus(n.children, frame, eval_node)
        case "returnstmt":
            return eval_return_stmt(n.children, frame, eval_func=eval_node)
        case "returnif":
            return eval_return_if(n.children, frame, eval_func=eval_node)
        case "throwstmt":
            return eval_throw_stmt(n.children, frame, eval_func=eval_node)
        case "breakstmt":
            return _eval_break_stmt(frame)
        case "continuestmt":
            return _eval_continue_stmt(frame)
        case "assignstmt":
            return eval_assign_stmt(
                n.children, frame, eval_node, apply_op, evaluate_index_operand
            )
        case "postfixif":
            return _postfix_eval_if(
                n.children, frame, eval_func=eval_node, truthy_fn=is_truthy
            )
        case "postfixunless":
            return _postfix_eval_unless(
                n.children, frame, eval_func=eval_node, truthy_fn=is_truthy
            )
        case "compound_assign":
            return eval_compound_assign(
                n.children, frame, eval_node, apply_op, evaluate_index_operand
            )
        case "fndef":
            return eval_fn_def(n.children, frame, eval_node)
        case "decorator_def":
            return eval_decorator_def(n.children, frame)
        case "deferstmt":
            return eval_defer_stmt(n.children, frame, eval_node)
        case "assert":
            return eval_assert(n.children, frame, eval_func=eval_node)
        case "bind":
            return eval_apply_assign(
                n.children, frame, eval_node, apply_op, evaluate_index_operand
            )
        case "subject":
            return get_subject(frame)
        case "keyexpr":
            return eval_node(n.children[0], frame) if n.children else ShkNull()
        case "destructure":
            # plain destructure must not create new names; walrus form does.
            return eval_destructure(
                n, frame, eval_node, create=False, allow_broadcast=False
            )
        case "destructure_walrus":
            return eval_destructure(
                n, frame, eval_node, create=True, allow_broadcast=True
            )
        case _:
            raise ShakarRuntimeError(f"Unknown node: {d}")


# ---------------- Toks ----------------


def _eval_token(t: Tok, frame: Frame) -> ShkValue:
    handler = _TOKEN_DISPATCH.get(t.type)
    if handler:
        return handler(t, frame)

    if t.type in (TT.IDENT, TT.ANY, TT.ALL, TT.OVER):
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
    "listcomp": lambda n, frame: eval_listcomp(n, frame, eval_node),
    "setcomp": lambda n, frame: eval_setcomp(n, frame, eval_node),
    "setliteral": lambda n, frame: eval_setliteral(n, frame, eval_node),
    "dictcomp": lambda n, frame: eval_dictcomp(n, frame, eval_node),
    "selectorliteral": lambda n, frame: eval_selectorliteral(n, frame, eval_node),
    "string_interp": lambda n, frame: eval_string_interp(n, frame, eval_node),
    "shell_string": lambda n, frame: eval_shell_string(n, frame, eval_node),
    "path_interp": lambda n, frame: eval_path_interp(n, frame, eval_node),
    "group": _eval_group,
    "no_anchor": _eval_group,
    "ternary": lambda n, frame: eval_ternary(n, frame, eval_node),
    "rebind_primary": lambda n, frame: eval_rebind_primary(
        n, frame, eval_node, apply_op, evaluate_index_operand
    ),
    "rebind_primary_grouped": lambda n, frame: eval_rebind_primary(
        n, frame, eval_node, apply_op, evaluate_index_operand
    ),
    "amp_lambda": eval_amp_lambda,
    "anonfn": lambda n, frame: eval_anonymous_fn(n.children, frame),
    "await_value": lambda n, frame: eval_await_value(n, frame, eval_node),
    "compare": lambda n, frame: eval_compare(n.children, frame, eval_node),
    "nullish": lambda n, frame: eval_nullish(n.children, frame, eval_node),
    "nullsafe": lambda n, frame: eval_nullsafe(n, frame, eval_node),
    "breakstmt": lambda _, frame: _eval_break_stmt(frame),
    "continuestmt": lambda _, frame: _eval_continue_stmt(frame),
    "awaitstmt": lambda n, frame: eval_await_stmt(n, frame, eval_node),
    "awaitanycall": lambda n, frame: eval_await_any_call(n, frame, eval_node),
    "awaitallcall": lambda n, frame: eval_await_all_call(n, frame, eval_node),
    "usingstmt": lambda n, frame: eval_using_stmt(n, frame, eval_node),
    "callstmt": lambda n, frame: eval_call_stmt(n, frame, eval_node),
    "ifstmt": lambda n, frame: eval_if_stmt(n, frame, eval_node),
    "whilestmt": lambda n, frame: eval_while_stmt(n, frame, eval_node),
    "fanoutblock": lambda n, frame: eval_fanout_block(
        n, frame, eval_node, apply_op, evaluate_index_operand
    ),
    "valuefan": lambda n, frame: eval_valuefan(
        eval_node(n.children[0], frame), n, frame, eval_node, apply_op
    ),
    "catchexpr": lambda n, frame: eval_catch_expr(n.children, frame, eval_node),
    "catchstmt": lambda n, frame: eval_catch_stmt(n.children, frame, eval_node),
    "slicearm_expr": lambda n, frame: (
        eval_node(n.children[0], frame) if n.children else ShkNull()
    ),
    "slicearm_empty": lambda _n, _frame: ShkNull(),
    "forin": lambda n, frame: eval_for_in(n, frame, eval_node),
    "forsubject": lambda n, frame: eval_for_subject(n, frame, eval_node),
    "forindexed": lambda n, frame: eval_for_indexed(n, frame, eval_node),
    "formap1": _eval_formap1,
    "formap2": lambda n, frame: eval_for_map2(n, frame, eval_node),
    "inlinebody": lambda n, frame: eval_inline_body(n, frame, eval_node),
    "indentblock": lambda n, frame: eval_indent_block(n, frame, eval_node),
    "onelineguard": lambda n, frame: eval_guard(n.children, frame, eval_node),
    "emitexpr": lambda n, frame: eval_emit_expr(n, frame, eval_node),
    "pack": lambda n, frame: ShkArray([eval_node(ch, frame) for ch in n.children]),
}

_TOKEN_DISPATCH: dict[TT, Callable[[Tok, Frame], ShkValue]] = {
    TT.NUMBER: token_number,
    TT.STRING: token_string,
    TT.RAW_STRING: token_string,
    TT.RAW_HASH_STRING: token_string,
    TT.PATH_STRING: token_path,
    TT.REGEX: token_regex,
    TT.TRUE: lambda _, __: ShkBool(True),
    TT.FALSE: lambda _, __: ShkBool(False),
    TT.NIL: lambda _, __: ShkNull(),
}
