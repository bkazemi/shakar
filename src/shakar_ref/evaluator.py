from __future__ import annotations

from typing import Callable, List, Optional
import pathlib
from types import SimpleNamespace

from .token_types import TT
from .lexer_rd import Lexer
from .stdlib import std_mixin
from .tree import Node, Tree, Tok, is_token, is_tree, tree_label
from .runtime import (
    Frame,
    ShkArray,
    ShkFan,
    ShkBool,
    ShkNil,
    ShkString,
    ShkValue,
    ShakarBreakSignal,
    ShakarContinueSignal,
    ShakarImportError,
    ShakarRuntimeError,
    ShakarTypeError,
    import_module,
    init_stdlib,
)

from .eval.common import (
    is_literal_node,
    modifier_from_node,
    token_number,
    token_duration,
    token_size,
    token_path,
    token_regex,
    token_string,
    validate_modifier,
    _resolve_error_span,
)
from .eval.selector import eval_selectorliteral
from .eval.control import (
    eval_assert,
    eval_if_stmt,
    eval_match_expr,
    eval_return_if,
    eval_return_stmt,
    eval_throw_stmt,
    eval_catch_expr,
    eval_catch_stmt,
    eval_try_stmt,
)
from .eval.helpers import (
    is_truthy,
    name_in_current_frame,
    check_cancel,
)
from .utils import debug_py_trace_enabled

from .eval.blocks import (
    eval_program,
    eval_body_node,
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

from .eval.channels import (
    eval_recv_expr,
    eval_send_expr,
    eval_spawn_expr,
    eval_wait_any_block,
    eval_wait_all_block,
    eval_wait_group_block,
    eval_wait_all_call,
    eval_wait_group_call,
)
from .eval.chains import (
    apply_op,
    evaluate_index_operand,
    eval_args_node,
    eval_args_node_with_named,
    call_value,
)
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
    eval_fan_chain,
    maybe_valuefan_broadcast,
)
from .eval.literals import (
    eval_array_literal,
    eval_env_interp,
    eval_env_string,
    eval_keyword_literal,
    eval_path_interp,
    eval_shell_bang,
    eval_shell_string,
    eval_string_interp,
)
from .eval.objects import eval_object
from .eval.fn import (
    eval_fn_def,
    eval_decorator_def,
    eval_anonymous_fn,
    eval_amp_lambda,
    eval_hook_stmt,
)
from .eval.using import eval_using_stmt

from .eval.bind import (
    eval_walrus,
    eval_assign_stmt,
    eval_compound_assign,
    eval_apply_assign,
    eval_rebind_primary,
)
from .eval.let import eval_let_stmt

EvalFn = Callable[[Node, Frame], ShkValue]


def _maybe_attach_location(exc: ShakarRuntimeError, node: Node, frame: Frame) -> None:
    if getattr(exc, "_augmented", False):
        return

    line, col, end_line, end_col = _resolve_error_span(node, frame)

    if line is None:
        return

    meta = SimpleNamespace(line=line, column=col)
    if end_line is not None:
        meta.end_line = end_line
    if end_col is not None:
        meta.end_column = end_col
    exc.shk_meta = meta
    exc._augmented = True  # type: ignore[attr-defined]


def _maybe_attach_call_stack(exc: ShakarRuntimeError, frame: Frame) -> None:
    if getattr(exc, "shk_call_stack", None) is not None:
        return
    call_stack = getattr(frame, "call_stack", None)
    if not call_stack:
        return
    exc.shk_call_stack = list(call_stack)


def _maybe_attach_py_trace(exc: ShakarRuntimeError) -> None:
    if getattr(exc, "shk_py_trace", None) is not None:
        return
    if not debug_py_trace_enabled():
        return
    if exc.__traceback__ is None:
        return
    exc.shk_py_trace = exc.__traceback__


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
        _maybe_attach_call_stack(e, frame)
        _maybe_attach_py_trace(e)
        _maybe_attach_location(e, ast, frame)
        raise


# ---------------- Core evaluator ----------------


def eval_node(n: Node, frame: Frame) -> ShkValue:
    check_cancel(frame)
    try:
        return _eval_node_inner(n, frame)
    except ShakarRuntimeError as e:
        _maybe_attach_call_stack(e, frame)
        _maybe_attach_py_trace(e)
        _maybe_attach_location(e, n, frame)
        raise


def _eval_node_inner(n: Node, frame: Frame) -> ShkValue:
    if is_literal_node(n):
        return n

    if is_token(n):
        return _eval_token(n, frame)

    d = n.data
    handler = _NODE_DISPATCH.get(d)
    if handler:
        return handler(n, frame)

    raise ShakarRuntimeError(f"Unknown node: {d}")


# ---------------- Toks ----------------


def _eval_token(t: Tok, frame: Frame) -> ShkValue:
    handler = _TOKEN_DISPATCH.get(t.type)
    if handler:
        return handler(t, frame)

    if t.type in (TT.IDENT, TT.OVER):
        return frame.get(t.value)

    raise ShakarRuntimeError(f"Unhandled token {t.type}:{t.value}")


def _eval_implicit_chain(ops: List[Tree], frame: Frame) -> ShkValue:
    """Evaluate `.foo().bar` style chains using the current subject anchor."""
    val = get_subject(frame)

    if isinstance(val, ShkFan):
        return eval_fan_chain(val, ops, frame, eval_node)

    for i, op in enumerate(ops):
        val = apply_op(val, op, frame, eval_node)

        # Valuefan with trailing ops: switch to fan broadcasting
        fan_result = maybe_valuefan_broadcast(val, op, ops, i + 1, frame, eval_node)
        if fan_result:
            return fan_result

    return val


def _eval_formap1(n: Tree, frame: Frame) -> ShkValue:
    child = n.children[0] if n.children else None

    if not is_tree(child):
        raise ShakarRuntimeError("Malformed indexed loop")

    return eval_for_indexed(child, frame, eval_node)


# ---------------- Grouping / dispatch ----------------


def _eval_group(n: Tree, frame: Frame) -> ShkValue:
    child = n.children[0] if n.children else None
    if child is None:
        return ShkNil()

    saved = frame.dot

    try:
        return eval_node(child, frame)
    finally:
        frame.dot = saved


def _eval_wrapper_node(n: Tree, frame: Frame) -> ShkValue:
    if len(n.children) == 1:
        return eval_node(n.children[0], frame)

    if len(n.children) == 0 and n.data == "literal":
        return eval_keyword_literal(n)

    raise ShakarRuntimeError(
        f"Unsupported wrapper shape {n.data} with {len(n.children)} children"
    )


def _eval_stmt_node(n: Tree, frame: Frame) -> ShkValue:
    if len(n.children) == 1:
        return eval_node(n.children[0], frame)

    return eval_program(n.children, frame, eval_node)


def _eval_program_node(n: Tree, frame: Frame) -> ShkValue:
    return eval_program(n.children, frame, eval_node)


# ---- Expression handlers ----


def _eval_array(n: Tree, frame: Frame) -> ShkValue:
    return eval_array_literal(n, frame, eval_node)


def _eval_fan_literal(n: Tree, frame: Frame) -> ShkValue:
    modifiers = None
    items_node = None

    for child in n.children:
        if is_tree(child) and child.data == "fan_modifiers":
            modifiers = child
        elif is_tree(child) and child.data == "fan_items":
            items_node = child

    if modifiers:
        modifier_name, modifier_tok = modifier_from_node(modifiers)
        if modifier_name:
            validate_modifier("fan", modifier_name, modifier_tok)
        raise ShakarRuntimeError("Fan modifiers are not supported yet")

    if items_node is None:
        return ShkFan([])

    items = [eval_node(ch, frame) for ch in items_node.children]
    return ShkFan(items)


def _eval_object(n: Tree, frame: Frame) -> ShkValue:
    return eval_object(n, frame, eval_node)


def _eval_unary(n: Tree, frame: Frame) -> ShkValue:
    return eval_unary(n.children[0], n.children[1], frame, eval_node)


def _eval_pow(n: Tree, frame: Frame) -> ShkValue:
    return eval_infix(n.children, frame, eval_node, right_assoc_ops={"**", "POW"})


def _eval_infix(n: Tree, frame: Frame) -> ShkValue:
    return eval_infix(n.children, frame, eval_node)


def _eval_explicit_chain(n: Tree, frame: Frame) -> ShkValue:
    return eval_explicit_chain(n, frame, eval_node)


def _eval_implicit_chain_node(n: Tree, frame: Frame) -> ShkValue:
    return _eval_implicit_chain(n.children, frame)


def _eval_spread(_n: Tree, _frame: Frame) -> ShkValue:
    raise ShakarRuntimeError(
        "Spread operator is only valid in array/object literals and call arguments"
    )


def _eval_call(n: Tree, frame: Frame) -> ShkValue:
    args_node = n.children[0] if n.children else None
    positional, named, interleaved = eval_args_node_with_named(
        args_node, frame, eval_node
    )

    return call_value(
        frame.get(""),
        positional,
        frame,
        eval_node,
        named=named,
        interleaved=interleaved,
        call_node=n,
    )


def _eval_and(n: Tree, frame: Frame) -> ShkValue:
    return eval_logical("and", n.children, frame, eval_node)


def _eval_or(n: Tree, frame: Frame) -> ShkValue:
    return eval_logical("or", n.children, frame, eval_node)


def _eval_walrus(n: Tree, frame: Frame) -> ShkValue:
    return eval_walrus(n.children, frame, eval_node)


def _eval_ternary(n: Tree, frame: Frame) -> ShkValue:
    return eval_ternary(n, frame, eval_node)


def _eval_compare(n: Tree, frame: Frame) -> ShkValue:
    return eval_compare(n.children, frame, eval_node)


def _eval_nullish(n: Tree, frame: Frame) -> ShkValue:
    return eval_nullish(n.children, frame, eval_node)


def _eval_nullsafe(n: Tree, frame: Frame) -> ShkValue:
    return eval_nullsafe(n, frame, eval_node)


def _eval_recv(n: Tree, frame: Frame) -> ShkValue:
    return eval_recv_expr(n, frame, eval_node)


def _eval_send(n: Tree, frame: Frame) -> ShkValue:
    return eval_send_expr(n, frame, eval_node)


def _eval_spawn(n: Tree, frame: Frame) -> ShkValue:
    return eval_spawn_expr(n, frame, eval_node)


def _eval_waitanyblock(n: Tree, frame: Frame) -> ShkValue:
    return eval_wait_any_block(n, frame, eval_node)


def _eval_waitallblock(n: Tree, frame: Frame) -> ShkValue:
    return eval_wait_all_block(n, frame, eval_node)


def _eval_waitgroupblock(n: Tree, frame: Frame) -> ShkValue:
    return eval_wait_group_block(n, frame, eval_node)


def _eval_waitallcall(n: Tree, frame: Frame) -> ShkValue:
    return eval_wait_all_call(n, frame, eval_node)


def _eval_waitgroupcall(n: Tree, frame: Frame) -> ShkValue:
    return eval_wait_group_call(n, frame, eval_node)


def _eval_wait_modifier_stub(kind: str, n: Tree) -> ShkValue:
    """Shared handler for wait modifier block/call stubs."""
    modifier_name, modifier_tok = modifier_from_node(n)
    if not modifier_name:
        raise ShakarRuntimeError(f"Malformed wait modifier {kind}")
    validate_modifier("wait", modifier_name, modifier_tok)
    raise ShakarRuntimeError(
        f"Internal error: known wait modifier should use a dedicated wait {kind} node"
    )


def _eval_waitmodifierblock(n: Tree, _frame: Frame) -> ShkValue:
    return _eval_wait_modifier_stub("block", n)


def _eval_waitmodifiercall(n: Tree, _frame: Frame) -> ShkValue:
    return _eval_wait_modifier_stub("call", n)


def _eval_subject(_n: Tree, frame: Frame) -> ShkValue:
    return get_subject(frame)


def _eval_optional_child(n: Tree, frame: Frame) -> ShkValue:
    """Evaluate single optional child, returning nil if absent."""
    return eval_node(n.children[0], frame) if n.children else ShkNil()


def _eval_slicearm_empty(_n: Tree, _frame: Frame) -> ShkValue:
    return ShkNil()


def _eval_pack(n: Tree, frame: Frame) -> ShkValue:
    return ShkArray([eval_node(ch, frame) for ch in n.children])


# ---- Statement handlers ----


def _eval_returnstmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_return_stmt(n.children, frame, eval_fn=eval_node)


def _eval_returnif(n: Tree, frame: Frame) -> ShkValue:
    return eval_return_if(n.children, frame, eval_fn=eval_node)


def _eval_throwstmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_throw_stmt(n.children, frame, eval_fn=eval_node)


def _eval_assignstmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_assign_stmt(
        n.children, frame, eval_node, apply_op, evaluate_index_operand
    )


def _eval_let(n: Tree, frame: Frame) -> ShkValue:
    return eval_let_stmt(n, frame, eval_node, apply_op, evaluate_index_operand)


def _eval_postfixif(n: Tree, frame: Frame) -> ShkValue:
    return _postfix_eval_if(n.children, frame, eval_fn=eval_node, truthy_fn=is_truthy)


def _eval_postfixunless(n: Tree, frame: Frame) -> ShkValue:
    return _postfix_eval_unless(
        n.children, frame, eval_fn=eval_node, truthy_fn=is_truthy
    )


def _eval_compound_assign(n: Tree, frame: Frame) -> ShkValue:
    return eval_compound_assign(
        n.children, frame, eval_node, apply_op, evaluate_index_operand
    )


def _eval_fndef(n: Tree, frame: Frame) -> ShkValue:
    return eval_fn_def(n.children, frame, eval_node)


def _eval_decorator_def(n: Tree, frame: Frame) -> ShkValue:
    return eval_decorator_def(n.children, frame)


def _eval_hook(n: Tree, frame: Frame) -> ShkValue:
    return eval_hook_stmt(n, frame, eval_node)


def _eval_deferstmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_defer_stmt(n.children, frame, eval_node)


def _eval_assert(n: Tree, frame: Frame) -> ShkValue:
    return eval_assert(n.children, frame, eval_fn=eval_node)


def _eval_bind(n: Tree, frame: Frame) -> ShkValue:
    return eval_apply_assign(
        n.children, frame, eval_node, apply_op, evaluate_index_operand
    )


def _eval_destructure(n: Tree, frame: Frame) -> ShkValue:
    return eval_destructure(n, frame, eval_node, create=False, allow_broadcast=False)


def _eval_destructure_walrus(n: Tree, frame: Frame) -> ShkValue:
    return eval_destructure(n, frame, eval_node, create=True, allow_broadcast=True)


def _eval_usingstmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_using_stmt(n, frame, eval_node)


def _eval_callstmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_call_stmt(n, frame, eval_node)


def _eval_catchexpr(n: Tree, frame: Frame) -> ShkValue:
    return eval_catch_expr(n.children, frame, eval_node)


def _eval_catchstmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_catch_stmt(n.children, frame, eval_node)


def _eval_trystmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_try_stmt(n.children, frame, eval_node)


# ---- Control flow handlers ----


def _eval_ifstmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_if_stmt(n, frame, eval_node)


def _eval_matchexpr(n: Tree, frame: Frame) -> ShkValue:
    return eval_match_expr(n, frame, eval_node)


def _eval_whilestmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_while_stmt(n, frame, eval_node)


def _eval_breakstmt_node(_n: Tree, _frame: Frame) -> ShkValue:
    raise ShakarBreakSignal()


def _eval_continuestmt_node(_n: Tree, _frame: Frame) -> ShkValue:
    raise ShakarContinueSignal()


# ---- Loop handlers ----


def _eval_forin(n: Tree, frame: Frame) -> ShkValue:
    return eval_for_in(n, frame, eval_node)


def _eval_forsubject(n: Tree, frame: Frame) -> ShkValue:
    return eval_for_subject(n, frame, eval_node)


def _eval_forindexed(n: Tree, frame: Frame) -> ShkValue:
    return eval_for_indexed(n, frame, eval_node)


def _eval_formap2(n: Tree, frame: Frame) -> ShkValue:
    return eval_for_map2(n, frame, eval_node)


# ---- Comprehension/literal handlers ----


def _eval_listcomp(n: Tree, frame: Frame) -> ShkValue:
    return eval_listcomp(n, frame, eval_node)


def _eval_setcomp(n: Tree, frame: Frame) -> ShkValue:
    return eval_setcomp(n, frame, eval_node)


def _eval_setliteral(n: Tree, frame: Frame) -> ShkValue:
    return eval_setliteral(n, frame, eval_node)


def _eval_dictcomp(n: Tree, frame: Frame) -> ShkValue:
    return eval_dictcomp(n, frame, eval_node)


def _eval_selectorliteral(n: Tree, frame: Frame) -> ShkValue:
    return eval_selectorliteral(n, frame, eval_node)


def _eval_string_interp(n: Tree, frame: Frame) -> ShkValue:
    return eval_string_interp(n, frame, eval_node)


def _eval_shell_string(n: Tree, frame: Frame) -> ShkValue:
    return eval_shell_string(n, frame, eval_node)


def _eval_shell_bang(n: Tree, frame: Frame) -> ShkValue:
    return eval_shell_bang(n, frame, eval_node)


def _eval_path_interp(n: Tree, frame: Frame) -> ShkValue:
    return eval_path_interp(n, frame, eval_node)


def _eval_env_string(n: Tree, frame: Frame) -> ShkValue:
    return eval_env_string(n, frame, eval_node)


def _eval_env_interp(n: Tree, frame: Frame) -> ShkValue:
    return eval_env_interp(n, frame, eval_node)


# ---- Block/body handlers ----


def _eval_body(n: Tree, frame: Frame) -> ShkValue:
    return eval_body_node(n, frame, eval_node)


def _eval_onelineguard(n: Tree, frame: Frame) -> ShkValue:
    return eval_guard(n.children, frame, eval_node)


def _eval_emitexpr(n: Tree, frame: Frame) -> ShkValue:
    return eval_emit_expr(n, frame, eval_node)


# ---- Import handlers ----


def _is_ident(name: str) -> bool:
    if not name:
        return False

    first = name[0]
    if not (("A" <= first <= "Z") or ("a" <= first <= "z") or first == "_"):
        return False

    for ch in name[1:]:
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch == "_":
            continue
        return False

    return True


def _import_name_from_node(node: Node, frame: Frame) -> str:
    value = eval_node(node, frame)
    if not isinstance(value, ShkString):
        raise ShakarTypeError("Expected module name string after 'import'")
    return value.value


def _default_import_bind(name: str) -> str:
    path = pathlib.Path(name)
    base = path.stem if path.suffix == ".shk" else path.name
    if not base:
        raise ShakarImportError("Import name cannot be empty")
    if not _is_ident(base) or base in Lexer.KEYWORDS:
        raise ShakarImportError(
            f"Import name '{base}' is not a valid identifier; use 'bind' to set a name"
        )

    return base


def _eval_import_expr(n: Tree, frame: Frame) -> ShkValue:
    if not n.children:
        raise ShakarRuntimeError("Expected module name string after 'import'")
    module_name = _import_name_from_node(n.children[0], frame)
    return import_module(module_name, frame)


def _eval_import_stmt(n: Tree, frame: Frame) -> ShkValue:
    if not n.children:
        raise ShakarRuntimeError("Expected module name string after 'import'")

    module_name = _import_name_from_node(n.children[0], frame)
    module = import_module(module_name, frame)

    bind_name = _default_import_bind(module_name)
    if len(n.children) > 1:
        name_node = n.children[1]
        if is_token(name_node) and name_node.type == TT.IDENT:
            bind_name = str(name_node.value)
        else:
            raise ShakarRuntimeError("Expected identifier after 'bind'")

    if name_in_current_frame(frame, bind_name):
        raise ShakarImportError(
            f"Import name '{bind_name}' already defined in this scope"
        )

    frame.define(bind_name, module)
    return module


def _eval_import_destructure(n: Tree, frame: Frame) -> ShkValue:
    if len(n.children) != 2:
        raise ShakarRuntimeError(
            "Expected import names and module string after 'import[...]'"
        )

    names_node, module_node = n.children
    if not is_tree(names_node) or names_node.data != "import_names":
        raise ShakarRuntimeError("Expected import names in brackets")

    names: list[str] = []
    for child in names_node.children:
        if is_token(child) and child.type == TT.IDENT:
            names.append(str(child.value))
        else:
            raise ShakarRuntimeError("Expected identifiers in import list")

    if len(set(names)) != len(names):
        raise ShakarImportError("Import destructure names must be unique")

    module_name = _import_name_from_node(module_node, frame)
    module = import_module(module_name, frame)
    slots = getattr(module, "slots", None)
    if slots is None:
        raise ShakarImportError("Imported value is not a module")

    for name in names:
        if name not in slots:
            raise ShakarImportError(f"Module '{module_name}' has no export '{name}'")
        if name_in_current_frame(frame, name):
            raise ShakarImportError(
                f"Import name '{name}' already defined in this scope"
            )

    last_value: ShkValue = ShkNil()
    for name in names:
        value = slots[name]
        frame.define(name, value)
        last_value = value

    return last_value


def _eval_import_mixin(n: Tree, frame: Frame) -> ShkValue:
    if not n.children:
        raise ShakarRuntimeError("Expected module name string after 'import[*]'")

    module_name = _import_name_from_node(n.children[0], frame)
    module = import_module(module_name, frame)
    return std_mixin(frame, None, [module])


# ---- Fanout/rebind handlers ----


def _eval_fanoutblock(n: Tree, frame: Frame) -> ShkValue:
    return eval_fanout_block(n, frame, eval_node, apply_op, evaluate_index_operand)


def _eval_valuefan(n: Tree, frame: Frame) -> ShkValue:
    return eval_valuefan(eval_node(n.children[0], frame), n, frame, eval_node, apply_op)


def _eval_rebind_primary(n: Tree, frame: Frame) -> ShkValue:
    return eval_rebind_primary(n, frame, eval_node, apply_op, evaluate_index_operand)


def _eval_anonfn(n: Tree, frame: Frame) -> ShkValue:
    return eval_anonymous_fn(n.children, frame)


# ---- Dispatch table ----


_NODE_DISPATCH: dict[str, Callable[[Tree, Frame], ShkValue]] = {
    # Structural
    "start_noindent": _eval_program_node,
    "start_indented": _eval_program_node,
    "stmtlist": _eval_program_node,
    "stmt": _eval_stmt_node,
    "literal": _eval_wrapper_node,
    "primary": _eval_wrapper_node,
    "expr": _eval_wrapper_node,
    # Expressions
    "array": _eval_array,
    "fan_literal": _eval_fan_literal,
    "object": _eval_object,
    "unary": _eval_unary,
    "pow": _eval_pow,
    "mul": _eval_infix,
    "add": _eval_infix,
    "explicit_chain": _eval_explicit_chain,
    "implicit_chain": _eval_implicit_chain_node,
    "spread": _eval_spread,
    "call": _eval_call,
    "and": _eval_and,
    "or": _eval_or,
    "walrus": _eval_walrus,
    "ternary": _eval_ternary,
    "compare": _eval_compare,
    "matchexpr": _eval_matchexpr,
    "nullish": _eval_nullish,
    "nullsafe": _eval_nullsafe,
    "recv": _eval_recv,
    "send": _eval_send,
    "spawn": _eval_spawn,
    "subject": _eval_subject,
    "keyexpr": _eval_optional_child,
    "slicearm_expr": _eval_optional_child,
    "slicearm_empty": _eval_slicearm_empty,
    "pack": _eval_pack,
    # Statements
    "returnstmt": _eval_returnstmt,
    "returnif": _eval_returnif,
    "throwstmt": _eval_throwstmt,
    "assignstmt": _eval_assignstmt,
    "let": _eval_let,
    "postfixif": _eval_postfixif,
    "postfixunless": _eval_postfixunless,
    "compound_assign": _eval_compound_assign,
    "fndef": _eval_fndef,
    "decorator_def": _eval_decorator_def,
    "hook": _eval_hook,
    "deferstmt": _eval_deferstmt,
    "assert": _eval_assert,
    "bind": _eval_bind,
    "destructure": _eval_destructure,
    "destructure_walrus": _eval_destructure_walrus,
    "waitanyblock": _eval_waitanyblock,
    "waitallblock": _eval_waitallblock,
    "waitgroupblock": _eval_waitgroupblock,
    "waitmodifierblock": _eval_waitmodifierblock,
    "waitallcall": _eval_waitallcall,
    "waitgroupcall": _eval_waitgroupcall,
    "waitmodifiercall": _eval_waitmodifiercall,
    "usingstmt": _eval_usingstmt,
    "callstmt": _eval_callstmt,
    "catchexpr": _eval_catchexpr,
    "catchstmt": _eval_catchstmt,
    "trystmt": _eval_trystmt,
    # Control flow
    "ifstmt": _eval_ifstmt,
    "whilestmt": _eval_whilestmt,
    "breakstmt": _eval_breakstmt_node,
    "continuestmt": _eval_continuestmt_node,
    # Loops
    "forin": _eval_forin,
    "forsubject": _eval_forsubject,
    "forindexed": _eval_forindexed,
    "formap1": _eval_formap1,
    "formap2": _eval_formap2,
    # Comprehensions/literals
    "listcomp": _eval_listcomp,
    "setcomp": _eval_setcomp,
    "setliteral": _eval_setliteral,
    "dictcomp": _eval_dictcomp,
    "selectorliteral": _eval_selectorliteral,
    "string_interp": _eval_string_interp,
    "shell_string": _eval_shell_string,
    "shell_bang": _eval_shell_bang,
    "path_interp": _eval_path_interp,
    "env_string": _eval_env_string,
    "env_interp": _eval_env_interp,
    # Blocks/bodies
    "group": _eval_group,
    "noanchor_expr": _eval_group,
    "body": _eval_body,
    "onelineguard": _eval_onelineguard,
    "emitexpr": _eval_emitexpr,
    "import_expr": _eval_import_expr,
    # Fanout/rebind
    "fanoutblock": _eval_fanoutblock,
    "valuefan": _eval_valuefan,
    "rebind_primary": _eval_rebind_primary,
    # Functions
    "amp_lambda": eval_amp_lambda,
    "anonfn": _eval_anonfn,
    # Import statements
    "import_stmt": _eval_import_stmt,
    "import_destructure": _eval_import_destructure,
    "import_mixin": _eval_import_mixin,
}

_TOKEN_DISPATCH: dict[TT, Callable[[Tok, Frame], ShkValue]] = {
    TT.NUMBER: token_number,
    TT.DURATION: token_duration,
    TT.SIZE: token_size,
    TT.STRING: token_string,
    TT.RAW_STRING: token_string,
    TT.RAW_HASH_STRING: token_string,
    TT.PATH_STRING: token_path,
    TT.REGEX: token_regex,
    TT.TRUE: lambda _, __: ShkBool(True),
    TT.FALSE: lambda _, __: ShkBool(False),
    TT.NIL: lambda _, __: ShkNil(),
}
