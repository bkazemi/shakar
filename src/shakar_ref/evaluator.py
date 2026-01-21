from __future__ import annotations

from typing import Callable, List, Optional
import pathlib
from types import SimpleNamespace

from .token_types import TT
from .lexer_rd import Lexer
from .stdlib import std_mixin
from .tree import Node, Tree, Tok, is_token, is_tree, node_meta, tree_label
from .runtime import (
    Frame,
    ShkArray,
    ShkFan,
    ShkBool,
    ShkNull,
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
    token_number,
    token_duration,
    token_size,
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
    name_in_current_frame,
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
    eval_fan_chain,
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

    if isinstance(val, ShkFan):
        return eval_fan_chain(val, ops, frame, eval_node)

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

    if modifiers is not None:
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
    args = eval_args_node(args_node, frame, eval_node)

    return call_value(frame.get(""), args, frame, eval_node)


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


def _eval_await_value(n: Tree, frame: Frame) -> ShkValue:
    return eval_await_value(n, frame, eval_node)


def _eval_subject(_n: Tree, frame: Frame) -> ShkValue:
    return get_subject(frame)


def _eval_keyexpr(n: Tree, frame: Frame) -> ShkValue:
    return eval_node(n.children[0], frame) if n.children else ShkNull()


def _eval_slicearm_expr(n: Tree, frame: Frame) -> ShkValue:
    return eval_node(n.children[0], frame) if n.children else ShkNull()


def _eval_slicearm_empty(_n: Tree, _frame: Frame) -> ShkValue:
    return ShkNull()


def _eval_pack(n: Tree, frame: Frame) -> ShkValue:
    return ShkArray([eval_node(ch, frame) for ch in n.children])


# ---- Statement handlers ----


def _eval_returnstmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_return_stmt(n.children, frame, eval_func=eval_node)


def _eval_returnif(n: Tree, frame: Frame) -> ShkValue:
    return eval_return_if(n.children, frame, eval_func=eval_node)


def _eval_throwstmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_throw_stmt(n.children, frame, eval_func=eval_node)


def _eval_assignstmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_assign_stmt(
        n.children, frame, eval_node, apply_op, evaluate_index_operand
    )


def _eval_let(n: Tree, frame: Frame) -> ShkValue:
    return eval_let_stmt(n, frame, eval_node, apply_op, evaluate_index_operand)


def _eval_postfixif(n: Tree, frame: Frame) -> ShkValue:
    return _postfix_eval_if(n.children, frame, eval_func=eval_node, truthy_fn=is_truthy)


def _eval_postfixunless(n: Tree, frame: Frame) -> ShkValue:
    return _postfix_eval_unless(
        n.children, frame, eval_func=eval_node, truthy_fn=is_truthy
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
    return eval_assert(n.children, frame, eval_func=eval_node)


def _eval_bind(n: Tree, frame: Frame) -> ShkValue:
    return eval_apply_assign(
        n.children, frame, eval_node, apply_op, evaluate_index_operand
    )


def _eval_destructure(n: Tree, frame: Frame) -> ShkValue:
    return eval_destructure(n, frame, eval_node, create=False, allow_broadcast=False)


def _eval_destructure_walrus(n: Tree, frame: Frame) -> ShkValue:
    return eval_destructure(n, frame, eval_node, create=True, allow_broadcast=True)


def _eval_awaitstmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_await_stmt(n, frame, eval_node)


def _eval_awaitanycall(n: Tree, frame: Frame) -> ShkValue:
    return eval_await_any_call(n, frame, eval_node)


def _eval_awaitallcall(n: Tree, frame: Frame) -> ShkValue:
    return eval_await_all_call(n, frame, eval_node)


def _eval_usingstmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_using_stmt(n, frame, eval_node)


def _eval_callstmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_call_stmt(n, frame, eval_node)


def _eval_catchexpr(n: Tree, frame: Frame) -> ShkValue:
    return eval_catch_expr(n.children, frame, eval_node)


def _eval_catchstmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_catch_stmt(n.children, frame, eval_node)


# ---- Control flow handlers ----


def _eval_ifstmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_if_stmt(n, frame, eval_node)


def _eval_whilestmt(n: Tree, frame: Frame) -> ShkValue:
    return eval_while_stmt(n, frame, eval_node)


def _eval_breakstmt_node(_n: Tree, frame: Frame) -> ShkValue:
    return _eval_break_stmt(frame)


def _eval_continuestmt_node(_n: Tree, frame: Frame) -> ShkValue:
    return _eval_continue_stmt(frame)


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


def _eval_inlinebody(n: Tree, frame: Frame) -> ShkValue:
    return eval_inline_body(n, frame, eval_node)


def _eval_indentblock(n: Tree, frame: Frame) -> ShkValue:
    return eval_indent_block(n, frame, eval_node)


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

    last_value: ShkValue = ShkNull()
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
    return std_mixin(frame, [module])


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
    "nullish": _eval_nullish,
    "nullsafe": _eval_nullsafe,
    "await_value": _eval_await_value,
    "subject": _eval_subject,
    "keyexpr": _eval_keyexpr,
    "slicearm_expr": _eval_slicearm_expr,
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
    "awaitstmt": _eval_awaitstmt,
    "awaitanycall": _eval_awaitanycall,
    "awaitallcall": _eval_awaitallcall,
    "usingstmt": _eval_usingstmt,
    "callstmt": _eval_callstmt,
    "catchexpr": _eval_catchexpr,
    "catchstmt": _eval_catchstmt,
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
    "no_anchor": _eval_group,
    "inlinebody": _eval_inlinebody,
    "indentblock": _eval_indentblock,
    "onelineguard": _eval_onelineguard,
    "emitexpr": _eval_emitexpr,
    "import_expr": _eval_import_expr,
    # Fanout/rebind
    "fanoutblock": _eval_fanoutblock,
    "valuefan": _eval_valuefan,
    "rebind_primary": _eval_rebind_primary,
    "rebind_primary_grouped": _eval_rebind_primary,
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
    TT.NIL: lambda _, __: ShkNull(),
}
