from __future__ import annotations

import hashlib
from typing import Callable, Dict, List, Optional
import pathlib

from .token_types import TT
from .lexer_rd import Lexer
from .stdlib import std_mixin
from .tree import (
    Node,
    Tree,
    Tok,
    is_token,
    is_tree,
    tree_children,
    tree_label,
    unwrap_node,
)
from .runtime import (
    Frame,
    ShkArray,
    ShkFan,
    ShkBool,
    ShkNil,
    LazyOnceThunk,
    OnceBinding,
    ShkString,
    ShkValue,
    ShakarBreakSignal,
    ShakarContinueSignal,
    ShakarImportError,
    ShakarRuntimeError,
    ShakarTypeError,
    OnceSiteState,
    _STATIC_ONCE_CELLS,
    _STATIC_ONCE_CELLS_LOCK,
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
    eval_assign_pun,
)
from .eval.let import eval_let_stmt
from .types import SourceSpanInfo

EvalFn = Callable[[Node, Frame], ShkValue]


def _append_once_fingerprint(node: Node, parts: List[str]) -> None:
    if is_token(node):
        tok_type = node.type.name if hasattr(node.type, "name") else str(node.type)
        parts.append(f"T:{tok_type}:{node.value!r}:{node.line}:{node.column};")
        return

    if not is_tree(node):
        parts.append("N:unknown;")
        return

    parts.append(f"N:{node.data}:{len(node.children)}:")
    meta = node.meta
    if meta:
        parts.append(f"M:{meta.line}:{meta.column}:{meta.end_line}:{meta.end_column}:")

    attrs = node.attrs
    if attrs:
        for key in sorted(attrs.keys()):
            if key == "once_id":
                continue
            parts.append(f"A:{key}={attrs[key]!r}:")

    for child in node.children:
        _append_once_fingerprint(child, parts)

    parts.append(");")


def _deterministic_once_id(node: Tree) -> int:
    """Compute a stable fallback id for lowered once_expr nodes missing attrs."""
    parts: List[str] = []
    _append_once_fingerprint(node, parts)
    digest = hashlib.sha256("".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def once_node_id(node: Tree) -> int:
    """Read once-site id, materializing deterministic fallback if attrs are absent."""
    attrs = node.attrs
    if attrs and "once_id" in attrs:
        once_id = attrs["once_id"]
        if isinstance(once_id, int):
            return once_id
        raise ShakarRuntimeError("once_expr node has invalid once_id metadata")

    once_id = _deterministic_once_id(node)
    if attrs:
        attrs["once_id"] = once_id
    else:
        node.attrs = {"once_id": once_id}

    return once_id


def enrich_error_context(exc: ShakarRuntimeError, node: Node, frame: Frame) -> None:
    context = exc.context

    if context.call_stack is None:
        call_stack = frame.call_stack_snapshot()
        if call_stack:
            context.call_stack = call_stack

    if context.py_trace is None and debug_py_trace_enabled() and exc.__traceback__:
        context.py_trace = exc.__traceback__

    if context.span is None:
        line, col, end_line, end_col = _resolve_error_span(node, frame)
        if line is not None and col is not None:
            context.span = SourceSpanInfo(
                line=line,
                column=col,
                end_line=end_line,
                end_column=end_col,
            )

    context.enriched = True


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
        result = eval_node(ast, frame)

        return result
    except ShakarRuntimeError as e:
        enrich_error_context(e, ast, frame)
        raise


# ---------------- Core evaluator ----------------


def eval_node(n: Node, frame: Frame) -> ShkValue:
    check_cancel(frame)
    try:
        return _eval_node_inner(n, frame)
    except ShakarRuntimeError as e:
        enrich_error_context(e, n, frame)
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


def _once_flags(node: Tree) -> tuple[bool, bool]:
    """Extract `static` and `lazy` modifier flags from a once expression."""
    is_static = False
    is_lazy = False

    for child in node.children:
        if tree_label(child) != "once_modifiers":
            continue
        for modifier in child.children:
            if not is_token(modifier):
                continue
            if modifier.value == "static":
                is_static = True
            elif modifier.value == "lazy":
                is_lazy = True

    return is_static, is_lazy


def _once_body(node: Tree) -> Node:
    """Return the expression body from a once expression node."""
    if not node.children:
        raise ShakarRuntimeError("Malformed once expression")
    return node.children[-1]


def _build_once_binding(node: Tree, frame: Frame) -> OnceBinding:
    """Build once binding from an once_expr AST node.

    Uses the caller's frame directly so side effects (walrus bindings,
    assignments) are visible in the surrounding scope.
    """
    is_static, _ = _once_flags(node)
    body = _once_body(node)

    # Detect multi-statement block body (indented or braced with stmtlist).
    # Inline single-expression bodies (body with inline=True and no stmtlist)
    # are NOT blocks — they evaluate in the current frame.
    body_is_block = False
    if tree_label(body) == "body":
        attrs = getattr(body, "attrs", None)
        if attrs and not attrs.get("inline", True):
            # Indented block => always a block body
            body_is_block = True
        else:
            # Inline brace block: has stmtlist child
            for ch in tree_children(body):
                if tree_label(ch) == "stmtlist":
                    body_is_block = True
                    break

    return OnceBinding(
        node_id=once_node_id(node),
        body=body,
        def_frame=frame,
        fn_frame=frame.get_function_frame(),
        is_static=is_static,
        is_block=body_is_block,
    )


def eval_once_binding(node: Tree, frame: Frame) -> OnceBinding:
    """Build once binding metadata without resolving the once body."""
    return _build_once_binding(node, frame)


def _once_site_for_binding(binding: OnceBinding) -> OnceSiteState:
    """Locate or create the OnceSiteState for a binding.

    Uses double-checked locking on the owning cell map (static global or
    per-function-frame) to safely get-or-create the site state."""

    if binding.is_static:
        cells = _STATIC_ONCE_CELLS
        map_lock = _STATIC_ONCE_CELLS_LOCK
        site = cells.get(binding.node_id)
        if site is None:
            with map_lock:
                site = cells.get(binding.node_id)
                if site is None:
                    site = OnceSiteState(is_block=binding.is_block)
                    cells[binding.node_id] = site
        return site

    return binding.fn_frame.get_or_create_once_site(
        binding.node_id, is_block=binding.is_block
    )


# -- State transition helpers ------------------------------------------------
# All once-site state mutations flow through these four functions.
# No code outside this group may write to OnceSiteState fields directly.


def _once_enter_eval(site: OnceSiteState, binding: OnceBinding) -> None:
    """Transition: uninitialized => evaluating.

    Must be called under site.lock. Raises on circular re-entry."""

    if site.status == "evaluating":
        raise ShakarRuntimeError("circular once initialization")

    site.status = "evaluating"


def _once_commit_success(
    site: OnceSiteState,
    value: ShkValue,
    binding: OnceBinding,
    block_binding_names: Optional[List[str]] = None,
) -> None:
    """Transition: evaluating => initialized.

    Stores the cached value. For block-form bodies, also records the
    names introduced by the block and the source frame for future
    rematerialization."""

    site.value = value
    site.status = "initialized"

    if binding.is_block and block_binding_names is not None:
        site.block_binding_names = block_binding_names
        # TODO: retains the full call frame for the process lifetime;
        # consider snapshotting only the needed bindings in the C port.
        site.block_source_frame = binding.def_frame


def _once_commit_error(site: OnceSiteState, exc: ShakarRuntimeError) -> None:
    """Transition: evaluating => failed.

    Stores the error for replay on subsequent access."""

    site.error = exc
    site.status = "failed"


def _once_materialize_into_frame(site: OnceSiteState, binding: OnceBinding) -> None:
    """Project cached block-local names into the current definition frame.

    When the source frame (where the block body ran) differs from the
    current def_frame (e.g. static once in a new call), installs aliases
    that delegate reads and writes to the source frame. This keeps
    closures and materialized bindings coherent -- both reference the same
    underlying storage.

    On re-entry in the same scope (e.g. once block in a while loop), the
    names already exist in def_frame and are skipped."""

    if not site.is_block or not site.block_binding_names:
        return

    source = site.block_source_frame
    target = binding.def_frame

    # Same frame (e.g. while loop re-entry) -- bindings are already live.
    if target is source:
        return

    if source is None:
        raise ShakarRuntimeError("once block rematerialization missing source frame")

    pending: list[str] = []
    for name in site.block_binding_names:
        existing_alias = target.get_block_alias(name)
        if existing_alias is source:
            # Same once site re-entry (e.g. while loop) -- already installed.
            continue
        pending.append(name)

    # Validate all names before installing any alias so rematerialization is
    # atomic when duplicate-name errors occur.
    for name in pending:
        # Enforce the same duplicate-definition check that fires during
        # first initialization -- catches conflicts with vars, let scopes,
        # lazy once bindings, and aliases from a different static once site.
        target.ensure_name_available_in_current_scope(name)

    for name in pending:
        target.set_block_alias(name, source)


def _read_initialized_site(site: OnceSiteState) -> ShkValue:
    """Return cached value or re-raise cached error for a terminal site."""

    if site.status == "failed":
        assert site.error is not None
        raise site.error

    if site.value is None:
        raise ShakarRuntimeError("once initializer did not produce a value")

    return site.value


def _resolve_once(binding: OnceBinding) -> ShkValue:
    """Resolve a once binding, evaluating at most once per cell key.

    Acquires the site lock for the full initialization or cache-hit path."""

    site = _once_site_for_binding(binding)

    with site.lock:
        # Cache hit: replay value/error and rematerialize block names.
        if site.status in ("initialized", "failed"):
            value = _read_initialized_site(site)
            _once_materialize_into_frame(site, binding)

            return value

        _once_enter_eval(site, binding)

        try:
            block_names: Optional[List[str]] = None

            if binding.is_block:
                # Snapshot names before block eval to diff new bindings.
                prior_names = set(binding.def_frame.local_var_names())
                # Evaluate directly in def_frame so closures capture the same
                # scope that receives the materialized bindings.
                value = eval_body_node(
                    binding.body,
                    binding.def_frame,
                    eval_node,
                )
                # Cache the names (not values) introduced by the block.
                # Rematerialization reads live values from the source frame
                # so closures and outer bindings stay coherent across calls.
                block_names = [
                    name
                    for name in binding.def_frame.local_var_names()
                    if name not in prior_names
                ]
            else:
                value = eval_node(binding.body, binding.def_frame)

            _once_commit_success(site, value, binding, block_names)

            return value
        except ShakarRuntimeError as exc:
            _once_commit_error(site, exc)
            raise
        finally:
            if site.status == "evaluating":
                site.status = "uninitialized"


def _register_lazy_once(
    once_node: Tree,
    walrus_node: Tree,
    frame: Frame,
    *,
    resolve: bool,
) -> ShkValue:
    """Register a lazy once-walrus binding, deferring evaluation to first read.

    Extracts the name and value expression from the walrus body, builds an
    OnceBinding with body = value_expr (not the full walrus), and stores a
    LazyOnceThunk in frame.lazy_once.  When the name is first read via
    frame.get(), the thunk resolves through the normal once-cell machinery.
    """
    from .eval.common import expect_ident_token

    if len(walrus_node.children) != 2:
        raise ShakarRuntimeError("Malformed once walrus expression")

    name_node, value_node = walrus_node.children
    name = expect_ident_token(name_node, "once walrus target")

    # Build binding with body = value expression (skip the walrus wrapper
    # so resolution produces the value directly without re-defining the name).
    binding = _build_once_binding(once_node, frame)
    binding.body = value_node

    site_id = binding.node_id
    existing_site_id = frame.local_lazy_once_site_id(name)

    # Repeated execution of the same lexical site (for example in loops)
    # must reuse prior registration. Conflicting declarations must keep
    # normal duplicate-name errors.
    if existing_site_id == site_id:
        if not frame.has_local_lazy_once(name) and not frame.has_local_var(name):
            frame.register_lazy_once(
                name,
                LazyOnceThunk(resolve=lambda: _resolve_once(binding)),
            )
    else:
        frame.ensure_name_available_in_current_scope(name)
        frame.register_lazy_once(
            name,
            LazyOnceThunk(resolve=lambda: _resolve_once(binding)),
        )
        frame.register_lazy_once_site(name, site_id)

    # In value contexts, produce the bound value (which resolves lazily on read).
    # In discard contexts, keep registration-only behavior.
    if resolve:
        return frame.get(name)

    return ShkNil()


def _eval_once_expr(n: Tree, frame: Frame) -> ShkValue:
    """Evaluate once expression: eager by default, lazy only with [lazy] modifier.

    All forms are eager unless ``[lazy]`` is specified.  ``once[lazy]: x := expr``
    defers evaluation until ``x`` is first read.  Block bodies
    (``once:\\n  stmts``) evaluate in a child frame for scoping.  Statement-discard
    mode is carried on ``n.attrs['discard']`` for lazy once walrus registration.
    """
    _, is_lazy = _once_flags(n)
    body = _once_body(n)
    discard = bool(n.attrs and n.attrs.get("discard"))

    # Lazy path: defer walrus body until first read of the name.
    # Parser validates that lazy always has a walrus body (inline, not block).
    if is_lazy:
        walrus_node = unwrap_node(body, {"expr", "stmt"})
        return _register_lazy_once(n, walrus_node, frame, resolve=not discard)

    # Eager path: bare expr, eager walrus, or block body.
    binding = _build_once_binding(n, frame)
    value = _resolve_once(binding)

    # For eager walrus (once: x := expr), the walrus body only runs on
    # first evaluation. On subsequent hits (cached), re-define the name
    # so the binding is visible in the current frame.
    inner = unwrap_node(body, {"expr", "stmt"})
    if tree_label(inner) == "walrus":
        from .eval.common import expect_ident_token

        inner_children = tree_children(inner)
        name = expect_ident_token(inner_children[0], "once walrus target")
        if not frame.has_local_var(name):
            frame.define_new_ident(name, value)

    return value


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

    saved_dot = frame.dot

    try:
        return eval_node(child, frame)
    finally:
        frame.dot = saved_dot


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
        if tree_label(child) == "fan_modifiers":
            modifiers = child
        elif tree_label(child) == "fan_items":
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
    if tree_label(names_node) != "import_names":
        raise ShakarRuntimeError("Expected import names in brackets")

    names: list[str] = []
    for child in tree_children(names_node):
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


def _eval_assign_pun(n: Tree, frame: Frame) -> ShkValue:
    return eval_assign_pun(n, frame, eval_node, apply_op, evaluate_index_operand)


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
    "once_expr": _eval_once_expr,
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
    "assign_pun": _eval_assign_pun,
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
