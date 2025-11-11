
from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional
import math
from lark import Tree, Token

from contextlib import contextmanager

from shakar_runtime import (
    BoundMethod,
    BuiltinMethod,
    DeferEntry,
    Descriptor,
    Env,
    ShkArray,
    ShkBool,
    ShkFn,
    ShkNull,
    ShkNumber,
    ShkObject,
    ShkSelector,
    ShkString,
    ShakarArityError,
    ShakarAssertionError,
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

from shakar_utils import fanout_values, normalize_object_key, shk_equals, value_in_list
from shakar_tree import (
    child_by_label,
    is_token_node,
    is_tree_node,
    node_meta,
    tree_children,
    tree_label,
)

from eval.selector import (
    eval_selectorliteral,
    evaluate_selectorlist,
    apply_selectors_to_value,
    selector_iter_values,
)

from eval.destructure import (
    evaluate_destructure_rhs,
    assign_pattern as destructure_assign_pattern,
    infer_implicit_binders as destructure_infer_implicit_binders,
    apply_comp_binders as destructure_apply_comp_binders,
)

from eval.mutation import (
    set_field_value,
    set_index_value,
    index_value,
    slice_value,
    get_field_value,
)

class _RebindContext:
    """Tracks an assignable slot (identifier or field) so tail ops can write back."""
    __slots__ = ("value", "setter")

    def __init__(self, value: Any, setter: Callable[[Any], None]) -> None:
        self.value = value
        self.setter = setter

class _FanContext:
    """Represents `.={a,b}` fan assignments; stores every target's context."""
    __slots__ = ("contexts", "values")

    def __init__(self, contexts: List[_RebindContext]) -> None:
        self.contexts = contexts
        self.values = [ctx.value for ctx in contexts]

    def update_from_contexts(self) -> None:
        self.values = [ctx.value for ctx in self.contexts]

    def snapshot(self) -> List[Any]:
        return list(self.values)

# ---------------- Public API ----------------

def eval_expr(ast: Any, env: Optional[Env]=None, source: Optional[str]=None) -> Any:
    init_stdlib()
    if env is None:
        env = Env(source=source)
    else:
        if source is not None:
            env.source = source
        elif not hasattr(env, 'source'):
            env.source = None
    return eval_node(ast, env)

# ---------------- Core evaluator ----------------

def _token_kind(node: Any) -> Optional[str]:
    return node.type if is_token_node(node) else None

def _expect_ident_token(node: Any, context: str) -> str:
    if is_token_node(node) and _token_kind(node) == 'IDENT':
        return node.value
    raise ShakarRuntimeError(f"{context} must be an identifier")

def _ident_token_value(node: Any) -> Optional[str]:
    if is_token_node(node) and _token_kind(node) == 'IDENT':
        return node.value
    return None

def _is_literal_node(node: Any) -> bool:
    return not isinstance(node, (Tree, Token))

def _get_source_segment(node: Any, env: Env) -> Optional[str]:
    source = getattr(env, 'source', None)
    if source is None:
        return None
    meta = node_meta(node)
    if meta is None:
        return None
    start = getattr(meta, "start_pos", None)
    end = getattr(meta, "end_pos", None)
    if start is None or end is None:
        return None
    return source[start:end]

def eval_node(n: Any, env: Env) -> Any:
    if _is_literal_node(n):
        return n
    if is_token_node(n):
        return _eval_token(n, env)

    d = n.data
    handler = _NODE_DISPATCH.get(d)
    if handler is not None:
        return handler(n, env)
    match d:
        # common wrapper nodes (delegate to single child)
        case 'start_noindent' | 'start_indented' | 'stmtlist':
            return _eval_program(n.children, env)
        case 'stmt':
            return _eval_program(n.children, env)
        case 'literal' | 'primary' | 'expr' | 'expr_nc':
            if len(n.children) == 1:
                return eval_node(n.children[0], env)
            if len(n.children) == 0 and d == 'literal':
                return _eval_keyword_literal(n)
            raise ShakarRuntimeError(f"Unsupported wrapper shape {d} with {len(n.children)} children")
        case 'array':
            return ShkArray([eval_node(c, env) for c in n.children])
        case 'object':
            return _eval_object(n, env)
        case 'unary' | 'unary_nc':
            op, rhs_node = n.children
            return _eval_unary(op, rhs_node, env)
        case 'pow' | 'pow_nc':
            return _eval_infix(n.children, env, right_assoc_ops={'**', 'POW'})
        case 'mul' | 'mul_nc' | 'add' | 'add_nc':
            return _eval_infix(n.children, env)
        case 'explicit_chain':
            head, *ops = n.children
            if ops and tree_label(ops[-1]) in {'incr', 'decr'}:
                tail = ops[-1]
                # ++/-- mutate the final receiver; resolve assignable context first.
                context = _resolve_chain_assignment(head, ops[:-1], env)
                delta = 1 if tree_label(tail) == 'incr' else -1
                if isinstance(context, _FanContext):
                    raise ShakarRuntimeError("++/-- not supported on field fan assignments")
                old_val, _ = _apply_numeric_delta(context, delta)
                return old_val
            val = eval_node(head, env)
            for op in ops:
                val = _apply_op(val, op, env)
            if isinstance(val, _RebindContext):
                final = val.value
                val.setter(final)
                return final
            if isinstance(val, _FanContext):
                return ShkArray(val.snapshot())
            return val
        case 'implicit_chain':
            return _eval_implicit_chain(n.children, env)
        case 'call':
            args_node = n.children[0] if n.children else None
            args = _eval_args_node(args_node, env)
            cal = env.get('')  # unreachable in practice
            return _call_value(cal, args, env)
        case 'and' | 'or' | 'and_nc' | 'or_nc':
            return _eval_logical(d, n.children, env)
        case 'walrus' | 'walrus_nc':
            return _eval_walrus(n.children, env)
        case 'returnstmt':
            return _eval_return_stmt(n.children, env)
        case 'breakstmt':
            return _eval_break_stmt(env)
        case 'continuestmt':
            return _eval_continue_stmt(env)
        case 'assignstmt':
            return _eval_assign_stmt(n.children, env)
        case 'compound_assign':
            return _eval_compound_assign(n.children, env)
        case 'fndef':
            return _eval_fn_def(n.children, env)
        case 'deferstmt':
            return _eval_defer_stmt(n.children, env)
        case 'assert':
            return _eval_assert(n.children, env)
        case 'bind' | 'bind_nc':
            return _eval_apply_assign(n.children, env)
        case 'subject':
            return _get_subject(env)
        case 'keyexpr' | 'keyexpr_nc':
            return eval_node(n.children[0], env) if n.children else ShkNull()
        case 'destructure':
            return _eval_destructure(n, env, create=False, allow_broadcast=False)
        case 'destructure_walrus':
            return _eval_destructure(n, env, create=True, allow_broadcast=True)
        case _:
            raise ShakarRuntimeError(f"Unknown node: {d}")

# ---------------- Tokens ----------------

def _eval_token(t: Token, env: Env) -> Any:
    handler = _TOKEN_DISPATCH.get(t.type)
    if handler:
        return handler(t, env)
    if t.type == 'IDENT':
        return env.get(t.value)
    raise ShakarRuntimeError(f"Unhandled token {t.type}:{t.value}")

def _eval_keyword_literal(node: Tree) -> Any:
    meta = node_meta(node)
    if meta is None:
        raise ShakarRuntimeError("Missing metadata for literal")
    end = getattr(meta, "end_pos", None)
    start = getattr(meta, "start_pos", None)
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

def _eval_program(children: List[Any], env: Env, allow_loop_control: bool=False) -> Any:
    """Run a stmt list under a fresh defer scope, returning last value."""
    result: Any = ShkNull()
    skip_tokens = {'SEMI', '_NL', 'INDENT', 'DEDENT'}
    _push_defer_scope(env)
    try:
        try:
            for child in children:
                tok_kind = _token_kind(child)
                if tok_kind in skip_tokens:
                    continue
                result = eval_node(child, env)
        except ShakarBreakSignal:
            if allow_loop_control:
                raise
            raise ShakarRuntimeError("break outside of a loop") from None
        except ShakarContinueSignal:
            if allow_loop_control:
                raise
            raise ShakarRuntimeError("continue outside of a loop") from None
    finally:
        _pop_defer_scope(env)
    return result

def _get_subject(env: Env) -> Any:
    if env.dot is None:
        raise ShakarRuntimeError("No subject available for '.'")
    return env.dot

def _push_defer_scope(env: Env) -> None:
    # each scope owns its own LIFO queue of entries; nested scopes flush before parents.
    env.push_defer_frame()

def _pop_defer_scope(env: Env) -> None:
    entries = env.pop_defer_frame()
    if entries:
        _run_defer_entries(entries)

_DEFER_UNVISITED = 0  # entry not touched yet
_DEFER_VISITING = 1   # DFS currently walking this entry's deps
_DEFER_DONE = 2       # entry executed (or scheduled) already

def _run_defer_entries(entries: List[DeferEntry]) -> None:
    if not entries:
        return
    label_map: dict[str, int] = {}
    for idx, entry in enumerate(entries):
        if entry.label:
            label_map[entry.label] = idx
    state = [_DEFER_UNVISITED] * len(entries)  # per-entry visit status for topo walk

    def run_index(idx: int) -> None:
        marker = state[idx]
        if marker == _DEFER_DONE:
            return
        if marker == _DEFER_VISITING:
            raise ShakarRuntimeError("Defer dependency cycle detected")
        state[idx] = _DEFER_VISITING
        entry = entries[idx]
        for dep in entry.deps:
            dep_idx = label_map.get(dep)
            if dep_idx is None:
                raise ShakarRuntimeError(f"Unknown defer handle '{dep}'")
            run_index(dep_idx)
        state[idx] = _DEFER_DONE
        entry.thunk()

    # entries still execute in overall LIFO order; deps may cause earlier entries to run first.
    for idx in reversed(range(len(entries))):
        run_index(idx)

def _schedule_defer(env: Env, thunk: Callable[[], None], label: str | None=None, deps: List[str] | None=None) -> None:
    if not env.has_defer_frame():
        raise ShakarRuntimeError("Cannot use defer outside of a block")
    frame = env.current_defer_frame()
    entry = DeferEntry(thunk=thunk, label=label, deps=list(deps or []))
    if label:
        for existing in frame:
            if existing.label == label:
                raise ShakarRuntimeError(f"Duplicate defer handle '{label}'")
    frame.append(entry)  # defer runs when the owning scope unwinds

def _eval_implicit_chain(ops: List[Any], env: Env) -> Any:
    """Evaluate `.foo().bar` style chains using the current subject anchor."""
    val = _get_subject(env)
    for op in ops:
        val = _apply_op(val, op, env)
    return val

def _eval_inline_body(node: Any, env: Env, allow_loop_control: bool=False) -> Any:
    """Execute a single-statement body or `{}` block attached to colon headers."""
    if tree_label(node) == 'inlinebody':
        for child in tree_children(node):
            if tree_label(child) == 'stmtlist':
                return _eval_program(child.children, env, allow_loop_control=allow_loop_control)
        if not tree_children(node):
            return ShkNull()
        return eval_node(node.children[0], env)
    return eval_node(node, env)

def _eval_indent_block(node: Tree, env: Env, allow_loop_control: bool=False) -> Any:
    return _eval_program(node.children, env, allow_loop_control=allow_loop_control)

def _eval_oneline_guard(children: List[Any], env: Env) -> Any:
    branches: List[Tree] = []
    else_body: Tree | None = None
    for child in children:
        data = tree_label(child)
        if data == 'guardbranch':
            branches.append(child)
        elif data == 'inlinebody':
            else_body = child
    outer_dot = env.dot
    for branch in branches:
        if not is_tree_node(branch) or len(branch.children) != 2:
            raise ShakarRuntimeError("Malformed guard branch")
        cond_node, body_node = branch.children
        with _temporary_subject(env, outer_dot):
            cond_val = eval_node(cond_node, env)
        if _is_truthy(cond_val):
            with _temporary_subject(env, outer_dot):
                return _eval_inline_body(body_node, env)
    if else_body is not None:
        with _temporary_subject(env, outer_dot):
            return _eval_inline_body(else_body, env)
    env.dot = outer_dot
    return ShkNull()

# ---------------- Assignment ----------------

def _eval_walrus(children: List[Any], env: Env) -> Any:
    """Implements `name := expr` inline assignments."""
    if len(children) != 2:
        raise ShakarRuntimeError("Malformed walrus expression")
    name_node, value_node = children
    name = _expect_ident_token(name_node, "Walrus target")
    value = eval_node(value_node, env)
    return _assign_ident(name, value, env, create=True)

def _eval_assign_stmt(children: List[Any], env: Env) -> Any:
    """Handles simple `lhs = rhs` statements (no destructuring)."""
    if len(children) < 2:
        raise ShakarRuntimeError("Malformed assignment statement")
    lvalue_node = children[0]
    value_node = children[-1]
    value = eval_node(value_node, env)
    _assign_lvalue(lvalue_node, value, env, create=False)
    return ShkNull()

def _eval_return_stmt(children: List[Any], env: Env) -> Any:
    """implements `return` with optional expression, unwinding via control signal."""
    if _current_function_env(env) is None:
        raise ShakarRuntimeError("return outside of a function")
    value = eval_node(children[0], env) if children else ShkNull()
    raise ShakarReturnSignal(value)

def _eval_break_stmt(env: Env) -> Any:
    raise ShakarBreakSignal()

def _eval_continue_stmt(env: Env) -> Any:
    raise ShakarContinueSignal()

def _eval_defer_stmt(children: List[Any], env: Env) -> Any:
    """Schedule a deferred thunk, unpacking optional label/dependency metadata."""
    if not children:
        raise ShakarRuntimeError("Malformed defer statement")
    idx = 0
    label = None
    # parser guarantees the shape [label? , body , deps?]; walk in that order.
    if is_tree_node(children[0]) and tree_label(children[0]) == 'deferlabel':
        label = _expect_ident_token(children[0].children[0], "Defer label")
        idx += 1
    if idx >= len(children):
        raise ShakarRuntimeError("Missing deferred body")
    body_wrapper = children[idx]  # either a simple call node or Tree('deferblock', [...])
    idx += 1
    deps: List[str] = []
    if idx < len(children):
        deps_node = children[idx]
        if is_tree_node(deps_node) and tree_label(deps_node) == 'deferdeps':
            deps = [
                _expect_ident_token(tok, "Defer dependency")
                for tok in tree_children(deps_node)
                if is_token_node(tok)
            ]
            idx += 1
    if idx != len(children):
        raise ShakarRuntimeError("Unexpected defer statement shape")
    body_kind = 'block' if is_tree_node(body_wrapper) and tree_label(body_wrapper) == 'deferblock' else 'call'
    if body_kind == 'block':
        # unpack the actual inline/indent block the parser wrapped in deferblock
        payload = body_wrapper.children[0] if is_tree_node(body_wrapper) and body_wrapper.children else Tree('inlinebody', [])
    else:
        payload = body_wrapper
    saved_dot = env.dot
    source = getattr(env, 'source', None)

    def thunk() -> None:
        child_env = Env(parent=env, dot=saved_dot, source=source)
        _push_defer_scope(child_env)
        try:
            if body_kind == 'block':
                _eval_inline_body(payload, child_env)
            else:
                eval_node(payload, child_env)
        finally:
            _pop_defer_scope(child_env)

    _schedule_defer(env, thunk, label=label, deps=deps)
    return ShkNull()

def _eval_fn_def(children: List[Any], env: Env) -> Any:
    if not children:
        raise ShakarRuntimeError("Malformed function definition")
    name = _expect_ident_token(children[0], "Function name")
    params_node = None
    body_node = None
    for node in children[1:]:
        if params_node is None and is_tree_node(node) and tree_label(node) == 'paramlist':
            params_node = node
        elif body_node is None and is_tree_node(node) and tree_label(node) in {'inlinebody', 'indentblock'}:
            body_node = node
    if body_node is None:
        body_node = Tree('inlinebody', [])
    params = _extract_param_names(params_node, context="function definition")
    fn_value = ShkFn(params=params, body=body_node, env=Env(parent=env, dot=None))
    _assign_ident(name, fn_value, env, create=True)
    return ShkNull()

def _eval_assert(children: List[Any], env: Env) -> Any:
    """Evaluate `assert expr [, message]`, raising ShakarAssertionError when falsy."""
    if not children:
        raise ShakarRuntimeError("Malformed assert statement")
    cond_val = eval_node(children[0], env)
    if _is_truthy(cond_val):
        return ShkNull()
    message = f"Assertion failed: {_assert_source_snippet(children[0], env)}"
    if len(children) > 1:
        msg_val = eval_node(children[1], env)
        message = _stringify(msg_val)
    raise ShakarAssertionError(message)

def _eval_if_stmt(n: Tree, env: Env) -> Any:
    children = tree_children(n)
    cond_node = None
    body_node = None
    elif_clauses: list[tuple[Any, Any]] = []
    else_body = None
    for child in children:
        if is_token_node(child):
            continue
        label = tree_label(child)
        if label == 'elifclause':
            clause_cond, clause_body = _extract_clause(child, label='elif')
            elif_clauses.append((clause_cond, clause_body))
            continue
        if label == 'elseclause':
            _, else_body = _extract_clause(child, label='else')
            continue
        if cond_node is None:
            cond_node = child
        elif body_node is None:
            body_node = child
    if cond_node is None or body_node is None:
        raise ShakarRuntimeError("Malformed if statement")
    if _is_truthy(eval_node(cond_node, env)):
        return _execute_loop_body(body_node, env)
    for clause_cond, clause_body in elif_clauses:
        if _is_truthy(eval_node(clause_cond, env)):
            return _execute_loop_body(clause_body, env)
    if else_body is not None:
        return _execute_loop_body(else_body, env)
    return ShkNull()

def _eval_compound_assign(children: List[Any], env: Env) -> Any:
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
    current_value = _read_lvalue_value(lvalue_node, env)
    rhs_value = eval_node(rhs_node, env)
    new_value = _apply_binary_operator(op_symbol, current_value, rhs_value)
    _assign_lvalue(lvalue_node, new_value, env, create=False)
    return ShkNull()

def _eval_apply_assign(children: List[Any], env: Env) -> Any:
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
    return _apply_assign(lvalue_node, rhs_node, env)

def _eval_for_in(n: Tree, env: Env) -> Any:
    pattern_node = None
    iter_expr = None
    body_node = None
    after_in = False
    for child in tree_children(n):
        if is_tree_node(child) and tree_label(child) == 'pattern':
            pattern_node = child
            continue
        if is_token_node(child):
            kind = _token_kind(child)
            if kind == 'FOR':
                continue
            if kind == 'IN':
                after_in = True
                continue
            if kind == 'COLON':
                continue
            if pattern_node is None and kind == 'IDENT':
                pattern_node = Tree('pattern', [child])
                continue
            if after_in and iter_expr is None:
                iter_expr = child
            continue
        if iter_expr is None:
            iter_expr = child
        else:
            body_node = child
    if iter_expr is None or body_node is None:
        raise ShakarRuntimeError("Malformed for-in loop")
    if pattern_node is None:
        raise ShakarRuntimeError("For-in loop missing pattern")
    iter_source = eval_node(iter_expr, env)
    iterable = _iterable_values(iter_source)
    outer_dot = env.dot
    object_pairs: list[tuple[str, Any]] | None = None
    if isinstance(iter_source, ShkObject):
        object_pairs = list(iter_source.slots.items())
    try:
        for idx, value in enumerate(iterable):
            loop_env = Env(parent=env, dot=outer_dot)
            assigned = value
            if object_pairs and _pattern_requires_object_pair(pattern_node):
                key, val = object_pairs[idx]
                assigned = ShkArray([ShkString(key), val])
            _assign_pattern_value(pattern_node, assigned, loop_env, create=True, allow_broadcast=False)
            try:
                _execute_loop_body(body_node, loop_env)
            except ShakarContinueSignal:
                continue
            except ShakarBreakSignal:
                break
    finally:
        env.dot = outer_dot
    return ShkNull()

def _eval_for_subject(n: Tree, env: Env) -> Any:
    iter_expr = None
    body_node = None
    for child in tree_children(n):
        if is_token_node(child):
            continue
        if iter_expr is None:
            iter_expr = child
        else:
            body_node = child
    if iter_expr is None or body_node is None:
        raise ShakarRuntimeError("Malformed subjectful for loop")
    iterable = _iterable_values(eval_node(iter_expr, env))
    outer_dot = env.dot
    try:
        for value in iterable:
            loop_env = Env(parent=env, dot=value)
            try:
                _execute_loop_body(body_node, loop_env)
            except ShakarContinueSignal:
                continue
            except ShakarBreakSignal:
                break
    finally:
        env.dot = outer_dot
    return ShkNull()

def _eval_for_indexed(n: Tree, env: Env) -> Any:
    if n is None:
        raise ShakarRuntimeError("Malformed indexed loop")
    children = tree_children(n)
    binder_nodes: list[Tree] = []
    for child in children:
        if is_tree_node(child) and tree_label(child) in {'binderpattern', 'hoist', 'pattern'}:
            binder_nodes.append(child)
    if not binder_nodes:
        raise ShakarRuntimeError("Indexed loop missing binder")
    iter_expr, body_node = _extract_loop_iter_and_body(children)
    if iter_expr is None or body_node is None:
        raise ShakarRuntimeError("Malformed indexed loop")
    binders = [_coerce_loop_binder(node) for node in binder_nodes]
    iterable = eval_node(iter_expr, env)
    entries = _iter_indexed_entries(iterable, len(binders))
    outer_dot = env.dot
    try:
        for subject, binder_values in entries:
            loop_env = Env(parent=env, dot=subject)
            for binder, binder_value in zip(binders, binder_values):
                target_env = env if binder.get('hoist') else loop_env
                _assign_pattern_value(binder['pattern'], binder_value, target_env, create=True, allow_broadcast=False)
            try:
                _execute_loop_body(body_node, loop_env)
            except ShakarContinueSignal:
                continue
            except ShakarBreakSignal:
                break
    finally:
        env.dot = outer_dot
    return ShkNull()

def _eval_for_map2(n: Tree, env: Env) -> Any:
    return _eval_for_indexed(n, env)

def _eval_destructure(n: Tree, env: Env, create: bool, allow_broadcast: bool) -> Any:
    """Evaluate `a, b := expr` destructures with optional broadcast semantics."""
    if len(n.children) != 2:
        raise ShakarRuntimeError("Malformed destructure")
    pattern_list, rhs_node = n.children
    patterns = [c for c in tree_children(pattern_list) if tree_label(c) == 'pattern']
    if not patterns:
        raise ShakarRuntimeError("Empty destructure pattern")
    values, result = evaluate_destructure_rhs(eval_node, rhs_node, env, len(patterns), allow_broadcast)
    for pat, val in zip(patterns, values):
        _assign_pattern_value(pat, val, env, create, allow_broadcast)
    return result if allow_broadcast else ShkNull()

def _assign_ident(name: str, value: Any, env: Env, create: bool) -> Any:
    try:
        env.set(name, value)
    except ShakarRuntimeError:
        if create:
            env.define(name, value)
        else:
            raise
    return value

def _assign_pattern_value(pattern: Tree, value: Any, env: Env, create: bool, allow_broadcast: bool) -> None:
    def _assign_ident_wrapper(name: str, val: Any, target_env: Env, create_flag: bool) -> None:
        _assign_ident(name, val, target_env, create=create_flag)
    destructure_assign_pattern(eval_node, _assign_ident_wrapper, pattern, value, env, create, allow_broadcast)

def _apply_comp_binders_wrapper(binders: list[dict[str, Any]], element: Any, iter_env: Env, outer_env: Env) -> None:
    destructure_apply_comp_binders(
        lambda pattern, val, target_env, create, allow_broadcast: _assign_pattern_value(pattern, val, target_env, create, allow_broadcast),
        binders,
        element,
        iter_env,
        outer_env,
    )

def _assign_lvalue(node: Any, value: Any, env: Env, create: bool) -> Any:
    if not is_tree_node(node) or tree_label(node) != 'lvalue':
        raise ShakarRuntimeError("Invalid assignment target")
    if not node.children:
        raise ShakarRuntimeError("Empty lvalue")
    head, *ops = node.children
    if not ops and is_token_node(head) and _token_kind(head) == 'IDENT':
        return _assign_ident(head.value, value, env, create=create)
    target = eval_node(head, env)
    if not ops:
        raise ShakarRuntimeError("Malformed lvalue")
    for op in ops[:-1]:
        target = _apply_op(target, op, env)
    final_op = ops[-1]
    label = tree_label(final_op)
    match label:
        case 'field' | 'fieldsel':
            field_name = _expect_ident_token(final_op.children[0], "Field assignment")
            return set_field_value(target, field_name, value, env, create=create)
        case 'lv_index':
            idx_val = _evaluate_index_operand(final_op, env)
            return set_index_value(target, idx_val, value, env)
        case 'fieldfan':
            fieldlist_node = child_by_label(final_op, 'fieldlist')
            if fieldlist_node is None:
                raise ShakarRuntimeError("Malformed field fan-out list")
            names = [tok.value for tok in fieldlist_node.children if _token_kind(tok) == 'IDENT']
            if not names:
                raise ShakarRuntimeError("Empty field fan-out list")
            vals = fanout_values(value, len(names))
            for name, val in zip(names, vals):
                set_field_value(target, name, val, env, create=create)
            return value
    raise ShakarRuntimeError("Unsupported assignment target")

def _read_lvalue_value(node: Any, env: Env) -> Any:
    if not is_tree_node(node) or tree_label(node) != 'lvalue':
        raise ShakarRuntimeError("Invalid lvalue")
    if not node.children:
        raise ShakarRuntimeError("Empty lvalue")
    head, *ops = node.children
    if not ops and is_token_node(head) and _token_kind(head) == 'IDENT':
        return env.get(head.value)
    target = eval_node(head, env)
    if not ops:
        raise ShakarRuntimeError("Malformed lvalue")
    for op in ops[:-1]:
        target = _apply_op(target, op, env)
    final_op = ops[-1]
    label = tree_label(final_op)
    match label:
        case 'field' | 'fieldsel':
            field_name = _expect_ident_token(final_op.children[0], "Field value access")
            return get_field_value(target, field_name, env)
        case 'lv_index':
            idx_val = _evaluate_index_operand(final_op, env)
            return index_value(target, idx_val, env)
    raise ShakarRuntimeError("Compound assignment not supported for this target")

def _apply_assign(lvalue_node: Tree, rhs_node: Tree, env: Env) -> Any:
    head, *ops = lvalue_node.children
    if not ops and _token_kind(head) == 'IDENT':
        target = env.get(head.value)
        old_val = target
        rhs_env = Env(parent=env, dot=old_val)
        new_val = eval_node(rhs_node, rhs_env)
        _assign_ident(head.value, new_val, env, create=False)
        return new_val
    target = eval_node(head, env)
    if isinstance(target, _RebindContext):
        target = target.value
    if not ops:
        raise ShakarRuntimeError("Malformed apply-assign target")
    for op in ops[:-1]:
        target = _apply_op(target, op, env)
    final_op = ops[-1]
    label = tree_label(final_op)
    match label:
        case 'field' | 'fieldsel':
            field_name = _expect_ident_token(final_op.children[0], "Field assignment")
            old_val = get_field_value(target, field_name, env)
            rhs_env = Env(parent=env, dot=old_val)
            new_val = eval_node(rhs_node, rhs_env)
            set_field_value(target, field_name, new_val, env, create=False)
            return new_val
        case 'lv_index':
            idx_val = _evaluate_index_operand(final_op, env)
            old_val = index_value(target, idx_val, env)
            rhs_env = Env(parent=env, dot=old_val)
            new_val = eval_node(rhs_node, rhs_env)
            set_index_value(target, idx_val, new_val, env)
            return new_val
        case 'fieldfan':
            fieldlist_node = child_by_label(final_op, 'fieldlist')
            if fieldlist_node is None:
                raise ShakarRuntimeError("Malformed field fan-out list")
            names = [tok.value for tok in fieldlist_node.children if _token_kind(tok) == 'IDENT']
            if not names:
                raise ShakarRuntimeError("Empty field fan-out list")
            results: list[Any] = []
            for name in names:
                old_val = get_field_value(target, name, env)
                rhs_env = Env(parent=env, dot=old_val)
                new_val = eval_node(rhs_node, rhs_env)
                set_field_value(target, name, new_val, env, create=False)
                results.append(new_val)
            return ShkArray(results)
    raise ShakarRuntimeError("Unsupported apply-assign target")

def _evaluate_index_operand(index_node: Tree, env: Env) -> Any:
    expr_node = _index_expr_from_children(index_node.children)
    return eval_node(expr_node, env)

def _collect_free_identifiers(node: Any, callback) -> None:
    skip_nodes = {'field', 'fieldsel', 'fieldfan', 'fieldlist', 'key_ident', 'key_string'}

    def walk(n: Any) -> None:
        if is_token_node(n):
            if _token_kind(n) == 'IDENT':
                callback(n.value)
            return
        if is_tree_node(n):
            if tree_label(n) == 'amp_lambda':
                return
            if tree_label(n) in skip_nodes:
                return
            for ch in n.children:
                walk(ch)

    walk(node)

def _prepare_comprehension(n: Tree, env: Env, head_nodes: list[Any]) -> tuple[Any, list[dict[str, Any]], Tree | None]:
    comphead = child_by_label(n, 'comphead')
    if comphead is None:
        raise ShakarRuntimeError("Malformed comprehension")
    ifclause = child_by_label(n, 'ifclause')
    iter_expr_node, binders = _parse_comphead(comphead)
    if not binders:
        implicit_names = destructure_infer_implicit_binders(
            head_nodes,
            ifclause,
            env,
            lambda expr, callback: _collect_free_identifiers(expr, callback)
        )
        for name in implicit_names:
            pattern = Tree('pattern', [Token('IDENT', name)])
            binders.append({'pattern': pattern, 'hoist': False})
    iter_val = eval_node(iter_expr_node, env)
    return iter_val, binders, ifclause

def _iterate_comprehension(n: Tree, env: Env, head_nodes: list[Any]) -> Iterable[tuple[Any, Env]]:
    iter_val, binders, ifclause = _prepare_comprehension(n, env, head_nodes)
    outer_dot = env.dot
    try:
        for element in _iterable_values(iter_val):
            iter_env = Env(parent=env, dot=element)
            _apply_comp_binders_wrapper(binders, element, iter_env, env)
            if ifclause is not None:
                cond_node = ifclause.children[-1] if ifclause.children else None
                if cond_node is None:
                    raise ShakarRuntimeError("Malformed comprehension guard")
                cond_val = eval_node(cond_node, iter_env)
                if not _is_truthy(cond_val):
                    continue
            yield element, iter_env
    finally:
        env.dot = outer_dot

def _eval_listcomp(n: Tree, env: Env) -> ShkArray:
    """Evaluate `[expr over xs if cond]` comprehensions using explicit binder envs."""
    body = n.children[0] if n.children else None
    if body is None:
        raise ShakarRuntimeError("Malformed list comprehension")
    items = [eval_node(body, iter_env) for _, iter_env in _iterate_comprehension(n, env, [body])]
    return ShkArray(items)

def _eval_setcomp(n: Tree, env: Env) -> ShkArray:
    body = n.children[0] if n.children else None
    if body is None:
        raise ShakarRuntimeError("Malformed set comprehension")
    items: list[Any] = []
    for _, iter_env in _iterate_comprehension(n, env, [body]):
        result = eval_node(body, iter_env)
        if not value_in_list(items, result):
            items.append(result)
    return ShkArray(items)

def _eval_setliteral(n: Tree, env: Env) -> ShkArray:
    """Set literals desugar to arrays internally; maintain order while deduping."""
    items: list[Any] = []
    for child in tree_children(n):
        val = eval_node(child, env)
        if not value_in_list(items, val):
            items.append(val)
    return ShkArray(items)

def _eval_dictcomp(n: Tree, env: Env) -> ShkObject:
    if len(n.children) < 3:
        raise ShakarRuntimeError("Malformed dict comprehension")
    key_node = n.children[0]
    value_node = n.children[1]
    slots: dict[str, Any] = {}
    for _, iter_env in _iterate_comprehension(n, env, [key_node, value_node]):
        key_val = eval_node(key_node, iter_env)
        value_val = eval_node(value_node, iter_env)
        key_str = normalize_object_key(key_val)
        slots[key_str] = value_val
    return ShkObject(slots)

def _parse_comphead(node: Tree) -> tuple[Any, list[dict[str, Any]]]:
    overspec = child_by_label(node, 'overspec')
    if overspec is None:
        raise ShakarRuntimeError("Malformed comprehension head")
    return _parse_overspec(overspec)

def _parse_overspec(node: Tree) -> tuple[Any, list[dict[str, Any]]]:
    children = list(node.children)
    binders: list[dict[str, Any]] = []
    if not children:
        raise ShakarRuntimeError("Malformed overspec")
    first = children[0]
    if tree_label(first) == 'binderlist':
        if len(children) < 2:
            raise ShakarRuntimeError("Binder list requires a source")
        iter_expr_node = children[1]
        for bp in first.children:
            bp_label = tree_label(bp)
            if bp_label == 'binderpattern' and bp.children:
                pattern_node = bp.children[0]
                pattern_label = tree_label(pattern_node)
                if pattern_label == 'pattern' and pattern_node.children:
                    child = pattern_node.children[0]
                    if tree_label(child) == 'pattern_list':
                        raise ShakarRuntimeError("Binder list cannot use parentheses")
                binders.append({'pattern': bp.children[0], 'hoist': False})
            elif bp_label == 'hoist' and bp.children:
                tok = bp.children[0]
                pattern = Tree('pattern', [tok])
                binders.append({'pattern': pattern, 'hoist': True})
        return iter_expr_node, binders
    iter_expr_node = children[0]
    if len(children) > 1:
        pattern = children[1]
        binders.append({'pattern': pattern, 'hoist': False})
    return iter_expr_node, binders

def _extract_loop_iter_and_body(children: List[Any]) -> tuple[Any | None, Any | None]:
    iter_expr = None
    body_node = None
    for child in children:
        if is_tree_node(child):
            label = tree_label(child)
            if label in {'binderpattern', 'hoist', 'pattern'}:
                continue
            if label in {'inlinebody', 'indentblock'}:
                body_node = child
            elif iter_expr is None:
                iter_expr = child
            else:
                body_node = child
        else:
            kind = _token_kind(child)
            if kind in {'FOR', 'IN', 'LSQB', 'RSQB', 'COMMA', 'COLON'}:
                continue
            if iter_expr is None:
                iter_expr = child
            else:
                body_node = child
    return iter_expr, body_node

def _extract_clause(node: Tree, label: str) -> tuple[Any | None, Any]:
    nodes = [child for child in tree_children(node) if not is_token_node(child)]
    if label == 'else':
        if not nodes:
            raise ShakarRuntimeError("Malformed else clause")
        return None, nodes[0]
    cond_node = nodes[0] if nodes else None
    body_node = nodes[1] if len(nodes) > 1 else None
    if cond_node is None:
        raise ShakarRuntimeError("Malformed elif clause")
    if body_node is None:
        raise ShakarRuntimeError("Malformed clause body")
    return cond_node, body_node

def _coerce_loop_binder(node: Tree) -> dict[str, Any]:
    target = node
    if tree_label(target) == 'binderpattern' and target.children:
        target = target.children[0]
    if tree_label(target) == 'hoist':
        tok = target.children[0] if target.children else None
        if tok is None or not is_token_node(tok):
            raise ShakarRuntimeError("Malformed hoisted binder")
        pattern = Tree('pattern', [tok])
        return {'pattern': pattern, 'hoist': True}
    if tree_label(target) == 'pattern':
        return {'pattern': target, 'hoist': False}
    if is_token_node(target) and _token_kind(target) == 'IDENT':
        pattern = Tree('pattern', [target])
        return {'pattern': pattern, 'hoist': False}
    raise ShakarRuntimeError("Malformed binder pattern")

def _pattern_requires_object_pair(pattern: Tree) -> bool:
    if not is_tree_node(pattern):
        return False
    return any(
        tree_label(child) == 'pattern_list'
        and sum(1 for elem in tree_children(child) if tree_label(elem) == 'pattern') >= 2
        for child in tree_children(pattern)
    )

def _execute_loop_body(body_node: Any, env: Env) -> Any:
    label = tree_label(body_node) if is_tree_node(body_node) else None
    if label == 'inlinebody':
        return _eval_inline_body(body_node, env, allow_loop_control=True)
    if label == 'indentblock':
        return _eval_indent_block(body_node, env, allow_loop_control=True)
    return eval_node(body_node, env)

def _iter_indexed_entries(value: Any, binder_count: int) -> list[tuple[Any, list[Any]]]:
    if binder_count <= 0:
        raise ShakarRuntimeError("Indexed loop requires at least one binder")
    if binder_count > 2:
        raise ShakarRuntimeError("Indexed loop supports at most two binders")
    entries: list[tuple[Any, list[Any]]] = []
    match value:
        case ShkArray(items=items):
            for idx, item in enumerate(items):
                binders = [ShkNumber(float(idx))]
                if binder_count > 1:
                    binders.append(item)
                entries.append((item, binders[:binder_count]))
        case ShkString(value=s):
            for idx, ch in enumerate(s):
                char = ShkString(ch)
                binders = [ShkNumber(float(idx))]
                if binder_count > 1:
                    binders.append(char)
                entries.append((char, binders[:binder_count]))
        case ShkObject(slots=slots):
            for key, val in slots.items():
                binders = [ShkString(key)]
                if binder_count > 1:
                    binders.append(val)
                entries.append((val, binders[:binder_count]))
        case ShkSelector():
            values = selector_iter_values(value)
            for idx, sel in enumerate(values):
                binders = [ShkNumber(float(idx))]
                if binder_count > 1:
                    binders.append(sel)
                entries.append((sel, binders[:binder_count]))
        case _:
            raise ShakarTypeError(f"Cannot use indexed loop on {type(value).__name__}")
    return entries

def _iterable_values(value: Any) -> list[Any]:
    match value:
        case ShkNull():
            return []
        case ShkArray(items=items):
            return list(items)
        case ShkString(value=s):
            return [ShkString(ch) for ch in s]
        case ShkObject(slots=slots):
            return [ShkString(k) for k in slots.keys()]
        case ShkSelector():
            return selector_iter_values(value)
        case _:
            raise ShakarTypeError(f"Cannot iterate over {type(value).__name__}")
# ---------------- Comparison ----------------

def _eval_compare(children: List[Any], env: Env) -> Any:
    if not children:
        return ShkNull()

    if len(children) == 1:
        return eval_node(children[0], env)

    subject = eval_node(children[0], env)
    idx = 1
    joiner = 'and'
    last_comp: Optional[str] = None
    agg: Optional[bool] = None

    while idx < len(children):
        node = children[idx]
        token_kind = _token_kind(node)
        if token_kind == 'AND':
            joiner = 'and'
            idx += 1
            continue
        if token_kind == 'OR':
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
            rhs = eval_node(children[idx], env)
            idx += 1
        else:
            if last_comp is None:
                raise ShakarRuntimeError("Comparator required in comparison chain")
            comp = last_comp
            rhs = eval_node(node, env)
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

def _eval_logical(kind: str, children: List[Any], env: Env) -> Any:
    if not children:
        return ShkNull()
    normalized = 'and' if 'and' in kind else 'or'
    prev_dot = env.dot
    try:
        if normalized == 'and':
            last_val: Any = ShkBool(True)
            for child in children:
                val = eval_node(child, env)
                if _retargets_anchor(child):
                    env.dot = val
                last_val = val
                if not _is_truthy(val):
                    return val
            return last_val
        last_val: Any = ShkBool(False)
        for child in children:
            val = eval_node(child, env)
            if _retargets_anchor(child):
                env.dot = val
            last_val = val
            if _is_truthy(val):
                return val
        return last_val
    finally:
        env.dot = prev_dot

def _eval_nullish(children: List[Any], env: Env) -> Any:
    exprs = [child for child in children if not (is_token_node(child) and _token_kind(child) == 'NULLISH')]
    if not exprs:
        return ShkNull()
    current = eval_node(exprs[0], env)
    for expr in exprs[1:]:
        if not isinstance(current, ShkNull):
            return current
        current = eval_node(expr, env)
    return current

def _eval_nullsafe(children: List[Any], env: Env) -> Any:
    if not children:
        return ShkNull()
    expr = children[0]
    saved_dot = env.dot
    try:
        return _eval_chain_nullsafe(expr, env)
    finally:
        env.dot = saved_dot

def _eval_chain_nullsafe(node: Any, env: Env) -> Any:
    if not is_tree_node(node) or tree_label(node) != 'explicit_chain':
        return eval_node(node, env)
    children = tree_children(node)
    if not children:
        return ShkNull()
    head = children[0]
    current = eval_node(head, env)
    if isinstance(current, ShkNull):
        return ShkNull()
    for op in children[1:]:
        try:
            current = _apply_op(current, op, env)
        except (ShakarRuntimeError, ShakarTypeError) as err:
            if _nullsafe_recovers(err, current):
                return ShkNull()
            raise
        if isinstance(current, _RebindContext):
            current = current.value
        if isinstance(current, ShkNull):
            return ShkNull()
    return current

def _nullsafe_recovers(err: Exception, recv: Any) -> bool:
    if isinstance(recv, ShkNull):
        return True
    return isinstance(err, (ShakarKeyError, ShakarIndexError))

def _is_truthy(val: Any) -> bool:
    match val:
        case ShkBool(value=b):
            return b
        case ShkNull():
            return False
        case ShkNumber(value=num):
            return num != 0
        case ShkString(value=s):
            return bool(s)
        case ShkArray(items=items):
            return bool(items)
        case ShkObject(slots=slots):
            return bool(slots)
        case _:
            return True

def _retargets_anchor(node: Any) -> bool:
    if is_token_node(node):
        return _token_kind(node) not in {'SEMI', '_NL', 'INDENT', 'DEDENT'}
    if is_tree_node(node):
        return tree_label(node) not in {'implicit_chain', 'subject', 'group'}
    return True

# ---------------- Arithmetic ----------------

def _normalize_unary_op(op_node: Any, env: Env) -> Any:
    if tree_label(op_node) == 'unaryprefixop':
        if op_node.children:
            return _normalize_unary_op(op_node.children[0], env)
        src = getattr(env, 'source', None)
        meta = node_meta(op_node)
        if src is not None and meta is not None:
            return src[meta.start_pos:meta.end_pos]
        return ''
    return op_node

def _eval_unary(op_node: Any, rhs_node: Any, env: Env) -> Any:
    op_norm = _normalize_unary_op(op_node, env)
    op_value = op_norm.value if isinstance(op_norm, Token) else op_norm

    if op_value in ('++', '--'):
        context = _resolve_assignable_node(rhs_node, env)
        _, new_val = _apply_numeric_delta(context, 1 if op_value == '++' else -1)
        return new_val

    rhs = eval_node(rhs_node, env)

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

def _eval_infix(children: List[Any], env: Env, right_assoc_ops: set|None=None) -> Any:
    if not children:
        return ShkNull()

    if right_assoc_ops and _all_ops_in(children, right_assoc_ops):
        vals = [eval_node(children[i], env) for i in range(0, len(children), 2)]
        acc = vals[-1]
        for i in range(len(vals)-2, -1, -1):
            lhs, rhs = vals[i], acc
            _require_number(lhs); _require_number(rhs)
            acc = ShkNumber(lhs.value ** rhs.value)
        return acc

    it = iter(children)
    acc = eval_node(next(it), env)
    for x in it:
        op = _as_op(x)
        rhs = eval_node(next(it), env)
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

def _stringify(value: Any) -> str:
    if isinstance(value, ShkString):
        return value.value
    inner = getattr(value, 'value', value)
    return str(inner)

def _assert_source_snippet(node: Any, env: Env) -> str:
    src = getattr(env, 'source', None)
    if src is not None:
        start, end = _node_source_span(node)
        if start is not None and end is not None and 0 <= start < end <= len(src):
            snippet = src[start:end].strip()
            if snippet:
                return snippet
    rendered = _render_expr(node)
    return rendered if rendered else "<expr>"

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

def _render_expr(node: Any) -> str:
    if is_token_node(node):
        return node.value
    if not is_tree_node(node):
        return str(node)
    parts: List[str] = []
    for child in tree_children(node):
        rendered = _render_expr(child)
        if rendered:
            parts.append(rendered)
    return " ".join(parts)

def _node_source_span(node: Any) -> tuple[int | None, int | None]:
    meta = node_meta(node)
    start = getattr(meta, 'start_pos', None)
    end = getattr(meta, 'end_pos', None)
    if start is not None and end is not None:
        return start, end
    if is_tree_node(node):
        child_spans = [_node_source_span(child) for child in tree_children(node)]
        child_starts = [s for s, _ in child_spans if s is not None]
        child_ends = [e for _, e in child_spans if e is not None]
        if child_starts and child_ends:
            return min(child_starts), max(child_ends)
    return None, None

def _require_number(v: Any) -> None:
    if not isinstance(v, ShkNumber):
        raise ShakarTypeError("Expected number")

def _make_ident_context(name: str, env: Env) -> _RebindContext:
    """Produce a `_RebindContext` for identifier assignments so tail ops can persist."""
    value = env.get(name)
    def setter(new_value: Any) -> None:
        _assign_ident(name, new_value, env, create=False)
        env.dot = new_value
    return _RebindContext(value, setter)

def _current_function_env(env: Env) -> Env | None:
    """Walk parents to find the nearest function-call environment marker."""
    cur = env
    while cur is not None:
        if cur.is_function_env():
            return cur
        cur = getattr(cur, "parent", None)
    return None

def _token_number(t: Token, _: Env) -> ShkNumber:
    return ShkNumber(float(t.value))

def _token_string(t: Token, _: Env) -> ShkString:
    v = t.value
    if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
        v = v[1:-1]
    return ShkString(v)

def _resolve_assignable_node(node: Any, env: Env) -> Any:
    while is_tree_node(node) and tree_label(node) in {'primary', 'group', 'group_expr'} and len(node.children) == 1:
        node = node.children[0]

    if is_token_node(node) and _token_kind(node) == 'IDENT':
        return _make_ident_context(node.value, env)

    if is_tree_node(node):
        label = tree_label(node)
        if label == 'rebind_primary':
            ctx = eval_node(node, env)
            if isinstance(ctx, _RebindContext):
                return ctx
            raise ShakarRuntimeError("Rebind primary did not produce a context")
        if label == 'explicit_chain':
            if not node.children:
                raise ShakarRuntimeError("Malformed explicit chain")
            head = node.children[0]
            ops = list(node.children[1:])
            return _resolve_chain_assignment(head, ops, env)
        if label in {'expr', 'expr_nc'} and node.children:
            return _resolve_assignable_node(node.children[0], env)

    raise ShakarRuntimeError("Increment target must be assignable")

def _resolve_chain_assignment(head_node: Any, ops: List[Any], env: Env) -> Any:
    """Walk `a.b[0]` and return the final assignable context (object slot, index, etc.)."""
    if not ops:
        return _resolve_assignable_node(head_node, env)

    current = eval_node(head_node, env)
    if isinstance(current, _RebindContext):
        current = current.value

    for idx, op in enumerate(ops):
        is_last = idx == len(ops) - 1
        label = tree_label(op)

        if is_last and label in {'field', 'fieldsel'}:
            field_name = _expect_ident_token(op.children[0], "Field access")
            owner = current
            value = get_field_value(owner, field_name, env)
            def setter(new_value: Any, owner: Any=owner, field_name: str=field_name) -> None:
                set_field_value(owner, field_name, new_value, env, create=False)
                env.dot = new_value
            return _RebindContext(value, setter)

        if is_last and label == 'index':
            owner = current
            idx_value = _evaluate_index_operand(op, env)
            value = index_value(owner, idx_value, env)
            def setter(new_value: Any, owner: Any=owner, idx_value: Any=idx_value) -> None:
                set_index_value(owner, idx_value, new_value, env)
                env.dot = new_value
            return _RebindContext(value, setter)

        if is_last and label == 'fieldfan':
            return _build_fieldfan_context(current, op, env)

        current = _apply_op(current, op, env)
        if isinstance(current, _RebindContext):
            current = current.value

    raise ShakarRuntimeError("Increment target must end with a field or index")

def _apply_numeric_delta(ref: _RebindContext, delta: int) -> tuple[Any, Any]:
    current = ref.value
    _require_number(current)
    new_val = ShkNumber(current.value + delta)
    ref.setter(new_val)
    ref.value = new_val
    return current, new_val

def _build_fieldfan_context(owner: Any, fan_node: Tree, env: Env) -> _FanContext:
    fieldlist_node = child_by_label(fan_node, 'fieldlist')
    if fieldlist_node is None:
        raise ShakarRuntimeError("Malformed field fan")
    names = [tok.value for tok in tree_children(fieldlist_node) if is_token_node(tok) and _token_kind(tok) == 'IDENT']
    if not names:
        raise ShakarRuntimeError("Field fan requires at least one identifier")
    contexts: List[_RebindContext] = []
    for name in names:
        value = get_field_value(owner, name, env)
        def setter(new_value: Any, owner: Any=owner, field_name: str=name) -> None:
            set_field_value(owner, field_name, new_value, env, create=False)
            env.dot = new_value
        contexts.append(_RebindContext(value, setter))
    return _FanContext(contexts)

def _apply_fan_op(fan: _FanContext, op: Tree, env: Env) -> _FanContext:
    new_contexts: List[_RebindContext] = []
    new_values: List[Any] = []
    has_contexts = False
    has_values = False
    for ctx in fan.contexts:
        res = _apply_op(ctx, op, env)
        if isinstance(res, _FanContext):
            if res.contexts:
                new_contexts.extend(res.contexts)
                has_contexts = True
            else:
                new_values.extend(res.values)
                has_values = True
        elif isinstance(res, _RebindContext):
            new_contexts.append(res)
            has_contexts = True
        else:
            new_values.append(res)
            has_values = True
    if not fan.contexts and not new_contexts and not new_values:
        return fan
    if has_contexts and has_values:
        raise ShakarRuntimeError("Mixed field fan results not supported")
    if has_contexts:
        fan.contexts = new_contexts
        fan.update_from_contexts()
    else:
        fan.contexts = []
        fan.values = new_values
    return fan

def _extract_param_names(params_node: Any, context: str="parameter list") -> List[str]:
    if params_node is None:
        return []
    names: List[str] = []
    for p in tree_children(params_node):
        name = _ident_token_value(p)
        if name is not None:
            names.append(name)
            continue
        kind = _token_kind(p)
        if kind in {'COMMA'}:
            continue
        raise ShakarRuntimeError(f"Unsupported parameter node in {context}: {p}")
    return names

# ---------------- Chains ----------------

def _apply_op(recv: Any, op: Tree, env: Env) -> Any:
    """Apply one explicit-chain operation (call/index/member) to `recv`."""
    if isinstance(recv, _FanContext):
        return _apply_fan_op(recv, op, env)

    context = None
    if isinstance(recv, _RebindContext):
        context = recv
        recv = context.value

    d = op.data
    if d in {'field', 'fieldsel'}:
        field_name = _expect_ident_token(op.children[0], "Field access")
        result = get_field_value(recv, field_name, env)
    elif d == 'index':
        result = _apply_index_operation(recv, op, env)
    elif d == 'slicesel':
        result = _apply_slice(recv, op.children, env)
    elif d == 'fieldfan':
        return _build_fieldfan_context(recv, op, env)
    elif d == 'call':
        args = _eval_args_node(op.children[0] if op.children else None, env)
        result = _call_value(recv, args, env)
    elif d == 'method':
        method_name = _expect_ident_token(op.children[0], "Method call")
        args = _eval_args_node(op.children[1] if len(op.children)>1 else None, env)
        try:
            result = call_builtin_method(recv, method_name, args, env)
        except ShakarMethodNotFound:
            cal = get_field_value(recv, method_name, env)
            if isinstance(cal, BoundMethod):
                result = call_shkfn(cal.fn, args, subject=cal.subject, caller_env=env)
            elif isinstance(cal, ShkFn):
                result = call_shkfn(cal, args, subject=recv, caller_env=env)
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

def _eval_args_node(args_node: Any, env: Env) -> List[Any]:
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
        return [eval_node(n, env) for n in flatten(args_node)]
    if isinstance(args_node, list):  # deliberate: args_node can be pure python list from visitors
        res: List[Any] = []
        for n in args_node:
            res.extend(flatten(n))
        return [eval_node(n, env) for n in res]
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

def _apply_slice(recv: Any, arms: List[Any], env: Env) -> Any:
    def arm_to_py(node: Any) -> int | None:
        if tree_label(node) == 'emptyexpr':
            return None
        value = eval_node(node, env)
        if isinstance(value, ShkNumber):
            return int(value.value)
        return None
    start, stop, step = map(arm_to_py, arms)
    return slice_value(recv, start, stop, step)

# ---------------- Selectors ----------------

def _apply_index_operation(recv: Any, op: Tree, env: Env) -> Any:
    selectorlist = child_by_label(op, 'selectorlist')
    if selectorlist is None:
        expr_node = _index_expr_from_children(op.children)
        idx_val = eval_node(expr_node, env)
        return index_value(recv, idx_val, env)
    selectors = evaluate_selectorlist(selectorlist, env, eval_node)
    return apply_selectors_to_value(recv, selectors)

def _eval_group(n: Tree, env: Env) -> Any:
    child = n.children[0] if n.children else None
    if child is None:
        return ShkNull()
    saved = env.dot
    try:
        return eval_node(child, env)
    finally:
        env.dot = saved

def _eval_ternary(n: Tree, env: Env) -> Any:
    if len(n.children) != 3:
        raise ShakarRuntimeError("Malformed ternary expression")
    cond_node, true_node, false_node = n.children
    cond_val = eval_node(cond_node, env)
    if _is_truthy(cond_val):
        return eval_node(true_node, env)
    return eval_node(false_node, env)

def _eval_rebind_primary(n: Tree, env: Env) -> Any:
    if not n.children:
        raise ShakarRuntimeError("Missing identifier for rebind")
    target = n.children[0]
    if not is_token_node(target) or _token_kind(target) != 'IDENT':
        raise ShakarRuntimeError("Rebind target must be identifier")
    name = target.value
    value = env.get(name)

    def setter(new_value: Any) -> None:
        _assign_ident(name, new_value, env, create=False)
        env.dot = new_value

    env.dot = value
    return _RebindContext(value, setter)

def _eval_optional_expr(node: Any, env: Env) -> Any:
    if node is None:
        return None
    return eval_node(node, env)

def _call_value(cal: Any, args: List[Any], env: Env) -> Any:
    match cal:
        case BoundMethod(fn=fn, subject=subject):
            return call_shkfn(fn, args, subject=subject, caller_env=env)
        case BuiltinMethod(name=name, subject=subject):
            return call_builtin_method(subject, name, args, env)
        case StdlibFunction(fn=fn, arity=arity):
            if arity is not None and len(args) != arity:
                raise ShakarArityError(f"Function expects {arity} args; got {len(args)}")
            return fn(env, args)
        case ShkFn():
            return call_shkfn(cal, args, subject=None, caller_env=env)
        case _:
            raise ShakarTypeError(f"Cannot call value of type {type(cal).__name__}")

# ---------------- Objects ----------------

def _eval_object(n: Tree, env: Env) -> ShkObject:
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
                    method_fn = ShkFn(params=params, body=val_node, env=Env(parent=env))
                    slots[name] = method_fn
                    return
                key = _eval_key(key_node, env)
                val = eval_node(val_node, env)
                slots[str(key)] = val
            case 'obj_get':
                name_tok, body = item.children
                if name_tok is None:
                    raise ShakarRuntimeError("Getter missing name")
                key = name_tok.value
                getter_fn = ShkFn(params=None, body=body, env=Env(parent=env))
                _install_descriptor(key, getter=getter_fn)
            case 'obj_set':
                name_tok, param_tok, body = item.children
                if name_tok is None or param_tok is None:
                    raise ShakarRuntimeError("Setter missing name or parameter")
                key = name_tok.value
                setter_fn = ShkFn(params=[param_tok.value], body=body, env=Env(parent=env))
                _install_descriptor(key, setter=setter_fn)
            case 'obj_method':
                name_tok, params_node, body = item.children
                if name_tok is None:
                    raise ShakarRuntimeError("Method missing name")
                param_names = _extract_params(params_node)
                method_fn = ShkFn(params=param_names, body=body, env=Env(parent=env))
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

def _eval_key(k: Any, env: Env) -> Any:
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
                v = eval_node(k.children[0], env)
                return v.value if isinstance(v, ShkString) else v

    if is_token_node(k) and _token_kind(k) in ('IDENT','STRING'):
        return k.value.strip('"').strip("'")

    return eval_node(k, env)

def _eval_amp_lambda(n: Tree, env: Env) -> ShkFn:
    if len(n.children) == 1:
        return ShkFn(params=None, body=n.children[0], env=Env(parent=env, dot=None))

    if len(n.children) == 2:
        params_node, body = n.children
        params = _extract_param_names(params_node, context="amp_lambda")
        return ShkFn(params=params, body=body, env=Env(parent=env, dot=None))

    raise ShakarRuntimeError("amp_lambda malformed")

@contextmanager
def _temporary_subject(env: Env, dot: Any) -> Iterable[None]:
    """Temporarily bind `env.dot` while evaluating nested subjectful constructs."""
    prev = env.dot
    env.dot = dot
    try:
        yield
    finally:
        env.dot = prev

_NODE_DISPATCH: dict[str, Callable[[Tree, Env], Any]] = {
    'listcomp': _eval_listcomp,
    'setcomp': _eval_setcomp,
    'setliteral': _eval_setliteral,
    'dictcomp': _eval_dictcomp,
    'selectorliteral': lambda n, env: eval_selectorliteral(n, env, eval_node),
    'group': _eval_group,
    'ternary': _eval_ternary,
    'rebind_primary': _eval_rebind_primary,
    'amp_lambda': _eval_amp_lambda,
    'compare': lambda n, env: _eval_compare(n.children, env),
    'compare_nc': lambda n, env: _eval_compare(n.children, env),
    'nullish': lambda n, env: _eval_nullish(n.children, env),
    'nullsafe': lambda n, env: _eval_nullsafe(n.children, env),
    'breakstmt': lambda _, env: _eval_break_stmt(env),
    'continuestmt': lambda _, env: _eval_continue_stmt(env),
    'ifstmt': _eval_if_stmt,
    'forin': _eval_for_in,
    'forsubject': _eval_for_subject,
    'forindexed': _eval_for_indexed,
    'formap1': lambda n, env: _eval_for_indexed(n.children[0] if n.children else None, env),
    'formap2': _eval_for_map2,
    'inlinebody': _eval_inline_body,
    'indentblock': _eval_indent_block,
    'onelineguard': lambda n, env: _eval_oneline_guard(n.children, env),
}

_TOKEN_DISPATCH: dict[str, Callable[[Token, Env], Any]] = {
    'NUMBER': _token_number,
    'STRING': _token_string,
    'TRUE': lambda _, __: ShkBool(True),
    'FALSE': lambda _, __: ShkBool(False),
}
