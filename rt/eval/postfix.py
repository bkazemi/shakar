from __future__ import annotations

from typing import Any, Callable, Iterable, List, Sequence

from shakar_runtime import Env, ShkNull, ShakarRuntimeError
from shakar_tree import find_tree_by_label, is_token_node, tree_children


def define_new_ident(name: str, value: Any, env: Env) -> Any:
    """Introduce a new binding in the current scope; error if it already exists."""
    vars_dict = getattr(env, "vars", None)
    if vars_dict is not None and name in vars_dict:
        raise ShakarRuntimeError(f"Name '{name}' already defined in this scope")
    env.define(name, value)
    return value


def _walrus_target_name(node: Any) -> str:
    children = tree_children(node)
    if not children:
        raise ShakarRuntimeError("Malformed walrus expression")
    target = children[0]
    if is_token_node(target):
        return target.value
    if isinstance(target, str):
        return target
    raise ShakarRuntimeError("Unsupported walrus target")


def _split_postfix_children(children: Sequence[Any], keyword_tokens: Iterable[str]) -> tuple[Any, Any]:
    keywords = set(keyword_tokens)
    semantic: List[Any] = []
    for ch in children:
        if is_token_node(ch) and ch.type in keywords:
            continue
        semantic.append(ch)
    if len(semantic) != 2:
        raise ShakarRuntimeError("Malformed postfix statement")
    return semantic[0], semantic[1]


def eval_postfix_if(
    children: Sequence[Any],
    env: Env,
    *,
    eval_func: Callable[[Any, Env], Any],
    truthy_fn: Callable[[Any], bool],
) -> Any:
    stmt_node, cond_node = _split_postfix_children(children, {"IF"})
    return _eval_postfix_guard(stmt_node, cond_node, env, eval_func, truthy_fn, run_on_truthy=True)


def eval_postfix_unless(
    children: Sequence[Any],
    env: Env,
    *,
    eval_func: Callable[[Any, Env], Any],
    truthy_fn: Callable[[Any], bool],
) -> Any:
    stmt_node, cond_node = _split_postfix_children(children, {"UNLESS"})
    return _eval_postfix_guard(stmt_node, cond_node, env, eval_func, truthy_fn, run_on_truthy=False)


def _eval_postfix_guard(
    stmt_node: Any,
    cond_node: Any,
    env: Env,
    eval_func: Callable[[Any, Env], Any],
    truthy_fn: Callable[[Any], bool],
    *,
    run_on_truthy: bool,
) -> Any:
    walrus_name = None
    walrus_node = find_tree_by_label(stmt_node, {"walrus", "walrus_nc"})
    if walrus_node is not None:
        walrus_name = _walrus_target_name(walrus_node)
    cond_val = eval_func(cond_node, env)
    cond_truthy = truthy_fn(cond_val)
    should_run = cond_truthy if run_on_truthy else not cond_truthy
    if should_run:
        return eval_func(stmt_node, env)
    if walrus_name is not None:
        define_new_ident(walrus_name, ShkNull(), env)
    return ShkNull()
