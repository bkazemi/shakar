from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Coroutine, Iterable, List, TypeVar

from ..runtime import Frame, ShkNull, ShkObject, ShkString, ShkValue, ShakarRuntimeError
from ..tree import TreeNode, child_by_label, is_token, is_tree, tree_children, tree_label
from .blocks import eval_body_node, run_body_with_subject, temporary_bindings
from .common import token_kind as _token_kind

EvalFunc = Callable[[Any, Frame], Any]
_T = TypeVar("_T")

def _wrap_awaitable(value: Any) -> Coroutine[Any, Any, Any]:
    if inspect.isawaitable(value):
        async def _forward() -> Any:
            return await value
        return _forward()

    async def _immediate() -> Any:
        return value

    return _immediate()

def _run_asyncio(coro: Coroutine[Any, Any, _T]) -> _T:
    try:
        asyncio.get_running_loop()
    except RuntimeError: # no active event loop, ok to run
        return asyncio.run(coro)

    raise ShakarRuntimeError("await constructs cannot run inside an active event loop")

async def _await_any_async(entries: list[tuple[dict[str, Any], Any]]) -> tuple[dict[str, Any], Any]:
    tasks: dict[asyncio.Task[Any], dict[str, Any]] = {}

    for arm, value in entries:
        task: asyncio.Task[Any] = asyncio.create_task(_wrap_awaitable(value))
        tasks[task] = arm

    errors: list[Exception] = []

    pending: set[asyncio.Task[Any]] = set(tasks.keys())
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            arm = tasks.pop(task)

            try:
                result = task.result()
            except Exception as exc:
                errors.append(exc)
                continue

            for other in pending:
                other.cancel()

            await asyncio.gather(*pending, return_exceptions=True)
            return arm, result

    if errors:
        raise errors[-1]

    raise ShakarRuntimeError("await[any] arms did not produce a value")

async def _await_all_async(entries: list[tuple[dict[str, Any], Any]]) -> list[tuple[dict[str, Any], Any]]:
    coros = [_wrap_awaitable(value) for _, value in entries]
    results = await asyncio.gather(*coros)

    return [(entries[i][0], results[i]) for i in range(len(entries))]

def await_any_entries(entries: list[tuple[dict[str, Any], Any]]) -> tuple[dict[str, Any], Any]:
    return _run_asyncio(_await_any_async(entries))

def await_all_entries(entries: list[tuple[dict[str, Any], Any]]) -> list[tuple[dict[str, Any], Any]]:
    return _run_asyncio(_await_all_async(entries))

def resolve_await_result(value: Any) -> Any:
    if inspect.isawaitable(value):
        return _run_asyncio(_wrap_awaitable(value))

    return value

def _collect_await_arms(list_node: TreeNode, prefix: str) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []

    for idx, arm_node in enumerate(tree_children(list_node)):
        children = tree_children(arm_node)
        label = None
        expr_node = None
        body_node = None
        pos = 0

        if children and is_token(children[0]) and _token_kind(children[0]) == 'IDENT':
            label = children[0].value
            pos = 1

        if pos >= len(children):
            raise ShakarRuntimeError("Malformed await arm")

        expr_node = children[pos]
        pos += 1

        if pos < len(children):
            candidate = children[pos]

            if is_tree(candidate) and tree_label(candidate) in {'inlinebody', 'indentblock'}:
                body_node = candidate

        arms.append({
            'label': label or f"{prefix}{idx}",
            'expr': expr_node,
            'body': body_node,
        })

    return arms

def _extract_trailing_body(children: List[Any]) -> Any | None:
    seen_rpar = False

    for child in children:
        if is_token(child):
            if _token_kind(child) == 'RPAR':
                seen_rpar = True
            continue

        if seen_rpar and tree_label(child) in {'inlinebody', 'indentblock'}:
            return child

    return None

def _await_winner_object(label: str, value: Any) -> ShkObject:
    return ShkObject({'winner': ShkString(label), 'value': value})

def eval_await_value(n: TreeNode, frame: Frame, eval_func: EvalFunc) -> Any:
    for child in tree_children(n):
        if not is_token(child):
            value = eval_func(child, frame)
            return resolve_await_result(value)

    raise ShakarRuntimeError("Malformed await expression")

def eval_await_stmt(n: TreeNode, frame: Frame, eval_func: EvalFunc) -> Any:
    expr_node = None
    body_node = None

    for child in tree_children(n):
        if is_tree(child) and tree_label(child) in {'inlinebody', 'indentblock'}:
            body_node = child
        elif not is_token(child) and expr_node is None:
            expr_node = child

    if expr_node is None or body_node is None:
        raise ShakarRuntimeError("Malformed await statement")

    value = resolve_await_result(eval_func(expr_node, frame))
    run_body_with_subject(body_node, frame, value, eval_func)

    return ShkNull()

def eval_await_any_call(n: TreeNode, frame: Frame, eval_func: EvalFunc) -> Any:
    arms_node = child_by_label(n, 'anyarmlist')
    if arms_node is None:
        raise ShakarRuntimeError("await[any] missing arm list")

    arms = _collect_await_arms(arms_node, "arm")
    if not arms:
        raise ShakarRuntimeError("await[any] requires at least one arm")

    trailing_body = _extract_trailing_body(tree_children(n))
    per_arm_bodies = [arm for arm in arms if arm['body'] is not None]

    if trailing_body is not None and per_arm_bodies:
        raise ShakarRuntimeError("await[any] cannot mix per-arm bodies with a trailing body")

    entries: list[tuple[dict[str, Any], Any]] = []
    errors: list[ShakarRuntimeError] = []

    for arm in arms:
        try:
            value = eval_func(arm['expr'], frame)
        except ShakarRuntimeError as err:
            errors.append(err)
            continue
        entries.append((arm, value))

    if not entries:
        if errors:
            raise errors[-1]
        raise ShakarRuntimeError("await[any] arms did not produce a value")

    winner_arm, winner_value = await_any_entries(entries)

    if winner_arm['body'] is not None:
        return run_body_with_subject(winner_arm['body'], frame, winner_value, eval_func)

    if trailing_body is not None:
        bindings = {'winner': ShkString(winner_arm['label'])}
        return run_body_with_subject(trailing_body, frame, winner_value, eval_func, bindings)

    return _await_winner_object(winner_arm['label'], winner_value)

def eval_await_all_call(n: TreeNode, frame: Frame, eval_func: EvalFunc) -> Any:
    arms_node = child_by_label(n, 'allarmlist')
    if arms_node is None:
        raise ShakarRuntimeError("await[all] missing arm list")

    arms = _collect_await_arms(arms_node, "arm")
    if not arms:
        raise ShakarRuntimeError("await[all] requires at least one arm")

    if any(arm['body'] is not None for arm in arms):
        raise ShakarRuntimeError("await[all] does not support per-arm bodies")

    entries: list[tuple[dict[str, Any], Any]] = []

    for arm in arms:
        value = eval_func(arm['expr'], frame)
        entries.append((arm, value))

    resolved = await_all_entries(entries)
    results_object: dict[str, ShkValue] = {arm['label']: value for arm, value in resolved}
    trailing_body = _extract_trailing_body(tree_children(n))

    if trailing_body is not None:
        with temporary_bindings(frame, results_object):
            return eval_body_node(trailing_body, frame, eval_func)
    return ShkObject(results_object)
