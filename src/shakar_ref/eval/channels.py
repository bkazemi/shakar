from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ..runtime import (
    CancelToken,
    Frame,
    ShkArray,
    ShkBool,
    ShkChannel,
    ShkResultChannel,
    ShkDuration,
    ShkFan,
    ShkNull,
    ShkNumber,
    ShkObject,
    ShkSelector,
    ShkDecorator,
    ShkFn,
    ShkValue,
    BoundMethod,
    BuiltinMethod,
    DecoratorContinuation,
    StdlibFunction,
    ShakarAllChannelsClosed,
    ShakarCancelledError,
    ShakarChannelClosedEmpty,
    ShakarRuntimeError,
    ShakarTypeError,
)
from ..tree import Node, Tree, is_tree, tree_children, tree_label
from .blocks import eval_body_node
from .chains import call_value
from .common import expect_ident_token
from .helpers import check_cancel
from .postfix import define_new_ident
from .selector import selector_iter_values

EvalFunc = Callable[[Node, Frame], ShkValue]


@dataclass
class _RecvCase:
    channel: ShkChannel
    binder: Optional[str]
    body: Tree


@dataclass
class _SendCase:
    channel: ShkChannel
    value: ShkValue
    body: Tree


@dataclass
class _TimeoutCase:
    deadline: float
    body: Tree


def _coerce_channel(value: ShkValue) -> ShkChannel:
    if isinstance(value, ShkNull):
        raise ShakarRuntimeError("Send/receive on nil channel")
    if not isinstance(value, ShkChannel):
        raise ShakarTypeError("Expected channel")
    return value


def _duration_to_seconds(value: ShkValue) -> float:
    if isinstance(value, ShkDuration):
        return max(0.0, float(value.nanos) / 1_000_000_000.0)
    if isinstance(value, ShkNumber):
        return max(0.0, float(value.value) / 1000.0)
    raise ShakarTypeError("timeout expects duration or number")


def _spawn_iterable_values(value: ShkValue) -> Optional[list[ShkValue]]:
    # Spawn-iterable is intentionally limited to concrete Shakar iterables.
    if isinstance(value, ShkArray):
        return list(value.items)
    if isinstance(value, ShkFan):
        return list(value.items)
    if isinstance(value, ShkSelector):
        return list(selector_iter_values(value))
    return None


def _is_callable_value(value: ShkValue) -> bool:
    return isinstance(
        value,
        (
            ShkFn,
            BoundMethod,
            BuiltinMethod,
            StdlibFunction,
            DecoratorContinuation,
            ShkDecorator,
        ),
    )


def _spawn_callable(frame: Frame, value: ShkValue, eval_func: EvalFunc) -> ShkChannel:
    def thunk(spawn_frame: Frame) -> ShkValue:
        return call_value(value, [], spawn_frame, eval_func)

    return _spawn_task(frame, thunk)


def eval_recv_expr(n: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    if not n.children:
        raise ShakarRuntimeError("Malformed receive expression")
    chan_val = eval_func(n.children[0], frame)
    chan = _coerce_channel(chan_val)
    return chan.recv_value(cancel_token=frame.cancel_token)


def eval_send_expr(n: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    if len(n.children) != 2:
        raise ShakarRuntimeError("Malformed send expression")
    value = eval_func(n.children[0], frame)
    chan_val = eval_func(n.children[1], frame)
    chan = _coerce_channel(chan_val)
    sent = chan.send_with_cancel(value, frame.cancel_token)
    return ShkBool(sent)


def _spawn_frame(parent: Frame, token: CancelToken) -> Frame:
    return Frame(
        parent=parent,
        dot=parent.dot,
        emit_target=parent.emit_target,
        cancel_token=token,
        source=parent.source,
        source_path=parent.source_path,
    )


def _spawn_task(frame: Frame, thunk: Callable[[Frame], ShkValue]) -> ShkResultChannel:
    cancel_token = CancelToken()
    result_ch = ShkResultChannel(cancel_token)

    def runner() -> None:
        spawn_frame = _spawn_frame(frame, cancel_token)
        try:
            result = thunk(spawn_frame)
            check_cancel(spawn_frame)
            if cancel_token.cancelled():
                return
            result_ch.send_result(result)
        except ShakarCancelledError:
            try:
                result_ch.close()
            except ShakarRuntimeError:
                pass
        except Exception as exc:
            result_ch.send_error(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return result_ch


def eval_spawn_expr(n: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    if not n.children:
        raise ShakarRuntimeError("Malformed spawn expression")
    expr = n.children[0]

    def thunk(spawn_frame: Frame) -> ShkValue:
        if is_tree(expr) and tree_label(expr) in {"inlinebody", "indentblock"}:
            return eval_body_node(expr, spawn_frame, eval_func)
        return eval_func(expr, spawn_frame)

    expr_node = _unwrap_expr_node(expr)
    if is_tree(expr_node) and tree_label(expr_node) in {"inlinebody", "indentblock"}:
        return _spawn_task(frame, thunk)
    if _explicit_call_in_node(expr):
        return _spawn_task(frame, thunk)

    # Non-call expressions are evaluated in the caller so we can detect iterable
    # spawn (arrays/fans/selectors). Non-iterables keep the old behavior: a task
    # that just returns the computed value.
    value = eval_func(expr, frame)
    iterable = _spawn_iterable_values(value)
    if iterable is None:
        return _spawn_task(frame, lambda _spawn_frame, v=value: v)

    channels: list[ShkChannel] = []
    for item in iterable:
        if isinstance(item, ShkChannel):
            raise ShakarTypeError("spawn iterable cannot contain channels")
        if not _is_callable_value(item):
            raise ShakarTypeError("spawn iterable expects callable elements")
        channels.append(_spawn_callable(frame, item, eval_func))

    if isinstance(value, ShkFan):
        return ShkFan(channels)
    return ShkArray(channels)


def eval_wait_any_block(n: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    recv_cases: list[_RecvCase] = []
    send_cases: list[_SendCase] = []
    timeout_case: Optional[_TimeoutCase] = None
    default_body: Optional[Tree] = None

    for arm in tree_children(n):
        label = tree_label(arm)
        children = list(tree_children(arm))

        if label == "waitany_default":
            if default_body is not None:
                raise ShakarRuntimeError("wait[any] has multiple default arms")
            if not children:
                raise ShakarRuntimeError("wait[any] default missing body")
            default_body = children[0]
            continue

        if label == "waitany_timeout":
            if len(children) != 2:
                raise ShakarRuntimeError("wait[any] timeout missing body")
            timeout_expr, body = children
            duration = eval_func(timeout_expr, frame)
            deadline = time.monotonic() + _duration_to_seconds(duration)
            timeout_case = _TimeoutCase(deadline=deadline, body=body)
            continue

        if label != "waitany_arm" or len(children) != 2:
            raise ShakarRuntimeError("Malformed wait[any] arm")

        head, body = children
        head = _unwrap_expr_node(head)
        if is_tree(head) and tree_label(head) == "send":
            value_expr, chan_expr = tree_children(head)
            value = eval_func(value_expr, frame)
            chan = _coerce_channel(eval_func(chan_expr, frame))
            send_cases.append(_SendCase(channel=chan, value=value, body=body))
            continue

        binder = None
        recv_expr = head
        if is_tree(head) and tree_label(head) == "walrus":
            name_node, value_node = tree_children(head)
            binder = expect_ident_token(name_node, "wait[any] walrus target")
            recv_expr = value_node

        recv_expr = _unwrap_expr_node(recv_expr)
        if not (is_tree(recv_expr) and tree_label(recv_expr) == "recv"):
            raise ShakarRuntimeError("wait[any] arm must be send or receive")

        recv_children = tree_children(recv_expr)
        if not recv_children:
            raise ShakarRuntimeError("wait[any] receive missing channel")
        chan = _coerce_channel(eval_func(recv_children[0], frame))
        recv_cases.append(_RecvCase(channel=chan, binder=binder, body=body))

    if not (recv_cases or send_cases or timeout_case or default_body):
        raise ShakarRuntimeError("wait[any] requires at least one arm")

    while True:
        check_cancel(frame)
        closed = 0

        cases = [("recv", c) for c in recv_cases] + [("send", c) for c in send_cases]
        random.shuffle(cases)

        for kind, case in cases:
            if kind == "recv":
                status, value = case.channel.try_recv_value()
                if status == "ready":
                    if case.binder is not None:
                        define_new_ident(case.binder, value, frame)
                    return eval_body_node(case.body, frame, eval_func)
                if status == "cancelled":
                    raise ShakarCancelledError("Spawn task cancelled")
                if status == "closed":
                    closed += 1
                continue

            sent, closed_flag = case.channel.try_send(case.value)
            if sent:
                return eval_body_node(case.body, frame, eval_func)
            if closed_flag:
                closed += 1

        if default_body is not None:
            return eval_body_node(default_body, frame, eval_func)

        if timeout_case is not None:
            now = time.monotonic()
            if now >= timeout_case.deadline:
                return eval_body_node(timeout_case.body, frame, eval_func)

        # Closed channels cannot reopen, so if all cases are closed and there's no
        # timeout to wait for, we can fail immediately.
        if cases and closed == len(cases) and timeout_case is None:
            raise ShakarAllChannelsClosed("All channels closed")

        event = threading.Event()
        for kind, case in cases:
            case.channel.register_waiter(event)
        try:
            timeout = None
            if timeout_case is not None:
                timeout = max(0.0, timeout_case.deadline - time.monotonic())
            _wait_event_with_cancel(event, timeout, frame)
        finally:
            for kind, case in cases:
                case.channel.unregister_waiter(event)


def _explicit_call_in_node(node: Node) -> bool:
    node = _unwrap_expr_node(node)
    if not is_tree(node):
        return False
    if tree_label(node) != "explicit_chain":
        return False
    for op in tree_children(node)[1:]:
        if not is_tree(op):
            continue
        if tree_label(op) in {"call", "method", "lambdacall1", "lambdacalln"}:
            return True
    return False


def _unwrap_expr_node(node: Node) -> Node:
    cur = node
    while is_tree(cur) and tree_label(cur) in {"expr", "group", "group_expr"}:
        children = tree_children(cur)
        if len(children) != 1:
            break
        cur = children[0]
    return cur


def _maybe_call_value(
    node: Node, value: ShkValue, frame: Frame, eval_func: EvalFunc
) -> ShkValue:
    if _explicit_call_in_node(node):
        return value
    if isinstance(value, (ShkChannel, ShkNull)):
        return value
    if isinstance(
        value,
        (ShkFn, BoundMethod, BuiltinMethod, StdlibFunction, DecoratorContinuation),
    ):
        return call_value(value, [], frame, eval_func)
    return value


def _wait_for_any_channel(
    pending: list[tuple[int, ShkChannel]],
    *,
    frame: Frame,
    allow_closed_skip: bool,
) -> tuple[int, ShkValue]:
    while True:
        check_cancel(frame)
        random.shuffle(pending)

        for idx, chan in pending:
            status, value = chan.try_recv_value()
            if status == "ready":
                return idx, value
            if status == "cancelled":
                raise ShakarCancelledError("Spawn task cancelled")
            if status == "closed":
                if allow_closed_skip:
                    continue
                raise ShakarChannelClosedEmpty("Channel closed without value")

        if not pending:
            raise ShakarAllChannelsClosed("All channels closed")

        event = threading.Event()
        for _, chan in pending:
            chan.register_waiter(event)
        try:
            _wait_event_with_cancel(event, None, frame)
        finally:
            for _, chan in pending:
                chan.unregister_waiter(event)


def _wait_event_with_cancel(
    event: threading.Event, timeout: Optional[float], frame: Frame
) -> None:
    if frame.cancel_token is None:
        event.wait(timeout=timeout)
        return

    if timeout is None:
        while True:
            event.wait(timeout=0.05)
            if event.is_set():
                return
            check_cancel(frame)
    else:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            event.wait(timeout=min(0.05, remaining))
            if event.is_set():
                return
            check_cancel(frame)


def _cancel_pending(pending: list[ShkChannel]) -> None:
    for chan in pending:
        if chan.is_result_channel():
            try:
                chan.close()
            except ShakarRuntimeError:
                continue


def eval_wait_all_block(n: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    arms = []
    for arm in tree_children(n):
        if tree_label(arm) != "waitallarm":
            raise ShakarRuntimeError("Malformed wait[all] arm")
        label_node, expr = tree_children(arm)
        name = expect_ident_token(label_node, "wait[all] label")
        arms.append((name, expr))

    if not arms:
        raise ShakarRuntimeError("wait[all] requires at least one arm")

    def _spawn_arm(node: Node) -> ShkChannel:
        def thunk(spawn_frame: Frame) -> ShkValue:
            return _maybe_call_value(
                node, eval_func(node, spawn_frame), spawn_frame, eval_func
            )

        return _spawn_task(frame, thunk)

    channels: list[ShkChannel] = []
    for _, expr in arms:
        channels.append(_spawn_arm(expr))

    pending = list(enumerate(channels))
    results: dict[str, ShkValue] = {}
    try:
        while pending:
            idx, value = _wait_for_any_channel(
                pending, frame=frame, allow_closed_skip=False
            )
            name, _ = arms[idx]
            results[name] = value
            pending = [(i, ch) for i, ch in pending if i != idx]
    except Exception as exc:
        _cancel_pending([ch for _, ch in pending])
        raise exc

    return ShkObject(results)


def eval_wait_group_block(n: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    arms = []
    for arm in tree_children(n):
        if tree_label(arm) != "waitgrouparm":
            raise ShakarRuntimeError("Malformed wait[group] arm")
        children = list(tree_children(arm))
        if len(children) != 1:
            raise ShakarRuntimeError("Malformed wait[group] arm")
        arms.append(children[0])

    if not arms:
        raise ShakarRuntimeError("wait[group] requires at least one arm")

    def _spawn_arm(node: Node) -> ShkChannel:
        def thunk(spawn_frame: Frame) -> ShkValue:
            return _maybe_call_value(
                node, eval_func(node, spawn_frame), spawn_frame, eval_func
            )

        return _spawn_task(frame, thunk)

    channels: list[ShkChannel] = []
    for expr in arms:
        channels.append(_spawn_arm(expr))

    pending = list(enumerate(channels))
    try:
        while pending:
            idx, _ = _wait_for_any_channel(
                pending, frame=frame, allow_closed_skip=False
            )
            pending = [(i, ch) for i, ch in pending if i != idx]
    except Exception as exc:
        _cancel_pending([ch for _, ch in pending])
        raise exc

    return ShkNull()


def eval_wait_all_call(n: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    if not n.children:
        raise ShakarRuntimeError("wait[all](tasks) missing argument")
    value = eval_func(n.children[0], frame)
    if not isinstance(value, ShkArray):
        raise ShakarTypeError("wait[all](tasks) expects an array of channels")
    channels: list[ShkChannel] = []
    for item in value.items:
        if not isinstance(item, ShkChannel):
            raise ShakarTypeError("wait[all](tasks) expects channel values")
        channels.append(item)

    pending = list(enumerate(channels))
    results: list[ShkValue] = [ShkNull()] * len(channels)
    try:
        while pending:
            idx, val = _wait_for_any_channel(
                pending, frame=frame, allow_closed_skip=False
            )
            results[idx] = val
            pending = [(i, ch) for i, ch in pending if i != idx]
    except Exception as exc:
        _cancel_pending([ch for _, ch in pending])
        raise exc

    return ShkArray(results)


def eval_wait_group_call(n: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    if not n.children:
        raise ShakarRuntimeError("wait[group](tasks) missing argument")
    value = eval_func(n.children[0], frame)
    if not isinstance(value, ShkArray):
        raise ShakarTypeError("wait[group](tasks) expects an array of channels")
    channels: list[ShkChannel] = []
    for item in value.items:
        if not isinstance(item, ShkChannel):
            raise ShakarTypeError("wait[group](tasks) expects channel values")
        channels.append(item)

    pending = list(enumerate(channels))
    try:
        while pending:
            idx, _ = _wait_for_any_channel(
                pending, frame=frame, allow_closed_skip=False
            )
            pending = [(i, ch) for i, ch in pending if i != idx]
    except Exception as exc:
        _cancel_pending([ch for _, ch in pending])
        raise exc

    return ShkNull()
