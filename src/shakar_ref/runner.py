from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from .parser_rd import parse_source, ParseError
from .lexer_rd import LexError
from .ast_transforms import Prune, looks_like_offside
from .lower import lower
from .evaluator import eval_expr
from .runtime import ShkValue
from .runtime import Frame, init_stdlib
from .types import ShkNil, ShakarRuntimeError
from .utils import debug_py_trace_enabled
from .tree import Tree

_STMT_LABELS = frozenset(
    {
        "walrus",
        "destructure_walrus",
        "assignstmt",
        "let",
        "compound_assign",
        "fndef",
        "decorator_def",
        "hook",
        "deferstmt",
        "assert",
        "bind",
        "destructure",
        "returnstmt",
        "returnif",
        "throwstmt",
        "postfixif",
        "postfixunless",
        "ifstmt",
        "whilestmt",
        "breakstmt",
        "continuestmt",
        "forin",
        "forsubject",
        "forindexed",
        "formap1",
        "formap2",
        "waitanyblock",
        "waitallblock",
        "waitgroupblock",
        "waitallcall",
        "waitgroupcall",
        "usingstmt",
        "callstmt",
        "catchstmt",
        "import_stmt",
        "import_destructure",
        "import_mixin",
    }
)


def _last_is_stmt(ast: object) -> bool:
    """Check if the last meaningful node in the AST is a statement."""
    if isinstance(ast, Tree):
        label = getattr(ast, "data", None)
        if label in ("stmtlist", "start_noindent", "start_indented"):
            # Find last tree child (skip tokens like SEMI/NEWLINE)
            for child in reversed(ast.children):
                if isinstance(child, Tree):
                    return _last_is_stmt(child)

            return False

        return label in _STMT_LABELS

    return False


def _parse_and_lower(src: str, use_indenter: Optional[bool] = None):
    if use_indenter is None:
        preferred = looks_like_offside(src)
        attempts = [preferred, not preferred]
    else:
        attempts = [use_indenter]

    last_error: Optional[Exception] = None
    tree = None

    for flag in attempts:
        try:
            tree = parse_source(src, use_indenter=flag)
            break
        except (ParseError, LexError) as exc:
            last_error = exc

    if tree is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Parser failed without producing a parse tree")

    ast = Prune().transform(tree)
    ast2 = lower(ast)

    # Unwrap start_* roots only when they contain a single child; otherwise keep
    # them so statement lists execute in order during evaluation.
    d = getattr(ast2, "data", None)
    if d in ("start_noindent", "start_indented"):
        children = getattr(ast2, "children", None)
        if children and len(children) == 1:
            ast2 = children[0]

    return ast2


def run(
    src: str,
    grammar_path: Optional[str] = None,
    use_indenter: Optional[bool] = None,
    grammar_variant: str = "default",
    source_path: Optional[str] = None,
) -> ShkValue:  # grammar_path/variant kept for CLI compatibility; ignored
    init_stdlib()
    ast2 = _parse_and_lower(src, use_indenter=use_indenter)
    return eval_expr(ast2, Frame(source=src, source_path=source_path), source=src)


def run_with_env(
    src: str,
    source_path: Optional[str] = None,
) -> Frame:
    """Run source and return the frame with all defined names accessible."""
    init_stdlib()
    ast2 = _parse_and_lower(src)
    frame = Frame(source=src, source_path=source_path)
    eval_expr(ast2, frame, source=src)
    return frame


def eval_in_env(src: str, frame: Frame) -> ShkValue:
    """Evaluate an expression in an existing frame/environment."""
    ast2 = _parse_and_lower(src)
    return eval_expr(ast2, frame, source=src)


def repl_eval(src: str, frame: Frame) -> tuple[ShkValue, bool]:
    """Evaluate src in a persistent frame. Returns (result, is_statement)."""
    ast2 = _parse_and_lower(src)
    is_stmt = _last_is_stmt(ast2)
    result = eval_expr(ast2, frame, source=src)
    return result, is_stmt


def _load_source(arg: Optional[str]) -> tuple[str, Optional[str]]:
    """
    Resolve CLI input into source text.
    - None or "-" => read stdin.
    - Existing path => read file contents.
    - Otherwise treat the argument as literal source.
    """

    if arg is None or arg == "-":
        data = sys.stdin.read()
        if not data:
            raise SystemExit("No input provided on stdin")
        return data, None

    candidate = Path(arg)
    if candidate.exists():
        return candidate.read_text(encoding="utf-8"), str(candidate)

    return arg, None


def main() -> None:
    grammar_variant = "default"
    grammar_path = None
    arg = None
    show_tree = False
    enable_py_trace = False
    start_repl = False
    it = iter(sys.argv[1:])

    for token in it:
        if token == "--tree":
            show_tree = True
        elif token == "--py-trace":
            enable_py_trace = True
        elif token == "--repl":
            start_repl = True
        elif arg is None:
            arg = token
        else:
            raise SystemExit(f"Unexpected argument: {token}")

    if start_repl or (arg is None and not show_tree and sys.stdin.isatty()):
        try:
            from .repl import repl
        except ImportError:
            raise SystemExit(
                "REPL requires 'prompt_toolkit'. Install it with: pip install prompt_toolkit"
            )

        repl()
        return

    arg = arg or "-"
    source, source_path = _load_source(arg)

    if show_tree:
        # Parse and show AST without executing
        init_stdlib()
        ast2 = _parse_and_lower(source)
        print(ast2.pretty())
    else:
        if enable_py_trace:
            os.environ["SHAKAR_DEBUG_PY_TRACE"] = "1"
        try:
            init_stdlib()
            ast2 = _parse_and_lower(source)
            stmt = _last_is_stmt(ast2)
            result = eval_expr(
                ast2, Frame(source=source, source_path=source_path), source=source
            )
        except ShakarRuntimeError as exc:
            print(str(exc), file=sys.stderr)
            if debug_py_trace_enabled():
                import traceback

                tb = getattr(exc, "shk_py_trace", None)
                if tb is not None:
                    print("\nPython traceback:", file=sys.stderr)
                    print("".join(traceback.format_tb(tb)), file=sys.stderr, end="")
            raise SystemExit(1) from None
        else:
            if not stmt and not isinstance(result, ShkNil):
                print(result)
        finally:
            if enable_py_trace:
                os.environ.pop("SHAKAR_DEBUG_PY_TRACE", None)


if __name__ == "__main__":
    main()
