from __future__ import annotations

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


def run(src: str, grammar_path: Optional[str]=None, use_indenter: Optional[bool]=None, grammar_variant: str="default") -> ShkValue:  # grammar_path/variant kept for CLI compatibility; ignored
    init_stdlib()

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
    d = getattr(ast2, 'data', None)

    if d in ('start_noindent', 'start_indented'):
        children = getattr(ast2, 'children', None)

        if children and len(children) == 1:
            ast2 = children[0]

    return eval_expr(ast2, Frame(source=src), source=src)


def _load_source(arg: Optional[str]) -> str:
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
        return data

    candidate = Path(arg)
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")

    return arg


def main() -> None:
    grammar_variant = "default"
    grammar_path = None
    arg = None
    show_tree = False
    it = iter(sys.argv[1:])

    for token in it:
        if token == "--tree":
            show_tree = True
        elif arg is None:
            arg = token
        else:
            raise SystemExit(f"Unexpected argument: {token}")

    arg = arg or "-"
    source = _load_source(arg)

    if show_tree:
        # Parse and show AST without executing
        init_stdlib()
        preferred = looks_like_offside(source)
        attempts = [preferred, not preferred]

        last_error = None
        tree = None
        for flag in attempts:
            try:
                tree = parse_source(source, use_indenter=flag)
                break
            except (ParseError, LexError) as exc:
                last_error = exc

        if tree is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Parser failed without producing a parse tree")

        ast = Prune().transform(tree)
        ast2 = lower(ast)
        print(ast2.pretty())
    else:
        print(run(source, grammar_path=grammar_path, grammar_variant=grammar_variant))

if __name__ == "__main__":
    main()
