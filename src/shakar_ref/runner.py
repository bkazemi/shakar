from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from lark import Lark, UnexpectedInput
from .parse_auto import build_parser  # use the project's builder (lexer remap + indenter)
from .parse_auto import Prune, looks_like_offside
from .lower import lower
from .evaluator import eval_expr
from .runtime import ShkValue
from .runtime import Frame, init_stdlib

REPO_ROOT = Path(__file__).resolve().parents[2]

def _read_grammar(grammar_path: Optional[str], variant: str="default") -> str:
    if grammar_path:
        p = Path(grammar_path)
        if p.exists():
            return p.read_text(encoding="utf-8")

    # fallback: sibling grammar files at the repository root
    parent_dir = REPO_ROOT

    if variant == "lalr":
        cand = parent_dir / "grammar_lalr.lark"
        if cand.exists():
            return cand.read_text(encoding="utf-8")

    fallback = parent_dir / "grammar.lark"
    if fallback.exists():
        return fallback.read_text(encoding="utf-8")

    raise FileNotFoundError("grammar.lark not found. pass an explicit path")

def make_parser(grammar_path: Optional[str]=None, use_indenter: bool=False, start_sym: Optional[str]=None, grammar_variant: str="default") -> Lark:
    g = _read_grammar(grammar_path, variant=grammar_variant)

    if start_sym is None:
        start_sym = "start_indented" if use_indenter else "start_noindent"

    parser_kind = "earley"
    if grammar_variant == "lalr":
        parser_kind = "lalr"

    parser: Lark = build_parser(g, parser_kind=parser_kind, use_indenter=use_indenter, start_sym=start_sym)
    return parser

def run(src: str, grammar_path: Optional[str]=None, use_indenter: Optional[bool]=None, grammar_variant: str="default") -> ShkValue:
    init_stdlib()

    if use_indenter is None:
        preferred = looks_like_offside(src)
        attempts = [preferred, not preferred]
    else:
        attempts = [use_indenter]

    last_error: Optional[UnexpectedInput] = None
    tree = None

    for flag in attempts:
        parser = make_parser(grammar_path, use_indenter=flag, grammar_variant=grammar_variant)
        try:
            tree = parser.parse(src)
            break
        except UnexpectedInput as exc:
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
    it = iter(sys.argv[1:])

    for token in it:
        if token == "--lalr":
            grammar_variant = "lalr"
            continue

        if token.startswith("--grammar="):
            grammar_path = token.split("=", 1)[1]
            continue

        if token == "--grammar":
            try:
                grammar_path = next(it)
            except StopIteration:
                raise SystemExit("--grammar flag requires a path") from None
            continue

        if arg is None:
            arg = token
        else:
            raise SystemExit(f"Unexpected argument: {token}")

    arg = arg or "-"
    source = _load_source(arg)
    print(run(source, grammar_path=grammar_path, grammar_variant=grammar_variant))

if __name__ == "__main__":
    main()
