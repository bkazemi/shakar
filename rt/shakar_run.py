from __future__ import annotations

from pathlib import Path
from lark import Lark
import sys
from shakar_parse_auto import build_parser  # use the project's builder (lexer remap + indenter)
from shakar_parse_auto import Prune
from shakar_lower import lower
from shakar_eval import eval_expr
from shakar_runtime import Frame, init_stdlib

def _read_grammar(grammar_path: str|None, variant: str="default") -> str:
    if grammar_path:
        p = Path(grammar_path)
        if p.exists():
            return p.read_text(encoding="utf-8")
    # Fallback: sibling grammar files next to this file
    parent_dir = Path(__file__).resolve().parent.parent
    if variant == "lalr":
        cand = parent_dir / "grammar_lalr.lark"
        if cand.exists():
            return cand.read_text(encoding="utf-8")
    fallback = parent_dir / "grammar.lark"
    if fallback.exists():
        return fallback.read_text(encoding="utf-8")
    raise FileNotFoundError("grammar.lark not found. pass an explicit path")

def make_parser(grammar_path: str|None=None, use_indenter: bool=False, start_sym: str|None=None, grammar_variant: str="default") -> Lark:
    g = _read_grammar(grammar_path, variant=grammar_variant)
    if start_sym is None:
        start_sym = "start_indented" if use_indenter else "start_noindent"
    parser_kind = "earley"
    if grammar_variant == "lalr":
        parser_kind = "lalr"
    return build_parser(g, parser_kind=parser_kind, use_indenter=use_indenter, start_sym=start_sym)

def run(src: str, grammar_path: str|None=None, use_indenter: bool=False, grammar_variant: str="default") -> object:
    init_stdlib()
    parser = make_parser(grammar_path, use_indenter=use_indenter, grammar_variant=grammar_variant)
    tree = parser.parse(src)
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

def _load_source(arg: str | None) -> str:
    """
    Resolve CLI input into source text.
    - None or "-" => read stdin.
    - Existing path => read file contents.
    - Otherwise treat the argument as literal source.
    """
    if arg in (None, "-"):
        data = sys.stdin.read()
        if not data:
            raise SystemExit("No input provided on stdin")
        return data
    candidate = Path(arg)
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return arg

if __name__ == "__main__":
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
