# lark.old/

This directory contains the legacy Lark-based parser (`parse_auto.py`) that has been **deprecated** in favor of the recursive-descent parser (`src/shakar_ref/parser_rd.py`).

## Current Status

**Lark deprecation is essentially complete.** The RD parser is now the default and handles all sanity tests successfully:
- All grammar/parser tests pass
- All AST scenario tests pass
- All runtime tests pass

## Remaining Usage

None in the main runtime or sanity scripts. `parse_auto.py` is kept for reference only.

## Dependencies

`parse_auto.py` requires the `lark` package. All other code in the project uses local `Tree`/`Tok` implementations from `src/shakar_ref/tree.py`.

## Future

If `parse_auto.py` is no longer needed, this directory (and the `lark` dependency) can be removed entirely.
