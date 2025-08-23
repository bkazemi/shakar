# Shakar Lang — Working Design Notes (Pre‑implementation, v0.1)

> **Changelog (nits pass — v0.1 freeze clarifications):**
> - Added: int overflow = throws (v0.1); set Int = i64, Float = f64; clarified leading‑zero floats (no `.5`).
> - Added: assignment split — `:=` (introduce+yield, expr) vs `=` (update‑only, stmt), with errors and no `x := y := …`.
> - Added: selector list write restrictions — multi‑selector and slice LHS assignment are disallowed in v0.1.
> - Added: `!x` listed alongside `not x` in Unary; binary `??` precedence note affirmed.
> - Affirmed: single‑line comments `#...`; ternary `cond ? a : b`.

> **Status:** concept/spec notes for early compiler & toolchain.
> **Audience:** language implementers and contributors.
> **Mantra:** *Sweet syntax, obvious desugar, zero ceremony.*
> **Philosophy:** keep the core tiny and predictable; push ergonomics into first‑class, deterministic sugars.

This is a **living technical spec**. It front‑loads design choices to avoid “oops” reversals once implementation starts. Every surface sugar has a deterministic desugar to a small, boring core.

- ✅ **Committed**: part of v0.1 surface.
- 🧪 **Experimental**: likely to ship; behind a flag.
- ❓ **Considering**: not in v0.1; tracked for later.

---

## 1) Language goals & invariants

- **Ergonomics > ceremony**; sugars for common patterns.
- **Expression‑local magic only.** Implicit subject `.` never crosses statement boundaries; it follows the **anchor stack** rule (§4).
- **Truthiness:** `nil`, `false`, `0` **and** `0.0`, `""`, `[]`, `{}` are falsey; everything else truthy.
- **Evaluation:** eager, short‑circuit `and`/`or`.
- **Errors:** exceptions; one‑statement handlers via `catch` / `@@`.
- **Strings:**  **Raw strings:** `raw"…"` (no interpolation; escapes processed) and `raw#"…"#` (no interpolation; no escapes; exactly one `#` in v0.1).
  Examples: `raw"Line1\nLine2"`, `raw#"C:\\path\\to\\file"#`, `raw#"he said "hi""#`.
 immutable UTF‑8 with zero‑copy views/ropes + leak‑avoidance heuristics (§12).
- **Objects:** records + descriptors (getters/setters); no class system required for v0.1 (§13).
- **Two habitats:** CLI scripting (Python niche) and safe embedding (Lua niche).

---

## 2) Lexical & literals (v0.1)

- **Identifiers:** `[_A-Za-z][_A-Za-z0-9]*`. Case‑sensitive. Unicode ids ❓ (later).
- **Comments:** `#` to end‑of‑line.
- **Whitespace & layout:** blocks introduced by `:` and indentation (spaces only).
- **Literals:**
  - **Nil/bool:** `nil`, `true`, `false`
  - **Integers:** 64‑bit signed (`Int`). Underscore separators allowed: `1_000_000`. Bases: `0b1010`, `0o755`, `0xFF_EC`. **Overflow throws** in v0.1.
  - **Floats:** IEEE‑754 double (`Float`). `0.5`, `1.0`, `1e-9`, `1_234.5e6`. **Leading zero required** (no `.5`).
  - **Strings:** `"…"`, with escapes `\n \t \\ \" \u{…}`. Multiline strings ❓ (later).
  - **Arrays:** `[1, 2, 3]`
  - **Records:** `{ key: value, other: 2 }` (plain values; getters/setters have contextual `get`/`set`, see §10).
  - **Ranges (as values):** `0..10`, `0..<10` produce **Range** objects (iterables).

---

## 3) Operators & precedence (complete)
**Set and Map algebra precedence:** `*` for set/map intersection participates at the multiplicative tier. `+`, `-`, and set/map `^` participate at the additive tier. Numeric bitwise `^` remains behind the `bitwise_symbols` gate and stays in the bitwise tier.


**Associativity**: unless noted, binary operators are **left‑associative**.
**Precedence table (high → low):**

1. **Postfix**: member `.`, call `()`, index `[]`
2. **Unary**: `-x`, `+x`, boolean `not x` / `!x`, `~x` (bitwise not; gated via `bitwise_symbols`, otherwise use `bit.not(x)`)
3. **Power**: `x ** y` (right‑associative) ✅
4. **Multiplicative**: `*`, `/` (float), `//` (floor int), `%`
5. **Additive / concat**: `+`, `-`
   - `+` adds numbers; concatenates **strings** (rope) and **arrays** (copy‑on‑append semantics).
6. **Shifts** (gated via `bitwise_symbols`): `<<`, `>>` (arithmetic for ints)
7. **Bitwise AND/XOR/OR** (gated via `bitwise_symbols`): `&`, `^`, `|`
   - Note: `&` is also used as the **lambda sigil on the callee** (`map&(...)`). Infix `&` is bitwise; parser disambiguates.
   - `|` at **start of line** is reserved for **punctuation guards** (§7); infix `|` is bitwise OR.
8. **Comparison**: `<`, `<=`, `>`, `>=`, `==`, `!=`, `is`, `is not`, `in`, `not in`
9. **Nil‑safe chain**: `??(expr)` (prefix form; treated as a primary) ✅
10. **Walrus & apply-assign**: `:=`, `.=` (both expression-valued; bind tighter than `and`/`or`, lower than postfix) ✅
11. **Boolean**: `and`, `or` (short‑circuit, value‑yielding) ✅
12. **Ternary**: `cond ? then : else` ✅
12.5 **Nil‑coalescing**: `a ?? b` (returns `a` unless `a` is `nil`, otherwise `b`; right‑associative; binds tighter than `or`).
13. **Assignment** (statements): `=`, compound assigns, `or=`, statement‑subject `=x or y` ✅

### 3.1 Unary
- `$x` (no-anchor; evaluates `x` but does **not** make it an explicit subject/anchor)
- `-x`, `+x` (numeric)
- `not x` (boolean)
- `!x` (boolean NOT; alias of `not x`)
- `~x` (bitwise not on ints; gated by `bitwise_symbols` — otherwise use `bit.not(x)`)

### 3.2 Arithmetic
- **Maps (type-directed overloads by key):**
  - `M + N` merge (RHS wins conflicts on shared keys).
  - `M * N` key intersection (only keys present in both; values taken from RHS).
  - `M - N` key difference (keys in `M` not in `N`).
  - `M ^ N` symmetric key difference (keys in exactly one side; values from the contributing side).

- **Strings and arrays (repeat):** `s * n` and `n * s` repeat a string; `a * n` and `n * a` repeat an array. `n` must be an Int with `n >= 0`; otherwise error.

- **Sets (type-directed overloads):**
  - `A + B` union; `A - B` difference; `A * B` intersection; `A ^ B` symmetric difference.
  - **Precedence:** follows token families (`*` multiplicative; `+`/`-` additive; `^` XOR family). Use parentheses when mixing.
  - **Purity:** these binary operators return a **new** Set; operands are not mutated.

- `/` yields float; `//` floor‑div for ints (works on floats too, returns int via floor).
- `%` is remainder; sign follows dividend (like Python).
- `**` exponentiation (right‑assoc).

### 3.3 Bitwise (via std/bit)
- Symbolic bitwise operators are **not in the v0.1 core**. Use `std/bit` functions:
  - `bit.and(x,y)`, `bit.or(x,y)`, `bit.xor(x,y)`, `bit.not(x)`
  - `bit.shl(x,n)`, `bit.shr(x,n)`
- Teams that need symbolic operators can enable the **`bitwise_symbols`** feature gate (see §19). When enabled, tokens `& | ^ << >> ~` are available with the usual precedence, and one-line guards still reserve `|`/`|:` only **after the first `:`** and only at **bracket-depth 0**.

- Errors on non‑int operands (use explicit conversions).

### 3.4 Comparison & identity
- **Comparison comma-chains (CCC):**
  - **Enter chain mode:** after a comparison on a subject followed by a comma (e.g., `S op X,`).
  - **Subject `S`:** the first explicit operand of the first comparison; **evaluate `S` once** for the chain.
  - **Leg form (only after commas):** optional `and|or` then optional comparator `op` then `Expr`.
    - **Carry-forward comparator**: allowed when the leg's joiner is **absent or `and`**; uses the most recent explicit comparator.
    - **After `or` the comparator is required** to remain in the chain (no carry-forward across `or`).
  - **Joiner is sticky**: defaults to `and` and persists until changed by `, or` / `, and`.
  - **Leaving chain mode:** any `and|or` **without a preceding comma** ends the chain and resumes normal boolean parsing.
  - **Desugaring:** expand left-to-right to `S op_i Expr_i` joined by the current sticky joiner; then apply normal precedence (`and` binds tighter than `or`) with short-circuit evaluation.
  - **Parentheses:** `( … )` may nest CCCs inside expressions without affecting the chain's subject/evaluation rules.
  - **Selectors in comparisons:** with backtick selector-literals `` `Sel` ``:
    - ``S == `Sel` `` ⇒ **any()** membership; ``S != `Sel` `` ⇒ **all()** not-in.
    - ``S <op> `Sel` `` for `<, <=, >, >=` ⇒ **all()**; reduce to `min/max(Sel)` honoring range openness.
      - Examples: ``a < `1..5` `` ⇒ `a < 1`; ``a >= `lo, hi` `` ⇒ `a >= max(lo, hi)`.

  ```shakar
  # CCC sanity
  a = 7; b = 3; c = 10
  assert a > 1, < 9, != 8                  # AND-only
  assert (a > 10, or == 0, or == 1) == ((a > 10) or (a == 0) or (a == 1))

  # Chain continues across OR only with explicit comparator
  assert (a > 10, or >= c) == ((a > 10) or (a >= c))
  # ERROR (leave chain or add comparator): a > 10, or b

  # Leaving chain mode with plain boolean OR (no comma)
  assert (a > 10 or b) == ((a > 10) or b)

  # Carry-forward comparator on AND legs
  assert a == b+4, 8-1                      # ⇒ a == (b+4) and a == (8-1)

  # Selector-literal legs
  assert 4 < `5..10`, != `7, 8`             # ⇒ (4 < 5) and (4 != 7) and (4 != 8)
  ```
- `==`, `!=` are value equality; type‑aware. Cross-type compares: numeric Int vs Float equality compares by value (for example, `3 == 3.0` is true); other cross-type comparisons error (except nil equality).
- Ordering `< <= > >=` defined for **numbers** and **strings** (lexicographic by bytes of normalized UTF‑8). Arrays/records ordering ❓ (not in v0.1).
- `is`, `is not`/`!is` check identity (same object/storage). For strings/views, identity means same `(base, off, len)`; value equality may still be true when identity is false.
- **Identity negation:** `!is` is a single token, equivalent to `is not`. Write `a !is b` (not `a ! is b`).

### 3.5 Membership
- `x in y`:
  - strings: substring
  - arrays: element membership (`==`)
  - records: **key** membership
- `x not in y` / `x !in y` are the negation (synonyms).

### 3.6 Boolean (words only)
- `and`, `or` (short‑circuit, yield last evaluated operand). **No `&&`/`||` in core.**

### 3.7 Ternary
```shakar
x = cond ? a : b
```
- Precedence **lower** than `or`. Associates right: `a ? b : c ? d : e` as `a ? b : (c ? d : e)`.

### 3.8 Assignment forms
- **Map compounds (type-directed):** when the LHS is a Map, mutate in place following the same semantics as the binary forms: `+=`, `-=`, `*=`, `^=`.

- **Set compounds (type-directed):** when the **LHS is a Set**, mutate in place:
  - `A += B` (union-update), `A -= B` (difference-update), `A *= B` (intersection-update), `A ^= B` (symmetric-diff-update).
  - `^=` is available for Sets even when numeric bitwise symbols are gated off; it’s type-directed like `^`.

- Simple: `name = expr` (statement)
- **Compound**: `+= -= *= /= //= %= **=` (core); `<<= >>= &= ^= |=` (gated via `bitwise_symbols`)
- **Defaults**: `or=` and statement‑subject `=x or y` (see §6)
- **Walrus**: `name := expr` (expression, see §5)
 - **Apply-assign**: `LHS .= RHS` (**expression**).
   - Inside `RHS`, `.` = **old value of `LHS`**; then the result writes back to `LHS`.
   - **Yields** the **updated value of `LHS`**, usable in larger expressions.

---

### Increments
- Prefix and postfix `++` / `--` are allowed.
- **Postfix is terminal**: at most once, and only at the **end** of a member/selector chain (e.g., `xs[i]++`).
- Requires an **lvalue**; applying to non-lvalues is an error.

### 3.8.1 Multi-field assign fan-out
#### Surface

```shakar
=user.{name, email}.trim()

user.{first, last} .= .title()

=user.address.{city, street}.title()
```

#### Desugar

```shakar
=user.name.trim(); =user.email.trim()

user.first .= .title(); user.last .= .title()

=user.address.city.title(); =user.address.street.title()
```

#### Rules

- Only after a path; `.{…}` here is not a set literal.
- In `.=` RHS, `.` is the old per-field value.
- Left-to-right order; duplicates collapse; missing fields are errors.

### 3.9 Deep map merge `+>` and `+>=`

#### Surface

```shakar
C = A +> B
cfg +>= env
```

#### Semantics

- Map vs map: recursive key-wise merge; RHS wins on conflicts.
- If either side at a path is not a map, RHS replaces LHS at that path.
- Arrays, sets, strings, numbers: replaced wholesale (no deep behavior).
- Not commutative: `A +> B` may differ from `B +> A`.

#### Precedence

- Additive tier with `+ - ^` (map algebra family).

#### Errors

- `+>=` requires LHS to be a map; otherwise error. `+>` always yields a value per rules above.


### 3.10 Map index with default

#### Surface

```shakar
count = hits["/home", default: 0]
port  = env["PORT", default: "3000"]
value = cfg.db[host, default: "localhost"]
```

#### Semantics

- Maps only. Arrays/strings unchanged (OOB still throws).
- If key exists ⇒ return stored value.
- If key missing ⇒ return the `default:` expression (no throw).
- `default:` is a named argument to the `[]` operator; it doesn’t modify storage.
- Works in selector chains: `cfg.db["host", default: "localhost"]`.

#### Method alternative

- `m.get(key, default: v)` remains valid.

## 4) Implicit subject `.` — anchor stack

### 4.1 Where `.` comes from (binders)
- **Statement-subject assign** `=LHS tail` at statement start: inside the statement, `.` = old `LHS`. After evaluation, assign `LHS = result`. Example: `=a.trim()` means `a = a.trim()`.


- **Statement-subject assign** `=LHS tail` (at **statement start**): within the statement, `.` = **old value of `LHS`**; after evaluation, assign **`LHS = result`**.

`.` exists only inside constructs that **bind** it:

- **Apply-assign** `LHS .= RHS`: inside `RHS`, `.` = **old value of `LHS`**; result writes back to `LHS`.
- **Subjectful loop** `for Expr:` / `for[i] Expr:`: in the loop body (per iteration), `.` = **current element** (and `i` = view index if present).
- **Lambda callee sigil** (e.g., `map&(...)`): inside the lambda body, `.` = **the parameter**.
- **`await[any]`**:
  - Per-arm body: in the winning arm’s body, `.` = **that arm’s result**.
  - Trailing body: `.` = **winning value** and `winner` = label.
- **Selectors** `base[ sel1, sel2, … ]`: inside each selector, `.` = **the `base`** (not the element).

> **Not creators:** one-line guards and plain blocks do not create a subject; use a subjectful `for`, a walrus temporary, or a grouping with an explicit subject.

### 4.2 Leading-dot chains
Within a grouping that has an anchor (see 4.3), any **leading dot** applies to that anchor:

- Property: `.k`
- Call: `.fn(args)`
- Index/selector: `.[i:j]`, `.[sel1, sel2]`

Each step’s **result becomes the next receiver** within that same chain, but the **anchor** stays the same unless retargeted (4.3) or shadowed by a new binder (4.1).

### 4.3 Grouping & the anchor stack

- **Comparison comma-chain legs are _no-anchor_.** The chain's subject is the first explicit operand; `$`/`%` do not retarget it; selector/base rules and `.=`/`=LHS` dot rules are unchanged.

- **Pop on exit:** after an inner grouping ends, the anchor reverts to the prior one; sibling leading-dot terms continue to use that prior anchor.

Example: `a and (b) and .c()`  ⇒  `a and b and a.c()`

- **No-anchor `$`**: `$expr` evaluates but does **not** create/retarget an explicit subject. Leading‑dot chains remain anchored to the prior subject.
  Example: `a and $b and .c` ⇒ `a and b and a.c`
Grouping constructs **push/pop** the current anchor:

- **Push** on entering: parentheses `(`…`)`, lambda bodies, comprehension heads/bodies, and `await[…]` per-arm / trailing bodies.
- **Retarget** inside a grouping by mentioning a new **explicit subject** (e.g., `b`, `user.profile[0]`).
- **Pop** on exit: restore the prior anchor.

Example:
```shakar
a and (b and .x()) and .y()   # `.x()` anchored to `b`; `.y()` anchored to `a`
```

### 4.4 Interactions & shadowing
- **Nested binders shadow**: inner binders (4.1) temporarily set `.` for their extent, then restore.
- **Loop × apply-assign**: in `for[i] xs: xs[i] .= RHS`, inside `RHS` the subject is **old `xs[i]`**, not the loop element.
 - **After `.=` within the same grouping**: leading-dot chains anchor to the explicit `LHS`; after the write they observe the **updated** `LHS`.
  - In the RHS, selectors may start from `.`; within selectors `.` refers to the **old LHS base** (e.g., `.[i]`).
   Example: `(xs[0] .= .trim()) and .hasPrefix("a")`.

### 4.5 Illegals & invariants
- **Invalid statement-subject**: `=LHS` with no tail (no effect) — use `.=` or provide a tail. `=.trim()` is illegal (free `.`).

- `.` is **never an lvalue**: `. = …` is illegal.

- **No free dot**: using `.` outside an active binder/anchor is illegal.

### 4.6 Normative laws (concise) — 2025-08-17

1) **Anchor law (SHALL).** The first explicit subject in a grouping sets the **anchor**; all leading-dot chains in that grouping apply to that anchor until retarget or pop.

2) **Selector law (SHALL).** Inside `base[ … ]`, `.` = **base**. Outside, the value of `base[ … ]` may itself retarget the anchor.

3) **Leading-dot chain law (SHALL).** Within a chain, each step’s result becomes the next **receiver**; the **anchor** remains as set by the grouping unless retargeted.

4) **Binder-shadowing law (SHALL).** Inner binders (lambda, `await`, subjectful `for`, apply-assign RHS) temporarily set `.` for their dynamic extent, then restore.

5) **Illegals & locality (SHALL NOT / SHALL).**
   - `.` is never an lvalue (`. = …` illegal).
   - No free `.` outside an active binder/anchor.

#### Minimal conformance checks
```shakar
# Apply-assign with selector on old LHS
xs = [[1,2,3], [4]]
xs .= .[0]
assert xs == [1,2,3]

# Expression-valued: sibling leading-dot sees updated LHS
(xs .= .[0]) and .len == 3

# Statement-subject vs apply-assign symmetry
s = "  A  "
=s.trim()
t = "  B  "; t .= .trim()
assert s == "A" and t == "B"
```

```shakar
# Statement-subject assignment writes back to the same path
s = "  hi  "
=s.trim()
assert s == "hi"

user = { name: "  Bob  " }
=user.name.trim()
assert user.name == "Bob"

xs = [" A ", "b"]
=xs[0].trim()
assert xs[0] == "A"

# Errors
# ERROR: =user.name     # missing tail
# ERROR: =.trim()       # free '.' cannot be the subject
```

```shakar
# Paren pop resumes prior anchor
(a := { c: &() true }); a and (true) and .c()   # ⇒ a and true and a.c()

# Apply-assign then sibling leading-dot sees updated LHS
xs = ["  a", "b  "]
(xs[0] .= .trim()) and .hasPrefix("a")
```

```shakar
# Range as value vs selector:
sum := 0
for i in 0..3: sum += i           # 0+1+2+3
ix, vals = [], []
xs = [10,11,12,13]
for[i] xs[1..3]: ix.append(i); vals.append(.)
assert ix == [0,1] and vals == [11,12]
```

```shakar
# Anchor + grouping
a and (b and .x()) and .y()      # `.x()` anchored to b; `.y()` anchored to a

# Selector inside vs outside
users[ .len-1 ]                  # inside: '.' = users
(users[0] and .name)             # outside: '.' = users[0]

# Binder shadowing: loop × apply-assign
for[i] names: names[i] .= .trim()   # in RHS: '.' = old names[i]

# Local grouping:
(u.trim()).len > 0 and u.has("x") # after expr, '.' reverts
```

_Formal anchor rules: see §20 Grammar sketch (Anchor semantics notes)._
## 5) Walrus `:=` — expression assignment + anchor

**Surface:**
```shakar
u := makeUser() and .isValid()
```

**Semantics:** `LHS := RHS` **assigns** the result of `RHS` to `LHS`, **yields** that same value as the expression result, and **anchors `.` to `LHS`** for:
- the remainder of the **containing expression** at the same nesting level, and
- **within `RHS` after** its head value is computed.

**Precedence:** higher than `and`/`or`, lower than postfix (`.`, `[]`, `()`), tighter than `catch`/`@@` binding to its immediate RHS.

**Desugar (conceptual):**
```shakar
u := makeUser() and .isValid()
# ⇒ tmp0 = makeUser()
# ⇒ u = tmp0
# ⇒ tmp0 and u.isValid()
```

**Use `bind` when you don't want an operator:**
```shakar
makeUser() bind u and .isValid()
```

**Interaction with statement‑rebinding `=`:**
```shakar
=a.lower() and .allLower()
# ⇒ tmp0 = a.lower(); a = tmp0; tmp0 and a.allLower()
```

---

### 5.1 Assignment forms — `:=` vs `=` (v0.1)

**Intent split**
- `name := expr` — **introduce & yield**: creates a new binding in the **current lexical scope** and **returns** the assigned value. Error if `name` already exists **in the same scope** (use `=` to update).
- `name = expr` — **update only**: assigns to an **existing** binding. Statement‑only: using `=` where a value is required is a compile‑time error. Error if `name` does **not** exist (use `:=` to introduce).

**Destructuring**
- Arrays: `[head, ...rest] := xs`
- Records: `{id, name} := user`
Updates use `=` on existing lvalues: `user.name = "New"`.

**Chaining**
- Disallow `x := y := 0` in v0.1 (write as two assignments or a destructure).

**LValues**
- Member/index updates are normal updates: `obj.field = v`, `arr[i] = v`.
- **Selector lists and slices may not appear on the LHS** in v0.1 (no multi‑write or slice assignment).
## 6) Defaults, guards, safe access (committed)
- **`or=`** and **statement‑subject** `=x or y` (statement head) desugar to `if not …: … = …`.
- **`?ret expr`** early‑returns if expr is truthy.
- **Postfix guards**: `stmt if cond`, `stmt unless cond`.
- **Nil‑safe chain**: `??(expr)` turns a deep deref/call chain into a nil‑propagating expression.

### Using

Purpose: scoped resource management with guaranteed cleanup.

**Surface**
```shakar
using expr:
  block

using expr bind name:
  block

using[name] expr:
  block
```

**Semantics**
1) Evaluate `expr` to `r`.
2) **Enter**:
   - If `r.using_enter()` exists, call it. Let its result be `v`.
   - Else if `r.enter()` exists, call it. Let its result be `v`.
   - Else `v = r`.
   If `bind name` is present, bind `name = v` for the block only.
3) Run `block`. Using does not change `.`.

**Dot semantics**
- `using` is not subjectful. It does not create or retarget the anchor.
- Inside the block, `.` is whatever an enclosing binder set. If there is no binder, a free `.` is an error.
- Preferred style: call through the bound name.
  ```shakar
  using connect(url) bind conn:
    conn.send(payload)
    conn.closeWhenIdle()
  ```
- If you need dot on the resource in a larger expression, use a temporary with walrus:
  ```shakar
  using openFile(p) bind f:
    (tmp := f).write(data)
  ```
- `using` is not subjectful. It does not create or retarget the anchor.
- Inside the block, `.` is whatever an enclosing binder set. If no binder is active, a free `.` is an error.

4) **Exit** (always runs, even on error or `return`):
   - If `r.using_exit(err?)` exists, call it with an error value if the block threw, or with no arg if it did not.
   - Else if `r.exit(err?)` exists, call it with the same rule.
   - Else if `r.close()` exists, call `close()` with no args.
   - If the chosen exit method returns truthy, the original block error is suppressed. Errors thrown by the exit method are raised; if both the block and exit fail, both error contexts are preserved in diagnostics.

**Notes**
- Single resource per `using`. Nest for multiple resources.
- `bind` is optional. Without `bind`, the enter value is not named.

**Examples**
```shakar
# File handle that defines using_enter/using_exit
using openFile(path):
  processFile(.)

# Socket with enter/exit methods and a bound name
using connect(url) bind conn:
  conn.send(payload)

# Object with only close()
using[f] getTempFile():
  f.write(data)
# close() is called after the block
```

---

## 7) Punctuation guards (drop `if/elif/else` noise)

**Multi‑line chain**
```shakar
ready():
  start()

| retries < 3:
  retry()

|:
  log("not ready")
```

**One‑line chain**
```shakar
ready(): start() | retries < 3: retry() |: log("not ready")
user and .active: process(user) |: log("inactive")
err: ?ret err |: log("ok")
```

**Binding of `|:` (nearest‑else rule):** After a head’s first `:`, any `|`/`|:` at the same bracket depth binds to the **innermost open guard** (nearest head). Wrap inner guards in `(...)` or `{...}` if you want the following `|:` to bind to an outer guard.
**Rules**
**Alias:** `||` and `||:` are accepted as input in one‑line guards and are normalized by the formatter to `|` / `|:`. They are recognized only after the first `:` and only at bracket‑depth 0.
- **Head:** `Expr ":"` starts a chain.
- **Or‑branch (elif):** same indent, `| Expr ":"`.
- **Else:** same indent, `|:` (no space). Else is optional.
- **Disambiguation:** if the head ends with named‑arg colons or a dict literal, wrap head: `(send to: "Ali"): …`, `({k:v}): …`.
- **Don’t mix** punctuation and keywords within **the same chain**. Keywords remain available (`if/elif/else`).

Desugars 1:1 to `if/elif/else`.

---

- **Guard heads with named args:** If a guard head contains `:`, **parenthesize the head** to disambiguate `Head: Body`, e.g., `(send "bob", subject: "Hi"): log("sent").
## 8) Control flow (loops & flow keywords)
- Iterating the `nil` **literal** is a compile-time error; iterating a variable that evaluates to `nil` is a **no-op**.

### Concurrency — `await[any]` / `await[all]`

- `timeout` default unit: **milliseconds**; supports suffixes `ms`, `s`, `m`.

**Goals:** one `await` per construct, no repetition; clear and cancellable.

**`await[any]`** (two exclusive shapes)

A) **Per‑arm bodies (no trailing body):**
```shakar
await[any](
  fetchUser(id): show(.),
  fetchFromCache(id): show(.),
  timeout 200): showFallback()
)
```
- Heads start concurrently. The **first successful** arm runs its body with **`.` = that arm’s result**. Others are canceled.
- Returns that body’s value.

B) **Trailing body (no per‑arm bodies):**
```shakar
await[any](
  user:  fetchUser(id),
  cache: fetchFromCache(id),
  timeout 200
):
  # '.' is the winning value; 'winner' is "user" | "cache" | "timeout"
  show(.)
```
- Heads start concurrently. When one wins, run the trailing block once with **`.` = winning value** and **`winner`** bound to its label.
- Returns the trailing block’s value.

**No trailing body**: returns a record (for programmatic use):
```shakar
r := await[any]( user: fetchUser(id), cache: fetchFromCache(id), timeout 200 )
# r = { winner: "user"|"cache"|"timeout", value: <result> }
```

**`await[all]`** (trailing body or record form)

With trailing body:
```shakar
await[all](
  user:  fetchUser(id),
  posts: fetchPosts(id)
):
  show(user, posts)
```
- Start all heads, await all, then run the body once with **named results** in scope.
- If any head fails, the whole expression throws (wrap a head with `catch` to handle locally).

Without trailing body:
```shakar
g := await[all]( user: fetchUser(id), posts: fetchPosts(id) )
use(g.user, g.posts)
```

**Notes**
- `await[...]` uses **parentheses**, not `{}`; curly braces signify **data literals** in Shakar.
- Do **not** mix per‑arm bodies and a trailing body in the same call (linted).
- `.` refers to the value defined by the construct as described above; it does **not** capture outer `.` from lambdas.

- **for‑in**: `for x in iterable: block`
- Ranges as values: `for i in 0..n: …` and `for i in 0..<n: …`  # iterate a numeric Range
- Selector lists (indexed view): `for[i] xs[ sel1, sel2, … ]: …`  # iterate the concatenated view; i is view index
- Note: the same `a..b` syntax is a **selector** inside `[]`, and a **Range value** elsewhere.
  - Iteration over `nil` is **no‑op** (safe): `for x in maybeNil: …` does nothing if `maybeNil` is nil.
- **break**, **continue**: loop controls.
- **return**: function return.
- **defer** / **using**: resource management (§11).
- **while/until** ❓: not in v0.1 unless demanded; `for` over `Range` covers most cases.

---

### Subjectful `for` (implicit element)

**Form:**
```shakar
for Expr:
  BODY
```
Binds **`.`** to each element of `Expr` for the duration of each iteration. This is sugar for a `for-in` where the
loop variable is immediately made the subject for the body.

- **Indexed arrays/sets:**
  ```shakar
  for[i] xs:
    # i is the index; '.' is the element
    process(i, .)
  ```
- **Maps/records:**
  ```shakar
  for[k] m:
    # '.' is m[k]; 'k' is the key (string key for records; any key for maps)
    use(k, .)

  for[k, v] m:
    # both names bound; '.' is 'v' (the value)
    use(k, v)
  ```

**Rules:**
- `.` is **per-iteration** and **does not leak** outside the loop body.
- Nested subjectful constructs (lambdas `map&`, apply-assign `.=`, `await[any]` arm bodies) **rebind** `.` inside their own scopes.
- A nested `for` **rebinds** `.`; if you need the outer element, name it once (`outer := .`) before entering the inner loop.
- Works with any iterable (`Array`, `Set`, `Map`, `Record`, custom iterables).
- `break` / `continue` behave as in `for-in`.

**Desugar (conceptual):**
```shakar
for xs:
  BODY
# ≈
for __it in xs:
  =__it: BODY
```

### Loop binders

Binder forms:
- `for xs:` — subjectful loop; body sees `.` = element.
- `for[i] xs:` — indexed; `i` = view index; `.` = element.
- `for[k] m:` / `for[k, v] m:` — map/record; `k` = key, `.` = value (or `v` if bound).
- `for … in <selectors>:` — iterate a **selector list** (ranges, slices, indices).

**Hoisted binders (`^name`) — final v0.1 semantics**
- **Meaning:** Bind this loop variable to an **outer-scope** binding; if none exists, **create** it in the immediately enclosing (non-loop) scope with initial `nil`.
- **Per-iteration:** Assign the current loop value to the hoisted binding each iteration. If the loop runs 0 times, a pre-existing binding is unchanged; a newly hoisted binding remains `nil`.
- **Where:** Works in any binder list: `for[^i] 0..n: …`, `for[^k, ^v] m: …`, `for[^i] xs[ sel1, sel2 ]: …`.
- **Errors:** It is illegal to use both `^x` and `x` for the same name in one binder list, or to repeat the same `^name` within a binder list.
- **Closures:** Inner closures capture the **single hoisted binding** (mutations visible across iterations).

Examples:
```shakar
# Hoist-and-declare
for[^idx] 0..4:
  use(idx)
print(idx)     # => 3; if loop didn’t run, idx == nil

sum := 0
for[j, ^sum] arr:
  sum = sum + .
# sum visible after loop; j is loop-local
```

- **for‑in**: `for x in iterable: block`
  - Range values: `for i in 0..n: …`, `for i in 0..<n: …`
  - Iteration over `nil` is **no‑op** (safe): `for x in maybeNil: …` does nothing if `maybeNil` is nil.
- **break**, **continue**: loop controls.
- **return**: function return.
- **defer** / **using**: resource management (§11).
- **while/until** ❓: not in v0.1 unless demanded; `for` over `Range` covers most cases.

---

## 9) Destructuring (no braces)

```shakar
x, y = get_pair()
name, age = user.profile
k, v = pair
```

---

## 10) Comprehensions (`over` + contextual `for`) & binding

**Subject & `.`** — Inside a comprehension, the implicit subject `.` is the current element exactly like in a **subjectful `for`** (`for xs:`). Anything you can do with `.` in a subjectful loop works the same way in the comprehension head/filters.

**Equivalences:**
```shakar
# subjectful for
for names:
  .trim(): emit(.) |: ()

# comprehension (same subject semantics)
cleaned = [ .trim() over names if .trim() ]
```

**Preferred:** `over`. **Contextual alias:** `for` allowed only inside `[]`, `{}` comps.

```shakar
names = [ .trim() over lines if .len > 0 ]
uniq  = set{ .lower() over tokens }
byId  = map{ .id: .; over users if .active }
```

**Explicit binding:** `bind` (does not remove `.`)
```shakar
admins = [ u.name over users bind u if u.role == "admin" ]
pairs  = { k: v over map.items() bind k, v }
```

**Desugar (list):**
```
lines.map&(.trim()).filter&(.len > 0).to_list()
```

---

- **`bind` scope:** names introduced by `bind` are visible **throughout the comprehension** (head, filters, body), but not outside it.
- **Binder list sugar on `over`:** `over[binders] src` is sugar for `over src bind binders`. Records: one binder yields the key, two binders yield key and value. It does not remove `.` in the body.

**Examples (binder list sugar):**
```shakar
pairs = { k: v over[k, v] map.items() }
names = [ u.name over[u] users if u.role == "admin" ]
sums = [ a + b over[a, b] aAndB ]
```

- **Illegal combination:** `over[...] Expr bind ...` is not allowed. The formatter drops the redundant `bind` and keeps the bracket form.

## 11) Named‑arg calls (paren‑light)

```shakar
fn send(to, subject, body): …
send "bob@x.com", subject: "Hi", body: "…"
```
If a head expression with named args is used as a punctuation‑guard head, wrap it in `(...)` (§7).

---

## 12) Strings & performance (critical)

## 12.5) Slicing & selector lists
- **Ranges as values vs slices:** ranges as values do not clamp and support an optional `:step` (negative step allowed; step 0 is an error). Slices inside `[]` still clamp; index selectors throw on out-of-bounds.


## 12.6) Regex helpers

- Literal: `r"..."/imsx` (flags optional).
- Methods on regex value: `.test(str) -> Bool`, `.match(str) -> Match?`, `.replace(str, repl) -> Str`.
- Timeouts and engine selection are library concerns (not syntax).

**Slices (arrays & strings):**
```shakar
a[i:j]         # half‑open: i ≤ k < j
a[i:j:step]    # step ≠ 0; supports negative steps for reverse
a[:j]          # from start
a[i:]          # to end
a[:]           # whole view
```
- **Negative indices** allowed (`-1` is last).
- **Reverse requires explicit negative step:** `a[i:j:-1]`. We **do not** auto‑reverse when `i > j`.
- **Indexing** `a[i]` **throws** if out‑of‑bounds.
- **Slices** **clamp** to `[0..len]` and never throw; an inverted range with positive step yields `[]`.
- **Strict slicing**: `slice!(a, i, j, step?)` throws on any out‑of‑bounds (no clamping).

**Selector lists** (multiple selectors inside `[]`, comma‑separated):
```shakar
xs[:5, 10:15:-2]      # concat first five with a reverse slice
xs[0, 3, 5:]          # specific indices plus a tail
xs[1 : .len-1]        # drop first and last (half‑open excludes last)
```
- The overall result is the **concatenation** of each selector’s result, in order.
- **Per‑selector steps** are allowed.
- **Index selectors**: OOB → **throw**.  **Slice selectors**: **clamp**.
- Inside each selector expression, **`.` = the base** (e.g., `xs`) for that selector only.
**LHS restrictions (v0.1):** selector lists and slices are **expression‑only**; they cannot be used as assignment targets. Use single index/property updates instead.

- Immutable UTF‑8 with **views** for slices/trim/drop and **ropes** for concat.
- **Unique‑owner compaction** avoids substring leaks (thresholds: keep <¼ or >64KiB slack).
- Controls: `.own()`, `.compact()`, `.materialize()`.
- Unicode indexing counts **characters**; `bytes` view for byte‑wise operations.
- FFI always materializes exact bytes (+NUL) and caches pointer.

What this buys:
```shakar
=s[1:]            # view or in-place compact when unique
=s.trim()         # view (offset/len tweaked)
=path[.len-1]     # view of last segment
```

For heavy edits, use `TextBuilder/Buf` and `freeze()` back to `Str`.

---

## 13) Object model (records + descriptors)

- **Record**: map `name → Slot`.
- **Slot**: `Plain(value)` or `Descriptor{ getter?: fn, setter?: fn }`.
- **Property read**: Plain → value; Descriptor.getter → call with implicit `self`.
- **Property write**: Plain → set; Descriptor.setter → call with `self, value`; strict mode can forbid setting Plain when a descriptor exists.
- **Methods**: `obj.m(args)` sets `self=obj` for the call frame.
- **Contextual `get/set`** inside record literals desugar to `Descriptor` slots.
- **Decorator‑based getters/setters** set flags or return descriptor wrappers so assignment installs the right slot.
- Getter must be **nullary**; setter **arity 1**.

To pass a callable for a getter: `&(obj.prop())`.

---

### Records vs Maps — literals & access
- **Records**: `{ … }` with **string keys** (identifier keys sugar to strings). Dot access (`rec.k`) and bracket access (`rec["k"]`).
- **Maps**: `map{ … }` with **any keys**; **semicolon-delimited** entries inside `map{ … }`; **bracket-only** access (`m[key]`).
- **Empty literals**: `{}`, `set{}`, `map{}`.
- **Equality & iteration**: both structural, order-insensitive; iteration preserves insertion order.
## 14) Decorators (no HOF boilerplate) & lambdas
- Decorators & hooks are **core** in v0.1.

- **Decorators**: inside `decorator` bodies, `f` and `args` are implicit; if the body does not `return`, implicitly `return f(args)`.
- **Lambdas**: `map&(.trim())` (single arg implicit `.`), `zipWith&[a,b](a+b)` (multi‑arg). `&` is a **lambda‑on-callee** sigil.

---

## 15) Errors, catch, assert, dbg, events

- **One‑stmt catch**:
  ```shakar
  val = risky() catch e => handle(e)
  val = risky() @@ e => handle(e)   # shorthand
  risky() catch { log("oops") }     # statement form
  ```
- **assert expr, "msg"** raises if falsey; build can strip or keep.
- **dbg expr** logs and returns expr; can be stripped in release.
- **Events**: `hook "name" => &(handler(.))` ⇒ `Event.on(name, handler)`

---

## 16) Keywords (v0.1)

**Reserved (cannot be identifiers):**
`and, or, not, if, elif, else, for, in, break, continue, return, assert, using, defer, catch, decorator, decorate, hook, get, set, bind, over, true, false, nil`

**Contextual:**
- `for` in comprehensions (`[...] for src …` as alias for `over`)
- `get`/`set` only **inside record literals**

**Punctuation (syntax, not keywords):**
- `.=` apply-assign
- `|` and `|:` at line start for guard chains
- `?ret` (statement head)
- `??(expr)` nil‑safe chain
- `:=` walrus

---

## 17) Type model (pragmatic dynamic)

- **Numbers**: `Int` (64‑bit signed; **overflow throws** in v0.1) and `Float` (64‑bit IEEE). Arithmetic follows Python‑like precedence; `/` produces float, `//` floor‑div produces `Int`.
- **Bool**: `true/false`
- **Str**: immutable UTF‑8
- **Array**: ordered, dynamic
- **Record**: key→value map (string keys); descriptors for getters/setters
- **Func**: user/native functions
- **Range**: iterable numeric range
- **Nil**: null value

Type predicates/methods live in stdlib (`isInt(x)`, `typeOf(x)`), not as syntax.

---

## 18) Tooling & ecosystem

- **Formatter** (`shk fmt`): canonical spacing; can normalize punctuation vs keywords per project style.
- **REPL** (`shk repl`): auto‑print last value; `:desugar` to show lowerings.
- **Linter** (initial rules):
  - Suggest `a or= b` / `=a or b` over `a = a or b`.
  - Warn on long implicit `.` chains; suggest `bind`.
  - Don’t mix punctuation and keyword guards in the same chain.
  - Warn on tiny views over huge bases; suggest `.compact()`/`.own()`.
- **Diagnostics with fix‑its**:
  - “`.` cannot stand alone; start from a subject or use `bind`.”
  - “Guard head ends with named args; wrap with parentheses.”
- **AI‑friendly**:
  - `shk --desugar` and `--resugar` for round‑trips.
  - Machine‑readable feature manifest (`shakar.features.json`) describing allowed sugars/style picks.
  - Lenient parse `--recover` to auto‑fix common near‑misses.

---

## 18.5) Tracing (built-in span helper)

`trace` is a function with block sugar. It measures duration, captures errors, and returns the body’s value.

```shakar
trace "load_users":
  u := getUsers()
  log(u.len)

# expression form returns the body value
users = trace "map+trim": users.map&(.trim())

# with attributes
trace "fetch" { user: id, cold: fromNetwork }:
  await[any](
    fetchUser(id): .,
    timeout 200: []
  )
```
## 19) Feature gating & style profiles

Project config (e.g. `shakar.toml`) with **per‑module overrides**:
```toml
[shakar]
version = "0.1"
allow   = ["punctuation_guards","getter_properties","using","defer","dbg"]
deny    = ["keyword_aliases","autocall_nullary_getters"]

[style]
prefer_guards           = "punctuation"   # or "keywords"
prefer_default_assign   = "or="           # or "stmt_subject"
max_inline_guard_branches = 3

[module."core/auth"]
deny  = ["implicit_lambdas"]
allow = ["?ret"]
```

- **Version pinning** protects against future keyword additions.
- **Allow/deny** lists gate syntax families & stdlib sugars.
- Formatter honors style preferences (e.g. guard style).

---

### Formatter / Lints (normative)
- **Style -- `over[...]` vs `bind`:** prefer `over[...]` when you have more than one binder. Do not use both `over[...]` and `bind` in the same head.
- **Style -- Guard heads with paren-light calls:** discouraged; the formatter may auto-paren such heads (see §7).
- **Tokenization -- ``!in``:** write as a single token with no internal space: `a !in b` (not `a ! in b`). Parser treats `! in` as unary `!` then `in`.
- **Style -- Comparison comma-chains with `or`:** When a comma-chain switches to `or`, prefer either repeating `, or` for each subsequent leg (e.g., `a > 10, or == 0, or == 1`) or grouping the `or` cluster in parentheses (e.g., `a > 10, or (== 0, == 1)`). This improves readability; semantics are unchanged (joiner is sticky; a comparator is required after `or` to remain in the chain).
- **Flag free `.`**: bare `.` outside an active binder/anchor is an error.
- **Auto-paren guard heads with `:`**: when a guard head contains `:`, the formatter inserts parentheses; style check warns if missing.

**Feature gate note (bitwise_symbols):**
- Numeric bitwise operators `& | ^ << >> ~` and their compound forms are behind the **`bitwise_symbols`** gate (default **denied**).
- This gate applies **only to numeric operators**. Set/Map algebra (e.g., `A ^ B` symmetric diff) is **always enabled** (type-directed).
## 20) Grammar sketch (EBNF‑ish; implementation may vary)
```ebnf
(* Shakar grammar — proper EBNF. Lexical tokens: IDENT, STRING, NUMBER, NEWLINE, INDENT, DEDENT. *)

(* ===== Core expressions ===== *)

Expr            ::= OrExpr ;

OrExpr          ::= AndExpr ( "or" AndExpr )* ;
AndExpr         ::= NullishExpr ( "and" NullishExpr )* ;
NullishExpr     ::= CompareExpr ( "??" CompareExpr )* ;
CompareExpr     ::= AddExpr ( CmpOp AddExpr )* ;

CmpOp           ::= "==" | "!=" | "<" | "<=" | ">" | ">=" | "is" | "is not" | "!is" | "in" | "!in" | "not in" ;

AddExpr         ::= MulExpr ( AddOp MulExpr )* ;
AddOp           ::= "+" | "-" | "^" | DeepMergeOp ;

MulExpr         ::= UnaryExpr ( MulOp UnaryExpr )* ;
MulOp           ::= "*" | "/" | "%" ;

UnaryExpr       ::= UnaryPrefixOp UnaryExpr | PostfixExpr ;
UnaryPrefixOp   ::= "+" | "-" | "not" | "!" ;

PostfixExpr     ::= Primary ( Postfix )* ;
Postfix         ::= "." IDENT
                  | "[" SelectorList ("," "default" ":" Expr)? "]"
                  | "(" ArgList? ")" ;

SelectorList    ::= Selector ("," Selector)* ;
Selector        ::= IndexSel | SliceSel ;
IndexSel        ::= Expr ;  (* evaluates to int/key; OOB on arrays throws *)
SliceSel        ::= OptExpr ":" OptExpr (":" Expr)? ;
OptExpr         ::= /* empty */ | Expr ;

Primary         ::= IDENT
                  | Literal
                  | "(" Expr ")"
                  | SubjectExpr
                  ;

Literal         ::= STRING | NUMBER | "nil" | "true" | "false" ;

SubjectExpr     ::= IDENT ( Postfix )+ ;
ImplicitUse     ::= "." ( Postfix )+ ;  (* only in contexts with an implicit subject *)

RangeExpr       ::= Expr ".." Expr (":" Expr)? | Expr "..<" Expr (":" Expr)? ;

(* ===== Assignment forms ===== *)

ApplyAssign     ::= LValue ".=" Expr ;
StmtSubjectAssign ::= "=" LValue StmtTail ;  (* '.' = old LHS; writes back to same LHS path *)
DeepMergeOp     ::= "+>" ;
DeepMergeAssign ::= LValue "+>=" Expr ;
AssignOr        ::= LValue "or=" Expr ;

StmtTail        ::= Expr ;

LValue          ::= IDENT ( Postfix )* ( FieldFan )? ;
FieldList       ::= IDENT ("," IDENT)* ;
FieldFan        ::= "." "{" FieldList "}" ;

(* ===== Statements & blocks ===== *)

SimpleStmt      ::= Expr
                  | ApplyAssign
                  | AssignOr
                  | Destructure
                  | GuardReturn
                  | Dbg
                  | Assert
                  | UsingStmt
                  | DeferStmt
                  | Hook
                  | CatchStmt
                  | PostfixIf
                  | PostfixUnless ;

PostfixIf       ::= SimpleStmt "if" Expr ;
PostfixUnless   ::= SimpleStmt "unless" Expr ;

Block           ::= NEWLINE INDENT Stmt+ DEDENT ;
InlineBlock     ::= "{" InlineStmt* "}" ;
InlineStmt      ::= SimpleStmt ;

Stmt            ::= SimpleStmt
                  | ForIn | ForSubject | ForIndexed | ForMap1 | ForMap2
                  | AwaitAnyCall | AwaitAllCall ;

(* ===== Control flow ===== *)

GuardReturn     ::= "?ret" Expr ;

GuardChain      ::= GuardHead GuardOr* GuardElse? ;
GuardHead       ::= Expr ":" NEWLINE INDENT Block DEDENT ;
GuardOr         ::= ", or" NEWLINE INDENT Block DEDENT ;
GuardElse       ::= "|:" NEWLINE INDENT Block DEDENT ;

OneLineGuard    ::= GuardBranch ("|" GuardBranch)* ("|:" InlineBody)? ;
GuardBranch     ::= Expr ":" InlineBody ;
InlineBody      ::= SimpleStmt | "{" InlineStmt* "}" ;

(* ===== Loops ===== *)

ForIn           ::= "for" IDENT "in" Expr ":" Block ;
ForSubject      ::= "for" Expr ":" Block ;
ForIndexed      ::= "for" "[" IDENT "]" Expr ":" Block ;
ForMap1         ::= "for" "[" IDENT "]" Expr ":" Block ;  (* '.' = value; IDENT = key *)
ForMap2         ::= "for" "[" IDENT "," IDENT "]" Expr ":" Block ;  (* '.' = value; key,value bound *)

(* ===== Selectors (per-selector step allowed) ===== *)

Indexing        ::= PostfixExpr "[" SelectorList "]" ;  (* kept for reference; covered by Postfix *)

(* ===== Calls & lambdas ===== *)

Callee          ::= PostfixExpr ;
Call            ::= Callee "(" ArgList? ")"
                  | Callee ArgListNamedMixed ;

ArgList         ::= Expr ("," Expr)* ;
ArgListNamedMixed ::= (Expr ("," Expr)*)? ("," NamedArg ("," NamedArg)*)? ;
NamedArg        ::= IDENT ":" Expr ;

LambdaCall1     ::= Callee "&" "(" Expr ")" ;
LambdaCallN     ::= Callee "&" "[" ParamList "]" "(" Expr ")" ;
ParamList       ::= IDENT ("," IDENT)* ;

(* ===== Comprehensions ===== *)

CompHead        ::= ("over" | "for") OverSpec ;
OverSpec        ::= "[" BinderList "]" Expr | Expr ("bind" Pattern)? ;
BinderList      ::= Pattern ("," Pattern)* ;
IfClause        ::= "if" Expr? ;

ListComp        ::= "[" Expr CompHead IfClause? "]" ;
SetComp         ::= "{" Expr CompHead IfClause? "}" ;
DictComp        ::= "{" Expr ":" Expr CompHead IfClause? "}" ;

Pattern         ::= IDENT | Pattern "," Pattern | "(" Pattern ("," Pattern)* ")" ;
Destructure     ::= Pattern "=" Expr ;

(* ===== Concurrency ===== *)

AwaitAnyCall    ::= "await" "[" "any" "]" "(" AnyArmList OptComma ")" (":" InlineBlock)? ;
AwaitAllCall    ::= "await" "[" "all" "]" "(" AllArmList OptComma ")" (":" InlineBlock)? ;
AnyArmList      ::= Expr ("," Expr)* ;
AllArmList      ::= Expr ("," Expr)* ;
OptComma        ::= /* empty */ | "," ;

(* ===== Error handling / hooks ===== *)

CatchExpr       ::= Expr "catch" IDENT? "=>" Expr ;
CatchStmt       ::= Expr "catch" Block ;
CatchCute       ::= Expr "@@" IDENT? "=>" Expr ;

Hook            ::= "hook" STRING "=>" LambdaExpr ;
LambdaExpr      ::= "(" ParamList? ")" "=>" (Expr | Block) ;

(* ===== Records ===== *)

RecordItem      ::= IDENT ":" Expr
                  | "get" IDENT "(" ")" ":" Block
                  | "set" IDENT "(" IDENT ")" ":" Block ;

(* ===== Using / Defer / Assert / Debug ===== *)

DeferStmt       ::= "defer" SimpleCall ;
SimpleCall      ::= Callee "(" ArgList? ")" ;
UsingStmt       ::= "using" Expr ("bind" IDENT)? ":" Block ;

Assert          ::= "assert" Expr ("," Expr)? ;
Dbg             ::= "dbg" (Expr ("," Expr)?) ;
```

---

#### Anchor semantics (normative notes)
```
Primary      := Ident | Literal | "(" Expr ")"
Call         := "(" ArgList? ")"
Selector     := "[" SelectorList "]"
MemberExpr   := Primary ( "." Ident | Call | Selector )*

# AnchorScopes: ParenthesizedExpression, LambdaBody, CompHead, CompBody, AwaitBody
# push/pop the current anchor.
# The first MemberExpr evaluated in an AnchorScope retargets the anchor for that scope.
# Lead-dot chains ( "." (Ident | Call | Selector) ) use the current anchor as receiver;
# chain steps advance the receiver, not the anchor.
# Inside Selector expressions, '.' denotes the base (the MemberExpr before '[').
```
## 21) Considering / undecided

- **Keyword aliases (macro‑lite)**: project remaps (disabled by default).
- **Autocall any nullary method**: off by default; explicit `getter` is core.
- **Copy‑update `with`** sugar: syntax TBA; method `obj.with(...)` works now.
- **Pipes `|>`**: maybe later; redundant for v0.1.
- **Nested/multi‑source comprehensions**: later via `bind (a,b) over zip(xs,ys)`.
- **Word range aliases `to`/`until`**: optional later.
- **While/until loops**: consider only if demanded; `for` over `Range` covers most cases.
- **Sticky subject (prefix `%`)**: `%expr` sets the **anchor** to `expr` and marks it **sticky** for the current expression: child groupings do not retarget unless another `%` or a new explicit subject appears. **Does not affect** selector bases or `.=`/`=LHS` tails (their `.` rules still win).

---

## 22) Implementation plan (v0.1)

**Phase 1 — Parser & IR**
- Implement grammar above; produce a compact IR.
- Author **desugar tables** per feature; ensure 1:1 mapping.

**Phase 2 — VM**
- Bytecode interpreter in C (embeddable).
- GC with support for `Str` views/ropes; compaction hooks.
- Records + descriptor slots; implicit `self` for getter/setter/method calls.

**Phase 3 — Stdlib (focused)**
- `fs`, `path`, `json/yaml`, `http`, `time`, `process`, `Event`.
- String APIs aligned with view/rope model (`.own()`, `.materialize()`, `.bytes`).

**Phase 4 — Tooling**
- `shk fmt`, `shk repl --desugar`, initial lints, feature manifest JSON.

**Phase 5 — Feature gates**
- Project + per‑module allow/deny with helpful diagnostics.

**Exit criteria**
- Guard‑heavy and data‑munging demos show 30–50% LOC reduction vs Python/JS.
- Desugar views are stable and readable.
- No substring leaks in memory profiles; heuristics documented & tunable.

---

## 23) Appendix — Examples
### Subjectful loops

```shakar
# Pick longest non-empty name
max := ""
for names:
  ((t := .trim()) and t.len > max.len): max = t

# Indexed variant
for[i] xs:
  log(i, .)

# Maps/records
emails = map{ u.id: u.email; over users }
for[k] emails:
  send(k, .)
for[k, v] emails:
  send(k, v)
```

### Selector lists (indices + slices)
```shakar
xs[:5, 10:15:-2]      # concat of two slices; per-slice step allowed
xs[1: .len-1]         # half-open; excludes last
xs[5:2:-1]            # explicit reverse via negative step
```

```shakar
# Arithmetic & assignment
x = 10
x += 5 * 2
y = 7 // 2          # 3
z = 2 ** 10         # 1024

# Bitwise
mask = bit.or(bit.shl(1, 8), 0x0F)
ok   = bit.and(flags, mask) != 0

# Ternary
level = debug ? "debug" : "info"

# Membership
"a" in "shakar"      # true
3 in [1,2,3]         # true
"id" in { id: 1 }    # true

# Rebinding + anchor
=a.lower() and .isAscii()

# Walrus + anchor
u := makeUser() and .isValid()

# Punctuation guard (one-line)
ready(): start() | retries < 3: retry() |: log("nope")

# Comprehension
names = [ .trim() over lines if .len > 0 ]

# Nil-safe chain
name = ??(user.profile.name.trim()) or "guest"

# Getters
box = {
  items: [1,2,3]
  get size(): items.len
}
print(box.size)
```
