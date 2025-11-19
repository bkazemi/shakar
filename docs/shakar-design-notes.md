# Shakar Lang — Working Design Notes (Pre-implementation, v0.1)

> **Changelog (nits pass — v0.1 freeze clarifications):**
> - Added: int overflow = throws (v0.1); set Int = i64, Float = f64; clarified leading-zero floats (no `.5`).
> - Added: assignment split — `:=` (introduce+yield, expr) vs `=` (update-only, stmt), with errors and no `x := y := …`.
> - Added: selector list write restrictions — multi-selector and slice LHS assignment are disallowed in v0.1.
> - Added: `!x` listed alongside `not x` in Unary; binary `??` precedence note affirmed.
> - Affirmed: single-line comments `#...`; ternary `cond ? a : b`.

> **Status:** concept/spec notes for early compiler & toolchain.
> **Audience:** language implementers and contributors.
> **Mantra:** *Sweet syntax, obvious desugar, zero ceremony.*
> **Philosophy:** keep the core tiny and predictable; push ergonomics into first-class, deterministic sugars.

This is a **living technical spec**. It front-loads design choices to avoid “oops” reversals once implementation starts. Every surface sugar has a deterministic desugar to a small, boring core.

- ✅ **Committed**: part of v0.1 surface.
- 🧪 **Experimental**: likely to ship; behind a flag.
- ❓ **Considering**: not in v0.1; tracked for later.

---

## 1) Language goals & invariants

- **Ergonomics > ceremony**; sugars for common patterns.
- **Expression-local magic only.** Implicit subject `.` never crosses statement boundaries; it follows the **anchor stack** rule (§4).
- **Truthiness:** `nil`, `false`, `0` **and** `0.0`, `""`, `[]`, `{}` are falsey; everything else truthy.
- **Evaluation:** eager, short-circuit `and`/`or`.
- **Errors:** exceptions; one-statement handlers via `catch` / `@@`.
- **Strings:**  **Raw strings:** `raw"…"` (no interpolation; escapes processed) and `raw#"…"#` (no interpolation; no escapes; exactly one `#` in v0.1).
  Examples: `raw"Line1\nLine2"`, `raw#"C:\\path\\to\\file"#`, `raw#"he said "hi""#`.
  immutable UTF-8 with zero-copy views/ropes + leak-avoidance heuristics (§12).
- **Objects:** objects + descriptors (getters/setters); no class system required for v0.1 (§13).
- **Two habitats:** CLI scripting (Python niche) and safe embedding (Lua niche).

---

## 2) Lexical & literals (v0.1)

- **Identifiers:** `[_A-Za-z][_A-Za-z0-9]*`. Case-sensitive. Unicode ids ❓ (later).
- **Comments:** `#` to end-of-line.
- **Whitespace & layout:** blocks introduced by `:` and indentation (spaces only).
- **Semicolons (statements):** `;` is a hard statement delimiter at top level and inside braced inline suites `{ ... }`. Multiple statements may share a line using semicolons. Grammar shape: `stmtlist := stmt (SEMI stmt)* SEMI?`.
- **Inline suites after `:`:** exactly **one** simple statement is allowed. To include multiple, wrap a braced inline suite on the right of the colon: `{ stmtlist }`.

- **Literals:**
  - **Nil/bool:** `nil`, `true`, `false`
  - **Integers:** 64-bit signed (`Int`). Underscore separators allowed: `1_000_000`. Bases: `0b1010`, `0o755`, `0xFF_EC`. **Overflow throws** in v0.1.
  - **Floats:** IEEE-754 double (`Float`). `0.5`, `1.0`, `1e-9`, `1_234.5e6`. **Leading zero required** (no `.5`).
  - **Strings:** `"…"`, with escapes `\n \t \\ \" \u{…}`. Multiline strings ❓ (later).
  - **Arrays:** `[1, 2, 3]`
  - **Objects:** `{ key: value, other: 2 }` (plain values; getters/setters have contextual `get`/`set`, see §10).
  - **Selector literals (as values):** backtick selectors like `` `1:10` `` produce **Selector** values (views/iterables). Default stop is **inclusive**; use `<stop` for exclusive (e.g., `` `[1:<10]` ``).
---

## 3) Operators & precedence (complete)

**Set and Object algebra precedence:** `*` for set/object intersection participates at the multiplicative tier. `+`, `-`, and set/obj `^` participate at the additive tier. Numeric bitwise `^` remains behind the `bitwise_symbols` gate and stays in the bitwise tier.

**Associativity**: unless noted, binary operators are **left-associative**.
**Precedence table (high → low):**

1. **Postfix**: member `.`, call `()`, index `[]`
2. **Unary**: `-x` boolean `not x` / `!x`, `~x` (bitwise not; gated via `bitwise_symbols`, otherwise use `bit.not(x)`)
3. **Power**: `x ** y` (right-associative) ✅
4. **Multiplicative**: `*`, `/` (float), `//` (floor int), `%`
5. **Additive / concat**: `+`, `-`, `+>` (deep object merge)
   - `+` adds numbers; concatenates **strings** (rope) and **arrays** (copy-on-append semantics).
6. **Shifts** (gated via `bitwise_symbols`): `<<`, `>>` (arithmetic for ints)
7. **Bitwise AND/XOR/OR** (gated via `bitwise_symbols`): `&`, `^`, `|`
   - Note: `&` is also used as the **lambda sigil on the callee** (`map&(...)`). Infix `&` is bitwise; parser disambiguates.
   - `|` at **start of line** is reserved for **punctuation guards** (§7); infix `|` is bitwise OR.
8. **Comparison**: `<`, `<=`, `>`, `>=`, `==`, `!=`, `is`, `is not`, `!is`, `in`, `!in`, `not in`
9. **Nil-safe chain**: `??(expr)` (prefix form; treated as a primary) ✅
10. **Nil-coalescing**: `a ?? b` (returns `a` unless `a` is `nil`, otherwise `b`; right-associative; binds tighter than `or`).
11. **Walrus & apply-assign**: `:=`, `.=` (both expression-valued; bind tighter than `and`/`or`, lower than postfix) ✅
12. **Boolean**: `and`, `or` (short-circuit, value-yielding) ✅
13. **Ternary**: `cond ? then : else` ✅
14. **Assignment** (statements): `=`, compound assigns, `or=`, statement-subject `=x or y` ✅

### 3.1 Unary

- `$x` (no-anchor; evaluates `x` but does **not** make it an explicit subject/anchor)
- `-x` (numeric)
- `not x` (boolean)
- `!x` (boolean NOT; alias of `not x`)
- `~x` (bitwise not on ints; gated by `bitwise_symbols` — otherwise use `bit.not(x)`)

### 3.2 Arithmetic
- **Objects (type-directed overloads by key):**
  - `M + N` merge (RHS wins conflicts on shared keys).
  - `M * N` key intersection (only keys present in both; values taken from RHS).
  - `M - N` key difference (keys in `M` not in `N`).
  - `M ^ N` symmetric key difference (keys in exactly one side; values from the contributing side).
- **Strings and arrays (repeat):** `s * n` and `n * s` repeat a string; `a * n` and `n * a` repeat an array. `n` must be an Int with `n >= 0`; otherwise error.
- **Sets (type-directed overloads):**
  - `A + B` union; `A - B` difference; `A * B` intersection; `A ^ B` symmetric difference.
  - **Precedence:** follows token families (`*` multiplicative; `+`/`-` additive; `^` XOR family). Use parentheses when mixing.
  - **Purity:** these binary operators return a **new** Set; operands are not mutated.
- `/` yields float; `//` floor-div for ints (works on floats too, returns int via floor).
- `%` is remainder; sign follows dividend (like Python).
- `**` exponentiation (right-assoc).

### 3.3 Bitwise (via std/bit)
- Symbolic bitwise operators are **not in the v0.1 core**. Use `std/bit` functions:
  - `bit.and(x,y)`, `bit.or(x,y)`, `bit.xor(x,y)`, `bit.not(x)`
  - `bit.shl(x,n)`, `bit.shr(x,n)`
- Teams that need symbolic operators can enable the **`bitwise_symbols`** feature gate (see §19). When enabled, tokens `& | ^ << >> ~` are available with the usual precedence, and one-line guards still reserve `|`/`|:` only **after the first `:`** and only at **bracket-depth 0**.
- Errors on non-int operands (use explicit conversions).

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
      - Examples: ``a < `1:<5` `` ⇒ `a < 1`; ``a >= `lo, hi` `` ⇒ `a >= max(lo, hi)`.

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
  assert 4 < `5:10`, != `7, 8`             # ⇒ (4 < 5) and (4 != 7) and (4 != 8)
  ```
- `==`, `!=` are value equality; type-aware. Cross-type compares: numeric Int vs Float equality compares by value (for example, `3 == 3.0` is true); other cross-type comparisons error (except nil equality).
- Ordering `< <= > >=` defined for **numbers** and **strings** (lexicographic by bytes of normalized UTF-8). Arrays/objects ordering ❓ (not in v0.1).
- `is`, `is not`/`!is` check identity (same object/storage). For strings/views, identity means same `(base, off, len)`; value equality may still be true when identity is false.
- **Identity negation:** `!is` is a single token, equivalent to `is not`. Write `a !is b` (not `a ! is b`).

### 3.5 Membership
- `x in y`:
  - strings: substring
  - arrays: element membership (`==`)
  - objects: **key** membership
- `x not in y` / `x !in y` are the negation (synonyms).

### 3.6 Boolean (words only)
- `and`, `or` (short-circuit, yield last evaluated operand). **No `&&`/`||` in core.**

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
- **Defaults**: `or=` and statement-subject `=x or y` (see §6)
- **Walrus**: `name := expr` (expression, see §5)
 - **Apply-assign**: `LHS .= RHS` (**expression**).
   - Inside `RHS`, `.` = **old value of `LHS`**; then the result writes back to `LHS`.
   - **Yields** the **updated value of `LHS`**, usable in larger expressions.
#### Idioms: `.=` with coalesce

```shakar
# 1) Sanitize strings with a constant fallback
user.{name, email} .= ??(.trim().lower()) ?? "unknown"

# 2) Keep the old value if the transform yields nil (explicit LHS fallback)
user.nick .= .trim() ?? user.nick

# 3) Numeric clamp with constant fallback
profile.age .= .clamp(0, 120) ?? 0

# 4) Selector chains with coalesce
cfg["db", default: {}]["host"].= .trim() ?? "localhost"

# 5) Fan-out with safe normalization and constant fallback
user.{phone, altPhone} .= .digits() ?? ""
```

**Notes**
- Bare `.` cannot stand alone. Do not write `a .= . ?? x` or `?? .`.
- When you must keep the old value on failure, use the explicit LHS name in the fallback, e.g. `a .= ??(.transform()) ?? a`.
- If you only want a default when the slot is nil (no transform), prefer plain assignment with `??`: `a = a ?? default`.

- **LValue shape**: assignment targets are identifiers with member (`.field`) and/or selector (`[index]`) chains; **calls are not allowed on the left-hand side**. Grouped statement-subject `=(LHS)` picks that exact `LHS` as the destination; it still has to appear at the beginning of the statement and the grouped expression must be a legitimate lvalue.

**Statement-subject rules**

- `=name<tail>` desugars to `name = name<tail>`. `name` must already exist, and `<tail>` must perform work; `=name.field` (no call/fan-out/selector/`.=`) is rejected because it has no effect.
- Ungrouped heads therefore require a “real” tail (call, fan-out, selector, `.=` …). If all you need is to point at a deeper slot and keep writing back to the outer identifier, wrap the identifier: `=(name).field`.
- Grouped heads `=(lvalue)<tail>` pick that exact `lvalue` as the destination. The grouped `lvalue` must itself be assignable (identifier with selectors or fan-out, no calls). This is the only way to mutate a nested slot inline, e.g. `=(user.profile.email).trim()`.
- The statement-subject must start the statement; you cannot drop `=…` in the middle of expressions.
- Fan-outs behave the same regardless of grouping—`=(user.{name, email})` is redundant because grouping only changes which head counts as the destination.
- Grouping with only the identifier (e.g. `=(user)` before `<tail>`) is purely for keeping that identifier as the thing being rewritten while `<tail>` walks somewhere else. Use it sparingly.

---

### Increments
- Prefix and postfix `++` / `--` are allowed.
- **Postfix is terminal**: at most once, and only at the **end** of a member/selector chain (e.g., `xs[i]++`).
- Requires an **lvalue**; applying to non-lvalues is an error.
- Prefix returns the updated value; postfix evaluates to the pre-update value.
- Operates on numeric slots only; attempting to increment non-numeric targets raises a runtime type error.

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
- Rebind chains can fan out directly: `=user.{name, email}.trim()` trims both slots and evaluates to `["trimmedName", "trimmedEmail"]`.

### 3.9 Deep object merge `+>` and `+>=`

#### Surface

```shakar
C = A +> B
cfg +>= env
```

#### Semantics

- Object vs object: recursive key-wise merge; RHS wins on conflicts.
- If either side at a path is not an object, RHS replaces LHS at that path.
- Arrays, sets, strings, numbers: replaced wholesale (no deep behavior).
- Not commutative: `A +> B` may differ from `B +> A`.

#### Precedence

- Additive tier with `+ - ^` (object algebra family).

#### Errors

- `+>=` requires LHS to be an object; otherwise error. `+>` always yields a value per rules above.

### 3.10 Object index with default

#### Surface

```shakar
count = hits["/home", default: 0]
port  = env["PORT", default: "3000"]
value = cfg.db[host, default: "localhost"]
```

#### Semantics
- Objects only; using `default:` on a non-object receiver is a **static error**.
- If key exists ⇒ return stored value.
- If key missing ⇒ return the `default:` expression (no throw).
- `default:` is a named argument to the `[]` operator; it doesn’t modify storage.
- The `default:` expression is **evaluated only if** the key is missing (lazy).
- Arrays and strings: `default:` is **not accepted**; indexing semantics unchanged (OOB still throws).
- Works in selector chains: `cfg["db", default: {}]["host", default: "localhost"]`.

#### Method alternative

- `m.get(key, default: v)` remains valid.

### 3.11 Placeholder partials `?`

#### Surface

```shakar
between = inRange(?, 5, 10)              # (x) => inRange(x, 5, 10)
mix     = blend(?, ?, 0.25)              # (a, b) => blend(a, b, 0.25)
xs.map&(get(?, id))                      # (x) => get(x, id)
```

#### Desugar

```text
inRange(?, 5, 10)         ⇒   (x) => inRange(x, 5, 10)
blend(?, ?, 0.25)         ⇒   (a, b) => blend(a, b, 0.25)
get(?, id)                ⇒   (x) => get(x, id)
```

#### Rules

- A `?` appearing among the **immediate arguments** of a call expression creates a lambda with **one parameter per hole**, left-to-right.
- Each hole is a distinct parameter; `blend(?, ?, 0.25)` receives two parameters `(a, b)`.
- Works in both free and method calls; named args allowed (holes in named values participate in left-to-right order).
- **No immediate invocation**: `blend(?, ?, 0.25)` yields a function; invoking it inline (e.g., `blend(?, ?, 0.25)(1, 2)`) is illegal—store or pass the partial first.
- **No ternary conflict**: `?` is recognized **only inside a call’s argument list**; elsewhere, `?` is not a valid expression token.
- **Style**: allowed with a single hole, but prefer `&` path-lambdas for one-arg cases: `xs.map&(.trim())`.

## 4) Implicit subject `.` - anchor stack

### 4.1 Where `.` comes from (binders)
- **Statement-subject assign** `=LHS<tail>` (at **statement start**): within the statement, `.` = **old value of `LHS`**; after evaluation, assign **`LHS = result`**.

`.` exists only inside constructs that **bind** it:

- **Apply-assign** `LHS .= RHS`: inside `RHS`, `.` = **old value of `LHS`**; result writes back to `LHS`.
- **Subjectful loop** `for Expr:` / `for[i] Expr:`: in the loop body (per iteration), `.` = **current element** (and `i` = view index if present).
- **Lambda callee sigil** (e.g., `map&(...)`): inside the lambda body, `.` = **the parameter**.
- **`await(expr)` / `await expr`**:
  - Trailing body: `.` = **resolved value**; returns the body’s result.
  - No trailing body: returns the resolved value (no binder).
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

- **No-anchor `$`**: `$expr` evaluates but does **not** create/retarget an explicit subject. Leading-dot chains remain anchored to the prior subject. Implementation-wise it simply saves and restores the current anchor around the expression, so inner leading dots still see the previous subject while siblings ignore the result.
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
- **Invalid statement-subject**: `=LHS` with no `<tail>` (no effect) — use `.=` or provide `<tail>`. `=.trim()` is illegal (free `.`).
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

user = { profile: { contact: { name: "  Ada " } }, flags: [] }
=(user).profile.contact.name.trim()  # rewrites the whole object, not just the nested name
assert user == "Ada"

=(user.profile.contact.name).trim()  # same as user.profile.contact.name = user.profile.contact.name.trim()

xs = [" A ", "b"]
=xs[0].trim()
assert xs[0] == "A"

# Errors
# ERROR: =user.name     # missing tail
# ERROR: =.trim()       # free '.' cannot be the subject
# ERROR: =(user + other).trim()   # grouped head must be a pure lvalue
```

```shakar
# Paren pop resumes prior anchor
(a := { c: &() true }); a and (true) and .c()   # ⇒ a and true and a.c()

# Apply-assign then sibling leading-dot sees updated LHS
xs = ["  a", "b  "]
(xs[0] .= .trim()) and .hasPrefix("a")
```

```shakar
# Selector as value vs selector:
sum := 0
for i in `0:3`: sum += i           # 0+1+2+3
ix, vals := [], []
xs = [10,11,12,13]
for[i] xs[1:3]: ix.append(i); vals.append(.)
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

**Interaction with statement-rebinding `=`:**
```shakar
=a.lower() and .allLower()
# ⇒ tmp0 = a.lower(); a = tmp0; tmp0 and a.allLower()
```

---

### 5.1 Assignment forms — `:=` vs `=` (v0.1)

**Intent split**
- `name := expr` — **introduce & yield**: creates a new binding in the **current lexical scope** and **returns** the assigned value. Error if `name` already exists **in the same scope** (use `=` to update).
- `name = expr` — **update only**: assigns to an **existing** binding. Statement-only: using `=` where a value is required is a compile-time error. Error if `name` does **not** exist (use `:=` to introduce).

**Destructuring**
- Arrays: `[head, ...rest] := xs`
- Objects: `{id, name} := user`
Updates use `=` on existing lvalues: `user.name = "New"`.

**Chaining**
- Disallow `x := y := 0` in v0.1 (write as two assignments or a destructure).

**LValues**
- Member/index updates are normal updates: `obj.field = v`, `arr[i] = v`.
- **Selector lists and slices may not appear on the LHS** in v0.1 (no multi-write or slice assignment).
## 6) Defaults, guards, safe access (committed)
- **`or=`** and **statement-subject** `=x or y` (statement head) desugar to `if not …: … = …`.
- **`?ret expr`** early-returns if expr is truthy.
- **Postfix conditionals**: `stmt if cond` (run when truthy) and `stmt unless cond` (run when falsey). When the statement is a walrus (`name := expr if cond`), the runtime assigns `name := nil` before evaluating the guard, so a failing condition leaves the binding at `nil` without running `expr`.
- **Nil-safe chain**: `??(expr)` turns a deep deref/call chain into a nil-propagating expression.

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

### Defer

Purpose: delay cleanup work until the current block finishes (including early exits).

**Surface**
```shakar
defer closeHandle()
defer log("done") after cleanup
defer cleanup: closeHandle()
defer cleanup:
  closeHandle()
  log("closed")
defer final after (prepare, audit):
  close()
  report()
```

- Without a handle, defers run strictly in **LIFO** order when the surrounding block completes, mirroring Go semantics.
- `handle` (the identifier between `defer` and the colon) labels a defer so that other defers in the same block can depend on it.
- `after (h1, h2, …)` is optional. When present, the defer runs only after all listed handles finish. Handles can be declared later in the source; scheduling resolves the dependency graph before execution.
- Handles are per-block; reusing a handle in the same block raises `ShakarRuntimeError`.
- Referencing an unknown handle or forming a cycle (direct or indirect) raises a runtime error.
- Bodies come in two shapes:
  - **Simple call** (no colon): `defer closeHandle()` — only valid when there is **no handle**. You may add a trailing `after` here (`defer log("done") after cleanup`), but the defer itself stays anonymous.
  - **Block body** (colon + inline or indented block): `defer cleanup: closeHandle()` or `defer cleanup:\n  closeHandle()`
- Any defer that declares a handle or uses `after` **must** use the colon form unless it is the anonymous simple-call shape above. In the block form, place `after` in the header: `defer cleanup after close: ...` (wrap the handles in parentheses if you need to specify zero or multiple dependencies).
- Block bodies execute in their own child frame that inherits the surrounding scope and subject `.`. Nested defers inside the body stack independently and flush before the parent defer completes.
- `defer` may only appear inside executable blocks (functions, guards, loops, etc.); using it at the absolute top level raises `ShakarRuntimeError`.

**Examples**
```shakar
fn run():
  defer cleanup: conn.close()
  conn := connect()
  use(conn)

fn ordered():
  defer second after first: log("second")
  defer first: log("first")
# prints "firstsecond"
```

---

## 7) Punctuation guards (drop `if/elif/else` noise)
> **Terminology:** *Guard* is the construct. A `Expr ":" Body` is a *guard branch*. Multiple branches separated by `|` form a *guard group*. *Chain* refers only to **postfix chains** (field/call/index).

**Multi-line guard**
```shakar
ready():
  start()

| retries < 3:
  retry()

|:
  log("not ready")
```

**One-line guard**
```shakar
ready(): start() | retries < 3: retry() |: log("not ready")
user and .active: process(user) |: log("inactive")
??(conn.open): use(conn) |: log("connection closed")
```

**Binding of `|:` (nearest-else rule):** After a head’s first `:`, any `|`/`|:` at the same bracket depth binds to the **innermost open guard** (nearest head). Wrap inner guards in `(...)` or `{...}` if you want the following `|:` to bind to an outer guard.
**Rules**
**Alias:** `||` and `||:` are accepted as input in one-line guards and are normalized by the formatter to `|` / `|:`. They are recognized only after the first `:` and only at bracket-depth 0.
- **Head:** `Expr ":"` starts a guard group.
- **Or-branch (elif):** same indent, `| Expr ":"`.
- **Else:** same indent, `|:` (no space). Else is optional.
- **Disambiguation:** guard heads that end with a dict literal should still be wrapped: `({k:v}): …`. Function calls already use parentheses (`send(to: "Ali"): …`), so named arguments stay unambiguous.
- **Don’t mix** punctuation and keywords within **the same guard group**. Keywords remain available (`if/elif/else`).

Desugars 1:1 to `if/elif/else`.

---

- **`await` one-liners are statements, not guards.** `await expr: body` and `await(expr): body` parse as `awaitstmt` so the trailing body binds to `await` (not to a guard). Parenthesized `(await f()): …` may parse as a guard head; allowed for now and style-lintable.

## 8) Control flow (loops & flow keywords)
- Iterating the `nil` **literal** is a compile-time error; iterating a variable that evaluates to `nil` is a **no-op**.

### Concurrency — `await()` / `await[any]` / `await[all]`

**Single await** — both forms are equivalent:

```shakar
v := await(expr)
v := await expr

# With trailing body (binder `.` = resolved value)
await(fetchUser(id)): show(.)
await fetchUser(id): show(.)
```
- **Clarity**: prefer `await(expr)` when adjacent to binary operators.

**Single await** — `await(expr)` waits for a single future/task. Optional trailing body runs with `.` bound to the resolved value.

```shakar
# without body: just get the value
user := await(fetchUser(id))

# with trailing body: use `.` inside
await(fetchUser(id)): show(.)
```

- `timeout` default unit: **milliseconds**; supports suffixes `ms`, `s`, `m`.
- Stdlib helper: `sleep(ms)` returns an awaitable; use `await sleep(100)` or drop it inside `await[any]/await[all]` arms to stage async workflows without custom futures.

**Goals:** one `await` per construct, no repetition; clear and cancellable.

**`await[any]`** (two exclusive shapes)

A) **Per-arm bodies (no trailing body):**
```shakar
await[any](
  fetchUser(id): show(.),
  fetchFromCache(id): show(.),
  timeout 200): showFallback()
)
```
- Heads start concurrently. The **first successful** arm runs its body with **`.` = that arm’s result**. Others are canceled.
- Returns that body’s value.

B) **Trailing body (no per-arm bodies):**
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

**No trailing body**: returns an object (for programmatic use):
```shakar
r := await[any]( user: fetchUser(id), cache: fetchFromCache(id), timeout 200 )
# r = { winner: "user"|"cache"|"timeout", value: <result> }
```

**`await[all]`** (trailing body or object form)

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
- Do **not** mix per-arm bodies and a trailing body in the same call (linted).
- `.` refers to the value defined by the construct as described above; it does **not** capture outer `.` from lambdas.
- **for-in**: `for x in iterable: block`
- Selector literals as values: ``for i in `0:<n`: …``  # iterate a numeric selector
- Selector lists (indexed view): `for[i] xs[ sel1, sel2, … ]: …`  # iterate the concatenated view; i is view index
- Note: the same `a:b` syntax is a **selector** inside `[]`, and a **Selector value** elsewhere.
  - Iteration over `nil` is **no-op** (safe): `for x in maybeNil: …` does nothing if `maybeNil` is nil.
- **break**, **continue**: loop controls.
- **return**: function return.
- **defer** / **using**: resource management (§11).
- **while/until** ❓: not in v0.1 unless demanded; `for` over `selector literal` covers most cases.

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
- **Objects:**
  ```shakar
  for[k] m:
    # '.' is m[k]; 'k' is the key use dot when key is an identifier; bracket for all keys
    use(k, .)

  for[k, v] m:
    # both names bound; '.' is 'v' (the value)
    use(k, v)

  for (k, v) in m:
    # destructuring form; equivalent to for[k, v] m:
    use(k, v)
  ```

**Rules:**
- `.` is **per-iteration** and **does not leak** outside the loop body.
- Nested subjectful constructs (lambdas `map&`, apply-assign `.=`, `await[any]` arm bodies) **rebind** `.` inside their own scopes.
- A nested `for` **rebinds** `.`; if you need the outer element, name it once (`outer := .`) before entering the inner loop.
- Works with any iterable (`Array`, `Set`, `Map`, `Object`, custom iterables).
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
- `for[k] m:` / `for[k, v] m:` — object; `k` = key, `.` = value (or `v` if bound). Equivalent destructuring form: `for (k, v) in m:` (parentheses optional when binding only identifiers: `for k, v in m:`).
- `for … in <selectors>:` — iterate a **selector list** (selector values (ranges), slices, indices).

**Hoisted binders (`^name`) — final v0.1 semantics**
- **Meaning:** Bind this loop variable to an **outer-scope** binding; if none exists, **create** it in the immediately enclosing (non-loop) scope with initial `nil`.
- **Per-iteration:** Assign the current loop value to the hoisted binding each iteration. If the loop runs 0 times, a pre-existing binding is unchanged; a newly hoisted binding remains `nil`.
- **Where:** Works in any binder list: `for[^i] 0:n: …`, `for[^k, ^v] m: …`, `for[^i] xs[ sel1, sel2 ]: …`.
- **Errors:** It is illegal to use both `^x` and `x` for the same name in one binder list, or to repeat the same `^name` within a binder list.
- **Closures:** Inner closures capture the **single hoisted binding** (mutations visible across iterations).

Examples:
```shakar
# Hoist-and-declare
for[^idx] `0:4`:
  use(idx)
print(idx)     # => 3; if loop didn’t run, idx == nil

sum := 0
for[j, ^sum] arr:
  sum = sum + .
# sum visible after loop; j is loop-local
```

- **for-in**: `for x in iterable: block`
  - Selector values: `` for i in `0:n`: … ``, `` for i in `0:<n`: … ``
  - Iteration over `nil` is **no-op** (safe): `for x in maybeNil: …` does nothing if `maybeNil` is nil.
- **break**, **continue**: loop controls.
- **return**: function return.
- **defer** / **using**: resource management (§11).
- **while/until** ❓: not in v0.1 unless demanded; `for` over `Selector literal` covers most cases.

---

## 9) Destructuring (no braces)
**Broadcast note (to finalize):** `a, b := 1` broadcasts the single RHS value to each LHS target, evaluating the RHS **once** and assigning aliases where applicable. LHS may be flat identifiers in v0.1; nested patterns TBD. Error messaging MUST distinguish (a) illegal `a = 1, 2` single-target with comma on RHS (site-local pack not allowed without a comma LHS), from (b) valid broadcast with `:=`.


```shakar
x, y := get_pair()
name, age := user.profile
k, v := pair
```

- **LHS requires a `pattern_list` (2+).** `a = 1, 2` is an error; use `a, b = 1, 2`. Applies to both `=` and `:=`.

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
byId  = { .id: . over users if .active }
```

**Explicit binding (when needed):** `bind` (does not remove `.`). You can also rely on **implicit head binders** when the head uses free, unqualified names.
```shakar
admins = [ u.name over users bind u if u.role == "admin" ]
pairs  = { k: v over obj.items() bind k, v }
```

**Desugar (list):**
```
lines.map&(.trim()).filter&(.len > 0).to_list()
```

---

- **`bind` scope:** names introduced by `bind` are visible **throughout the comprehension** (head, filters, body), but not outside it.
- **Binder list sugar on `over`:** `over[binders] src` is sugar for `over src bind binders`. Objects: one binder yields the key, two binders yield key and value. It does not remove `.` in the body.

**Examples (binder list sugar):**
```shakar
pairs = { k: v over[k, v] map.items() }
names = [ u.name over[u] users if u.role == "admin" ]
sums = [ a + b over[a, b] aAndB ]
```

- **Illegal combination:** `over[...] Expr bind ...` is not allowed. The formatter drops the redundant `bind` and keeps the bracket form.

### Implicit head binders (objects / lists / sets)
In a comprehension head, free, unqualified identifiers are treated as temporary binders so you don’t have to write a separate `bind` list when the intent is obvious.

**Where it applies**
- **Object heads:** `{ key : value [if pred] over iter }` — free idents in `key`, then `value`, then `pred` bind to components of each element from `iter` in **first-use order**.
- **List heads:** `[ elem [if pred] over iter ]` — free idents in `elem` then `pred` bind likewise.
- **Set heads:** `{ elem [if pred] over iter }` — same rule as lists.

**Hard gates (so it can’t leak)**
- Only in **comprehension heads** (and optional `if` guards). Not in loops.
- Identifiers must be **free** at that site and **unqualified** (plain names). If a name resolves in the surrounding scope, it is **not** a binder.
- Ignore occurrences **inside nested lambdas** within the head/guard.
- This rule does **not** remove or alter the subject `.` — you can still use `.` in the head as usual.

**Arity check**
- Let N = number of distinct head-binders; M = arity of the element.
- Require N ≤ M (checked statically when possible, otherwise at runtime with a one-step unpack).
- Error example: “Head introduces 2 names (k, v) but source produced a 1-component value.”

**Examples**
```shakar
{ k: v over dict.items() if v > 0 }
[ f(k, v) over dict.items() ]
{ id over users if id > 0 }      # set comp
```
## 11) Named-arg calls

All invocations use parentheses, even when passing named arguments. This keeps the syntax unambiguous in guards and postfix contexts.

```shakar
fn send(to, subject, body): …
send("bob@x.com", subject: "Hi", body: "…")
```

## 12) Strings & performance (critical)

- **Interpolation syntax**
  - Use `{expr}` inside a single- or double-quoted string to splice the value of any expression into the final string. Examples: `"Hello {user.name}! score={user.score}"` and `'user: {user.name}'`.
  - Braces used literally must be doubled: write `{{` for `{` and `}}` for `}`. Single `}` outside of an interpolation is an error.
  - Expressions are parsed with the normal grammar (so nested selectors, ternaries, etc., all work) and inherit the surrounding scope/subject. Empty braces (`{}`) are illegal.
  - Evaluation order is left-to-right; each expression is evaluated and stringified immediately (`ShkString` contributes its `.value`, numbers render numerically, everything else goes through the standard `stringify` helper).
  - Strings with no `{` remain simple literals; interpolation is only enabled once a single brace-insertion appears.

## 12.5) Slicing & selector lists
- **Selector literals as values** use backticks with the body (e.g., `` `lo:<hi` `` or `` `1:<5, 11:15:2, 20` ``). Indexing continues to use brackets: `xs[lo:<hi]`.
- **Selector literals as values vs slices:** selector literals as values do not clamp; slices inside `[]` still clamp; index selectors throw on out-of-bounds.
- **Selector literal interpolation**: inside backticks, use `{expr}` to embed any expression as a bound or index, e.g., `` `1:{hi-1}` `` or `` `{lo+1}:<{hi}` ``. Top-level separators `:` and `,` split slices/items; `{…}` shields internal punctuation. Use **bare identifiers/numbers** without braces when possible (e.g., `` `lo:<hi` ``). **Parentheses are not used at top level** inside selector literals; if you need grouping, put it **inside** `{…}`.

## 12.6) Regex helpers

- Literal: `r"..."/imsx` (flags optional).
- Methods on regex value: `.test(str) -> Bool`, `.match(str) -> Match?`, `.replace(str, repl) -> Str`.
- Timeouts and engine selection are library concerns (not syntax).

**Slices (arrays & strings):**
```shakar
a[i:j]         # half-open: i ≤ k < j
a[i:j:step]    # step ≠ 0; supports negative steps for reverse
a[:j]          # from start
a[i:]          # to end
a[:]           # whole view
```
- **Negative indices** allowed (`-1` is last).
- **Reverse requires explicit negative step:** `a[i:j:-1]`. We **do not** auto-reverse when `i > j`.
- **Indexing** `a[i]` **throws** if out-of-bounds.
- **Slices** **clamp** to `[0:len]` and never throw; an inverted range with positive step yields `[]`.
- **Strict slicing**: `slice!(a, i, j, step?)` throws on any out-of-bounds (no clamping).

**Selector lists** (multiple selectors inside `[]`, comma-separated):
```shakar
xs[:5, 10:15:-2]      # concat first five with a reverse slice
xs[0, 3, 5:]          # specific indices plus a tail
xs[1 : .len-1]        # drop first and last (half-open excludes last)
```
- The overall result is the **concatenation** of each selector’s result, in order.
- **Selector values inside `[]`**: backtick selector literals are valid items and are **flattened** (spliced) into the selector list at that position.
- **Examples**:
  - `xs[`1:10`]`  ≡  `xs[1:10]`
  - `xs[`1:10, 11:15:2`, 20]`  ≡  `xs[1:10, 11:15:2, 20]`
  - `customSelector := `1:10, someIdx`; xs[customSelector]`  # define & reuse a selector value
- **Type checks**: each item must evaluate to either an integer index or a selector value; otherwise it is a type error.
- **Per-selector steps** are allowed.
- **Index selectors**: OOB → **throw**.  **Slice selectors**: **clamp**.
- Inside each selector expression, **`.` = the base** (e.g., `xs`) for that selector only.
**LHS restrictions (v0.1):** selector lists and slices are **expression-only**; they cannot be used as assignment targets. Use single index/property updates instead.
- Immutable UTF-8 with **views** for slices/trim/drop and **ropes** for concat.
- **Unique-owner compaction** avoids substring leaks (thresholds: keep <¼ or >64KiB slack).
- Controls: `.own()`, `.compact()`, `.materialize()`.
- Unicode indexing counts **characters**; `bytes` view for byte-wise operations.
- FFI always materializes exact bytes (+NUL) and caches pointer.

What this buys:
```shakar
=s[1:]            # view or in-place compact when unique
=s.trim()         # view (offset/len tweaked)
=path[.len-1]     # view of last segment
```

For heavy edits, use `TextBuilder/Buf` and `freeze()` back to `Str`.

---

## 13) Object model (objects and descriptors)

- **Object**: map `name → Slot`.
- **Slot**: `Plain(value)` or `Descriptor{ getter?: fn, setter?: fn }`.
- **Property read**: Plain → value; Descriptor.getter → call with implicit `self`.
- **Property write**: Plain → set; Descriptor.setter → call with `self, value`; strict mode can forbid setting Plain when a descriptor exists.
- **Methods**: `obj.m(args)` sets `self=obj` for the call frame.
- **Contextual `get/set`** inside object literals desugar to `Descriptor` slots.
- **Decorator-based getters/setters** set flags or return descriptor wrappers so assignment installs the right slot.
- Getter must be **nullary**; setter **arity 1**.

To pass a callable for a getter: `&(obj.prop())`.

---

### Objects — unified literal
Shakar uses a single object literal `{ ... }`. Prior distinctions between records and maps are removed. The validator tags objects as `closed` or `open` based on keys and spreads.

**Keys**
- Identifier: `a: expr`
- String: `"a-b": expr`
- Computed: `[expr]: expr`

**Getters and setters**
- `get key: expr` or an indented block after `:`
- `set key(x): expr` or an indented block after `:`
- Getter arity 0; setter arity 1; implicit `self` is the object.

**Access**
- `obj.key` only for identifier keys (participates in method fusion).
- `obj[expr]` and `obj["k"]` for all keys.

**Shape tagging**
- `closed`: all keys are identifiers and there are no computed keys or spreads. Optimizable hidden shape. Duplicate identifier keys are errors.
- `open`: otherwise. Dictionary path. Last write wins on duplicates.

**Validator**
- Enforce getter/setter arities.
- In `closed` objects: forbid duplicate identifier keys.
- Require bracket access for non-identifier keys.

**Printer**
- Preserve order; print identifier keys bare; quote string keys; keep computed keys in `[ ]`.
- Allow block bodies after `:` for getters, setters, and field values.

**Examples**
```shakar
user = {
  id: 1,
  name: "Ada",
  [1+2]: 3,
  get size: 1,
  set size(x): x
}

user.name       # dot for identifier
user["name"]     # bracket also ok
user[1+2]       # computed
```
## 14) Decorators (no HOF boilerplate) & lambdas
- Decorators & hooks are **core** in v0.1. Decorators use a dedicated `decorator` form so that wrappers do not require manual higher-order plumbing.
- **Decorator definitions**: `decorator name(params?): body`. Inside the body the **next callable** is exposed as `f` and the current **positional arguments** are available as a mutable `args` array. Mutate `args`, reassign it entirely, or short-circuit by `return`ing a value. When the body finishes without a `return`, the runtime implicitly executes `return f(args)` so you always get the original call unless you opt out.
- **Decorator application**: prefix `fn` definitions with `@decorator` lines. Expressions are evaluated **top to bottom**, but the decorator closest to the `fn` wraps the function **first** (`@outer` runs after `@inner`). Parameterized decorators behave the same way (`@memoize(256)`), and bare `@decorator` is shorthand for calling a parameterless decorator. Decorator expressions evaluate to either a `decorator` value or a configured decorator instance; anything else is rejected.
- **`args` semantics**: `args` is an ordinary `Array` of Shakar values. Use `args[0]` / `args[1]` to inspect or tweak inputs, or rebind `args := [new, args[1], …]` to completely replace them. The object lives in the decorator’s own function frame, so you may define helper locals or call other functions before deciding whether to call `f(args)`.
- **Lambdas**: `map&(.trim())` (single arg implicit `.`), `zipWith&[a,b](a+b)` (multi-arg). `&` is a **lambda-on-callee** sigil.

---

### Amp-lambda: implicit parameters at known-arity call sites
**When it triggers.** Only when `&( body )` is passed to a call site whose arity **N** is known (e.g., `map`=1, `filter`=1, `zipWith`=2).

**How parameters are inferred**
- Collect **free, unqualified identifiers** used as **bases** in `body` (e.g., `a`, `a.lower()`, `a[0]`, `f(a)`), scanning left→right. Occurrences inside nested lambdas are ignored.
- If the body contains the **subject** `.` anywhere, inference is disabled for that lambda. Use either `.` **or** implicit parameters, not both.
- If #distinct_free > N: hard error (name the extras).
- If #distinct_free < N: behavior depends on the callee’s `implicit_params` policy: `exact` → error; `pad` → pad with anonymous parameters (ignored); `off` → error.
- If a name resolves in the surrounding scope, it **captures** that outer value (it is not a binder). Use an explicit parameter list if you intend to shadow: `&[x](...)` or `&[x,y](...)`.

**Examples (valid under current call shape: lambda-first)**
```shakar
zipWith&(left + right)(xs, ys)      # infers &[left, right](left+right)
pairs.filter&(value > 0)            # unary site → &[value](value>0)
lines.map&(.trim())                 # uses subject — no inference
```
**Errors**
- “Cannot mix `.` subject with implicit parameters; choose one style.”
- “Implicit parameters found: a,b,c but callee expects 2.”
- “`x` is already bound here; implicit parameters require free names (use `&[x](...)` to shadow).”

**Notes**
- No grammar changes; this is a validator/desugar step that fills the param list in the canonical `amp_lambda` node.

#### Callee policy for amp-lambda implicit parameters
**Declaration (decorator):** Use `@implicit_params(policy)` on the callee definition to control inference at its call sites. Valid `policy` values are `exact` (**default if omitted**), `pad`, and `off`.

```shakar
@implicit_params('exact')   # default; explicit for clarity
fn zipWith(xs, ys, f) { ... }

@implicit_params('pad')
fn mapWithIndex(xs, f) { ... }

@implicit_params('off')
fn reduce(xs, init, f) { ... }

struct Dict {
  @implicit_params('pad')
  fn mapWithKey(self, f) { ... }
}
```

Known-arity callees declare an `implicit_params` policy that controls whether implicit parameters may be inferred and whether missing parameters may be ignored:

- `exact` (**default**): the number of distinct free, unqualified names in the body must equal the callee arity **N**.
- `pad`: the body may use fewer than **N** names; missing positions are filled with anonymous parameters that are not usable in the body.
- `off`: implicit parameters are disabled; write an explicit parameter list (`&[params](...)`).

### Anonymous `fn` expressions

- `fn(params?): body` is an expression that produces a callable value. Body syntax matches named `fn` (colon + inline or indented block).
- Example: `inc := fn(x): { x + 1 }; inc(5)` → `6`.
- Zero-arg IIFE sugar: `fn(()): body` defines a zero-arg anonymous function and immediately invokes it (desugars to `(fn(): body)()`).
- Expression bodies without braces execute when the function is invoked, not when the `fn` literal is evaluated, so `handler := fn(): print("awd")` simply stores the function until you call `handler()`. Braces remain the preferred way to wrap multi-statement bodies.

**Rules that still apply**
- Only active at **known-arity** call sites.
- **No mixing** with the subject `.` inside the same lambda body.
- Names that resolve in the surrounding scope are **captures**, not implicit parameters.

**Examples**
```shakar
# exact (default)
zipWith&(l + r)(xs, ys)            # ok: 2 names, arity 2
zipWith&(l)(xs, ys)                # error: expects 2 (policy = exact)

# pad (callee opted in)
mapWithIndex&(item * 2)(xs)        # ok: arity 2; pads the second param (ignored)
filterWithKey&(key.starts_with("x"))(dict)  # ok: uses only key

# off (callee opted out)
reduce&(acc + x)(xs, 0)            # error: implicit params disabled; write &[acc, x](...)
```
## 15) Errors, catch, assert, dbg, events

- **Catch expressions**:
  ```shakar
  val = risky() catch err: recover(err)
  val = risky() @@ err: recover(err)          # @@ is shorthand for catch
  fallback = risky() catch: "ok"              # omit binder to rely on .
  user := risky() catch (ValidationError, ParseError) bind err: err.payload
  ```
  `catch` (and `@@`) evaluate the left side; if it succeeds, the original value is returned unchanged. If a `ShakarRuntimeError` (or subclass) escapes, the handler runs instead and its value becomes the result. Inside the handler:
  - Without a type guard, `catch err:` binds the payload (omit `err` to rely solely on `.`). With a guard, write `catch (Type, …) bind name:` to bind while filtering.
  - `.` (the subject) always points to the same payload object.
  - The payload exposes `.type` (exception class name), `.message` (stringified message), `.key` for `ShakarKeyError`, and `.method` for `ShakarMethodNotFound`.
  - Optional `(TypeA, TypeB)` parentheses immediately after `catch` restrict the handler to those exception types; follow them with `bind name` if you need a payload alias. Non-matching errors bubble out automatically.
  - Bare `catch` (no binder, no parentheses) is a catch-all.
  - `throw` inside the handler rethrows the current payload; `throw expr` raises a new error from `expr`.
- **Catch statements**:
  ```shakar
  risky() catch err: { log(err.message) }
  risky() catch: {
    log("oops")
    alert(.type)
  }
  ```
  The statement form mirrors the expression semantics but discards the original value (always producing `nil`). The body may be an inline `:` block or an indented block, and it only executes if the left-hand expression fails.
- **assert expr, "msg"** raises if falsey; build can strip or keep.
- **throw [expr]** re-raises the current catch payload when the expression is omitted, or raises a new `ShakarRuntimeError` from the provided value (strings become messages, structured objects set `.type/.message`).
- `error(type, message, data?)` (stdlib helper) builds a tagged error object. `throw error("ValidationError", "bad", info)` produces a payload with `.type == "ValidationError"`, `.message == "bad"`, and `.data == info`, enabling `catch (ValidationError) bind err:` guards.
- **dbg expr** logs and returns expr; can be stripped in release.
- **Events**: `hook "name": .emit()` ⇒ `Event.on(name, &( .emit()))`

---

## 16) Keywords (v0.1)

**Reserved (cannot be identifiers):**
`and, or, not, if, elif, else, for, in, break, continue, return, assert, using, defer, catch, decorator, decorate, hook, fn, get, set, bind, over, true, false, nil`

**Contextual:**
- `for` in comprehensions (`[...] for src …` as alias for `over`)
- `get`/`set` only **inside object literals**

**Punctuation (syntax, not keywords):**
- `.=` apply-assign
- `|` and `|:` at line start for guard chains
- `?ret` (statement head)
- `??(expr)` nil-safe chain
- `:=` walrus

---

## 17) Type model (pragmatic dynamic)

- **Numbers**: `Int` (64-bit signed; **overflow throws** in v0.1) and `Float` (64-bit IEEE). Arithmetic follows Python-like precedence; `/` produces float, `//` floor-div produces `Int`.
- **Bool**: `true/false`
- **Str**: immutable UTF-8
- **Array**: ordered, dynamic
- **Object**: key→value map (string keys); descriptors for getters/setters
- **Func**: user/native functions
- **Selector literal**: iterable numeric range
- **Nil**: null value

Type predicates/methods live in stdlib (`isInt(x)`, `typeOf(x)`), not as syntax.

---

## 18) Tooling & ecosystem

- **Formatter** (`shk fmt`): canonical spacing; can normalize punctuation vs keywords per project style.
- **REPL** (`shk repl`): auto-print last value; `:desugar` to show lowerings.
- **Linter** (initial rules):
  - Suggest `a or= b` / `=a or b` over `a = a or b`.
  - Warn on long implicit `.` chains; suggest `bind`.
  - Don’t mix punctuation and keyword guards in the same chain.
  - Warn on tiny views over huge bases; suggest `.compact()`/`.own()`.
- **Diagnostics with fix-its**:
  - “`.` cannot stand alone; start from a subject or use `bind`.”
- **AI-friendly**:
  - `shk --desugar` and `--resugar` for round-trips.
  - Machine-readable feature manifest (`shakar.features.json`) describing allowed sugars/style picks.
  - Lenient parse `--recover` to auto-fix common near-misses.

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

Project config (e.g. `shakar.toml`) with **per-module overrides**:
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
- **Terminology**: `IndentBlock` means the off-side indented form (`NEWLINE INDENT … DEDENT`). `InlineBody` is either a single `SimpleStmt` or a braced inline block `{…}`. Punctuation guards remain **block-only** via `IndentBlock`.

- **Unary `+` is invalid**: write `x`, not `+x`. Use explicit conversions (`int(x)`, `float(x)`) instead of operator coercions.
- **Selector literals (backticks) — interpolation style**: prefer bare names/numbers for bounds; require `{…}` for any non-trivial expression. Do **not** use top-level `(…)` in selector literals; use `{…}` instead. Avoid spaces just inside braces: `{a+1}`, not `{ a + 1 }`.
- **Discourage inline backticks inside `[]`**: `` xs[`1:10`] `` is allowed but discouraged; prefer flatted form `` xs[1:10] `` when the literal is the only selector and binded selectors elsewhere `` sel := `5:10`; xs[1, sel] ``. The linter should warn.
- **Discourage inline backticks in collection literals**: `` [1, `1:10`, 20] `` is allowed but discouraged; prefer `[1, sel, 20]` with `` sel := `1:10` ``.
  * **Exception**: short REPL snippets and small examples may use inline backticks for brevity.
- **Selector values in `[]`**: when a backtick selector literal appears inside `[]`, the formatter should inline its body (flatten), preserving order.
- **Selector values in collections**: backtick selector literals are ordinary values in arrays/objects/sets; **no flattening** occurs in collection literals.
- **Selector literal values use backticks** around the selector **body** (no brackets inside). Example: `` `1:<5, 11:15:2, 20` ``.
- **Brackets `[]` are not used for selector values**. Use brackets only for indexing, binders (e.g. `for[i]`), and array literals.
- **Commas in lists**: parser accepts both `.{a,b}` and `.{a, b}`. Formatter emits a single space after each comma across all comma-separated lists (field fan-out `.{...}`, argument lists, binder lists, patterns). No space before commas; no spaces around braces.
- **Field fan-out braces**: prefer `user.{a, b}` (no space between `.` and `{`). Trailing comma only when the list is multiline.
- **Map index with default**: prefer `m[key, default: expr]` with exactly one space after the comma and no spaces around `:`. Multiline default only if the entire index breaks across lines.
- **Placeholder partials `?`**: for single-hole cases, prefer `&` path lambdas (e.g., `xs.map&(.trim())`). Use `?` when there are **2+ holes** (e.g., `blend(?, ?, 0.25)`). Avoid mixing `&` and `?` within the same call; when readability suffers, switch to a named-arg lambda.
- **Style — head binders:** prefer **implicit head binders** when the head’s names are free and unqualified; use `over[...]` or `bind` when you need to shadow, disambiguate arity, or improve readability. Do **not** mix `over[...]` and `bind` in the same head.
- **Tokenization -- ``!in``:** write as a single token with no internal space: `a !in b` (not `a ! in b`). Parser treats `! in` as unary `!` then `in`.
- **Tokenization -- selector interpolation braces:** Inside a backtick selector literal, `{` begins interpolation; the interpolation region is parsed as a normal expression with balanced-brace counting; braces contained in string or character literals or comments do not affect the balance. The selector literal ends at the next backtick after the selector grammar completes. Comments are not permitted inside selector bodies. There are no escape sequences in selector bodies.
- **Style -- Comparison comma-chains with `or`:** When a comma-chain switches to `or`, prefer either repeating `, or` for each subsequent leg (e.g., `a > 10, or == 0, or == 1`) or grouping the `or` cluster in parentheses (e.g., `a > 10, or (== 0, == 1)`). This improves readability; semantics are unchanged (joiner is sticky; a comparator is required after `or` to remain in the chain).
- **Flag free `.`**: bare `.` outside an active binder/anchor is an error.
- **Auto-paren guard heads with `:`**: when a guard head contains `:`, the formatter inserts parentheses; style check warns if missing.
- **Bullets**: use `-` everywhere; do not mix `*` in lists.
- **Tight lists**: no blank lines between sibling bullets. Within an item, allow one blank line before a code fence or a continuation paragraph; continuation lines and fences are indented by two spaces.
- **Spacing normalization**: remove trailing spaces; collapse 3+ blank lines to one; ensure exactly one blank line after `##`/`###` headings; remove blank lines between sibling bullets at the same indent.
- **Await any/all body size**: prefer `InlineBody` only for a single simple statement; use `{ ... }` or an indented block for multi-stmt bodies.

**Feature gate note (bitwise_symbols):**
- Numeric bitwise operators `& | ^ << >> ~` and their compound forms are behind the **`bitwise_symbols`** gate (default **denied**).
- This gate applies **only to numeric operators**. Set/Map algebra (e.g., `A ^ B` symmetric diff) is **always enabled** (type-directed).

## 20) Grammar sketch (EBNF-ish; implementation may vary)
```ebnf
(* Shakar grammar — proper EBNF. Lexical tokens: IDENT, STRING, NUMBER, NEWLINE, INDENT, DEDENT. *)

(* ===== Core expressions ===== *)

Expr            ::= TernaryExpr ;
TernaryExpr     ::= OrExpr ( "?" Expr ":" TernaryExpr )? ;

OrExpr          ::= AndExpr ( "or" AndExpr )* ;
AndExpr         ::= BindExpr ( "and" BindExpr )* ;
NullishExpr     ::= CompareExpr ( "??" CompareExpr )* ;
WalrusExpr      ::= NullishExpr | IDENT ":=" Expr ;
BindExpr       ::= WalrusExpr | LValue ".=" BindExpr ;  (* right-assoc; expression form of .= *)

CompareExpr     ::= AddExpr ( CmpOp AddExpr | "," ( ("and" | "or")? (CmpOp)? AddExpr ) )* ;

CmpOp           ::= "==" | "!=" | "<" | "<=" | ">" | ">=" | "is" | "is not" | "!is" | "in" | "!in" | "not in" ;

AddExpr         ::= MulExpr ( AddOp MulExpr )* ;
AddOp           ::= "+" | "-" | "^" | DeepMergeOp ;
MulExpr         ::= PowExpr ( MulOp PowExpr )* ;
MulOp           ::= "*" | "/" | "%" ;

UnaryExpr       ::= UnaryPrefixOp UnaryExpr | PostfixExpr ;
PowExpr         ::= UnaryExpr ( "**" PowExpr )? ;
UnaryPrefixOp   ::= "-" | "not" | "!" | "$" | "~" | "++" | "--" | "await" ;

PostfixExpr     ::= Primary ( Postfix )* ( PostfixIncr )?
                  | "." ( IDENT | "(" ArgList? ")" | "[" SelectorList ("," "default" ":" Expr)? "]" ) ( Postfix )* ( PostfixIncr )? ;
Postfix         ::= "." IDENT
                  | "[" SelectorList ("," "default" ":" Expr)? "]"
                  | "(" ArgList? ")" ;
PostfixIncr     ::= "++" | "--" ;

SelectorList    ::= Selector ("," Selector)* ;
Selector        ::= IndexSel | SliceSel ;
IndexSel        ::= Expr ;  (* evaluates to int/key; OOB on arrays throws *)
SliceSel        ::= OptExpr ":" OptExpr (":" Expr)? ;

OptExpr         ::= Expr? ;
OptStop         ::= "<" Expr | Expr? ;
Primary         ::= IDENT
                  | Literal
                  | "(" Expr ")"
                  | SelectorLiteral
                  | NullSafe

                  ;

Literal         ::= STRING | NUMBER | "nil" | "true" | "false" ;
SelectorLiteral ::= "`" SelList "`" ;  (* backtick selector literal; first-class value with {expr} interpolation *)
SelList         ::= SelItem ("," SelItem)* ;
SelItem         ::= SliceItem | IndexItem ;
SliceItem       ::= SelAtom? ":" SelOptStop (":" SelAtom)? ;
SelOptStop      ::= "<" SelAtom | SelAtom? ;
IndexItem       ::= SelAtom ;
SelAtom         ::= Interp | IDENT | NUMBER ;
Interp          ::= "{" Expr "}" ;

(* SubjectExpr removed; covered by PostfixExpr *)
(* ImplicitUse is contextual via anchor stack; not a nonterminal here. *)

NullSafe        ::= "??" "(" Expr ")" ;

(* ===== Assignment forms ===== *)

(* ApplyAssign     ::= LValue ".=" Expr ; //redundant, in BindExpr  *)
StmtSubjectAssign ::= "=" LValue StmtTail ;  (* '.' = old LHS; writes back to same LHS path *)
DeepMergeOp     ::= "+>" ;
DeepMergeAssign ::= LValue "+>=" Expr ;
AssignOr        ::= LValue "or=" Expr ;

StmtTail        ::= Expr ;

LValuePostfix ::= "." IDENT | "[" SelectorList "]" ;
PrimaryLHS ::= IDENT | "(" LValue ")" ;
LValue ::= PrimaryLHS ( LValuePostfix )* ( FieldFan )? ;
FieldList       ::= IDENT ("," IDENT)* ;
FieldFan        ::= "." "{" FieldList "}" ;

RebindStmt ::= "=" Expr ;

(* ===== Statements & blocks ===== *)
BaseSimpleStmt  ::= RebindStmt
                  | Expr

                  | AssignOr
                  | Destructure
                  | ReturnIf
                  | Dbg
                  | Assert
                  | UsingStmt
                  | DeferStmt
                  | Hook
                  | CatchStmt
                  | AwaitStmt
                  | AwaitAnyCall
                  | AwaitAllCall
                  | IfStmt ;

SimpleStmt      ::= BaseSimpleStmt
                  | PostfixIf
                  | PostfixUnless ;

PostfixIf       ::= BaseSimpleStmt "if" Expr ;
PostfixUnless   ::= BaseSimpleStmt "unless" Expr ;

IndentBlock     ::= NEWLINE INDENT Stmt+ DEDENT ;
InlineStmt      ::= SimpleStmt ;

Stmt            ::= SimpleStmt
                  | ForIn | ForSubject | ForIndexed | ForMap1 | ForMap2
                  | AwaitAnyCall | AwaitAllCall ;

(* ===== Control flow ===== *)

IfStmt        ::= "if" Expr ":" (InlineBody | IndentBlock) ElifClause* ElseClause? ;
ElifClause    ::= "elif" Expr ":" (InlineBody | IndentBlock) ;
ElseClause    ::= "else" ":" (InlineBody | IndentBlock) ;

ReturnIf      ::= "?ret" Expr ;

GuardChain     ::= GuardHead GuardOr* GuardElse? ;
GuardHead      ::= Expr ":" IndentBlock ;
GuardOr        ::= ("|" | "||") Expr ":" IndentBlock ;
GuardElse      ::= ("|:" | "||:") IndentBlock ;
OneLineGuard   ::= GuardBranch ("|" GuardBranch)* ("|:" InlineBody)? ;
GuardBranch    ::= Expr ":" InlineBody ;
InlineBody     ::= SimpleStmt | "{" InlineStmt* "}" ;

(* ===== Loops ===== *)

ForIn           ::= "for" IDENT "in" Expr ":" (InlineBody | IndentBlock) ;
ForSubject      ::= "for" Expr ":" (InlineBody | IndentBlock) ;
ForIndexed      ::= "for" "[" IDENT "]" Expr ":" (InlineBody | IndentBlock) ;
ForMap1         ::= ForIndexed ;  (* alias; '.' = value; IDENT = key *)
ForMap2         ::= "for" "[" IDENT "," IDENT "]" Expr ":" (InlineBody | IndentBlock) ;  (* '.' = value; key,value bound *)

(* ===== Selectors (per-selector step allowed) ===== *)

Indexing        ::= PostfixExpr "[" SelectorList "]" ;  (* kept for reference; covered by Postfix *)

(* ===== Calls & lambdas ===== *)

Callee          ::= PostfixExpr ;
Call            ::= Callee "(" ArgList? ")"
                  | Callee ArgListNamedMixed ;

ArgList         ::= Arg ("," Arg)* ;
Arg             ::= Expr | HoleExpr ;
ArgListNamedMixed ::= (Arg ("," Arg)*)? ("," NamedArg ("," NamedArg)*)? ;
NamedArg        ::= IDENT ":" Expr ;
(* Placeholder partials in calls *)
HoleExpr       ::= "?" ;
(* If any HoleExpr appears among the immediate arguments of a Call, the Call desugars to a lambda with positional holes left-to-right. Each hole becomes a distinct parameter. *)


LambdaCall1     ::= Callee "&" "(" Expr ")" ;
LambdaCallN     ::= Callee "&" "[" ParamList "]" "(" Expr ")" ;
ParamList       ::= IDENT ("," IDENT)* ;

(* ===== Comprehensions ===== *)

CompHead        ::= ("over" | "for") OverSpec ;
OverSpec        ::= "[" PatternList "]" Expr | Expr ("bind" PatternHead)? ;
IfClause        ::= "if" Expr ;

ListComp        ::= "[" Expr CompHead IfClause? "]" ;
SetComp         ::= "{" Expr CompHead IfClause? "}" ;
DictComp        ::= "{" Expr ":" Expr CompHead IfClause? "}" ;

Pattern         ::= IDENT | "(" Pattern ("," Pattern)* ")" ;
PatternList     ::= Pattern "," Pattern ("," Pattern)* ;
PatternHead     ::= PatternList | Pattern ;   (* use at heads that allow a,b *)

DestrRHS        ::= Expr ( "," Expr )+ ; (* site-local pack, not a value *)
Destructure     ::= PatternList "=" ( DestrRHS | Expr )
                  | Pattern "=" Expr
                  | PatternList ":=" ( DestrRHS | Expr )
                  | Pattern ":=" Expr ;


(* ===== Concurrency ===== *)
AwaitStmt       ::= "await" ( "(" Expr ")" | Expr ) ":" (InlineBody | IndentBlock) ;
AwaitAnyCall    ::= "await" "[" "any" "]" "(" AnyArmList OptComma ")" ( ":" (InlineBody | IndentBlock) )? ;
AwaitAllCall    ::= "await" "[" "all" "]" "(" AllArmList OptComma ")" ( ":" (InlineBody | IndentBlock) )? ;
AnyArmList      ::= AnyArm ("," AnyArm)* ;
AllArmList      ::= AnyArm ("," AnyArm)* ;
AnyArm          ::= (IDENT ":")? Expr ( ":" (InlineBody | IndentBlock) )? | "timeout" Expr ( ":" (InlineBody | IndentBlock) )? ;
OptComma        ::= /* empty */ | "," ;

(* ===== Error handling / hooks ===== *)

CatchExpr       ::= Expr ("catch" | "@@") CatchBinder? CatchTypes? ":" Expr ;
CatchStmt       ::= Expr "catch" CatchBinder? CatchTypes? ":" (InlineBody | IndentBlock) ;
CatchBinder     ::= IDENT | "bind" IDENT ;
CatchTypes      ::= "(" IDENT ("," IDENT)* ")" ;

Hook            ::= "hook" STRING ":" Body ;
LambdaExpr      ::= "(" ParamList? ")" "=>" (Expr | IndentBlock) ;

(* ===== Objects ===== *)

ObjectItem      ::= IDENT ":" Expr
                  | "get" IDENT "(" ")" ":" IndentBlock
                  | "set" IDENT "(" IDENT ")" ":" IndentBlock ;

(* ===== Using / Defer / Assert / Debug ===== *)

DeferStmt       ::= "defer" ( SimpleCall DeferAfter? | DeferLabel? DeferAfter? DeferBody ) ;
DeferLabel      ::= IDENT ;
DeferBody       ::= ":" (InlineBody | IndentBlock) ;
DeferAfter      ::= "after" ( IDENT | "(" (IDENT ("," IDENT)*)? ")" ) ;
SimpleCall      ::= Callee "(" ArgList? ")" ;
UsingStmt       ::= "using" ( "[" IDENT "]" )? Expr ("bind" IDENT)? ":" IndentBlock ;
Assert          ::= "assert" Expr ("," Expr)? ;
Dbg             ::= "dbg" (Expr ("," Expr)?) ;
```

---

#### Anchor semantics (normative notes)
> **Note:** Bare parentheses never start an implicit-chain. Use `.(...)` to call the ambient anchor. Examples: OK `.(x)`, `.(x,y)`, `.foo`, `.foo()`; **Never** `(x)`, `(x,y)` as implicit-chain heads.

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
- **Conditional apply-assign `.?=`**: compute RHS with old LHS as `.` and **assign only if non-nil**. Today use `=<LHS> ??(.transform()) ?? .` or `<LHS> .= ??(.transform()) ?? .`.
- **Keyword aliases (macro-lite)**: project remaps (disabled by default).
- **Autocall any nullary method**: off by default; explicit `getter` is core.
- **Copy-update `with`** sugar: syntax TBA; method `obj.with(...)` works now.
- **Pipes `|>`**: maybe later; redundant for v0.1.
- **Nested/multi-source comprehensions**: later via `bind (a,b) over zip(xs,ys)`.
- **Word range aliases `to`/`until`**: optional later.
- **While/until loops**: consider only if demanded; `for` over `Selector value` covers most cases.
- **Sticky subject (prefix `%`)**: `%expr` sets the **anchor** to `expr` and marks it **sticky** for the current expression: child groupings do not retarget unless another `%` or a new explicit subject appears. **Does not affect** selector bases or `.=`/`=LHS` tails (their `.` rules still win).

---

## 22) Implementation plan (v0.1)

**Phase 1 — Parser & IR**
- Implement grammar above; produce a compact IR.
- Author **desugar tables** per feature; ensure 1:1 mapping.

**Phase 2 — VM**
- Bytecode interpreter in C (embeddable).
- GC with support for `Str` views/ropes; compaction hooks.
- Objects + descriptor slots; implicit `self` for getter/setter/method calls.

**Phase 3 — Stdlib (focused)**
- `fs`, `path`, `json/yaml`, `http`, `time`, `process`, `Event`.
- String APIs aligned with view/rope model (`.own()`, `.materialize()`, `.bytes`).

**Phase 4 — Tooling**
- `shk fmt`, `shk repl --desugar`, initial lints, feature manifest JSON.

**Phase 5 — Feature gates**
- Project + per-module allow/deny with helpful diagnostics.

**Exit criteria**
- Guard-heavy and data-munging demos show 30–50% LOC reduction vs Python/JS.
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

# Objects
emails = { u.id: u.email; over users }
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
