# Shakar Lang — Working Design Notes

Status: concept/spec notes for early compiler & toolchain.
Audience: language implementers and contributors.
Mantra: *Sweet syntax, obvious desugar, zero ceremony.*
Philosophy: keep the core tiny and predictable; push ergonomics into first-class, deterministic sugars.
This is a living technical spec. Every surface sugar has a deterministic desugar to a small, boring core.

- ✅ **Committed**: part of v0.1 surface.
- 🧪 **Experimental**: likely to ship; behind a flag.
- ❓ **Considering**: not in v0.1; tracked for later.

---

## Philosophy & Invariants

- **Ergonomics over ceremony**; sugars are explicit and desugar cleanly.
- **Expression-local magic only**: implicit subject `.` never crosses statement boundaries; it follows the anchor stack rules.
- **Truthiness**: `nil`, `false`, `0`, `0.0`, `""`, `[]`, `{}` are falsey; everything else is truthy.
- **Evaluation**: eager; `and`/`or` short-circuit.
- **Errors**: exceptions; one-statement handlers via `catch`/`@@`.
- **Strings**: raw forms `raw"…"` (escapes processed) and `raw#"…"#` (no interpolation; no escapes; exactly one `#` in v0.1). Examples: `raw"Line1\nLine2"`, `raw#"C:\\path\\file"#`, `raw#"he said "hi""#`.
- **Objects**: objects with getters/setters; no class system required in v0.1.
- **Habitats**: CLI scripting (Python niche) and safe embedding (Lua niche).

## Lexical Structure

- **Identifiers**: `[_A-Za-z][_A-Za-z0-9]*`; case-sensitive. Unicode ids ❓ (later).
- **Comments**: `#` to end-of-line.
- **Whitespace & layout**: indentation (spaces only) after `:` starts blocks.
- **Semicolons**: hard statement delimiters at top level and inside braced inline suites `{ ... }`. Multiple statements may share a line. Grammar shape: `stmtlist := stmt (SEMI stmt)* SEMI?`.
- **Inline suites after `:`**: exactly one simple statement. Wrap a braced inline suite on the right of the colon for multiple statements.
- **Reserved keywords**: `and, or, not, if, elif, else, for, in, break, continue, return, assert, using, defer, catch, decorator, decorate, hook, fn, get, set, bind, over, true, false, nil`.
- **Contextual keywords**: `for` (in comprehensions), `get`/`set` (inside object literals).
- **Punctuation tokens**: `.=` (apply-assign), `|`/`|:` (guards), `?ret` (statement head), `??(expr)` (nil-safe chain), `:=` (walrus).

## The Anchor Model (`.`)

### Binders (where `.` comes from)

- **Statement-subject** `=LHS<tail>` at statement start: inside the statement, `.` is the pre-update value of `LHS`; result writes back to `LHS`.
- **Apply-assign** `LHS .= RHS`: inside `RHS`, `.` is the old value of `LHS`; result writes back to `LHS`.
- **Subjectful loop** `for Expr:` / `for[i] Expr:`: per iteration, `.` is the element (and `i` the view index if present).
- **Lambda callee sigil** (e.g., `map&(...)`): inside the lambda body, `.` is the parameter.
- **`await(expr)` / `await expr`**: trailing body runs with `.` = resolved value; without a body, returns the value with no binder.
- **`await[any]`**: per-arm body uses `.` = that arm’s result; trailing body uses `.` = winning value and `winner` label.
- **Selectors** `base[ sel1, sel2, … ]`: inside each selector, `.` = the `base`.
- **Not creators**: one-line guards and plain blocks do not create a subject.

### Leading-dot chains

Within a grouping that has an anchor, any leading dot applies to that anchor. Each step’s result becomes the next receiver within the chain; the anchor stays until explicitly retargeted or popped.

### Grouping & the anchor stack

- **Comparison comma-chain legs are no-anchor.** The chain’s subject is the first explicit operand.
- **Pop on exit**: leaving a grouping restores the prior anchor; sibling leading dots reuse it. Example: `a and (b) and .c()` ⇒ `a and b and a.c()`.
- **No-anchor `$`**: `$expr` evaluates without retargeting the anchor; siblings keep using the previous anchor. Example: `a and $b and .c` ⇒ `a and b and a.c`.
- **Push/pop scopes**: parentheses, lambda bodies, comprehension heads/bodies, and `await[...]` bodies push the current anchor; they restore it on exit. Retarget inside by mentioning a new explicit subject.

### Interactions & shadowing

- Inner binders temporarily set `.` then restore.
- In `for[i] xs: xs[i] .= RHS`, `.` inside `RHS` is the old `xs[i]`, not the loop element.
- After `.=` within a grouping, sibling leading dots anchor to the explicit `LHS` and see the updated value. Within selectors in that RHS, `.` is the old base. Example: `(xs[0] .= .trim()) and .hasPrefix("a")`.

### Illegals & locality

- `.` is never an lvalue (`. = …` illegal).
- No free `.` outside an active binder/anchor.
- Invalid statement-subject: `=LHS` with no tail, `=.trim()`, or grouped heads that are not pure lvalues.

### Normative laws

1. **Anchor law**: the first explicit subject in a grouping sets the anchor for leading-dot chains until retarget or pop.
2. **Selector law**: inside `base[ … ]`, `.` = base; outside, the result may retarget the anchor.
3. **Leading-dot chain law**: each step’s result becomes the next receiver; the anchor is stable unless retargeted.
4. **Binder-shadowing law**: inner binders (lambda, `await`, subjectful `for`, apply-assign RHS) shadow `.` for their extent, then restore.
5. **Illegals/locality**: `. = …` is invalid; free `.` is invalid.

#### Conformance checks

```shakar
# Apply-assign with selector on old LHS + expression-valued anchor
xs = [[1,2,3], [4]]
xs .= .[0]
assert xs == [1,2,3]
(xs .= .[0]) and .len == 3
```

```shakar
# Statement-subject vs apply-assign symmetry
s = "  A  "
=s.trim()
t = "  B  "; t .= .trim()
assert s == "A" and t == "B"
```

```shakar
# Statement-subject writes back to the same path
s = "  hi  "
=s.trim()
assert s == "hi"
user = { name: "  Bob  " }
=user.name.trim()
assert user.name == "Bob"
user = { profile: { contact: { name: "  Ada " } }, flags: [] }
=(user).profile.contact.name.trim()
assert user == "Ada"
=(user.profile.contact.name).trim()
xs = [" A ", "b"]
=xs[0].trim()
# Errors
# =user.name       # missing tail
# =.trim()         # free '.'
# =(user + other).trim()   # grouped head must be a pure lvalue
```

```shakar
# Anchor + grouping, selector scope, binder shadowing
a and (b and .x()) and .y()      # '.x()' anchored to b; '.y()' anchored to a
users[ .len-1 ]                  # inside selector: '.' = users
(users[0] and .name)             # outside selector: '.' = users[0]
for[i] names: names[i] .= .trim()   # in RHS: '.' = old names[i]
```

```shakar
# Selector as value vs selector
sum := 0
for i in `0:3`: sum += i           # 0+1+2+3
ix, vals := [], []
xs = [10,11,12,13]
for[i] xs[1:3]: ix.append(i); vals.append(.)
assert ix == [0,1] and vals == [11,12]
```

---

## Data Types & Literals

### Core types

- **Int** (`i64`, overflow throws), **Float** (`f64`), **Bool**, **Str** (immutable UTF-8), **Array**, **Object** (map with descriptors), **Func**, **Selector literal** (iterable numeric range), **Nil**. Type predicates live in the stdlib (`isInt`, `typeOf`); no type grammar.

### Primitive literals

- `nil`, `true`, `false`.
- **Integers**: 64-bit signed; underscores allowed; bases `0b 0o 0x`; overflow throws.
- **Floats**: IEEE-754 double; leading zero required (`0.5`, not `.5`).
- **Strings**: `"…"`, `'…'` with escapes `\n \t \\ \" \u{…}`. Multiline ❓.
- **Arrays**: `[1, 2, 3]`.
- **Objects**: `{ key: value }` (getters/setters contextual, below).
- **Selector literals (values)**: backtick selectors like `` `1:10` `` produce Selector values (views/iterables). Default stop is inclusive; use `<stop` for exclusive (e.g., `` `[1:<10]` ``).

### Strings & interpolation

- Interpolation: `{expr}` inside quoted strings; braces used literally must be doubled. Empty `{}` is illegal. Expressions parse normally and use surrounding scope/subject; evaluation is left-to-right with immediate stringification.
- Strings without `{` stay as simple literals.
- Raw strings: `raw"…"` (escapes processed) and `raw#"…"#` (no interpolation, no escapes; one `#` in v0.1).
- String model: immutable UTF-8 with zero-copy **views** for slices/trim/drop and **ropes** for concat. Unique-owner **compaction** avoids substring leaks (thresholds: keep <¼ or >64KiB slack). Controls: `.own()`, `.compact()`, `.materialize()`. Unicode indexing counts characters; `bytes` view for byte-wise ops. FFI materializes exact bytes (+NUL).
- Examples keep view semantics:
  ```shakar
  =s[1:]            # view or in-place compact when unique
  =s.trim()         # view (offset/len tweaked)
  =path[.len-1]     # view of last segment
  ```

### Shell strings (`sh"..."`)

- Literal: `sh"..."` or `sh'...'`; evaluates to a lazy `Command` (no auto-exec).
- Execution: `cmd.run()` returns stdout (`Str`) on success; non-zero exit raises `CommandError` with payload `{ cmd, code, stdout, stderr }`. Future `!cmd` may execute inline; stdlib will grow capture/streaming helpers.
- Interpolation & safety: `{expr}` is auto-quoted; arrays expand to multiple quoted args. `{{expr}}` splices raw/unsafe tokens.
- Pass-through: pipes, redirects, `&&` forwarded to the system shell.
- Example:
  ```shakar
  files := ["a.txt", "b 1.txt"]
  cmd := sh"ls -l {files} | grep 'x' > {outfile}"
  res := cmd.run()
  ```

### Selector values, slices, and selectors

- **Selector literals as values** use backticks: `` `lo:<hi` `` or `` `1:<5, 11:15:2, 20` ``. Inside backticks, `{expr}` embeds bounds or indices; `{…}` shields punctuation. Prefer bare identifiers/numbers; no top-level parentheses (use `{…}` inside if needed).
- **Selectors inside `[]`**: selector literals are valid items and are flattened into the selector list at that position. Example: `xs[`1:10`]` ≡ `xs[1:10]`.
- **Selector values in collections**: backtick literals are ordinary values in arrays/objects/sets; no flattening there.
- **Type checks**: each selector list item must be int index or selector value; otherwise a type error. Per-selector steps allowed.
- **Inside selectors**: `.` = base (e.g., `xs`) for that selector only.
- **Slices (arrays & strings)**:
  ```shakar
  a[i:j]         # half-open i <= k < j
  a[i:j:step]    # step != 0; negative step for reverse
  a[:j]; a[i:]; a[:]   # clamp to [0:len]
  a[0:-1]        # positive start, negative stop allowed
  a[-3:-1]       # both negative allowed
  a[-1:2]        # ERROR: negative start with positive stop disallowed
  ```
  Negative indices allowed; no auto-reverse when `i > j`. Indexing `a[i]` throws OOB; slices clamp; inverted positive-step yields `[]`. Slice restriction: negative start with positive stop is an error (ambiguous semantics). Positive start with negative stop is valid (e.g., `arr[0:-1]` for "all but last"). Strict slicing: `slice!(a, i, j, step?)` throws on any OOB.
- **Selector lists** (multiple selectors inside `[]`): concatenation of each selector’s result, in order. Index selectors throw on OOB; slice selectors clamp. LHS restriction: selector lists and slices are expression-only in v0.1 (no LHS assignment).
- **Selector literals in comparisons**: see Comparison & Identity for CCC handling.
- **Examples**:
  ```shakar
  xs[:5, 10:15:-2]
  xs[1 : .len-1]
  xs[5:2:-1]
  customSelector := `1:10, someIdx`; xs[customSelector]
  ```

### Objects & descriptors

- Object = map `name → Slot`; Slot = `Plain(value)` or `Descriptor{ getter?, setter? }`.
- Property read: Plain → value; Descriptor.getter → call with implicit `self`.
- Property write: Plain → set; Descriptor.setter → call with `self, value`; strict mode may forbid setting Plain when descriptor exists.
- Methods: `obj.m(args)` sets `self=obj`.
- Contextual `get/set` in object literals desugar to descriptor slots. Getter arity 0; setter arity 1.
- Literal keys: identifier (`a: expr`), string (`"a-b": expr`), computed (`[expr]: expr`).
- Access: `obj.key` for identifier keys (method fusion), `obj[expr]`/`obj["k"]` for all keys.
- Shape tagging: `closed` when all keys are identifiers and no computed keys or spreads (duplicate identifier keys are errors); `open` otherwise (dict path; last write wins).
- Validator: enforce getter/setter arities; forbid duplicate identifier keys in closed shapes; require bracket access for non-ident keys. Printer preserves order; block bodies allowed after `:`.
- Example:
  ```shakar
  user = {
    id: 1,
    name: "Ada",
    [1+2]: 3,
    get size: 1,
    set size(x): x
  }
  user.name
  user["name"]
  user[1+2]
  ```

### Structural match (`~`)

- **Syntax**: `Value ~ Schema` (comparison-tier precedence).
- **Semantics**: recursive per RHS node:
  - If RHS is a **Union**, check if LHS matches any alternative: `Union(T1, T2, ...)` succeeds if `LHS ~ T1` or `LHS ~ T2` or ...
  - If RHS is a type (e.g., `Str`, `Int`), check `type(LHS) == RHS`.
  - If RHS is a Selector, check `LHS in RHS`.
  - If RHS is an Object, LHS must be an Object and each RHS key must exist with `LHS[K] ~ RHS[K]`.
    - **Optional fields**: If RHS value is `Optional(Schema)`, missing key is allowed. If present, must match inner schema.
  - Otherwise use value equality `LHS == RHS`.
- **Goal**: schema validation for JSON-like structures.
- **Optional field syntax**: `key?: Schema` desugars to `key: Optional(Schema)`.
- **Union types**: `Union(Schema1, Schema2, ...)` matches if value satisfies any alternative.
- **Examples**:
  ```shakar
  UserSchema := {
    name: Str,
    age: `18:120`,
    role: "admin"
  }
  if payload ~ UserSchema: process(payload)

  # Optional fields
  ApiResponse := {
    status: Int,
    data?: Object,
    error?: Str
  }
  {} ~ {age?: Int}              # true (missing optional is OK)
  {age: 30} ~ {age?: Int}       # true (present and valid)
  {age: "bad"} ~ {age?: Int}    # false (present but wrong type)

  # Union types
  5 ~ Union(Int, Str)           # true (Int alternative matches)
  "hi" ~ Union(Int, Str)        # true (Str alternative matches)
  true ~ Union(Int, Str)        # false (no alternative matches)

  # Nullable pattern
  result ~ Union(Int, Nil)      # allows Int or nil

  # Complex schemas
  {id: 1} ~ {id: Union(Int, Str)}          # true
  {id: "abc"} ~ {id: Union(Int, Str)}      # true
  {name: "Alice", age?: Union(Int, Str)}   # combines optional and union
  ```

---

## Expressions & Operators

### Precedence & associativity

- Set/object algebra precedence: `*` (intersection) is multiplicative; `+`, `-`, `^` additive. Numeric bitwise `^` remains gated.
- Associativity: binary operators are left-associative unless noted.
- Order (high → low):
  1) Postfix: member `.`, call `()`, index `[]`
  2) Unary: `-x`, `not x`/`!x`, `$x`
  3) Power: `x ** y` (right-assoc)
  4) Multiplicative: `*`, `/`, `//`, `%`
  5) Additive / concat: `+`, `-`, `+>` (deep object merge)
  6) Shifts (gated): `<<`, `>>`
  7) Bitwise (gated): `&`, `^`, `|` (infix)
  8) Comparison: `<`, `<=`, `>`, `>=`, `==`, `!=`, `is`, `is not`, `!is`, `in`, `!in`, `not in`, `~`
  9) Nil-safe chain: `??(expr)` (prefix primary)
  10) Nil-coalescing: `a ?? b` (right-assoc; tighter than `or`)
  11) Walrus & apply-assign: `:=`, `.=` (expression-valued; tighter than `and`/`or`, lower than postfix)
  12) Boolean: `and`, `or` (short-circuit, value-yielding)
  13) Ternary: `cond ? then : else`
  14) Assignment statements: `=`, compounds, `or=`, statement-subject `=x or y`

### Arithmetic & algebra

- **Objects**: `M + N` merge (RHS wins conflicts); `M * N` key intersection (values from RHS); `M - N` key difference; `M ^ N` symmetric key difference.
- **Strings/arrays repeat**: `s * n`/`n * s`, `a * n`/`n * a` (Int `n >= 0`, else error).
- **Sets**: `A + B` union; `A - B` difference; `A * B` intersection; `A ^ B` symmetric diff. Operators produce new sets (no mutation).
- `/` yields float; `//` floor-div (ints or floats, returns int via floor); `%` remainder with dividend sign; `**` exponentiation (right-assoc).

### Bitwise (via `std/bit`)

- Symbolic bitwise operators are gated by `bitwise_symbols`. Use `bit.and`, `bit.or`, `bit.xor`, `bit.not`, `bit.shl`, `bit.shr` otherwise.
- When gated on, tokens `& | ^ << >>` use usual precedence; one-line guards still reserve `|/|:` only after the first `:` at bracket-depth 0. Errors on non-int operands.
- Note: Unary bitwise NOT (`~`) is NOT supported even when `bitwise_symbols` is enabled. Use `bit.not(x)` for bitwise negation.

### Comparison & identity (CCC)

- **Comparison comma-chains (CCC)**:
  - Enter chain after a comparison followed by a comma (`S op X,`). Subject `S` evaluates once.
  - Leg form (after commas): optional `and|or`, optional comparator, then `Expr`. Carry-forward comparator allowed when joiner absent or `and`; after `or`, comparator required. Joiner is sticky (`and` by default) until changed by `, or`/`, and`.
  - Leaving chain: any `and|or` without a comma leaves chain mode and resumes normal boolean parsing.
  - Desugar: expand left→right to `S op_i Expr_i` joined by current sticky joiner; apply normal precedence (`and` tighter than `or`).
  - **Disambiguation**: Commas disambiguate via lookahead (`, or`/`, and`/`, <cmpop>` → CCC) and context. Without explicit markers, bare `, <value>` is treated as a separator in function arguments, array literals, and destructure packs. Parentheses reset context and allow CCC: `f((x == 5, < 10))` passes a single CCC result; `[1, (a > 0, < 10)]` is a 2-element array.
  - Parentheses nest CCCs without affecting subject/eval rules.
  - **Selectors in comparisons**: with backtick selector literals, `S == \`Sel\`` ⇒ any(); `S != \`Sel\`` ⇒ all not-in; `S <op> \`Sel\`` for `< <= > >=` ⇒ all(), reducing to `min/max(Sel)` with openness respected (e.g., ``a < `1:<5` `` ⇒ `a < 1`; ``a >= `lo, hi` `` ⇒ `a >= max(lo, hi)`).
  - **Examples**:
    ```shakar
    a = 7; b = 3; c = 10
    assert a > 1, < 9, != 8
    assert (a > 10, or == 0, or == 1) == ((a > 10) or (a == 0) or (a == 1))
    assert (a > 10, or >= c) == ((a > 10) or (a >= c))
    assert (a > 10 or b) == ((a > 10) or b)
    assert a == b+4, 8-1
    assert 4 < `5:10`, != `7, 8`
    ```
- `==`, `!=` are value equality (type-aware). Int vs Float equality compares by value (`3 == 3.0` true); other cross-type comparisons error (except nil equality).
- Ordering `< <= > >=` defined for numbers and strings (lexicographic by normalized UTF-8 bytes); arrays/objects ordering ❓.
- Identity: `is`, `is not`/`!is` check identity (same object/storage). For strings/views, identity = same `(base, off, len)` even if values equal. `!is` is a single token.
- **Structural match** `~` shares the comparison tier; semantics in Structural match above.

### Membership

- `x in y`: substring for strings; element membership for arrays (`==`); key membership for objects.
- `x not in y` / `x !in y` are negations.

### Boolean & ternary

- `and`, `or` (short-circuit, yield last evaluated operand); no `&&/||` in core.
- Ternary `cond ? a : b` (right-assoc; lower than `or`).

### Null safety

- Nil-safe chain `??(expr)` turns a deep deref/call chain into nil-propagating expression.
- Nil-coalescing `a ?? b` returns `a` unless `a` is `nil`, otherwise `b` (right-assoc, tighter than `or`).
- Allow paren-free nullsafe `?? expr` sugar (equivalent to `??(expr)`, tighter than infix `??`, capturing the following postfix chain) ❓

### Assignment & mutation

- **Binding vs update**:
  - `name := expr` introduces a new binding in the current lexical scope, yields the value, and errors if `name` already exists in that scope.
  - `name = expr` updates an existing binding; statement-only; errors if `name` does not exist.
  - Destructuring: arrays `[head, ...rest] := xs`; objects `{id, name} := user`; updates use `=`. Chaining `x := y := 0` is disallowed.
- **Walrus `:=` anchor**: assigns `RHS` to `LHS`, yields the value, and anchors `.` to `LHS` for the remainder of the containing expression at that level and within `RHS` after its head value is computed. Precedence tighter than `and`/`or`, lower than postfix. Example/desugar:
  ```shakar
  u := makeUser() and .isValid()
  # tmp0 = makeUser(); u = tmp0; tmp0 and u.isValid()
  ```
  Use `bind` when an operator is not desired: `makeUser() bind u and .isValid()`.
- **Subjects and LValues**: lvalues are identifiers with member/selector chains; calls are not allowed on the LHS. Grouped statement-subject `=(LHS)` must be an lvalue.
  - Statement-subject rules: `=name<tail>` desugars to `name = name<tail>`; `name` must exist and `<tail>` must do work (e.g., call/selector/fan-out/`.=`). `=name.field` with no effect is rejected. Grouped heads pick that exact lvalue as the destination (e.g., `=(user.profile.email).trim()`). The statement-subject must start the statement.
- **Apply-assign `.=`**: inside `RHS`, `.` = old `LHS`; result writes back and the expression yields the updated value. Notes: bare `.` cannot stand alone; to keep old value on failure, use explicit fallback (`a .= ??(.transform()) ?? a`). If only defaulting, prefer `a = a ?? default`. Idioms:
  ```shakar
  user.{name, email} .= ??(.trim().lower()) ?? "unknown"
  user.nick .= .trim() ?? user.nick
  profile.age .= .clamp(0, 120) ?? 0
  cfg["db", default: {}]["host"].= .trim() ?? "localhost"
  user.{phone, altPhone} .= .digits() ?? ""
  ```
  After `.=` in a grouping, sibling leading dots anchor to `LHS` and see the updated value.
- **Compound & defaults**: map compounds `+= -= *= ^=` mutate maps in place with object-algebra semantics; set compounds `+= -= *= ^=` mutate sets in place. Simple compounds `+= -= *= /= //= %= **=` (bitwise variants gated). Defaults: `or=` and statement-subject `=x or y` desugar to `if not …: … = …`.
- **Increments**: prefix/postfix `++/--` on lvalues (numeric only). Postfix terminal and only at end of member/selector chain; prefix returns updated value, postfix the pre-update value.
- **Multi-field assign fan-out**:
  - Surface:
    ```shakar
    =user.{name, email}.trim()
    user.{first, last} .= .title()
    =user.address.{city, street}.title()
    state.{a, b}.c = 5              # fieldfan chaining
    state.{a, b}.c .= . + 1         # fieldfan chain with apply-assign
    ```
  - Desugar:
    ```shakar
    =user.name.trim(); =user.email.trim()
    user.first .= .title(); user.last .= .title()
    =user.address.city.title(); =user.address.street.title()
    state.a.c = 5; state.b.c = 5
    state.a.c .= . + 1; state.b.c .= . + 1
    ```
  - Rules: only after a path; left-to-right order; duplicates are errors; missing fields error; `.=` RHS uses per-field old value; chain rebinds can fan out and yield list of updated values. Fieldfan may appear mid-chain with further field/index access after the fanout; each fanned target is independently traversed for subsequent segments.
- **Fanout block statement**:
  - Surface:
    ```shakar
    state{
      .cur = .next
      .x += 1
      .name .= .trim()
    }
    ```
  - Desugar:
    ```shakar
    state.cur = state.next
    state.x += 1
    state.name .= .trim()  # RHS sees old state.name as `.`
    ```
  - Semantics: `.` anchors to `state`; each clause starts with `.` and uses `=/.=/+= -= *= /= //= %= **=`; `.=` RHS sees the old field value as `.` then resets `.` to the base for the next clause. Clauses run top→down; base eval once; missing/duplicate fields error; result unused (statement-only).
  - Selector fanout (subtle!): path segments may include index selectors and slices. If a selector yields multiple elements (e.g., `.rows[1:3]`), the clause broadcasts to each selected element, preserving left→right order. Example:
    ```shakar
    state := {rows: [{v:1}, {v:3}, {v:5}]}
    state{ .rows[1:3].v = 0 }   # rows -> [{v:1},{v:0},{v:0}]
    state{ .rows[1][0].v += 5 } # works through nested indices
    ```
    Broadcasting only happens when the selector produces multiple targets (slice or multi-selector); single-index selectors behave as a single target. Duplicate target detection still applies on the resolved targets.
  - Single-clause fanout: allowed when the clause either targets multiple elements (e.g., `.rows[1:5]`) **or** is an implicit-subject rewrite on both sides (e.g., `state{ .cur = .next }`). Prefer a plain assignment when the fanout block adds no clarity.
  - Mode guidance: In a prospective “strict” mode, bare-path selector broadcasts (e.g., `state.rows[1:3].v = 0`) would be rejected unless wrapped in braces (`state{ .rows[1:3].v = 0 }` or `state.{rows[1:3].v} = 0`). In normal mode, tooling/lints should warn on such bare-path selector broadcasts and recommend the braced form for clarity.
- **Fanout value + call auto-spread**:
  ```shakar
  values = state.{a, b, c}      # [state.a, state.b, state.c]
  fn call(): someFn(1, state.{a, b}, 2)  # spreads to positional args
  ```
  Items may be field names or chained postfix off a field (calls/indexes). Evaluates base once, items left→right; missing/duplicate fields error. In arglists the resulting tuple/array auto-flattens; illegal in named-arg position unless wrapped (e.g., `f(xs: [state.{a,b}])`). LHS fanout semantics stay the same; shared duplicate/missing checks.
  Expression-style fanout blocks (`base{ .c - .d }` returning a value) remain in the "considering" bucket.
# Apply-assign with selector on old LHS + expression-valued anchor
- **Deep object merge `+>` and `+>=`**: recursive key-wise merge where RHS wins; non-object at a path replaces LHS at that path; arrays/sets/strings/numbers replace wholesale. Additive-tier precedence. `+>=` mutates LHS (must be object or error); `+>` yields value.
- **Object index with default**:
  ```shakar
  count = hits["/home", default: 0]
  port  = env["PORT", default: "3000"]
  value = cfg["db", default: {}]["host", default: "localhost"]
  ```
  Works on objects only (static error otherwise). If key exists, return value; if missing, return `default:` expression (lazy). Arrays/strings reject `default:` (OOB still throws).
- **Destructuring & broadcast**: `a, b := 1` broadcasts a single RHS value (evaluated once) to each LHS target when arity requires. LHS requires a pattern list for multiple targets (`a = 1, 2` is an error; use `a, b = 1, 2`). Nested patterns supported via parentheses: `a, (b, c) := [1, [2, 3]]` destructures nested arrays.
  - **Destructure contracts**: Per-identifier contracts supported via `ident ~ Schema` syntax. Example: `a ~ Int, b ~ Str := get_pair()` validates each value before binding. Contracts are optional and can be mixed: `id ~ Int, name, age := data`. Contracts can be combined with nested patterns: `a ~ Int, (b, c) := [10, [20, 30]]`.

### Expression examples

```shakar
# Arithmetic & assignment
x = 10
x += 5 * 2
y = 7 // 2          # 3
z = 2 ** 10         # 1024
mask = bit.or(bit.shl(1, 8), 0x0F)
ok   = bit.and(flags, mask) != 0
level = debug ? "debug" : "info"
"a" in "shakar"
3 in [1,2,3]
"id" in { id: 1 }
=a.lower() and .allLower()
u := makeUser() and .isValid()
```

---

## Control Flow

### Guards & postfix conditionals

- **Punctuation guards** (desugar 1:1 to if/elif/else). Syntax: head `Expr: Body`, or-branches `| Expr: Body`, else `|: Body`. One-line form chains with `|`/`|:`; multi-line uses indentation. `||`/`||:` accepted in one-line guards and normalized to `|`/`|:` (only after first `:` at bracket-depth 0). Nearest-else rule: after a head’s first `:`, any `|`/`|:` at the same bracket depth binds to the innermost open guard; wrap inner guards to bind outward. Guard heads ending with dict literals should be parenthesized. Do not mix punctuation and keyword guards in the same group.
- **Examples**:
  ```shakar
  ready():
    start()
  | retries < 3:
    retry()
  |:
    log("not ready")
  ready(): start() | retries < 3: retry() |: log("not ready")
  user and .active: process(user) |: log("inactive")
  ??(conn.open): use(conn) |: log("connection closed")
  ```
- `await` one-liners are statements, not guards. `(await f()): …` may parse as guard head; style-lintable.
- **Postfix conditionals**: `stmt if cond`, `stmt unless cond`. If the statement is a walrus, runtime sets the binding to `nil` before the guard so a failing condition leaves it at `nil` without running `expr`.
- **Early return**: `?ret expr` returns early if `expr` is truthy.

### Loops

- Iterating the `nil` literal is a compile-time error; iterating a variable that evaluates to `nil` is a no-op.
- Forms:
  - `for Expr:` subjectful loop; body sees `.` = element.
  - `for[i] Expr:` indexed view; `i` = view index; `.` = element.
  - `for[k] m:` / `for[k, v] m:` on objects; `.` = value (or `v` if bound). Destructuring alias: `for (k, v) in m:`.
  - `for … in <selectors>:` iterate selector lists/values.
  - `while expr:` standard while loop; condition re-evaluated each iteration; supports inline or indented bodies.
- Rules: `.` is per-iteration and does not leak. Nested subjectful constructs rebind `.`; name outer value if needed before nesting. Works with any iterable.
- **Hoisted binders `^name`**: bind loop variable to an outer-scope binding (create it with `nil` if absent). Each iteration assigns current value. Illegal to mix `^x` and `x` or repeat `^name` in a binder list. Closures capture the single hoisted binding. Examples:
  ```shakar
  for[^idx] `0:4`:
    use(idx)
  sum := 0
  for[j, ^sum] arr:
    sum = sum + .
  ```
- Controls: `break`, `continue`; `return` from functions; resource constructs (`defer`, `using`) allowed inside.
- **Loop examples**:
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
  for[k] emails: send(k, .)
  for[k, v] emails: send(k, v)
  ```

### Comprehensions

- Use `over` (preferred) or contextual `for` inside `[]`, `{}` comps. Implicit subject `.` is the current element (like a subjectful `for`). Example:
  ```shakar
  names = [ .trim() over lines if .len > 0 ]
  uniq  = set{ .lower() over tokens }
  byId  = { .id: . over users if .active }
  ```
- `bind` introduces names without removing `.`; binder list sugar `over[binders] src` is sugar for `over src bind binders`. Objects: one binder yields key, two yield key/value.
- Implicit head binders: free, unqualified identifiers in heads (and optional `if` guards) auto-bind to components of each element, first-use order, capped by arity. Only in heads/guards; names resolving in outer scope are captures, not binders; occurrences inside nested lambdas ignored. Error if distinct head-binders exceed element arity.
- Illegal: `over[...] Expr bind ...` (formatter drops redundant `bind`).
- Scope: names introduced by `bind`/implicit binders visible throughout the comprehension, not outside.
- Desugar (list): `lines.map&(.trim()).filter&(.len > 0).to_list()`.

### Errors, assertions, diagnostics

- **Catch expressions**:
  ```shakar
  val = risky() catch err: recover(err)
  val = risky() @@ err: recover(err)
  fallback = risky() catch: "ok"
  user := risky() catch (ValidationError, ParseError) bind err: err.payload
  ```
  Evaluate left; on success return original value. On `ShakarRuntimeError` (or subclass), run handler and use its value. Without a type guard, `catch err:` binds payload (omit binder to rely on `.`). With a guard: `catch (Type, …) bind name:` filters. Payload exposes `.type`, `.message`, `.key` (for `ShakarKeyError`), `.method` (for `ShakarMethodNotFound`). Bare `catch` is catch-all. `throw` inside handler rethrows; `throw expr` raises new error.
- **Catch statements** mirror expression semantics but discard original value; bodies execute only on failure.
- **Inline vs block handlers:** `expr catch err: handler_expr` is expression-valued (usable in walrus/assign chains without parentheses). A block-bodied catch (`expr catch err: { … }` or newline/indent) is statement-valued and yields `nil`. Use inline form when you need the handler’s value; use a block when you need multiple statements or don’t care about the value.
- **assert expr, "msg"`** raises if falsey; build can strip.
- **throw [expr]** re-raises current payload when expression omitted; otherwise raises new `ShakarRuntimeError` from value (strings → message; objects set `.type/.message`).
- **Helpers**: `error(type, message, data?)` builds tagged payload; `dbg expr` logs and returns expr (strip-able).
- **Events**: `hook "name": .emit()` ⇒ `Event.on(name, &( .emit()))`.

---

## Functions & Abstractions

- **Named-arg calls**: all invocations use parentheses, even with named args. Example:
  ```shakar
  fn send(to, subject, body): …
  send("bob@x.com", subject: "Hi", body: "…")
  ```

### Function forms

- `fn(params?): body` is an expression producing a callable value. Bodies match named `fn` syntax (inline or indented). Zero-arg IIFE sugar: `fn(()): body` desugars to `(fn(): body)()`. Expression bodies execute on call, not on literal creation. Lambdas with `&` covered below.
- **Type contracts**: Parameters may specify schemas using `param ~ Schema` syntax. Return values may specify schemas using `fn(params) ~ ReturnSchema:` syntax. Both desugar to runtime assertions. See the Structural Match section for details.

### Decorators

- Define with `decorator name(params?): body`; inside, `f` is the next callable and `args` is a mutable array of positional arguments. Mutate `args`, reassign it, or `return` to short-circuit; implicit `return f(args)` when body falls through.
- Apply with `@decorator` lines above `fn` definitions. Expressions evaluate top→bottom; the closest decorator wraps first (so outer runs after inner). Parameterized decorators behave the same; bare `@decorator` calls parameterless decorator. Decorator expressions must produce a decorator/instance.
- `args` is ordinary `Array`; mutate elements or rebind entirely. Helpers allowed before calling `f(args)`.

### Amp-lambdas (`&`) and implicit parameters

- `map&(.trim())` (single arg implicit subject) or `zipWith&[a,b](a+b)` (explicit params). `&` lives on the callee.
- **Implicit parameter inference** at known-arity call sites: collect free, unqualified identifiers used as bases in the body (ignore inside nested lambdas), left→right. If body uses `.` anywhere, inference is disabled (choose implicit params or subject, not both).
- Errors: distinct free names > arity; names that resolve in surrounding scope are captures (use `&[x](...)` to shadow). No mixing of `.` with implicit params.
- Examples:
  ```shakar
  zipWith&(left + right)(xs, ys)      # infers &[left, right](left+right)
  pairs.filter&(value > 0)            # unary site -> &[value](value>0)
  lines.map&(.trim())                 # uses subject; no inference
  ```
- Callee policy via `@implicit_params(policy)` (`exact` default, `pad`, `off`):
  ```shakar
  @implicit_params('exact') fn zipWith(xs, ys, f) { ... }
  @implicit_params('pad')   fn mapWithIndex(xs, f) { ... }
  @implicit_params('off')   fn reduce(xs, init, f) { ... }
  struct Dict { @implicit_params('pad') fn mapWithKey(self, f) { ... } }
  ```
  `exact`: #distinct free names must equal arity N. `pad`: may use fewer; missing positions become anonymous/ignored. `off`: disable inference (write explicit params).

### Placeholder partials `?`

- A `?` among the immediate arguments of a call produces a lambda with one parameter per hole, left→right. Each hole is distinct; named args participate in order.
- Works for free or method calls; yields a function (no immediate invocation). `?` is recognized only inside a call’s argument list.
- Prefer `&` path-lambdas for single holes; use `?` when 2+ holes increase clarity. Examples:
  ```shakar
  between = inRange(?, 5, 10)              # (x) => inRange(x, 5, 10)
  mix     = blend(?, ?, 0.25)              # (a, b) => blend(a, b, 0.25)
  xs.map&(get(?, id))                      # (x) => get(x, id)
  ```

### Regex helpers

- Literal: `r"..."/imsx` (flags optional). Methods: `.test(str) -> Bool`, `.match(str) -> Match?`, `.replace(str, repl) -> Str`. Timeouts/engine selection are library concerns.

---

## Concurrency & Resources

### Await

- **Single await**: `await(expr)` or `await expr` (equivalent). Trailing body runs with `.` = resolved value; without body, returns value. Prefer `await(expr)` near binary ops for clarity.
- **Await[any]** (two shapes):
  - Per-arm bodies (no trailing body): arms start concurrently; first successful arm runs its body with `.` = arm result; others canceled; returns that body.
  - Trailing body (no per-arm bodies): starts all heads; when one wins, run trailing block once with `.` = winning value and `winner` label; returns trailing block result. Without trailing body, returns `{ winner, value }`.
  Example:
  ```shakar
  await[any](
    fetchUser(id): show(.),
    fetchFromCache(id): show(.),
    timeout 200): showFallback()
  )
  r := await[any]( user: fetchUser(id), cache: fetchFromCache(id), timeout 200 )
  ```
- **Await[all]**: start all heads; with trailing body, run once after all resolve with named results in scope; without trailing body, return object of results. If any head fails, the expression throws unless caught per-head.
  ```shakar
  await[all]( user: fetchUser(id), posts: fetchPosts(id) ):
    show(user, posts)
  g := await[all]( user: fetchUser(id), posts: fetchPosts(id) )
  ```
- Notes: uses parentheses (not `{}`); do not mix per-arm bodies with trailing body in one call. `.` refers to construct-defined value (not outer lambdas). `timeout` default unit ms; suffixes `ms/s/m`. `sleep(ms)` returns awaitable for staging async workflows.

### Resource management

- **using**: scoped resource with guaranteed cleanup (inline or indented).
  1) Evaluate `expr` to `r`.
  2) Enter: call `r.using_enter()` if present, else `r.enter()`, else `v = r`. Optional `bind name` binds `name = v` for the block.
  3) Run block. `using` is not subjectful; `.` stays whatever an enclosing binder set (free `.` errors).
  4) Exit always runs (even on error/return): call `r.using_exit(err?)` else `r.exit(err?)` else `r.close()`. If exit method returns truthy, original block error is suppressed. Errors in exit propagate; dual failures preserve both contexts.
  - Single resource per `using`; nest for multiples. Without `bind`, enter value is unnamed. Preferred style: call through bound name; use walrus temporary if dot on resource is needed.
    - Pragmatics: when `expr` is a bare identifier and no `bind`/handle is given, `using` implicitly rebinds that identifier to the enter value for the body (restored on exit).
  - Examples:
    ```shakar
    using openFile(path): processFile(.)
    using connect(url) bind conn:
      conn.send(payload)
    using[f] getTempFile():
      f.write(data)
    ```
- **defer**: schedule work for block exit (includes early exits).
  - Shapes: `defer closeHandle()`, `defer log("done") after cleanup`, `defer cleanup: closeHandle()`, multiline bodies allowed.
  - Without a handle, defers run in LIFO order. Handles label defers; `after (h1, h2, …)` enforces dependencies (resolved before execution; unknown handles or cycles error). Handles are per-block; reuse errors.
  - Any defer with a handle or `after` must use the colon form unless the simple anonymous-call shape is used. Place `after` in header for block form (`defer cleanup after close: ...`). Nested defers inside a defer body flush before parent defer completes. `defer` may appear only inside executable blocks (not absolute top level).
  - Examples:
    ```shakar
    fn run():
      defer cleanup: conn.close()
      conn := connect()
      use(conn)
    fn ordered():
      defer second after first: log("second")
      defer first: log("first")
    ```

---

## Tooling & Ecosystem

### CLI & developer tools

- **Formatter** (`shk fmt`): canonical spacing; can normalize punctuation vs keyword guards per style.
- **REPL** (`shk repl`): auto-print last value; `:desugar` shows lowerings.
- **Linter** (initial rules): suggest `a or= b` / `=a or b` over `a = a or b`; warn on long implicit `.` chains; prefer chosen guard style; warn on tiny views over huge bases (suggest `.compact()`/`.own()`).
- **Diagnostics**: fix-its like “`.` cannot stand alone; start from a subject or use `bind`.”
- **AI-friendly**: `shk --desugar` / `--resugar`; machine-readable feature manifest `shakar.features.json`; lenient parse `--recover` to auto-fix common near-misses.

### Tracing (built-in span helper)

- `trace` is a function with block sugar; measures duration, captures errors, returns body value. Example:
  ```shakar
  trace "load_users":
    u := getUsers()
    log(u.len)
  users = trace "map+trim": users.map&(.trim())
  trace "fetch" { user: id, cold: fromNetwork }:
    await[any]( fetchUser(id): ., timeout 200: [] )
  ```

### Feature gating & style profiles

- Project config (e.g., `shakar.toml`) with per-module overrides:
  ```shakar
  [shakar]
  version = "0.1"
  allow   = ["punctuation_guards","getter_properties","using","defer","dbg"]
  deny    = ["keyword_aliases","autocall_nullary_getters"]

  [style]
  prefer_guards           = "punctuation"
  prefer_default_assign   = "or="
  max_inline_guard_branches = 3

  [module."core/auth"]
  deny  = ["implicit_lambdas"]
  allow = ["?ret"]
  ```
- Version pinning guards against future keyword additions. Allow/deny gate syntax families & stdlib sugars; formatter honors style preferences. Bitwise operators stay behind `bitwise_symbols` (set/Map algebra unaffected).

### Formatter / lint rules (normative)

- `IndentBlock` = off-side indented form; `InlineBody` = single `SimpleStmt` or braced inline `{…}`. Punctuation guards remain block-only via `IndentBlock`.
- **Unary `+` invalid**; use explicit conversions.
- Selector literal style: prefer bare names/numbers for bounds; require `{…}` for non-trivial expressions; no top-level `(…)` inside backticks.
- Discourage inline backticks inside `[]` (allowed but warn); prefer flattened form unless reused selector values.
- Selector values in `[]` are flattened; selector values in collection literals are not. Selector literal values use backticks around the body (no brackets inside).
- Commas: single space after commas; no space before; no spaces around braces (e.g., `user.{a, b}`).
- Map index with default: `m[key, default: expr]` with one space after comma, none around `:`.
- Placeholder partials: prefer `&` for single hole; use `?` when 2+ holes; avoid mixing `&` and `?` in same call.
- Head binders: prefer implicit head binders when free/unqualified; use `over[...]`/`bind` to shadow/disambiguate; do not mix `over[...]` and `bind` in same head.
- Tokenization: write `!in` as single token (`a !in b`); selector interpolation braces parse normal expressions (no comments).
- Comparison comma-chains with `or`: prefer repeating `, or` or grouping the `or` cluster.
- Flag free `.` as error.
- Auto-paren guard heads with `:`; warn if missing.
- Lists stay tight: no blank lines between sibling bullets; collapse 3+ blank lines to one; one blank line after headings; trailing spaces removed; formatter emits bullets with `-`.
- Await any/all body size: prefer inline body only for single simple stmt; otherwise use block.
- Feature gate note: numeric bitwise operators `& | ^ << >>` and compounds gated by `bitwise_symbols`; set/map algebra `^` always enabled. Unary bitwise NOT (`~`) is NOT supported; use `bit.not(x)`.

---

## Appendix: Grammar Sketch

```shakar
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
UnaryPrefixOp   ::= "-" | "not" | "!" | "$" | "++" | "--" | "await" ;

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
WhileStmt       ::= "while" Expr ":" (InlineBody | IndentBlock) ;

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

#### Anchor semantics notes

```shakar
# Bare parentheses never start an implicit-chain. Use " .(...) " to call the ambient anchor.
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

---

## Roadmap (v0.2)

- **Match constructs** (Rust model): match statement (exhaustive, block-based) and match guards inside guard chains (disambiguated by `match`).
  ```shakar
  match subject:
  | { type: "A" }: ...
  | { type: "B" }: ...
  for events:
    .type == "system": handle_sys(.)
    | match { x, y }: handle_geo(x, y)
    | match { age } and age > 18: handle_adult(age)
    |: log("ignored")
  ```
- **Structural match JIT**: compile schema literals for `~` into cached bytecode validators.
- **Type contracts**: ✅ Implemented.
  - **Parameter contracts**: Use `fn name(arg ~ Schema, ...)` syntax. Desugars to runtime asserts injected at function entry.
  - **Return contracts**: Use `fn name(args...) ~ ReturnSchema:` syntax. Validates return value matches schema at runtime.
  ```shakar
  # Parameter contracts
  fn process(id ~ Int, user ~ UserSchema):
    # Auto-injected: assert id ~ Int
    # Auto-injected: assert user ~ UserSchema
    ...

  # Return contracts
  fn divide(a ~ Int, b ~ Int) ~ Float:
    a / b  # Validates return value ~ Float

  # Combined parameter and return contracts
  fn safe_divide(a ~ Int, b ~ Int) ~ Union(Float, Nil):
    if b == 0:
      nil
    else:
      a / b

  # Works with inline bodies
  fn inc(x ~ Int) ~ Int: x + 1

  # Works with anonymous functions
  square := fn(x ~ Int) ~ Int: x * x
  ```
- **Optional fields**: ✅ Implemented. Use `key?: Schema` syntax or `Optional(Schema)` function for optional object fields in schemas.
- **Union types**: ✅ Implemented. Use `Union(Schema1, Schema2, ...)` to allow multiple type alternatives. Future: `Type1 | Type2` syntax (requires parser context disambiguation).
- **Conditional apply-assign `.?=`**: assign only if RHS (evaluated with old LHS as `.`) is non-nil; today use `=<LHS> ??(.transform()) ?? .` or `<LHS> .= ??(.transform()) ?? .`.
- **Keyword aliases**, **autocall nullary methods**, **copy-update `with` sugar**, **pipes `|>`**, **nested/multi-source comprehensions**, **word range aliases `to`/`until`**, **until loops**, **sticky subject `%expr`** (anchor stays sticky; selectors/`.=`/statement-subject tails still win).
