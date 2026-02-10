# Shakar

**A subjectful scripting language.**

Shakar is a work-in-progress general-purpose scripting language. It targets boilerplate like repeated variable names, defensive null checks, and verbose read-transform-write cycles. Its core idea is **subjectful flow**, where the language tracks what you're operating on so you don't have to repeat it. Chains, guards, fan-outs, and apply-assign all share a single implicit-subject model that keeps code compact without hiding control flow.

The **source of truth** for the language is the [design notes](docs/shakar-design-notes.md). It describes the grammar, implicit subject rules for `.`, and the rest of the semantics in detail.

[**Playground**](https://b.shirkadeh.org/shakar) | [**Design Notes**](docs/shakar-design-notes.md) | [**Grammar**](grammar.ebnf)

---

## Design Goals

*   **Subject-Oriented**: Most scripting involves reading a value, transforming it, and writing it back. Shakar makes that the default path instead of a pattern you hand-wire every time.
*   **Zero Ceremony**: Implicit main, eager evaluation, and dynamic typing suitable for scripting.
*   **Tooling First**: The grammar is developed with a recursive-descent parser (Tree-sitter grammar exists but is currently unmaintained).
*   **Deterministic Lowering**: Desugaring guarantees deterministic behavioral lowering to core forms; it does not guarantee that lowered IR/AST prints as valid roundtrippable Shakar source.

## Setup & Usage

### Prerequisites

*   Python 3.10 or higher
*   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bkazemi/shakar.git
    cd shakar
    ```

2.  **Install in editable mode:**
    ```bash
    pip install -e .
    ```

### Running the Reference Interpreter

The reference implementation (`src/shakar_ref`) is a Python-based parser and interpreter used for validating the language design.

Run a script file:
```bash
shakar sample.shk
```

Or execute code from stdin:
```bash
echo 'print("Hello from Shakar!")' | shakar -
```

Start the interactive REPL (or just run `shakar` with no args).
Requires `prompt_toolkit` (installed automatically with `pip install -e .`):
```bash
shakar --repl
```

Inspect the parse tree with `--tree`:
```bash
shakar --tree path/to/file.shk
```

---

## Highlights

The following features are implemented in the reference runtime. See the [design notes](docs/shakar-design-notes.md) for the full specification.

### 1. Subjectful Statements and Apply-Assign

Shakar has a first-class notion of a "statement subject". A line that starts with `=LHS<tail>` evaluates `<tail>` with `.` bound to the current value of `LHS`, then writes the final result back to `LHS`:

```shakar
# boring, fully spelled
user.name = user.name.trim().title() ?? "guest"

# statement-subject sugar
=user.name.trim().title() ?? "guest"

line := "name=  Ada "
=line.split("=").last().trim().title()  # still assigns back to `line`

=(user).profile.contact.name.trim()  # grouping: write lands on `user`, not deeper
```

The plain form (`=name<tail>`) writes back to the deepest field walked — `=user.profile.email.trim()` writes back to `user.profile.email`. Grouping with `=(user)` overrides that: no matter how deep the tail goes, the write lands on `user`. Use that escape hatch sparingly.

Apply-assign `.=` does the same thing at expression level: inside the right-hand side, `.` is the old value of the left-hand side and the result is written back in place:

```shakar
user.name .= .trim().title() ?? user.name

# fan-out over a selector
user.{name, email} .= ??(.trim().lower()) ?? "unknown"

cfg["db", default: {}]["host"] .= .trim() ?? "localhost"
```

Both forms are built on the same [implicit-subject / anchor model](docs/shakar-design-notes.md#the-anchor-model-) rules, which makes "update this thing based on its old value" cheap to write without making the flow mysterious.

#### Implicit Subject and Anchor Stack

The implicit subject `.` only exists inside constructs that explicitly bind it (`=LHS<tail>`, `.=`, subjectful `for` loops, selectors, amp-lambdas). It never leaks across statements.

The first explicit subject in an expression sets the **anchor**; leading-dot chains hang off it:

```shakar
user and .profile and .id
# `user` sets the anchor
# `.profile` → user.profile, `.id` → user.id
```

Parentheses isolate anchor changes — retargeting inside a group doesn't leak out:

```shakar
user and (other and .name) and .id
# inside group: `other` retargets, `.name` → other.name
# `.id` uses outer anchor → user.id
```

`$expr` suppresses retargeting on a single unit without parens:

```shakar
for items:
  .price > $config.limits.max: flag(.)
# $config doesn't retarget; `.price` and `.` use the loop element
```

See the [anchor model](docs/shakar-design-notes.md#the-anchor-model-) in the design notes for the full rules.

### 2. Control Flow Guards

Vertical punctuation guards replace `if/elif/else` chains to keep the primary logic flow linear. See [Guards & postfix conditionals](docs/shakar-design-notes.md#guards--postfix-conditionals).

```shakar
ready():
  start()
| retries < 3:
  retry()
|:
  log("not ready")
```

Each `| expr:` is an `elif`, and `|:` is the final `else`. Single-line forms (`stmt if cond` and `if cond: stmt`) are still available.

### 3. Selectors, Fan-out & Broadcasting

[Selectors](docs/shakar-design-notes.md#selector-values-slices-and-selectors) address multiple fields or regions of a structure in one shot. Fan literals broadcast an operation across multiple values.

```shakar
# Update multiple fields at once
user.{name, email} .= .trim()

# Fieldfan chaining: navigate fields after the fanout
state.{a, b}.nested .= .update()

# Fan-out block: anchors `.` to state, runs clauses top→down
state{ .cur = .next; .x += 1; .name .= .trim() }

# Fan literal: broadcast a method call
fan{db, cache, worker}.restart()

# Selector literals in comparisons
level == `warn, error`, or >= `critical:`:
  notify(level)

# Destructure directly from a selector list
key, val := arr[0,1]  # binds arr[0] → key, arr[1] → val
```

Selector literals use backticks and have precise semantics for ranges, lists, and membership; the [design notes](docs/shakar-design-notes.md#selector-values-slices-and-selectors) cover the details.

### 4. UFCS (Universal Function Call Syntax)

Any in-scope callable can be invoked with method syntax. When the receiver doesn’t have a matching method or builtin, Shakar falls back to a scope lookup and calls the function with the receiver as its subject/first arg.

```shakar
fn double(x): x * 2
5.double()            # UFCS → double(5)

"hello".print()       # UFCS → print("hello")

fn clamp(x, lo, hi):
  x < lo: return lo
  x > hi: return hi
  x

volume := 120
volume.clamp(0, 100)

for paths:
  .read().print()
```

UFCS only applies to **method-call syntax** (`recv.name(...)`). Plain field access (`recv.name`) never falls back.

### 5. Collection Flow & In-place Mutation

Built-in collection methods leverage UFCS and [Amp-lambdas](#8-other-features) to provide concise transformation and filtering. Shakar distinguishes between **pure** transformations (creating new arrays) and **in-place** mutations (modifying the original for performance).

```shakar
arr := [1, 2, 3, 4]

# Pure: returns new arrays
doubled := arr.map&(. * 2)
evens   := arr.filter&(. % 2 == 0)

# In-place: modifies `arr` directly (Zero allocation)
arr.update&(. + 10)  # arr is now [11, 12, 13, 14]
arr.keep&(. > 12)    # arr is now [13, 14]

# Objects: update all values
config := { timeout: 5sec, retries: 3 }
config.update&(. * 2) # config is now { timeout: 10sec, retries: 6 }
```

### 6. Comparison Comma-Chains (CCC)

CCC locks in a subject once and lets you stream more comparisons with commas, avoiding repeated left operands. The joiner defaults to `and` and sticks until changed; `and` legs can inherit the previous comparator, but `, or` legs must restate the comparator they want. See [Comparison & identity (CCC)](docs/shakar-design-notes.md#comparison--identity-ccc).

```shakar
temp > 40, < 60, != 55:   # temp > 40 and temp < 60 and temp != 55
  fan.on()

# AND legs inherit comparator, OR legs must restate
alert := temp > 70, or >= limits.max   # temp > 70 or temp >= limits.max
ok    := temp >= 45, and <= 65, or == preferred   # temp >= 45 and temp <= 65 or temp == preferred
```

### 7. Decorators

Attach lightweight wrappers to functions. Decorator bodies see `f` (the next callable) and `args` (positional arguments as a mutable array). If the body finishes without `return`, the runtime calls `f(args)` automatically. Stacking is deterministic: the decorator closest to `fn` runs first. See [Decorators](docs/shakar-design-notes.md#decorators).

```shakar
decorator log_calls():
  log += [args[0]]

decorator with_prefix(prefix):
  args[0] = prefix + args[0]

@log_calls
@with_prefix("[prefix] ")
fn announce(msg):
  msg

announce("ready")   # => "[prefix] ready"
log                 # => ["ready"]
```

### 8. Call Blocks and Emit `>`

Call blocks bind a callable as an emit target for the duration of a block. Inside, `>` invokes that target. See [Call blocks](docs/shakar-design-notes.md#call-blocks-call-and-emit-).

```shakar
call self.expect:       # binds self.expect as the emit target
  > TT.IN               # self.expect(TT.IN)
  expr := > TT.EXPR     # expr := self.expect(TT.EXPR)
  > TT.COLON             # self.expect(TT.COLON)
```

### 9. Concurrency (Experimental)

The reference runtime includes prototypes for structured concurrency primitives. *Note: These are active areas of design experimentation.* See [Channels & Wait](docs/shakar-design-notes.md#channels--wait).

```shakar
# Channels
ch := channel(1)
"msg" -> ch

# Spawning & Waiting
task := spawn fetch_data()
result := wait task
```

### 8. Other Features

*   **Nil-safe chains & coalescing**: Prefix `??(...)` evaluates a chain and short-circuits to `nil` on the first failure; infix `??` is null-coalescing. Together: `nickname := ??(user.profile.nick.trim()) ?? "guest"`. See [Null safety](docs/shakar-design-notes.md#null-safety).
*   **Match expressions**: `match val: pattern: body` — see [Match Expression](docs/shakar-design-notes.md#match-expression-v01).
*   **Amp-lambdas**: `&(. * 2)` and `&[a, b](a + b)` — see [Amp-lambdas](docs/shakar-design-notes.md#amp-lambdas--and-implicit-parameters).
*   **Imports**: `import "path"`, `import[a, b] "path"` — see [Modules & Imports](docs/shakar-design-notes.md#modules--imports).

---

## Development Status

*   **Spec**: [`docs/shakar-design-notes.md`](docs/shakar-design-notes.md) is the source of truth.
*   **Runtime**: `src/shakar_ref/` contains the Python reference implementation.
*   **Grammar**: `tree-sitter-shakar/` contains the Tree-sitter grammar (currently unmaintained; may lag behind the spec).
*   **Hooks / decorators**: implemented.
*   **Concurrency**: channels, spawn, wait — experimental.
*   **Tests**: run `pytest tests/ --tb=short` after semantic changes; it exercises parser, AST, and runtime behavior end-to-end.

### Running Tests

To verify the grammar and runtime against regression tests:

```bash
pytest tests/ --tb=short
```

---

This is early research. Expect breaking changes while the surface syntax settles. Always treat the [design notes](docs/shakar-design-notes.md) as the source of truth. Contributions that improve the spec, runtime, or tests are welcome :-)
