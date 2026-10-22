# Pseudocode syntax

Python 3 syntax, plus four extensions. Fenced as ```python (for highlighting); e.g. masked.md.

1. **Prose expressions.** A backtick span, `` `tgt tokens whose centres fall in cube` ``, is an expression whose value the
   prose describes. It may stand anywhere an expression may (`visible = `ctx tokens outside cube``, `sigreg(`tgt_emb over
   batch`)`), holds no backticks and doesn't cross lines. (Python 2's backtick `repr`, removed in Python 3, is not meant.)
2. **Trailing-operator continuation.** A line ending in a binary operator (`+ - * / // % @ ** & | ^ << >> and or ??`, a
   comparison, or `,`) continues on the next line, at any indentation:

   ```python
   loss = mse(pred_emb, target) +
           lamb * sigreg(`tgt_emb over batch`)
   ```

   Python's own continuation inside brackets still holds; a line beginning with an operator does not continue the line above.
3. **`go` statements**, as in Go. `go f(args)` runs the call concurrently and moves straight on: the arguments are
   evaluated first, in the spawning thread; the call's result is discarded. Its operand must be a call, and `go` is a
   statement, not an expression, so `go` is not a name. Anything that waits for or hears from the spawned call is written
   out (a queue, `` `wait for every worker` ``):

   ```python
   for shard in shards:
       go encode(shard, out=queue)
   ```
4. **`??` nullish coalescing**, as in JS/TS. `a ?? b` is `a` unless `a` is None, then `b`; `b` is evaluated only then.
   Unlike `a or b` it keeps a falsy `a` (0, `""`, an empty list) and never asks a tensor for its truth value:
   `config = config ?? MaskConfig()`. It binds more loosely than `or` and more tightly than `x if c else y`, chains
   left to right (`a ?? b ?? c`: the first that isn't None), and needs parentheses when mixed with `and` or `or`, as in JS.
   JS's `?.` optional chaining is deliberately not included: missing data fails an assertion rather than travel on as None
   (CLAUDE.md).

Everything else is Python: names are ASCII, short and meaningful (single letters only for names that live a line or two),
and names used without definition get their domain in a header comment, `name : type, meaning   value`. Tensor dims in
comments follow CLAUDE.md (`B N D`, no commas).

Checking a block: replace each backtick span with a placeholder name, join each line that ends in an operator to the next,
drop the `go` that begins a statement, rewrite `a ?? b` as `(a if a is not None else b)`, then `ast.parse` the result.
