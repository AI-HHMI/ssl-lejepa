"""Small helpers for experiment scripts."""

import inspect
from hashlib import sha256
import subprocess
import sys
import numpy as np


def git_provenance(repo="."):
    """Return HEAD, tracked working-copy diff, and its SHA-256 (Spearmint format).

    Includes staged and unstaged changes; excludes untracked files. Does not log.
    """
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    commit_id = git("rev-parse", "HEAD")
    diff = git("diff", "HEAD")
    return {"commit_id": commit_id, "diff_hash": sha256(diff.encode()).hexdigest(), "diff": diff}


def logish_samples(base, factors, final, N):
    base = np.array(base, dtype='int32')
    res = []
    res.append(base.copy())
    for i in range(N+1):
        j = i % len(factors)
        res.append(base * factors[j])
        if j==len(factors)-1:
            base = base * final
    res = [list(int(xi) for xi in x) for x in res]
    return res

def _entrypoints(namespace):
    return {
        name: fn for name, fn in namespace.items()
        if inspect.isfunction(fn) and fn.__module__ == namespace["__name__"]
        and fn.__qualname__ == name
    }


def call_entrypoint(name, *args, namespace=None):
    """Call a script function with positional string/int arguments."""
    if namespace is None:
        namespace = sys._getframe(1).f_globals
    functions = _entrypoints(namespace)
    if name not in functions:
        raise SystemExit(f"Unknown entrypoint {name!r}. Choose from: {', '.join(sorted(functions))}")
    fn = functions[name]
    signature = inspect.signature(fn)
    try:
        bound = signature.bind(*args)
    except TypeError as exc:
        raise SystemExit(f"{name}{signature}: {exc}")

    def convert(value, annotation):
        if annotation in (str, "str"):
            return value
        try:
            return int(value)
        except ValueError:
            if annotation in (int, "int"):
                raise SystemExit(f"{name}{signature}: expected an integer, got {value!r}")
            return value

    for key, value in bound.arguments.items():
        parameter = signature.parameters[key]
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            bound.arguments[key] = tuple(convert(v, parameter.annotation) for v in value)
        else:
            bound.arguments[key] = convert(value, parameter.annotation)
    return fn(*bound.args, **bound.kwargs)


def pick_entrypoint(namespace=None):
    """Pick and call a top-level function from the caller (or supplied globals)."""
    if namespace is None:
        namespace = sys._getframe(1).f_globals
    functions = _entrypoints(namespace)
    try:
        choice = subprocess.run(
            ["fzf", "--prompt=Entry point> ", "--height=40%", "--reverse"],
            input="\n".join(sorted(functions)), capture_output=True, text=True,
        )
    except FileNotFoundError:
        raise SystemExit("Install fzf to use the entrypoint picker.")
    if choice.returncode in (1, 130):
        return  # No match or cancelled.
    choice.check_returncode()
    fn = functions[choice.stdout.strip()]
    signature = inspect.signature(fn)
    args = []
    if signature.parameters:
        try:
            value = input(f"{fn.__name__}{signature} argument (blank to omit): ")
        except (EOFError, KeyboardInterrupt):
            return
        if value:
            args.append(value)
    return call_entrypoint(fn.__name__, *args, namespace=namespace)
