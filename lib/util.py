"""Small helpers for experiment scripts."""

import inspect
import subprocess
import sys


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
