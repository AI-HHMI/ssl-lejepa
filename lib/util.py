"""Small helpers for experiment scripts."""

import inspect
import subprocess
import sys


def pick_entrypoint(namespace=None):
    """Pick and call a top-level function from the caller (or supplied globals)."""
    if namespace is None:
        namespace = sys._getframe(1).f_globals
    functions = {
        name: fn for name, fn in namespace.items()
        if inspect.isfunction(fn) and fn.__module__ == namespace["__name__"]
        and fn.__qualname__ == name and fn is not pick_entrypoint
    }
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
            annotation = next(iter(signature.parameters.values())).annotation
            try:
                value = value if annotation in (str, "str") else int(value)
            except ValueError:
                pass
            args.append(value)
    try:
        signature.bind(*args)
    except TypeError as exc:
        raise SystemExit(f"{fn.__name__}{signature}: {exc}")
    return fn(*args)
