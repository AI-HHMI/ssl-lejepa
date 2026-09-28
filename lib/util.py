"""Small helpers for experiment scripts."""

import inspect
import json
import os
import re
from hashlib import sha256
from pathlib import Path
import socket
import subprocess
import sys
import time
import numpy as np
import shutil

def trash(path) -> Path:
    """Empty dir `path` (under outdir/) for a fresh run: its old contents move to outdir/.trash/<path>/<time>/,
    so a resubmission never destroys results. Empty .trash by hand; pull.sh doesn't copy it."""
    path = Path(path)
    if path.is_dir() and any(path.iterdir()):
        dest = Path("outdir/.trash") / path.relative_to("outdir") / time.strftime("%Y%m%d-%H%M%S")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, dest)
    path.mkdir(parents=True, exist_ok=True)
    return path


def snapshot(paths, dest) -> Path:
    """Copy files/dirs (relative to the cwd) into dest, for jobs that must run exactly this code.

    Skipped if dest already holds identical content, so repeated calls (one per job in a sweep) don't rewrite
    files a started job may be importing. Each file is written to a temp name and os.replace()d into place.
    Always (re)writes dest/provenance.json with the commit being copied: see code_provenance().
    """
    dest = Path(dest)
    files = sorted(f for p in map(Path, paths) for f in ([p] if p.is_file() else p.rglob("*.py")) if "__pycache__" not in f.parts)
    assert files, f"nothing to snapshot in {paths}"
    digest = sha256(b"".join(str(f).encode() + b"\0" + f.read_bytes() for f in files)).hexdigest()
    stamp = dest / ".sha256"
    if not (stamp.is_file() and stamp.read_text() == digest):
        for f in files:
            (dest / f).parent.mkdir(parents=True, exist_ok=True)
            tmp = dest / f"{f}.tmp"
            shutil.copy2(f, tmp)
            tmp.replace(dest / f)
        stamp.write_text(digest)
    tmp = dest / "provenance.json.tmp"
    tmp.write_text(json.dumps(git_provenance()))
    tmp.replace(dest / "provenance.json")
    return dest


def repo_root(path=".") -> Path:
    """Return the absolute Git repository root containing the given directory."""
    return Path(subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"], text=True,
    ).strip())


def git_provenance(repo=".") -> dict:
    """HEAD commit, its subject line, and whether tracked files differ from it (dirty) plus that diff's SHA-256.

    With jj (colocated), HEAD is the working-copy commit's parent, so dirty means @ has changes; files jj tracks
    in @ that git doesn't yet know about (new files) don't count. Does not log.
    """
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    diff = git("diff", "HEAD")
    return {"commit_id": git("rev-parse", "HEAD"), "subject": git("log", "-1", "--format=%s"),
            "dirty": bool(diff), "diff_hash": sha256(diff.encode()).hexdigest()}


def code_provenance() -> dict:
    """git_provenance() of the code actually running: a .tmpcode snapshot's recorded commit (snapshot()) when
    the script runs from one, since the shared checkout's HEAD may have moved on since submission."""
    recorded = Path(sys.argv[0]).resolve().parent / "provenance.json"
    return json.loads(recorded.read_text()) if recorded.is_file() else git_provenance()


def log_command(argv: list[str]):
    """Append one JSON line per entrypoint call to outdir/_log/commands-<host>.jsonl: time, host, argv, LSF
    job and code_provenance(). One file per host, since appends from several hosts to one NFS file can
    interleave. Rank 0 only under torchrun. pull.sh copies the cluster's and protects the local one."""
    if os.environ.get("RANK", "0") != "0":
        return
    host = socket.gethostname().split(".")[0]
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": host, "argv": argv,
           "lsf_job": os.environ.get("LSB_JOBID"), **code_provenance()}
    path = Path("outdir/_log") / f"commands-{host}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


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

def json_equal(a, b) -> bool:
    """Equality that survives a JSON round-trip: tuples and lists compare as sequences, dicts by key."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_equal(a[k], b[k]) for k in a)
    return a == b


def trace_summary(trace_path: str | Path) -> dict[str, float]:
    """Averages over the profiled steps of a chrome trace.

    step_ms: wall time per ProfilerStep. gpu_busy: fraction of that wall time covered by GPU
    kernels/memcpys (overlaps merged). NN_PHASE_ms: host time per step in each `NN_` record_function
    phase; forward/backward host time is launch time unless the host blocks on the GPU.
    """
    events = json.loads(Path(trace_path).read_text())["traceEvents"]
    steps = [e for e in events if e.get("ph") == "X" and e.get("name", "").startswith("ProfilerStep#")]
    assert steps, f"no ProfilerStep# spans in {trace_path}"
    t0 = min(e["ts"] for e in steps)
    t1 = max(e["ts"] + e["dur"] for e in steps)
    intervals = sorted(
        (e["ts"], e["ts"] + e["dur"]) for e in events
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    )
    busy, end = 0.0, t0
    for a, b in intervals:
        a, b = max(a, end), min(b, t1)
        if b > a:
            busy += b - a
            end = b
    phases: dict[str, float] = {}
    for e in events:
        b1 = e.get("ph") == "X" and e.get("cat") == "user_annotation"
        b2 = re.match(r"\d\d_", e.get("name", "")) is not None
        if b1 and b2 and t0 <= e["ts"] < t1:
            phases[e["name"]] = phases.get(e["name"], 0.0) + e["dur"]
    n = len(steps)
    return {
        "step_ms": (t1 - t0) / n / 1e3,
        "gpu_busy": busy / (t1 - t0),
        **{f"{k}_ms": v / n / 1e3 for k, v in sorted(phases.items())},
    }


def profile_device_ms(path: str | Path) -> tuple[float, dict[str, float]]:
    """GPU ms per profiled step from a Benchmark profile.out: the total, and each row of its self-device-time table.

    Rows overlap (a CompiledFxGraph call's time includes its kernels'), so only sum disjoint rows, e.g. kernels
    picked by name. Row names are truncated by the table.
    """
    text = Path(path).read_text()
    steps = re.search(r"recorded steps \(zero-based\): (\d+)\.\.(\d+)", text)
    total = re.search(r"Self CUDA time total: ([\d.]+)(us|ms|s)", text)
    assert steps and total and "SORTED BY SELF DEVICE TIME" in text, f"{path} is not a CUDA Benchmark profile.out"
    n = int(steps[2]) - int(steps[1]) + 1
    scale = {"us": 1e-3, "ms": 1.0, "s": 1e3}
    rows: dict[str, float] = {}
    for line in text.split("SORTED BY SELF DEVICE TIME")[1].split("Self CPU time total")[0].splitlines():
        f = re.split(r"\s{2,}", line.strip())  # Name, Self CPU %, Self CPU, ..., Self CUDA (7th), ..., # of Calls
        t = re.fullmatch(r"([\d.]+)(us|ms|s)", f[6]) if len(f) == 11 else None
        if t:
            rows[f[0]] = rows.get(f[0], 0.0) + float(t[1]) * scale[t[2]] / n
    return float(total[1]) * scale[total[2]] / n, rows


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
    log_command([sys.argv[0], name, *map(str, args)])
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
