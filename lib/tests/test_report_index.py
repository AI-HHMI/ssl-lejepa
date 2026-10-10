"""Tests for reports/index.py: the site indexes every report page by research thread, dated by its experiment's
original commit, and links results/ and outdir/, nothing else."""

import json
import os
import subprocess

import pytest

from reports import index, page


def commit(repo, date: str, message: str, *paths: str) -> str:
    """A commit in repo at this committer date; its full id."""
    env = {**os.environ, "GIT_COMMITTER_DATE": date, "GIT_AUTHOR_DATE": date}
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, env=env, check=True, capture_output=True, text=True).stdout.strip()
    if paths:
        run("add", *paths)
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "--allow-empty", "-q", "-m", message)
    return run("rev-parse", "HEAD")


def ran(run_dir, *commit_ids):
    """runs.json rows for jobs that ran from these commits."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "runs.json").write_text("".join(json.dumps({"fn": "run", "commit_id": c}) + "\n" for c in commit_ids))


def test_index_dates_each_experiment_by_its_original_commit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    old = commit(tmp_path, "2026-09-20T12:00:00", "exp: e00/b")
    late = commit(tmp_path, "2026-10-05T12:00:00", "a later probe of b")
    newer = commit(tmp_path, "2026-09-25T12:00:00", "exp: e00/a")
    page.write("e00/b", "What does b ask? <b>bold</b> & more.", [], [("S", [("c", "x")])])
    page.write("e00/a", "What does a ask?", [], [("S", [("c", "x")])])
    ran(tmp_path / "outdir/e00/b/d0", late)
    ran(tmp_path / "outdir/e00/b/d1", old, late)  # b's original commit is the oldest any of its runs ran from
    ran(tmp_path / "outdir/e00/a/d0", newer)
    (tmp_path / "results/perf_journey.html").write_text("<title>perf journey</title><p>Throughput over time.</p>")
    commit(tmp_path, "2026-09-22T12:00:00", "perf page", "-f", "results/perf_journey.html")  # results/ may be ignored
    (tmp_path / "results/perf_journey.html").write_text("<title>perf journey</title><p>Throughput over time, updated.</p>")
    commit(tmp_path, "2026-10-09T12:00:00", "perf page edit", "-f", "results/perf_journey.html")  # dated by the first commit
    monkeypatch.setattr(index, "THREADS", {"Speed": ["e00/b", "perf_journey"], "Empty": ["e00/never-ran"]})
    index.main()
    index.main()  # rerunnable: the links are already there
    site = (tmp_path / "site/index.html").read_text()
    assert f"<time>2026-09-20<br><code>{old[:8]}</code></time>" in site and late[:8] not in site
    # threads in order, newest experiment first within each (perf page 09-22 before b 09-20); unlisted pages under Unsorted
    assert site.index("id='speed'") < site.index("results/perf_journey.html") < site.index("results/e00/b/report.html") \
        < site.index("id='unsorted'") < site.index("results/e00/a/report.html")
    assert "id='empty'" not in site and "<b>2</b>research threads" in site and "<b>2026-09-25</b>newest experiment" in site
    assert "What does b ask? bold &amp; more." in site and ">e00/b<" in site and site.count("class='card wide'") == 2
    assert sorted(p.name for p in (tmp_path / "site").iterdir()) == ["index.html", "outdir", "results"]
    assert (tmp_path / "site/results/e00/a/report.html").is_file() and (tmp_path / "site/outdir/e00/a/d0").is_dir()


def test_index_refuses_to_replace_a_real_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    page.write("e00/a", "What does a ask?", [], [("S", [("c", "x")])])
    (tmp_path / "site/outdir").mkdir(parents=True)
    with pytest.raises(AssertionError, match="isn't a link"):
        index.main()
