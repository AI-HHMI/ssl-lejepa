"""Scoring a branch with mia-evals, and keeping the best of a trunk's: mia_pq runs mia-evals as a library (a git dependency on its
lib-namespace branch, see pyproject.toml; its commit is recorded in each score's provenance, mia_evals_commit) on the affinities a
branch wrote, and keep_if_best moves the best branch's weights to the trunk's best.pt."""

import fcntl
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from lib.task_utils import CODE_ROOT
from lib.util import code_provenance

BOARD_CONFIGS = {"neuron-instance": "mia_evals/gary_comparison_neuron_instance/mws.toml"}  # Trunk.board: ours, truth_kind = "instances"


def mia_pq(t, savedir: str) -> float:
    """Score the affinities in savedir (probe/{fit,test}, written by a branch's probe or finetune) as mia-evals' neuron-instance task on trunk t's board
    (mutex watershed + size filter, fitted on the fit block, reported on the test block); panoptic quality, which ranks, comes
    back (VOI and ARE are in the record).

    Writes the record into the run's own dir, savedir/mia_evals/<task>/records/*.json (pull.sh brings it down;
    analysis_plots.mia_evals_table reads it), not mia-evals' repo. resolved_config.json (params) and git_commit.txt (the
    commit that ran) go in the savedir first: mia-evals copies them into the record from run_dir.
    Calls mia_evals.evaluate.cmd_score (what `mia-evals score` runs) with this commit's board config. Deletes savedir/probe/
    (the artifacts it read) once the record is written.
    """
    from argparse import Namespace

    from mia_evals import evaluate
    savedir = Path(savedir)
    assert (savedir / "probe/test").is_dir() and (savedir / "probe/fit").is_dir(), f"no probe artifacts in {savedir}: probe or finetune first"
    (savedir / "resolved_config.json").write_text(json.dumps({"savedir": str(savedir), "trunk": asdict(t)}) + "\n")
    (savedir / "git_commit.txt").write_text(code_provenance()["commit_id"] + "\n")
    config = CODE_ROOT / BOARD_CONFIGS[t.board]  # the snapshot's copy when run as a job
    root = savedir.resolve()
    evaluate.cmd_score(Namespace(
        config=config, test=root / "probe/test", val=root / "probe/fit", leaderboard=root / "mia_evals", run_dir=root,
        no_scored=True,  # don't keep the 5.8 GB post-processed labelling per run
        scored_out=None, scratch=None))
    shutil.rmtree(savedir / "probe")  # the artifacts are scored (the record is in mia_evals/): GBs per run, not kept
    record_file = sorted(savedir.glob("mia_evals/*/records/*.json"))[-1]
    return json.loads(record_file.read_text())["scores"]["voxel_instance"]["pq"]

def keep_if_best(t, savedir: str, score: float):
    """Keep the branch's weights (savedir/checkpoint.pt, its end) as its trunk's best.pt if score beats the best so far (best.json, in the
    trunk's dir); else delete them. The scores of one trunk's branches finish in any order, so the compare-and-move is under a lock
    held only for it (a rename, not a copy)."""
    trunk, weights = Path(savedir).parent, Path(savedir) / "checkpoint.pt"
    step = int(Path(savedir).name.split("_")[1])
    with open(trunk / "best.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # released when the file closes
        best = json.loads((trunk / "best.json").read_text())["score"] if (trunk / "best.json").is_file() else float("-inf")
        if score > best:
            weights.replace(trunk / "best.pt")
            (trunk / "best.json").write_text(json.dumps({"score": score, "step": step, "board": t.board, "from": str(savedir)}) + "\n")
        else:
            weights.unlink()
    print(f"{savedir}: score {score:.4f} {'beats' if score > best else 'does not beat'} the best so far, {best:.4f}", flush=True)
