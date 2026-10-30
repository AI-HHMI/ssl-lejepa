"""e13-03-extend: e13's mae and views trunks, 100k more steps from their final states (step 100k) on a full H200 node each: does 12 h
of pretraining more help a linear probe on hemibrain? (mask7's is e13-02.) The branches are the trunk's weights as they are
(anneal_steps=0), finetuned on RTX 6000s. mask5's line is added when its trunk is done: submitall skips the runs submitted.

Declarative: every run shares BASE (its Train) and differs by the groups its line names; its schedule is its Trunk's fields. What a
run does (tasks.py): it pretrains in segments, branches after each (finetune, score), and keeps the best branch.
Before submitting, e13's runs need a last.pt (their trunks predate it): in outdir/e00/e13-cellmap-12h/<run>/
`ln step_0100000/start.pt last.pt`, a hard link its pending branch job can't delete.
Run it: `sh jrun.sh exp experiment.py submitall`, then `resubmit_task <start or task> <run or job file>`; `python experiment.py -h` lists the
commands and tasks."""

from dataclasses import replace
from functools import partial

from lib.task_entry import main
from lib.task_utils import Experiment
from lib.training import Data, Encoder, Mae, Masked, Optim, Speed, Train, Views
from tasks import TASKS, Trunk, start

DATA = ("hemibrain", "fish2", "liconn_mouse", "microns", "cellmap")
BATCH = 32  # per GPU
mask = partial(Masked, shift=32, lamb=10 / BATCH, sigreg_over="both")  # the targets are a second view, shifted up to 32 voxels from the context's

BASE = Train(
    data=Data(sets=DATA, power=0.4, window=(128, 128, 128), view=(96, 96, 96)),
    encoder=Encoder(qk_norm=True),
    optim=Optim(batch_size=BATCH, weight_decay=0.05, adam_beta2=0.95),
    speed=Speed(cudagraphs=True),
)

e13 = "outdir/e00/e13-cellmap-12h"
run = partial(Trunk, eval_data="hemibrain", segments=10, segment_steps=10_000, anneal_steps=0)  # a line of the table below
EXPERIMENT = Experiment("e00/e13-03-extend", start, {
    "mae":   run(train=replace(BASE, model=Mae(decoder=(512, 8))), init_from=f"{e13}/mae/last.pt"),
    "views": run(train=replace(BASE, model=Views(local=96, batch_views=True, lamb=0.1)), init_from=f"{e13}/views/last.pt"),
})

if __name__ == "__main__":
    main(EXPERIMENT, TASKS)
