"""e13-cellmap-12h: e12's six runs in this format (e12's are named d0-d5). Do CellMap's organelle EM, the masked cube's size (7^3 tokens,
e12's, or e11's 5^3) and the training objective change what 12 h of pretraining gives a linear probe on hemibrain?

Declarative: every run shares BASE (its Train) and differs by the groups its line names; its schedule is its Trunk's fields. What a
run does (tasks.py): it pretrains in segments, branches at its start and after each, scores each branch, and keeps its best.
Run it: `sh jrun.sh exp experiment.py submitall`, then `resubmit <task> <run or job file>`; `python experiment.py -h` lists the
commands and tasks."""

from dataclasses import replace
from functools import partial

from lib.task_utils import B300, H200, Experiment, main
from lib.training import Data, Encoder, Mae, Masked, Optim, Speed, Train, Views
from tasks import TASKS, Trunk, trunk

DATA = ("hemibrain", "fish2", "liconn_mouse", "microns", "cellmap")
BATCH = 32  # per GPU
mask = partial(Masked, shift=32, lamb=10 / BATCH, sigreg_over="both")  # the targets are a second view, shifted up to 32 voxels from the context's

BASE = Train(
    data=Data(sets=DATA, power=0.4, window=(128, 128, 128), view=(96, 96, 96)),
    encoder=Encoder(qk_norm=True),
    optim=Optim(batch_size=BATCH, weight_decay=0.05, adam_beta2=0.95),
    speed=Speed(cudagraphs=True),
)

run = partial(Trunk, eval_data="hemibrain", segments=10, anneal_steps=10_000, anneal_to=0.01, hours=12)  # a line of the table below
EXPERIMENT = Experiment("e00/e13-cellmap-12h", trunk, {
    "mae":          run(train=replace(BASE, model=Mae(decoder=(512, 8)),                       hardware=B300), segment_steps=10_000),
    "views":        run(train=replace(BASE, model=Views(local=96, batch_views=True, lamb=0.1), hardware=B300), segment_steps=10_000),
    "mask7":        run(train=replace(BASE, model=mask(cube=56),                               hardware=B300), segment_steps=10_000),
    "mask5":        run(train=replace(BASE, model=mask(cube=40),                               hardware=B300), segment_steps=10_000),
    "mask5-nocell": run(train=replace(BASE, model=mask(cube=40),                               hardware=B300, data=replace(BASE.data, sets=DATA[:-1])), segment_steps=10_000),
    "mask5-cell":   run(train=replace(BASE, model=mask(cube=40),                               hardware=B300, data=replace(BASE.data, sets=("cellmap",))), segment_steps=10_000),
})

if __name__ == "__main__":
    main(EXPERIMENT, TASKS)
