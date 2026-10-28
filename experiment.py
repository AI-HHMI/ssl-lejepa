"""e13-cellmap-12h: e12's six runs in this format (e12's are named d0-d5). Do CellMap's organelle EM, the masked cube's size (7^3 tokens,
e12's, or e11's 5^3) and the training objective change what 12 h of pretraining gives a linear probe on hemibrain? Every run shares
BASE (qk_norm, the lr falling linearly to 0 over total_steps, milestones that double from 5k steps, each evaluated as saved) and
differs by the fields its line names. total_steps is the pace we expect in 12 h less ~5%, so the lr reaches 0 as the budget ends (a
slower run stops before it): e12's for the MAE and views, a guess for the masked runs (`lsf benchmark mask7` first).
Run it: `sh jrun.sh exp experiment.py submitall`, then `lsf <task> <run>`; `python experiment.py -h` lists the commands and tasks."""

from functools import partial

from lib.jobs import B300, CPUS_PER_GPU, H200, Experiment, main
from lib.specs import Mae, Masked, Train, Views

DATA = ("hemibrain", "fish2", "liconn_mouse", "microns", "cellmap")
BATCH = 32  # per GPU
mask = partial(Masked, shift=32, lamb=10 / BATCH, sigreg_over="both")  # the targets are a second view, shifted up to 32 voxels from the context's

BASE = Train(
    data=DATA, data_power=0.4, eval_on="hemibrain",
    patch_size=(128, 128, 128), global_size=(96, 96, 96), batch_size=BATCH, n_workers=CPUS_PER_GPU - 1,
    max_hours=12, milestone_every=5_000, milestone_doubling=True,
    lr_decay=True, qk_norm=True, weight_decay=0.05, adam_beta2=0.95, cudagraphs=True,
)

EXPERIMENT = Experiment("e00/e13-cellmap-12h", {
    "mae":          BASE.but(model=Mae(decoder=(512, 8)),                       total_steps=325_000, **H200),
    "views":        BASE.but(model=Views(local=96, batch_views=True, lamb=0.1), total_steps=100_000, **H200),
    "mask7":        BASE.but(model=mask(cube=56),                               total_steps=62_500,  **B300),
    "mask5":        BASE.but(model=mask(cube=40),                               total_steps=62_500,  **B300),
    "mask7-nocell": BASE.but(model=mask(cube=56),                               total_steps=125_000, **H200, data=DATA[:-1]),
    "mask7-cell":   BASE.but(model=mask(cube=56),                               total_steps=62_500,  **B300, data=("cellmap",)),
})

if __name__ == "__main__":
    main(EXPERIMENT)
