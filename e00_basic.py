from llm.models import Lejepa, LejepaConfig
import lmd_catalog as lmd
from miao.config import MiaoConfig
from miao import VolumeDataset
from rich import print as pprint

volumes = [x.to_miao() for x in lmd.all() if "flyliconn" in x.name]
mcfg = MiaoConfig(
    volumes=volumes,
    patch_size=[104, 232, 232],
    resolutions=[[25.0, 10.0, 10.0]],
    samples_per_epoch=1000,
    sampling="random",
    output_axes="lzyx",
)
# dl = VolumeDataset(mcfg)
pprint(mcfg)

savedir = "outdir/e00/main/basic/"

cfg = LejepaConfig(
    n_layers = 12,
    width = 512,
    views = 'basic',
    profile=savedir + 'profile.out', ## Turn on profiling, which uses pytorch profiler and writes to this file
)
model = Lejepa(cfg)
pprint(model)



