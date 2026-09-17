from llm.models import Lejepa, LejepaConfig
import lmd_catalog as lmd
from miao.config import MiaoConfig
from miao import VolumeDataset
from rich import print as pprint
import os

lmd.set_data_root("/Volumes/miaai/lmd-v0.0.1/data")
volumes = [x.to_miao() for x in lmd.all() if "flyliconn" in x.name]
mcfg = MiaoConfig(
    volumes=volumes,
    patch_size=[104, 232, 232],
    resolutions=[[25.0, 10.0, 10.0]],
    samples_per_epoch=1000,
    sampling="random",
    output_axes="lzyx",
)
dl = VolumeDataset(mcfg)
pprint(mcfg)
pprint(dl[0])

import sys
sys.exit(0)

savedir = "outdir/e00/main/basic/"
os.makedirs(savedir, exist_ok=True)

cfg = LejepaConfig(
    n_layers = 12,
    width = 512,
    views = 'basic',
    profile=savedir + 'profile.out', ## Turn on profiling, which uses pytorch profiler and writes to this file
)
model = Lejepa(cfg)
pprint(model)

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"We're using torch device {device} .")
net = model.to(device)
opt = torch.optim.Adam(model.parameters(), lr = 1e-4)

for ep in range(100):
    x = torch.rand(mcfg.patch_size)
    out = model(x)
    out.loss.backwards()
    # init_weights(net)
    opt.step()
    opt.zero_grad()
    print("\033[F",end='') ## move cursor UP one line 
    # print(f"finished epoch {ep+1}/{100}, loss={torch.rand():4f}, dt={dt:4f}, rate={N_pix/dt:5f} Mpix/s", end='\n',flush=True)
    print(f"finished epoch {ep+1}/{100}, loss={torch.rand():4f},", end='\n',flush=True)

