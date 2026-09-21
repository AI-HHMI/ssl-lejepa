from __future__ import annotations

from dataclasses import dataclass

from lib.models import Lejepa, LejepaConfig
import lmd_catalog as lmd
from miao.config import MiaoConfig
from miao import VolumeDataset
from rich import print as pprint
import os, sys
import torch
import json
import time

# savedir = "outdir/e00/main/basic/"

@dataclass(slots=True)
class Params:
    savedir: str = "outdir/e00/main/basic/"
    patch_size: tuple[int,int,int] = (104, 232, 232)
    
def allparams():
    params = []
    ps = [
        (2**3  , 2**3*3, 2**3*3),
        (2**2*3, 2**2*9, 2**2*9),
        (2**4  , 2**4*3, 2**4*3),
        (2**3*3, 2**3*9, 2**3*9),
        (2**5  , 2**5*3, 2**5*3),
        (2**4*3, 2**4*9, 2**4*9),
        (2**6  , 2**6*3, 2**6*3),
        (2**5*3, 2**5*9, 2**5*9),
    ]
    for i, _p in enumerate(ps):
        p = Params()
        p.patch_size = _p
        p.savedir = f"outdir/e00/main/basic/d{i}/"
        params.append(p)
    return params

def run(n:int):
    par = allparams()[n]
    # lmd.set_data_root("/Volumes/miaai/lmd-v0.0.1/data")
    # volumes = [x.to_miao() for x in lmd.all() if "flyliconn" in x.name]
    volumes = [x.to_miao() for x in lmd.all() if x.name == "exm-drosophila-flyliconn-matt-260601-60X-B4-2-045/crop-001"]
    pprint(volumes)

    mcfg = MiaoConfig(
        volumes=volumes,
        patch_size=par.patch_size,
        resolutions=[[25.0, 10.0, 10.0]],
        samples_per_epoch=1000,
        sampling="random",
        output_axes="lzyx",
    )
    dl = VolumeDataset(mcfg)
    # pprint(mcfg)
    # pprint(dl[0])

    os.makedirs(par.savedir, exist_ok=True)
    metrics_file = open(par.savedir + "metrics.jsonl", 'a')

    cfg = LejepaConfig(
        n_layers = 12,
        width = 512,
        views = 'basic',
        profile=par.savedir + 'profile.out', ## Turn on profiling, which uses pytorch profiler and writes to this file
    )
    model = Lejepa(cfg)
    pprint(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"We're using torch device {device} .")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = 1e-4)
    # init_weights(net)
    n_epoch = 1_000

    for ep in range(n_epoch):
        # x = torch.rand(mcfg.patch_size)
        x = dl[ep]
        out = model(x['img'])
        out.loss.backwards()
        opt.step()
        opt.zero_grad()

        if ep%10==0:
            metrics_file.write(json.dumps({"tbl":"metrics", "epoch":ep, "time":time.time(), "loss":float(out.loss.detach().item())}) + '\n')
            metrics_file.flush()

        print("\033[F",end='') ## move cursor UP one line 
        # print(f"finished epoch {ep+1}/{100}, loss={torch.rand():4f}, dt={dt:4f}, rate={N_pix/dt:5f} Mpix/s", end='\n',flush=True)
        print(f"finished epoch {ep+1}/{n_epoch}, loss={out.loss.detach():4f},", end='\n',flush=True)


def runlsf(n:int):
    import subprocess
    par = allparams()[n]
    RUN_NAME = "e00_basic"
    NUM_GPUS = 1
    cmd = f""" bsub -J {RUN_NAME} \
        -W 46:00 \
        -P miaai \
        -n {NUM_GPUS} \
        -R "span[hosts=1]" \
        -gpu "num={NUM_GPUS}:mode=exclusive_process" \
        -q gpu_h100 \
        -o {par.savedir}/job_%J.log \
        uv run python e00_basic.py {n}
        """
    subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"Submitted {RUN_NAME} {n} to LSF.")

def runmany():
    for i in range(len(allparams())):
        runlsf(i)

def test():
    x = allparams()
    for xi in x:
        pprint(xi)

if __name__=="__main__":
    import sys
    print(sys.argv)
    if len(sys.argv) == 1:
        assert False, "we need cmdline args"
    if sys.argv[1] == 'many':
        runmany()
    elif sys.argv[1] == 'lsf':
        runlsf(int(sys.argv[2]))
    elif sys.argv[1] == 'test':
        test()
    else:
        run(int(sys.argv[1]))
