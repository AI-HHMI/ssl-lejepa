from __future__ import annotations

from dataclasses import dataclass, field, asdict
from functools import reduce

from lib.models import Lejepa, LejepaConfig
from lib.util import pick_entrypoint
import lmd_catalog as lmd
from miao.config import MiaoConfig
from miao import VolumeDataset
from rich import print as pprint
import os, sys
import json
import time


@dataclass(slots=True)
class Params:
    savedir: str = "outdir/e00/main/basic/"
    # patch_size: list[int] = [104, 232, 232]
    patch_size: list[int] = field(default_factory=lambda: [104, 232, 232])
    batch_size: int = 42
    n_epoch: int = 1000
    
def allparams():
    params = []
    ps = [
        [2**3  , 2**3*3, 2**3*3],
        [2**2*3, 2**2*9, 2**2*9],
        [2**4  , 2**4*3, 2**4*3],
        [2**3*3, 2**3*9, 2**3*9],
        [2**5  , 2**5*3, 2**5*3],
        [2**4*3, 2**4*9, 2**4*9],
        [2**6  , 2**6*3, 2**6*3],
        [2**5*3, 2**5*9, 2**5*9],
    ]
    for i, _p in enumerate(ps):
        p = Params()
        p.patch_size = _p
        p.savedir = f"outdir/e00/main/basic/d{i}/"
        params.append(p)
    return params

def run(n:int):
    import torch

    par : Params = allparams()[n]
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
    # pprint(dl[0]['img'].shape)

    os.makedirs(par.savedir, exist_ok=True)
    metrics_file = open(par.savedir + "metrics.jsonl", 'a')

    cfg = LejepaConfig(
        n_layers = 12,
        width = 512,
        views = 'basic',
        profile=par.savedir + 'profile.out', ## Turn on profiling, which uses pytorch profiler and writes to this file
        lamb = 0.1,
    )
    model = Lejepa(cfg)
    pprint(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"We're using torch device {device} .")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = 1e-4)
    # init_weights(net)
    # n_epoch = 1_000

    def batch(ep):
        stack = [dl[par.batch_size*ep + i] for i in range(par.batch_size)]
        b = torch.stack([x['img'] for x in stack])
        return b

    for ep in range(par.n_epoch):
        x = batch(ep)
        out = model(x)
        out.loss.backwards()
        opt.step()
        opt.zero_grad()

        if ep%10==0:
            metrics_file.write(json.dumps({"tbl":"metrics", "epoch":ep, "time":time.time(), "loss":float(out.loss.detach().item())}) + '\n')
            metrics_file.flush()

        print("\033[F",end='') ## move cursor UP one line 
        # print(f"finished epoch {ep+1}/{100}, loss={torch.rand():4f}, dt={dt:4f}, rate={N_pix/dt:5f} Mpix/s", end='\n',flush=True)
        print(f"finished epoch {ep+1}/{par.n_epoch}, loss={out.loss.detach():4f},", end='\n',flush=True)


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
        -q gpu_a100 \
        -o {par.savedir}/job_%J.log \
        uv run python e00_basic.py {n}
        """
    subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"Submitted {RUN_NAME} {n} to LSF.")

def runmany():
    for i in range(len(allparams())):
        runlsf(i)

def runmany_sequential():
    for i in range(len(allparams())):
        run(i)

def analysis():
    import pandas
    import plotly.express as px
    def loadAndFuse(par:Params):
        # metr = json.load(open(par.savedir + "metrics.jsonl", "r"))
        try:
            with open(par.savedir + "metrics.jsonl") as f:
                metr = [json.loads(line) for line in f if line.strip()]
            tabl = [{**m, **asdict(par)} for m in metr]
            return tabl
        except:
            return []
    params = [loadAndFuse(p) for p in allparams()]
    res = list(reduce(lambda a,b: a+b, params))

    for r in res:
        r['patch_size'] = tuple(r['patch_size'])
    res = pandas.DataFrame(res)
    pl = px.scatter(res, x="epoch", y="loss", color="patch_size")
    pl.show()
    print(res)

def test():
    x = lmd.all()
    for xi in x:
        pprint(xi)

if __name__ == "__main__":
    import sys
    print(sys.argv)
    if len(sys.argv) == 1:
        pick_entrypoint()
    elif sys.argv[1] == 'many':
        runmany()
    elif sys.argv[1] == 'lsf':
        runlsf(int(sys.argv[2]))
    elif sys.argv[1] == 'test':
        test()
    elif sys.argv[1] == 'anl':
        analysis()
    elif sys.argv[1] == 'seq':
        runmany_sequential()
    else:
        run(int(sys.argv[1]))
