from llm.models import Lejepa, LejepaConfig
import lmd_catalog as lmd
from miao.config import MiaoConfig
from miao import VolumeDataset
from rich import print as pprint
import os, sys
import torch

def run():
    # lmd.set_data_root("/Volumes/miaai/lmd-v0.0.1/data")
    # volumes = [x.to_miao() for x in lmd.all() if "flyliconn" in x.name]
    volumes = [x.to_miao() for x in lmd.all() if x.name == "exm-drosophila-flyliconn-matt-260601-60X-B4-2-045/crop-001"]
    pprint(volumes)

    mcfg = MiaoConfig(
        volumes=volumes,
        patch_size=[104, 232, 232],
        resolutions=[[25.0, 10.0, 10.0]],
        samples_per_epoch=1000,
        sampling="random",
        output_axes="lzyx",
    )
    dl = VolumeDataset(mcfg)
    # pprint(mcfg)
    # pprint(dl[0])

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"We're using torch device {device} .")
    net = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = 1e-4)
    # init_weights(net)

    for ep in range(100):
        # x = torch.rand(mcfg.patch_size)
        x = dl[ep]
        out = model(x['img'])
        out.loss.backwards()
        opt.step()
        opt.zero_grad()
        print("\033[F",end='') ## move cursor UP one line 
        # print(f"finished epoch {ep+1}/{100}, loss={torch.rand():4f}, dt={dt:4f}, rate={N_pix/dt:5f} Mpix/s", end='\n',flush=True)
        print(f"finished epoch {ep+1}/{100}, loss={out.loss.detach():4f},", end='\n',flush=True)


def rungpu():
    import subprocess
    RUN_NAME = "e00_basic"
    NUM_GPUS = 1
    cmd = f""" bsub -J {RUN_NAME} \
        -W 46:00 \
        -P miaai \
        -n {NUM_GPUS} \
        -R "span[hosts=1]" \
        -gpu "num={NUM_GPUS}:mode=exclusive_process" \
        -q gpu_h100 \
        -o ./logs/${RUN_NAME}.log \
        uv run python e00_basic.py
        """
    subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"Submitted {RUN_NAME} to LSF.")

if __name__=="__main__":
    import sys
    print(sys.argv)
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu':
        rungpu()
    else:
        run()
