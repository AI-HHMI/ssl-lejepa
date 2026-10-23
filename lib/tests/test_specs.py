"""Tests for lib.specs: a spec's model, its milestones and the weights it loads."""

from dataclasses import replace

import torch

from lib import specs
from lib.encoders import EncoderConfig
from lib.models import Lejepa, LejepaConfig, MaeConfig, ProjectorConfig


def test_mae_decoder_sets_the_mae_models_decoder():
    c = specs.model_config(specs.Train(objective="mae", mae_decoder=(512, 8)))
    assert isinstance(c, MaeConfig) and (c.decoder_dim, c.decoder_depth, c.decoder_heads) == (512, 8, 16)


def test_load_checkpoint_reads_a_milestone_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    par = specs.Train(patch_size=(8, 8, 8), views="displace", global_size=(8, 8, 8))
    monkeypatch.setattr(specs, "lejepa_config", lambda par: LejepaConfig(
        encoder=EncoderConfig(depth=1, embed_dim=16, num_heads=2, patch_size=(4, 4, 4)), projector=ProjectorConfig(dim=8, hidden=16)))
    model = Lejepa(specs.lejepa_config(par))
    path = tmp_path / "step_0000130.pt"
    torch.save({"step": 130, "model": model.state_dict()}, path)
    loaded, step = specs.load_checkpoint(specs.Eval(savedir=str(tmp_path), train=par, init_from=str(path)))
    assert step == 130 and torch.equal(loaded.encoder.blocks[0].attn.qkv.weight.cpu(), model.encoder.blocks[0].attn.qkv.weight)


def test_lejepa_config_takes_the_mask_block_and_lamb_from_params():
    config = specs.lejepa_config(specs.Train(mask_block=0.45, lamb=0.02))
    assert config.mask.block == 0.45 and config.lamb == 0.02


def test_milestones_are_every_n_steps_or_every_tenth():
    assert specs.milestone_steps(specs.Train(steps_per_epoch=1300)) == list(range(130, 1301, 130))
    assert specs.milestone_steps(specs.Train(steps_per_epoch=100, milestone_every=30)) == [30, 60, 90]  # the cap lists them
    doubling = specs.Train(steps_per_epoch=100, milestone_every=10, milestone_doubling=True)
    assert specs.milestone_steps(doubling) == [10, 20, 40, 80]
    assert specs.milestone_steps(replace(doubling, steps_per_epoch=80)) == [10, 20, 40, 80]  # the cap itself, if on the doubling
