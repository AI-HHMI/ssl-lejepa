"""Tests for lib.specs: a spec's model and the weights it loads."""

from dataclasses import asdict, replace

import pytest
import torch

from lib import specs
from lib.encoders import EncoderConfig
from lib.models import Lejepa, LejepaConfig, MaeConfig, ProjectorConfig


def test_mae_decoder_sets_the_mae_models_decoder():
    c = specs.model_config(specs.Train(model=specs.Mae(decoder=(512, 8))))
    assert isinstance(c, MaeConfig) and (c.decoder_dim, c.decoder_depth, c.decoder_heads) == (512, 8, 16)


def test_load_checkpoint_reads_a_milestone_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    par = specs.Train(patch_size=(8, 8, 8), global_size=(8, 8, 8))
    monkeypatch.setattr(specs, "model_config", lambda par: LejepaConfig(
        encoder=EncoderConfig(depth=1, embed_dim=16, num_heads=2, patch_size=(4, 4, 4)), projector=ProjectorConfig(dim=8, hidden=16)))
    model = Lejepa(specs.model_config(par))
    path = tmp_path / "step_0000130.pt"
    torch.save({"step": 130, "model": model.state_dict()}, path)
    loaded, step = specs.load_checkpoint(specs.Eval(savedir=str(tmp_path), train=par, init_from=str(path)))
    assert step == 130 and torch.equal(loaded.encoder.blocks[0].attn.qkv.weight.cpu(), model.encoder.blocks[0].attn.qkv.weight)


def test_each_objective_takes_its_own_knobs_from_the_spec():
    masked = specs.model_config(specs.Train(model=specs.Masked(cube=40, shift=16, lamb=0.02), global_size=(96, 96, 96)))
    assert masked.objective == "mask" and masked.mask.cube == 5 and masked.lamb == 0.02  # tokens: the cube's voxels over the patch of 8
    assert masked.view.global_size == (112, 112, 112) and masked.view.n_local == 0  # the context's region, and the shift
    views = specs.model_config(specs.Train(model=specs.Views(local=64, batch_views=True)))
    assert views.objective == "views" and views.batch_views and views.view.local_size == (64, 64, 64) and views.view.n_local == 4
    with pytest.raises(AssertionError, match="multiple of the patch"):
        specs.model_config(specs.Train(model=specs.Masked(cube=44)))


def test_the_models_round_trip_through_a_spec_json():
    from lib.util import parse
    for model in (specs.Mae(decoder=(512, 8)), specs.Masked(cube=56, sigreg_over="both"), specs.Views(local=96)):
        par = specs.Train(model=model)
        assert parse(specs.Train, asdict(par)) == par and type(parse(specs.Train, asdict(par)).model) is type(model)
