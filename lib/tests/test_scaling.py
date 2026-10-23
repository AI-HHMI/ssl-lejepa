"""Tests for scaling: the scaling-law grid's parameter counts, step times and compute budgets."""

from types import SimpleNamespace

import pytest

import scaling


def test_lejepa_n_params_builds_the_model_once_per_size(tmp_path, monkeypatch):
    built = []

    def fake_lejepa(config):
        e = config.encoder
        built.append((e.depth, e.embed_dim))
        return SimpleNamespace(_n_encoder_params=e.depth * e.embed_dim, _n_predictor_params=5, _n_projector_params=7)

    monkeypatch.setattr(scaling, "N_PARAMS_CACHE", tmp_path / ".cache/n_params.json")
    monkeypatch.setattr(scaling, "Lejepa", fake_lejepa)
    assert scaling.lejepa_n_params(4, 256) == (1024, 5, 7)
    assert scaling.lejepa_n_params(4, 256) == (1024, 5, 7)  # read back from the cache file
    assert scaling.lejepa_n_params(6, 384) == (2304, 5, 7)
    assert built == [(4, 256), (6, 384)]


def test_scaling_budgets_are_m_hours_at_the_anchor_speed(monkeypatch):
    monkeypatch.setattr(scaling, "lejepa_n_params", lambda n_layers, width: (n_layers * width ** 2, n_layers * width ** 2 // 3, 1000))
    assert scaling.step_seconds("m", 84) == pytest.approx(scaling.ANCHOR_STEP_S_M_B84)  # the anchor itself
    assert scaling.step_seconds("l", 84) > scaling.step_seconds("m", 84) > scaling.step_seconds("m", 42)
    budgets = scaling.scaling_budgets()
    m_steps_per_hour = 3600 / scaling.step_seconds("m", scaling.SCALING_BATCH)
    for b, hours in scaling.BUDGET_HOURS_AT_M.items():  # m trains BUDGET_HOURS_AT_M hours at each budget
        assert budgets[b] / (scaling.sample_flops("m") * scaling.SCALING_BATCH) == pytest.approx(hours * m_steps_per_hour)
