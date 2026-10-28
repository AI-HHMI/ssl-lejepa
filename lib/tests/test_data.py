"""Tests for lib.data: every dataset has the same parts, no held-out volume is trained on, and training volumes are sampled by size ^ power."""

import math

import pytest

from lib.data import DATASETS, Dataset, sampling


def held_out(d: Dataset):
    return [*d.val, *([d.fit] if d.fit else []), *d.test, *([d.tune] if d.tune else [])]


def overlap(a, b) -> bool:
    return a.crop == b.crop and all(max(al, bl) < min(ah, bh) for (al, ah), (bl, bh) in zip(a.box, b.box))


@pytest.mark.parametrize("name", list(DATASETS))
def test_a_dataset_has_train_and_val_volumes_and_labels_only_where_it_reads_out(name):
    d = DATASETS[name]
    assert d.train and d.val and all(not v.label for v in d.train)  # pretraining reads images only
    labelled = [d.fit, *d.test, d.tune] if d.fit else []
    assert (d.fit is None) == (not d.test) and all(v is None or v.label for v in labelled)  # a readout is fit and scored, or neither
    assert d.tune is None or d.tune.name and all(v.name for v in d.test), "mia-evals scores only volumes it names"


@pytest.mark.parametrize("name", list(DATASETS))
def test_no_held_out_volume_is_trained_on(name):
    d = DATASETS[name]
    assert not [(v.crop, t.crop) for v in held_out(d) for t in d.train if overlap(v, t)]


def test_datasets_share_no_train_crop():
    crops = [v.crop for d in DATASETS.values() for v in d.train]
    assert len(crops) == len(set(crops))


@pytest.mark.parametrize("power", [0.0, 0.4, 1.0])
def test_a_datasets_share_and_a_volumes_within_it_go_as_voxels_to_the_power(power):
    names = tuple(DATASETS)
    weights = sampling(names, power)
    assert sum(w for _, w in weights) == pytest.approx(1)
    share = {n: sum(w for v, w in weights if v in DATASETS[n].train) for n in names}
    size = {n: sum(v.voxels for v in DATASETS[n].train) for n in names}
    for n in names:
        assert share[n] / share["hemibrain"] == pytest.approx((size[n] / size["hemibrain"]) ** power)
    cellmap = {v.crop: w for v, w in weights if v in DATASETS["cellmap"].train}
    big, small = (max(DATASETS["cellmap"].train, key=lambda v: v.voxels), min(DATASETS["cellmap"].train, key=lambda v: v.voxels))
    assert cellmap[big.crop] / cellmap[small.crop] == pytest.approx((big.voxels / small.voxels) ** power)


def test_power_zero_samples_datasets_equally_and_each_ones_volumes_equally():
    weights = sampling(("fish2", "microns"), 0.0)
    assert [w for v, w in weights if v in DATASETS["fish2"].train] == [0.125] * 4 and [w for v, w in weights if v in DATASETS["microns"].train] == [0.5]


def test_sampling_wants_a_dataset_named():
    with pytest.raises(AssertionError, match="name datasets"):
        sampling((), 0.4)
    with pytest.raises(AssertionError, match="no dataset 'hemibrain_eb'"):
        sampling(("hemibrain_eb",), 0.4)
