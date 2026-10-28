"""Evals of a loaded model's weights, each given its model (load_checkpoint's), the Train it was trained by, the dataset (lib.data.DATASETS
name) to measure it on, and the savedir to write its artifacts to: PCA maps
of an encoder's tokens (pca_maps), a linear affinity probe (probe), and a UNETR fitted on labels (finetune). The tasks and their
run records are experiment.py's."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np

from lib.data import Volume, dataset
from lib.inference import boundary_png, mean_scores, pca_rgb, score_block, tile_features, tile_logits, write_affinities
from lib.readers import PROBE_SIDE, crop, inner, read_block
from lib.training import Train
from lib.util import *

FINETUNE_STEPS = 10_000  # finetune: fit steps (lib.decoders.unetr.fit_unetr, batch 4), scored at 100, 200, 400, ... and the last


def pca_maps(encoder, par: Train, savedir: Path, step: int, data: str):
    """PCA maps of patch-token embeddings on a crop of the middle of dataset data's first val volume -> savedir/pca2.png, pca.json.

    pca2.png rows: the middle z-slice of each of the crop's 3 tile layers (a token's tile is the 96^3 view the encoder
    saw at once, so rows from one layer would share one tile code). Columns: EM | PCA of tokens | PCA after subtracting
    each tile's mean token | PCA fitted within each tile (colors are not comparable across tiles) | pre-LayerNorm
    token L2 norm.
    """
    import torch
    from PIL import Image
    device = next(encoder.parameters()).device
    encoder.eval()

    # 3 tiles deep in z so each row of pca2.png is a different tile.
    shape = (3 * par.data.view[0], 4 * par.data.view[1], 4 * par.data.view[2])
    img = crop(dataset(data).val[0], shape)

    # Tokens from non-overlapping tiles at the training global-view size, stitched into one grid.
    tile, patch = par.data.view, encoder.patch_embed.patch_size
    assert all(s % t == 0 for s, t in zip(shape, tile)), f"crop {shape} must tile by {tile}"
    grid = [s // p for s, p in zip(shape, patch)]
    feats = torch.empty(*grid, encoder.config.embed_dim)  # Z Y X D, tokens (after the final LayerNorm)
    norms = torch.empty(*grid)  # Z Y X, token norms before the final LayerNorm
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for z, y, x in product(*(range(0, s, t) for s, t in zip(shape, tile))):
            r = encoder.forward_residual(img[None, :, z:z + tile[0], y:y + tile[1], x:x + tile[2]].to(device))
            t = encoder.norm(r)
            tz, ty, tx = (o // p for o, p in zip((z, y, x), patch))
            gz, gy, gx = (t_ // p for t_, p in zip(tile, patch))
            feats[tz:tz + gz, ty:ty + gy, tx:tx + gx] = t[0].float().reshape(gz, gy, gx, -1).cpu()
            norms[tz:tz + gz, ty:ty + gy, tx:tx + gx] = r[0].float().norm(dim=-1).reshape(gz, gy, gx).cpu()

    # Tile-centered tokens: subtract each tile's mean token, removing the per-view code and keeping within-view structure.
    g = [t_ // p for t_, p in zip(tile, patch)]  # tokens per tile side
    nt = [n // gi for n, gi in zip(grid, g)]  # tiles per axis
    tiled = feats.reshape(nt[0], g[0], nt[1], g[1], nt[2], g[2], encoder.config.embed_dim)
    centered = (tiled - tiled.mean(dim=(1, 3, 5), keepdim=True)).reshape(*grid, encoder.config.embed_dim)
    rgb, erank, var = pca_rgb(feats.reshape(-1, encoder.config.embed_dim), grid)
    rgb_c, erank_c, var_c = pca_rgb(centered.reshape(-1, encoder.config.embed_dim), grid)
    rgb_t = np.empty((*grid, 3), np.uint8)  # PCA fitted separately within each tile
    for i, j, k in product(*map(range, nt)):
        rgb_t[i * g[0]:(i + 1) * g[0], j * g[1]:(j + 1) * g[1], k * g[2]:(k + 1) * g[2]] = pca_rgb(tiled[i, :, j, :, k, :].reshape(-1, encoder.config.embed_dim), g)[0]
    between_tile = 1 - float(centered.var(dim=(0, 1, 2)).sum() / feats.var(dim=(0, 1, 2)).sum())

    # Pre-LayerNorm token norms: "register"-like tokens that store global information show up as sparse
    # high-norm outliers. (After the final LayerNorm every token's norm is ~sqrt(width), so those can't show this.)
    med = float(norms.median())
    lo, hi = torch.quantile(norms.flatten(), torch.tensor([0.01, 0.999]))
    norm_u8 = ((norms - lo) / (hi - lo)).clamp(0, 1).mul(255).byte().numpy()

    # Rows: the middle z-slice of each tile layer; columns: EM | PCA | tile-centered PCA | per-tile PCA | token norm, upsampled to voxels.
    up = lambda a: a.repeat(patch[1], axis=0).repeat(patch[2], axis=1)
    vgap = np.full((shape[1], 8, 3), 255, np.uint8)
    rows = []
    for gz in [g[0] // 2 + g[0] * i for i in range(nt[0])]:
        zv = gz * patch[0] + patch[0] // 2
        em = np.repeat((img[0, zv].numpy() * 255).astype(np.uint8)[..., None], 3, axis=2)
        nm = np.repeat(norm_u8[gz][..., None], 3, axis=2)
        rows.append(np.concatenate([em, vgap, up(rgb[gz]), vgap, up(rgb_c[gz]), vgap, up(rgb_t[gz]), vgap, up(nm)], axis=1))
    gap = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
    Image.fromarray(np.concatenate([r for row in rows for r in (row, gap)][:-1], axis=0)).save(savedir / "pca2.png")
    stats = {"tbl": "pca", "step": step, "effective_rank": erank, "explained_variance": var,
             "centered_effective_rank": erank_c, "centered_explained_variance": var_c,
             "between_tile_variance": between_tile,  # fraction of token variance explained by tile means
             "norm_median": med, "norm_p99": float(torch.quantile(norms.flatten(), 0.99)), "norm_max": float(norms.max()),
             "norm_outliers": int((norms > 2 * med).sum()), "n_tokens": norms.numel()}
    (savedir / "pca.json").write_text(json.dumps(stats) + "\n")
    print(f"{savedir}: step {step}, effective rank {erank:.1f}, top-3 variance {sum(var[:3]):.2f}, "
          f"between-tile variance {between_tile:.2f}, norm outliers (>2x median) {stats['norm_outliers']}/{norms.numel()}; wrote pca2.png")

def run_name(savedir: str) -> str:
    """mia-evals' row name for the run in savedir: its parts, letters, digits, _ and - only."""
    return "-".join(Path(savedir).parts[1:])

MIA_EVALS_SPLIT = {"tune": "fit", "test": "test"}  # mia-evals' names for the blocks it tunes on and scores on ([data.fit], [data.test])

def write_artifact(savedir: str, step: int, role: str, vol: Volume, pred):
    """vol's predictions pred (C Z Y X probabilities) as the mia-evals affinity artifact `score` reads, savedir/probe/<split>/<its
    name>.zarr, vol being the dataset's tune or test block (role) and split mia-evals' name for the role."""
    assert vol.name, f"{vol.crop} has no name in a mia-evals data config: mia-evals can't score it"
    (x0, _), (y0, _), (z0, _) = vol.box
    write_affinities(Path(savedir) / "probe" / MIA_EVALS_SPLIT[role] / f"{vol.name}.zarr", pred.permute(0, 3, 2, 1).cpu().numpy(),
                     origin=[x0, y0, z0], convention="sigmoid(logit), linear decoder",
                     run=run_name(savedir), step=step, axes="xyz", source_path=vol.path, source_label_key=f"labels/{vol.label}",
                     native_box=vol.box, scale=[1, 1, 1], annotated_box=vol.annotated, covers_full_box=False)


def probe(model, train: Train, data: str, savedir: str, step: int, standardize: bool = False):
    """Linear affinity probe (lib.probe) of model, frozen: one Linear per token to mia-evals' 6 affinity channels at every voxel
    of its patch, fitted on the fit block of dataset data (lib.data.DATASETS) against its instance labels, scored on
    its test blocks (averaged) as mia-evals' neuron-instance task. Data: the dataset's fit block fits the Linear; its test blocks
    score it; its tune block's predictions are also written (`score` picks the post-processing on them). Every block read at 8 nm,
    as training reads (read_block).
      savedir/probe.json          test-block BCE and boundary AP per channel (cheap, immediate)
      savedir/probe_fit.json      the fit curve every 100 steps: training loss, and BCE and boundary AP on a
                                  fixed held-out sample (monitoring only, nothing selects on it)
      savedir/probe/{fit,test}/<name>.zarr   mia-evals affinity artifacts, (6, X, Y, Z) float16, for
                                  `mia-evals score` with truth_kind = "instances" (reads the GT from the store)
      savedir/probe.png           test block, middle z: EM | true boundaries | predicted boundaries (probe_<k>.png
                                  for any further test blocks)
    Reads get whole tiles of context (the training view, tile) around each block (at most PROBE_SIDE a side), as in
    training; where a block's context leaves its volume, the image is edge-replicated.
    """
    import torch
    from lib.probe import AFFINITY_OFFSETS_XYZ, affinities, fit_probe, to_tokens, to_voxels
    tile = train.data.view
    encoder, device = model.encoder, next(model.parameters()).device
    p = encoder.patch_embed.patch_size
    ds = dataset(data)
    assert ds.tune, f"{savedir}: no mia-evals tune block for {data} (a dataset with a tune block), to score"
    assert len(set(p)) == 1, f"patch {p} must be cubic"
    offsets = [o[::-1] for o in AFFINITY_OFFSETS_XYZ]  # x y z offsets; the model sees z y x

    def tokens(img, m, size):  # the box's token features, N D
        f = tile_features(encoder, img, tile)
        t0, n = [mi // q for mi, q in zip(m, p)], [s // q for s, q in zip(size, p)]
        return f[t0[0]:t0[0] + n[0], t0[1]:t0[1] + n[1], t0[2]:t0[2] + n[2]].reshape(-1, f.shape[-1])

    tests = [read_block(v, tile, p, device) for v in ds.test]  # (img, labels, m, size) each
    img, labels, m, size = tests[0]
    aff, valid = affinities(labels, offsets, ds.test[0].sparse)
    feats = tokens(img, m, size)
    g = torch.Generator(device=device).manual_seed(0)
    i = torch.randperm(len(feats), device=device, generator=g)[:16384]  # held-out tokens, fit curve only
    held = (feats[i], to_tokens(aff, p[0])[i], to_tokens(valid, p[0])[i])
    img, labels, m, size = read_block(ds.fit, tile, p, device)
    aff, valid = affinities(labels, offsets, ds.fit.sparse)
    feats = tokens(img, m, size)
    del img, labels
    head, curve = fit_probe(feats, to_tokens(aff, p[0]), to_tokens(valid, p[0]), held, standardize)
    del feats, aff, valid, held

    def predict(img, m, size):  # C Z Y X sigmoid probabilities over the box
        with torch.no_grad():
            prob = torch.cat([torch.sigmoid(head(f.float())).half() for f in tokens(img, m, size).split(65536)])
        return to_voxels(prob, tuple(s // p[0] for s in size), p[0])

    (Path(savedir) / "probe_fit.json").write_text("".join(json.dumps({"tbl": "probe_fit", **c}) + "\n" for c in curve))
    per_block = []
    for k, (vol, (img, labels, m, size)) in enumerate(zip(ds.test, tests)):
        pred = predict(img, m, size)
        per_block.append(score_block(pred, labels, vol.sparse))
        boundary_png(Path(savedir) / ("probe.png" if k == 0 else f"probe_{k}.png"), inner(img, m, size), labels, pred)
        write_artifact(savedir, step, "test", vol, pred)
    img, _, m, size = read_block(ds.tune, tile, p, device)
    write_artifact(savedir, step, "tune", ds.tune, predict(img, m, size))
    stats = {"tbl": "probe", "step": step, "run": run_name(savedir), "data": data, "test_blocks": [v.crop for v in ds.test],
             **mean_scores(per_block)}
    (Path(savedir) / "probe.json").write_text(json.dumps(stats) + "\n")
    print(f"{savedir}: probe at step {step}: boundary AP (short-range mean) {stats['boundary_ap_short']:.3f}; wrote probe/", flush=True)

def predict_block(unetr, tile, img, m, size):
    """C Z Y X sigmoid probabilities of unetr over the box (m, size) of img, tiled with context (tile_logits)."""
    import torch
    return torch.sigmoid(inner(tile_logits(unetr, img, tile), m, size)).half()

def finetune(model, train: Train, data: str, savedir: str, step: int):
    """A UNETR (lib.decoders.unetr: skip connections into the encoder's own blocks) on model's frozen encoder, fitted for
    FINETUNE_STEPS to mia-evals' 6 affinity channels of the fit block of dataset data (a PROBE_SIDE cube, as probe
    reads it), on crops of the encoder's view size, and scored as it fits on the dataset's test blocks, tiled at that size; the
    tune block's predictions are written too, for `score`. Blocks read as probe reads them.
      savedir/finetune_curve.json   a first row (tbl "finetune_data"): the weights' step, the fit cube's size and labelled voxels
                              (the GT the fit had), the test blocks. Then the fit curve (tbl "finetune", fit_unetr's): the batch
                              loss every 100 steps, and at 100, 200, 400, ... steps and the last, the test blocks' scores
                              averaged, as probe.json has them.
      savedir/finetune.png    test block after the fit, middle z: EM | true boundaries | predicted boundaries
                              (finetune_<block>.png for any further test blocks), written last: the task finished
      savedir/probe/{fit,test}/   the fitted decoder's mia-evals affinity artifacts, as probe writes them
    """
    import torch
    from lib.decoders import Unetr, UnetrConfig, fit_unetr
    from lib.probe import AFFINITY_OFFSETS_XYZ, affinities
    torch.set_float32_matmul_precision(train.speed.f32mode)  # as the encoder trained: TF32 matmuls ("high"), not full fp32
    tile, p, device = train.data.view, model.encoder.patch_embed.patch_size, next(model.parameters()).device
    ds = dataset(data)
    assert ds.tune, f"{savedir}: no mia-evals tune block for {data} (a dataset with a tune block), to score"
    tests = [read_block(v, tile, p, device, context=True) for v in ds.test]  # (img, labels, m, size) each

    def test_scores(unetr) -> dict:
        return mean_scores([score_block(predict_block(unetr, tile, t_img, t_m, t_size), t_labels, v.sparse) for v, (t_img, t_labels, t_m, t_size) in zip(ds.test, tests)])

    unetr = Unetr(model.encoder, UnetrConfig(out_dim=len(AFFINITY_OFFSETS_XYZ))).to(device)
    model.encoder.requires_grad_(False)
    img, labels, m, size = read_block(ds.fit, tile, p, device, PROBE_SIDE)
    assert all(s >= q for s, q in zip(size, tile)), f"{savedir}: the fit cube {size} is smaller than a crop {tile}"
    aff, valid = affinities(labels, [o[::-1] for o in AFFINITY_OFFSETS_XYZ], ds.fit.sparse)  # x y z offsets; the model sees z y x
    data = {"tbl": "finetune_data", "fit_size": size, "fit_labelled": int((labels != 0).sum()), "test_blocks": [v.crop for v in ds.test]}
    curve = fit_unetr(unetr, inner(img, m, size)[None], aff, valid, tile, lambda: test_scores(unetr), steps=FINETUNE_STEPS)
    rows = [{**data, "step": step}] + [{"tbl": "finetune", **c} for c in curve]
    (Path(savedir) / "finetune_curve.json").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for k, (vol, (img, labels, m, size)) in enumerate(zip(ds.test, tests)):
        pred = predict_block(unetr, tile, img, m, size)
        write_artifact(savedir, step, "test", vol, pred)
        boundary_png(Path(savedir) / ("finetune.png" if k == 0 else f"finetune_{k}.png"), inner(img, m, size), labels, pred)
    img, _, m, size = read_block(ds.tune, tile, p, device, context=True)
    write_artifact(savedir, step, "tune", ds.tune, predict_block(unetr, tile, img, m, size))
    print(f"{savedir}: finetune of step {step} on {data['fit_size']} voxels of {data}: boundary AP (short-range mean) "
          f"{curve[-1]['boundary_ap_short']:.3f} after {FINETUNE_STEPS} steps", flush=True)
