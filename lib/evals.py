"""Evals of a set of weights, each given its model (load_checkpoint's) and writing its artifacts to its spec's savedir: PCA maps of
an encoder's tokens (pca_maps, pca_sizes), a linear affinity probe (probe), and a UNETR fitted on labels (finetune, tiles).
Entrypoints, their run records and milestone release are experiment.py's."""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import numpy as np

from lib.data import ProbeVolume, probe_set
from lib.inference import boundary_png, mean_scores, pca_rgb, score_block, tile_features, tile_logits, write_affinities
from lib.readers import PROBE_SIDE, held_out_crop, inner, read_block
from lib.specs import Eval, Finetune, Train
from lib.util import *

TEST_TILES = (16, 32, 64, 128, 256, 512)  # pca_sizes: the tile sides an encoder is run at, test time (2^3 to 64^3 tokens)


def pca_maps(encoder, par: Train, savedir: Path, step: int, data: str):
    """PCA maps of patch-token embeddings on a crop of dataset data's held-out region (lib.data.held_out) -> savedir/pca2.png, pca.json.

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
    shape = (3 * par.global_size[0], 4 * par.global_size[1], 4 * par.global_size[2])
    img = held_out_crop(data, shape)

    # Tokens from non-overlapping tiles at the training global-view size, stitched into one grid.
    tile, patch = par.global_size, encoder.patch_embed.patch_size
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

def pca_sizes(e: Eval, model, step: int):
    """The input size at test time: e's model's encoder (weights from step) run on one crop of its probe set's dataset's held-out region in non-overlapping cubic tiles
    of each TEST_TILES side, the crop 2 x 2 tiles of the largest in y x and one deep (so the smallest are many), ->
    savedir/pca_tile_<side>.png and a pca_sizes.json row per side. Each png is one z-slice, the same voxels at every side:
    EM | PCA of its tokens | PCA after subtracting each tile's mean token, fitted on that slice. The row's effective rank
    and explained variance are of 65536 tokens drawn from the whole crop, its between-tile variance (the share of token
    variance that is each tile's mean) of all of them."""
    import torch
    from PIL import Image
    encoder, device = model.encoder, next(model.parameters()).device
    p = encoder.patch_embed.patch_size
    big = max(TEST_TILES)
    shape = (big, 2 * big, 2 * big)
    img = held_out_crop(e.probe_on, shape)[0].to(device)  # Z Y X
    zt = shape[0] // 2 // p[0]  # the slice shown: the middle of a token row near the crop's middle
    em = np.repeat((img[zt * p[0] + p[0] // 2].cpu().numpy() * 255).astype(np.uint8)[..., None], 3, axis=2)
    up = lambda a: a[0].repeat(p[1], axis=0).repeat(p[2], axis=1)  # 1 Gy Gx 3 tokens -> Y X 3 voxels
    vgap = np.full((shape[1], 8, 3), 255, np.uint8)
    (Path(e.savedir) / "pca_sizes.json").write_text("")  # a redo (lsf pca_sizes) starts over
    for side in TEST_TILES:
        t0 = time.time()
        feats = tile_features(encoder, img, (side, side, side)).float()  # Gz Gy Gx D
        grid, d = feats.shape[:3], feats.shape[-1]
        g = side // p[0]  # tokens per tile side
        tiled = feats.reshape(grid[0] // g, g, grid[1] // g, g, grid[2] // g, g, d)
        centered = (tiled - tiled.mean(dim=(1, 3, 5), keepdim=True)).reshape(*grid, d)
        between = 1 - float(centered.var(dim=(0, 1, 2)).sum() / feats.var(dim=(0, 1, 2)).sum())
        sample = feats.reshape(-1, d)[torch.randperm(feats[..., 0].numel(), device=device, generator=torch.Generator(device).manual_seed(0))[:65536]]
        _, erank, var = pca_rgb(sample, (len(sample),))
        rgb, rgb_c = pca_rgb(feats[zt].reshape(-1, d), (1, *grid[1:]))[0], pca_rgb(centered[zt].reshape(-1, d), (1, *grid[1:]))[0]
        Image.fromarray(np.concatenate([em, vgap, up(rgb), vgap, up(rgb_c)], axis=1)).save(Path(e.savedir) / f"pca_tile_{side:03d}.png")
        row = {"tbl": "pca_sizes", "step": step, "tile": side, "tiles": (shape[0] // side) * (shape[1] // side) * (shape[2] // side),
               "effective_rank": erank, "explained_variance": var, "between_tile_variance": between, "seconds": time.time() - t0}
        with open(Path(e.savedir) / "pca_sizes.json", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"tile {side}: {row['tiles']} tiles, effective rank {erank:.1f}, between-tile variance {between:.3f}, {row['seconds']:.0f} s", flush=True)
        del feats, tiled, centered
        torch.cuda.empty_cache()

def probe(par: Eval, model, step: int):
    """Linear affinity probe (lib.probe) of the spec job's weights (or those of the eval.json it names), frozen: one Linear per
    token to mia-evals' 6 affinity channels at every voxel of its patch, fitted on the train block of its probe set
    (par.probe_on, lib.data.PROBE_SETS) against its instance labels, scored on its test blocks (averaged) as mia-evals'
    neuron-instance task. Every block read at 8 nm, as training reads (read_block).
      savedir/probe.json          test-block BCE and boundary AP per channel (cheap, immediate)
      savedir/probe_fit.json      the fit curve every 100 steps: training loss, and BCE and boundary AP on a
                                  fixed held-out sample (monitoring only, nothing selects on it)
    Weights: par.init_from (load_checkpoint): this run's checkpoint, an older run's, or random (the baseline).
      savedir/probe/{fit,test}/<probe_on>_{split}.zarr   mia-evals affinity artifacts, (6, X, Y, Z) float16, for
                                  `mia-evals score` with truth_kind = "instances" (reads the GT from the store)
      savedir/probe.png           test block, middle z: EM | true boundaries | predicted boundaries (probe_<block>.png
                                  for any further test blocks)
    Reads get whole tiles of context (the training view, tile) around each block (at most PROBE_SIDE a side), as in
    training; where a block's context leaves its volume, the image is edge-replicated.
    """
    import torch
    from lib.probe import AFFINITY_OFFSETS_XYZ, affinities, fit_probe, to_tokens, to_voxels
    tile = par.train.global_size
    encoder, device = model.encoder, next(model.parameters()).device
    p = encoder.patch_embed.patch_size
    sets = probe_set(par.probe_on)
    assert not par.score or "fit" in sets, f"{par.savedir}: no mia-evals fit block for {par.probe_on} (a probe set with a \"fit\" split), to score"
    assert len(sets["train"]) == 1, f"{par.probe_on}: the probe fits on one block"
    assert len(set(p)) == 1, f"patch {p} must be cubic"
    offsets = [o[::-1] for o in AFFINITY_OFFSETS_XYZ]  # x y z offsets; the model sees z y x

    def tokens(img, m, size):  # the box's token features, N D
        f = tile_features(encoder, img, tile)
        t0, n = [mi // q for mi, q in zip(m, p)], [s // q for s, q in zip(size, p)]
        return f[t0[0]:t0[0] + n[0], t0[1]:t0[1] + n[1], t0[2]:t0[2] + n[2]].reshape(-1, f.shape[-1])

    tests = [read_block(v, tile, p, device) for v in sets["test"]]  # (img, labels, m, size) each
    img, labels, m, size = tests[0]
    aff, valid = affinities(labels, offsets, sets["test"][0].sparse)
    feats = tokens(img, m, size)
    g = torch.Generator(device=device).manual_seed(0)
    i = torch.randperm(len(feats), device=device, generator=g)[:16384]  # held-out tokens, fit curve only
    held = (feats[i], to_tokens(aff, p[0])[i], to_tokens(valid, p[0])[i])
    img, labels, m, size = read_block(sets["train"][0], tile, p, device)
    aff, valid = affinities(labels, offsets, sets["train"][0].sparse)
    feats = tokens(img, m, size)
    del img, labels
    head, curve = fit_probe(feats, to_tokens(aff, p[0]), to_tokens(valid, p[0]), held)
    del feats, aff, valid, held

    def predict(img, m, size):  # C Z Y X sigmoid probabilities over the box
        with torch.no_grad():
            prob = torch.cat([torch.sigmoid(head(f.float())).half() for f in tokens(img, m, size).split(65536)])
        return to_voxels(prob, tuple(s // p[0] for s in size), p[0])

    run_name = "-".join(Path(par.savedir).parts[1:])  # mia-evals row name: letters, digits, _ and - only

    def artifact(split: str, vol: ProbeVolume, pred):  # for score
        (x0, _), (y0, _), (z0, _) = vol.box
        write_affinities(Path(par.savedir) / "probe" / split / f"{par.probe_on}_{split}.zarr", pred.permute(0, 3, 2, 1).cpu().numpy(),
                         origin=[x0, y0, z0], convention="sigmoid(logit), linear decoder",
                         run=run_name, step=step, axes="xyz", source_path=vol.path, source_label_key=f"labels/{vol.label_key}",
                         native_box=vol.box, scale=[1, 1, 1], annotated_box=vol.annotated, covers_full_box=False)

    (Path(par.savedir) / "probe_fit.json").write_text("".join(json.dumps({"tbl": "probe_fit", **c}) + "\n" for c in curve))
    per_block = []
    for k, (vol, (img, labels, m, size)) in enumerate(zip(sets["test"], tests)):
        pred = predict(img, m, size)
        per_block.append(score_block(pred, labels, vol.sparse))
        boundary_png(Path(par.savedir) / ("probe.png" if k == 0 else f"probe_{vol.name}.png"), inner(img, m, size), labels, pred)
        if par.score:
            artifact("test", vol, pred)
    if par.score:  # also the set's mia-evals fit block, whose predictions only feed the artifacts
        vol = sets["fit"][0]
        img, _, m, size = read_block(vol, tile, p, device)
        artifact("fit", vol, predict(img, m, size))
    stats = {"tbl": "probe", "step": step, "run": run_name, "probe_on": par.probe_on, "test_blocks": [v.name for v in sets["test"]],
             **mean_scores(per_block)}
    (Path(par.savedir) / "probe.json").write_text(json.dumps(stats) + "\n")
    print(f"{par.savedir}: probe at step {step}: boundary AP (short-range mean) {stats['boundary_ap_short']:.3f}{'; wrote probe/' * par.score}", flush=True)

def fit_decoder(par: Finetune, model, score=None):
    """finetune's fit: a UNETR on model's encoder (trained too unless par.freeze_encoder), fitted to the affinities of par's probe set's
    train block or of its par.fit_side cube, on crops of the encoder's view size: (the UNETR, fit_unetr's curve, the fit's
    finetune_data row without the weights' step). score(unetr) -> dict is fit_unetr's scoring, at its doubling steps."""
    import torch
    from lib.decoders import Unetr, UnetrConfig, fit_unetr
    from lib.probe import AFFINITY_OFFSETS_XYZ, affinities
    torch.set_float32_matmul_precision(par.train.f32mode)  # as the encoder trained: TF32 matmuls ("high"), not full fp32
    tile = par.train.global_size
    encoder, device = model.encoder, next(model.parameters()).device
    p = encoder.patch_embed.patch_size
    sets = probe_set(par.probe_on)
    side = par.fit_side or PROBE_SIDE
    assert len(sets["train"]) == 1, f"{par.probe_on}: the fit is on one block"
    assert side % p[0] == 0, f"{par.savedir}: fit_side {side} must be whole patches of {p[0]}"
    unetr = Unetr(encoder, UnetrConfig(out_dim=len(AFFINITY_OFFSETS_XYZ))).to(device)
    encoder.requires_grad_(not par.freeze_encoder)
    img, labels, m, size = read_block(sets["train"][0], tile, p, device, side)
    assert all(s >= t for s, t in zip(size, tile)), f"{par.savedir}: the fit cube {size} is smaller than a crop {tile}"
    aff, valid = affinities(labels, [o[::-1] for o in AFFINITY_OFFSETS_XYZ], sets["train"][0].sparse)  # x y z offsets; the model sees z y x
    data = {"tbl": "finetune_data", "fit_size": size, "fit_labelled": int((labels != 0).sum()), "test_blocks": [v.name for v in sets["test"]]}
    curve = fit_unetr(unetr, inner(img, m, size)[None], aff, valid, tile, (lambda: score(unetr)) if score else None, steps=par.steps)
    return unetr, curve, data

def predict_block(unetr, tile, img, m, size):
    """C Z Y X sigmoid probabilities of unetr over the box (m, size) of img, tiled with context (tile_logits)."""
    import torch
    return torch.sigmoid(inner(tile_logits(unetr, img, tile), m, size)).half()

def finetune(par: Finetune, model, step: int):
    """A UNETR (lib.decoders.unetr: skip connections into the encoder's own blocks) on the spec job's weights (or those of the
    finetune.json it names), fitted to mia-evals' 6 affinity channels of its probe set's (par.probe_on) train block, or of the
    par.fit_side cube about its centre: on crops of the encoder's view size, with the encoder trained too unless
    par.freeze_encoder, and scored as it fits on the set's test blocks, tiled at that size. Blocks read as probe reads them.
      savedir/finetune_curve.json   a first row (tbl "finetune_data"): the weights' step, the fit cube's size and labelled voxels
                              (the GT the fit had), the test blocks. Then the fit curve (tbl "finetune", fit_unetr's): the batch
                              loss every 100 steps, and at 100, 200, 400, ... steps and the last, the test blocks' scores
                              averaged, as probe.json has them. (Not finetune.json: that is the job's spec, dump's.)
      savedir/finetune.png    test block after the fit, middle z: EM | true boundaries | predicted boundaries
                              (finetune_<block>.png for any further test blocks), written last: the job finished
    """
    tile, p, device = par.train.global_size, model.encoder.patch_embed.patch_size, next(model.parameters()).device
    sets = probe_set(par.probe_on)
    tests = [read_block(v, tile, p, device, context=True) for v in sets["test"]]  # (img, labels, m, size) each

    def test_scores(unetr) -> dict:
        return mean_scores([score_block(predict_block(unetr, tile, t_img, t_m, t_size), t_labels, v.sparse) for v, (t_img, t_labels, t_m, t_size) in zip(sets["test"], tests)])

    unetr, curve, data = fit_decoder(par, model, test_scores)
    rows = [{**data, "step": step}] + [{"tbl": "finetune", **c} for c in curve]
    (Path(par.savedir) / "finetune_curve.json").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for k, (vol, (img, labels, m, size)) in enumerate(zip(sets["test"], tests)):
        boundary_png(Path(par.savedir) / ("finetune.png" if k == 0 else f"finetune_{vol.name}.png"), inner(img, m, size), labels, predict_block(unetr, tile, img, m, size))
    print(f"{par.savedir}: finetune of step {step} on {data['fit_size']} voxels of {par.probe_on}: boundary AP (short-range mean) "
          f"{curve[-1]['boundary_ap_short']:.3f} after {par.steps} steps", flush=True)

def tiles(par: Finetune, model, step: int):
    """finetune's fit, without its scoring, and tile_logits' blended prediction on its first test block: savedir/tiles.png,
    finetune.png's panels (middle z: EM | true boundaries | predicted boundaries). The fit is seeded, so these are finetune's
    predictions, redone with the current tiling; nothing else is written."""
    unetr, _, _ = fit_decoder(par, model)
    tile, p, device = par.train.global_size, model.encoder.patch_embed.patch_size, next(model.parameters()).device
    img, labels, m, size = read_block(probe_set(par.probe_on)["test"][0], tile, p, device, context=True)
    boundary_png(Path(par.savedir) / "tiles.png", inner(img, m, size), labels, predict_block(unetr, tile, img, m, size))
    print(f"{par.savedir}: wrote tiles.png, finetune of step {step} for {par.steps} steps", flush=True)

