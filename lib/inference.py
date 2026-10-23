"""Inference and scoring helpers that need no spec: tokens and logits of an encoder or UNETR over a volume in tiles, PCA colours,
and the mia-evals affinity artifacts, per-block scores and boundary pictures."""

from itertools import product
from pathlib import Path



def pca_rgb(f, size):
    """N D tokens of a size (... ) box -> (... 3 uint8: the top 3 principal components, each scaled 1st-99th percentile;
    effective rank, ~1 a collapse; top-10 explained variance)."""
    import torch
    f = f - f.mean(0)
    _, sv, vh = torch.linalg.svd(f, full_matrices=False)
    p = sv / sv.sum()
    erank = float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))
    pcs = (f @ vh[:3].T).reshape(*size, 3)
    lo, hi = torch.quantile(pcs.reshape(-1, 3), torch.tensor([0.01, 0.99], device=f.device), dim=0)
    rgb = ((pcs - lo) / (hi - lo)).clamp(0, 1).mul(255).byte().cpu().numpy()
    return rgb, erank, (sv ** 2 / (sv ** 2).sum())[:10].tolist()

def tile_features(encoder, img, tile):
    """Encoder tokens (after its final LayerNorm) of img (Z Y X in [0, 1], on the encoder's device, a multiple of
    tile), from non-overlapping tiles at the training global-view size, stitched: Gz Gy Gx D, bf16."""
    import torch
    p = encoder.patch_embed.patch_size
    assert all(s % t == 0 for s, t in zip(img.shape, tile)), f"{tuple(img.shape)} must tile by {tile}"
    g = [t // q for t, q in zip(tile, p)]  # tokens per tile side
    feats = torch.empty(*(s // q for s, q in zip(img.shape, p)), encoder.config.embed_dim, dtype=torch.bfloat16, device=img.device)
    with torch.no_grad(), torch.autocast(img.device.type, dtype=torch.bfloat16, enabled=img.device.type == "cuda"):
        for z, y, x in product(*(range(0, s, t) for s, t in zip(img.shape, tile))):
            t = encoder.forward_features(img[None, None, z:z + tile[0], y:y + tile[1], x:x + tile[2]])
            feats[z // p[0]:z // p[0] + g[0], y // p[1]:y // p[1] + g[1], x // p[2]:x // p[2] + g[2]] = t[0].reshape(*g, -1)
    return feats

def tile_logits(model, img, tile):
    """A dense model's (lib.decoders.unetr.Unetr) raw per-voxel logits over img (Z Y X in [0, 1], on the model's
    device, a multiple of tile), from tile-sized crops at the training global-view size, half a tile apart, blended:
    OutDim Z Y X. Unlike tile_features (whose Linear head needs no spatial context beyond one token's own patch),
    this tiles the whole model (encoder + decoder), since Unetr's skip connections need a full crop each call.
    A crop's voxels near its faces see little context, so each is weighted by a Hann window (0 at the faces, 1 at the
    centre; two crops half a tile apart sum to 1): the faces drop out where another crop covers them. The outer half tile
    of img has only one crop over it, so it is poorly predicted: read img with half a tile of context (read_block) and trim."""
    import torch
    assert all(s % t == 0 for s, t in zip(img.shape, tile)), f"{tuple(img.shape)} must tile by {tile}"
    ramp = [torch.sin(torch.pi * (torch.arange(t, device=img.device) + 0.5) / t) ** 2 for t in tile]  # t per axis
    starts = [range(0, s - t // 2, t // 2) for s, t in zip(img.shape, tile)]
    norm = [torch.zeros(s, device=img.device) for s in img.shape]  # per axis, the summed windows over it (the weights are separable)
    for n, r, ss, t in zip(norm, ramp, starts, tile):
        for s in ss:
            n[s:s + t] += r
    out = None
    with torch.no_grad():
        for z, y, x in product(*starts):
            logits = model(img[None, None, z:z + tile[0], y:y + tile[1], x:x + tile[2]])[0]  # OutDim tz ty tx
            if out is None:
                out = torch.zeros(logits.shape[0], *img.shape, dtype=logits.dtype, device=img.device)
            out[:, z:z + tile[0], y:y + tile[1], x:x + tile[2]] += logits * (ramp[0][:, None, None] * ramp[1][None, :, None] * ramp[2][None, None, :])
    assert out is not None, "img must be nonempty"
    return out.div_(norm[0][:, None, None]).div_(norm[1][None, :, None]).div_(norm[2][None, None, :])

def write_affinities(path: Path, array, **attrs):
    """A mia-evals affinity artifact (its artifact.write_artifact layout): one zarr array, Channel X Y Z, whose attrs say
    how to read it (kind, origin, convention, axes, native_box, scale, ...). mia-evals validates it when it scores it."""
    import zarr
    store = zarr.open_array(str(path), mode="w", shape=array.shape, dtype=array.dtype, chunks=tuple(min(256, s) for s in array.shape))
    store[:] = array
    store.attrs.update(kind="affinity", **attrs)

def score_block(pred, labels, sparse: bool) -> dict:
    """pred (C Z Y X affinity probabilities) against the affinities of labels (Z Y X instance ids; sparse: ProbeVolume.sparse),
    per channel: BCE, boundary AP (boundary = different objects) and the boundary base rate, on a fixed random subset (at most
    10M) of its valid pairs."""
    import torch
    from lib.probe import AFFINITY_OFFSETS_XYZ, affinities, average_precision
    aff, valid = affinities(labels, [o[::-1] for o in AFFINITY_OFFSETS_XYZ], sparse)  # x y z offsets; the model sees z y x
    g = torch.Generator(device=labels.device).manual_seed(0)
    s = {}
    for c, o in enumerate(AFFINITY_OFFSETS_XYZ):
        idx = valid[c].nonzero()
        idx = idx[torch.randint(len(idx), (min(len(idx), 10_000_000),), device=labels.device, generator=g)]
        q, t = pred[c][tuple(idx.T)].float(), aff[c][tuple(idx.T)].float()
        s[f"bce_{o}"] = float(torch.nn.functional.binary_cross_entropy(q.clamp(1e-6, 1 - 1e-6), t))
        s[f"boundary_ap_{o}"] = average_precision(1 - q, 1 - t)
        s[f"boundary_frac_{o}"] = float(1 - t.mean())
    return s

def mean_scores(per_block: list[dict]) -> dict:
    """score_block's dicts averaged over the test blocks (equally, as lmd_ssl_v1 weights its volumes), plus
    boundary_ap_short: the mean of the three short-range (+1) channels' AP."""
    from lib.probe import AFFINITY_OFFSETS_XYZ
    s = {k: sum(b[k] for b in per_block) / len(per_block) for k in per_block[0]}
    s["boundary_ap_short"] = sum(s[f"boundary_ap_{o}"] for o in AFFINITY_OFFSETS_XYZ[:3]) / 3
    return s

def boundary_png(path: Path, em, labels, pred):
    """A block's middle z: EM | true boundaries | predicted boundaries (1 - min of the 3 short-range affinities)."""
    import torch
    from PIL import Image
    from lib.probe import AFFINITY_OFFSETS_XYZ, affinities
    zm = labels.shape[0] // 2
    aff, _ = affinities(labels[zm:zm + 2], [o[::-1] for o in AFFINITY_OFFSETS_XYZ[:3]])  # short range: the next slice is enough
    row = [em[zm], 1 - aff[:, 0].float().amin(0), 1 - pred[:3, zm].float().amin(0)]
    gap = torch.ones(row[0].shape[0], 8, device=em.device)
    Image.fromarray((torch.cat([row[0], gap, row[1], gap, row[2]], 1) * 255).byte().cpu().numpy()).save(path)
