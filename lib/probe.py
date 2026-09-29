"""Linear probes on frozen encoder tokens: voxel-resolution affinity targets, the token <-> voxel layout of a
per-token linear head ("pixel-shuffle linear": each token predicts all voxels of its patch), its fit, and AP."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# mia-evals' neuron-instance affinity channels (its src/postprocess/mws.py SHORT_OFFSETS + LONG_OFFSETS), in the
# store's x y z axes: channel c at voxel p is 1 if p and p + offset_c belong to the same object.
AFFINITY_OFFSETS_XYZ = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (10, 0, 0), (0, 10, 0), (0, 0, 10)]

def affinities(labels: Tensor, offsets: list[tuple[int, ...]]) -> tuple[Tensor, Tensor]:
    """labels: Z Y X instance ids (0 = background); offsets: per channel (dz, dy, dx), all >= 0.
    Returns (aff, valid), both C Z Y X bool. aff: p and p + offset have the same nonzero id. valid: p + offset is
    inside the block and neither voxel is background (the loss and metrics ignore the rest)."""
    Z, Y, X = labels.shape
    aff = torch.zeros(len(offsets), Z, Y, X, dtype=torch.bool, device=labels.device)
    valid = torch.zeros_like(aff)
    for c, (dz, dy, dx) in enumerate(offsets):
        a, b = labels[:Z - dz, :Y - dy, :X - dx], labels[dz:, dy:, dx:]
        v = (a != 0) & (b != 0)
        aff[c, :Z - dz, :Y - dy, :X - dx] = (a == b) & v
        valid[c, :Z - dz, :Y - dy, :X - dx] = v
    return aff, valid

def to_tokens(vox: Tensor, p: int) -> Tensor:
    """C Z Y X voxels -> (Z/p Y/p X/p) x (C p p p): one row per token, holding its patch's voxels (probe targets)."""
    C, Z, Y, X = vox.shape
    return vox.reshape(C, Z // p, p, Y // p, p, X // p, p).permute(1, 3, 5, 0, 2, 4, 6).reshape(-1, C * p ** 3)

def to_voxels(tok: Tensor, grid: tuple[int, ...], p: int) -> Tensor:
    """Inverse of to_tokens: (Gz Gy Gx) x (C p p p) rows -> C Z Y X."""
    gz, gy, gx = grid
    C = tok.shape[1] // p ** 3
    return tok.reshape(gz, gy, gx, C, p, p, p).permute(3, 0, 4, 1, 5, 2, 6).reshape(C, gz * p, gy * p, gx * p)

def fit_probe(feats: Tensor, targets: Tensor, valid: Tensor) -> tuple[nn.Linear, list[tuple[int, float]]]:
    """Linear(D, K) from frozen token features (N D) to binary token targets (N K), BCE over the valid entries.
    Random token batches, Adam at a fixed lr; seeded, so a refit gives the same probe. Also returns the batch loss
    every 100 steps (the fit curve: flat by the end means STEPS was enough)."""
    STEPS, BATCH, LR = 3000, 4096, 1e-3
    head = nn.Linear(feats.shape[1], targets.shape[1]).to(feats.device)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    g = torch.Generator(device=feats.device).manual_seed(0)
    curve = []
    for step in range(STEPS):
        i = torch.randint(len(feats), (BATCH,), device=feats.device, generator=g)
        w = valid[i].float()
        loss = F.binary_cross_entropy_with_logits(head(feats[i].float()), targets[i].float(), weight=w, reduction="sum")
        loss = loss / w.sum().clamp_min(1)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0 or step == STEPS - 1:
            curve.append((step, loss.item()))
    return head, curve

def average_precision(scores: Tensor, labels: Tensor) -> float:
    """Average precision of 1-D scores for 1-D binary labels (area under the precision-recall curve, step-wise)."""
    hit = labels[scores.argsort(descending=True)].float()
    precision = hit.cumsum(0) / torch.arange(1, len(hit) + 1, device=hit.device)
    return float((precision * hit).sum() / hit.sum().clamp_min(1))
