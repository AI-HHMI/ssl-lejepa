# Masked LeJEPA, one training step

As trained in e10 (`Lejepa.mask_loss`, lib/models/lejepa.py): masked tokens predicted from a second view of the same region, offset by a fraction of a token.

```python
# inputs and hyperparameters (e10's masked encoder)
#   vol        : float[B, 1, Z, Y, X] in [0, 1]   EM volumes at 8 nm, Z, Y, X >= view + shift
#   B          : int, batch size                  32
#   view       : int, view side in voxels         64
#   shift      : int, max view offset in voxels   24   (0 <= offset <= shift per axis)
#   patch      : int, token side in voxels        8
#   N          : int, tokens per view             (view / patch)^3 = 512
#   M          : int, masked tokens               round(0.6 * view / patch)^3 = 5^3 = 125
#   Nvis       : int, visible context tokens      N - M = 387
#   D          : int, encoder width               512
#   Dproj      : int, projector output dim        128
#   lamb       : float, SIGReg weight             10 / B
# modules
#   encoder    : ViT, 12 blocks x 512 wide;       float[B, 1, view^3] -> float[B, N, D]   (keep=: visible tokens only)
#   predictor  : 4 transformer blocks, D wide;    (ctx: float[B, Nvis, D], queries: float[B, M, D]) -> float[B, M, D]
#   projector  : MLP D -> 2048 -> 2048 -> Dproj, batch norm
#   mask_token : learned float[D]
#   sincos     : 3D sin/cos position embedding,  float[..., 3] (fractional token coords ok) -> float[..., D]
#   sigreg     : sketched isotropic-Gaussian test, float[samples, dims] -> float scalar

# one training step
region   = random_crop(vol, view + shift)              # e.g. 64 + 24 = 88^3
ctx_org  = randint(0, shift, size=(B, 3))              # voxel origins, independent per sample
tgt_org  = randint(0, shift, size=(B, 3))
ctx_view = crop(region, ctx_org, view)                 # context view
tgt_view = crop(region, tgt_org, view)                 # target view: same content, shifted
offset   = (tgt_org - ctx_org) / patch                 # in tokens, usually fractional
grid     = view / patch                                # e.g. 8^3 tokens

# one masked cube per sample, inside both views
cube        = random_cube(grid, side=round(0.6 * grid), fits=offset)
visible     = `ctx tokens outside cube`
masked      = `tgt tokens whose centres fall in cube`
masked_pos  = `positions of masked in ctx grid`         # = token index + offset, fractional

tgt_tokens  = encoder(tgt_view)                         # B N D, full view; gradients flow (no EMA, no stop-grad)
ctx_tokens  = encoder(ctx_view, keep=visible)           # B Nvis D, never sees the cube
queries     = mask_token + sincos(masked_pos)
pred        = predictor(ctx_tokens + sincos(visible), queries)   # B M D

pred_emb    = projector(pred)                           # shared MLP -> B M Dproj
tgt_emb     = projector(tgt_tokens)                     # B N Dproj
target      = tgt_emb[masked]                           # B M Dproj

loss = mse(pred_emb, target) +
        lamb * ( sigreg(`tgt_emb over batch, per token position`) +
        (B / N) * sigreg(`tgt_emb over tokens, per sample`) )
```

Why the shift: with one view, a token's position embedding is enough information to predict its (entire) target (see experiments e02-e04).
Offsetting the target view by a random fraction of a token prior to encoding prevents this form of collapse.
There is no EMA teacher and no stop-gradient: SIGReg alone prevents collapse, applied both across the batch at each token position and across the tokens within each sample.
