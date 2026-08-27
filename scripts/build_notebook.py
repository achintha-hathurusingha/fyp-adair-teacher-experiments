"""Programmatically build AdaIR_3Degradation_Analysis.ipynb from the Excel
workbook + embeddings.npz + sample visuals produced by run_inference.py, then
execute it in-place (via nbconvert) so the delivered notebook already has
plots baked in.

Usage (on devon, adair-distill env):
  python build_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO / "AdaIR_3Degradation_Analysis.ipynb"

md = lambda src: nbf.v4.new_markdown_cell(src)
code = lambda src: nbf.v4.new_code_cell(src)

cells = []

cells.append(md("""\
# AdaIR — White-Box Analysis of the 3-Degradation Teacher

This notebook analyses how the frozen **AdaIR** teacher (3-degradation
all-in-one checkpoint: dehazing + deraining + denoising) behaves internally
across three degradation types, by instrumenting every **AFLB**
(Adaptive Frequency Learning Block — `FreModule` in the code) in its decoder.

* 300 test images: 100 Rain100L (derain), 100 SOTS-outdoor (dehaze), 100 BSD68 (denoise, sigma in {15,25,50})
* No retraining — pure forward-pass instrumentation of the released checkpoint
* Full numeric trail in `AdaIR_3Degradation_Analysis.xlsx`; full-precision tensors in `intermediates/`

See the workbook's **README** sheet for the complete code-name -> paper-concept mapping
(AFLB / MGB / FMiM / FMoM / H-L / L-H) — reproduced below for convenience.
"""))

cells.append(code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

REPO = Path('.').resolve()
CSV_DIR = REPO / 'csv_export'  # source of truth -- see README cell for why this isn't read from .xlsx here
DEG_COLORS = {'Rain': '#3b7dd8', 'Haze': '#d8853b', 'Noise': '#3bb273'}
DEG_ORDER = ['Rain', 'Haze', 'Noise']
AFLB_ORDER = ['AFLB1', 'AFLB2', 'AFLB3']

def _csv(sheet):
    matches = list(CSV_DIR.glob(f'*{sheet}.csv'))
    assert matches, f'no CSV for {sheet!r} in {CSV_DIR}'
    return pd.read_csv(matches[0])

readme_text = next(CSV_DIR.glob('*README.csv')).read_text(encoding='utf-8')
if readme_text.split('\\n', 1)[0].strip() != 'README':
    readme_lines = readme_text.split('\\n')
else:
    readme_lines = _csv('README')['README'].astype(str).tolist()

image_info = _csv('Image_Info')
mgb = _csv('MGB_Values')
freq = _csv('Frequency_Statistics')
fmim = _csv('FMiM_Statistics')
fmom = _csv('FMoM_Statistics')
stages = _csv('Transformer_Features')
output_metrics = _csv('Output_Metrics')
cross_comp = _csv('Cross_Degradation_Comparison')
pca_tsne = _csv('PCA_TSNE_Coordinates')
tensor_index = _csv('Tensor_File_Index')

print(f"{len(image_info)} images loaded")
image_info['Degradation'].value_counts()
"""))

cells.append(md("### Full architecture / column reference (README, verbatim)"))
cells.append(code("print('\\n'.join(readme_lines))"))

cells.append(md("""\
## 1. Restoration quality sanity check

Before trusting the internal statistics, confirm the teacher is actually
restoring these 300 images well (PSNR/SSIM per degradation).
"""))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for i, metric in enumerate(['PSNR', 'SSIM']):
    data = [image_info[image_info.Degradation == d][metric] for d in DEG_ORDER]
    bp = axes[i].boxplot(data, tick_labels=DEG_ORDER, patch_artist=True)
    for patch, d in zip(bp['boxes'], DEG_ORDER):
        patch.set_facecolor(DEG_COLORS[d])
        patch.set_alpha(0.6)
    axes[i].set_title(metric)
    axes[i].grid(alpha=0.3)
fig.suptitle('Restoration quality by degradation (3-degradation teacher)')
fig.tight_layout()
plt.show()
output_metrics
"""))

cells.append(md("""\
## 2. MGB — does AdaIR actually move its learned frequency boundary?

`alpha`/`beta` are the sigmoid-gated rates (per image, per AFLB) that decide
how large the retained low-frequency box is, before it gets floored by the
hardcoded `n=128` spatial normalisation inside `FreModule.fft()`.
"""))
cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, aflb in zip(axes, AFLB_ORDER):
    sub = mgb[mgb.AFLB == aflb]
    data = [sub[sub.Degradation == d]['alpha'] for d in DEG_ORDER]
    bp = ax.boxplot(data, tick_labels=DEG_ORDER, patch_artist=True)
    for patch, d in zip(bp['boxes'], DEG_ORDER):
        patch.set_facecolor(DEG_COLORS[d]); patch.set_alpha(0.6)
    ax.set_title(f'{aflb}: alpha (height rate)')
    ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
"""))
cells.append(code("""\
print("mask_area_pct summary (% of the CxHxW volume marked as 'low frequency'):")
display(mgb.groupby(['Degradation', 'AFLB'])['mask_area_pct'].describe()[['mean', 'std', 'max']])
"""))

cells.append(md("""\
### Finding: the frequency mask is degenerate at benchmark resolutions

`FreModule.fft()` computes the retained low-frequency half-width as
`h_ = int(h // 128 * alpha)`. At every AFLB, for every one of these 300
images (Rain100L/SOTS/BSD68 are all well under ~550px), the feature-map
spatial size at the relevant decoder stage is smaller than 128 in at least
one axis (AFLB1: ~40x60, AFLB2: ~80x120, AFLB3: ~160x240 for a typical
480x320 input) — the floor division `h // 128` truncates to 0 (or, for
AFLB3, to 1, then `int(1 * ~0.5)` still truncates to 0). **`mask_area_pct` is
~0 for essentially every image at every AFLB.** This means at these
resolutions AdaIR's low-frequency branch (`X_low`) is receiving an
all-zero mask -> its `raw_low` feature is the inverse-FFT of nothing, and
`raw_high` carries effectively the *entire* spectrum, not a genuine
high-pass split. Section 3 confirms this numerically (the low-energy
percentage below is ~0% for all three degradation types).

This is a property of the **released code at standard benchmark
resolutions**, not an artefact of this analysis pipeline (verified by
recomputing `h_`/`w_` directly from the saved `conv_feat` shapes and
`threshold` values in `intermediates/`).
"""))

cells.append(md("## 3. Frequency energy split — confirms the masking finding at scale"))
cells.append(code("""\
agg = freq.groupby(['Degradation', 'AFLB'])[['low_pct', 'high_pct']].mean().reset_index()
fig, ax = plt.subplots(figsize=(9, 4.5))
width = 0.25
x = np.arange(len(AFLB_ORDER))
for i, d in enumerate(DEG_ORDER):
    sub = agg[agg.Degradation == d].set_index('AFLB').reindex(AFLB_ORDER)
    ax.bar(x + i * width, sub['low_pct'], width, label=d, color=DEG_COLORS[d], alpha=0.8)
ax.set_xticks(x + width); ax.set_xticklabels(AFLB_ORDER)
ax.set_ylabel('mean low-frequency energy %'); ax.set_title('Low-frequency energy share by degradation x AFLB')
ax.legend(); ax.grid(alpha=0.3)
plt.show()
agg
"""))

cells.append(md("""\
## 4. FMiM — feature statistics across the mining stage

Mean / std / L2-energy fingerprints for `conv_feat`, the FFT magnitude, the
raw low/high split (pre cross-attention), and the mined low/high features
(post cross-attention with the decoder feature `y`).
"""))
cells.append(code("""\
feat_order = ['conv_feat', 'fft_magnitude', 'raw_low', 'raw_high', 'mined_low', 'mined_high']
piv = fmim.pivot_table(index=['AFLB', 'Feature'], columns='Degradation', values='energy', aggfunc='mean')
piv = piv.reindex(pd.MultiIndex.from_product([AFLB_ORDER, feat_order], names=['AFLB', 'Feature']))
piv[DEG_ORDER]
"""))
cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
for ax, aflb in zip(axes, AFLB_ORDER):
    sub = fmim[(fmim.AFLB == aflb) & (fmim.Feature.isin(['raw_low', 'raw_high', 'mined_low', 'mined_high']))]
    m = sub.groupby(['Feature', 'Degradation'])['energy'].mean().unstack()
    m = m.reindex(['raw_low', 'raw_high', 'mined_low', 'mined_high'])[DEG_ORDER]
    m.plot(kind='bar', ax=ax, color=[DEG_COLORS[d] for d in DEG_ORDER], alpha=0.8, legend=(aflb == 'AFLB1'))
    ax.set_title(aflb); ax.set_ylabel('mean L2 energy'); ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
fig.suptitle('FMiM feature energy: raw split vs. mined (post cross-attention) split')
fig.tight_layout()
plt.show()
"""))

cells.append(md("""\
## 5. FMoM — H-L / L-H cross-frequency attention

`hl_spatial_weight` (H-L unit, spatial attention computed from the high-freq
branch, applied to low) and `lh_channel_weight` (L-H unit, channel attention
from the low-freq branch, applied to high).
"""))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col, title in zip(axes, ['hl_mean', 'lh_mean'], ['H-L spatial weight (mean)', 'L-H channel weight (mean)']):
    m = fmom.groupby(['AFLB', 'Degradation'])[col].mean().unstack().reindex(AFLB_ORDER)[DEG_ORDER]
    m.plot(kind='bar', ax=ax, color=[DEG_COLORS[d] for d in DEG_ORDER], alpha=0.8)
    ax.set_title(title); ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
"""))

cells.append(md("""\
## 6. Does degradation-specific separation strengthen deeper in the network?

Mean L2 energy at every encoder/decoder stage plus each AFLB's output,
per degradation, in forward-pass order.
"""))
cells.append(code("""\
layer_order = ['encoder_level1', 'encoder_level2', 'encoder_level3', 'latent', 'AFLB1_output',
               'decoder_level3', 'AFLB2_output', 'decoder_level2', 'AFLB3_output',
               'decoder_level1']
m = stages.groupby(['Layer', 'Degradation'])['energy'].mean().unstack()
m = m.reindex(layer_order)[DEG_ORDER]
# normalise each layer's energy to Rain=1 so the three curves are comparable despite scale changes across the U-Net
m_norm = m.div(m['Rain'], axis=0)

fig, ax = plt.subplots(figsize=(11, 5))
for d in DEG_ORDER:
    ax.plot(layer_order, m_norm[d], marker='o', label=d, color=DEG_COLORS[d])
ax.axhline(1.0, color='gray', lw=0.5)
ax.set_xticklabels(layer_order, rotation=45, ha='right')
ax.set_ylabel('mean energy, normalised to Rain=1 per layer')
ax.set_title('Depth-wise feature energy divergence across degradations')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
m
"""))

cells.append(md("""\
## 7. PCA / t-SNE of pooled AFLB-output features

Each image is represented by concatenating the global-average-pooled output
of AFLB1, AFLB2, AFLB3 (a 384+192+96 = 672-d vector), matching the paper's
own claim that AdaIR learns discriminative degradation contexts.
"""))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (xcol, ycol, title) in zip(axes, [('PC1', 'PC2', 'PCA'), ('TSNE1', 'TSNE2', 't-SNE')]):
    for d in DEG_ORDER:
        sub = pca_tsne[pca_tsne.Degradation == d]
        ax.scatter(sub[xcol], sub[ycol], s=18, alpha=0.7, label=d, color=DEG_COLORS[d])
    ax.set_title(f'{title} of pooled AFLB-output embeddings'); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
"""))

cells.append(md("## 8. Sample pipeline visualisations (one per degradation)"))
cells.append(code("""\
visual_files = sorted((REPO / 'outputs' / 'visuals').glob('*_pipeline.png'))
fig, axes = plt.subplots(len(visual_files), 1, figsize=(16, 8 * len(visual_files)))
if len(visual_files) == 1:
    axes = [axes]
for ax, f in zip(axes, visual_files):
    ax.imshow(mpimg.imread(f)); ax.axis('off'); ax.set_title(f.stem)
fig.tight_layout()
plt.show()
"""))

cells.append(md("""\
## 9. Summary

* Restoration quality (section 1) confirms the checkpoint is behaving as the
  released 3-degradation teacher should.
* **The frequency-masking mechanism (MGB) is inactive at these benchmark
  resolutions** — `mask_area_pct` is ~0 for all 300 images at all 3 AFLBs, a
  direct consequence of the hardcoded `n=128` normalisation in
  `FreModule.fft()` versus feature maps that are smaller than 128px per side.
  In effect, `X_low` is empty and `X_high` inherits (almost) the whole
  spectrum. **Appendix A** independently re-verifies this from a fresh
  forward pass (not just the saved statistics) and empirically maps out the
  input resolution at which the mechanism *does* switch on.
* The correct, narrow claim (see Appendix A for the reasoning): the specific
  binary frequency-boundary mechanism in this `FreModule.fft()` path is
  inactive at the resolutions it is normally evaluated at — this is **not**
  evidence that "AdaIR doesn't use frequency information" (restoration
  quality above is strong), only that this particular mechanism, at these
  resolutions, isn't where that quality is coming from.
* Despite that, FMiM/FMoM statistics (sections 4-5) and the depth-wise energy
  trend (section 6) still show measurable, consistent separation between Rain
  / Haze / Noise — meaning the discriminative behaviour the paper attributes
  to frequency mining is, at least partly, coming from the cross-attention
  and gating stages downstream of the (non-functional) frequency split,
  rather than from the split itself.
* The PCA/t-SNE projection (section 7) shows how separable the three
  degradation types are in the pooled AFLB-output feature space.

Full per-image numbers: `AdaIR_3Degradation_Analysis.xlsx`.
Full-precision tensors for any (image, AFLB): `intermediates/<degradation>/<Image_ID>/aflb{1,2,3}.pt`,
indexed by the `Tensor_File_Index` sheet.
"""))

cells.append(md("""\
## Appendix A — Mechanism audit: is the zero mask a bug or real teacher behaviour?

The zero `mask_area_pct` in section 2 is surprising enough that it deserves
independent re-verification, not just a re-read of the same saved tensors
that produced it. This appendix reruns a **fresh forward pass** on `R001`
(`scripts/trace_single_image.py`) and checks the MGB -> mask -> FFT-split
pipeline against the literal AdaIR source, plus a **resolution sweep**
(`scripts/resolution_sweep.py`) that resizes one image up from 128px to
1024px and re-runs the full model at each size, to find the input resolution
at which the mechanism actually switches on.
"""))
cells.append(code("""\
trace = _csv('12_Trace_R001')
trace
"""))
cells.append(md("""\
Five independent checks, all against a fresh forward pass (not the cached
300-image tensors):

1. **`Ml + Mh == 1` everywhere** — mask complement is constructed correctly.
2. **`||F||^2 == ||F_low||^2 + ||F_high||^2`** (exact, not approximate, since
   the mask is binary and the two halves have disjoint support) — the energy
   split is self-consistent.
3. **Manual `ifft(F_low)` / `ifft(F_high)` reconstruction matches the saved
   `raw_low` / `raw_high` tensors** — the extraction code isn't silently
   diverging from what actually feeds the rest of the network.
4. **`mask.unique() == [0.0]`** — the low-frequency mask is genuinely,
   exactly empty, at every AFLB, for this image.
5. **`h_ = int((h // 128) * alpha)`** traced by hand: `h // 128` floors to 0
   at AFLB1/AFLB2 and to 1 at AFLB3 for a typical Rain100L input; even where
   it's 1, `int(1 * ~0.5)` still truncates to 0.
"""))
cells.append(code("""\
assert trace['verify_Ml_plus_Mh_eq_1'].all()
assert trace['verify_manual_ifft_low_matches'].all()
assert trace['verify_manual_ifft_high_matches'].all()
assert (trace['energy_rel_error'] < 1e-6).all()
assert (trace['mask_area_pct'] == 0).all()
print('All 5 checks pass on R001, all 3 AFLBs -- the zero mask is real, not an extraction bug.')
"""))
cells.append(md("### A.1 Visual trace (R001, all 3 AFLBs)"))
cells.append(code("""\
trace_files = sorted((REPO / 'outputs' / 'trace_R001').glob('*_trace.png'))
fig, axes = plt.subplots(len(trace_files), 1, figsize=(20, 4 * len(trace_files)))
if len(trace_files) == 1:
    axes = [axes]
for ax, f in zip(axes, trace_files):
    ax.imshow(mpimg.imread(f)); ax.axis('off')
fig.tight_layout()
plt.show()
"""))
cells.append(md("""\
FFT magnitude is a tight spike at the centre (DC/near-DC dominance is
expected for natural images); the low mask (Ml) is solid black (0 pixels
selected) at every AFLB; the high mask (Mh) is solid white; "high spectrum"
is essentially the full reconstructed signal, not a genuine high-pass
result — there is no split happening for this image at native resolution.
"""))

cells.append(md("""\
### A.2 Resolution sweep — when does the mechanism switch on?

Feature-map size at each AFLB is a fixed fraction of the *input* resolution
(three 2x downsamples before the latent): AFLB1 = H/8, AFLB2 = H/4,
AFLB3 = H/2. `h_` only becomes non-zero once `h // 128` climbs high enough
that `int((h // 128) * alpha)` stops truncating to 0 (alpha/beta sit close
to 0.5 in practice, so `h // 128` generally needs to reach ~3). One real
image (`rain-001.png`) was resized up from 128px to 1024px and pushed
through the **full** AdaIR forward pass at each size (1536px/2048px hit
GPU OOM on this 24GB card at fp32/batch=1 and were left untested).
"""))
cells.append(code("""\
sweep = _csv('11_Resolution_Sweep')
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for aflb in AFLB_ORDER:
    sub = sweep[sweep.AFLB == aflb].sort_values('input_size')
    axes[0].plot(sub.input_size, sub.mask_area_pct, marker='o', label=aflb)
    axes[1].plot(sub.input_size, sub.h_over_128, marker='o', label=aflb)
axes[0].set_xlabel('input resolution (px)'); axes[0].set_ylabel('mask_area_pct')
axes[0].set_title('Low-frequency mask activation vs. input resolution')
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].set_xlabel('input resolution (px)'); axes[1].set_ylabel('h // 128')
axes[1].set_title('Floor-divided feature size vs. input resolution')
axes[1].axhline(1, color='gray', lw=0.5, ls='--')
axes[1].legend(); axes[1].grid(alpha=0.3)
fig.tight_layout()
plt.show()

print('First activating input resolution per AFLB:')
for aflb in AFLB_ORDER:
    active = sweep[(sweep.AFLB == aflb) & (sweep.mask_active == True)]
    if len(active):
        print(f'  {aflb}: {int(active.input_size.min())}px')
    else:
        print(f'  {aflb}: not activated up to {int(sweep.input_size.max())}px (untested beyond, GPU OOM)')
sweep
"""))
cells.append(md("""\
**Empirical thresholds**: AFLB3 (shallowest AFLB, largest feature map)
activates first, at input resolution 768x768. AFLB2 and AFLB1 do not
activate anywhere in the tested range up to 1024x1024 — consistent with
needing roughly 4x and 8x more input resolution respectively (since their
feature maps are 4x and 8x smaller than AFLB3's), which pushes well past
what fits on a 24GB GPU at fp32/batch=1 for this architecture.

**Conclusion**: the mechanism is not fundamentally broken — it is a genuine
adaptive-boundary mechanism that *can* activate — but AdaIR's own published
benchmarks (Rain100L, SOTS, BSD68, all <=~550px) sit far below the
resolution where it ever turns on. This reframes the original question from
"how does the frequency mask differ between Rain/Haze/Noise?" to: **if this
mechanism is inert at benchmark resolution, where does AdaIR's restoration
quality actually come from, and is that mechanism worth distilling into an
NPU student at all?** — the question the next phase of this project (a
teacher mechanism audit, then a resolution/ablation study, before any
student design) is set up to answer.
"""))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "adair-distill", "language": "python", "name": "adair-distill"},
    "language_info": {"name": "python"},
}
nbf.write(nb, NOTEBOOK_PATH)
print(f"wrote {NOTEBOOK_PATH}")
