# TEST08-C — Compact Degradation State + Spatial Conditioning

## 1. Research Question

TEST07-B established that a student NAFNet can be trained to closely match a
frozen teacher's compact (16-dim) degradation-aware latent (cosine similarity
~0.99, probe accuracy ~96%), but that matching this embedding via an auxiliary
MSE loss alone did not improve restoration quality (mean ΔPSNR -0.79dB vs. an
undistilled baseline, consistent across 3 seeds). TEST08-C asks the natural
follow-up: **does actively using that same embedding to condition the spatial
restoration computation — rather than treating it purely as a feature-matching
target — convert the successfully transferred representation into a measurable
restoration improvement?** The central comparison is **C vs. B**, not C vs. A,
because B already demonstrates successful representation transfer; the question
is whether *using* that representation (H_COND) does anything B's passive
matching does not.

## 2. Motivation from TEST07-B

TEST07-B's key numbers that motivate this experiment: mean ΔPSNR(B-A) = -0.79dB
(all 3 seeds negative), with a striking degradation-specific split — Rain -2.59dB,
Haze -1.03dB, Noise **+1.24dB** (all 3 seeds positive for Noise). This pattern —
good representation transfer, no restoration benefit, and a degradation-dependent
sign flip — is exactly what H_COND predicts if the representation is present but
unused: the embedding "knows" about the degradation but the network has no
mechanism to act on that knowledge except through the shared, degradation-agnostic
decoder weights.

## 3. Models A/B/C

All three share the identical locked NAFNet M-arm backbone (read-only import from
`fyp-adair-distill`), composed via the same faithful forward-pass replica pattern
used since TEST04.

- **Model A**: plain NAFNet baseline. No KD, no conditioning.
- **Model B**: the validated TEST07-B configuration exactly reproduced — bottleneck
  → GAP+GMP(512) → `Linear(512,16)` = e_S, trained with
  `L = L_restore + 0.1·MSE(e_S, e_T)`. Reference distillation baseline.
- **Model C**: identical to B, but e_S additionally conditions the bottleneck via
  `F_cond = (1+γ)·F + β`, with `γ = 1 + G_γ(e_S)`, `β = G_β(e_S)` from two
  zero-initialized `Linear(16,256)` heads (so conditioning starts as an exact
  identity transform). **The same e_S is used for both the KD loss and the
  conditioning** — no separate embeddings — which isolates whether one learned
  state can both represent and control restoration, per the task's explicit
  design intent.
- **Model D** (oracle teacher-conditioned control): **NOT RUN**, per the task's
  explicit final instruction not to train it.

Model A/B numbers in this experiment **exactly reproduce TEST07-B's own results**
bit-for-bit (same seeds, same data, same architecture, fully deterministic) —
confirming both experiments' training pipelines are consistent and trustworthy.

## 4. Dataset and Training

TEST07-B's dataset was reused **read-only, unmodified** — same 80 train / 20 val
scene-disjoint split, same 8-crops-per-train-scene cache, same fixed validation
crops, same Rain/Haze/Noise synthesis. This is valid reuse per the task's own
rule ("reuse only if the exact training split and crop definition are identical")
since `test08_c` reads directly from `test07_b/results/dataset_manifest.csv` and
`test07_b/results/teacher_cache/` without touching those files.

Training: Adam, LR=2e-4, batch=8, 50 epochs, seeds {0,1,2}, identical across A/B/C
except the KD/conditioning branches — 9 total runs. All 9 completed cleanly using
the per-run-CSV-file pattern established after TEST07-B's shared-file race
condition (each run writes its own output file; a separate merge step combines
them — no shared-file writes during parallel execution this time).

## 5. Teacher Representation

Reused, unmodified, from TEST07-B: checkpoint SHA256-verified
(`f3822d9c...5937fb`), frozen throughout, `latent_pre` → GAP+GMP (768-dim) →
StandardScaler+PCA-16 fit on 1,920 training-split records only (leakage-safe),
explained variance 68.44%.

## 6. Conditioning Mechanism

Model C conditions **only the deepest bottleneck** (per spec — encoder, skips,
and decoder blocks are untouched), using a channel-wise, spatially-broadcast
affine transform: `F_cond[b,c,h,w] = (1+γ[b,c])·F[b,c,h,w] + β[b,c]`. No spatial
attention, no FFT, no dynamic convolution, no inference-time degradation
classifier — all per the task's explicit constraints. γ/β come from two
`Linear(16,256)` heads (2×16×256 + 2×256 = 8,704 extra parameters), zero-initialized
so training starts from an exact identity transform and any deviation is learned,
not imposed.

## 7. Restoration Results

Primary metric: last-5-epoch mean validation PSNR/SSIM (as in TEST07-B).

| Comparison | Mean Δ last5 PSNR (dB) | Std | 95% bootstrap CI | All 3 seeds same sign? |
|---|---|---|---|---|
| C − A | -0.519 | 0.090 | [-0.616, -0.437] | Yes, all negative |
| **C − B** | **+0.274** | 0.273 | **[+0.048, +0.577]** | **Yes, all positive** |
| B − A | -0.793 | 0.231 | [-1.014, -0.553] | Yes, all negative (reproduces TEST07-B) |

**C consistently beats B in all 3 seeds** on the primary metric, and the 95%
bootstrap CI excludes zero — but the per-seed magnitude shrinks noticeably
(+0.577, +0.196, +0.048 dB across seeds 0,1,2), so the *size* of the improvement
is not itself stable, only its direction. C still underperforms A overall.
ΔSSIM is positive for both C-A and C-B in all 3 seeds.

## 8. Degradation-Specific Results

This is where the picture sharpens — averaging would hide it:

| Degradation | C−B ΔPSNR (dB) | B−A ΔPSNR (dB) | C−A ΔPSNR (dB) |
|---|---|---|---|
| Rain | **+0.846** | -2.591 | -1.745 |
| Haze | -0.057 | -1.027 | -1.084 |
| Noise | +0.033 | +1.238 | +1.271 |

**Conditioning recovers roughly a third of B's Rain regression** (-2.59 → -1.75dB
vs. A, a +0.85dB recovery from B), while Haze is essentially unchanged by
conditioning (still as damaged as B) and Noise — already a B-vs-A win — is
preserved but not further improved by conditioning. So: **conditioning helps
specifically where the passive representation was hurting most (Rain), does
nothing for Haze, and doesn't erode the pre-existing Noise gain.**

## 9. Representation Results

| Representation | Mean accuracy | Std |
|---|---|---|
| Teacher PCA-16 | 96.1% | - |
| Model A bottleneck | 86.3% | 6.9pp |
| Model B bottleneck | 98.4% | 0.3pp |
| Model B e_S | 96.2% | 0.1pp |
| Model C bottleneck | 98.6% | 0.3pp |
| Model C e_S | 96.3% | 0.03pp |

B and C have **statistically indistinguishable representation quality** — both
match/exceed the teacher's own probe accuracy, both align with cosine similarity
~0.988. This is the critical control: **the restoration difference between B and
C cannot be explained by C having a "better" embedding** — B and C's embeddings
are essentially the same. The difference must come from what C *does* with that
embedding.

## 10. Conditioning Statistics

γ and β respond systematically and differently to each degradation (aggregated
over the 60 validation crops × 3 seeds):

| Degradation | γ mean | β mean | Relative bottleneck change |
|---|---|---|---|
| Rain | 0.721 | +0.018 | 0.470 |
| Haze | 0.634 | +0.422 | **0.655** |
| Noise | **1.362** | -0.376 | 0.526 |

Noise pushes γ well above 1 (feature amplification), Haze and Rain push γ below 1
(suppression), and β sign flips too (Haze positive, Noise negative). The
conditioning path is clearly **not** collapsing toward the identity (γ≈1, β≈0) —
it has learned a distinct, degradation-dependent modulation for each case, with
relative bottleneck change 0.47-0.66 (i.e., the conditioned bottleneck differs
from the raw bottleneck by 47-66% of its own norm — a substantial, not marginal,
modulation).

## 11. Random / Zero Controls

The critical causal-attribution test: does the restoration benefit come from the
*semantic content* of e_S, or merely from having an extra learned affine
transform available?

| Condition | Mean PSNR (dB) | Mean SSIM |
|---|---|---|
| **Learned e_S** | **26.95** | **0.833** |
| Zero embedding (≈identity) | 25.15 | 0.781 |
| Shuffled e_S (mismatched sample) | 23.43 | 0.737 |
| Random matched (same mean/std, no semantics) | 23.43 | 0.739 |

Learned conditioning beats every control by a wide margin (≥1.8dB over zero, ≥3.5dB
over random/shuffled). Critically, **random_matched and shuffled perform far worse
than even the zero/identity control** — an arbitrary or mismatched affine
transform actively *hurts* restoration rather than being a harmless no-op. This
confirms the conditioning benefit is not "any affine transform helps"; it requires
the *correct, learned, degradation-appropriate* signal.

## 12. Student-Side Embedding Intervention

For each validation scene, swapping in a donor degradation's e_S (same scene,
different degradation) in place of the recipient's own embedding:

| Recipient ← Donor | ΔPSNR vs. recipient's normal output |
|---|---|
| Haze ← Rain | -7.50 dB |
| Haze ← Noise | -6.58 dB |
| Noise ← Haze | -5.45 dB |
| Rain ← Haze | -5.45 dB |
| Noise ← Rain | -3.18 dB |
| Rain ← Noise | -1.00 dB |

**Every single direction causes a large restoration penalty** (-1.0 to -7.5dB) —
there is no direction where a mismatched embedding is harmless. This is direct,
causal evidence that e_S controls restoration behavior in a degradation-specific
way, not just decoratively. The asymmetry is itself informative: Haze as
recipient is hurt worst by any donor swap (-6.6 to -7.5dB), while Rain as
recipient tolerates a Noise donor comparatively well (-1.0dB) — consistent with
Rain and Noise's γ/β signatures being closer to each other than either is to
Haze's.

## 13. Complexity

| | Model A | Model B | Model C |
|---|---|---|---|
| Params | 7,371,923 | 7,380,131 | 7,388,835 |
| MACs @128px | 1,033,040,896 | 1,033,049,088 | 1,033,057,280 |

Model C adds 8,704 params over B (two `Linear(16,256)` heads) and negligible MACs
(the channel-wise affine is elementwise). Total overhead vs. A: +16,912 params
(+0.229%), +16,384 MACs (+0.0016%) — still effectively free relative to the 7.37M/
1.03GMACs backbone. Theoretical complexity only; no NPU latency claim.

## 14. Limitations

- N=3 seeds: all comparisons are exploratory evidence, not statistically
  significant claims. The C-B improvement, while consistent in direction across
  all 3 seeds, shrinks substantially in magnitude across seeds (0.577→0.196→0.048),
  which is a meaningful caveat on how large or reliable the effect size actually is.
- Conditioning was restricted to the deepest bottleneck only, per the task's
  explicit scope — multi-depth or encoder/skip conditioning was not tested and may
  behave very differently (in either direction).
- Haze remains essentially unrescued by this conditioning mechanism; whatever B's
  Haze failure mode is, single-depth affine bottleneck conditioning does not
  address it.
- Model D (oracle teacher-conditioned control) was not run, so we cannot yet
  separate "the student's own e_S is a slightly weaker conditioning signal than
  the teacher's" from "affine conditioning itself has a ceiling regardless of
  embedding source" — this remains open.

## 15. GO / NO-GO

Applying the pre-specified decision rule:

- **GO** requires C to consistently improve over B across seeds AND show
  meaningful gains in **at least two** degradation types. Only one (Rain) shows a
  meaningful, unambiguous recovery; Haze and Noise are essentially flat. **Not met.**
- **PARTIAL GO**: C≈B overall but strongly recovers one or more degradation
  failures, especially Rain/Haze → investigate degradation-specific or
  multi-depth conditioning. **This is what the evidence shows**: C's improvement
  over B (mean +0.27dB, all 3 seeds positive) is small and driven almost entirely
  by Rain's substantial recovery (+0.85dB), with Haze essentially untouched.
- **NO-GO** requires C≤B consistently with little bottleneck effect — **not met**:
  C beat B in all 3 seeds, and the conditioning demonstrably alters the bottleneck
  substantially (47-66% relative change) with clear causal control (Section 12).
- **Interesting failure** (bottleneck changes strongly but restoration doesn't
  improve) — **partially applies to Haze specifically**: Haze shows the largest
  relative bottleneck change (0.655) of any degradation, yet its restoration
  outcome versus B is flat. This is a genuine, degradation-localized instance of
  "the student is using the representation but the affine operator isn't
  expressive enough" — exactly the pattern that motivates low-rank dynamic
  spatial operators as raised in the task's own decision tree, but only for Haze,
  not universally.

**Decision: PARTIAL GO.** Spatial conditioning from the compact degradation
embedding is real, causal, and measurably helps restoration where the passive
representation was failing most (Rain). It does not close the gap to the
undistilled baseline overall, and it leaves Haze unaddressed. This is not evidence
that conditioning "doesn't work" — the random/shuffled controls and the embedding
intervention both rule out that interpretation — it is evidence that **single-depth
bottleneck-only affine conditioning is a real but incomplete mechanism**, and the
next step should extend or specialize it rather than abandon it.

**Recommendation for TEST09**: extend conditioning to multiple decoder depths
(not just the bottleneck) and/or test a degradation-specific or higher-rank
conditioning operator specifically targeting Haze's unrescued failure mode —
Haze shows the largest bottleneck modulation magnitude of any degradation
(Section 10) yet gets no restoration benefit from it, suggesting the channel-wise
affine form itself, not the embedding, is the bottleneck (pun intended) for Haze
specifically. A targeted low-rank spatial operator (not full spatial attention,
staying NPU-friendly per the project's constraints) for Haze is a concrete,
scoped next experiment, more promising than a full architecture change.
