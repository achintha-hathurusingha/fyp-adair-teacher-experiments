# TEST03 Design — Controlled Same-Scene Degradation Study

## 1. Confound TEST03 removes

TEST02 classified Rain100L-vs-SOTS-outdoor-vs-BSD68, not
rain-vs-haze-vs-noise in isolation — every "Rain" image came from
Rain100L, every "Haze" image from SOTS-outdoor, every "Noise" image from
BSD68. A linear probe scoring 71.7% on raw input pixels alone proves these
three *source datasets* are visually separable before any AdaIR
computation touches them; the 93-100% internal-stage accuracies could
therefore be partly (or, in the worst case, entirely) re-discovering
dataset identity rather than degradation type. TEST03 removes this by
synthesizing all three degradations from the **same 100 clean base
images**, so scene content is held fixed and only the degradation varies.

## 2. What remains controlled (unchanged from TEST01/TEST02)

- Checkpoint: `adair3d.ckpt`, loaded via the same strict loader (0
  missing/0 unexpected keys), unmodified.
- Architecture: unmodified, no retraining, no fine-tuning.
- Preprocessing: same `crop_img(base=16)` cropping convention.
- Feature-extraction locations: same 14 pipeline stages + 10 AFLB
  sub-features per AFLB, captured via the same non-intrusive hooks
  (`teacher-experiments/scripts/instrument.py`, reused unmodified).
- Pooling: same GAP+GMP concatenation.
- Classifiers: same Logistic Regression + Linear SVM, same random
  baseline (33.3%).
- Host/pinning: same `taskset -c 0-7,12-31` mitigation for devon's flaky
  cores 8-11; CSV-first, Excel-rendered-locally policy unchanged.

## 3. What changes

- **Image source**: 100 clean scenes (the Rain100L *ground-truth*
  `norain-*.png` images — natural photos, not previously used as inputs
  in TEST01/TEST02, only as derain targets) instead of three disjoint
  degraded datasets.
- **Degradation synthesis**: all three degradations (rain, haze, noise)
  are synthesized by TEST03 itself, deterministically, from the same
  clean image — not sourced from pre-existing degraded datasets.
- **Evaluation unit**: the *scene*, not the *image*. Cross-validation is
  grouped by `scene_id` (`GroupKFold`) so no fold ever trains on one
  degraded version of a scene and tests on another version of the *same*
  scene.
- **New analyses**: same-scene cross-degradation distance vs.
  same-degradation cross-scene distance (Phase 10-11) — the core new
  evidence TEST02 could not provide, directly separating "degradation
  effect" from "scene effect" on the representation.
- **Restoration ground truth**: the ORIGINAL clean image (not the
  synthetic degraded image) is used for PSNR/SSIM.

## 4. Degradation synthesis methods (exact, reproducible)

All three use a per-image deterministic seed (`np.random.RandomState(abs(hash(scene_id)) % 2**31)`,
same mechanism as the noise-seeding fix from TEST01) so every synthesis is
exactly reproducible from `scene_id` alone.

**Noise**: `I_noise = clip(I_clean + n, 0, 255)`, `n ~ N(0, sigma^2)`,
`sigma = 25` (fixed, matching the canonical mid-level AdaIR/AirNet
denoising protocol used throughout this project). One moderate condition
only (per the task's fallback instruction, to keep the primary experiment
at 100×3=300 images) — see `test03/results/manifest/degradation_parameters.csv`.

**Haze**: standard atmospheric scattering model,
`I_haze(x) = I_clean(x)*t(x) + A*(1-t(x))`, `t(x) = exp(-beta*d(x))`.
No real depth map exists for these images, and the task explicitly
forbids inventing physically meaningful depth — so `d(x)` is a **documented
synthetic proxy**, not a depth estimate: a linear vertical gradient,
normalized to `[0.3, 1.0]`, with row 0 (image top) = 1.0 ("far") and the
bottom row = 0.3 ("near"). This is a standard simplification for synthetic
haze (distant/sky regions are usually near the top of a natural photo) but
is explicitly *not* claimed to be accurate per-pixel depth. `A = 0.85`
(atmospheric light, fixed), `beta = 1.2` (extinction coefficient, fixed,
chosen to produce moderate, clearly visible but non-destructive haze on
this image set — verified visually in Phase 4). Fully deterministic (no
randomness beyond the fixed gradient), same for every image.

**Rain**: synthetic rain-streak layer, deterministic per image. Procedure:
(1) a sparse random point mask is drawn from `RandomState(seed)`, density
0.06% of pixels; (2) each point is rendered as a short line segment via
`cv2.line`, fixed length 18px, fixed angle 70° from horizontal (a
typical rain-streak angle), fixed stroke width 1px; (3) the streak layer is
Gaussian-blurred (`sigma=0.6`) to soften edges; (4) scaled by intensity
`0.55` and added to the clean image, then clipped to `[0,255]`. All
parameters fixed across images; only the RNG seed (hence streak
*positions*) varies per scene, ensuring every image gets a different but
reproducible rain pattern rather than a systematically identical overlay
that could itself become a confound.

Exact parameter values, and the per-image seed, are recorded for every one
of the 300 synthesized images in
`test03/results/manifest/degradation_parameters.csv` (Phase 2 requirement).

## 5. Evaluation design

- 100 scenes × 3 degradations = 300 images, extracted with the same 41
  pooled features as TEST02 (14 pipeline stages + 3×10 AFLB sub-features
  minus overlap in naming — see TEST02's `Feature_Trajectory` sheet for
  the canonical stage list, reused identically here for direct
  comparability in Phase 18).
- **Primary probe**: Logistic Regression + Linear SVM, 5-fold
  **GroupKFold** (group = `scene_id`) — every scene's three degraded
  versions are always in the same fold, together, eliminating any
  possibility of a fold "cheating" by having seen a same-scene example of
  another class.
- **Primary new evidence**: for every scene and every feature, compute
  `distance(rain_i, haze_i)`, `distance(rain_i, noise_i)`,
  `distance(haze_i, noise_i)` (same-scene, cross-degradation) and
  `distance(rain_i, rain_j)` for `i != j` (cross-scene, same-degradation),
  then `degradation_ratio = mean(D_degradation) / mean(D_scene)`. This is
  the direct test of the key scientific question: does changing *only* the
  degradation move the representation more than changing the *scene* does?
