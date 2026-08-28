# TEST19 — does PCA-16(latent_pre) separate the SAME image by degradation type?

## Question

TEST05/05.5 established PCA-16 of `latent_pre` (768-dim = GAP+GMP over
384 channels) retains strong degradation-classification accuracy in
aggregate. This asks a sharper question: for the SAME underlying image,
degraded three different ways, does the compact code actually push those
three representations apart by degradation — or does it stay dominated by
image content, with degradation only a secondary signal?

## Data

Reused TEST05's own already-extracted feature cache directly — no teacher
re-run needed. `test05/results/feature_analysis/latent_pre.npz`: 100 real
scenes x Rain/Haze/Noise = 300 rows, each scene's three degradation variants
sharing the same underlying content (TEST03's own same-scene triplet set).

## Method

`test19/scripts/same_image_pca16.py`. StandardScaler + PCA-16 (fit on all
300 rows, matching TEST05's own compact_embedding.py convention: 76.8%
variance explained). Two checks:

1. **Distance comparison**: same-scene/different-degradation pairwise
   distances vs different-scene/same-degradation pairwise distances, in
   PCA-16 space.
2. **Leave-scene-out classification**: 5-fold GroupKFold by scene_id (a
   scene's other two degradation variants never appear in that fold's
   training set — rules out the classifier memorizing scene identity
   rather than learning degradation structure), kNN(k=5) predicting
   degradation type.

## Results

| check | result |
|---|---:|
| same-scene, different-degradation mean distance | 34.469 |
| different-scene, same-degradation mean distance | 19.928 |
| ratio | **1.73x** |
| points closer to same-scene twin than nearest same-degradation point | 19/300 (6.3%) |
| leave-scene-out classification accuracy | **99.0% +/- 1.3%** (chance 33.3%) |

## Conclusion

PCA-16(latent_pre) genuinely separates by degradation type, not by image
content. The same photo, degraded three different ways, lands 1.73x
farther apart in the compact code than two different photos sharing the
same degradation do. 93.7% of points have a closer same-degradation
neighbor than their own same-scene counterpart. And the separation
generalizes to scenes never seen with any of their degradation variants
during training (99.0% leave-scene-out accuracy) -- this is not the
classifier learning per-scene shortcuts.

**Consequence for kd_feature**: direct, positive supporting evidence that
`e_D` (the compact `latent_pre` projection being distilled into the
student's `middle_blks`) carries robust, generalizable degradation
information rather than incidental image-specific structure. Complements
TEST05.5's own audit, which validated the `z_T -> e_D` pathway causally;
this shows the resulting code is also cleanly separated by the thing it is
meant to represent.
