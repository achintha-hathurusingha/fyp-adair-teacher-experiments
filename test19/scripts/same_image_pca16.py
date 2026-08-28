"""Does PCA-16(latent_pre) separate the SAME image's different-degradation
variants, or does it cluster by image content instead?

Reuses TEST05's own already-extracted feature cache (100 scenes x
Rain/Haze/Noise = 300 rows, latent_pre GAP+GMP), matching TEST05/05.5's own
768-dim = concat(GAP,GMP) convention exactly. No teacher re-run needed --
this data already exists.

Two checks:
  1. Direct distance comparison: for each point, is its SAME-SCENE
     different-degradation counterpart closer or farther than a typical
     SAME-DEGRADATION different-scene point? If degradation genuinely
     dominates the compact code, same-scene/different-degradation pairs
     should be FAR apart (large distance) -- the opposite of what "the code
     mostly encodes image content" would predict.
  2. Leave-scene-out classification: GroupKFold by scene_id (leakage-safe,
     matching TEST05.5's own standard -- a plain random split would leak
     scene identity across train/test, since each scene contributes one
     point per degradation). If PCA-16 correctly separates by degradation
     even when a scene's OTHER degradation variants are held out of
     training entirely, that is strong, non-trivial evidence for real
     separation, not just "the classifier memorized which scene is which."
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

d = np.load('/home/minura/teacher-experiments/test05/results/feature_analysis/latent_pre.npz',
            allow_pickle=True)
X_full = np.concatenate([d['X_gap'], d['X_gmp']], axis=1)  # (300, 768)
deg = np.array(d['degradation'])
scene = np.array(d['scene_id'])
n = len(deg)
print(f"loaded {n} rows, {len(set(scene))} scenes, degs={sorted(set(deg))}")

scaler = StandardScaler()
Xs = scaler.fit_transform(X_full)
pca = PCA(n_components=16, random_state=0)
X16 = pca.fit_transform(Xs)
print(f"PCA-16 explains {pca.explained_variance_ratio_.sum()*100:.1f}% variance")

# ---- Check 1: same-scene vs same-degradation distances ----
D = cdist(X16, X16, metric='euclidean')
np.fill_diagonal(D, np.nan)

same_scene_dists = []   # same scene, different degradation (the key question)
same_deg_dists = []     # same degradation, different scene (reference)
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        if scene[i] == scene[j] and deg[i] != deg[j]:
            same_scene_dists.append(D[i, j])
        elif scene[i] != scene[j] and deg[i] == deg[j]:
            same_deg_dists.append(D[i, j])

same_scene_dists = np.array(same_scene_dists)  # has duplicates (i,j)+(j,i), fine for a mean
same_deg_dists = np.array(same_deg_dists)

print(f"\n--- Check 1: distances in PCA-16 space ---")
print(f"same scene, DIFFERENT degradation : mean={same_scene_dists.mean():.3f}  "
      f"median={np.median(same_scene_dists):.3f}  n={len(same_scene_dists)}")
print(f"different scene, SAME degradation : mean={same_deg_dists.mean():.3f}  "
      f"median={np.median(same_deg_dists):.3f}  n={len(same_deg_dists)}")
ratio = same_scene_dists.mean() / same_deg_dists.mean()
print(f"ratio (same-scene-diff-deg / diff-scene-same-deg) = {ratio:.2f}")
print("  > 1  => degradation dominates (same image's variants ARE separated by degradation)")
print("  < 1  => content dominates (same image's variants stay clustered together)")

# Per-scene: is each point closer to its OWN scene's other-degradation twin,
# or to the nearest same-degradation point from a different scene?
closer_to_own_scene = 0
for i in range(n):
    own_scene_mask = (scene == scene[i]) & (deg != deg[i])
    other_deg_mask = (scene != scene[i]) & (deg == deg[i])
    d_own = D[i, own_scene_mask].min()
    d_other = D[i, other_deg_mask].min()
    if d_own < d_other:
        closer_to_own_scene += 1
print(f"\npoints whose NEAREST same-scene(diff-deg) neighbor is closer than their "
      f"nearest same-degradation(diff-scene) neighbor: {closer_to_own_scene}/{n} "
      f"({100*closer_to_own_scene/n:.1f}%)")
print("  low %  => degradation separation dominates over content (expected if e_D is a good signal)")
print("  high % => content/scene identity dominates over degradation")

# ---- Check 2: leave-scene-out degradation classification ----
print(f"\n--- Check 2: leave-scene-out (GroupKFold) classification accuracy ---")
gkf = GroupKFold(n_splits=5)
accs = []
for train_idx, test_idx in gkf.split(X16, deg, groups=scene):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X16[train_idx], deg[train_idx])
    pred = clf.predict(X16[test_idx])
    accs.append(accuracy_score(deg[test_idx], pred))
print(f"5-fold leave-scene-out accuracy: {np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%  "
      f"(chance = {100/3:.1f}%)")
print("Scenes in test folds NEVER appear in training (any degradation) -- this rules out "
      "the classifier just memorizing scene identity.")
