# R2R Exact Recipe (Phase 0)

Sources: (a) the CVPR 2026 paper PDF at `C:\Users\User\Documents\FYP\R@R\Wang_..._paper (1).pdf`
(cited as "paper, p.N"), (b) the official code clone at `~/FYP/R2R` on `devon`
(`github.com/cscxwang/R2R`, cited as `file:line`). Every row has a source. Anything
not directly observed is marked UNKNOWN — nothing here is inferred or defaulted.

## Architecture

| Item | Value | Source |
|---|---|---|
| Backbone | U-shaped NAFBlock encoder-decoder | paper §3, Fig.2(a) |
| Encoder blocks | `[1,1,1,28]` (4 levels) | paper p.5 "Implementation Details"; `net/model_3D.py:372` default `enc_blk_nums=[1,1,1,28]` |
| Decoder blocks | `[1,1,1,1]` | paper p.5; `net/model_3D.py:372` default `dec_blk_nums=[1,1,1,1]` |
| Width (base channels) | 32 | `net/model_3D.py:372` default `width=32` |
| Degradation amalgamator backbone | 4-level NAF-style encoder, `[2,2,4,8]` blocks | paper p.4 §3.1; `net/model_3D.py:378` `NAFMemEncoder(..., enc_blk_nums=[2,2,4,8], ...)` |
| Key dim `C_k` | 64 (= width×2) | paper p.5 "Ck and Cv are set to 64 and 512"; `net/model_3D.py:375` `key_dim = width * 2` |
| Value dim `C_v` | 512 (= width×16) | same as above; `net/model_3D.py:376` `value_dim = width * 16` |
| Bank capacity per task `M` | 64 (best in ablation; range tested 24/32/64/128) | paper Table 5, p.8; `net/model_3D.py:395` `DegradationMemory(opt, T_max=64, ...)` |
| Input/output channels | 3 (RGB) | `net/model_3D.py:381-384` `img_channel=3` on `intro`/`ending` convs |
| Retrieval mechanism | Cosine-normalized similarity, local-mean masked softmax over task windows, gated-conv fusion | paper §3.2 Eq.1-3; `net/feature_bank_3D.py:252` `comprehensive_attention_processing` |
| Normalization inside NAFBlock | UNKNOWN — not independently verified; NAFBlock is imported, its definition file was not read line-by-line in this pass | — |
| Activation (SimpleGate etc.) | UNKNOWN — same caveat as above | — |
| Skip connections | Present (encoder features `encs` passed into decoder) | `net/model_3D.py`: `x, encs = self.encoder(x)` ... `x = self.decoder(x, encs, read_out)`; visually confirmed in paper Fig.2(a) |
| Params (paper-reported) | 19.7M | paper Table 4, p.7 |
| MACs (paper-reported, 224×224 input) | 12G | paper Table 4, p.7 |
| Peak memory (paper-reported) | 846M | paper Table 4, p.7 |

## Loss

| Item | Value | Source |
|---|---|---|
| Total loss | `L = L_pixel + λd·L_deg + λm·L_match + λf·L_fft` | paper Eq.5, p.4 |
| `L_pixel` | `‖x̂ − x‖₁` | paper Eq. (unnumbered, p.4, "L_pixel defined as...") |
| `L_fft` | `(1/P)‖F(x̂) − F(x)‖₁`, dual-domain L1 (spatial+FFT) | paper Eq.4, p.4 |
| `L_deg` | cross-entropy on degradation amalgamator's auxiliary classification head | paper p.4 §3.3 |
| `L_match` | cross-entropy on matching-stage task assignment | paper p.4 §3.3 |
| `λd, λm, λf` | 0.1, 0.1, 0.125 | paper p.4, "empirically set to 0.1, 0.1, and 0.125" |

## Optimizer / Schedule

| Item | Value | Source |
|---|---|---|
| Optimizer | Adam, β1=0.9, β2=0.999 | paper p.5 "Implementation Details" |
| Initial LR | 2×10⁻⁴ | paper p.5 |
| Training length | 240 epochs (pretrain) | paper p.5 |
| Finetune length | 30 epochs at LR=1×10⁻⁶ | paper p.5 |
| Two-stage protocol | `pretrain` (from scratch) → `finetune` (loads pretrain's final `last.ckpt` + final prompt bank `last`, continues training; finetune does NOT update the prompt bank — reused as fixed memory) | README.md ("Training" section, `~/FYP/R2R/README.md`) |
| Patch size (pretrain) | 128×128 | paper p.5 |
| Patch size (finetune) | 224×224 | paper p.5 |
| Batch size | 64 | paper p.5 |
| Augmentation | random horizontal + vertical flips | paper p.5 |
| Hardware used by paper authors | single NVIDIA RTX 5090, <40 GPU-hours total | paper p.5 |
| Random seed (training) | UNKNOWN — not set/documented in the public train scripts inspected | — |
| Random seed (testing) | 0 (both `np.random.seed(0)` and `torch.manual_seed(0)`) | `test_3D.py` main block, confirmed by direct read |

## Data

| Task | Train source | Test source | Degradation synthesis | Source |
|---|---|---|---|---|
| Denoise | BSD400 + WED, combined | BSD68 (`cbsd68/`) | AWGN added on the fly at σ ∈ {15,25,50}, `noise = randn(shape); clip(clean + noise*σ, 0, 255)` | paper p.5-6 "Datasets"; `utils/dataset_utils.py:533-536` `_add_gaussian_noise` |
| Derain | Rain100L pairs | Rain100L | none (paired dataset) | paper p.6 |
| Dehaze | RESIDE synthetic (`synthetic/part1,2...`) + real (`original/`) | SOTS (outdoor) | none (paired dataset) | paper p.6; README "Data Preparation" |
| Deblur (5D only) | GoPro | GoPro | none (paired) | README |
| Low-light (5D only) | LOL-v1 | LOL-v1 | none (paired) | README |

## Evaluation

| Item | Value | Source |
|---|---|---|
| PSNR | `skimage.metrics.peak_signal_noise_ratio(clean, restored, data_range=1)` | `utils/val_utils.py:52-66`, direct read of cloned repo |
| SSIM | `skimage.metrics.structural_similarity(clean, restored, data_range=1, channel_axis=-1)` | same file, same function |
| Color space | RGB, full 3-channel — no Y/luma-only conversion | `utils/dataset_utils.py:542` `Image.open(...).convert('RGB')`; metric operates on the RGB tensor directly |
| Image range | float `[0,1]` (via `ToTensor()`), explicitly `np.clip(..., 0, 1)` before metric call | `utils/val_utils.py:54-55` |
| Border/shave policy | none — full image, no cropping | confirmed by reading `compute_psnr_ssim`, no shave logic present |
| Averaging | per-image PSNR/SSIM computed then arithmetic-averaged across the test set (`AverageMeter`) | `test_3D.py` `test_Denoise`/`test_Derain_Dehaze` |
| Inference tiling | none for the standard benchmark eval — `patch_inference(..., tile=None, ...)`, i.e. full-image single forward pass | `test_3D.py:31` call sites in `test_Denoise`/`test_Derain_Dehaze` |
| Checkpoint used for eval | the single final `last.ckpt` + final prompt bank `last` from the **3D** (denoise+derain+dehaze) `finetune` run — NOT an "epoch 15/25/50" checkpoint | README "Saved Files"; `net/model_3D.py:490` `R2RTest` loads `ckpt_path+prompts_name+".ckpt"` |

## CRITICAL CORRECTION — "Epoch 15/25/50" is a misread

The TEST17-R brief's table headers ("Epoch 15", "Epoch 25", "Epoch 50") are incorrect.
**15, 25, 50 are Gaussian-noise standard-deviation (σ) levels for the denoising task,
not training epochs.** R2R trains a single model to convergence (240 pretrain epochs +
30 finetune epochs, per the table above) and then evaluates that one fixed checkpoint
three times, once at each of three noise severities (σ=15, σ=25, σ=50), on the same
68-image CBSD68 test set. There is no "epoch 15 checkpoint" to reproduce — reproducing
these three numbers means training **one** model for the full 270-epoch schedule, then
running inference at three noise levels, not training three separate partial runs.

This was independently verified empirically: the official released checkpoint
(`train_ckpt_3D_f/last.ckpt` + `save_prompts_3D/last`, a single checkpoint) was
evaluated once and produced all three numbers in one run:

```
Denoise sigma=15: psnr: 34.10, ssim: 0.9356
Denoise sigma=25: psnr: 31.45, ssim: 0.8931
Denoise sigma=50: psnr: 28.22, ssim: 0.8064
```

matching the paper's Table 1 to the hundredth of a dB. See `target_origin.md` for
full detail on how these numbers were produced.
