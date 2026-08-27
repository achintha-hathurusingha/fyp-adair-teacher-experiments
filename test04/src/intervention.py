"""TEST04 core mechanism: a manual, faithful replica of AdaIR.forward()
(net/model.py lines 426-475, verified against source -- see
report/forward_graph_audit.md) that accepts an `overrides` dict to
substitute any intermediate tensor at its exact production point, then
continues the SAME recipient computation (recipient's inp_img, recipient's
skip connections unless also overridden, recipient's weights) from there.

This does NOT modify net/model.py or the checkpoint -- it calls the
already-loaded model's submodules directly, in the same order and with the
same arguments the original forward() uses. Verified bit-identical to
model(img) with no overrides (see verify_manual_forward_matches_model).

Intervention points (override dict keys):
  'enc1', 'enc2', 'enc3'   -- encoder skip-connection tensors
  'latent_pre'              -- encoder bottleneck, BEFORE AFLB1
  'aflb1_out'                -- AFLB1 (fre1) output
  'aflb2_out'                -- AFLB2 (fre2) output
  'aflb3_out'                -- AFLB3 (fre3) output
"""
from __future__ import annotations

import torch


INTERMEDIATE_KEYS = ["enc1", "enc2", "enc3", "latent_pre", "aflb1_out", "aflb2_out", "aflb3_out"]


def manual_forward(model, inp_img: torch.Tensor, overrides: dict[str, torch.Tensor] | None = None) -> dict:
    """Faithful manual replay of AdaIR.forward() with optional tensor
    substitution at any of INTERMEDIATE_KEYS. Returns a dict of every
    intermediate tensor plus the final 'output'.
    """
    net = model.net if hasattr(model, "net") else model
    overrides = overrides or {}

    for key, t in overrides.items():
        assert key in INTERMEDIATE_KEYS, f"unknown override key {key!r}"

    def sub(key, computed):
        if key in overrides:
            donor = overrides[key]
            assert donor.shape == computed.shape, (
                f"override {key!r} shape mismatch: donor {tuple(donor.shape)} "
                f"vs recipient {tuple(computed.shape)}")
            return donor.to(computed.device, computed.dtype)
        return computed

    inp_enc_level1 = net.patch_embed(inp_img)
    out_enc_level1 = sub("enc1", net.encoder_level1(inp_enc_level1))

    inp_enc_level2 = net.down1_2(out_enc_level1)
    out_enc_level2 = sub("enc2", net.encoder_level2(inp_enc_level2))

    inp_enc_level3 = net.down2_3(out_enc_level2)
    out_enc_level3 = sub("enc3", net.encoder_level3(inp_enc_level3))

    inp_enc_level4 = net.down3_4(out_enc_level3)
    latent_pre = sub("latent_pre", net.latent(inp_enc_level4))

    aflb1_out = net.fre1(inp_img, latent_pre) if net.decoder else latent_pre
    aflb1_out = sub("aflb1_out", aflb1_out)

    inp_dec_level3 = net.up4_3(aflb1_out)
    inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
    inp_dec_level3 = net.reduce_chan_level3(inp_dec_level3)
    out_dec_level3 = net.decoder_level3(inp_dec_level3)

    aflb2_out = net.fre2(inp_img, out_dec_level3) if net.decoder else out_dec_level3
    aflb2_out = sub("aflb2_out", aflb2_out)

    inp_dec_level2 = net.up3_2(aflb2_out)
    inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
    inp_dec_level2 = net.reduce_chan_level2(inp_dec_level2)
    out_dec_level2 = net.decoder_level2(inp_dec_level2)

    aflb3_out = net.fre3(inp_img, out_dec_level2) if net.decoder else out_dec_level2
    aflb3_out = sub("aflb3_out", aflb3_out)

    inp_dec_level1 = net.up2_1(aflb3_out)
    inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
    out_dec_level1 = net.decoder_level1(inp_dec_level1)

    refine = net.refinement(out_dec_level1)
    output = net.output(refine) + inp_img

    return {
        "enc1": out_enc_level1, "enc2": out_enc_level2, "enc3": out_enc_level3,
        "latent_pre": latent_pre, "aflb1_out": aflb1_out, "dec3": out_dec_level3,
        "aflb2_out": aflb2_out, "dec2": out_dec_level2, "aflb3_out": aflb3_out,
        "dec1": out_dec_level1, "refine": refine, "output": output,
    }


@torch.no_grad()
def verify_manual_forward_matches_model(model, inp_img: torch.Tensor) -> dict:
    """Sanity check (part of Phase 4/16): manual_forward with NO overrides
    must be bit-identical (or extremely close, allowing for op-order fp
    nondeterminism) to calling the model directly."""
    ref = model(inp_img)
    man = manual_forward(model, inp_img)["output"]
    diff = (ref - man).abs()
    return {
        "max_abs_diff": diff.max().item(), "mean_abs_diff": diff.mean().item(),
        "matches": bool(torch.allclose(ref, man, atol=1e-5)),
    }


def sanity_check(t: torch.Tensor) -> dict:
    """Phase 16 per-intervention sanity check."""
    x = t.detach().float()
    return {
        "nan_count": int(torch.isnan(x).sum().item()), "inf_count": int(torch.isinf(x).sum().item()),
        "min": x.min().item() if x.numel() else float("nan"), "max": x.max().item() if x.numel() else float("nan"),
        "mean": x.mean().item() if x.numel() else float("nan"), "std": x.std(unbiased=False).item() if x.numel() else float("nan"),
        "all_finite": bool(torch.isfinite(x).all().item()),
    }
