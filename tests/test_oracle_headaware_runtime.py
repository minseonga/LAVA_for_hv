import csv
from pathlib import Path

import torch

from llava.model.ais_gating import AISGatingConfig, AISGatingRuntime


def _write_role_csv(path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=["id", "candidate_patch_idx", "candidate_rank", "role_label"],
        )
        wr.writeheader()
        wr.writerow({"id": "42", "candidate_patch_idx": 1, "candidate_rank": 0, "role_label": "supportive"})
        wr.writerow({"id": "42", "candidate_patch_idx": 2, "candidate_rank": 0, "role_label": "harmful"})


def _make_inputs():
    # 8 key columns; image tokens are columns [2,3,4,5].
    image_mask = torch.zeros(1, 8, dtype=torch.bool)
    image_mask[:, 2:6] = True
    logits = torch.randn(1, 2, 1, 8)
    return image_mask, logits


def test_oracle_bias_only_late_and_image_cols(tmp_path: Path):
    role_csv = tmp_path / "roles.csv"
    _write_role_csv(role_csv)

    cfg = AISGatingConfig(
        enable_ais_gating=True,
        ais_use_oracle_roles=True,
        ais_oracle_role_csv=str(role_csv),
        ais_arm="bipolar",
        ais_oracle_lambda_pos=0.3,
        ais_oracle_lambda_neg=0.4,
        ais_late_start=1,
        ais_late_end=3,
    )
    rt = AISGatingRuntime(cfg)
    rt.configure(ais_harmful_heads="1:0", ais_faithful_heads="1:1")

    image_mask, logits = _make_inputs()
    rt.begin_generation(image_mask, sample_ids=["42"])

    # Early layer untouched.
    assert rt.compute_bias(layer_idx=0, attn_logits_masked=logits, attention_mask=None) is None

    # Late layer gets oracle bias.
    bias = rt.compute_bias(layer_idx=1, attn_logits_masked=logits, attention_mask=None)
    assert bias is not None

    # Text columns unchanged.
    assert (bias[:, :, :, ~image_mask[0]] == 0).all()

    # Image columns: harmful head0 on assertive patch idx2 -> seq col4 (positive penalty).
    # Faithful head1 on supportive patch idx1 -> seq col3 (negative penalty / boost).
    assert float(bias[0, 0, 0, 4].item()) > 0.0
    assert float(bias[0, 1, 0, 3].item()) < 0.0


def test_oracle_zero_lambda_is_noop(tmp_path: Path):
    role_csv = tmp_path / "roles.csv"
    _write_role_csv(role_csv)

    cfg = AISGatingConfig(
        enable_ais_gating=True,
        ais_use_oracle_roles=True,
        ais_oracle_role_csv=str(role_csv),
        ais_arm="bipolar",
        ais_oracle_lambda_pos=0.0,
        ais_oracle_lambda_neg=0.0,
        ais_late_start=1,
        ais_late_end=3,
    )
    rt = AISGatingRuntime(cfg)
    rt.configure(ais_harmful_heads="1:0", ais_faithful_heads="1:1")
    image_mask, logits = _make_inputs()
    rt.begin_generation(image_mask, sample_ids=["42"])
    bias = rt.compute_bias(layer_idx=1, attn_logits_masked=logits, attention_mask=None)
    assert bias is not None
    assert torch.allclose(bias, torch.zeros_like(bias))


def test_oracle_not_disabled_by_gamma_zero(tmp_path: Path):
    role_csv = tmp_path / "roles.csv"
    _write_role_csv(role_csv)

    cfg = AISGatingConfig(
        enable_ais_gating=True,
        ais_gamma=0.0,  # oracle path should still stay active.
        ais_use_oracle_roles=True,
        ais_oracle_role_csv=str(role_csv),
    )
    rt = AISGatingRuntime(cfg)
    image_mask, _ = _make_inputs()
    rt.begin_generation(image_mask, sample_ids=["42"])
    assert rt.active is True

