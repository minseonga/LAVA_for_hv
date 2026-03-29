import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llava.model.ais_gating import (
    AISGatingConfig,
    AISGatingRuntime,
    apply_ais_column_penalty,
    install_ais_gating_hooks,
)


class TestAISGatingSanity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _tiny_llama(self):
        cfg = LlamaConfig(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
        model = LlamaForCausalLM(cfg).eval()
        return model

    def test_disabled_equals_baseline(self):
        model = self._tiny_llama()
        input_ids = torch.randint(low=0, high=96, size=(2, 7), dtype=torch.long)
        attn = torch.ones_like(input_ids)

        with torch.no_grad():
            base = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits

        rt = AISGatingRuntime(AISGatingConfig(enable_ais_gating=False, ais_gamma=1.0))
        install_ais_gating_hooks(model, runtime=rt)
        rt.begin_generation(torch.zeros((2, 7), dtype=torch.bool))
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        rt.end_generation()

        self.assertTrue(torch.equal(base, out), "Disabled AIS should be exactly baseline.")

    def test_gamma_zero_equals_baseline(self):
        model = self._tiny_llama()
        input_ids = torch.randint(low=0, high=96, size=(2, 7), dtype=torch.long)
        attn = torch.ones_like(input_ids)

        with torch.no_grad():
            base = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits

        rt = AISGatingRuntime(AISGatingConfig(enable_ais_gating=True, ais_gamma=0.0))
        install_ais_gating_hooks(model, runtime=rt)
        rt.begin_generation(torch.zeros((2, 7), dtype=torch.bool))
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        rt.end_generation()

        self.assertTrue(torch.equal(base, out), "Gamma=0 should be exactly baseline.")

    def test_only_image_columns_changed(self):
        logits = torch.randn(1, 2, 3, 6)
        image_mask = torch.tensor([[False, True, False, True, False, False]])
        penalty = torch.zeros(1, 2, 6)
        penalty[:, :, 1] = 0.3
        penalty[:, :, 3] = 0.7
        out = apply_ais_column_penalty(logits, image_mask=image_mask, penalty_last_query=penalty)

        # Non-last query rows unchanged.
        self.assertTrue(torch.equal(logits[:, :, :-1, :], out[:, :, :-1, :]))
        # Last query text columns unchanged.
        text_cols = torch.tensor([0, 2, 4, 5], dtype=torch.long)
        self.assertTrue(torch.equal(logits[:, :, -1, text_cols], out[:, :, -1, text_cols]))
        # Last query image columns changed.
        self.assertFalse(torch.equal(logits[:, :, -1, 1], out[:, :, -1, 1]))
        self.assertFalse(torch.equal(logits[:, :, -1, 3], out[:, :, -1, 3]))

    def test_only_late_layers_apply_bias(self):
        cfg = AISGatingConfig(
            enable_ais_gating=True,
            ais_early_start=0,
            ais_early_end=0,
            ais_late_start=2,
            ais_late_end=2,
            ais_topk=1,
            ais_tau=-1.0,  # always trigger if any image token exists
            ais_gamma=1.0,
            ais_eps=1e-6,
            ais_debug_log=True,
        )
        rt = AISGatingRuntime(cfg)
        rt.begin_generation(torch.tensor([[True, True, False, False]], dtype=torch.bool))

        # Early layer support.
        early_logits = torch.tensor(
            [
                [
                    [[6.0, 1.0, -2.0, -2.0]],
                    [[6.0, 1.0, -2.0, -2.0]],
                ]
            ],
            dtype=torch.float32,
        )  # [B=1,H=2,Q=1,K=4]
        b0 = rt.compute_bias(layer_idx=0, attn_logits_masked=early_logits, attention_mask=None)
        self.assertTrue(b0 is None or torch.allclose(b0, torch.zeros_like(early_logits)))

        # Mid layer (not late) should have no bias.
        mid_logits = torch.tensor(
            [
                [
                    [[2.0, 2.0, -2.0, -2.0]],
                    [[2.0, 2.0, -2.0, -2.0]],
                ]
            ],
            dtype=torch.float32,
        )
        b1 = rt.compute_bias(layer_idx=1, attn_logits_masked=mid_logits, attention_mask=None)
        self.assertIsNone(b1)

        # Late layer should return non-zero bias on image columns only.
        late_logits = torch.tensor(
            [
                [
                    [[1.0, 6.0, -2.0, -2.0]],
                    [[1.0, 3.0, -2.0, -2.0]],
                ]
            ],
            dtype=torch.float32,
        )
        b2 = rt.compute_bias(layer_idx=2, attn_logits_masked=late_logits, attention_mask=None)
        self.assertIsNotNone(b2)
        self.assertGreater(float(torch.abs(b2[:, :, -1, :2]).sum().item()), 0.0)
        self.assertEqual(float(torch.abs(b2[:, :, -1, 2:]).sum().item()), 0.0)

        # Debug rows should exist for late layer.
        rows = rt.get_debug_rows(reset=False)
        self.assertGreaterEqual(len(rows), 1)
        rt.end_generation()

    def test_headset_arm_masks_work(self):
        # Harmful-only: only configured harmful head receives suppression.
        cfg_harm = AISGatingConfig(
            enable_ais_gating=True,
            ais_early_start=0,
            ais_early_end=0,
            ais_late_start=1,
            ais_late_end=1,
            ais_topk=1,
            ais_tau=-1.0,
            ais_gamma=1.0,
            ais_arm="harmful_only",
            ais_harmful_heads="1:0",
            ais_use_dynamic_omega=False,
        )
        rt_harm = AISGatingRuntime(cfg_harm)
        rt_harm.begin_generation(torch.tensor([[True, True, False, False]], dtype=torch.bool))

        early_logits = torch.tensor(
            [[[[6.0, 1.0, -2.0, -2.0]], [[6.0, 1.0, -2.0, -2.0]]]],
            dtype=torch.float32,
        )
        _ = rt_harm.compute_bias(layer_idx=0, attn_logits_masked=early_logits, attention_mask=None)

        late_logits = torch.tensor(
            [[[[1.0, 6.0, -2.0, -2.0]], [[1.0, 6.0, -2.0, -2.0]]]],
            dtype=torch.float32,
        )
        b_harm = rt_harm.compute_bias(layer_idx=1, attn_logits_masked=late_logits, attention_mask=None)
        self.assertIsNotNone(b_harm)
        self.assertGreater(float(torch.abs(b_harm[:, 0, -1, :2]).sum().item()), 0.0)
        self.assertEqual(float(torch.abs(b_harm[:, 1, -1, :2]).sum().item()), 0.0)
        rt_harm.end_generation()

        # Faithful-only: configured faithful head receives negative bias (boost after subtraction).
        cfg_faith = AISGatingConfig(
            enable_ais_gating=True,
            ais_early_start=0,
            ais_early_end=0,
            ais_late_start=1,
            ais_late_end=1,
            ais_topk=1,
            ais_tau=-1.0,
            ais_gamma=1.0,
            ais_arm="faithful_only",
            ais_faithful_heads="1:1",
            ais_faithful_boost=1.0,
        )
        rt_faith = AISGatingRuntime(cfg_faith)
        rt_faith.begin_generation(torch.tensor([[True, True, False, False]], dtype=torch.bool))
        _ = rt_faith.compute_bias(layer_idx=0, attn_logits_masked=early_logits, attention_mask=None)

        late_logits_2 = torch.tensor(
            [[[[1.0, 6.0, -2.0, -2.0]], [[1.0, 6.0, -2.0, -2.0]]]],
            dtype=torch.float32,
        )
        b_faith = rt_faith.compute_bias(layer_idx=1, attn_logits_masked=late_logits_2, attention_mask=None)
        self.assertIsNotNone(b_faith)
        self.assertLess(float(b_faith[:, 1, -1, :2].min().item()), 0.0)
        self.assertEqual(float(torch.abs(b_faith[:, 0, -1, :2]).sum().item()), 0.0)
        rt_faith.end_generation()

    def test_budget_mode_fixed_mass(self):
        cfg = AISGatingConfig(
            enable_ais_gating=True,
            ais_early_start=0,
            ais_early_end=0,
            ais_late_start=1,
            ais_late_end=1,
            ais_topk=1,
            ais_tau=999.0,  # ignored in budget mode
            ais_gamma=1.0,
            ais_arm="harmful_only",
            ais_use_budget_routing=True,
            ais_budget_total=0.8,
            ais_harmful_top_ratio=0.5,
            ais_budget_patch_topk=1,
        )
        rt = AISGatingRuntime(cfg)
        rt.begin_generation(torch.tensor([[True, True, False, False]], dtype=torch.bool))

        early_logits = torch.tensor(
            [[[[6.0, 1.0, -2.0, -2.0]], [[6.0, 1.0, -2.0, -2.0]]]],
            dtype=torch.float32,
        )
        _ = rt.compute_bias(layer_idx=0, attn_logits_masked=early_logits, attention_mask=None)

        late_logits = torch.tensor(
            [[[[1.0, 6.0, -2.0, -2.0]], [[2.0, 5.0, -2.0, -2.0]]]],
            dtype=torch.float32,
        )
        b = rt.compute_bias(layer_idx=1, attn_logits_masked=late_logits, attention_mask=None)
        self.assertIsNotNone(b)

        # Image columns only.
        self.assertEqual(float(torch.abs(b[:, :, -1, 2:]).sum().item()), 0.0)

        # Sparse fixed-dose routing: each selected (head, patch) gets fixed dose budget_layer.
        img = b[:, :, -1, :2]
        nz = img[img > 0]
        self.assertGreaterEqual(int(nz.numel()), 1)
        self.assertAlmostEqual(float(nz.max().item()), 0.8, places=4)
        rt.end_generation()


if __name__ == "__main__":
    unittest.main()
