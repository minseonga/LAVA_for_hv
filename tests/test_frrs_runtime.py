import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llava.model.ais_gating import AISGatingConfig, AISGatingRuntime, install_ais_gating_hooks
from llava.model.frrs import FRRS


class TestFRRSModule(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def _dummy_feats(self, bsz: int, k_img: int):
        x = torch.rand(bsz, k_img)
        return {
            "A": x.clone(),
            "C": x.clone(),
            "E": x.clone(),
            "D": x.clone(),
        }

    def test_alpha_beta_zero_is_noop(self):
        m = FRRS(alpha=0.0, beta=0.0, head_mode="static")
        out = torch.randn(2, 4, 2, 8)
        v = torch.randn(2, 4, 8, 8)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[:, [1, 3, 5]] = True
        feats = self._dummy_feats(2, 3)
        y, dbg = m(
            attn_output=out,
            value_states=v,
            image_mask=mask,
            feats=feats,
            faithful_head_mask=torch.tensor([1, 0, 1, 0], dtype=torch.float32),
            harmful_head_mask=torch.tensor([0, 1, 0, 1], dtype=torch.float32),
        )
        self.assertTrue(torch.equal(y, out))
        self.assertEqual(float(dbg["frrs_delta_abs_mean"]), 0.0)

    def test_only_last_row_changed(self):
        m = FRRS(
            alpha=0.6,
            beta=0.0,
            arm="supportive",
            tau_c=10.0,
            tau_e=10.0,
            k_c=1.0,
            k_e=1.0,
            head_mode="static",
        )
        out = torch.randn(2, 4, 3, 8)
        v = torch.randn(2, 4, 8, 8)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[:, [1, 2, 6]] = True
        feats = self._dummy_feats(2, 3)
        y, _ = m(
            attn_output=out,
            value_states=v,
            image_mask=mask,
            feats=feats,
            faithful_head_mask=torch.tensor([1, 1, 0, 0], dtype=torch.float32),
        )
        self.assertTrue(torch.equal(y[:, :, :-1, :], out[:, :, :-1, :]))
        self.assertGreater(float((y[:, :, -1, :] - out[:, :, -1, :]).abs().sum().item()), 0.0)

    def test_head_masks_respected(self):
        m = FRRS(
            alpha=0.8,
            beta=0.0,
            arm="supportive",
            tau_c=10.0,
            tau_e=10.0,
            k_c=1.0,
            k_e=1.0,
            head_mode="static",
        )
        out = torch.randn(1, 4, 2, 8)
        v = torch.randn(1, 4, 8, 8)
        mask = torch.zeros(1, 8, dtype=torch.bool)
        mask[:, [1, 2, 6]] = True
        feats = self._dummy_feats(1, 3)
        y, _ = m(
            attn_output=out,
            value_states=v,
            image_mask=mask,
            feats=feats,
            faithful_head_mask=torch.tensor([0, 1, 0, 0], dtype=torch.float32),
        )
        delta = (y - out).abs().sum(dim=(-1, -2))[0]
        self.assertEqual(float(delta[0].item()), 0.0)
        self.assertGreater(float(delta[1].item()), 0.0)
        self.assertEqual(float(delta[2].item()), 0.0)
        self.assertEqual(float(delta[3].item()), 0.0)


class TestFRRSRuntime(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def _tiny_llama(self):
        cfg = LlamaConfig(
            vocab_size=101,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
        return LlamaForCausalLM(cfg).eval()

    def test_disabled_equals_baseline(self):
        model = self._tiny_llama()
        input_ids = torch.randint(low=0, high=100, size=(2, 6), dtype=torch.long)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            base = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits

        rt = AISGatingRuntime(AISGatingConfig(enable_frrs=False))
        install_ais_gating_hooks(model, runtime=rt)
        rt.begin_generation(torch.zeros((2, 6), dtype=torch.bool))
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        rt.end_generation()
        self.assertTrue(torch.equal(base, out))

    def test_zero_strength_equals_baseline(self):
        model = self._tiny_llama()
        input_ids = torch.randint(low=0, high=100, size=(2, 6), dtype=torch.long)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            base = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits

        cfg = AISGatingConfig(
            enable_frrs=True,
            frrs_late_start=0,
            frrs_late_end=3,
            frrs_alpha=0.0,
            frrs_beta=0.0,
            frrs_head_mode="static",
            ais_faithful_heads="0:0,1:0,2:0,3:0",
        )
        rt = AISGatingRuntime(cfg)
        install_ais_gating_hooks(model, runtime=rt)
        feats = {
            "A": torch.rand(2, 2),
            "C": torch.rand(2, 2),
            "E": torch.rand(2, 2),
            "D": torch.rand(2, 2),
        }
        mask = torch.zeros((2, 6), dtype=torch.bool)
        mask[:, [1, 2]] = True
        rt.begin_generation(mask, frrs_feats=feats)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        rt.end_generation()
        self.assertTrue(torch.equal(base, out))

    def test_apply_frrs_late_only_and_last_row(self):
        cfg = AISGatingConfig(
            enable_frrs=True,
            frrs_late_start=2,
            frrs_late_end=2,
            frrs_alpha=0.5,
            frrs_beta=0.0,
            frrs_tau_c=10.0,
            frrs_tau_e=10.0,
            frrs_k_c=1.0,
            frrs_k_e=1.0,
            frrs_arm="supportive",
            frrs_head_mode="static",
            ais_faithful_heads="2:1,2:3",
        )
        rt = AISGatingRuntime(cfg)
        mask = torch.zeros((2, 8), dtype=torch.bool)
        mask[0, [1, 3, 5]] = True
        mask[1, [0, 4, 7]] = True
        feats = {
            "A": torch.rand(2, 3),
            "C": torch.rand(2, 3),
            "E": torch.rand(2, 3),
            "D": torch.rand(2, 3),
        }
        rt.begin_generation(mask, frrs_feats=feats)

        out = torch.randn(2, 4, 2, 8)
        val = torch.randn(2, 4, 8, 8)
        y0 = rt.apply_frrs_output_steering(layer_idx=1, attn_output=out, value_states=val, attention_mask=None)
        self.assertTrue(torch.equal(y0, out))

        y1 = rt.apply_frrs_output_steering(layer_idx=2, attn_output=out, value_states=val, attention_mask=None)
        self.assertTrue(torch.equal(y1[:, :, :-1, :], out[:, :, :-1, :]))
        self.assertGreater(float((y1[:, :, -1, :] - out[:, :, -1, :]).abs().sum().item()), 0.0)
        rt.end_generation()


if __name__ == "__main__":
    unittest.main()
