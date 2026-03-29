import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llava.model.ais_gating import AISGatingConfig, AISGatingRuntime, install_ais_gating_hooks
from llava.model.rfhar import RFHAR


class _RFHARFixedRF(RFHAR):
    """Test helper: bypass RF formula and use feats['C'] directly."""

    def compute_rf(self, feats):
        rf = feats["C"].float()
        if rf.dim() == 1:
            rf = rf.unsqueeze(0)
        return rf


class TestRFHARModule(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _dummy_feats(self, bsz: int, k_img: int):
        x = torch.rand(bsz, k_img)
        return {"C": x.clone(), "A": x.clone(), "D": x.clone(), "B": x.clone()}

    def test_gamma_zero_is_noop(self):
        m = RFHAR(gamma=0.0, r_percent=0.2, lambda_penalty=0.5, eps=1e-6)
        logits = torch.randn(2, 4, 8)
        image_mask = torch.zeros(2, 8, dtype=torch.bool)
        image_mask[:, [1, 3, 5]] = True
        feats = self._dummy_feats(bsz=2, k_img=3)
        out, dbg = m(attn_logits_last=logits, image_mask=image_mask, feats=feats)
        self.assertTrue(torch.equal(out, logits))
        self.assertEqual(float(dbg["delta_abs_mean"]), 0.0)

    def test_only_image_columns_changed(self):
        m = RFHAR(gamma=0.5, r_percent=0.5, lambda_penalty=0.5, eps=1e-6)
        logits = torch.randn(2, 4, 7)
        image_mask = torch.zeros(2, 7, dtype=torch.bool)
        image_mask[0, [1, 4]] = True
        image_mask[1, [0, 6]] = True
        feats = self._dummy_feats(bsz=2, k_img=2)
        out, _ = m(attn_logits_last=logits, image_mask=image_mask, feats=feats)

        for b in range(2):
            text_cols = torch.nonzero(~image_mask[b], as_tuple=False).flatten()
            img_cols = torch.nonzero(image_mask[b], as_tuple=False).flatten()
            self.assertTrue(torch.equal(out[b, :, text_cols], logits[b, :, text_cols]))
            self.assertGreater(float((out[b, :, img_cols] - logits[b, :, img_cols]).abs().sum().item()), 0.0)

    def test_dynamic_head_sets_are_disjoint(self):
        m = RFHAR(gamma=0.5, r_percent=0.5, lambda_penalty=0.5, eps=1e-6)
        attn_img_norm = torch.rand(3, 6, 5)
        attn_img_norm = attn_img_norm / torch.clamp(attn_img_norm.sum(dim=-1, keepdim=True), min=1e-6)
        rf = torch.rand(3, 5)
        m_pos, m_neg, _ = m.dynamic_head_roles(attn_img_norm=attn_img_norm, rf=rf)
        overlap = ((m_pos > 0) & (m_neg > 0)).sum().item()
        self.assertEqual(int(overlap), 0)

    def test_image_only_renorm_drives_head_role(self):
        # Construct a case where full-sequence scores would pick head-1, but image-renorm picks head-0.
        m = _RFHARFixedRF(gamma=1.0, r_percent=0.5, lambda_penalty=0.5, eps=1e-6)
        logits = torch.tensor(
            [
                [
                    [4.0, 2.0, 0.0, 4.0],   # head 0: tiny image mass, mostly on image-1
                    [1.0, 2.0, 2.4, 1.0],   # head 1: large image mass, more on image-2
                ]
            ],
            dtype=torch.float32,
        )  # [B=1,H=2,K=4]
        image_mask = torch.tensor([[False, True, True, False]], dtype=torch.bool)
        feats = {
            "C": torch.tensor([[1.0, 0.0]], dtype=torch.float32),  # RF: image-1 high, image-2 low
            "A": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            "D": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            "B": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        }
        out, dbg = m(attn_logits_last=logits, image_mask=image_mask, feats=feats)
        delta = out - logits
        # On RF-high image token (col=1), head-0 should be boosted more than head-1.
        self.assertGreater(float(delta[0, 0, 1].item()), float(delta[0, 1, 1].item()))
        self.assertEqual(float(dbg["disjoint_ok"]), 1.0)

    def test_output_is_logits_not_softmax_probs(self):
        m = RFHAR(gamma=0.3, r_percent=0.5, lambda_penalty=0.5, eps=1e-6)
        logits = torch.randn(1, 2, 5)
        image_mask = torch.tensor([[True, True, False, False, False]], dtype=torch.bool)
        feats = self._dummy_feats(bsz=1, k_img=2)
        out, _ = m(attn_logits_last=logits, image_mask=image_mask, feats=feats)
        # A softmaxed tensor would sum to ~1 on last dim. Logits should not.
        self.assertFalse(torch.allclose(out.sum(dim=-1), torch.ones_like(out.sum(dim=-1)), atol=1e-4))


class TestRFHARRuntime(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)

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

        rt = AISGatingRuntime(AISGatingConfig(enable_rfhar=False))
        install_ais_gating_hooks(model, runtime=rt)
        rt.begin_generation(torch.zeros((2, 6), dtype=torch.bool))
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        rt.end_generation()
        self.assertTrue(torch.equal(base, out))

    def test_gamma_zero_equals_baseline(self):
        model = self._tiny_llama()
        input_ids = torch.randint(low=0, high=100, size=(2, 6), dtype=torch.long)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            base = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits

        rt = AISGatingRuntime(AISGatingConfig(enable_rfhar=True, rfhar_gamma=0.0))
        install_ais_gating_hooks(model, runtime=rt)
        feats = {
            "C": torch.ones(2, 2),
            "A": torch.ones(2, 2),
            "D": torch.zeros(2, 2),
            "B": torch.zeros(2, 2),
        }
        mask = torch.zeros((2, 6), dtype=torch.bool)
        mask[:, [1, 2]] = True
        rt.begin_generation(mask, rfhar_feats=feats)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        rt.end_generation()
        self.assertTrue(torch.equal(base, out))

    def test_only_last_query_row_changed_and_batch_aware(self):
        cfg = AISGatingConfig(
            enable_rfhar=True,
            rfhar_late_start=2,
            rfhar_late_end=2,
            rfhar_gamma=0.4,
            rfhar_r_percent=0.5,
            rfhar_lambda_penalty=0.5,
            rfhar_eps=1e-6,
        )
        rt = AISGatingRuntime(cfg)
        mask = torch.zeros((2, 8), dtype=torch.bool)
        mask[0, [1, 3, 5]] = True
        mask[1, [0, 4, 7]] = True
        feats = {
            "C": torch.rand(2, 3),
            "A": torch.rand(2, 3),
            "D": torch.rand(2, 3),
            "B": torch.rand(2, 3),
        }
        rt.begin_generation(mask, rfhar_feats=feats)

        logits = torch.randn(2, 4, 2, 8)
        # Non-late layer -> no bias.
        b0 = rt.compute_bias(layer_idx=1, attn_logits_masked=logits, attention_mask=None)
        self.assertIsNone(b0)

        b1 = rt.compute_bias(layer_idx=2, attn_logits_masked=logits, attention_mask=None)
        self.assertIsNotNone(b1)
        self.assertTrue(torch.equal(b1[:, :, :-1, :], torch.zeros_like(b1[:, :, :-1, :])))
        for b in range(2):
            text_cols = torch.nonzero(~mask[b], as_tuple=False).flatten()
            self.assertTrue(torch.equal(b1[b, :, -1, text_cols], torch.zeros_like(b1[b, :, -1, text_cols])))
        rt.end_generation()

    def test_alignment_mismatch_disables_bias(self):
        cfg = AISGatingConfig(
            enable_rfhar=True,
            rfhar_late_start=0,
            rfhar_late_end=0,
            rfhar_gamma=0.3,
        )
        rt = AISGatingRuntime(cfg)
        mask = torch.zeros((1, 6), dtype=torch.bool)
        mask[:, [1, 2, 3]] = True  # 3 image tokens
        feats = {
            "C": torch.rand(1, 2),  # mismatch K_img=2
            "A": torch.rand(1, 2),
            "D": torch.rand(1, 2),
            "B": torch.rand(1, 2),
        }
        rt.begin_generation(mask, rfhar_feats=feats)
        logits = torch.randn(1, 2, 1, 6)
        b = rt.compute_bias(layer_idx=0, attn_logits_masked=logits, attention_mask=None)
        self.assertIsNone(b)
        rt.end_generation()


if __name__ == "__main__":
    unittest.main()
