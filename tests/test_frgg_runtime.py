import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llava.model.ais_gating import AISGatingConfig, AISGatingRuntime, install_ais_gating_hooks
from llava.model.frgg import FRGG


class TestFRGGModule(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def _dummy_feats(self, bsz: int, k_img: int):
        x = torch.rand(bsz, k_img)
        return {"A": x.clone(), "C": x.clone(), "E": x.clone()}

    def test_gamma_zero_is_noop(self):
        m = FRGG(gamma=0.0)
        logits = torch.randn(2, 4, 8)
        image_mask = torch.zeros(2, 8, dtype=torch.bool)
        image_mask[:, [1, 3, 5]] = True
        feats = self._dummy_feats(bsz=2, k_img=3)
        faithful = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
        out, dbg = m(attn_logits_last=logits, image_mask=image_mask, feats=feats, faithful_head_mask=faithful)
        self.assertTrue(torch.equal(out, logits))
        self.assertEqual(float(dbg["frgg_delta_abs_mean"]), 0.0)

    def test_only_image_columns_changed(self):
        m = FRGG(gamma=0.4)
        logits = torch.randn(2, 4, 7)
        image_mask = torch.zeros(2, 7, dtype=torch.bool)
        image_mask[0, [1, 4]] = True
        image_mask[1, [0, 6]] = True
        feats = self._dummy_feats(bsz=2, k_img=2)
        faithful = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
        out, _ = m(attn_logits_last=logits, image_mask=image_mask, feats=feats, faithful_head_mask=faithful)

        for b in range(2):
            text_cols = torch.nonzero(~image_mask[b], as_tuple=False).flatten()
            img_cols = torch.nonzero(image_mask[b], as_tuple=False).flatten()
            self.assertTrue(torch.equal(out[b, :, text_cols], logits[b, :, text_cols]))
            self.assertGreater(float((out[b, :, img_cols] - logits[b, :, img_cols]).abs().sum().item()), 0.0)


class TestFRGGRuntime(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

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

        rt = AISGatingRuntime(AISGatingConfig(enable_frgg=False))
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

        cfg = AISGatingConfig(
            enable_frgg=True,
            frgg_late_start=0,
            frgg_late_end=3,
            frgg_gamma=0.0,
            ais_faithful_heads="0:0,1:0,2:0,3:0",
        )
        rt = AISGatingRuntime(cfg)
        install_ais_gating_hooks(model, runtime=rt)
        feats = {
            "A": torch.rand(2, 2),
            "C": torch.rand(2, 2),
            "E": torch.rand(2, 2),
        }
        mask = torch.zeros((2, 6), dtype=torch.bool)
        mask[:, [1, 2]] = True
        rt.begin_generation(mask, frgg_feats=feats)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        rt.end_generation()
        self.assertTrue(torch.equal(base, out))

    def test_only_late_last_row_and_image_columns_changed(self):
        cfg = AISGatingConfig(
            enable_frgg=True,
            frgg_late_start=2,
            frgg_late_end=2,
            frgg_gamma=0.5,
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
        }
        rt.begin_generation(mask, frgg_feats=feats)

        logits = torch.randn(2, 4, 2, 8)
        b0 = rt.compute_bias(layer_idx=1, attn_logits_masked=logits, attention_mask=None)
        self.assertIsNone(b0)

        b1 = rt.compute_bias(layer_idx=2, attn_logits_masked=logits, attention_mask=None)
        self.assertIsNotNone(b1)
        self.assertTrue(torch.equal(b1[:, :, :-1, :], torch.zeros_like(b1[:, :, :-1, :])))
        for b in range(2):
            text_cols = torch.nonzero(~mask[b], as_tuple=False).flatten()
            self.assertTrue(torch.equal(b1[b, :, -1, text_cols], torch.zeros_like(b1[b, :, -1, text_cols])))
        rt.end_generation()


if __name__ == "__main__":
    unittest.main()
