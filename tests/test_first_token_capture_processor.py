import unittest

import torch

from llava.eval.model_vqa_loader import FirstTokenCaptureAndBiasProcessor


class FirstTokenCaptureProcessorTest(unittest.TestCase):
    def test_capture_without_bias(self):
        p = FirstTokenCaptureAndBiasProcessor(
            prompt_len=5,
            yes_id=1,
            no_id=2,
            apply_bias=False,
        )
        input_ids = torch.zeros((1, 5), dtype=torch.long)
        scores = torch.tensor([[0.1, 1.25, -0.75, 0.0]], dtype=torch.float32)
        scores_before = scores.clone()

        out = p(input_ids, scores)
        cap = p.get_capture()

        self.assertIsNotNone(cap)
        self.assertAlmostEqual(cap["yes_logit_pre"], 1.25, places=6)
        self.assertAlmostEqual(cap["no_logit_pre"], -0.75, places=6)
        self.assertAlmostEqual(cap["yes_logit_post"], 1.25, places=6)
        self.assertAlmostEqual(cap["no_logit_post"], -0.75, places=6)
        self.assertAlmostEqual(cap["delta_margin"], 0.0, places=6)
        self.assertTrue(torch.allclose(out, scores_before))

    def test_capture_with_bias(self):
        p = FirstTokenCaptureAndBiasProcessor(
            prompt_len=7,
            yes_id=0,
            no_id=3,
            yes_bias=0.5,
            no_bias=-0.2,
            apply_bias=True,
        )
        input_ids = torch.zeros((1, 7), dtype=torch.long)
        scores = torch.tensor([[1.0, -0.1, 0.4, 0.6]], dtype=torch.float32)

        out = p(input_ids, scores)
        cap = p.get_capture()

        self.assertIsNotNone(cap)
        self.assertAlmostEqual(cap["yes_logit_pre"], 1.0, places=6)
        self.assertAlmostEqual(cap["no_logit_pre"], 0.6, places=6)
        self.assertAlmostEqual(cap["yes_logit_post"], 1.5, places=6)
        self.assertAlmostEqual(cap["no_logit_post"], 0.4, places=6)
        self.assertAlmostEqual(cap["margin_pre"], 0.4, places=6)
        self.assertAlmostEqual(cap["margin_post"], 1.1, places=6)
        self.assertAlmostEqual(cap["delta_margin"], 0.7, places=6)
        self.assertAlmostEqual(float(out[0, 0].item()), 1.5, places=6)
        self.assertAlmostEqual(float(out[0, 3].item()), 0.4, places=6)

    def test_capture_first_call_even_if_length_differs(self):
        p = FirstTokenCaptureAndBiasProcessor(
            prompt_len=999,  # intentionally mismatched
            yes_id=1,
            no_id=2,
            yes_bias=0.3,
            no_bias=0.0,
            apply_bias=True,
        )
        input_ids = torch.zeros((1, 1), dtype=torch.long)
        scores = torch.tensor([[0.0, 0.2, -0.1]], dtype=torch.float32)
        out = p(input_ids, scores)
        cap = p.get_capture()
        self.assertIsNotNone(cap)
        self.assertAlmostEqual(cap["yes_logit_pre"], 0.2, places=6)
        self.assertAlmostEqual(cap["yes_logit_post"], 0.5, places=6)
        self.assertAlmostEqual(float(out[0, 1].item()), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
