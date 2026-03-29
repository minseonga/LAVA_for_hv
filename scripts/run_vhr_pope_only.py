#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Run VHR native POPE inference only (no CHAIR loop).")
    ap.add_argument("--vhr_repo", type=str, default="/home/kms/VHR")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--method", type=str, default="vhr")
    ap.add_argument("--vhr_aug_ratio", type=float, default=2.0)
    ap.add_argument("--vhr_layers", type=int, default=14)
    ap.add_argument("--vhr_layer1", action="store_true")
    ap.add_argument("--vhr_filter", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_idx", type=int, default=0)
    args = ap.parse_args()

    vhr_repo = os.path.abspath(args.vhr_repo)
    os.chdir(vhr_repo)
    if vhr_repo not in sys.path:
        sys.path.insert(0, vhr_repo)

    import main as vhr_main  # noqa: WPS433 (runtime import intended)

    ns = argparse.Namespace(
        model_path=str(args.model_path),
        num_chunks=int(args.num_chunks),
        chunk_idx=int(args.chunk_idx),
        device=str(args.device),
        method=str(args.method),
        vhr_aug_ratio=float(args.vhr_aug_ratio),
        vhr_layers=int(args.vhr_layers),
        vhr_layer1=bool(args.vhr_layer1),
        vhr_filter=bool(args.vhr_filter),
        max_new_tokens=int(args.max_new_tokens),
        seed=int(args.seed),
        output_dir=os.path.abspath(args.output_dir),
    )

    runner = vhr_main.HallucinationTest(ns)
    for ds in ["pope_popular", "pope_adversarial", "pope_random"]:
        runner.infer_dataset(ds, int(args.seed))
    print("[done]", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()

