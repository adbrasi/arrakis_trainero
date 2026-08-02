"""Slider merge math: ΔW_slider == ΔW_pos − ΔW_neg (needs torch; runs on pod)."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parent.parent


@unittest.skipUnless(HAS_TORCH, "torch não instalado (roda no pod, dentro do venv de engine)")
class TestSliderMath(unittest.TestCase):
    def test_delta_w_is_exact_difference(self):
        import torch
        from safetensors.torch import load_file, save_file

        torch.manual_seed(0)

        def make_lora(rank, alpha, out_f=8, in_f=6):
            return {
                "lora_unet_block.lora_down.weight": torch.randn(rank, in_f),
                "lora_unet_block.lora_up.weight": torch.randn(out_f, rank),
                "lora_unet_block.alpha": torch.tensor(float(alpha)),
            }

        def delta_w(sd):
            down = sd["lora_unet_block.lora_down.weight"].float()
            up = sd["lora_unet_block.lora_up.weight"].float()
            alpha = float(sd["lora_unet_block.alpha"])
            return up @ down * (alpha / down.shape[0])

        pos, neg = make_lora(4, 2.0), make_lora(3, 3.0)
        with tempfile.TemporaryDirectory() as td:
            p, n, o = Path(td) / "p.safetensors", Path(td) / "n.safetensors", Path(td) / "o.safetensors"
            save_file(pos, str(p))
            save_file(neg, str(n))
            subprocess.run([sys.executable, str(ROOT / "trainero/tools/make_slider.py"),
                            str(p), str(n), str(o)], check=True)
            merged = load_file(str(o))
            expected = delta_w(pos) - delta_w(neg)
            torch.testing.assert_close(delta_w(merged), expected, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
