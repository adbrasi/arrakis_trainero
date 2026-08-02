#!/usr/bin/env python3
"""Build a concept-slider LoRA from a positive and a negative LoRA.

ΔW_slider = ΔW_pos − ΔW_neg, computed EXACTLY by rank concatenation:
for each module,
    down = cat(down_pos, down_neg)             (rank axis)
    up   = cat(up_pos·s_pos, −up_neg·s_neg)    (rank axis, s = alpha/rank)
    alpha = new_rank                           (so the merged scale is 1.0)

Works for any sd-scripts-format LoRA (musubi-tuner and sd-scripts both save
lora_*.lora_down.weight / .lora_up.weight / .alpha). Modules present in only
one file are kept with their sign. Runs inside an engine venv (needs torch).

Usage: make_slider.py POS.safetensors NEG.safetensors OUT.safetensors
"""

from __future__ import annotations

import sys

import torch
from safetensors.torch import load_file, save_file


def module_map(sd: dict) -> dict[str, dict]:
    mods: dict[str, dict] = {}
    for key, tensor in sd.items():
        for suffix, name in ((".lora_down.weight", "down"), (".lora_up.weight", "up"), (".alpha", "alpha")):
            if key.endswith(suffix):
                mods.setdefault(key[: -len(suffix)], {})[name] = tensor
                break
    return {k: v for k, v in mods.items() if "down" in v and "up" in v}


def scaled(mod: dict) -> tuple[torch.Tensor, torch.Tensor]:
    down = mod["down"].float()
    up = mod["up"].float()
    rank = down.shape[0]
    alpha = float(mod["alpha"].item()) if "alpha" in mod else float(rank)
    return down, up * (alpha / rank)


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    pos_path, neg_path, out_path = sys.argv[1:4]
    pos = module_map(load_file(pos_path))
    neg = module_map(load_file(neg_path))

    out: dict[str, torch.Tensor] = {}
    for name in sorted(set(pos) | set(neg)):
        parts_down, parts_up = [], []
        if name in pos:
            d, u = scaled(pos[name])
            parts_down.append(d)
            parts_up.append(u)
        if name in neg:
            d, u = scaled(neg[name])
            parts_down.append(d)
            parts_up.append(-u)
        down = torch.cat(parts_down, dim=0)
        up = torch.cat(parts_up, dim=1)
        rank = down.shape[0]
        out[f"{name}.lora_down.weight"] = down.to(torch.float16).contiguous()
        out[f"{name}.lora_up.weight"] = up.to(torch.float16).contiguous()
        out[f"{name}.alpha"] = torch.tensor(float(rank))

    save_file(out, out_path, metadata={"trainero_slider": "1",
                                       "pos": pos_path.rsplit("/", 1)[-1],
                                       "neg": neg_path.rsplit("/", 1)[-1]})
    print(f"slider salvo: {out_path} ({len(out) // 3} módulos)")


if __name__ == "__main__":
    main()
