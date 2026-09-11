from __future__ import annotations

import torch.nn as nn


def _tiny_transformer():
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(8, 8)
            self.to_k = nn.Linear(8, 8)
            self.to_v = nn.Linear(8, 8)
            self.to_out = nn.ModuleList([nn.Linear(8, 8)])

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = Attn()

    class Tr(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()

    return Tr()


def test_lora_target_search():
    from audiyo.lora import find_lora_targets

    found = find_lora_targets(_tiny_transformer())
    assert "to_q" in found
    assert "to_v" in found


def test_lora_freeze_check():
    from audiyo.lora import assert_only_adapter_trainable
    from audiyo.errors import ValidationError

    m = nn.Linear(4, 4)
    for p in m.parameters():
        p.requires_grad = True
    try:
        assert_only_adapter_trainable(m)
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")


def test_adapters_targets_match_lora():
    from audiyo.lora import find_lora_targets
    from audiyo.adapters import find_lora_targets as find_new

    assert find_lora_targets(_tiny_transformer()) == find_new(_tiny_transformer())
