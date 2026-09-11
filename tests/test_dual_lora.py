from __future__ import annotations

import torch


def make_dual():
    from audiyo.testkit import TinyDualMusic

    dual = TinyDualMusic()
    return {"llm": dual.language_model, "transformer": dual.transformer}


def grad_names(module) -> list:
    return sorted(n for n, p in module.named_parameters() if p.requires_grad)


def test_assumed_lists_are_labeled_assumed():
    from audiyo.adapters import ASSUMED_NOTE, MUSIC_DIT_TARGETS, MUSIC_LLM_TARGETS

    assert "Assumed" in ASSUMED_NOTE
    assert set(MUSIC_LLM_TARGETS) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert set(MUSIC_DIT_TARGETS) == {"to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"}


def test_assumed_lists_hit_proxy_modules():
    from audiyo.adapters import assumed_targets_for, find_lora_targets

    dual = make_dual()
    assert find_lora_targets(dual["llm"], assumed_targets_for("llm")) == list(assumed_targets_for("llm"))
    assert find_lora_targets(dual["transformer"], assumed_targets_for("transformer")) == list(assumed_targets_for("transformer"))


def test_llm_target_freezes_dit():
    from audiyo.adapters import assert_dual_frozen, attach_dual_lora

    dual = make_dual()
    adapters = attach_dual_lora(dual, target="llm", rank=2, alpha=2)
    assert set(adapters) == {"llm"}
    assert_dual_frozen(dual, adapters)
    assert all("lora_" in n.lower() for n in grad_names(adapters["llm"]))
    assert grad_names(dual["transformer"]) == []


def test_transformer_target_freezes_llm():
    from audiyo.adapters import assert_dual_frozen, attach_dual_lora

    dual = make_dual()
    adapters = attach_dual_lora(dual, target="transformer", rank=2, alpha=2)
    assert set(adapters) == {"transformer"}
    assert_dual_frozen(dual, adapters)
    assert grad_names(dual["llm"]) == []
    assert all("lora_" in n.lower() for n in grad_names(adapters["transformer"]))


def test_both_target_trains_both_bases_frozen():
    from audiyo.adapters import assert_dual_frozen, attach_dual_lora

    dual = make_dual()
    adapters = attach_dual_lora(dual, target="both", rank=2, alpha=2)
    assert set(adapters) == {"llm", "transformer"}
    counts = assert_dual_frozen(dual, adapters)
    assert counts["llm"] > 0
    assert counts["transformer"] > 0


def test_bad_target_rejected():
    from audiyo.adapters import attach_dual_lora
    from audiyo.errors import ValidationError

    try:
        attach_dual_lora(make_dual(), target="vocals")
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")


def test_dual_save_reload_matches():
    import json
    import os
    import tempfile

    from audiyo.adapters import attach_dual_lora, load_dual_adapters, save_dual_adapters

    d = tempfile.mkdtemp()
    out = os.path.join(d, "dual")
    dual = make_dual()
    adapters = attach_dual_lora(dual, target="both", rank=2, alpha=2)
    x = torch.randn(1, 4, 32)
    before = {}
    for r in adapters:
        before[r] = {k: v.detach().clone() for k, v in adapters[r].state_dict().items() if "lora_" in k}
        assert before[r]
    save_dual_adapters(adapters, out, "unit-test", 2, 2)
    manifest = json.load(open(os.path.join(out, "manifest.json")))
    assert manifest["roles"] == ["llm", "transformer"]
    assert manifest["base_model"] == "unit-test"
    assert os.path.isfile(os.path.join(out, "llm", "audiyo_adapter.json"))
    assert os.path.isfile(os.path.join(out, "transformer", "audiyo_adapter.json"))
    fresh = make_dual()
    reloaded = load_dual_adapters(fresh, out, target="both")
    for r in reloaded:
        after = {k: v.detach().clone() for k, v in reloaded[r].state_dict().items() if "lora_" in k}
        assert set(after) == set(before[r])
        for k in after:
            assert bool(torch.allclose(after[k], before[r][k], atol=1e-8))
        assert bool(torch.isfinite(reloaded[r](x)).all())


def test_dual_partial_reload():
    import os
    import tempfile

    from audiyo.adapters import attach_dual_lora, load_dual_adapters, save_dual_adapters

    d = tempfile.mkdtemp()
    out = os.path.join(d, "dual")
    dual = make_dual()
    adapters = attach_dual_lora(dual, target="both", rank=2, alpha=2)
    save_dual_adapters(adapters, out, "unit-test", 2, 2)
    fresh = make_dual()
    reloaded = load_dual_adapters(fresh, out, target="llm")
    assert set(reloaded) == {"llm"}


def test_role_validation_against_backend_roles():
    from audiyo.adapters import validate_target_for_roles
    from audiyo.errors import ValidationError

    assert validate_target_for_roles("both", ("llm", "transformer")) == ("llm", "transformer")
    assert validate_target_for_roles("transformer", ("transformer",)) == ("transformer",)
    try:
        validate_target_for_roles("llm", ("transformer",))
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")
