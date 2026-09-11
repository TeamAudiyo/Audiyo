from __future__ import annotations

VOCAB = 64


def target_next(prefix: tuple) -> int:
    return (prefix[-1] * 7 + 3) % VOCAB


def target_batch(prefixes: list) -> list:
    return [target_next(p) for p in prefixes]


def draft_next(prefix: tuple) -> int:
    prev = prefix[-1]
    if prev % 5 == 0:
        return (prev + 1) % VOCAB
    return (prev * 7 + 3) % VOCAB


def positional_next(prefix: tuple) -> int:
    return (len(prefix) * 5 + 1) % VOCAB


def positional_batch(prefixes: list) -> list:
    return [positional_next(p) for p in prefixes]


def test_speculative_matches_greedy_with_fewer_target_calls():
    from audiyo.backends.music_decode import greedy_decode, speculative_decode

    start = [11]
    want = greedy_decode(target_next, start, 40)
    got = speculative_decode(target_next, target_batch, draft_next, start, 40, gamma=4)
    assert got["ids"] == want["ids"]
    assert got["target_calls"] < want["target_calls"]
    assert got["accepted"] < got["total"]
    assert got["total"] == 40


def test_speculative_rejects_bad_input():
    from audiyo.backends.music_decode import speculative_decode
    from audiyo.errors import ValidationError

    try:
        speculative_decode(target_next, target_batch, draft_next, [], 10)
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")


def test_jacobi_converges_to_greedy_in_two_sweeps():
    from audiyo.backends.music_decode import greedy_decode, jacobi_decode

    start = [4]
    want = greedy_decode(positional_next, start, 24)
    got = jacobi_decode(positional_batch, start, 24)
    assert got["ids"] == want["ids"]
    assert got["sweeps"] == 2
    assert got["target_calls"] == 2


def test_overlap_plan_math_and_order():
    from audiyo.backends.music_decode import plan_overlap

    plan = plan_overlap(2.0, 5.0, 4)
    assert plan["sequential_span"] == 28.0
    assert plan["overlap_span"] == 22.0
    assert plan["speedup"] == round(28.0 / 22.0, 3)
    order = plan["order"]
    assert order[0] == ("ar", 0)
    assert order[-1] == ("flow", 3)
    for i in range(4):
        assert order.index(("ar", i)) < order.index(("flow", i))


def test_cached_oracle_never_recomputes_a_prefix():
    from audiyo.backends.music_decode import cached_oracle, greedy_decode

    cached = cached_oracle(positional_next)
    first = greedy_decode(cached, [4], 24)
    assert cached.stats()["real_calls"] == 24
    second = greedy_decode(cached, [4], 24)
    assert second["ids"] == first["ids"]
    assert cached.stats()["real_calls"] == 24


def test_model_target_validation():
    from audiyo.adapters import validate_target_for_roles

    assert validate_target_for_roles("both", ("llm", "transformer")) == ("llm", "transformer")


def test_model_llm_target_rejected_on_single_backend():
    from audiyo.adapters import validate_target_for_roles
    from audiyo.errors import ValidationError

    try:
        validate_target_for_roles("llm", ("transformer",))
    except ValidationError as exc:
        assert "only has" in str(exc)
        return
    raise AssertionError("expected ValidationError")
