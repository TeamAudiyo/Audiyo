from __future__ import annotations


def cached_oracle(fn):
    """Memoize a prefix oracle. Repeated prefixes cost one real call. Pairs well with Jacobi sweeps, which revisit the same prefixes every round."""
    cache: dict = {}
    calls = {"real": 0}

    def wrapped(prefix):
        key = tuple(prefix)
        if key not in cache:
            calls["real"] += 1
            cache[key] = fn(key)
        return cache[key]

    def stats():
        return {"real_calls": calls["real"], "cached_prefixes": len(cache)}

    wrapped.stats = stats
    return wrapped


def greedy_decode(target_next, start_ids: list, max_new: int) -> dict:
    ids = list(start_ids)
    calls = 0
    for _ in range(max_new):
        ids.append(target_next(tuple(ids)))
        calls += 1
    return {"ids": ids, "target_calls": calls}


def speculative_decode(target_next, target_batch, draft_next, start_ids: list, max_new: int, gamma: int = 4) -> dict:
    """Greedy speculative sampling. Each round drafts gamma tokens cheaply, verifies them with one batched target pass, keeps the accepted prefix, and appends one bonus token. Output always equals plain greedy decoding."""
    if not start_ids or max_new < 1 or gamma < 1:
        from ..errors import ValidationError

        raise ValidationError("Need non-empty start ids plus max_new and gamma of at least 1.")
    ids = list(start_ids)
    target_calls = 0
    draft_calls = 0
    accepted = 0
    produced = 0
    while produced < max_new:
        drafted: list = []
        probe = list(ids)
        for _ in range(min(gamma, max_new - produced)):
            nxt = draft_next(tuple(probe))
            draft_calls += 1
            drafted.append(nxt)
            probe.append(nxt)
        prefixes = []
        probe = list(ids)
        for tok in drafted:
            prefixes.append(tuple(probe))
            probe.append(tok)
        verified = target_batch(prefixes)
        target_calls += 1
        keep = 0
        for tok, want in zip(drafted, verified):
            if tok != want:
                break
            keep += 1
        ids.extend(drafted[:keep])
        accepted += keep
        produced += keep
        if keep < len(drafted):
            ids.append(verified[keep])
            produced += 1
        elif produced < max_new:
            ids.append(target_next(tuple(ids)))
            target_calls += 1
            produced += 1
    return {"ids": ids, "target_calls": target_calls, "draft_calls": draft_calls, "accepted": accepted, "total": produced}


def jacobi_decode(target_batch, start_ids: list, max_new: int, max_sweeps: int = 16) -> dict:
    """Parallel fixed-point decoding. Every position is re-predicted at once each sweep until the whole block stops changing. Converged output equals greedy decoding; wall-clock wins need a backend that truly batches."""
    if not start_ids or max_new < 1 or max_sweeps < 1:
        from ..errors import ValidationError

        raise ValidationError("Need non-empty start ids plus max_new and max_sweeps of at least 1.")
    current = list(start_ids) + [start_ids[-1]] * max_new
    sweeps = 0
    for _ in range(max_sweeps):
        sweeps += 1
        prefixes = [tuple(current[: len(start_ids) + i]) for i in range(max_new)]
        refreshed = target_batch(prefixes)
        block = list(current[len(start_ids) :])
        if refreshed == block:
            break
        current[len(start_ids) :] = refreshed
    return {"ids": current, "sweeps": sweeps, "target_calls": sweeps}


def plan_overlap(ar_per_chunk: float, flow_per_chunk: float, n_chunks: int) -> dict:
    """Software-pipelined schedule for the autoregressive and flow stages. Chunk i of flow runs after chunk i of the language model, while the language model starts chunk i plus one. Returns the dependency-valid order plus span math."""
    order: list = []
    for i in range(n_chunks):
        order.append(("ar", i))
        if i > 0:
            order.append(("flow", i - 1))
    order.append(("flow", n_chunks - 1))
    a = float(ar_per_chunk)
    f = float(flow_per_chunk)
    n = int(n_chunks)
    sequential = round(n * (a + f), 4)
    overlap = round(a + f + (n - 1) * max(a, f), 4)
    return {"order": order, "sequential_span": sequential, "overlap_span": overlap, "speedup": round(sequential / overlap, 3) if overlap else 0.0}
