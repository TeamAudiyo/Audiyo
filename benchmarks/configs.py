from __future__ import annotations

PROMPTS = [
    "Rain against a window with distant thunder",
    "A busy cafe with cups clinking",
    "Waves on a pebble beach at night",
]

DURATIONS = [4.0, 10.0, 20.0]
STEPS = [20, 50, 100]
SEEDS = [42]
MODES = ["performance", "balanced", "low", "minimal"]


def default_suite() -> dict:
    return {
        "prompts": list(PROMPTS),
        "durations": list(DURATIONS),
        "steps": list(STEPS),
        "seeds": list(SEEDS),
        "modes": list(MODES),
    }
