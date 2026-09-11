from __future__ import annotations

from contextlib import contextmanager

BYTES_PER_PARAM = {"float32": 4.0, "float16": 2.0, "bfloat16": 2.0, "int8": 1.0, "int4": 0.5}

STAGE_SPECS = (
    {"name": "structure", "params_billion": 8.6, "note": "8B global LLM plus 0.6B depth decoder run together inside the autoregressive loop"},
    {"name": "flow", "params_billion": 2.4, "note": "2.4B flow-matching transformer plus condition encoder, already chunked frame by frame"},
    {"name": "vocoder", "params_billion": 0.123, "note": "123M DAC-style waveform decoder"},
)

PRESET_STAGE_PLAN = {
    "performance": {"resident": "all", "llm_dtype": "bfloat16", "flow_dtype": "bfloat16", "note": "Everything resident. Needs about 23 GB. Fastest."},
    "balanced": {"resident": "one stage", "llm_dtype": "bfloat16", "flow_dtype": "bfloat16", "note": "Sequential stages in bfloat16. Peak is the structure stage."},
    "low": {"resident": "one stage", "llm_dtype": "int8", "flow_dtype": "bfloat16", "note": "Sequential stages with the LLM in 8-bit. Needs bitsandbytes."},
    "minimal": {"resident": "one stage", "llm_dtype": "int4", "flow_dtype": "bfloat16", "note": "Sequential stages with the LLM in 4-bit. Needs bitsandbytes. Slowest."},
}


def estimate_stage_gb(params_billion: float, dtype: str = "bfloat16", margin: float = 0.2) -> float:
    if dtype not in BYTES_PER_PARAM:
        from ..errors import ValidationError

        raise ValidationError("Unknown dtype " + repr(dtype) + ".")
    return round(float(params_billion) * BYTES_PER_PARAM[dtype] * (1.0 + margin), 2)


def estimate_plan(llm_dtype: str = "bfloat16", margin: float = 0.2) -> dict:
    stages = {}
    for spec in STAGE_SPECS:
        dtype = llm_dtype if spec["name"] == "structure" else "bfloat16"
        stages[spec["name"]] = estimate_stage_gb(spec["params_billion"], dtype, margin)
    return {"stages": stages, "peak_gb": round(max(stages.values()), 2), "llm_dtype": llm_dtype}


class StageOffloader:
    def __init__(self, stages: dict, device: str = "cpu"):
        if set(stages) != {"structure", "flow", "vocoder"}:
            from ..errors import ValidationError

            raise ValidationError("Stages must be structure, flow, and vocoder.")
        self.stages = stages
        self.device = device
        self.active: str | None = None
        self.order: list = []

    def activate(self, name: str) -> str:
        import torch

        if name not in self.stages:
            from ..errors import ValidationError

            raise ValidationError("Unknown stage " + repr(name) + ".")
        for stage_name, module in self.stages.items():
            module.to(self.device if stage_name == name else "cpu")
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        self.active = name
        self.order.append(name)
        return name

    @contextmanager
    def stage(self, name: str):
        self.activate(name)
        yield self.stages[name]

    def resident_names(self) -> list:
        if self.device == "cpu":
            return sorted(self.stages)
        return [self.active] if self.active else []
