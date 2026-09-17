from __future__ import annotations


def build_stable_call_kwargs(cfg, generator: object) -> dict:
    return {
        "prompt": cfg.prompt,
        "audio_end_in_s": cfg.audio_start_in_s + float(cfg.duration_seconds),
        "audio_start_in_s": cfg.audio_start_in_s,
        "num_inference_steps": cfg.num_inference_steps,
        "guidance_scale": cfg.guidance_scale,
        "negative_prompt": cfg.negative_prompt,
        "num_waveforms_per_prompt": cfg.num_waveforms_per_prompt,
        "eta": cfg.eta,
        "generator": generator,
        "output_type": "pt",
    }


def build_music_call_kwargs(prompt: str, lyrics: str | None, duration_seconds: float, num_inference_steps: int, generator: object) -> dict:
    from ..errors import ValidationError

    if lyrics is None or not str(lyrics).strip():
        raise ValidationError("Minimax-Music3 needs lyrics. Pass lyrics with section tags like [verse] on their own lines.")
    call_kwargs: dict = {
        "prompt": prompt,
        "lyrics": lyrics,
        "audio_duration": float(duration_seconds),
        "num_inference_steps": int(num_inference_steps),
        "generator": generator,
        "output_type": "pt",
    }
    return call_kwargs


def check_music_limits(audio_start_in_s: float, num_waveforms_per_prompt: int) -> None:
    from ..errors import ValidationError

    if float(audio_start_in_s) != 0.0:
        raise ValidationError("Minimax-Music3 always starts at 0. audio_start_in_s must be 0 for this backend.")
    if int(num_waveforms_per_prompt) != 1:
        raise ValidationError("Minimax-Music3 renders one clip per call. Use generate_variations with seeds for more takes.")
