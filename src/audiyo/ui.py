from __future__ import annotations


def need_gradio():
    try:
        import gradio
    except ImportError as exc:
        from .errors import DependencyError

        raise DependencyError("The interface needs gradio. Run pip install audiyo[ui] and retry.") from exc
    return gradio


def build_demo(default_checkpoint: str = "stabilityai/stable-audio-open-1.0"):
    gradio = need_gradio()
    from .config import SUPPORTED_CHECKPOINTS

    choices = list(SUPPORTED_CHECKPOINTS)

    def run(prompt, lyrics, checkpoint, memory_mode, duration, seed, steps, guidance, fade_out, normalize, trim):
        from .model import AudioModel
        from .postfx import apply_postfx
        from .quality import analyze_waveform

        model = AudioModel.from_pretrained(checkpoint=checkpoint, device="auto", memory_mode=memory_mode)
        lyrics_value = lyrics.strip() if isinstance(lyrics, str) and lyrics.strip() else None
        result = model.generate(
            prompt=prompt,
            duration_seconds=float(duration),
            seed=int(seed),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            lyrics=lyrics_value,
        )
        wav = result.waveform
        applied = {}
        if fade_out or normalize or trim:
            target = 0.89 if normalize else None
            wav, applied = apply_postfx(
                wav,
                result.sample_rate,
                fade_in_ms=20.0 if fade_out else 0.0,
                fade_out_ms=200.0 if fade_out else 0.0,
                normalize_peak=target,
                trim_silence=bool(trim),
            )
            result.waveform = wav
        metrics = analyze_waveform(wav, result.sample_rate)
        audio_out = (result.sample_rate, wav.T)
        summary = {
            "peak": metrics["peak"],
            "rms": metrics["rms"],
            "centroid_hz": metrics["spectral_centroid_hz"],
            "latency_s": result.performance.get("latency_s"),
            "postfx": applied,
        }
        return audio_out, summary

    with gradio.Blocks(title="Audiyo") as demo:
        gradio.Markdown("Generate audio, check levels, save what sounds right.")
        with gradio.Row():
            prompt = gradio.Textbox(label="Prompt", value="Rain against a window with distant thunder")
            checkpoint = gradio.Dropdown(label="Checkpoint", choices=choices, value=default_checkpoint)
        lyrics = gradio.Textbox(label="Lyrics, for music only, else leave blank", lines=3, value="")
        with gradio.Row():
            memory_mode = gradio.Dropdown(label="Memory mode", choices=["performance", "balanced", "low", "minimal"], value="balanced")
            duration = gradio.Slider(1, 47, value=10, step=0.5, label="Seconds")
            seed = gradio.Number(value=42, precision=0, label="Seed")
        with gradio.Row():
            steps = gradio.Slider(1, 200, value=50, step=1, label="Steps")
            guidance = gradio.Slider(0, 20, value=7, step=0.5, label="Guidance")
        with gradio.Row():
            fade_out = gradio.Checkbox(value=True, label="Gentle fade")
            normalize = gradio.Checkbox(value=False, label="Normalize peak")
            trim = gradio.Checkbox(value=False, label="Trim edge silence")
        go = gradio.Button("Generate")
        audio = gradio.Audio(label="Output")
        facts = gradio.JSON(label="Levels and timing")
        go.click(run, inputs=[prompt, lyrics, checkpoint, memory_mode, duration, seed, steps, guidance, fade_out, normalize, trim], outputs=[audio, facts])
    return demo


def launch_ui(port: int = 7860, share: bool = False, checkpoint: str = "stabilityai/stable-audio-open-1.0") -> None:
    demo = build_demo(default_checkpoint=checkpoint)
    demo.launch(server_port=int(port), share=bool(share))
