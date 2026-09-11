from __future__ import annotations

from .modules import TinyDiT, TinyProjection, TinyTextEncoder, TinyVAE
from .music import MUSIC_MAX_PROXY_SECONDS, MUSIC_SAMPLE_RATE, TinyDualMusic, TinyFlowBlock, TinyMusicPipeline, TinyQwenBlock, TinyStagedMusicPipeline, parse_lyrics_sections
from .pipeline import LATENT_DIM, SAMPLE_RATE, WINDOW_FRAMES, WINDOW_SAMPLES, TinyPipeline, build_tiny_model, count_parameters

__all__ = ["TinyDiT", "TinyProjection", "TinyTextEncoder", "TinyVAE", "LATENT_DIM", "SAMPLE_RATE", "WINDOW_FRAMES", "WINDOW_SAMPLES", "TinyPipeline", "build_tiny_model", "count_parameters", "MUSIC_MAX_PROXY_SECONDS", "MUSIC_SAMPLE_RATE", "TinyDualMusic", "TinyFlowBlock", "TinyMusicPipeline", "TinyQwenBlock", "TinyStagedMusicPipeline", "parse_lyrics_sections"]
