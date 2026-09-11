from __future__ import annotations

from . import config as config
from . import audio as audio
from . import backends as backends
from . import errors as errors
from . import memory as memory
from . import utils as utils
from . import inference as inference
from . import dataio as dataio
from . import adapters as adapters
from . import memopt as memopt
from . import benchutils as benchutils
from . import testkit as testkit
from ._version import __version__

__all__ = ["AudioModel", "AudioResult", "GenerationConfig", "FinetuneConfig", "__version__"]


def __getattr__(name: str):
    if name == "AudioModel":
        from .model import AudioModel

        return AudioModel
    if name == "AudioResult":
        from .audio import AudioResult

        return AudioResult
    if name == "GenerationConfig":
        from .config import GenerationConfig

        return GenerationConfig
    if name == "FinetuneConfig":
        from .config import FinetuneConfig

        return FinetuneConfig
    raise AttributeError(name)
