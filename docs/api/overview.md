# API overview

Map of where things live. Public imports stay the same.

* `audiyo.AudioModel` - in `src/audiyo/model.py`, helpers in `src/audiyo/inference/` and `src/audiyo/utils/`
* `audiyo.audio` - result plus loading, helpers in `src/audiyo/dataio/`
* `audiyo.config` - checkpoint, constants, configs
* `audiyo.datasets` - dataset discovery and prep
* `audiyo.memory` - presets and applier, split into `src/audiyo/memopt/`
* `audiyo.lora` - injection and storage, logic in `src/audiyo/adapters/`
* `audiyo.training` - training loop
* `audiyo.benchmark` - benchmark runner, helpers in `src/audiyo/benchutils/`
* `audiyo.hardware`, `audiyo.backend`, `audiyo.chunked`, `audiyo.errors`
* `audiyo.postfx` - fades, normalize, limiter, silence trim
* `audiyo.quality` - levels, spectrum, compare, listening sheets
* `audiyo.estimate` - load-time memory estimate before download
* `audiyo.configfile` - JSON config files for the CLI
* `audiyo.ui` - optional local web demo, needs pip install audiyo[ui]
