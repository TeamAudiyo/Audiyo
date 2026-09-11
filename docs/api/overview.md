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
