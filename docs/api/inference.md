# Inference modules

* `inference/generator.py` - builds a CPU generator from a seed. Seeds repeat on the same setup only.
* `inference/outputs.py` - pulls waveforms from pipeline output and enforces stereo shape.
* `inference/performance.py` - builds performance and settings dicts on AudioResult.
* `utils/memsnap.py` - memory snapshots, kept separate by source.
* `utils/timing.py` - timing helpers.
