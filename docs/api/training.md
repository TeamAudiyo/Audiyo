# Training modules

* `adapters/targets.py` - finds layers and checks LoRA targets.
* `adapters/meta.py` - reads and writes `audiyo_adapter.json`.
* `lora.py` - attaches adapters, checks only adapters train, saves and reloads.
* `training.py` - v-prediction loop, conditioning, resume, checks.
