# What is in this folder

Sample audio made with the bundled test model (`audiyo.testkit`), not with Stable Audio Open.

* `baseline_seed42.wav` - short clip for prompt "rain on a window", seed 42, before training. It sounds like noise. That is expected. The test model has random weights.
* `adapter_seed42.wav` - same prompt and seed, after a short LoRA run on a test tone. Also noise, but slightly different. That difference shows the adapter trained and changed output.
* `tone_data/` - the training clip (a synthetic tone plus caption).
* `adapter_run/` - the trained test adapter and trainer state.

## Real model audio

These three files came from the `cvssp/audioldm-s-full-v2` checkpoint (CC-BY-NC-SA-4.0, research and non-commercial use only):

* `rain_forest.wav` - "steady rain in a forest". Play this one first.
* `rain_window.wav` - "rain against a window with distant thunder". Quiet as saved from the model.
* `rain_window_loud.wav` - same audio with gain raised so it is easier to hear. Only the volume changed.

These are real diffusion output, not the test model. Whether they sound like rain is for you to judge. This pipeline is a side experiment only. Audiyo itself supports Stable Audio Open.
