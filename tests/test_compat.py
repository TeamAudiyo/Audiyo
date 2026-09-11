from __future__ import annotations


def test_audio_reexports_dataio_objects():
    import audiyo.audio as audio
    import audiyo.dataio as dataio
    import audiyo.dataio.channels as channels
    import audiyo.dataio.windows as windows

    assert audio.convert_channels is channels.convert_channels
    assert audio.crop_or_pad is windows.crop_or_pad
    assert audio.samples_for_duration is windows.samples_for_duration
    assert audio.SUPPORTED_EXTENSIONS is dataio.SUPPORTED_EXTENSIONS


def test_lora_reexports_adapters_objects():
    import audiyo.adapters.meta as meta
    import audiyo.adapters.targets as targets
    import audiyo.lora as lora

    assert lora.AdapterMeta is meta.AdapterMeta
    assert lora.DEFAULT_TARGET_MODULES is targets.DEFAULT_TARGET_MODULES
    assert lora.find_lora_targets is targets.find_lora_targets
    assert lora.list_linear_modules is targets.list_linear_modules
    assert lora.verify_targets_or_raise is targets.verify_targets_or_raise
    assert lora.read_adapter_meta is meta.read_meta


def test_memory_reexports_memopt_objects():
    import audiyo.memory as memory
    import audiyo.memopt as memopt
    import audiyo.memopt.presets as presets

    assert memory.PRESET_DOCS is presets.PRESET_DOCS
    assert memory.describe_presets is memopt.describe_presets


def test_model_uses_shared_helpers():
    import audiyo.inference.outputs as outputs
    import audiyo.inference.performance as performance
    import audiyo.model as model
    import audiyo.utils as utils

    assert model._dtype_object is utils.dtype_object
    assert model._cuda_mem is utils.cuda_snapshot
    assert model._system_mem_gb is utils.system_snapshot
    assert model.extract_first_waveform is outputs.extract_first_waveform
    assert model.build_performance is performance.build_performance
    assert model.build_settings is performance.build_settings
