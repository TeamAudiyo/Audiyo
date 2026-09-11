# Contributing to Audiyo

Thanks for looking. The process is light on purpose.

## Setup

```
git clone https://github.com/TeamAudiyo/Audiyo.git
cd Audiyo
pip install -e ".[all]"
```

Run the fast tests. They do not download the checkpoint:

```
pytest -q
```

Integration tests stay skipped unless you opt in with HF access. Set `AUDIYO_RUN_INTEGRATION=1`, then run:

```
pytest tests/integration_test.py -q
```

Try the CLI without a model download first:

```
audiyo --help
audiyo info
audiyo hardware
audiyo presets
```

## What to work on

Good first areas: better error messages, more dataset checks, benchmark reports from real hardware, docs fixes. Bigger changes (new models, new objectives) need an issue first so scope can be agreed. Version 0.1.0 stays single runnable model on purpose.

## Style

Code files start with code, not header comments, and contain no hash comments. Put explanations in docs or in PLAN.md, not inline. Keep lines around 100 characters. No hype words in docs. If you claim a speedup or saving, attach the benchmark header.

Before opening a pull request:

* `pytest -q` passes.
* New behavior has a test that runs without the checkpoint.
* User facing changes update the docs.
* No tokens, audio, credentials, or personal data in the diff.
* New Python files contain no `#` characters.

## Security

Do not use `eval` or `exec`. Do not add shell calls. Load checkpoints with `weights_only=True`. Never log tokens or user audio paths in full. If you touch error paths, pass them through `scrub_text`. Read SECURITY.md before changing auth, downloads, or file loading.

## Licenses

Audiyo code is Apache-2.0. Anything trained on or derived from the model weights follows the Stability AI Community License. Do not commit weights, adapters trained on private audio you cannot share, or clips you do not have rights to.
