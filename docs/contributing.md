# Contributing

Thanks for looking. This is small and the process is light.

## Setup

```
git clone https://github.com/TeamAudiyo/Audiyo.git
cd Audiyo
pip install -e ".[all]"
```

Run the fast tests:

```
pytest -q
```

They do not download the checkpoint. Integration tests stay skipped unless you opt in. Set `AUDIYO_RUN_INTEGRATION=1`, then run:

```
pytest tests/integration_test.py -q
```

You need HF access and stronger hardware for those.

## What to work on

Good first areas: better error messages, more dataset checks, benchmark reports, docs fixes. Bigger changes need an issue first. Version 0.0.1 stays single model on purpose.

## Style

Code files have no header comments and no hash comments. Put explanations in docs or PLAN.md. Keep lines around 100 characters. No hype words. Claims about speed or savings need a benchmark header.

## Checks before a pull request

* `pytest -q` passes
* New behavior has a test that runs without the checkpoint
* Docs updated if user facing behavior changed
* No tokens, audio, or personal data in the diff

## Licenses

Audiyo code is Apache-2.0. Anything trained on or derived from model weights follows the Stability AI Community License. Do not commit weights, private adapters, or clips you do not have rights to.
