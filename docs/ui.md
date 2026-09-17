# Simple interface

Three ways to skip flags.

## Web demo

```
pip install audiyo[ui]
audiyo ui
```

Opens a page with prompt, checkpoint, memory mode, length, seed, and a few tidy-up boxes. Press generate, listen, keep what you like. Add --share for a public link.

## Config files

```
audiyo init-config --out mysound.json
audiyo generate --config mysound.json
```

Edit the JSON in any editor. Flags still override the file.

## Size check

Before loading a big checkpoint:

```
audiyo estimate --checkpoint MiniMaxAI/MiniMax-Music3 --memory-mode balanced
audiyo estimate --checkpoint stabilityai/stable-audio-open-1.0 --memory-mode low --vram 8
```

It prints an estimate and a plain verdict: fits, tight, or unlikely. Treat it as guidance, not a promise. Length, steps, and drivers move the real number.
