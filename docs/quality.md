# Quality checks

Quick numbers plus your ears. Numbers never replace listening.

## Check one file

```
audiyo quality rain.wav
audiyo quality rain.wav --out metrics.json
```

You get peak, loudness, clipping, silence, brightness, and stereo width. Two hints print automatically: quiet file, and clipping.

## Compare two files

```
audiyo compare before.wav after.wav
```

Use it for full render versus tiled render, or before versus after an adapter. Same seed both times or the diff means nothing.

## Listening sheet

For a folder of renders:

```
audiyo sheet ./clips --out sheet.csv
```

Open the sheet, play each file, fill in rating and notes. That sheet is the real quality record. Keep it with the audio when you share results.
