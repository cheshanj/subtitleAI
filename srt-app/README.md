# Sinhala Subtitle Translator

Upload an English subtitle file, translate it to Sinhala with your fine-tuned
NLLB model, review and correct the result, then download it — in the same format
you uploaded.

## Features

- **Formats:** `.srt`, `.vtt`, `.ass`/`.ssa` (via `pysubs2`, preserving timing). MicroDVD `.sub` is not supported — it stores frame numbers, not timestamps.
- **Live progress + Stop:** see `312 / 1500 lines` and cancel long jobs.
- **Review & edit:** correct names/idioms in the table before exporting.
- **Model dropdown:** auto-discovers trained checkpoints under `../../models/`.
- **Sinhala cleanup:** conjunct-spacing fixes and line wrapping (toggleable).

## Setup

Copy your trained NLLB model folder somewhere the app can find it, e.g.:

```text
D:\Personal_Projects\Subtitle\models\nllb-sinhala-final
```

Any folder containing a `config.json` under `..\..\models\` is offered in the
dropdown automatically. You can also point at one explicitly:

```bat
set SINHALA_MODEL_DIR=D:\Personal_Projects\Subtitle\models\nllb-sinhala-v4
```

Install dependencies (into the environment you run the app with):

```bash
python -m pip install -r requirements.txt
```

## Run

Double-click `run.bat`, or:

```bash
python app.py
```

A browser tab opens automatically at the local Gradio URL.

## How it works

| File | Responsibility |
|------|----------------|
| `subtitle_io.py` | Load/save subtitle formats; strip tags; preserve timing. |
| `translator.py`  | NLLB model cache, batch translate, cancel, Sinhala fixes. UI-free. |
| `app.py`         | Gradio UI wiring the two together. |

## Notes

- First translation loads the model (cold start of ~10–30s); it's cached after.
- GPU is used automatically when available (fp16), else CPU (fp32).
- Skipped credit/watermark lines keep their original text in the output.

## License & notices

App code: [MIT](../LICENSE) © 2026 cheshanj.

⚠️ The **NLLB-200 model** this app runs is licensed **CC-BY-NC 4.0
(non-commercial)** by Meta AI — commercial use is not permitted, and this applies
to any model fine-tuned from it. Full third-party notices and attribution are in
the [project README](../README.md#third-party-notices--attribution).

