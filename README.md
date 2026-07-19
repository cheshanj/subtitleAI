# subtitleAI — English → Sinhala Subtitle Translator

Fine-tune [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M)
for English→Sinhala, then translate subtitle files through a local web app:
upload a subtitle → translate → review & edit → download.

---

## Repository layout

| Path | What it is |
|------|-----------|
| [`srt-app/`](srt-app/) | **The translator app** — a local Gradio UI. Upload `.srt`/`.vtt`/`.ass`, translate, edit, download. |
| [`sinhala-finetune/`](sinhala-finetune/) | Training package — `train.py`, `dataset.py`, `metrics.py`. |
| `*.ipynb` | Colab notebooks used to fine-tune the model (latest: `NLLB200_Sinhala_v4_hardened.ipynb`). |
| `clean_pairs_final.tsv` | Parallel training data (**not tracked in git** — see below). |

## Quick start (the app)

```bash
cd srt-app
python -m pip install -r requirements.txt
# put your trained model where the app can find it, e.g. ../models/nllb-sinhala-final
python app.py            # or double-click srt-app/run.bat on Windows
```

A browser tab opens at the local Gradio URL. Full app docs: [`srt-app/README.md`](srt-app/README.md).

## The model

- **Base:** `facebook/nllb-200-distilled-600M`
- **Direction:** `eng_Latn → sin_Sinh`
- **Training:** see the notebooks / `sinhala-finetune/`. Output is a standard
  Hugging Face model folder (`config.json`, tokenizer, weights).

The app auto-discovers any model folder containing a `config.json` under
`models/`, so you can keep several checkpoints (v3, v4, …) side by side and pick
one from the dropdown.

## Data & model files are not in git

`clean_pairs_final.tsv` (~64 MB) and trained model folders are **gitignored** —
data and large binaries don't belong in a code repo. To reproduce:

1. Place your parallel corpus at `clean_pairs_final.tsv` (tab-separated
   `english<TAB>sinhala`).
2. Run the training notebook / `sinhala-finetune/train.py`.
3. Copy the resulting model folder to `models/nllb-sinhala-final/`.

## License

[MIT](LICENSE) © cheshanj
