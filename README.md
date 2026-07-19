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

The **code** in this repository is released under the [MIT License](LICENSE).

```
Copyright (c) 2026 cheshanj
```

> **Important — the model is not MIT.** The MIT license covers this repository's
> source code only. It does **not** cover the NLLB-200 model weights, any model
> you fine-tune from them, or the training data. See "Third-party notices" below.

## Third-party notices & attribution

This project builds on the following works. You are responsible for complying
with each of their licenses when you use, fine-tune, or distribute derivatives.

| Component | Author | License | Notes |
|-----------|--------|---------|-------|
| [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) | Meta AI | **CC-BY-NC 4.0** | **Non-commercial only.** Fine-tuned derivatives inherit this restriction. |
| [Transformers](https://github.com/huggingface/transformers) | Hugging Face | Apache-2.0 | |
| [PyTorch](https://pytorch.org/) | PyTorch / Linux Foundation | BSD-3-Clause | |
| [Gradio](https://www.gradio.app/) | Hugging Face | Apache-2.0 | |
| [pysubs2](https://github.com/tkarabela/pysubs2) | Tomáš Karabela | MIT | |
| [SentencePiece](https://github.com/google/sentencepiece) | Google | Apache-2.0 | |

### ⚠️ NLLB-200 non-commercial restriction

NLLB-200 is distributed under **Creative Commons Attribution-NonCommercial 4.0
International (CC-BY-NC 4.0)**. This means:

- ✅ Research, personal, and educational use is permitted.
- ❌ **Commercial use is not permitted** — this includes selling translations,
  paid services, or any revenue-generating deployment built on the model or a
  model fine-tuned from it.
- 📌 A model you fine-tune from NLLB-200 is a derivative work and remains subject
  to CC-BY-NC 4.0. Redistributing it must preserve the same license and
  attribution to Meta AI.

If you need commercial use, replace NLLB-200 with a permissively licensed model.

### Attribution (NLLB)

> NLLB Team et al. *No Language Left Behind: Scaling Human-Centered Machine
> Translation.* Meta AI, 2022. https://arxiv.org/abs/2207.04672

```bibtex
@article{nllb2022,
  title   = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  author  = {{NLLB Team} and Costa-juss{\`a}, Marta R. and others},
  year    = {2022},
  journal = {arXiv preprint arXiv:2207.04672}
}
```

## Disclaimer

This software is provided "as is", without warranty of any kind, express or
implied. Machine translation output may contain errors; review translations
before relying on them. The author is not responsible for how translated
subtitles or trained models are used, nor for any content you translate — you
are responsible for holding the rights to any subtitle files you process and for
complying with all applicable third-party licenses listed above.

