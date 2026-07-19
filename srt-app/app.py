"""Gradio UI for the Sinhala subtitle translator.

Flow:
  1. Upload a subtitle file (.srt/.vtt/.ass/.ssa/.sub).
  2. Translate  -> shows an editable English -> Sinhala table + a draft file.
  3. Optionally correct rows in the table.
  4. Apply edits & export -> regenerates the download in the original format.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import threading
from pathlib import Path

import gradio as gr

import subtitle_io
import translator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = os.environ.get(
    "SINHALA_MODEL_DIR",
    str(REPO_ROOT / "models" / "nllb-sinhala-final"),
)
# Where to look for trained checkpoints to offer in the dropdown.
MODEL_SEARCH_DIRS = [REPO_ROOT / "models", REPO_ROOT]

# Cooperative cancel. Each translate() run installs its OWN Event under this lock
# so a new run can never clear a still-running previous run's flag (which would
# un-cancel a zombie job). Stop targets whichever Event is currently active.
_cancel_lock = threading.Lock()
_active_cancel: threading.Event | None = None


def _model_choices() -> list[str]:
    found = translator.discover_models(MODEL_SEARCH_DIRS)
    if DEFAULT_MODEL_DIR not in found and Path(DEFAULT_MODEL_DIR).exists():
        found.insert(0, DEFAULT_MODEL_DIR)
    if not found:
        # Still show the expected path so the user sees where it should live.
        found = [DEFAULT_MODEL_DIR]
    return found


def request_cancel():
    with _cancel_lock:
        if _active_cancel is not None:
            _active_cancel.set()
    return gr.update(value="Cancelling after the current batch…")


def translate(
    srt_file,
    model_dir,
    batch_size,
    num_beams,
    max_new_tokens,
    skip_credits,
    wrap_lines,
    fix_spacing,
    progress=gr.Progress(),
):
    global _active_cancel
    if not srt_file:
        raise gr.Error("Upload a subtitle file first.")
    if not model_dir or not Path(model_dir).exists():
        raise gr.Error(
            f"Model folder not found: {model_dir}\n"
            "Copy your trained NLLB model folder there, or pick one from the dropdown."
        )

    # Install this run's own cancel Event so Stop targets exactly this job.
    my_cancel = threading.Event()
    with _cancel_lock:
        _active_cancel = my_cancel

    # Copy the upload into our own temp dir so export can reload it later even
    # after Gradio has cleaned up the original upload file.
    stable_dir = Path(tempfile.mkdtemp(prefix="sinhala_src_"))
    stable_path = stable_dir / Path(srt_file).name
    shutil.copy2(srt_file, stable_path)

    try:
        # load() tries utf-8 then cp1252 internally; ValueError covers a decode
        # failure past both, an unsupported format, or an unparseable file.
        subs = subtitle_io.load(stable_path)
    except ValueError as exc:
        raise gr.Error(str(exc))
    plain = subs.plain_texts()

    # Decide which lines to actually translate.
    to_translate_idx: list[int] = []
    to_translate_txt: list[str] = []
    for i, text in enumerate(plain):
        if not text:
            continue
        if skip_credits and translator.should_skip_translation(text):
            continue
        to_translate_idx.append(i)
        to_translate_txt.append(text)

    progress(0.0, desc="Loading model…")

    def on_progress(frac, msg):
        progress(frac, desc=msg)

    try:
        translated = translator.translate_batches(
            to_translate_txt,
            model_dir=model_dir,
            batch_size=max(1, int(batch_size)),
            num_beams=max(1, int(num_beams)),
            max_new_tokens=max(16, int(max_new_tokens)),
            progress=on_progress,
            cancel=my_cancel.is_set,
        )
    except translator.TranslationCancelled:
        return [], None, "Translation cancelled.", ""

    # Build the editable table: one row per event. Untranslated lines show blank
    # Sinhala so the user can see they were skipped.
    trans_by_idx = dict(zip(to_translate_idx, translated))
    rows = [[plain[i], trans_by_idx.get(i, "")] for i in range(len(subs))]

    # Draft export straight away so there's a download even without edits,
    # honouring the user's wrap/fix choices.
    draft_path = _export(subs, rows, wrap_lines=wrap_lines, fix_spacing=fix_spacing)

    summary = (
        f"Translated {len(to_translate_idx)} lines, "
        f"skipped {len(subs) - len(to_translate_idx)}. "
        f"Format: {subs.fmt}. Edit any row below, then re-export."
    )
    # Stash the loaded file path so export can reload it (state carries the path).
    return rows, draft_path, summary, str(subs.source_path)


def _cell(value) -> str:
    """Coerce a Dataframe cell (str / None / NaN float) to a clean string."""
    if value is None:
        return ""
    if isinstance(value, float):  # NaN from an emptied cell
        return "" if math.isnan(value) else str(value)
    return str(value)


def _export(subs, rows, wrap_lines, fix_spacing):
    texts: list[str] = []
    for row in rows:
        si = _cell(row[1] if len(row) > 1 else "").strip()
        if not si:
            texts.append("")  # leave original line unchanged
            continue
        if fix_spacing:
            si = translator.fix_sinhala_spacing(si)
        if wrap_lines:
            si = translator.wrap_subtitle(si)
        texts.append(si)

    out_dir = Path(tempfile.mkdtemp(prefix="sinhala_srt_"))
    return str(subtitle_io.save(subs, texts, out_dir))


def export_edits(edited_rows, source_path, wrap_lines, fix_spacing):
    if not source_path or not Path(source_path).exists():
        raise gr.Error("Nothing to export yet — translate a file first.")
    if not edited_rows:
        raise gr.Error("The review table is empty — translate a file first.")
    try:
        subs = subtitle_io.load(source_path)
    except ValueError as exc:  # includes UnicodeDecodeError
        raise gr.Error(f"Could not reload the source file: {exc}")
    # With type="array" the Dataframe yields a list of [src, si] lists.
    rows = [[_cell(r[0]), _cell(r[1] if len(r) > 1 else "")] for r in edited_rows]
    if len(rows) != len(subs):
        raise gr.Error(
            "Row count no longer matches the source file — please don't add or "
            "remove rows in the table."
        )
    path = _export(subs, rows, wrap_lines, fix_spacing)
    return path, "Exported with your edits."


THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")


def build_app() -> gr.Blocks:
    import torch  # local import so the module imports even without a GPU stack

    default_batch = 16 if torch.cuda.is_available() else 4
    with gr.Blocks(title="Sinhala Subtitle Translator") as demo:
        source_state = gr.State(value="")

        gr.Markdown("## 🎬 Sinhala Subtitle Translator")
        gr.Markdown(
            "Upload an English subtitle file (`.srt`, `.vtt`, `.ass`), translate "
            "with your fine-tuned NLLB model, review/edit, and download."
        )

        with gr.Row():
            with gr.Column(scale=1):
                srt_file = gr.File(
                    label="Subtitle file",
                    file_types=list(subtitle_io.SUPPORTED_EXTS),
                    type="filepath",
                )
                _choices = _model_choices()
                model_dir = gr.Dropdown(
                    label="Model folder",
                    choices=_choices,
                    value=_choices[0],
                    allow_custom_value=True,
                )
                with gr.Accordion("Advanced settings", open=False):
                    batch_size = gr.Slider(1, 64, value=default_batch, step=1, label="Batch size")
                    num_beams = gr.Slider(1, 8, value=4, step=1, label="Beams")
                    max_new_tokens = gr.Slider(32, 256, value=128, step=8, label="Max new tokens")
                    skip_credits = gr.Checkbox(value=True, label="Skip credit/watermark lines")
                    wrap_lines = gr.Checkbox(value=True, label="Wrap long lines (~42 chars)")
                    fix_spacing = gr.Checkbox(value=True, label="Fix Sinhala conjunct spacing")

                with gr.Row():
                    translate_btn = gr.Button("Translate", variant="primary")
                    cancel_btn = gr.Button("Stop", variant="stop")

            with gr.Column(scale=2):
                status = gr.Textbox(label="Status", interactive=False)
                preview = gr.Dataframe(
                    headers=["English", "Sinhala (editable)"],
                    datatype=["str", "str"],
                    type="array",  # yields list-of-lists, not a pandas DataFrame
                    label="Review & edit (don't add or remove rows)",
                    wrap=True,
                    column_widths=["50%", "50%"],
                )
                with gr.Row():
                    export_btn = gr.Button("Apply edits & export", variant="secondary")
                output_file = gr.File(label="Download translated subtitle")

        translate_btn.click(
            fn=translate,
            inputs=[
                srt_file, model_dir, batch_size, num_beams, max_new_tokens,
                skip_credits, wrap_lines, fix_spacing,
            ],
            outputs=[preview, output_file, status, source_state],
        )
        # No cancels=[...]: translate() runs in a worker thread that Gradio can't
        # kill, so cancellation is cooperative via the per-run Event. Leaving the
        # event alive lets the "Translation cancelled." status reach the UI.
        cancel_btn.click(fn=request_cancel, inputs=None, outputs=[status])
        export_btn.click(
            fn=export_edits,
            inputs=[preview, source_state, wrap_lines, fix_spacing],
            outputs=[output_file, status],
        )

    return demo


if __name__ == "__main__":
    # Gradio 6 takes theme at launch() rather than on Blocks().
    build_app().queue().launch(theme=THEME, inbrowser=True)
