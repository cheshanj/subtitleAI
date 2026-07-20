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
import queue
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


# Human-readable dropdown labels -> internal device ids.
_DEVICE_VALUES = {
    "Auto (use GPU if available)": "auto",
    "CPU": "cpu",
    "GPU (CUDA)": "cuda",
}


def _device_choices_and_hint():
    """Dropdown choices plus a one-line note on what hardware was detected."""
    if translator.gpu_available():
        hint = f"✅ GPU detected: {translator.gpu_name()}"
    else:
        hint = "ℹ️ No CUDA GPU detected — CPU will be used. (GPU option needs a CUDA build of PyTorch.)"
    return list(_DEVICE_VALUES.keys()), hint


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
    device,
):
    global _active_cancel
    if not srt_file:
        raise gr.Error("Upload a subtitle file first.")
    if not model_dir or not Path(model_dir).exists():
        raise gr.Error(
            f"Model folder not found: {model_dir}\n"
            "Copy your trained NLLB model folder there, or pick one from the dropdown."
        )
    # Validate the device up front so "GPU unavailable" is a clean message.
    device_choice = _DEVICE_VALUES.get(device, "auto")
    try:
        translator.resolve_device(device_choice)
    except ValueError as exc:
        raise gr.Error(str(exc))

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

    total = len(to_translate_txt)

    # Show the loading bar immediately (indeterminate while the model loads).
    yield (
        gr.update(),                              # preview (unchanged)
        gr.update(),                              # output_file
        "⏳ Loading model…",                       # status
        gr.update(),                              # source_state
        gr.update(visible=True),                  # empty_state stays until done
        gr.update(interactive=False),             # export_btn
        gr.update(value=_progress_html(0, total, "Loading model…"), visible=True),
    )

    # Run the (blocking) translation in a worker thread and stream progress from
    # a queue so the UI updates live with "N / total lines".
    events: "queue.Queue" = queue.Queue()
    result: dict = {}

    def worker():
        def on_progress(frac, msg):
            events.put(("progress", frac, msg))
        try:
            out = translator.translate_batches(
                to_translate_txt,
                model_dir=model_dir,
                batch_size=max(1, int(batch_size)),
                num_beams=max(1, int(num_beams)),
                max_new_tokens=max(16, int(max_new_tokens)),
                device=device_choice,
                progress=on_progress,
                cancel=my_cancel.is_set,
            )
            result["translated"] = out
        except translator.TranslationCancelled:
            result["cancelled"] = True
        except Exception as exc:  # surface model/runtime errors to the UI
            result["error"] = str(exc)
        finally:
            events.put(("done", None, None))

    thr = threading.Thread(target=worker, daemon=True)
    thr.start()

    # Drain progress events until the worker signals done.
    while True:
        kind, frac, msg = events.get()
        if kind == "done":
            break
        yield (
            gr.update(), gr.update(),
            f"⏳ {msg}", gr.update(),
            gr.update(visible=True), gr.update(interactive=False),
            gr.update(value=_progress_html(frac, total, msg), visible=True),
        )
    thr.join()

    if result.get("cancelled"):
        yield (
            gr.update(value=[], visible=False), None,
            "⏹️ Translation cancelled.", "",
            gr.update(visible=True), gr.update(interactive=False),
            gr.update(visible=False),
        )
        return
    if result.get("error"):
        raise gr.Error(f"Translation failed: {result['error']}")

    translated = result["translated"]

    # Build the editable table: one row per event. Untranslated lines show blank
    # Sinhala so the user can see they were skipped.
    trans_by_idx = dict(zip(to_translate_idx, translated))
    rows = [[plain[i], trans_by_idx.get(i, "")] for i in range(len(subs))]

    # Draft export straight away so there's a download even without edits,
    # honouring the user's wrap/fix choices.
    draft_path = _export(subs, rows, wrap_lines=wrap_lines, fix_spacing=fix_spacing)

    summary = (
        f"✅ Translated {len(to_translate_idx)} lines, "
        f"skipped {len(subs) - len(to_translate_idx)}. "
        f"Format: {subs.fmt}. Edit any row below, then re-export."
    )
    # Stash the loaded file path so export can reload it (state carries the path).
    yield (
        gr.update(value=rows, visible=True),   # preview
        draft_path,                            # output_file
        summary,                               # status
        str(subs.source_path),                 # source_state
        gr.update(visible=False),              # empty_state (hide)
        gr.update(interactive=True),           # export_btn (enable)
        gr.update(visible=False),              # progress bar (hide)
    )


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
    return path, "✅ Exported with your edits."


# ---------------------------------------------------------------------------
# Theme — glassmorphism with an indigo -> purple accent, working in BOTH the
# light and dark palettes. Colours are driven by CSS variables (see CSS below)
# so the viewer's light/dark toggle restyles everything consistently; the theme
# object only sets the accent gradient shared by both modes.
# ---------------------------------------------------------------------------
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    spacing_size="lg",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #a855f7 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4f46e5 0%, #9333ea 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #6366f1 0%, #a855f7 100%)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #4f46e5 0%, #9333ea 100%)",
    button_primary_text_color="#ffffff",
    button_primary_border_color="rgba(255,255,255,0.15)",
)

# ---------------------------------------------------------------------------
# Custom CSS — glass cards, gradient hero, numbered steps, glow buttons.
# ---------------------------------------------------------------------------
CSS = """
/* Headings use a display-ish stack with no external @import — an external font
   import blocks first paint and leaves the loading screen up longer. */

/* ---------- Theme tokens: light defaults, dark overrides ---------- */
.gradio-container {
    --app-bg: radial-gradient(1200px 600px at 15% -10%, #eef2ff 0%, transparent 55%),
              radial-gradient(1000px 500px at 100% 0%, #faf5ff 0%, transparent 50%),
              linear-gradient(180deg, #f8fafc 0%, #eef1f8 100%);
    --card-bg: rgba(255, 255, 255, 0.72);
    --card-border: rgba(15, 23, 42, 0.08);
    --card-border-hover: rgba(99, 102, 241, 0.45);
    --card-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    --card-shadow-hover: 0 14px 40px rgba(99, 102, 241, 0.16);
    --text-strong: #0f172a;
    --text-muted: #64748b;
    --accent-soft-bg: rgba(99, 102, 241, 0.10);
    --accent-text: #4f46e5;
    --blur: 16px;
}
.dark.gradio-container, .dark .gradio-container {
    --app-bg: linear-gradient(160deg, #0b0d17 0%, #14162a 45%, #1b1030 100%);
    --card-bg: rgba(255, 255, 255, 0.045);
    --card-border: rgba(148, 163, 184, 0.14);
    --card-border-hover: rgba(129, 140, 248, 0.4);
    --card-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    --card-shadow-hover: 0 12px 40px rgba(99, 102, 241, 0.22);
    --text-strong: #e2e8f0;
    --text-muted: #94a3b8;
    --accent-soft-bg: rgba(99, 102, 241, 0.18);
    --accent-text: #c7d2fe;
    --blur: 18px;
}

body, .gradio-container { background: var(--app-bg) !important; background-attachment: fixed !important; }
.gradio-container {
    max-width: 1220px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
/* Gradio 6 can left-align the app's inner wrapper; force it centred. */
.gradio-container > .main,
.gradio-container .wrap > .contain,
.gradio-container .fillable { margin-left: auto !important; margin-right: auto !important; width: 100% !important; }

/* Strip Gradio's default chrome from the COLUMN that wraps each glass card,
   so we see only one card, not a card inside a bordered column. */
.gradio-container .column:has(> .glass-card),
.gradio-container div.column:has(> div > .glass-card) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
/* The glass card is itself a gr.Group; make sure the group wrapper is clean. */
.glass-card { overflow: visible !important; }

/* ---------- Hero header ---------- */
.hero { text-align: center; padding: 14px 0 2px; }
.hero h1 {
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; font-weight: 800; font-size: 2.35rem;
    margin: 0 0 8px; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 55%, #d946ef 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { color: var(--text-muted); font-size: 1.02rem; margin: 0; }
.hero .badges { margin-top: 12px; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
.hero .badge {
    font-size: 0.74rem; font-weight: 600; letter-spacing: .01em;
    padding: 4px 11px; border-radius: 999px;
    background: var(--accent-soft-bg); color: var(--accent-text);
    border: 1px solid var(--card-border);
}

/* ---------- Glass cards ---------- */
.glass-card {
    background: var(--card-bg) !important;
    backdrop-filter: blur(var(--blur)); -webkit-backdrop-filter: blur(var(--blur));
    border: 1px solid var(--card-border) !important;
    border-radius: 20px !important;
    box-shadow: var(--card-shadow);
    padding: 22px !important; margin-bottom: 18px;
    transition: border-color .25s ease, box-shadow .25s ease, transform .25s ease;
}
.glass-card:hover {
    border-color: var(--card-border-hover) !important;
    box-shadow: var(--card-shadow-hover);
}
/* Neutralise Gradio's default group/block chrome INSIDE a glass card so we
   don't get a bordered box within a bordered box. */
.glass-card > .form,
.glass-card .block,
.glass-card > div > .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ---------- Step headings ---------- */
.step-heading { display: flex; align-items: baseline; gap: 11px; margin-bottom: 16px; flex-wrap: wrap; }
.step-heading .step-badge { align-self: center; }
.step-heading .step-title { white-space: nowrap; }
.step-heading .step-badge {
    flex: 0 0 auto; width: 30px; height: 30px; border-radius: 999px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; font-weight: 700; font-size: 0.9rem; color: #fff;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.45);
}
.step-heading .step-title {
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; font-weight: 700; font-size: 1.08rem;
    color: var(--text-strong);
}
.step-heading .step-sub { font-size: 0.82rem; color: var(--text-muted); font-weight: 400; margin-left: 4px; }

/* ---------- Buttons ---------- */
.btn-glow, .btn-export, .btn-stop-custom {
    border-radius: 12px !important; font-weight: 600 !important;
    transition: transform .15s ease, box-shadow .15s ease !important;
}
.btn-glow:hover, .btn-export:hover, .btn-stop-custom:hover { transform: translateY(-1px); }
.btn-glow { box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35) !important; }
.btn-glow:hover { box-shadow: 0 10px 28px rgba(99, 102, 241, 0.5) !important; }
.btn-export {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: #fff !important; border: 1px solid rgba(255,255,255,0.15) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3) !important;
}
.btn-export:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    box-shadow: 0 10px 28px rgba(16, 185, 129, 0.45) !important;
}
.btn-stop-custom {
    background: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%) !important;
    color: #fff !important; box-shadow: 0 6px 20px rgba(244, 63, 94, 0.3) !important;
}
.btn-stop-custom:hover { box-shadow: 0 10px 28px rgba(244, 63, 94, 0.45) !important; }
/* Make the disabled export button unmistakably inactive. */
.btn-export:disabled, .btn-export[disabled] {
    background: var(--card-border) !important; color: var(--text-muted) !important;
    box-shadow: none !important; cursor: not-allowed !important; opacity: .75;
}

/* ---------- Status box ---------- */
.status-box textarea {
    border-left: 3px solid #818cf8 !important; font-weight: 500 !important;
    color: var(--text-strong) !important;
}

/* ---------- Review table ---------- */
.review-table table { border-radius: 14px !important; overflow: hidden; }
.review-table thead th {
    background: var(--accent-soft-bg) !important; color: var(--accent-text) !important;
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; font-weight: 600 !important;
}

/* ---------- Compact download box (avoid a huge empty dropzone) ---------- */
.download-box .empty { min-height: 0 !important; padding: 8px 0 !important; }
.download-box [data-testid="block-label"] { margin-bottom: 4px; }

/* ---------- Progress bar ---------- */
.pbar-wrap { margin: 6px 0 2px; }
.pbar-track {
    position: relative; height: 10px; border-radius: 999px; overflow: hidden;
    background: var(--card-border);
}
.pbar-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #a855f7, #d946ef);
    background-size: 200% 100%;
    transition: width .35s ease;
    box-shadow: 0 0 12px rgba(129, 140, 248, 0.55);
}
.pbar-indeterminate { animation: pbar-slide 1.15s ease-in-out infinite; }
@keyframes pbar-slide {
    0%   { margin-left: -35%; }
    100% { margin-left: 100%; }
}
.pbar-label {
    margin-top: 8px; font-size: 0.86rem; font-weight: 600;
    color: var(--accent-text); font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
    display: flex; align-items: center; gap: 8px;
}
.pbar-label::before {
    content: ""; width: 12px; height: 12px; border-radius: 50%;
    border: 2px solid var(--accent-text); border-top-color: transparent;
    animation: pbar-spin 0.7s linear infinite; flex: 0 0 auto;
}
@keyframes pbar-spin { to { transform: rotate(360deg); } }

/* ---------- Empty state ---------- */
.empty-state {
    text-align: center; padding: 30px 16px; color: var(--text-muted);
    border: 1px dashed var(--card-border); border-radius: 14px;
    background: linear-gradient(180deg, var(--accent-soft-bg), transparent);
}
.empty-state .es-icon { font-size: 2.2rem; margin-bottom: 8px; opacity: .9; }
.empty-state .es-title { font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; font-weight: 700; color: var(--text-strong); font-size: 1.02rem; margin-bottom: 4px; }
.empty-state .es-sub { font-size: 0.88rem; line-height: 1.5; }

footer { display: none !important; }
"""


def _step_heading(number: str, title: str, subtitle: str = "") -> str:
    sub_html = f'<span class="step-sub">{subtitle}</span>' if subtitle else ""
    return (
        '<div class="step-heading">'
        f'<span class="step-badge">{number}</span>'
        f'<span class="step-title">{title}</span>'
        f"{sub_html}"
        "</div>"
    )


def _progress_html(frac, total, msg: str) -> str:
    """A determinate progress bar; indeterminate shimmer when frac is 0/None."""
    if frac and frac > 0:
        pct = max(0, min(100, round(frac * 100)))
        bar = f'<div class="pbar-fill" style="width:{pct}%"></div>'
        label = f"{msg} · {pct}%"
    else:
        bar = '<div class="pbar-fill pbar-indeterminate" style="width:35%"></div>'
        label = msg
    return (
        '<div class="pbar-wrap">'
        f'<div class="pbar-track">{bar}</div>'
        f'<div class="pbar-label">{label}</div>'
        "</div>"
    )


EMPTY_STATE_HTML = (
    '<div class="empty-state">'
    '<div class="es-icon">🌐</div>'
    '<div class="es-title">No translation yet</div>'
    '<div class="es-sub">Upload a subtitle file and press <b>Translate</b>.<br>'
    "Your English → Sinhala lines will appear here, ready to review and edit.</div>"
    "</div>"
)


def build_app() -> gr.Blocks:
    import torch  # local import so the module imports even without a GPU stack

    default_batch = 16 if torch.cuda.is_available() else 4
    with gr.Blocks(title="Sinhala Subtitle Translator") as demo:
        source_state = gr.State(value="")

        gr.HTML(
            '<div class="hero"><h1>🎬 Sinhala Subtitle Translator</h1>'
            "<p>Upload an English subtitle file, translate with your fine-tuned "
            "NLLB model, review, and download.</p>"
            '<div class="badges">'
            '<span class="badge">.srt · .vtt · .ass</span>'
            '<span class="badge">NLLB-200</span>'
            '<span class="badge">Editable review</span>'
            "</div></div>"
        )

        with gr.Row(equal_height=True):
            # ---- Left: the input flow (upload -> settings -> translate) ----
            with gr.Column(scale=2):
                with gr.Group(elem_classes=["glass-card"]):
                    gr.HTML(_step_heading("1", "Upload &amp; model", "pick your file and checkpoint"))
                    srt_file = gr.File(
                        label="Subtitle file",
                        file_types=list(subtitle_io.SUPPORTED_EXTS),
                        type="filepath",
                        elem_classes=["upload-zone"],
                    )
                    _choices = _model_choices()
                    model_dir = gr.Dropdown(
                        label="Model folder",
                        choices=_choices,
                        value=_choices[0],
                        allow_custom_value=True,
                    )
                    _dev_choices, _dev_hint = _device_choices_and_hint()
                    device = gr.Dropdown(
                        label="Device",
                        choices=_dev_choices,
                        value=_dev_choices[0],
                        info=_dev_hint,
                    )

                    gr.HTML(_step_heading("2", "Settings", "tune quality vs. speed"))
                    with gr.Accordion("Advanced settings", open=False):
                        batch_size = gr.Slider(1, 64, value=default_batch, step=1, label="Batch size")
                        num_beams = gr.Slider(1, 8, value=4, step=1, label="Beams")
                        max_new_tokens = gr.Slider(32, 256, value=128, step=8, label="Max new tokens")
                        skip_credits = gr.Checkbox(value=True, label="Skip credit/watermark lines")
                        wrap_lines = gr.Checkbox(value=True, label="Wrap long lines (~42 chars)")
                        fix_spacing = gr.Checkbox(value=True, label="Fix Sinhala conjunct spacing")

                    with gr.Row():
                        translate_btn = gr.Button("Translate", variant="primary", scale=3, elem_classes=["btn-glow"])
                        cancel_btn = gr.Button("Stop", variant="stop", scale=1, elem_classes=["btn-stop-custom"])

            # ---- Right: the output flow (review/edit -> download) ----
            with gr.Column(scale=3):
                with gr.Group(elem_classes=["glass-card"]):
                    gr.HTML(_step_heading("3", "Review &amp; edit", "correct any line, then export"))
                    status = gr.Textbox(
                        label="Status",
                        value="Ready — upload a file and press Translate.",
                        interactive=False,
                        elem_classes=["status-box"],
                    )
                    progress_bar = gr.HTML(visible=False)
                    empty_state = gr.HTML(EMPTY_STATE_HTML)
                    preview = gr.Dataframe(
                        headers=["English", "Sinhala (editable)"],
                        datatype=["str", "str"],
                        type="array",  # yields list-of-lists, not a pandas DataFrame
                        label="Review & edit (don't add or remove rows)",
                        wrap=True,
                        column_widths=["50%", "50%"],
                        elem_classes=["review-table"],
                        visible=False,
                    )

                    gr.HTML(_step_heading("4", "Export &amp; download", "grab your finished subtitle"))
                    export_btn = gr.Button(
                        "Apply edits & export",
                        variant="secondary",
                        elem_classes=["btn-export"],
                        interactive=False,
                    )
                    output_file = gr.File(
                        label="Download translated subtitle",
                        elem_classes=["download-box"],
                    )

        translate_btn.click(
            fn=translate,
            inputs=[
                srt_file, model_dir, batch_size, num_beams, max_new_tokens,
                skip_credits, wrap_lines, fix_spacing, device,
            ],
            outputs=[preview, output_file, status, source_state, empty_state, export_btn, progress_bar],
            # Hide Gradio's default per-component spinner/timer overlay so only
            # our custom progress bar (progress_bar) is shown during the run.
            show_progress="hidden",
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
    # Gradio 6 takes theme/css at launch() rather than on Blocks(). No forced
    # theme: the app styles both light and dark and follows the viewer's toggle.
    build_app().queue().launch(theme=THEME, css=CSS, inbrowser=True)
