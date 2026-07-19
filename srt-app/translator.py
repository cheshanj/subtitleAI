"""NLLB translation core, independent of any UI.

Holds the model cache, Sinhala post-processing, and a cancellable, progress-
reporting batch translate. Keeping this UI-free means it can be unit-tested and
reused (CLI, tests, a future desktop shell) without importing Gradio.
"""
from __future__ import annotations

import gc
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SOURCE_LANG = "eng_Latn"
TARGET_LANG = "sin_Sinh"

# Phrases that mark a subtitle credit/watermark line rather than dialogue.
# Matched with word boundaries so "synced" doesn't trip on "our watches are
# synced" and ".com" doesn't trip on ordinary "amazon.com" mentions mid-line.
CREDIT_PATTERNS = [
    re.compile(r"\bsubtitles?\b", re.IGNORECASE),
    re.compile(r"\bprovided by\b", re.IGNORECASE),
    re.compile(r"\bre-?synced?\b", re.IGNORECASE),
    re.compile(r"\bedited by\b", re.IGNORECASE),
    re.compile(r"www\.\S+", re.IGNORECASE),
    re.compile(r"\bsync(?:ed|hronized)?\s+by\b", re.IGNORECASE),
]

# Zero-width-joiner fixes for Sinhala conjuncts the tokenizer tends to split.
SINHALA_REPLACEMENTS = {
    "ප් ර": "ප්‍ර",
    "ක් ර": "ක්‍ර",
    "ත් ර": "ත්‍ර",
    "ද් ර": "ද්‍ර",
    "ව් ය": "ව්‍ය",
    "ද් ය": "ද්‍ය",
    "න් ය": "න්‍ය",
    "ල් ය": "ල්‍ය",
    "ම් ය": "ම්‍ය",
    "බ් ර": "බ්‍ර",
    "ෆ් ර": "ෆ්‍ර",
    "ග් ර": "ග්‍ර",
    "ට් ර": "ට්‍ර",
    "ඩ් ර": "ඩ්‍ර",
    "ඉලෙක්ට් රෝ": "ඉලෙක්ට්‍රෝ",
    "මයික් රො": "මයික්‍රො",
}

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]


class TranslationCancelled(Exception):
    """Raised when a caller-supplied cancel check returns True mid-run."""


def should_skip_translation(plain_text: str) -> bool:
    return any(pat.search(plain_text) for pat in CREDIT_PATTERNS)


def fix_sinhala_spacing(text: str) -> str:
    for bad, good in SINHALA_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text


def wrap_subtitle(text: str, width: int = 42) -> str:
    words = text.split()
    if not words:
        return text
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        next_len = len(word) if not current else current_len + 1 + len(word)
        if current and next_len > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = next_len
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def discover_models(search_dirs: Iterable[Path]) -> list[str]:
    """Return paths of folders that look like a saved HF model (have config.json)."""
    found: list[str] = []
    for base in search_dirs:
        base = Path(base)
        if not base.exists():
            continue
        for config in base.glob("**/config.json"):
            model_dir = config.parent
            # Skip intermediate training checkpoints (checkpoint-500, etc.);
            # offer only final/saved model folders.
            if model_dir.name.startswith("checkpoint-"):
                continue
            found.append(str(model_dir.resolve()))
    # De-dupe while keeping order.
    seen: set[str] = set()
    unique = []
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def load_model(model_dir: str):
    """Resolve the path first, then hit the (single-slot) cache.

    Resolving before caching means ``models\\x``, ``models/x`` and the absolute
    path are one key, not three. maxsize=1 keeps only one model resident; when
    the user switches models the previous one is evicted and its GPU memory
    reclaimed, so two big NLLB checkpoints never sit in VRAM at once.
    """
    resolved = str(Path(model_dir).expanduser().resolve())
    if resolved not in _current_key:
        # Switching models: drop the old one and reclaim its memory first.
        _load_model_cached.cache_clear()
        _current_key.clear()
        _current_key.add(resolved)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return _load_model_cached(resolved)


# Tracks which resolved path is currently cached (maxsize=1 holds one model).
_current_key: set[str] = set()


@lru_cache(maxsize=1)
def _load_model_cached(model_dir: str):
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=dtype,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    tokenizer.src_lang = SOURCE_LANG
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)
    return tokenizer, model, forced_bos_token_id


def translate_batches(
    texts: list[str],
    model_dir: str,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
    progress: ProgressFn | None = None,
    cancel: CancelFn | None = None,
) -> list[str]:
    """Translate a flat list of source lines to Sinhala.

    ``progress(fraction, message)`` is called after each batch. ``cancel()`` is
    checked before each batch; returning True raises ``TranslationCancelled``.
    """
    if not texts:
        return []

    tokenizer, model, forced_bos_token_id = load_model(model_dir)
    outputs: list[str] = []
    total = len(texts)

    for start in range(0, total, batch_size):
        if cancel is not None and cancel():
            raise TranslationCancelled()

        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
            )

        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

        if progress is not None:
            done = min(start + batch_size, total)
            progress(done / total, f"Translated {done} / {total} lines")

    return [fix_sinhala_spacing(text.strip()) for text in outputs]
