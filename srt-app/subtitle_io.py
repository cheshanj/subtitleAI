"""Subtitle file loading/saving across formats (srt, vtt, ass/ssa).

Wraps ``pysubs2`` so the rest of the app can treat every supported format the
same way: a list of plain-text lines to translate, plus the ability to write the
translations back while preserving timing and (for ASS) styling.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pysubs2

# Formats pysubs2 can round-trip and that make sense for this tool.
# MicroDVD (.sub) is intentionally excluded: it stores frames, not timestamps,
# and needs an fps the app can't infer, so pysubs2 raises on load.
SUPPORTED_EXTS = (".srt", ".vtt", ".ass", ".ssa")

TAG_RE = re.compile(r"<[^>]+>")
# ASS inline override blocks, e.g. {\i1}, {\pos(…)} — keep them out of the text
# we send to the model, but we don't try to re-insert them per word.
ASS_TAG_RE = re.compile(r"\{[^}]*\}")


def strip_tags(text: str) -> str:
    """Return display text with markup and hard line breaks removed."""
    text = ASS_TAG_RE.sub("", text)
    text = TAG_RE.sub("", text)
    # pysubs2 uses \N for a hard break and \n as a soft one; normalise both.
    text = text.replace(r"\N", " ").replace(r"\n", " ")
    return text.replace("\n", " ").strip()


@dataclass
class Subtitles:
    """A loaded subtitle file plus the format it came from."""

    ssa: pysubs2.SSAFile
    fmt: str  # pysubs2 format id, e.g. "srt", "ass", "vtt"
    source_path: Path

    def plain_texts(self) -> list[str]:
        # Drawing commands (ASS vector graphics) and Comment events are not
        # dialogue; return "" so they're treated like skipped lines and left
        # untouched, while keeping index alignment with self.ssa.events.
        out: list[str] = []
        for ev in self.ssa.events:
            if getattr(ev, "is_drawing", False) or getattr(ev, "is_comment", False):
                out.append("")
            else:
                out.append(strip_tags(ev.text))
        return out

    def __len__(self) -> int:
        return len(self.ssa.events)


def load(path: str | Path) -> Subtitles:
    source_path = Path(path)
    ext = source_path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported subtitle format '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_EXTS)}"
        )
    # Prefer UTF-8; many scene-released subs are cp1252/latin-1, so fall back
    # rather than crashing on a stray non-UTF-8 byte.
    try:
        try:
            ssa = pysubs2.load(str(source_path), encoding="utf-8")
        except UnicodeDecodeError:
            ssa = pysubs2.load(str(source_path), encoding="cp1252")
    except pysubs2.exceptions.Pysubs2Error as exc:
        # Unparseable/unrecognised content, MicroDVD without fps, etc. Raised
        # from either load attempt, so both are wrapped.
        raise ValueError(f"Could not parse subtitle file: {exc}") from exc
    return Subtitles(ssa=ssa, fmt=ssa.format or ext.lstrip("."), source_path=source_path)


def save(subs: Subtitles, texts: list[str], out_dir: Path, suffix: str = "si") -> Path:
    """Write ``texts`` back into the events and save in the original format.

    ``texts`` must align 1:1 with ``subs.ssa.events``; an empty string leaves the
    original line unchanged (used for skipped credit lines).
    """
    if len(texts) != len(subs.ssa.events):
        raise ValueError(
            f"Text count ({len(texts)}) does not match event count "
            f"({len(subs.ssa.events)})."
        )

    for event, new_text in zip(subs.ssa.events, texts):
        if new_text:
            # Preserve hard line breaks as the format's own break token.
            event.text = new_text.replace("\n", r"\N")

    out_dir.mkdir(parents=True, exist_ok=True)
    ext = subs.source_path.suffix.lower()
    out_path = out_dir / f"{subs.source_path.stem}.{suffix}{ext}"
    subs.ssa.save(str(out_path), format_=subs.fmt, encoding="utf-8")
    return out_path
