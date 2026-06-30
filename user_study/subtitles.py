from __future__ import annotations

import html
import re
from pathlib import Path
from typing import List, Sequence, Tuple


SUB_RE = re.compile(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*:\s*(.+)$")
_TRIM_TOKEN_RE = re.compile(r"^\W+|\W+$")


def _normalize_text(text: object) -> str:
    return " ".join(html.unescape(str(text)).split()).strip()


def parse_subtitles(path: Path) -> List[Tuple[float, float, str]]:
    rows: List[Tuple[float, float, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = SUB_RE.match(line.strip())
            if not match:
                continue
            start = float(match.group(1))
            duration = float(match.group(2))
            text = _normalize_text(match.group(3))
            if not text:
                continue
            rows.append((start, start + duration, text))
    return rows


def clip_subtitles(
    subtitle_rows: Sequence[Tuple[float, float, str]],
    start_abs_sec: float,
    end_abs_sec: float,
) -> List[Tuple[float, float, str]]:
    rel: List[Tuple[float, float, str]] = []
    for start, end, text in subtitle_rows:
        if end <= start_abs_sec or start >= end_abs_sec:
            continue
        rel_start = max(0.0, start - start_abs_sec)
        rel_end = min(end_abs_sec - start_abs_sec, end - start_abs_sec)
        if rel_end - rel_start < 1e-3:
            continue
        rel.append((rel_start, rel_end, text))
    rel.sort(key=lambda row: row[0])
    return rel


def _token_keys(text: str) -> List[str]:
    keys: List[str] = []
    for token in text.split():
        cleaned = _TRIM_TOKEN_RE.sub("", token).casefold()
        keys.append(cleaned or token.casefold())
    return keys


def _find_subsequence(needle: Sequence[str], haystack: Sequence[str]) -> int:
    if not needle:
        return 0
    if len(needle) > len(haystack):
        return -1
    stop = len(haystack) - len(needle) + 1
    for idx in range(stop):
        if list(haystack[idx : idx + len(needle)]) == list(needle):
            return idx
    return -1


def _suffix_prefix_overlap(left: Sequence[str], right: Sequence[str]) -> int:
    max_overlap = min(len(left), len(right))
    for size in range(max_overlap, 0, -1):
        if list(left[-size:]) == list(right[:size]):
            return size
    return 0


def _collapse_progressive_fragments(texts: Sequence[str]) -> List[str]:
    phrases: List[str] = []
    phrase_keys: List[List[str]] = []

    for raw_text in texts:
        text = _normalize_text(raw_text)
        if not text:
            continue

        keys = _token_keys(text)
        if not phrases:
            phrases.append(text)
            phrase_keys.append(keys)
            continue

        prev_text = phrases[-1]
        prev_keys = phrase_keys[-1]
        if text.casefold() == prev_text.casefold():
            continue

        if _find_subsequence(prev_keys, keys) >= 0:
            phrases[-1] = text
            phrase_keys[-1] = keys
            continue

        if _find_subsequence(keys, prev_keys) >= 0:
            continue

        overlap = _suffix_prefix_overlap(prev_keys, keys)
        if overlap > 0:
            merged_tokens = prev_text.split() + text.split()[overlap:]
            merged_text = " ".join(merged_tokens)
            phrases[-1] = merged_text
            phrase_keys[-1] = _token_keys(merged_text)
            continue

        phrases.append(text)
        phrase_keys.append(keys)

    return phrases


def _pack_display_lines(texts: Sequence[str], max_chars_per_line: int) -> Tuple[str, str]:
    chunks: List[str] = []
    for text in texts:
        if not text:
            continue
        if chunks:
            merged = f"{chunks[-1]} {text}"
            if len(merged) <= max_chars_per_line:
                chunks[-1] = merged
                continue
        chunks.append(text)

    if not chunks:
        return "", ""
    if len(chunks) == 1:
        return "", chunks[0]
    if len(chunks) == 2:
        return chunks[0], chunks[1]
    return " ".join(chunks[:-1]), chunks[-1]


def _flush_same_start_group(
    group: Sequence[Tuple[float, float, str, int]],
    max_chars_per_line: int,
    grouped: List[Tuple[float, float, str, int]],
) -> None:
    if not group:
        return

    current_text = ""
    current_end = 0.0
    for _, end, text, order_idx in group:
        merged = f"{current_text} {text}".strip() if current_text else text
        if current_text and len(merged) > max_chars_per_line:
            grouped.append((group[0][0], current_end, current_text, order_idx - 1))
            current_text = text
            current_end = end
        else:
            current_text = merged
            current_end = max(current_end, end)

    if current_text:
        grouped.append((group[0][0], current_end, current_text, group[-1][3]))


def build_youtube_caption_states(
    subtitles: Sequence[Tuple[float, float, str]],
    max_chars_per_line: int = 44,
) -> List[Tuple[float, float, str, str]]:
    normalized: List[Tuple[float, float, str, int]] = []
    for idx, (start, end, text) in enumerate(sorted(subtitles, key=lambda row: row[0])):
        cleaned = _normalize_text(text)
        if not cleaned:
            continue
        norm_start = max(0.0, float(start))
        norm_end = max(norm_start, float(end))
        normalized.append((norm_start, norm_end, cleaned, idx))

    if not normalized:
        return []

    grouped: List[Tuple[float, float, str, int]] = []
    same_start_group: List[Tuple[float, float, str, int]] = []
    eps = 1e-6

    for item in normalized:
        if not same_start_group or abs(item[0] - same_start_group[0][0]) <= eps:
            same_start_group.append(item)
            continue
        _flush_same_start_group(same_start_group, max_chars_per_line, grouped)
        same_start_group = [item]

    _flush_same_start_group(same_start_group, max_chars_per_line, grouped)

    boundaries = sorted({time_pt for start, end, _, _ in grouped for time_pt in (start, end)})
    raw_states: List[Tuple[float, float, str, str]] = []

    for start, end in zip(boundaries, boundaries[1:]):
        if end - start < 0.08:
            continue

        active = [
            (caption_start, text, order_idx)
            for caption_start, caption_end, text, order_idx in grouped
            if caption_start <= start + eps and caption_end > start + eps
        ]
        if not active:
            continue

        active.sort(key=lambda row: (row[0], row[2]))
        active_texts = [text for _, text, _ in active]
        top_line, bottom_line = _pack_display_lines(
            _collapse_progressive_fragments(active_texts),
            max_chars_per_line=max_chars_per_line,
        )
        if not top_line and not bottom_line:
            continue
        raw_states.append((start, end, top_line, bottom_line))

    if not raw_states:
        return []

    states: List[Tuple[float, float, str, str]] = [raw_states[0]]
    for start, end, top_line, bottom_line in raw_states[1:]:
        prev_start, prev_end, prev_top, prev_bottom = states[-1]
        if top_line == prev_top and bottom_line == prev_bottom and abs(start - prev_end) <= eps:
            states[-1] = (prev_start, end, prev_top, prev_bottom)
        else:
            states.append((start, end, top_line, bottom_line))

    return states
