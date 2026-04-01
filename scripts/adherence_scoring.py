"""
Adherence Scoring Module

Provides three scoring tiers for evaluating lyric adherence:
1. Whisper Transcription Score - ASR transcription diffed against input lyrics
2. LRC Timing Score - timestamp analysis for skipped/compressed lines
3. Combined Adherence Score - weighted composite
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Section marker patterns ──────────────────────────────────────────────
_SECTION_RE = re.compile(
    r"^\[(?:Intro|Verse|Chorus|Bridge|Outro|Interlude|Pre-Chorus|Post-Chorus|Hook|Break|Instrumental|Refrain)"
    r"(?:\s*\d*)?\]\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_BRACKET_LINE_RE = re.compile(r"^\[.*?\]\s*$", re.MULTILINE)
_DIALOGUE_RE = re.compile(r'^".*?"$', re.MULTILINE)


@dataclass
class LineDiff:
    """Per-line diff result."""

    input_line: str
    status: str  # "matched", "missing", "partial", "repeated"
    matched_text: str = ""
    similarity: float = 0.0


@dataclass
class WhisperResult:
    """Result of Whisper transcription scoring."""

    raw_transcription: str
    cleaned_transcription: str
    cleaned_input: str
    overall_similarity: float
    line_diffs: List[LineDiff] = field(default_factory=list)
    missing_count: int = 0
    matched_count: int = 0
    partial_count: int = 0
    total_input_lines: int = 0


@dataclass
class LrcTimingResult:
    """Result of LRC timestamp analysis."""

    score: float
    total_lines: int
    healthy_lines: int
    skipped_lines: int  # duration < 1.0s
    compressed_lines: int  # overlapping with next line
    details: List[dict] = field(default_factory=list)


@dataclass
class AdherenceResult:
    """Combined adherence scoring result."""

    whisper: Optional[WhisperResult]
    lrc: Optional[LrcTimingResult]
    dit_score: float
    pmi_score: float
    combined_score: float
    whisper_score: float
    lrc_score: float


# ── Text Normalization ────────────────────────────────────────────────────


def normalize_lyrics(text: str, strip_sections: bool = True, strip_dialogue: bool = True) -> str:
    """Normalize lyrics text for comparison.

    Strips section markers, dialogue lines, punctuation, and normalises whitespace.
    """
    if not text:
        return ""

    result = text

    # Strip section markers like [Verse 1], [Chorus], etc.
    if strip_sections:
        result = _BRACKET_LINE_RE.sub("", result)

    # Strip dialogue lines (in quotes)
    if strip_dialogue:
        result = _DIALOGUE_RE.sub("", result)

    # Remove remaining punctuation except apostrophes (important for contractions)
    result = re.sub(r"[^\w\s']", " ", result)

    # Collapse whitespace and lowercase
    result = re.sub(r"\s+", " ", result).strip().lower()

    return result


def extract_lyric_lines(text: str) -> List[str]:
    """Extract meaningful lyric lines from raw lyrics text.

    Strips section markers, empty lines, and dialogue.
    Returns a list of normalised non-empty lines.
    """
    lines = []
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # Skip section markers
        if _SECTION_RE.match(stripped):
            continue
        if _BRACKET_LINE_RE.match(stripped):
            continue
        # Skip dialogue lines
        if _DIALOGUE_RE.match(stripped):
            continue
        # Normalize the line
        normalized = re.sub(r"[^\w\s']", " ", stripped)
        normalized = re.sub(r"\s+", " ", normalized).strip().lower()
        if normalized:
            lines.append(normalized)
    return lines


# ── Whisper Transcription Scoring ─────────────────────────────────────────


class WhisperScorer:
    """Manages Whisper model and transcription scoring."""

    def __init__(self, model_size: str = "small", device: str = "cuda", compute_type: str = "float16"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _ensure_model(self):
        """Lazy-load the Whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel

            print(f"[Whisper] Loading model '{self.model_size}' on {self.device} ({self.compute_type})...")
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            print("[Whisper] Model loaded.")

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """Transcribe an audio file and return the raw text."""
        self._ensure_model()
        segments, _info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            word_timestamps=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        # Collect all segment texts
        texts = []
        for segment in segments:
            texts.append(segment.text.strip())
        return " ".join(texts)

    def score(self, audio_path: str, input_lyrics: str, language: str = "en") -> WhisperResult:
        """Transcribe audio and compare against input lyrics."""
        raw_transcription = self.transcribe(audio_path, language)

        # Normalize both texts
        cleaned_transcription = normalize_lyrics(raw_transcription, strip_sections=False, strip_dialogue=False)
        cleaned_input = normalize_lyrics(input_lyrics, strip_sections=True, strip_dialogue=True)

        # Overall similarity
        overall_sim = difflib.SequenceMatcher(None, cleaned_input, cleaned_transcription).ratio()

        # Per-line analysis
        input_lines = extract_lyric_lines(input_lyrics)
        line_diffs = []
        missing_count = 0
        matched_count = 0
        partial_count = 0

        for input_line in input_lines:
            # Find best match in transcription
            best_sim = 0.0
            best_match = ""

            # Sliding window over transcription words
            trans_words = cleaned_transcription.split()
            line_words = input_line.split()
            window_size = len(line_words)

            if window_size == 0:
                continue

            for i in range(max(1, len(trans_words) - window_size + 1)):
                window = " ".join(trans_words[i : i + window_size + 2])  # slightly wider window
                sim = difflib.SequenceMatcher(None, input_line, window).ratio()
                if sim > best_sim:
                    best_sim = sim
                    best_match = window

            if best_sim >= 0.7:
                status = "matched"
                matched_count += 1
            elif best_sim >= 0.4:
                status = "partial"
                partial_count += 1
            else:
                status = "missing"
                missing_count += 1

            line_diffs.append(
                LineDiff(
                    input_line=input_line,
                    status=status,
                    matched_text=best_match,
                    similarity=best_sim,
                )
            )

        return WhisperResult(
            raw_transcription=raw_transcription,
            cleaned_transcription=cleaned_transcription,
            cleaned_input=cleaned_input,
            overall_similarity=overall_sim,
            line_diffs=line_diffs,
            missing_count=missing_count,
            matched_count=matched_count,
            partial_count=partial_count,
            total_input_lines=len(input_lines),
        )

    def unload(self):
        """Unload the Whisper model to free VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[Whisper] Model unloaded.")


# ── LRC Timing Scoring ────────────────────────────────────────────────────


def parse_lrc_lines(lrc_text: str) -> List[dict]:
    """Parse LRC text into timestamped lines.

    Returns list of dicts with keys: timestamp_s, text, is_section_marker.
    """
    lrc_re = re.compile(r"\[(\d{2}):(\d{2}\.\d{2})\](.*)")
    lines = []
    for raw_line in lrc_text.strip().split("\n"):
        m = lrc_re.match(raw_line.strip())
        if not m:
            continue
        minutes = int(m.group(1))
        seconds = float(m.group(2))
        text = m.group(3).strip()
        timestamp_s = minutes * 60 + seconds
        is_marker = bool(_BRACKET_LINE_RE.match(f"[{text}]")) if text.startswith("[") else False
        lines.append(
            {
                "timestamp_s": timestamp_s,
                "text": text,
                "is_section_marker": is_marker or text.startswith("["),
            }
        )
    return lines


def score_lrc_timing(lrc_text: str, total_duration: float = 0.0) -> LrcTimingResult:
    """Analyze LRC timestamps for timing anomalies.

    Flags lines with:
    - Duration < 1.0s as "likely skipped"
    - Overlapping timestamps as "compressed"
    """
    lines = parse_lrc_lines(lrc_text)

    if not lines:
        return LrcTimingResult(score=0.0, total_lines=0, healthy_lines=0, skipped_lines=0, compressed_lines=0)

    # Filter to content lines only (skip section markers)
    content_lines = [l for l in lines if not l["is_section_marker"]]

    if not content_lines:
        return LrcTimingResult(score=0.0, total_lines=0, healthy_lines=0, skipped_lines=0, compressed_lines=0)

    skipped = 0
    compressed = 0
    healthy = 0
    details = []

    for i, line in enumerate(content_lines):
        # Calculate duration to next line
        if i < len(content_lines) - 1:
            duration = content_lines[i + 1]["timestamp_s"] - line["timestamp_s"]
        elif total_duration > 0:
            duration = total_duration - line["timestamp_s"]
        else:
            duration = 5.0  # assume healthy for last line

        is_skipped = duration < 1.0
        is_compressed = duration < 0.5

        if is_skipped:
            skipped += 1
            status = "skipped"
        elif is_compressed:
            compressed += 1
            status = "compressed"
        else:
            healthy += 1
            status = "healthy"

        details.append(
            {
                "text": line["text"],
                "timestamp_s": line["timestamp_s"],
                "duration_s": round(duration, 2),
                "status": status,
            }
        )

    total = len(content_lines)
    score = healthy / total if total > 0 else 0.0

    return LrcTimingResult(
        score=round(score, 4),
        total_lines=total,
        healthy_lines=healthy,
        skipped_lines=skipped,
        compressed_lines=compressed,
        details=details,
    )


# ── Combined Scoring ──────────────────────────────────────────────────────


def compute_adherence_score(
    whisper_result: Optional[WhisperResult] = None,
    lrc_result: Optional[LrcTimingResult] = None,
    dit_score: float = 0.0,
    pmi_score: float = 0.0,
) -> AdherenceResult:
    """Compute weighted combined adherence score.

    Weights:
    - 40% Whisper transcription match
    - 35% DiT alignment score
    - 25% LRC timing score
    """
    whisper_score = whisper_result.overall_similarity if whisper_result else 0.0
    lrc_score = lrc_result.score if lrc_result else 0.0

    combined = 0.40 * whisper_score + 0.35 * dit_score + 0.25 * lrc_score

    return AdherenceResult(
        whisper=whisper_result,
        lrc=lrc_result,
        dit_score=dit_score,
        pmi_score=pmi_score,
        combined_score=round(combined, 4),
        whisper_score=round(whisper_score, 4),
        lrc_score=round(lrc_score, 4),
    )
