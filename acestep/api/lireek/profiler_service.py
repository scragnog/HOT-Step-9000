"""
Enhanced lyrics profiling service.

Takes a collection of song lyrics and produces a richly detailed LyricsProfile
by combining rule-based analysis with LLM-assisted deep analysis.

Rule-based analysis extracts:
- Phonetic rhyme schemes (via CMU Pronouncing Dictionary)
- Song structure blueprints (V-C-V-C-B-C patterns)
- Perspective / voice analysis (pronoun dominance)
- Syllable & meter heuristics (syllables per line, variance)
- Vocabulary fingerprinting (type-token ratio, contractions, distinctive words)
- Average verse / chorus line counts
- Representative lyric excerpts for generation

LLM-assisted analysis extracts:
- Themes, subjects, imagery patterns, narrative techniques
- Tone, mood, emotional arcs
- Vocabulary nuance the rules can't capture
- Signature phrases, verbal tics, word pairings
"""
import re
import json
import math
import logging
from collections import Counter
from typing import Optional

import nltk
from nltk.corpus import cmudict

from acestep.api.lireek.schemas import LyricsProfile, SongLyrics

logger = logging.getLogger(__name__)


# ── Robust JSON extraction ────────────────────────────────────────────────────

def _repair_json(text: str) -> str:
    """Attempt to fix common LLM JSON errors."""
    # Fix missing commas between string array elements:
    #   "line one"
    #   "line two"   <-- missing comma after first element
    text = re.sub(r'"\s*\n(\s*")', r'",\n\1', text)
    # Fix missing commas after ] when followed by " (next key)
    text = re.sub(r'\]\s*\n(\s*")', r'],\n\1', text)
    # Fix stray } after ] that prematurely closes the object:
    #   "key": [...]
    #   },              <-- stray, should just be a comma
    #   "next_key": ... 
    text = re.sub(r'\]\s*\n\s*},?\s*\n(\s*")', r'],\n\1', text)
    # Fix missing commas between } and " or { (object boundaries)
    text = re.sub(r'}\s*\n(\s*["{])', r'},\n\1', text)
    # Fix trailing commas before ] or }
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    return text


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from LLM output, tolerating preamble/postamble.

    Tries these strategies in order:
    0. Strip <think>...</think> blocks from reasoning models
    1. Direct json.loads (model returned pure JSON)
    2. Strip markdown code fences, then json.loads
    3. Find the first '{' and last '}' and parse the substring
    4. Repair common JSON errors (missing commas, stray braces) and retry
    5. Brute-force: progressively remove problematic lines and retry
    """
    # Strategy 0: strip reasoning model <think> blocks
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Also handle unclosed <think> tags (model cut off mid-thought)
    if "<think>" in stripped:
        stripped = re.sub(r"<think>.*", "", stripped, flags=re.DOTALL).strip()
    # Strip LM Studio GGUF channel-based thinking tokens:
    #   <|channel>thought ... <channel|>
    stripped = re.sub(r"<\|channel>thought.*?<channel\|>", "", stripped, flags=re.DOTALL).strip()
    if "<|channel>thought" in stripped:
        stripped = re.sub(r"<\|channel>thought.*", "", stripped, flags=re.DOTALL).strip()

    # Strategy 1: direct parse
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: strip code fences
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.MULTILINE).strip()
    try:
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 3: find outermost { ... }
    first_brace = clean.find("{")
    last_brace = clean.rfind("}")
    candidate = None
    if first_brace != -1 and last_brace > first_brace:
        candidate = clean[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 4: repair common JSON errors and retry
    to_repair = candidate or clean
    repaired = _repair_json(to_repair)
    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 5: brute-force — remove lines that are just "}" or "}," 
    # (stray premature closes) and retry
    lines = repaired.splitlines()
    for attempt in range(3):  # max 3 removals
        found = False
        for i, line in enumerate(lines):
            s = line.strip()
            if s in ("}", "},") and i > 0 and i < len(lines) - 1:
                # Check if removing this line lets us parse
                trial = "\n".join(lines[:i] + lines[i+1:])
                try:
                    return json.loads(trial)
                except (json.JSONDecodeError, ValueError):
                    # Remove this line and keep trying
                    lines = lines[:i] + lines[i+1:]
                    found = True
                    break
        if not found:
            break

    logger.warning("All JSON extraction strategies failed for response (%d chars)", len(text))
    return None

# ── CMU Pronouncing Dictionary ────────────────────────────────────────────────

try:
    _CMU_DICT = cmudict.dict()
except LookupError:
    nltk.download("cmudict", quiet=True)
    _CMU_DICT = cmudict.dict()


def _get_phones(word: str) -> Optional[list[str]]:
    """Get the phonetic representation of a word from CMU dict."""
    entries = _CMU_DICT.get(word.lower().strip("'\".,!?;:"))
    return entries[0] if entries else None


def _get_vowel_tail(phones: list[str], n: int = 3) -> tuple[str, ...]:
    """Get the last N vowel+consonant phones for rhyme comparison."""
    # Vowels have stress markers (digits), consonants don't
    result = []
    for p in reversed(phones):
        # Strip stress marker for comparison
        clean = re.sub(r"\d", "", p)
        result.append(clean)
        if len(result) >= n:
            break
    return tuple(reversed(result))


def _rhyme_quality(word_a: str, word_b: str) -> str:
    """Classify rhyme quality between two words.

    Returns: 'perfect', 'slant', 'assonance', or 'none'.
    """
    if word_a == word_b:
        return "perfect"  # identical words

    phones_a = _get_phones(word_a)
    phones_b = _get_phones(word_b)

    if not phones_a or not phones_b:
        # Fallback: last 2 chars matching
        a, b = word_a.lower(), word_b.lower()
        if len(a) >= 2 and len(b) >= 2 and a[-2:] == b[-2:]:
            return "slant"
        return "none"

    tail_a = _get_vowel_tail(phones_a, 3)
    tail_b = _get_vowel_tail(phones_b, 3)

    if tail_a == tail_b:
        return "perfect"

    # Check 2-phone tail
    tail_a2 = _get_vowel_tail(phones_a, 2)
    tail_b2 = _get_vowel_tail(phones_b, 2)
    if tail_a2 == tail_b2:
        return "perfect"

    # Assonance: same vowel sounds
    vowels_a = [re.sub(r"\d", "", p) for p in phones_a if re.search(r"\d", p)]
    vowels_b = [re.sub(r"\d", "", p) for p in phones_b if re.search(r"\d", p)]
    if vowels_a and vowels_b and vowels_a[-1] == vowels_b[-1]:
        # Same final vowel
        if tail_a2[:1] == tail_b2[:1] or len(set(tail_a) & set(tail_b)) >= 1:
            return "slant"
        return "assonance"

    # Last resort: any shared tail phones
    if len(set(tail_a) & set(tail_b)) >= 2:
        return "slant"

    return "none"


def _count_syllables_cmu(word: str) -> Optional[int]:
    """Count syllables using CMU dict (vowel phones have digits)."""
    phones = _get_phones(word)
    if not phones:
        return None
    return sum(1 for p in phones if re.search(r"\d", p))


def _count_syllables_heuristic(word: str) -> int:
    """Fallback syllable count using a simple heuristic."""
    word = word.lower().strip("'\".,!?;:-")
    if not word:
        return 0
    # Count vowel groups
    count = len(re.findall(r"[aeiouy]+", word))
    # Subtract silent e
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _count_syllables(word: str) -> int:
    """Count syllables using CMU dict with heuristic fallback."""
    cmu = _count_syllables_cmu(word)
    return cmu if cmu is not None else _count_syllables_heuristic(word)


# ── Section parsing ───────────────────────────────────────────────────────────

_SECTION_HEADER_RE = re.compile(r"^\[(.+?)\]$", re.IGNORECASE)

_SECTION_LABEL_MAP = {
    "verse": "V",
    "chorus": "C",
    "hook": "C",
    "bridge": "B",
    "pre-chorus": "PC",
    "prechorus": "PC",
    "post-chorus": "POC",
    "intro": "I",
    "outro": "O",
    "interlude": "IL",
    "refrain": "C",
    # Extended mappings for common Genius variants
    "breakdown": "B",
    "solo": "IL",
    "instrumental": "IL",
    "spoken": "V",
    "spoken word": "V",
    "rap": "V",
    "rap verse": "V",
    "drop": "C",
    "buildup": "PC",
    "build-up": "PC",
    "build up": "PC",
    "ad-lib": "IL",
    "adlib": "IL",
    "skit": "IL",
    "tag": "O",
}


def _normalise_section_label(raw_label: str) -> str:
    """Convert a section header like 'Verse 1' to a short code like 'V'."""
    lower = raw_label.lower().strip()
    for key, code in _SECTION_LABEL_MAP.items():
        if key in lower:
            return code
    # Unknown section — map to Interlude rather than mysterious "X"
    logger.debug("Unknown section label '%s' — mapping to Interlude", raw_label)
    return "IL"


def _split_into_sections(lyrics: str) -> list[tuple[str, list[str]]]:
    """Split lyrics into labelled sections.

    Returns a list of (label_code, lines) tuples, e.g.:
    [('V', ['line1', 'line2']), ('C', ['line3', 'line4']), ...]
    """
    sections: list[tuple[str, list[str]]] = []
    current_label = "X"
    current_lines: list[str] = []

    for line in lyrics.splitlines():
        stripped = line.strip()
        header_match = _SECTION_HEADER_RE.match(stripped)
        if header_match:
            if current_lines:
                sections.append((current_label, current_lines))
                current_lines = []
            current_label = _normalise_section_label(header_match.group(1))
        elif stripped == "":
            if current_lines:
                sections.append((current_label, current_lines))
                current_lines = []
        else:
            current_lines.append(stripped)

    if current_lines:
        sections.append((current_label, current_lines))

    return sections


def _sections_by_type(sections: list[tuple[str, list[str]]]) -> dict[str, list[list[str]]]:
    """Group sections by type code."""
    grouped: dict[str, list[list[str]]] = {}
    for label, lines in sections:
        grouped.setdefault(label, []).append(lines)
    return grouped


# ── Analysis functions ────────────────────────────────────────────────────────

def _get_last_word(line: str) -> str:
    """Extract the last meaningful word from a line."""
    words = re.findall(r"[a-zA-Z']+", line)
    return words[-1].lower() if words else ""


def _detect_rhyme_scheme(section_lines: list[str]) -> tuple[str, dict[str, int]]:
    """Detect rhyme scheme with phonetic matching.

    Returns (scheme_string, rhyme_quality_counts).
    """
    lines = section_lines[:8]
    endings = [_get_last_word(l) for l in lines]

    mapping: dict[str, str] = {}  # word -> letter
    letter_idx = 0
    scheme = []
    quality_counts: dict[str, int] = {"perfect": 0, "slant": 0, "assonance": 0}

    for word in endings:
        if not word:
            scheme.append("X")
            continue
        matched = None
        best_quality = "none"
        for existing_word, letter in mapping.items():
            q = _rhyme_quality(word, existing_word)
            if q in ("perfect", "slant", "assonance"):
                if matched is None or (q == "perfect"):
                    matched = letter
                    best_quality = q
                    if q == "perfect":
                        break
        if matched:
            scheme.append(matched)
            quality_counts[best_quality] += 1
        else:
            new_letter = chr(ord("A") + min(letter_idx, 25))
            mapping[word] = new_letter
            scheme.append(new_letter)
            letter_idx += 1

    return "".join(scheme), quality_counts


def _analyse_rhymes(all_songs: list[SongLyrics]) -> tuple[list[str], dict[str, int]]:
    """Analyse rhyme schemes and quality across all songs.

    Returns (top_schemes, aggregate_quality_counts).
    """
    scheme_counter: Counter = Counter()
    total_quality: dict[str, int] = {"perfect": 0, "slant": 0, "assonance": 0}

    for song in all_songs:
        sections = _split_into_sections(song.lyrics)
        for label, lines in sections:
            if label in ("V", "C") and len(lines) >= 2:
                scheme, quality = _detect_rhyme_scheme(lines)
                scheme_counter[scheme] += 1
                for k, v in quality.items():
                    total_quality[k] += v

    return [s for s, _ in scheme_counter.most_common(5)], total_quality


def _analyse_structure(all_songs: list[SongLyrics]) -> tuple[float, float, list[str]]:
    """Analyse verse/chorus lengths and song structure blueprints.

    Returns (avg_verse_lines, avg_chorus_lines, top_blueprints).
    """
    verse_sections: list[list[str]] = []
    chorus_sections: list[list[str]] = []
    blueprints: list[str] = []

    for song in all_songs:
        sections = _split_into_sections(song.lyrics)
        grouped = _sections_by_type(sections)
        verse_sections.extend(grouped.get("V", []))
        chorus_sections.extend(grouped.get("C", []))

        # Build blueprint: V-C-V-C-B-C
        if sections:
            # Deduplicate consecutive labels
            labels = []
            for label, _ in sections:
                if not labels or labels[-1] != label:
                    labels.append(label)
            blueprints.append("-".join(labels))

    avg_v = round(sum(len(s) for s in verse_sections) / max(len(verse_sections), 1), 1)
    avg_c = round(sum(len(s) for s in chorus_sections) / max(len(chorus_sections), 1), 1)

    # Top 3 most common blueprints — validate and normalise
    bp_counter = Counter(blueprints)
    top_bps = []
    for bp, _ in bp_counter.most_common(5):
        bp = _normalise_blueprint(bp)
        if bp and bp not in top_bps:
            top_bps.append(bp)
        if len(top_bps) >= 3:
            break

    # Fallback: if no usable blueprints, use a safe default
    if not top_bps:
        top_bps = ["V-C-V-C-B-C"]

    return avg_v, avg_c, top_bps


def _normalise_blueprint(bp: str) -> str:
    """Validate and fix a song structure blueprint.

    - Filters out unknown 'X' labels
    - If no chorus exists but bridge repeats: rename repeating bridges to chorus
    - Ensures at least one Chorus exists
    """
    parts = bp.split("-")
    # Remove any stray 'X' labels
    parts = [p for p in parts if p != "X"]
    if not parts:
        return ""

    # If no chorus but bridge appears multiple times, it's probably a mislabelled chorus
    has_chorus = "C" in parts
    bridge_count = parts.count("B")
    if not has_chorus and bridge_count >= 2:
        # Rename all but the last Bridge to Chorus (last one stays as actual bridge)
        result = []
        bridges_seen = 0
        for p in parts:
            if p == "B":
                bridges_seen += 1
                # Keep only the last bridge as 'B', rename others to 'C'
                if bridges_seen < bridge_count:
                    result.append("C")
                else:
                    result.append("B")
            else:
                result.append(p)
        parts = result

    return "-".join(parts)


def _analyse_perspective(all_songs: list[SongLyrics]) -> str:
    """Analyse perspective/voice across all songs.

    Returns a descriptive string like "First-person confessional (78% I/me/my)".
    """
    first_person = 0   # I, me, my, mine, myself, I'm, I've, I'll, I'd
    second_person = 0  # you, your, yours, yourself, you're, you've, you'll
    third_person = 0   # he, she, they, him, her, them, his, their

    first_words = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd", "im"}
    second_words = {"you", "your", "yours", "yourself", "you're", "you've", "you'll", "ya"}
    third_words = {"he", "she", "they", "him", "her", "them", "his", "their", "hers", "theirs"}

    for song in all_songs:
        words = re.findall(r"[a-zA-Z']+", song.lyrics.lower())
        for w in words:
            if w in first_words:
                first_person += 1
            elif w in second_words:
                second_person += 1
            elif w in third_words:
                third_person += 1

    total = first_person + second_person + third_person
    if total == 0:
        return "Indeterminate (no clear pronoun pattern)"

    p1 = round(100 * first_person / total)
    p2 = round(100 * second_person / total)
    p3 = round(100 * third_person / total)

    parts = []
    if p1 >= 50:
        parts.append(f"First-person dominant ({p1}% I/me/my)")
    if p2 >= 30:
        parts.append(f"Second-person address ({p2}% you/your)")
    if p3 >= 30:
        parts.append(f"Third-person narrative ({p3}% he/she/they)")

    if not parts:
        parts.append(f"Mixed voice ({p1}% first / {p2}% second / {p3}% third)")

    # Add style classification
    if p1 >= 70:
        parts.append("— confessional / introspective style")
    elif p1 >= 50 and p2 >= 20:
        parts.append("— conversational / direct address style")
    elif p2 >= 50:
        parts.append("— confrontational / accusatory style")
    elif p3 >= 40:
        parts.append("— storytelling / observational style")

    return " ".join(parts)


def _analyse_meter(all_songs: list[SongLyrics]) -> dict:
    """Analyse syllable counts and line length variance.

    Returns a dict with avg_syllables_per_line, syllable_std_dev,
    avg_words_per_line, line_length_range.
    """
    syllable_counts: list[int] = []
    word_counts: list[int] = []
    char_counts: list[int] = []

    for song in all_songs:
        for line in song.lyrics.splitlines():
            stripped = line.strip()
            if not stripped or _SECTION_HEADER_RE.match(stripped):
                continue
            words = re.findall(r"[a-zA-Z']+", stripped)
            if not words:
                continue
            syl = sum(_count_syllables(w) for w in words)
            syllable_counts.append(syl)
            word_counts.append(len(words))
            char_counts.append(len(stripped))

    if not syllable_counts:
        return {"avg_syllables_per_line": 0, "syllable_std_dev": 0,
                "avg_words_per_line": 0, "line_length_range": "0-0"}

    avg_syl = round(sum(syllable_counts) / len(syllable_counts), 1)
    std_dev = round(math.sqrt(sum((s - avg_syl) ** 2 for s in syllable_counts) / len(syllable_counts)), 1)
    avg_words = round(sum(word_counts) / len(word_counts), 1)
    min_chars = min(char_counts)
    max_chars = max(char_counts)

    return {
        "avg_syllables_per_line": avg_syl,
        "syllable_std_dev": std_dev,
        "avg_words_per_line": avg_words,
        "line_length_range": f"{min_chars}-{max_chars} chars",
    }


# Common English words to exclude from "distinctive" word analysis
_COMMON_WORDS = frozenset(
    "the a an and or but if in on at to for of is it its that this with"
    " from by as are was were be been being have has had do does did"
    " will would shall should can could may might must not no nor so"
    " than too very just all each every both few more most other some"
    " such any only same also how when where why what which who whom"
    " i me my mine we us our they them their he him his she her you your"
    " about after again against between into through during before up down"
    " out off over under there here then now get got like go going know"
    " want need make take come think say tell give see feel find keep"
    " let put seem still try call ask look show turn move live help"
    " start run write set play hold bring happen begin walk talk love"
    " well back even new way day man right old big long little much"
    " good great first last time thing part work world life hand"
    " oh yeah hey ah oh uh ooh la da na hoo hey".split()
)


def _analyse_vocabulary(all_songs: list[SongLyrics]) -> dict:
    """Analyse vocabulary characteristics.

    Returns dict with type_token_ratio, contraction_pct,
    profanity_pct, distinctive_words, total_words, unique_words.
    """
    all_words: list[str] = []
    contractions = 0
    profanity = 0

    contraction_patterns = re.compile(
        r"\b(?:i'm|i've|i'll|i'd|don't|doesn't|didn't|won't|wouldn't|can't|couldn't"
        r"|shouldn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't"
        r"|ain't|it's|that's|what's|there's|here's|who's|let's"
        r"|you're|you've|you'll|you'd|we're|we've|we'll|we'd"
        r"|they're|they've|they'll|they'd|he's|she's"
        r"|gonna|wanna|gotta|kinda|sorta|nothin'|somethin'|burnin'|growin'"
        r"|draggin'|shaggin'|feelin'|whinin')\b",
        re.IGNORECASE,
    )

    profanity_words = frozenset(
        "shit fuck fucking fucked damn damn ass hell bitch bastard crap piss dick".split()
    )

    for song in all_songs:
        text = song.lyrics.lower()
        words = re.findall(r"[a-zA-Z']+", text)
        all_words.extend(words)
        contractions += len(contraction_patterns.findall(text))
        for w in words:
            if w in profanity_words:
                profanity += 1

    total = len(all_words)
    if total == 0:
        return {"type_token_ratio": 0, "total_words": 0, "unique_words": 0,
                "contraction_pct": 0, "profanity_pct": 0, "distinctive_words": []}

    unique = set(all_words)
    ttr = round(len(unique) / total, 3)
    contraction_pct = round(100 * contractions / total, 1)
    profanity_pct = round(100 * profanity / total, 1)

    # Find distinctive words (frequent but not in the common set)
    word_freq = Counter(all_words)
    distinctive = [
        w for w, _ in word_freq.most_common(100)
        if w not in _COMMON_WORDS and len(w) > 2
    ][:15]

    return {
        "type_token_ratio": ttr,
        "total_words": total,
        "unique_words": len(unique),
        "contraction_pct": contraction_pct,
        "profanity_pct": profanity_pct,
        "distinctive_words": distinctive,
    }


def _analyse_line_length_variation(all_songs: list[SongLyrics]) -> dict:
    """Analyse line length variation per section type.

    Returns a dict with syllable distributions, per-section stats,
    and example short/long lines from the actual lyrics.
    """
    section_syllables: dict[str, list[int]] = {"V": [], "C": [], "B": []}
    all_syllable_counts: list[int] = []
    example_lines: dict[str, list[tuple[int, str]]] = {"short": [], "long": []}

    for song in all_songs:
        sections = _split_into_sections(song.lyrics)
        for label, lines in sections:
            for line in lines:
                words = re.findall(r"[a-zA-Z']+", line)
                if not words:
                    continue
                syl = sum(_count_syllables(w) for w in words)
                all_syllable_counts.append(syl)
                if label in section_syllables:
                    section_syllables[label].append(syl)
                example_lines["short" if syl <= 4 else "long"].append((syl, line.strip()))

    if not all_syllable_counts:
        return {}

    # Build histogram buckets
    total = len(all_syllable_counts)
    buckets = {"1-4": 0, "5-7": 0, "8-10": 0, "11-14": 0, "15+": 0}
    for s in all_syllable_counts:
        if s <= 4:
            buckets["1-4"] += 1
        elif s <= 7:
            buckets["5-7"] += 1
        elif s <= 10:
            buckets["8-10"] += 1
        elif s <= 14:
            buckets["11-14"] += 1
        else:
            buckets["15+"] += 1
    histogram = {k: round(100 * v / total) for k, v in buckets.items()}

    # Per-section stats
    per_section = {}
    for label, counts in section_syllables.items():
        if counts:
            per_section[label] = {
                "min": min(counts),
                "max": max(counts),
                "avg": round(sum(counts) / len(counts), 1),
                "std": round(math.sqrt(sum((s - sum(counts)/len(counts))**2 for s in counts) / len(counts)), 1),
            }

    # Pick example lines
    short_examples = sorted(example_lines["short"], key=lambda x: x[0])[:3]
    long_examples = sorted(example_lines["long"], key=lambda x: -x[0])[:3]

    return {
        "histogram": histogram,
        "per_section": per_section,
        "short_line_examples": [f"({s} syl) {line}" for s, line in short_examples],
        "long_line_examples": [f"({s} syl) {line}" for s, line in long_examples],
    }


def _analyse_repetition(all_songs: list[SongLyrics]) -> dict:
    """Analyse hook and repetition patterns across songs.

    Returns a dict with chorus_repetition_pct, common patterns,
    and example repeated lines.
    """
    chorus_total_lines = 0
    chorus_repeated_lines = 0
    verse_total_lines = 0
    verse_repeated_lines = 0
    hook_examples: list[str] = []

    for song in all_songs:
        sections = _split_into_sections(song.lyrics)
        for label, lines in sections:
            if label == "C" and len(lines) >= 2:
                chorus_total_lines += len(lines)
                # Count repeated lines within this chorus
                line_counts = Counter(l.strip().lower() for l in lines)
                for line_text, count in line_counts.items():
                    if count > 1:
                        chorus_repeated_lines += count
                        if len(hook_examples) < 5 and line_text:
                            hook_examples.append(line_text)
            elif label == "V" and len(lines) >= 2:
                verse_total_lines += len(lines)
                line_counts = Counter(l.strip().lower() for l in lines)
                for _, count in line_counts.items():
                    if count > 1:
                        verse_repeated_lines += count

    # Cross-section repetition: lines that appear in multiple sections
    all_lines: list[str] = []
    for song in all_songs:
        for line in song.lyrics.splitlines():
            stripped = line.strip()
            if stripped and not _SECTION_HEADER_RE.match(stripped):
                all_lines.append(stripped.lower())
    global_line_counts = Counter(all_lines)
    cross_section_repeats = sum(1 for _, c in global_line_counts.items() if c >= 3)

    chorus_rep_pct = round(100 * chorus_repeated_lines / max(chorus_total_lines, 1))
    verse_rep_pct = round(100 * verse_repeated_lines / max(verse_total_lines, 1))

    # Classify pattern
    if chorus_rep_pct >= 50:
        pattern = "heavy-hook: choruses built around repeated lines"
    elif chorus_rep_pct >= 20:
        pattern = "moderate-hook: choruses use some repeated lines"
    else:
        pattern = "low-repetition: choruses mostly unique lines"

    return {
        "chorus_repetition_pct": chorus_rep_pct,
        "verse_repetition_pct": verse_rep_pct,
        "cross_section_repeats": cross_section_repeats,
        "pattern": pattern,
        "hook_examples": hook_examples[:5],
    }


def _select_representative_excerpts(
    all_songs: list[SongLyrics],
    max_excerpts: int = 5,
) -> list[str]:
    """Select the most representative verse/chorus excerpts for generation.

    Picks sections that best represent the artist's typical style:
    sections with average-ish length and clear rhyme patterns.
    """
    candidates: list[tuple[float, str, str]] = []  # (score, title, text)

    for song in all_songs:
        sections = _split_into_sections(song.lyrics)
        for label, lines in sections:
            if label not in ("V", "C") or len(lines) < 2:
                continue
            text = "\n".join(lines)
            # Score: prefer moderate-length sections (4-8 lines)
            length_score = 1.0 - abs(len(lines) - 6) * 0.15
            # Prefer sections with rhymes
            _, quality = _detect_rhyme_scheme(lines)
            rhyme_score = (quality["perfect"] * 1.0 + quality["slant"] * 0.6 + quality["assonance"] * 0.3)
            total_score = length_score + rhyme_score
            section_name = "Verse" if label == "V" else "Chorus"
            candidates.append((total_score, f"{song.title} ({section_name})", text))

    # Sort by score descending, pick top N
    candidates.sort(key=lambda x: x[0], reverse=True)
    excerpts = []
    seen_texts = set()
    for _, title, text in candidates:
        # Avoid duplicate/very similar excerpts
        if text not in seen_texts:
            seen_texts.add(text)
            excerpts.append(f"[{title}]\n{text}")
            if len(excerpts) >= max_excerpts:
                break

    return excerpts


# ── LLM prompts (split into smaller chunks for reliability) ──────────────────

_PROFILE_COMMON_PREAMBLE = """You are an expert musicologist and lyric analyst.
You will be given an artist's song lyrics and statistical analysis.

CRITICAL FORMAT RULES:
- Return ONLY a valid JSON object. No other text before or after.
- ALL values must be FLAT — plain strings or arrays of plain strings.
- Do NOT use nested objects, sub-keys, or arrays of objects.
- Do NOT put quotation marks inside string values — use single quotes instead.
- Be deeply specific and cite actual examples from the lyrics."""

_PROFILE_PROMPT_1 = _PROFILE_COMMON_PREAMBLE + """

Return JSON with exactly these 3 keys:
{
  "themes": ["theme 1 with specific examples cited", "theme 2 with examples", "etc"],
  "common_subjects": ["subject/motif 1 with examples", "subject 2 with examples", "etc"],
  "vocabulary_notes": "One detailed paragraph about vocabulary style, register, slang, metaphors, favourite words/phrases, citing specific examples"
}

Example of CORRECT format:
{"themes": ["Apocalyptic imagery - references to 'burning cities' and 'ash' in multiple songs"], "common_subjects": ["Fire as transformation metaphor"], "vocabulary_notes": "Heavy use of concrete nouns..."}

Do NOT return objects like {"theme": "x", "description": "y"} inside arrays."""

_PROFILE_PROMPT_2 = _PROFILE_COMMON_PREAMBLE + """

Return JSON with exactly these 3 keys:
{
  "tone_and_mood": "One detailed paragraph about emotional tone, mood shifts, irony/sarcasm/sincerity, citing examples",
  "structural_patterns": "One detailed paragraph about song structure beyond basic V-C-B, how ideas develop, repetition patterns, citing examples",
  "narrative_techniques": "One detailed paragraph about storytelling techniques, perspective shifts, dialogue, scene-setting, citing examples"
}

ALL values must be plain strings (paragraphs). No arrays, no nested objects."""

_PROFILE_PROMPT_3 = _PROFILE_COMMON_PREAMBLE + """

Return JSON with exactly these 4 keys:
{
  "imagery_patterns": "One detailed paragraph about recurring imagery types with specific examples cited",
  "signature_devices": "One detailed paragraph about verbal tics, signature phrases, recurring word pairings",
  "emotional_arc": "One detailed paragraph about how emotions develop within songs — build, release, cycle",
  "raw_summary": "A 3-4 paragraph prose summary synthesising the artist's complete lyrical style into a practical writing guide"
}

ALL values must be plain strings (paragraphs). No arrays, no nested objects."""

# Keep the old constant around as an alias for any external references
PROFILE_SYSTEM_PROMPT = _PROFILE_PROMPT_1


def _build_profile_prompt(
    artist: str,
    album: Optional[str],
    songs: list[SongLyrics],
    rule_based_stats: dict,
) -> str:
    """Build the user prompt for the LLM profiler, including all lyrics."""
    header = f"Artist: {artist}"
    if album:
        header += f"\nAlbum: {album}"
    header += f"\nSongs analysed: {len(songs)}"

    # Format rule-based stats
    stats_section = "\n=== RULE-BASED ANALYSIS ===\n"
    stats_section += f"Average verse length: {rule_based_stats['avg_verse_lines']} lines\n"
    stats_section += f"Average chorus length: {rule_based_stats['avg_chorus_lines']} lines\n"
    stats_section += f"Top rhyme schemes: {', '.join(rule_based_stats['rhyme_schemes'])}\n"
    rq = rule_based_stats['rhyme_quality']
    stats_section += f"Rhyme quality breakdown: {rq['perfect']} perfect, {rq['slant']} slant, {rq['assonance']} assonance\n"
    stats_section += f"Structure blueprints: {', '.join(rule_based_stats['structure_blueprints'])}\n"
    stats_section += f"Perspective: {rule_based_stats['perspective']}\n"
    ms = rule_based_stats['meter_stats']
    stats_section += f"Meter: avg {ms['avg_syllables_per_line']} syllables/line (σ={ms['syllable_std_dev']}), {ms['avg_words_per_line']} words/line, range {ms['line_length_range']}\n"
    vs = rule_based_stats['vocabulary_stats']
    stats_section += f"Vocabulary: {vs['total_words']} total words, {vs['unique_words']} unique, TTR={vs['type_token_ratio']}\n"
    stats_section += f"Contractions: {vs['contraction_pct']}% of words\n"
    stats_section += f"Profanity: {vs['profanity_pct']}% of words\n"
    stats_section += f"Distinctive words: {', '.join(vs['distinctive_words'])}\n"

    # Line length variation
    llv = ms.get("line_length_variation", {})
    if llv:
        hist = llv.get("histogram", {})
        if hist:
            hist_str = ", ".join(f"{k}: {v}%" for k, v in hist.items())
            stats_section += f"Syllable distribution: {hist_str}\n"

    # Repetition stats
    rs = rule_based_stats.get('repetition_stats', {})
    if rs:
        stats_section += f"Chorus repetition: {rs.get('chorus_repetition_pct', 0)}% of chorus lines are repeats\n"
        stats_section += f"Repetition pattern: {rs.get('pattern', 'unknown')}\n"
        hook_ex = rs.get("hook_examples", [])
        if hook_ex:
            stats_section += f"Hook examples: {'; '.join(hook_ex[:3])}\n"

    # Include ALL lyrics
    lyrics_section = "\n=== COMPLETE LYRICS ===\n\n"
    for song in songs:
        lyrics_section += f"--- {song.title} ---\n{song.lyrics}\n\n"

    return header + stats_section + lyrics_section


def _coerce_str(value) -> str:
    """Coerce a value to string — joins lists with newlines for Pydantic compatibility."""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value) if value else ""


# ── Public API ────────────────────────────────────────────────────────────────

_MAX_RETRIES = 2  # total attempts per call


def _llm_call_with_retry(
    llm_call,
    system_prompt: str,
    user_prompt: str,
    label: str,
    on_phase=None,
) -> tuple[str, dict]:
    """Call the LLM and validate JSON extraction; retry once on failure."""
    for attempt in range(1, _MAX_RETRIES + 1):
        if attempt > 1:
            logger.warning("%s: retry %d/%d", label, attempt, _MAX_RETRIES)
            if on_phase:
                on_phase(f"{label} (retry {attempt}/{_MAX_RETRIES})…")

        raw = llm_call(system_prompt, user_prompt)
        data = _extract_json(raw)
        if data:
            logger.info("%s: got %d keys", label, len(data))
            return raw, data
        logger.warning("%s: JSON extraction failed (attempt %d)", label, attempt)

    logger.error("%s: all %d attempts failed, using empty data", label, _MAX_RETRIES)
    return raw, {}


def build_profile(
    artist: str,
    album: Optional[str],
    songs: list[SongLyrics],
    llm_call,  # callable(system_prompt: str, user_prompt: str) -> str
    on_phase=None,  # optional callback for streaming progress
) -> LyricsProfile:
    """Build a richly detailed LyricsProfile from a list of songs.

    Makes 4 smaller LLM calls instead of 1 large one, for reliability
    with smaller reasoning models that struggle with large JSON output.
    Each call validates JSON and retries once on failure.
    """
    logger.info("Building profile for %s (%d songs)...", artist, len(songs))

    # Rule-based pre-analysis
    avg_verse, avg_chorus, blueprints = _analyse_structure(songs)
    rhyme_schemes, rhyme_quality = _analyse_rhymes(songs)
    perspective = _analyse_perspective(songs)
    meter_stats = _analyse_meter(songs)
    vocabulary_stats = _analyse_vocabulary(songs)
    line_length_variation = _analyse_line_length_variation(songs)
    repetition_stats = _analyse_repetition(songs)
    excerpts = _select_representative_excerpts(songs)

    # Merge line-length variation into meter_stats for the profile
    meter_stats["line_length_variation"] = line_length_variation

    rule_based_stats = {
        "avg_verse_lines": avg_verse,
        "avg_chorus_lines": avg_chorus,
        "rhyme_schemes": rhyme_schemes,
        "rhyme_quality": rhyme_quality,
        "structure_blueprints": blueprints,
        "perspective": perspective,
        "meter_stats": meter_stats,
        "vocabulary_stats": vocabulary_stats,
        "repetition_stats": repetition_stats,
    }

    logger.info("Rule-based analysis complete. Starting LLM analysis (4 calls)...")

    user_prompt = _build_profile_prompt(artist, album, songs, rule_based_stats)
    raw_responses: list[str] = []
    merged_data: dict = {}

    # Call 1: Themes, subjects, vocabulary
    if on_phase:
        on_phase("Analysing themes & vocabulary… (1/4)")
    raw1, data1 = _llm_call_with_retry(
        llm_call, _PROFILE_PROMPT_1, user_prompt, "Call 1/4 (themes)", on_phase
    )
    raw_responses.append(raw1)
    merged_data.update(data1)

    # Call 2: Tone, structure, narrative
    if on_phase:
        on_phase("Analysing tone & structure… (2/4)")
    raw2, data2 = _llm_call_with_retry(
        llm_call, _PROFILE_PROMPT_2, user_prompt, "Call 2/4 (tone)", on_phase
    )
    raw_responses.append(raw2)
    merged_data.update(data2)

    # Call 3: Imagery, signature devices, emotional arc, summary
    if on_phase:
        on_phase("Analysing imagery & signature… (3/4)")
    raw3, data3 = _llm_call_with_retry(
        llm_call, _PROFILE_PROMPT_3, user_prompt, "Call 3/4 (imagery)", on_phase
    )
    raw_responses.append(raw3)
    merged_data.update(data3)

    # Call 4: Song subjects
    if on_phase:
        on_phase("Analysing song subjects… (4/4)")
    logger.info("LLM call 4/4: song subjects")
    song_subjects, subject_categories = _analyse_song_subjects(songs, llm_call)

    raw_combined = "\n\n---\n\n".join(raw_responses)

    return LyricsProfile(
        artist=artist,
        album=album,
        themes=merged_data.get("themes", []),
        common_subjects=merged_data.get("common_subjects", []),
        rhyme_schemes=merged_data.get("rhyme_schemes", rhyme_schemes),
        avg_verse_lines=merged_data.get("avg_verse_lines", avg_verse),
        avg_chorus_lines=merged_data.get("avg_chorus_lines", avg_chorus),
        vocabulary_notes=_coerce_str(merged_data.get("vocabulary_notes", "")),
        tone_and_mood=_coerce_str(merged_data.get("tone_and_mood", "")),
        structural_patterns=_coerce_str(merged_data.get("structural_patterns", "")),
        additional_notes=_coerce_str(merged_data.get("additional_notes", "")),
        raw_summary=_coerce_str(merged_data.get("raw_summary", raw_combined)),
        raw_llm_response=raw_combined,
        # Enhanced fields
        structure_blueprints=blueprints,
        perspective=perspective,
        meter_stats=meter_stats,
        vocabulary_stats=vocabulary_stats,
        representative_excerpts=excerpts,
        narrative_techniques=_coerce_str(merged_data.get("narrative_techniques", "")),
        imagery_patterns=_coerce_str(merged_data.get("imagery_patterns", "")),
        signature_devices=_coerce_str(merged_data.get("signature_devices", "")),
        emotional_arc=_coerce_str(merged_data.get("emotional_arc", "")),
        rhyme_quality=rhyme_quality,
        song_subjects=song_subjects,
        subject_categories=subject_categories,
        repetition_stats=repetition_stats,
    )


# ── Subject analysis ──────────────────────────────────────────────────────────

_SUBJECT_SYSTEM_PROMPT = """You are a music analyst. For each song provided, write a ONE-SENTENCE summary of what the song is about — its core subject, not its style.

Then group all the subjects into 5-10 thematic categories that describe the range of topics this artist writes about.

Return JSON in exactly this format:
{
  "song_subjects": {
    "Song Title": "one sentence about what this specific song is about",
    ...
  },
  "subject_categories": ["category1", "category2", ...]
}

Be specific and concrete. "Basket Case" is about "anxiety and self-doubt about one's own mental stability", NOT "emotions" or "feelings".
Do NOT include any text outside the JSON object.
"""


def _analyse_song_subjects(
    songs: list[SongLyrics],
    llm_call,
) -> tuple[dict[str, str], list[str]]:
    """Ask the LLM to summarise each song's subject and group into categories."""
    song_list = "\n\n".join(
        f"--- {song.title} ---\n{song.lyrics[:500]}"  # First 500 chars is enough for subject
        for song in songs
    )
    user_prompt = f"Analyse the subjects of these {len(songs)} songs:\n\n{song_list}"

    raw = llm_call(_SUBJECT_SYSTEM_PROMPT, user_prompt)
    data = _extract_json(raw)
    if data:
        return data.get("song_subjects", {}), data.get("subject_categories", [])
    logger.warning("LLM did not return valid JSON for subjects; using fallback.")
    return {}, []

