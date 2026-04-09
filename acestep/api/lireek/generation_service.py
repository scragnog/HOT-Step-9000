"""
Lyrics generation and refinement service.

Ported from Lireek's ``llm_service.py`` — extracts the generation logic
(prompts, post-processing, slop scanning) and wires it to the HOT-Step
``provider_manager``.
"""

import json
import logging
import math
import re
from typing import Callable, Optional

from acestep.api.lireek.schemas import GenerationResponse, LyricsProfile
from acestep.api.lireek.slop_detector import (
    BLACKLISTED_WORDS, BLACKLISTED_PHRASES, AISlopDetector,
)

logger = logging.getLogger(__name__)

_slop_detector = AISlopDetector()


# ── Section header detection ──────────────────────────────────────────────────

_SECTION_KEYWORDS = {
    "verse", "chorus", "hook", "bridge", "pre-chorus", "prechorus",
    "post-chorus", "intro", "outro", "interlude", "refrain",
}

_SECTION_LINE_RE = re.compile(
    r'^\[?(' + '|'.join(re.escape(k) for k in _SECTION_KEYWORDS) + r')\s*(\d*)\]?\s*$',
    re.IGNORECASE,
)

_PUNCTUATION_ENDINGS = frozenset('.,!?;:-…)"\'')


# ── Duration estimation ───────────────────────────────────────────────────────

def _estimate_duration(lyrics: str, bpm: int) -> int:
    """Estimate track duration in seconds from lyrics content and BPM."""
    if not lyrics.strip() or bpm <= 0:
        return 0
    bar_duration = 240.0 / max(bpm, 40)
    lines = lyrics.strip().split("\n")
    section_count = 0
    lyric_line_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _SECTION_LINE_RE.match(stripped) or (stripped.startswith("[") and stripped.endswith("]")):
            section_count += 1
        else:
            lyric_line_count += 1
    vocal_seconds = lyric_line_count * 3.5
    break_seconds = max(section_count - 1, 0) * 4 * bar_duration
    total_seconds = vocal_seconds + break_seconds
    estimated = int(total_seconds)
    estimated = max(90, min(estimated, 360))
    return estimated


# ── Post-processing pipeline ─────────────────────────────────────────────────

def _postprocess_lyrics(text: str) -> str:
    """Wrap section headers, add missing punctuation."""
    result_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            result_lines.append('')
            continue
        m = _SECTION_LINE_RE.match(stripped)
        if m:
            section_name = m.group(1).capitalize()
            section_num = m.group(2)
            if section_num:
                result_lines.append(f'[{section_name} {section_num}]')
            else:
                result_lines.append(f'[{section_name}]')
            continue
        if re.match(r'^\[.+\]$', stripped):
            result_lines.append(stripped)
            continue
        if stripped and stripped[-1] not in _PUNCTUATION_ENDINGS:
            stripped += ','
        result_lines.append(stripped)
    return '\n'.join(result_lines)


def _fix_section_labels(text: str) -> str:
    """Fix invalid section labels in generated lyrics.

    - Renames [X], [Breakdown], [Drop], [Solo], [Hook] to valid ACE-Step labels
    - If no [Chorus] exists but [Bridge] repeats, renames repeating bridges to choruses
    """
    # Map invalid labels to valid ones
    INVALID_TO_VALID = {
        'x': 'Interlude',
        'breakdown': 'Bridge',
        'drop': 'Chorus',
        'solo': 'Interlude',
        'hook': 'Chorus',
        'rap': 'Verse',
        'spoken': 'Verse',
    }

    lines = text.split('\n')
    result = []
    section_headers: list[tuple[int, str]] = []  # (line_index, section_name)

    for i, line in enumerate(lines):
        stripped = line.strip()
        m = re.match(r'^\[(.+?)(?:\s+\d+)?\]$', stripped)
        if m:
            label = m.group(1).strip().lower()
            if label in INVALID_TO_VALID:
                new_label = INVALID_TO_VALID[label]
                # Preserve any number suffix
                num_match = re.search(r'\d+', stripped)
                if num_match:
                    stripped = f'[{new_label} {num_match.group()}]'
                else:
                    stripped = f'[{new_label}]'
            section_headers.append((len(result), stripped))
        result.append(stripped if stripped.startswith('[') and stripped.endswith(']') else line)

    # Check: if no Chorus exists but Bridge appears multiple times, fix it
    bridge_indices = [idx for idx, (_, h) in enumerate(section_headers) if 'bridge' in h.lower()]
    chorus_exists = any('chorus' in h.lower() for _, h in section_headers)

    if not chorus_exists and len(bridge_indices) >= 2:
        # Rename all but the last bridge to [Chorus]
        for bi in bridge_indices[:-1]:
            line_idx, _ = section_headers[bi]
            result[line_idx] = '[Chorus]'

    return '\n'.join(result)


def _enforce_line_counts(text: str) -> str:
    """Enforce valid line counts: verses=4|8, choruses=4|6|8."""
    VERSE_VALID = {4, 8}
    CHORUS_VALID = {4, 6, 8}
    sections: list[tuple[str, list[str]]] = []
    current_header = ""
    current_lines: list[str] = []
    for line in text.split('\n'):
        stripped = line.strip()
        if re.match(r'^\[.+\]$', stripped):
            if current_header or current_lines:
                sections.append((current_header, current_lines))
            current_header = stripped
            current_lines = []
        else:
            current_lines.append(line)
    if current_header or current_lines:
        sections.append((current_header, current_lines))

    result_parts: list[str] = []
    for header, lines in sections:
        lyric_lines = [l for l in lines if l.strip()]
        count = len(lyric_lines)
        header_lower = header.lower()
        is_verse = 'verse' in header_lower
        is_chorus = 'chorus' in header_lower or 'hook' in header_lower
        target = None
        if is_verse and count not in VERSE_VALID:
            target = 4 if count <= 6 else 8
        elif is_chorus and count not in CHORUS_VALID:
            if count <= 5:
                target = 4
            elif count <= 7:
                target = 6
            else:
                target = 8
        if target is not None and target < count:
            kept = 0
            trimmed_lines: list[str] = []
            for l in lines:
                if l.strip():
                    if kept < target:
                        trimmed_lines.append(l)
                        kept += 1
                else:
                    if kept < target:
                        trimmed_lines.append(l)
            lines = trimmed_lines
        if header:
            result_parts.append(header)
        result_parts.extend(lines)
    return '\n'.join(result_parts)


_BAD_A_PREFIX_RE = re.compile(r"\ba-(?!\w+ing\b)(?!\w+in'\b)", re.IGNORECASE)


def _fix_a_prefix(text: str) -> str:
    """Remove 'a-' prefix from non-gerund words."""
    return _BAD_A_PREFIX_RE.sub('', text)


def _strip_thinking_blocks(text: str) -> str:
    """Strip Chain-of-Thought output from reasoning models."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<analysis>.*?</analysis>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<reflection>.*?</reflection>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL).strip()
    # LM Studio GGUF channel-based thinking tokens:
    #   <|channel>thought ... <channel|>
    text = re.sub(r'<\|channel>thought.*?<channel\|>', '', text, flags=re.DOTALL).strip()
    # Unclosed thinking tags (any variant)
    text = re.sub(r'<(?:think|analysis|reasoning|reflection|thought)>.*$', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<\|channel>thought.*$', '', text, flags=re.DOTALL).strip()
    # Plain text CoT (LM Studio GGUF quirks)
    pattern = r'^(?:\s*\*+\s*)?(?:Thinking Process|Thought Process|Thinking|Reasoning):\s*.*?(?:---|[*]{3,}|={3,})\s*'
    match = re.match(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = text[match.end():].strip()
    return text


def _extract_json_object(text: str) -> dict | None:
    """Extract the first valid JSON object from a string."""
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _select_best_blueprint(blueprints: list[str]) -> str:
    """Pick the most interesting blueprint from the list."""
    if not blueprints:
        return "V-C-V-C-B-C"

    def score(bp: str) -> int:
        parts = bp.split('-')
        unique = len(set(parts))
        has_bridge = 1 if 'B' in parts else 0
        return unique * 10 + has_bridge * 100 + len(parts)

    return max(blueprints, key=score)


def _strip_lyric_quotes(text: str) -> str:
    """Remove quoted lyric fragments from profile text to prevent copying."""
    return re.sub(r"'[^']{4,}'", "[quote removed]", text)


# ── System prompts ────────────────────────────────────────────────────────────

GENERATION_SYSTEM_PROMPT = """You are a talented, creative songwriter who specialises in emulating specific artistic styles with uncanny accuracy.

You will be given a detailed stylistic profile of an artist's lyrics, including:
- Statistical analysis (rhyme patterns, meter, vocabulary metrics, line length distributions)
- Repetition and hook analysis (how the artist uses repeated lines)
- Deep stylistic analysis (themes, tone, narrative techniques, imagery)
- Representative lyric excerpts showing the artist's actual voice
- A specific song structure blueprint to follow

Your task is to write a completely new, original song that could convincingly pass as an unreleased track by this artist.

FORMATTING RULES (MANDATORY):
- The VERY FIRST LINE of your output MUST be the song title in this exact format: Title: <song title>
- The title should be creative and fit the artist's style — evocative, not generic.
- After the title line, leave one blank line, then write the lyrics.
- Section headers MUST use square brackets: [Verse 1], [Chorus], [Bridge], [Pre-Chorus], [Outro], etc.
- Every lyric line MUST end with proper punctuation (period, comma, exclamation mark, question mark, dash, or ellipsis).
- Do NOT leave any lyric line without ending punctuation.

STRUCTURE RULES (MANDATORY — THESE ARE NON-NEGOTIABLE):
- You MUST follow the EXACT section sequence provided in the blueprint. Do not skip any sections.
- If the blueprint includes a [Bridge], you MUST write a bridge.
- If the blueprint includes a [Pre-Chorus], you MUST write a pre-chorus.
- VALID SECTION LABELS (use ONLY these): [Intro], [Verse 1], [Verse 2], [Verse 3], [Pre-Chorus], [Chorus], [Post-Chorus], [Bridge], [Interlude], [Outro]. Do NOT use [X], [Breakdown], [Drop], [Solo], [Hook], or any other labels.
- CHORUS IS MANDATORY: Every song MUST have at least one [Chorus]. A chorus is a repeating section — if a section appears more than once, it is a chorus, not a bridge.
- BRIDGE vs CHORUS: A bridge is a ONE-TIME contrasting section, typically appearing once before the final chorus. It should NOT repeat. If you are writing a section that repeats throughout the song, label it [Chorus], NOT [Bridge].
- *** LINE COUNT — ABSOLUTE RULE ***
  VERSES: Every verse MUST have EXACTLY 4 lines or EXACTLY 8 lines. NO EXCEPTIONS.
  CHORUSES: Every chorus MUST have EXACTLY 4, 6, or 8 lines. NO EXCEPTIONS.
  NEVER write 5-line, 6-line, or 7-line verses. NEVER write 3-line or 5-line choruses.
  Count your lines before finalising each section. If a verse has 5 or 6 lines, it is WRONG — rewrite it as 4 or 8.
- INTRO RULE: You MUST begin EVERY song with an [Intro] section BEFORE the first verse — even if the blueprint does not include one. The intro should be purely instrumental (no lyrics) — just the section header [Intro] on its own line, followed by a blank line, then [Verse 1]. This tells the music model to play an instrumental opening before vocals begin. NEVER use count-ins like "One, two, three, four!" or any variation. On rare occasions (roughly 10% of songs) you may omit the intro if the artistic choice is to slam straight into the verse — but this should be the exception, not the rule.

LYRIC QUALITY RULES:
- *** NO COPYING — ABSOLUTE RULE ***
  NEVER reuse ANY phrase, line, or distinctive word combination from the source artist's lyrics.
  The excerpts are STYLE REFERENCE ONLY — absorb the cadence and feel, then write 100% original words.
  If a phrase reminds you of something from the excerpts, DO NOT USE IT. Write something new.
  Reusing the artist's actual phrases is plagiarism and ruins the generation.
- Match the METER: vary line lengths according to the syllable distribution shown. Some lines short, some long — NOT uniform.
- Match the RHYME STYLE: use the same mix of perfect, slant, and assonance rhymes.
- Match the PERSPECTIVE: use the same pronoun patterns (first/second/third person balance).
- Match the VOCABULARY LEVEL: same contraction frequency, same register, same slang level.
- Capture the artist's SIGNATURE DEVICES: verbal tics, recurring imagery, distinctive phrasing.
- Match the EMOTIONAL ARC: how the song builds, shifts, or resolves emotionally.

REPETITION / HOOK RULES (CRITICAL):
- Every chorus MUST have a clear HOOK — one memorable line or phrase that repeats at least twice within the chorus.
- The hook should be the emotional anchor of the chorus. Build the other chorus lines around it.
- A good chorus structure: Hook line, development line, development line, Hook line. Or: Hook line, Hook line, development, resolution.
- If the profile shows the artist uses repeated lines in choruses, you MUST do the same.
- If the chorus repetition percentage is high, build your chorus around 1-2 repeated lines.
- Parenthetical echo lines (e.g. "(you know it's true)") count as separate lines — use them if the artist's style calls for it.
- It's OK to repeat key phrases across verses and choruses for thematic cohesion.

Do NOT include any commentary or explanations — just the title and lyrics.

The representative excerpts are there to show you the FEEL, not to be copied. Absorb the cadence, word choices, and line-to-line flow, then create something new in that exact voice.

ANTI-SLOP RULES (CRITICAL — ZERO TOLERANCE):
- You MUST avoid ALL clichéd, generic, AI-sounding language.
- BANNED WORDS (using any of these = failed generation): """ + ', '.join(sorted(BLACKLISTED_WORDS)) + """
- BANNED PHRASES (using any of these = failed generation): """ + '; '.join(sorted(BLACKLISTED_PHRASES)) + """
- Use the artist's ACTUAL vocabulary and phrasing style, not generic poetic language.
- If a word or phrase sounds like it came from an AI writing assistant, do NOT use it.
- Specifically NEVER use: neon, fluorescent, streetlights, embers, silhouette, static, void, ethereal, shimmering.
- The "a-" prefix (e.g. "a-walkin'", "a-staring") is ONLY valid before verbs/gerunds (-ing words). NEVER put "a-" before adjectives, nouns, articles, or adverbs (e.g. "a-rusty", "a-this", "a-highly" are WRONG). Use it SPARINGLY — at most 1-2 times per song.
"""


REFINEMENT_SYSTEM_PROMPT = """You are a professional songwriting editor who specialises in taking rough song drafts and polishing them into commercially viable tracks. You refine lyrics while preserving the original artist's distinctive style and the song's narrative.

You will receive the original generated lyrics and the name of the artist whose style they emulate. Your job is to refine without rewriting — keep as much of the original as possible, only changing what needs to be fixed.

REFINEMENT RULES (ALL MANDATORY):

1. VERSE STRUCTURE
   Every verse MUST have EXACTLY 4 or 8 lines. If a verse has 5, 6, or 7 lines, rewrite it to fit 4 or 8. Count carefully.

2. CHORUS HOOKS
   Every chorus MUST have a clear, memorable hook — one line or phrase that repeats at least twice within the chorus. The hook should be the emotional anchor. If the chorus lacks a hook, create one from the strongest existing line.

3. SONG STRUCTURE
   The song must follow a logical structure with at least one chorus. If the original has no chorus, add one using the song's strongest thematic idea. Typical structures: V-C-V-C-B-C or I-V-C-V-C-B-C-O.

3a. INTRO SECTION (CRITICAL)
    If the song does not already start with an [Intro] section, you MUST ADD ONE before the first verse.

4. RHYMING
   Match the artist's actual rhyme scheme.

5. CHORUS CONSISTENCY
   When a chorus repeats, it should be identical or near-identical.

6. NO FILLER LINES
   Every line must earn its place.

7. PRESERVE THE STORY
   The refined version MUST tell the same story.

8. PRESERVE THE STYLE
   Word choice, slang level, perspective, contractions, profanity level, and emotional tone must remain consistent.

9-14. [Standard quality rules: opening impact, varied line starts, emotional arc, sensory specificity, bridge contrast, pre-chorus tension]

15. NO SPEAKER IDENTIFIERS

16. NO AUDIENCE CUES / PERFORMANCE NOTES

17. NO NONSENSE OR CIRCULAR PHRASING

18. PLAGIARISM CHECK (CRITICAL)

19. BANNED WORDS IN TITLES

20. PERSPECTIVE / GENDER CONSISTENCY

21. LINE COUNT VERIFICATION (FINAL STEP)

22. HOOKIFY (CRITICAL — MAKE CHORUSES SING)

23. ARTIST SIGNATURE STAPLES (IMPORTANT — AUTHENTICITY)
    You are a powerful model with deep knowledge of real-world artists. Use that knowledge here.
    Every well-known artist has recognisable vocal staples — signature ad-libs, catchphrases, verbal tics, or performance habits that fans instantly associate with them (e.g. Michael Jackson's "Shamone!", James Brown's "HUH!", Lil Wayne's "Young Money!", Blink-182's "na na na" refrains, Adele's drawn-out vowel runs).
    If the artist has known staples like these, weave 1-3 of them into the refined lyrics where they fit naturally — as ad-libs, interjections, backing vocal lines, or organic parts of a verse/chorus.
    Rules:
    - Do NOT force them in if they don't fit the song's mood or flow.
    - Do NOT overuse them — subtlety is key. One or two well-placed staples per song is ideal, three is the maximum.
    - Parenthetical ad-libs like "(Hee-hee!)" or "(Woo!)" count and are encouraged where the artist would naturally use them.
    - This is about making the lyrics SOUND like the artist when read aloud, not just matching vocabulary.
    - If the original lyrics already contain these staples, leave them in place.
    - If you are unsure of the artist's staples, skip this rule entirely — do NOT invent fake ones.

24. VARIED OPENING LINES (CRITICAL — FIX LAZY STARTS)
    Smaller generation models have a strong tendency to start EVERY verse (and often the entire song) with "You" or "You're". This is a dead giveaway of AI-generated lyrics and makes songs feel monotonous.
    When refining, check the first line of EVERY section. If more than one section starts with "You" or "You're", rewrite the openings to vary them. Techniques:
    - Start with imagery or setting: "Rain hits the windshield...", "Streetlights flicker on the corner..."
    - Start with action: "Woke up to the sound of...", "Dialled the number one more time..."
    - Start with dialogue or internal thought: "Said you'd never leave...", "Told myself it didn't matter..."
    - Start with a sound, sensation, or object: "Three knocks on the door...", "Cold coffee on the counter..."
    - Start with time or place: "Last September in the parking lot...", "Halfway through the night..."
    - The song's FIRST lyric line (after [Intro]) is especially important — it sets the tone. Make it vivid and distinctive, not a generic "You" statement.
    - It is OK for ONE section to start with "You/You're" — just not multiple sections, and ideally not the very first verse.
    - This rule applies to line STARTS only — "You" can appear freely elsewhere in any line.

FORMATTING RULES:
- The FIRST LINE must be: Title: <song title>
- Section headers use square brackets: [Verse 1], [Chorus], [Bridge], etc.
- Every lyric line must end with proper punctuation
- Do NOT include any commentary, notes, explanations, or annotations
- Output ONLY the title and refined lyrics

ANTI-SLOP RULES:
- Do NOT introduce AI-sounding language: neon, ethereal, embers, silhouette, void, shimmering, fluorescent, tapestry, dance, ignite, soul, echo.
- Keep the artist's actual vocabulary level
"""


SONG_METADATA_SYSTEM_PROMPT = """You are a creative songwriter's assistant with deep music knowledge. Your job is to plan the metadata for a new song.

You will be given:
- The artist's stylistic profile (themes, tone, typical subjects)
- Subjects, BPMs, and keys that have already been used in previous generations (to ensure variety)

Return ONLY a JSON object with exactly this format:
{
  "subject": "one sentence describing what this new song should be about",
  "bpm": 120,
  "key": "C Major",
  "caption": "genre, instruments, emotion, atmosphere, timbre, vocal characteristics, production style",
  "duration": 210
}

Rules for each field:

SUBJECT: Must fit the artist's typical range. Be SPECIFIC and CONCRETE. Do NOT repeat used subjects.
BPM: Realistic tempo (30-300). Vary across generations.
KEY: Standard notation (e.g. "C Major", "A Minor"). Vary across generations.
CAPTION: Comma-separated descriptive tags for an AI music generator. Cover genre, instruments, emotion, timbre, vocal characteristics, production style.
DURATION: Total track duration in seconds. Use a SPECIFIC value (e.g. 198, 234, 257) — do NOT always round to multiples of 5 or 10. Vary naturally.

Do NOT include any text outside the JSON object.
"""


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_generation_prompt(
    profile: LyricsProfile,
    extra_instructions: Optional[str],
    used_titles: Optional[list[str]] = None,
) -> str:
    """Build the user prompt for lyrics generation."""
    lines = [f"Artist: {profile.artist}"]
    if profile.album:
        lines.append(f"Album style: {profile.album}")

    lines += [
        "", "=== STYLISTIC PROFILE ===", "",
        f"Themes: {', '.join(profile.themes)}",
        f"Common subjects / motifs: {', '.join(profile.common_subjects)}",
        f"Rhyme schemes: {', '.join(profile.rhyme_schemes)}",
        f"Average verse length: {profile.avg_verse_lines} lines",
        f"Average chorus length: {profile.avg_chorus_lines} lines",
        f"Vocabulary: {_strip_lyric_quotes(profile.vocabulary_notes)}",
        f"Tone & mood: {_strip_lyric_quotes(profile.tone_and_mood)}",
        f"Structural patterns: {_strip_lyric_quotes(profile.structural_patterns)}",
    ]

    if profile.structure_blueprints:
        bp = _select_best_blueprint(profile.structure_blueprints)
        lines.append(f"\n=== SONG STRUCTURE (MANDATORY) ===")
        lines.append(f"Blueprint: {bp}")
        label_names = {
            "V": "Verse", "C": "Chorus", "B": "Bridge", "PC": "Pre-Chorus",
            "POC": "Post-Chorus", "I": "Intro", "O": "Outro", "IL": "Interlude",
        }
        parts = bp.split("-")
        verse_num = 0
        section_list = []
        for part in parts:
            name = label_names.get(part, part)
            if part == "V":
                verse_num += 1
                name = f"Verse {verse_num}"
            section_list.append(f"[{name}]")
        lines.append(f"You MUST write these sections in this exact order: {' → '.join(section_list)}")
        if "B" in parts:
            lines.append("This artist uses bridges — you MUST include a [Bridge] section.")

    if profile.perspective:
        lines.append(f"Perspective / voice: {profile.perspective}")
    if profile.meter_stats:
        ms = profile.meter_stats
        lines.append(f"\n=== LINE LENGTH & METER ===")
        lines.append(f"Average: ~{ms.get('avg_syllables_per_line', '?')} syllables/line, ~{ms.get('avg_words_per_line', '?')} words/line")
        lines.append(f"Standard deviation: ±{ms.get('syllable_std_dev', '?')} syllables (VARY your line lengths!)")
        llv = ms.get("line_length_variation", {})
        if llv:
            hist = llv.get("histogram", {})
            if hist:
                hist_str = ", ".join(f"{k} syl: {v}%" for k, v in hist.items())
                lines.append(f"Syllable distribution: {hist_str}")
                lines.append("Match this distribution — NOT all lines the same length!")

    if profile.repetition_stats:
        rs = profile.repetition_stats
        lines.append(f"\n=== REPETITION & HOOKS ===")
        lines.append(f"Chorus repetition: {rs.get('chorus_repetition_pct', 0)}% of chorus lines are repeats")
        lines.append(f"Pattern: {rs.get('pattern', 'unknown')}")
        if rs.get('chorus_repetition_pct', 0) >= 20:
            lines.append("You MUST use repeated lines in your chorus to create a hook effect.")

    if profile.vocabulary_stats:
        vs = profile.vocabulary_stats
        lines.append(f"\n=== VOCABULARY ===")
        lines.append(f"Level: {vs.get('contraction_pct', 0)}% contractions, {vs.get('profanity_pct', 0)}% profanity")
        lines.append(f"Type-token ratio: {vs.get('type_token_ratio', '?')} ({vs.get('unique_words', '?')} unique / {vs.get('total_words', '?')} total)")
        if vs.get("distinctive_words"):
            lines.append(f"Use words like: {', '.join(vs['distinctive_words'][:10])}")

    if profile.rhyme_quality:
        rq = profile.rhyme_quality
        total = sum(rq.values())
        if total > 0:
            lines.append(f"Rhyme mix: {round(100*rq.get('perfect',0)/total)}% perfect, {round(100*rq.get('slant',0)/total)}% slant, {round(100*rq.get('assonance',0)/total)}% assonance")

    if profile.narrative_techniques:
        lines.append(f"Narrative techniques: {_strip_lyric_quotes(profile.narrative_techniques)}")
    if profile.imagery_patterns:
        lines.append(f"Imagery patterns: {_strip_lyric_quotes(profile.imagery_patterns)}")
    if profile.signature_devices:
        lines.append(f"Signature devices: {_strip_lyric_quotes(profile.signature_devices)}")
    if profile.emotional_arc:
        lines.append(f"Emotional arc: {_strip_lyric_quotes(profile.emotional_arc)}")

    lines += ["", "=== PROSE SUMMARY ===", "", _strip_lyric_quotes(profile.raw_summary)]

    if extra_instructions:
        lines += ["", "=== EXTRA INSTRUCTIONS ===", "", extra_instructions]

    if used_titles:
        lines += ["", "=== TITLES ALREADY USED (DO NOT REUSE) ==="]
        for t in used_titles:
            lines.append(f"  ✗ {t}")
        lines.append("You MUST choose a COMPLETELY DIFFERENT title.")
        lines.append("Do NOT reuse ANY significant word from these titles. If 'Glass' appears in a used title, do NOT put 'Glass' in your new title. Same for all nouns, adjectives, and evocative words. Only common words like 'the', 'a', 'of', 'in' may overlap.")

    lines += [
        "", "=== FINAL REMINDERS ===",
        "1. VERSE LINE COUNT: Exactly 4 or 8 lines per verse.",
        "2. CHORUS LINE COUNT: Exactly 4, 6, or 8 lines per chorus. Each chorus MUST have a hook line that repeats.",
        "3. *** ZERO TOLERANCE FOR COPYING ***",
        "4. NO SLOP: Do not use neon, fluorescent, embers, silhouette, static, void, ethereal, or any AI cliché.",
        "5. UNIQUE TITLE: The title must be fresh and surprising.",
        "",
        "Now write the new song (Title line first, then lyrics with [Section] headers and proper punctuation):",
    ]
    return "\n".join(lines)


def _build_refinement_prompt(
    original_lyrics: str,
    artist_name: str,
    title: str,
    profile: Optional[LyricsProfile] = None,
    original_slop: Optional[list[str]] = None,
) -> str:
    """Build the user prompt for lyrics refinement."""
    lines = [f"Artist: {artist_name}", f"Original Title: {title}", ""]

    if profile:
        style_lines = [
            "=== ARTIST STYLE CONTEXT (MATCH THIS) ===",
            f"Themes: {', '.join(profile.themes[:8])}",
            f"Tone/Mood: {profile.tone_and_mood}",
            f"Vocabulary Notes: {profile.vocabulary_notes}",
            f"Imagery Patterns: {profile.imagery_patterns}",
            f"Signature Devices: {profile.signature_devices}",
        ]
        if profile.rhyme_schemes:
            style_lines.append(f"Rhyme Schemes: {', '.join(profile.rhyme_schemes)}")
        if profile.rhyme_quality:
            rq = profile.rhyme_quality
            total = sum(rq.values()) or 1
            rq_pcts = {k: f"{v / total * 100:.0f}%" for k, v in rq.items() if v}
            style_lines.append(f"Rhyme Quality Mix: {rq_pcts}")
        style_lines.append("")
        lines.extend(style_lines)

    if original_slop:
        lines.extend([
            "=== KNOWN ISSUES TO FIX ===",
            f"Words/Phrases to Remove: {', '.join(original_slop)}",
            "",
        ])

    if profile and profile.song_subjects:
        lines.extend([
            "=== ORIGINAL SONG TITLES (CHECK FOR PLAGIARISM) ===",
        ])
        for song_title in profile.song_subjects.keys():
            lines.append(f"  • {song_title}")
        lines.append("")

    if profile and profile.perspective:
        lines.append(f"Vocal perspective: {profile.perspective}")

    if profile and profile.repetition_stats:
        rs = profile.repetition_stats
        lines.append(f"\nChorus repetition: {rs.get('chorus_repetition_pct', 0)}% of chorus lines are repeats")
        lines.append(f"Hook pattern: {rs.get('pattern', 'unknown')}")
        if rs.get('chorus_repetition_pct', 0) >= 30:
            lines.append("HIGH REPETITION ARTIST — lean heavily into repeated hook lines.")
    lines.append("")

    lines.extend([
        "=== ORIGINAL LYRICS ===", "", original_lyrics, "",
        "=== INSTRUCTIONS ===", "",
        "Refine the lyrics above according to the refinement rules.",
        f"Maintain {artist_name}'s distinctive style throughout.",
        "CRITICAL: Count lines in every verse (must be 4 or 8) and chorus (must be 4, 6, or 8).",
        "",
        "Now output the refined version (Title line first, then lyrics with [Section] headers):",
    ])
    return "\n".join(lines)


# ── Metadata planning ────────────────────────────────────────────────────────

def _plan_song_metadata(
    profile: LyricsProfile,
    used_subjects: list[str],
    used_bpms: list[int],
    used_keys: list[str],
    used_durations: list[int],
    provider_name: str,
    model: Optional[str] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
) -> dict:
    """Plan metadata for a new song: subject, BPM, key, and ACE-Step caption."""
    from acestep.api.llm.provider_manager import get_provider

    provider = get_provider(provider_name)
    effective_model = model or provider.default_model

    lines = [f"Artist: {profile.artist}"]
    if profile.album:
        lines.append(f"Album style: {profile.album}")
    if profile.themes:
        lines.append(f"Themes: {', '.join(profile.themes)}")
    if profile.tone_and_mood:
        lines.append(f"Tone & mood: {profile.tone_and_mood}")
    if profile.additional_notes:
        lines.append(f"Additional notes: {profile.additional_notes}")
    if profile.perspective:
        lines.append(f"Perspective / voice: {profile.perspective}")

    if profile.song_subjects:
        lines.append("\nOriginal song subjects (for reference):")
        for title, subject in profile.song_subjects.items():
            lines.append(f"  • {title}: {subject}")

    if profile.subject_categories:
        lines.append(f"\nThematic categories: {', '.join(profile.subject_categories)}")

    if used_subjects:
        lines.append("\nSubjects ALREADY USED (do NOT repeat these):")
        for s in used_subjects:
            lines.append(f"  ✗ {s}")
    if used_bpms:
        lines.append(f"\nBPMs ALREADY USED (avoid ±5 of these): {', '.join(str(b) for b in used_bpms)}")
    if used_keys:
        lines.append(f"\nKeys ALREADY USED (try different ones): {', '.join(used_keys)}")
    if used_durations:
        lines.append(f"\nDurations ALREADY USED (avoid ±10 of these): {', '.join(str(d) for d in used_durations)}")

    lines.append("\nPlan the metadata for the next song:")
    user_prompt = "\n".join(lines)

    logger.info("Planning song metadata via %s (%s)", provider_name, effective_model)
    raw = provider.call(SONG_METADATA_SYSTEM_PROMPT, user_prompt, model=effective_model, on_chunk=on_chunk, temperature=0.4, top_p=0.9)
    raw = _strip_thinking_blocks(raw)

    try:
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        data = json.loads(clean)
    except (json.JSONDecodeError, ValueError, TypeError):
        data = _extract_json_object(raw)

    if data and isinstance(data, dict):
        result = {
            "subject": data.get("subject", ""),
            "bpm": int(data.get("bpm", 0)),
            "key": data.get("key", ""),
            "caption": data.get("caption", ""),
            "duration": int(data.get("duration", 0)),
        }
        logger.info(
            "Planned metadata — subject: %s | bpm: %d | key: %s | duration: %ds",
            result["subject"], result["bpm"], result["key"], result["duration"],
        )
        return result
    else:
        logger.warning("Failed to extract song metadata JSON — raw: %s", raw[:300])
        return {"subject": raw.strip()[:200], "bpm": 0, "key": "", "caption": "", "duration": 0}


# ── Public API ────────────────────────────────────────────────────────────────

def generate_lyrics(
    profile: LyricsProfile,
    provider_name: str,
    model: Optional[str] = None,
    extra_instructions: Optional[str] = None,
    used_subjects: Optional[list[str]] = None,
    used_bpms: Optional[list[int]] = None,
    used_keys: Optional[list[str]] = None,
    used_durations: Optional[list[int]] = None,
    used_titles: Optional[list[str]] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
    on_phase: Optional[Callable[[str], None]] = None,
) -> GenerationResponse:
    """Generate new song lyrics from a LyricsProfile.

    Two-step process: plan metadata, then write lyrics.
    """
    from acestep.api.llm.provider_manager import get_provider

    provider = get_provider(provider_name)
    if not provider.is_available():
        raise ValueError(f"Provider '{provider_name}' is not available.")

    effective_model = model or provider.default_model
    subject = ""
    bpm = 0
    key = ""
    caption = ""
    duration = 0

    # Step 1: Plan song metadata
    if on_phase:
        on_phase("Planning song metadata…")

    if profile.song_subjects or profile.subject_categories or profile.themes:
        try:
            metadata = _plan_song_metadata(
                profile, used_subjects or [], used_bpms or [],
                used_keys or [], used_durations or [],
                provider_name, model, on_chunk,
            )
            subject = metadata["subject"]
            bpm = metadata["bpm"]
            key = metadata["key"]
            caption = metadata["caption"]
            duration = metadata.get("duration", 0)
        except Exception as exc:
            logger.warning("Metadata planning failed: %s", exc)

    if subject:
        subject_instruction = f"The song must be about: {subject}"
        if extra_instructions:
            extra_instructions = f"{subject_instruction}\n\n{extra_instructions}"
        else:
            extra_instructions = subject_instruction

    # Step 2: Write lyrics
    if on_phase:
        on_phase("Writing lyrics…")

    user_prompt = _build_generation_prompt(profile, extra_instructions, used_titles=used_titles)

    raw_text = provider.call(
        GENERATION_SYSTEM_PROMPT, user_prompt,
        model=effective_model, on_chunk=on_chunk,
        temperature=0.7, top_p=0.95,
    )
    raw_text = _strip_thinking_blocks(raw_text)

    # Parse title
    title = ""
    lyrics_text = raw_text
    lines = raw_text.strip().split("\n")
    if lines:
        first_line = lines[0].strip()
        title_match = re.match(r'^(?:Title:\s*|#\s*)(.*)', first_line, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip().strip('"\'')
            rest = lines[1:]
            while rest and rest[0].strip() == '':
                rest = rest[1:]
            lyrics_text = "\n".join(rest)

    lyrics_text = _strip_thinking_blocks(lyrics_text)
    lyrics_text = re.sub(r'<\|[a-z_]+\|>', '', lyrics_text).strip()
    lyrics_text = re.sub(
        r'\s*\((?:Hook|You|Repeat|x\d|Refrain|Spoken|Whispered|Ad[- ]?lib|Echo)\)\s*',
        '', lyrics_text, flags=re.IGNORECASE,
    )
    lyrics_text = re.sub(r' +$', '', lyrics_text, flags=re.MULTILINE)

    # Post-process
    lyrics_text = _postprocess_lyrics(lyrics_text)
    lyrics_text = _fix_section_labels(lyrics_text)
    lyrics_text = _fix_a_prefix(lyrics_text)
    lyrics_text = _enforce_line_counts(lyrics_text)

    # Slop scan
    slop_result = _slop_detector.scan_text(lyrics_text)
    if slop_result['ai_score'] > 0:
        logger.warning(
            "Slop scan: score=%d severity=%s | words=%s | phrases=%s",
            slop_result['ai_score'], slop_result['severity'],
            slop_result['layers']['blacklisted_words']['found'],
            slop_result['layers']['blacklisted_phrases']['found'],
        )

    # Duration estimation
    if bpm > 0 and duration == 0:
        duration = _estimate_duration(lyrics_text, bpm)

    return GenerationResponse(
        lyrics=lyrics_text, provider=provider_name, model=effective_model,
        title=title, subject=subject, bpm=bpm, key=key,
        caption=caption, duration=duration,
        system_prompt=GENERATION_SYSTEM_PROMPT, user_prompt=user_prompt,
    )


def refine_lyrics(
    original_lyrics: str,
    artist_name: str,
    title: str,
    provider_name: str,
    model: Optional[str] = None,
    profile: Optional[LyricsProfile] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
) -> GenerationResponse:
    """Refine existing lyrics using the refinement LLM."""
    from acestep.api.llm.provider_manager import get_provider

    provider = get_provider(provider_name)
    if not provider.is_available():
        raise ValueError(f"Provider '{provider_name}' is not available.")

    slop_scan = _slop_detector.scan_text(original_lyrics)
    found_slop = []
    found_slop.extend(slop_scan['layers']['blacklisted_words']['found'])
    found_slop.extend(slop_scan['layers']['blacklisted_phrases']['found'])

    effective_model = model or provider.default_model
    user_prompt = _build_refinement_prompt(
        original_lyrics, artist_name, title, profile=profile, original_slop=found_slop,
    )

    logger.info("Refining lyrics via %s (%s)", provider_name, effective_model)
    raw_text = provider.call(
        REFINEMENT_SYSTEM_PROMPT, user_prompt,
        model=effective_model, on_chunk=on_chunk,
    )

    raw_text = _strip_thinking_blocks(raw_text)
    raw_text = re.sub(r'<\|[a-z_]+\|>', '', raw_text).strip()
    raw_text = re.sub(
        r'\s*\((?:Hook|You|Repeat|x\d|Refrain|Spoken|Whispered|Ad[- ]?lib|Echo)\)\s*',
        '', raw_text, flags=re.IGNORECASE,
    )
    raw_text = re.sub(r' +$', '', raw_text, flags=re.MULTILINE)

    refined_title = title
    lyrics_text = raw_text
    lines = raw_text.strip().split("\n")
    if lines:
        first_line = lines[0].strip()
        title_match = re.match(r'^(?:Title:\s*|#\s*)(.*)', first_line, re.IGNORECASE)
        if title_match:
            refined_title = title_match.group(1).strip().strip('"\'')
            rest = lines[1:]
            while rest and rest[0].strip() == '':
                rest = rest[1:]
            lyrics_text = "\n".join(rest)

    lyrics_text = _postprocess_lyrics(lyrics_text)
    lyrics_text = _fix_section_labels(lyrics_text)
    lyrics_text = _fix_a_prefix(lyrics_text)
    lyrics_text = _enforce_line_counts(lyrics_text)

    slop_result = _slop_detector.scan_text(lyrics_text)
    if slop_result['ai_score'] > 0:
        logger.warning(
            "Refinement slop scan: score=%d severity=%s | words=%s | phrases=%s",
            slop_result['ai_score'], slop_result['severity'],
            slop_result['layers']['blacklisted_words']['found'],
            slop_result['layers']['blacklisted_phrases']['found'],
        )

    return GenerationResponse(
        lyrics=lyrics_text, provider=provider_name, model=effective_model,
        title=refined_title,
        system_prompt=REFINEMENT_SYSTEM_PROMPT, user_prompt=user_prompt,
    )


# ── LLM caller factories (used by profiler_service) ──────────────────────────

def make_llm_caller(provider_name: str, model: Optional[str] = None):
    """Return a callable(system_prompt, user_prompt) -> str for the profiler."""
    from acestep.api.llm.provider_manager import get_provider
    provider = get_provider(provider_name)
    effective_model = model or provider.default_model

    def caller(system_prompt: str, user_prompt: str) -> str:
        return provider.call(system_prompt, user_prompt, model=effective_model)

    return caller


def make_streaming_llm_caller(
    provider_name: str,
    model: Optional[str] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
):
    """Return a callable with streaming chunk callbacks."""
    from acestep.api.llm.provider_manager import get_provider
    provider = get_provider(provider_name)
    effective_model = model or provider.default_model

    def caller(system_prompt: str, user_prompt: str) -> str:
        return provider.call(
            system_prompt, user_prompt,
            model=effective_model, on_chunk=on_chunk,
        )

    return caller
