"""
Enhanced AI-Slop Detection and Prevention System (v2).
6-layer defense to ensure generated lyrics feel authentic to the source artist.

Layers:
  1. Blacklisted words (expanded to ~100 terms)
  2. Blacklisted phrases (expanded to ~75 phrases)
  3. Regex pattern detection (refined, fewer false positives)
  4. Structural analysis (line uniformity, repetitive grammar)
  5. Artist fingerprint comparison (TTR, line stats, vocab overlap)
  6. Statistical anomaly detection (function words, hapax ratio, opener variety)
"""

import re
import json
import math
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Layer 1 — Blacklisted Words
# ---------------------------------------------------------------------------

BLACKLISTED_WORDS: Set[str] = {
    # Visual clichés
    "neon", "streetlights", "streetlight", "silhouette", "silhouettes",
    "tapestry", "mosaic", "kaleidoscope", "prism",

    # Action clichés
    "yearning", "beckons", "beckoning", "cascading", "cascade",
    "unfurling", "unfurl",

    # Emotional clichés
    "bittersweet", "melancholy", "wistful", "poignant",
    "ethereal", "ephemeral",

    # Abstract concepts
    "symphony", "harmonize", "harmonizing",
    "crossroads",

    # Overused metaphors
    "phoenix", "labyrinth", "soaring",

    # Generic intensity
    "pulsing", "pulsating", "throbbing", "vibrant",
    "vivid", "luminous", "radiant", "shimmering",

    # Time clichés
    "hourglass", "timeless",

    # Nature clichés
    "tempest",

    # Existential clichés
    "essence", "consciousness", "realm", "dimension",

    # Modern AI clichés (2024-2026 patterns)
    "unraveling", "unravel", "ember", "embers", "ignite", "ignites",
    "resonate", "resonates", "reverberate", "reverberates",

    # Faux-poetic
    "amidst", "entwined", "intertwined", "ablaze",

    # Cosmic clichés
    "constellation", "constellations", "cosmos", "infinite", "infinity",
    "void",

    # Over-emotional constructions
    "shattering", "hollowed",

    # Synesthesia clichés
    "crimson sky", "velvet night",

    # Generic AI title/filler words
    "static", "catalyst", "paradox", "paradigm", "mantra",
    "epitome", "chronicles", "solace", "juxtaposition",
    "serenity", "resilience", "dichotomy", "transcend",
    "transcendence", "metamorphosis", "pinnacle",

    # Lighting clichés
    "fluorescent", "halogen",
}

# ---------------------------------------------------------------------------
# Layer 2 — Blacklisted Phrases
# ---------------------------------------------------------------------------

BLACKLISTED_PHRASES: Set[str] = {
    # Original set (kept)
    "reaching up to the sky", "beneath the streetlights",
    "under the streetlights", "neon lights", "neon glow", "neon dreams",
    "echoes in the night", "whispers in the dark",
    "shadows dance", "dancing shadows",
    "tapestry of dreams", "symphony of", "kaleidoscope of",
    "mosaic of emotions", "bittersweet memories", "fleeting moments",
    "sands of time", "rising from the ashes", "like a phoenix",
    "tangled web", "labyrinth of", "journey begins", "path ahead",
    "crossroads of", "fabric of reality", "threads of fate",
    "ocean of tears", "sea of faces", "waves of emotion",
    "storm within", "tempest raging",
    "essence of", "realm of", "universe within",
    "consciousness expands", "vivid dreams",
    "radiant light", "pulsing with", "cascading down",
    "ethereal beauty", "melancholy mood",
    "beckons me", "yearning for",

    # Expanded set — modern AI giveaways
    "paint the sky", "written in the stars", "dance with the devil",
    "scream into the void", "drown in your eyes", "heart on my sleeve",
    "break the chains", "find my voice", "lost in the moment",
    "through the fire", "edge of forever", "weight of the world",
    "paint a picture", "piece by piece", "shattered glass",
    "hollow eyes", "burning bridges", "chase the sun",
    "bleeding heart", "silent scream", "torn apart",
    "crumbling walls", "whisper your name", "dust settles",
    "ghost of you", "ashes to ashes", "taste of freedom",
    "colors of the wind", "sound of silence",
    "in this moment", "against the tide", "into the unknown",
    "carry the weight", "unravel the truth", "embers glow",
    "spark ignites", "constellations align", "resonates within",
    "let it all go", "rise above it all",
}

# ---------------------------------------------------------------------------
# Layer 3 — Regex Patterns (refined to reduce false positives)
# ---------------------------------------------------------------------------

_AI_PATTERNS_RAW = [
    # Abstract-of-abstract: "the tapestry of dreams", "the fabric of time"
    # Only flag when the head noun is itself an AI-slop word
    r"\b(tapestry|fabric|symphony|kaleidoscope|mosaic|labyrinth|maze|void)\s+of\s+\w+\b",

    # Simile with AI-favourite nouns
    r"\blike\s+a\s+(phoenix|symphony|kaleidoscope|constellation|ember)\b",

    # Double-gerund (pulsing throbbing, cascading unfurling)
    r"\b\w+ing\s+\w+ing\b",

    # "the [noun] of my [noun]" — overly poetic possessive
    r"\bthe\s+\w+\s+of\s+my\s+\w+\b",

    # Lines starting with "In the [abstract]" — AI loves this opener
    r"(?m)^In\s+the\s+(darkness|silence|shadows|distance|stillness|emptiness)\b",
]

AI_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _AI_PATTERNS_RAW]


# ---------------------------------------------------------------------------
# Layer 4 — Structural Analysis
# ---------------------------------------------------------------------------

def _analyze_structure(lines: List[str]) -> Dict[str, Any]:
    """
    Detect structural uniformity that signals AI generation.

    Real lyrics are messy — lines vary in length, structure deviates.
    AI tends to produce perfectly balanced stanzas.
    """
    issues: List[str] = []
    score = 0

    if len(lines) < 4:
        return {"issues": issues, "score": score}

    # Word counts per line
    word_counts = [len(line.split()) for line in lines if line.strip()]
    if not word_counts:
        return {"issues": issues, "score": score}

    mean_wc = sum(word_counts) / len(word_counts)
    variance = sum((wc - mean_wc) ** 2 for wc in word_counts) / len(word_counts)
    stddev = math.sqrt(variance) if variance > 0 else 0

    # Very low standard deviation = suspiciously uniform
    if len(word_counts) >= 6 and stddev < 1.0:
        issues.append(f"Line lengths are suspiciously uniform (stddev={stddev:.2f})")
        score += 10

    # Check for identical sentence starters (first word repetition)
    first_words = [line.split()[0].lower() for line in lines
                   if line.strip() and line.split()]
    if first_words:
        most_common_first, count = Counter(first_words).most_common(1)[0]
        ratio = count / len(first_words)
        if ratio > 0.5 and count >= 3:
            issues.append(f"Over-repetitive line starter '{most_common_first}' "
                          f"({count}/{len(first_words)} = {ratio:.0%})")
            score += 8

    # Check for "list poem" pattern — all lines follow same grammatical shape
    # Simplified: check if most lines start with the same POS-like pattern
    patterns = []
    for line in lines:
        words = line.strip().split()
        if len(words) >= 2:
            patterns.append(f"{words[0].lower()}_{len(words)}")
    if patterns:
        most_common_pat, pat_count = Counter(patterns).most_common(1)[0]
        if pat_count >= 4 and pat_count / len(patterns) > 0.4:
            issues.append("Many lines follow the same structural pattern (list poem)")
            score += 7

    return {"issues": issues, "score": score}


# ---------------------------------------------------------------------------
# Layer 5 — Artist Fingerprint Comparison
# ---------------------------------------------------------------------------

def _compare_fingerprint(text: str, fingerprint: Optional[Dict] = None
                         ) -> Dict[str, Any]:
    """
    Compare generated text statistics against the source artist's fingerprint.
    If no fingerprint is available, returns a neutral result.
    """
    issues: List[str] = []
    score = 0

    if fingerprint is None:
        return {"issues": issues, "score": score}

    words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text.lower())
    if len(words) < 10:
        return {"issues": issues, "score": score}

    # Vocabulary overlap
    safe_vocab = set(fingerprint.get("safe_vocabulary", []))
    if safe_vocab:
        word_set = set(words)
        overlap = len(word_set & safe_vocab) / len(word_set) if word_set else 0
        if overlap < 0.5:
            issues.append(f"Low vocabulary overlap with source artist "
                          f"({overlap:.0%} vs expected ≥50%)")
            score += 12

    # Type-Token Ratio comparison
    artist_ttr = fingerprint.get("type_token_ratio")
    if artist_ttr is not None:
        gen_ttr = len(set(words)) / len(words)
        ttr_diff = abs(gen_ttr - artist_ttr)
        if ttr_diff > 0.15:
            issues.append(f"TTR mismatch: generated={gen_ttr:.3f}, "
                          f"artist={artist_ttr:.3f} (diff={ttr_diff:.3f})")
            score += 8

    # Line length stddev comparison
    artist_line_stddev = fingerprint.get("line_length_stddev")
    if artist_line_stddev is not None:
        gen_lines = [l for l in text.split("\n") if l.strip()
                     and not l.strip().startswith("[")]
        if gen_lines:
            gen_wc = [len(l.split()) for l in gen_lines]
            gen_mean = sum(gen_wc) / len(gen_wc)
            gen_var = sum((w - gen_mean) ** 2 for w in gen_wc) / len(gen_wc)
            gen_stddev = math.sqrt(gen_var) if gen_var > 0 else 0
            stddev_diff = abs(gen_stddev - artist_line_stddev)
            if stddev_diff > 2.0:
                issues.append(f"Line length variation mismatch: "
                              f"generated stddev={gen_stddev:.1f}, "
                              f"artist={artist_line_stddev:.1f}")
                score += 6

    return {"issues": issues, "score": score}


# ---------------------------------------------------------------------------
# Layer 6 — Statistical Anomaly Detection
# ---------------------------------------------------------------------------

FUNCTION_WORDS = {"the", "a", "an", "of", "in", "to", "and", "is", "it",
                  "for", "on", "with", "as", "at", "by", "from", "or",
                  "but", "not", "be", "are", "was", "were", "been", "this",
                  "that", "which", "who", "what", "if", "so", "my", "your",
                  "we", "they", "i", "you", "he", "she", "me", "us"}


def _detect_anomalies(text: str) -> Dict[str, Any]:
    """Detect statistical anomalies in word usage."""
    issues: List[str] = []
    score = 0

    words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text.lower())
    if len(words) < 20:
        return {"issues": issues, "score": score}

    word_freq = Counter(words)
    total = len(words)
    unique = len(word_freq)

    # Hapax legomena ratio (words used exactly once)
    hapax = sum(1 for w, c in word_freq.items() if c == 1)
    hapax_ratio = hapax / unique if unique else 0
    # AI text typically has higher hapax ratio (too many unique words)
    if hapax_ratio > 0.85 and unique > 20:
        issues.append(f"Very high hapax ratio ({hapax_ratio:.2f}) — "
                      f"too many unique-once words")
        score += 6

    # Function word density
    func_count = sum(word_freq.get(fw, 0) for fw in FUNCTION_WORDS)
    func_ratio = func_count / total if total else 0
    # Real lyrics tend to have lower function word density than prose
    # AI-generated lyrics often read like prose (func ratio > 0.35)
    if func_ratio > 0.38:
        issues.append(f"High function word density ({func_ratio:.2f}) — "
                      f"reads more like prose than lyrics")
        score += 5

    # Sentence opener variety (first word of each line)
    lines = [l.strip() for l in text.split("\n")
             if l.strip() and not l.strip().startswith("[")
             and not l.strip().startswith("(")]
    if len(lines) >= 6:
        openers = [l.split()[0].lower() for l in lines if l.split()]
        opener_unique_ratio = len(set(openers)) / len(openers)
        if opener_unique_ratio < 0.3:
            issues.append(f"Very low line opener variety "
                          f"({opener_unique_ratio:.2f})")
            score += 5
        elif opener_unique_ratio > 0.95 and len(openers) > 8:
            # Paradoxically, PERFECT variety is also suspicious
            issues.append("Suspiciously perfect line opener variety — "
                          "real lyrics naturally repeat some starters")
            score += 3

    return {"issues": issues, "score": score}


# ===========================================================================
# Main Detector Class
# ===========================================================================

class AISlopDetector:
    """
    Enhanced AI-slop detection with 6-layer defense.

    Usage:
        detector = AISlopDetector()
        result = detector.scan_text("Some lyrics here...")
        print(result['severity'], result['ai_score'])
    """

    def __init__(self):
        self.blacklisted_words = BLACKLISTED_WORDS
        self.blacklisted_phrases = BLACKLISTED_PHRASES
        self.patterns = AI_PATTERNS

    # -- Layer 1 --
    def check_word(self, word: str) -> bool:
        """Check if a word is in the blacklist."""
        return word.lower().strip() in self.blacklisted_words

    # -- Layer 2 --
    def check_phrases(self, text: str) -> List[str]:
        """Return all blacklisted phrases found in text."""
        text_lower = text.lower()
        return [p for p in self.blacklisted_phrases if p in text_lower]

    # -- Layer 3 --
    def check_patterns(self, text: str) -> List[Tuple[str, str]]:
        """Return (pattern_desc, matched_text) for AI writing patterns."""
        found: List[Tuple[str, str]] = []
        for pat in self.patterns:
            for m in pat.finditer(text):
                found.append((pat.pattern, m.group()))
        return found

    # -- Full scan --
    def scan_text(self, text: str,
                  fingerprint: Optional[Dict] = None,
                  statistical_weight: float = 1.0) -> Dict[str, Any]:
        """
        Comprehensive 6-layer scan.

        Args:
            text: The text to scan.
            fingerprint: Optional artist fingerprint dict from the analyzer.
            statistical_weight: 0.0–1.0 multiplier for layers 4-6 (structural,
                fingerprint, statistical). Lower values reduce the impact of
                TTR mismatch, hapax ratio, line uniformity etc.

        Returns:
            Dict with keys: ai_score, severity, is_likely_ai, layers
        """
        # Remove section tags for analysis
        clean = re.sub(r"\[.*?\]", "", text)
        clean = re.sub(r"\(.*?\)", "", clean)  # Remove performance notes

        words = re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", clean)
        lines = [l.strip() for l in clean.split("\n")
                 if l.strip() and len(l.strip()) > 1]

        # Layer 1: Blacklisted words (full weight — always bad)
        bad_words = [w for w in words if self.check_word(w)]
        l1_score = len(bad_words) * 10

        # Layer 2: Blacklisted phrases (full weight — always bad)
        bad_phrases = self.check_phrases(clean)
        l2_score = len(bad_phrases) * 20

        # Layer 3: Regex patterns (full weight — always bad)
        bad_patterns = self.check_patterns(clean)
        l3_score = len(bad_patterns) * 5

        # Layer 4: Structural analysis (weighted)
        l4 = _analyze_structure(lines)

        # Layer 5: Fingerprint comparison (weighted)
        l5 = _compare_fingerprint(clean, fingerprint)

        # Layer 6: Statistical anomalies (weighted)
        l6 = _detect_anomalies(clean)

        # Apply weight to statistical layers (4-6)
        sw = max(0.0, min(1.0, statistical_weight))
        l4_weighted = int(l4["score"] * sw)
        l5_weighted = int(l5["score"] * sw)
        l6_weighted = int(l6["score"] * sw)

        total = l1_score + l2_score + l3_score + l4_weighted + l5_weighted + l6_weighted

        return {
            "ai_score": total,
            "severity": ("high" if total > 30
                         else "medium" if total > 15
                         else "low"),
            "is_likely_ai": total > 15,
            "layers": {
                "blacklisted_words": {
                    "score": l1_score,
                    "found": list(set(bad_words)),
                },
                "blacklisted_phrases": {
                    "score": l2_score,
                    "found": bad_phrases,
                },
                "pattern_matches": {
                    "score": l3_score,
                    "found": [(p, m) for p, m in bad_patterns[:10]],
                },
                "structural": {
                    "score": l4_weighted,
                    "raw_score": l4["score"],
                    "issues": l4["issues"],
                },
                "fingerprint": {
                    "score": l5_weighted,
                    "raw_score": l5["score"],
                    "issues": l5["issues"],
                },
                "statistical": {
                    "score": l6_weighted,
                    "raw_score": l6["score"],
                    "issues": l6["issues"],
                },
            },
        }

    # -- Vocabulary management --
    def generate_safe_vocabulary(self, analysis: Dict) -> Set[str]:
        """Generate a safe vocabulary from an analysis dict."""
        vocab = set(analysis.get("vocabulary", {}).get("all_words_list", []))
        return {w for w in vocab if not self.check_word(w)}

    def create_vocabulary_constraint(
        self, analysis_dicts: List[Dict], output_file: str
    ) -> Dict:
        """Create a vocabulary constraint file from multiple analyses."""
        all_safe: Set[str] = set()
        word_freq: Counter = Counter()

        for analysis in analysis_dicts:
            safe = self.generate_safe_vocabulary(analysis)
            all_safe.update(safe)

            freq = analysis.get("vocabulary", {}).get("word_frequency", {})
            for word, count in freq.items():
                if word in safe:
                    word_freq[word] += count

        constraint = {
            "total_safe_words": len(all_safe),
            "safe_vocabulary": sorted(all_safe),
            "word_frequencies": dict(word_freq.most_common(500)),
            "blacklisted_words": sorted(self.blacklisted_words),
            "blacklisted_phrases": sorted(self.blacklisted_phrases),
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(constraint, f, indent=2)

        return constraint
