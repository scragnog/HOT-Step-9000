"""
prompt_manager.py — File-backed prompt storage with runtime variable interpolation.

Prompts are stored as .txt files in the prompts/ directory alongside this module.
If a file exists, its content is used; otherwise, the default constant is returned.
Variables like {{BLACKLISTED_WORDS}} are interpolated at runtime.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Registry of prompt names → their Python-default values (set below after constants are defined)
_DEFAULTS: dict[str, str] = {}


def _ensure_dir():
    _PROMPTS_DIR.mkdir(exist_ok=True)


def register_default(name: str, default_text: str) -> None:
    """Register the in-code default for a named prompt."""
    _DEFAULTS[name] = default_text


def load_prompt(name: str, variables: dict[str, str] | None = None) -> str:
    """Load a prompt by name.

    1. If prompts/{name}.txt exists, read it.
    2. Otherwise, use the registered default.
    3. Interpolate any {{VAR}} placeholders from `variables`.
    """
    file_path = _PROMPTS_DIR / f"{name}.txt"

    if file_path.exists():
        text = file_path.read_text(encoding="utf-8")
        logger.debug("Loaded prompt '%s' from file", name)
    elif name in _DEFAULTS:
        text = _DEFAULTS[name]
        logger.debug("Loaded prompt '%s' from default", name)
    else:
        logger.warning("No prompt found for '%s'", name)
        return ""

    if variables:
        for key, value in variables.items():
            text = text.replace(f"{{{{{key}}}}}", value)

    return text


def save_prompt(name: str, content: str) -> Path:
    """Save a prompt to prompts/{name}.txt."""
    _ensure_dir()
    file_path = _PROMPTS_DIR / f"{name}.txt"
    file_path.write_text(content, encoding="utf-8")
    logger.info("Saved prompt '%s' to %s", name, file_path)
    return file_path


def list_prompts() -> list[dict[str, str]]:
    """List all known prompts with name, source, and content."""
    results = []
    all_names = set(_DEFAULTS.keys())

    # Also include any .txt files in prompts/ dir
    if _PROMPTS_DIR.exists():
        for f in _PROMPTS_DIR.glob("*.txt"):
            all_names.add(f.stem)

    for name in sorted(all_names):
        file_path = _PROMPTS_DIR / f"{name}.txt"
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            source = "file"
        elif name in _DEFAULTS:
            content = _DEFAULTS[name]
            source = "default"
        else:
            continue

        results.append({
            "name": name,
            "source": source,
            "content": content,
            "has_default": name in _DEFAULTS,
        })

    return results


def reset_prompt(name: str) -> bool:
    """Delete the file override, reverting to the default."""
    file_path = _PROMPTS_DIR / f"{name}.txt"
    if file_path.exists():
        file_path.unlink()
        logger.info("Reset prompt '%s' to default", name)
        return True
    return False
