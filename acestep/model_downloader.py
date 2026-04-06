"""
ACE-Step Model Downloader

This module provides functionality to download models from HuggingFace Hub or ModelScope.
It supports automatic downloading when models are not found locally,
with intelligent fallback between download sources.
"""

import os
import sys
import hashlib
import shutil
import argparse
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from loguru import logger


# =============================================================================
# Model Code File Sync (GitHub repo -> checkpoint directories)
# =============================================================================

# Mapping from checkpoint directory name to source model variant in acestep/models/
_CHECKPOINT_TO_VARIANT: Dict[str, str] = {
    "acestep-v15-turbo": "turbo",
    "acestep-v15-sft": "sft",
    "acestep-v15-base": "base",
    # SFT variants (base-SFT uses the same model code as SFT)
    "acestep-v15-base-sft-fix-inst": "sft",
    # Turbo variants all share the turbo model code
    "acestep-v15-turbo-shift1": "turbo",
    "acestep-v15-turbo-shift3": "turbo",
    "acestep-v15-turbo-continuous": "turbo",
    "acestep-v15-turbo-fix-inst-shift3": "turbo",
    "acestep-v15-turbo-fix-inst-shift-continuous": "turbo",
    "acestep-v15-turbo-fix-inst-shift-dynamic": "turbo",
    "acestep-v15-turbo-rl": "turbo",
    # XL (4B DiT) variants have their own model code under acestep/models/xl_*/
    "acestep-v15-xl-base": "xl_base",
    "acestep-v15-xl-sft": "xl_sft",
    "acestep-v15-xl-turbo": "xl_turbo",
}

# Weight file extensions that transformers/diffusers look for when loading models
_WEIGHT_FILE_EXTENSIONS = {".safetensors", ".bin", ".ckpt", ".h5", ".msgpack"}

_INCOMPLETE_DOWNLOAD_GUIDANCE = (
    "\n"
    "============================================================\n"
    "  INCOMPLETE MODEL DOWNLOAD DETECTED\n"
    "============================================================\n"
    "  The model directory exists but is missing weight files.\n"
    "  This usually means a previous download was interrupted.\n"
    "\n"
    "  To fix this, try one of:\n"
    "    1. Re-run install.bat (recommended)\n"
    "    2. Run: python -m acestep.model_downloader --force\n"
    "    3. Delete the checkpoints folder and re-run install.bat\n"
    "============================================================\n"
)


def _has_weight_files(model_dir: Path) -> bool:
    """Check if a model directory contains actual weight files.

    A directory that exists but has no weight files (e.g. only config.json)
    indicates an interrupted download and should be treated as missing.
    """
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    for f in model_dir.iterdir():
        if f.is_file() and f.suffix in _WEIGHT_FILE_EXTENSIONS:
            return True
    return False


def _get_models_source_dir() -> Path:
    """Get the acestep/models/ directory (authoritative source for model code)."""
    return Path(__file__).resolve().parent / "models"


def _file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_code_mismatch(model_name: str, checkpoints_dir) -> List[str]:
    """
    Compare .py files in acestep/models/{variant}/ with those in the checkpoint directory.

    Args:
        model_name: Checkpoint directory name (e.g. "acestep-v15-turbo")
        checkpoints_dir: Path to the checkpoints root directory

    Returns:
        List of filenames that differ (empty list if all match or model_name is unknown)
    """
    variant = _CHECKPOINT_TO_VARIANT.get(model_name)
    if variant is None:
        return []

    source_dir = _get_models_source_dir() / variant
    if not source_dir.exists():
        return []

    if isinstance(checkpoints_dir, str):
        checkpoints_dir = Path(checkpoints_dir)
    target_dir = checkpoints_dir / model_name

    mismatched = []
    for src_file in source_dir.glob("*.py"):
        if src_file.name == "__init__.py":
            continue
        dst_file = target_dir / src_file.name
        if not dst_file.exists():
            mismatched.append(src_file.name)
        elif _file_hash(src_file) != _file_hash(dst_file):
            mismatched.append(src_file.name)

    return mismatched


def _sync_model_code_files(model_name: str, checkpoints_dir) -> List[str]:
    """
    Copy .py files from acestep/models/{variant}/ into the checkpoint directory,
    overwriting the HuggingFace-downloaded versions.

    Args:
        model_name: Checkpoint directory name (e.g. "acestep-v15-turbo")
        checkpoints_dir: Path to the checkpoints root directory

    Returns:
        List of filenames that were synced (empty if model_name is unknown or no source)
    """
    variant = _CHECKPOINT_TO_VARIANT.get(model_name)
    if variant is None:
        return []

    source_dir = _get_models_source_dir() / variant
    if not source_dir.exists():
        logger.warning(f"[Model Sync] Source directory not found: {source_dir}")
        return []

    if isinstance(checkpoints_dir, str):
        checkpoints_dir = Path(checkpoints_dir)
    target_dir = checkpoints_dir / model_name
    if not target_dir.exists():
        logger.warning(f"[Model Sync] Target directory not found: {target_dir}")
        return []

    synced = []
    for src_file in source_dir.glob("*.py"):
        if src_file.name == "__init__.py":
            continue
        dst_file = target_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        synced.append(src_file.name)
        logger.debug(f"[Model Sync] Synced {src_file.name} -> {dst_file}")

    return synced


# =============================================================================
# Network Detection & Smart Download
# =============================================================================

def _can_access_google(timeout: float = 3.0) -> bool:
    """
    Check if Google is accessible (to determine HuggingFace vs ModelScope).

    Args:
        timeout: Connection timeout in seconds

    Returns:
        True if Google is accessible, False otherwise
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(("www.google.com", 443))
        return True
    except (socket.timeout, socket.error, OSError):
        return False
    finally:
        sock.close()


def _download_from_huggingface_internal(
    repo_id: str,
    local_dir: Path,
    token: Optional[str] = None,
) -> None:
    """
    Internal function to download from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "ACE-Step/Ace-Step1.5")
        local_dir: Local directory to save the model
        token: HuggingFace token for private repos (optional)

    Raises:
        Exception: If download fails
    """
    from huggingface_hub import snapshot_download

    logger.info(f"[Model Download] Downloading from HuggingFace: {repo_id} -> {local_dir}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
    )


def _download_from_modelscope_internal(
    repo_id: str,
    local_dir: Path,
) -> None:
    """
    Internal function to download from ModelScope.

    Args:
        repo_id: ModelScope repository ID (e.g., "ACE-Step/Ace-Step1.5")
        local_dir: Local directory to save the model

    Raises:
        Exception: If download fails
    """
    from modelscope import snapshot_download

    logger.info(f"[Model Download] Downloading from ModelScope: {repo_id} -> {local_dir}")

    snapshot_download(
        model_id=repo_id,
        local_dir=str(local_dir),
    )


def _smart_download(
    repo_id: str,
    local_dir: Path,
    token: Optional[str] = None,
    prefer_source: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Smart download with automatic fallback between HuggingFace and ModelScope.

    Automatically detects network environment and chooses the best download source.
    If the primary source fails, automatically falls back to the alternative.

    Args:
        repo_id: Repository ID (same format for both HF and ModelScope)
        local_dir: Local directory to save the model
        token: HuggingFace token for private repos (optional)
        prefer_source: Preferred download source ("huggingface", "modelscope", or None for auto-detect)

    Returns:
        Tuple of (success, message)
    """
    # Ensure directory exists
    local_dir.mkdir(parents=True, exist_ok=True)

    # Determine primary source
    if prefer_source == "huggingface":
        use_huggingface_first = True
        logger.info("[Model Download] User preference: HuggingFace Hub")
    elif prefer_source == "modelscope":
        use_huggingface_first = False
        logger.info("[Model Download] User preference: ModelScope")
    else:
        # Auto-detect network environment
        can_access_google = _can_access_google()
        use_huggingface_first = can_access_google
        logger.info(f"[Model Download] Auto-detected: {'HuggingFace Hub' if can_access_google else 'ModelScope'}")

    if use_huggingface_first:
        logger.info("[Model Download] Using HuggingFace Hub...")
        try:
            _download_from_huggingface_internal(repo_id, local_dir, token)
            return True, f"Successfully downloaded from HuggingFace: {repo_id}"
        except Exception as e:
            logger.warning(f"[Model Download] HuggingFace download failed: {e}")
            logger.info("[Model Download] Falling back to ModelScope...")
            try:
                _download_from_modelscope_internal(repo_id, local_dir)
                return True, f"Successfully downloaded from ModelScope: {repo_id}"
            except Exception as e2:
                error_msg = f"Both HuggingFace and ModelScope downloads failed. HF: {e}, MS: {e2}"
                logger.error(error_msg)
                return False, error_msg
    else:
        logger.info("[Model Download] Using ModelScope...")
        try:
            _download_from_modelscope_internal(repo_id, local_dir)
            return True, f"Successfully downloaded from ModelScope: {repo_id}"
        except Exception as e:
            logger.warning(f"[Model Download] ModelScope download failed: {e}")
            logger.info("[Model Download] Falling back to HuggingFace Hub...")
            try:
                _download_from_huggingface_internal(repo_id, local_dir, token)
                return True, f"Successfully downloaded from HuggingFace: {repo_id}"
            except Exception as e2:
                error_msg = f"Both ModelScope and HuggingFace downloads failed. MS: {e}, HF: {e2}"
                logger.error(error_msg)
                return False, error_msg


# =============================================================================
# Model Registry
# =============================================================================
# Main model contains core components (vae, text_encoder, default DiT)
MAIN_MODEL_REPO = "ACE-Step/Ace-Step1.5"

# Sub-models that can be downloaded separately into the checkpoints directory
SUBMODEL_REGISTRY: Dict[str, str] = {
    # LM models
    "acestep-5Hz-lm-0.6B": "ACE-Step/acestep-5Hz-lm-0.6B",
    "acestep-5Hz-lm-4B": "ACE-Step/acestep-5Hz-lm-4B",
    # DiT models
    "acestep-v15-turbo-shift3": "ACE-Step/acestep-v15-turbo-shift3",
    "acestep-v15-sft": "ACE-Step/acestep-v15-sft",
    "acestep-v15-base": "ACE-Step/acestep-v15-base",
    "acestep-v15-turbo-shift1": "ACE-Step/acestep-v15-turbo-shift1",
    "acestep-v15-turbo-continuous": "ACE-Step/acestep-v15-turbo-continuous",
    # XL (4B DiT) models — auto-download from HuggingFace when selected
    "acestep-v15-xl-base": "ACE-Step/acestep-v15-xl-base",
    "acestep-v15-xl-sft": "ACE-Step/acestep-v15-xl-sft",
    "acestep-v15-xl-turbo": "ACE-Step/acestep-v15-xl-turbo",
}

# Components that come from the main model repo (ACE-Step/Ace-Step1.5)
MAIN_MODEL_COMPONENTS = [
    "acestep-v15-turbo",      # Default DiT model
    "vae",                     # VAE for audio encoding/decoding
    "Qwen3-Embedding-0.6B",    # Text encoder
    "acestep-5Hz-lm-1.7B",     # Default LM model (1.7B)
]

# Default LM model (included in main model)
DEFAULT_LM_MODEL = "acestep-5Hz-lm-1.7B"

# GGUF quantized LM models (for llama-cpp-python backend)
# Maps LM model name -> quant -> (repo_id, filename, approx_file_size_bytes)
# Ordered: smallest VRAM first within each model
GGUF_REGISTRY: Dict[str, Dict[str, Tuple[str, str, int]]] = {
    "acestep-5Hz-lm-0.6B": {
        "Q8_0": ("Serveurperso/ACE-Step-1.5-GGUF", "acestep-5Hz-lm-0.6B-Q8_0.gguf", 710_000_000),
        "BF16": ("Serveurperso/ACE-Step-1.5-GGUF", "acestep-5Hz-lm-0.6B-BF16.gguf", 1_340_000_000),
    },
    "acestep-5Hz-lm-1.7B": {
        "Q8_0": ("Serveurperso/ACE-Step-1.5-GGUF", "acestep-5Hz-lm-1.7B-Q8_0.gguf", 1_980_000_000),
        "BF16": ("Serveurperso/ACE-Step-1.5-GGUF", "acestep-5Hz-lm-1.7B-BF16.gguf", 3_740_000_000),
    },
    "acestep-5Hz-lm-4B": {
        "Q5_K_M": ("Serveurperso/ACE-Step-1.5-GGUF", "acestep-5Hz-lm-4B-Q5_K_M.gguf", 3_030_000_000),
        "Q6_K": ("Serveurperso/ACE-Step-1.5-GGUF", "acestep-5Hz-lm-4B-Q6_K.gguf", 3_500_000_000),
        "Q8_0": ("Serveurperso/ACE-Step-1.5-GGUF", "acestep-5Hz-lm-4B-Q8_0.gguf", 4_530_000_000),
        "BF16": ("Serveurperso/ACE-Step-1.5-GGUF", "acestep-5Hz-lm-4B-BF16.gguf", 8_540_000_000),
    },
}


def get_project_root() -> Path:
    """Get the project root directory.

    Returns the directory set by the ``ACESTEP_PROJECT_ROOT`` environment
    variable when present, otherwise the current working directory.  Using
    the working directory (rather than ``__file__``) keeps the checkpoints
    folder next to where the user launched the process, regardless of whether
    the package was installed via ``pip install .`` or run from source.
    """
    env_root = os.environ.get("ACESTEP_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(os.getcwd())


def get_checkpoints_dir(custom_dir: Optional[str] = None) -> Path:
    """Get the checkpoints directory path."""
    if custom_dir:
        return Path(custom_dir)
    return get_project_root() / "checkpoints"


def check_main_model_exists(checkpoints_dir: Optional[Path] = None) -> bool:
    """
    Check if the main model components exist in the checkpoints directory.

    Validates that each component directory contains actual weight files,
    not just metadata/config from an interrupted download.

    Returns:
        True if all main model components exist with weight files, False otherwise.
    """
    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    elif isinstance(checkpoints_dir, str):
        checkpoints_dir = Path(checkpoints_dir)

    for component in MAIN_MODEL_COMPONENTS:
        component_path = checkpoints_dir / component
        if not component_path.exists():
            return False
        if not _has_weight_files(component_path):
            logger.warning(
                f"[Model Check] Directory exists but missing weight files: {component_path}"
            )
            print(_INCOMPLETE_DOWNLOAD_GUIDANCE)
            return False
    return True


def check_model_exists(model_name: str, checkpoints_dir: Optional[Path] = None) -> bool:
    """
    Check if a specific model exists in the checkpoints directory.

    Validates that the model directory contains actual weight files,
    not just metadata/config from an interrupted download.

    Args:
        model_name: Name of the model to check
        checkpoints_dir: Custom checkpoints directory (optional)

    Returns:
        True if the model exists with weight files, False otherwise.
    """
    if not model_name:
        logger.warning("[check_model_exists] Empty model_name; treating as missing.")
        return False
    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    elif isinstance(checkpoints_dir, str):
        checkpoints_dir = Path(checkpoints_dir)

    model_path = checkpoints_dir / model_name
    if not model_path.exists():
        return False
    if not _has_weight_files(model_path):
        logger.warning(
            f"[Model Check] Directory exists but missing weight files: {model_path}"
        )
        print(_INCOMPLETE_DOWNLOAD_GUIDANCE)
        return False
    return True


def list_available_models() -> Dict[str, str]:
    """
    List all available models for download.
    
    Returns:
        Dictionary mapping local names to HuggingFace repo IDs.
    """
    models = {
        "main": MAIN_MODEL_REPO,
        **SUBMODEL_REGISTRY
    }
    return models


def download_main_model(
    checkpoints_dir: Optional[Path] = None,
    force: bool = False,
    token: Optional[str] = None,
    prefer_source: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Download the main ACE-Step model from HuggingFace or ModelScope.

    The main model includes:
    - acestep-v15-turbo (default DiT model)
    - vae (audio encoder/decoder)
    - Qwen3-Embedding-0.6B (text encoder)
    - acestep-5Hz-lm-1.7B (default LM model)

    Args:
        checkpoints_dir: Custom checkpoints directory (optional)
        force: Force re-download even if model exists
        token: HuggingFace token for private repos (optional)
        prefer_source: Preferred download source ("huggingface", "modelscope", or None for auto-detect)

    Returns:
        Tuple of (success, message)
    """
    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    elif isinstance(checkpoints_dir, str):
        checkpoints_dir = Path(checkpoints_dir)

    # Ensure checkpoints directory exists
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if not force and check_main_model_exists(checkpoints_dir):
        return True, f"Main model already exists at {checkpoints_dir}"

    print(f"Downloading main model from {MAIN_MODEL_REPO}...")
    print(f"Destination: {checkpoints_dir}")
    print("This may take a while depending on your internet connection...")

    # Use smart download with automatic fallback
    success, msg = _smart_download(MAIN_MODEL_REPO, checkpoints_dir, token, prefer_source)
    if success:
        # Sync model code files for all DiT components in the main model
        for component in MAIN_MODEL_COMPONENTS:
            if component in _CHECKPOINT_TO_VARIANT:
                synced = _sync_model_code_files(component, checkpoints_dir)
                if synced:
                    logger.info(f"[Model Download] Synced code files for {component}: {synced}")
    return success, msg


def download_submodel(
    model_name: str,
    checkpoints_dir: Optional[Path] = None,
    force: bool = False,
    token: Optional[str] = None,
    prefer_source: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Download a specific sub-model from HuggingFace or ModelScope.

    Args:
        model_name: Name of the model to download (must be in SUBMODEL_REGISTRY)
        checkpoints_dir: Custom checkpoints directory (optional)
        force: Force re-download even if model exists
        token: HuggingFace token for private repos (optional)
        prefer_source: Preferred download source ("huggingface", "modelscope", or None for auto-detect)

    Returns:
        Tuple of (success, message)
    """
    if model_name not in SUBMODEL_REGISTRY:
        available = ", ".join(SUBMODEL_REGISTRY.keys())
        return False, f"Unknown model '{model_name}'. Available models: {available}"

    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    elif isinstance(checkpoints_dir, str):
        checkpoints_dir = Path(checkpoints_dir)

    # Ensure checkpoints directory exists
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoints_dir / model_name

    if not force and model_path.exists():
        if not _has_weight_files(model_path):
            logger.warning(
                f"[Model Download] Ghost directory detected for '{model_name}' "
                f"at {model_path} — re-downloading..."
            )
            print(_INCOMPLETE_DOWNLOAD_GUIDANCE)
        else:
            return True, f"Model '{model_name}' already exists at {model_path}"

    repo_id = SUBMODEL_REGISTRY[model_name]

    print(f"Downloading {model_name} from {repo_id}...")
    print(f"Destination: {model_path}")

    # Use smart download with automatic fallback
    success, msg = _smart_download(repo_id, model_path, token, prefer_source)
    if success and model_name in _CHECKPOINT_TO_VARIANT:
        # Sync model code files after successful download
        synced = _sync_model_code_files(model_name, checkpoints_dir)
        if synced:
            logger.info(f"[Model Download] Synced code files for {model_name}: {synced}")
    return success, msg


def download_all_models(
    checkpoints_dir: Optional[Path] = None,
    force: bool = False,
    token: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """
    Download all available models.
    
    Args:
        checkpoints_dir: Custom checkpoints directory (optional)
        force: Force re-download even if models exist
        token: HuggingFace token for private repos (optional)
    
    Returns:
        Tuple of (all_success, list of messages)
    """
    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    elif isinstance(checkpoints_dir, str):
        checkpoints_dir = Path(checkpoints_dir)

    messages = []
    all_success = True
    
    # Download main model first
    success, msg = download_main_model(checkpoints_dir, force, token)
    messages.append(msg)
    if not success:
        all_success = False
    
    # Download all sub-models
    for model_name in SUBMODEL_REGISTRY:
        success, msg = download_submodel(model_name, checkpoints_dir, force, token)
        messages.append(msg)
        if not success:
            all_success = False
    
    return all_success, messages


def ensure_main_model(
    checkpoints_dir: Optional[Path] = None,
    token: Optional[str] = None,
    prefer_source: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Ensure the main model is available, downloading if necessary.

    This function is designed to be called during initialization.
    It will only download if the model doesn't exist.

    Args:
        checkpoints_dir: Custom checkpoints directory (optional)
        token: HuggingFace token for private repos (optional)
        prefer_source: Preferred download source ("huggingface", "modelscope", or None for auto-detect)

    Returns:
        Tuple of (success, message)
    """
    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()

    if check_main_model_exists(checkpoints_dir):
        return True, "Main model is available"

    print("\n" + "=" * 60)
    print("Main model not found. Starting automatic download...")
    print("=" * 60 + "\n")

    return download_main_model(checkpoints_dir, token=token, prefer_source=prefer_source)


def ensure_lm_model(
    model_name: Optional[str] = None,
    checkpoints_dir: Optional[Path] = None,
    token: Optional[str] = None,
    prefer_source: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Ensure an LM model is available, downloading if necessary.

    Args:
        model_name: Name of the LM model (defaults to DEFAULT_LM_MODEL)
        checkpoints_dir: Custom checkpoints directory (optional)
        token: HuggingFace token for private repos (optional)
        prefer_source: Preferred download source ("huggingface", "modelscope", or None for auto-detect)

    Returns:
        Tuple of (success, message)
    """
    if model_name is None:
        model_name = DEFAULT_LM_MODEL

    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    elif isinstance(checkpoints_dir, str):
        checkpoints_dir = Path(checkpoints_dir)

    if check_model_exists(model_name, checkpoints_dir):
        return True, f"LM model '{model_name}' is available"

    # Check if this is a known LM model
    if model_name not in SUBMODEL_REGISTRY:
        # Check if it might be a variant name
        for known_model in SUBMODEL_REGISTRY:
            if "lm" in known_model.lower() and model_name.lower() in known_model.lower():
                model_name = known_model
                break
        else:
            return False, f"Unknown LM model: {model_name}"

    print("\n" + "=" * 60)
    print(f"LM model '{model_name}' not found. Starting automatic download...")
    print("=" * 60 + "\n")

    return download_submodel(model_name, checkpoints_dir, token=token, prefer_source=prefer_source)


def ensure_dit_model(
    model_name: str,
    checkpoints_dir: Optional[Path] = None,
    token: Optional[str] = None,
    prefer_source: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Ensure a DiT model is available, downloading if necessary.

    Args:
        model_name: Name of the DiT model
        checkpoints_dir: Custom checkpoints directory (optional)
        token: HuggingFace token for private repos (optional)
        prefer_source: Preferred download source ("huggingface", "modelscope", or None for auto-detect)

    Returns:
        Tuple of (success, message)
    """
    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    elif isinstance(checkpoints_dir, str):
        checkpoints_dir = Path(checkpoints_dir)

    if check_model_exists(model_name, checkpoints_dir):
        return True, f"DiT model '{model_name}' is available"

    # Check if this is the default turbo model (part of main)
    if model_name == "acestep-v15-turbo":
        return ensure_main_model(checkpoints_dir, token, prefer_source)

    # Check if it's a known sub-model
    if model_name in SUBMODEL_REGISTRY:
        print("\n" + "=" * 60)
        print(f"DiT model '{model_name}' not found. Starting automatic download...")
        print("=" * 60 + "\n")
        return download_submodel(model_name, checkpoints_dir, token=token, prefer_source=prefer_source)

    if not model_name:
        return False, "Unknown DiT model: '' (pass None for default or choose a valid model)"
    return False, f"Unknown DiT model: {model_name}"


def download_gguf_model(
    lm_model_name: str,
    quant: Optional[str] = None,
    checkpoints_dir: Optional[Path] = None,
    force: bool = False,
) -> Tuple[bool, str]:
    """Download a GGUF quantized LM model file.

    Places the GGUF file inside the corresponding LM model directory
    so that ``_find_gguf_file()`` in ``llm_inference.py`` discovers it.

    Args:
        lm_model_name: LM model name (e.g. "acestep-5Hz-lm-4B")
        quant: Quantization type (e.g. "Q5_K_M", "Q8_0"). None = best available.
        checkpoints_dir: Custom checkpoints directory
        force: Force re-download even if file exists

    Returns:
        (success, message)
    """
    if lm_model_name not in GGUF_REGISTRY:
        available = ", ".join(GGUF_REGISTRY.keys())
        return False, f"No GGUF models available for '{lm_model_name}'. Available: {available}"

    quant_options = GGUF_REGISTRY[lm_model_name]
    if quant is None:
        # Pick best available
        quant = next(iter(quant_options))
    elif quant not in quant_options:
        available_quants = ", ".join(quant_options.keys())
        return False, f"Quant '{quant}' not available for {lm_model_name}. Available: {available_quants}"

    repo_id, filename, approx_size = quant_options[quant]

    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    elif isinstance(checkpoints_dir, str):
        checkpoints_dir = Path(checkpoints_dir)

    # Place GGUF file inside the LM model directory
    model_dir = checkpoints_dir / lm_model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    dest_path = model_dir / filename

    if not force and dest_path.exists() and dest_path.stat().st_size > 0:
        return True, f"GGUF model already exists: {dest_path}"

    size_gb = approx_size / (1024 ** 3)
    print(f"Downloading GGUF model: {filename} (~{size_gb:.1f} GB)")
    print(f"  Source: {repo_id}")
    print(f"  Destination: {dest_path}")

    try:
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        logger.info(f"[GGUF Download] Downloaded: {downloaded_path}")
        return True, f"✅ GGUF model downloaded: {dest_path}"
    except Exception as e:
        return False, f"❌ Failed to download GGUF model: {e}"


def ensure_gguf_model(
    lm_model_name: str,
    quant: Optional[str] = None,
    checkpoints_dir: Optional[Path] = None,
) -> Tuple[bool, str]:
    """Ensure a GGUF model is available, downloading if necessary."""
    if checkpoints_dir is None:
        checkpoints_dir = get_checkpoints_dir()
    elif isinstance(checkpoints_dir, str):
        checkpoints_dir = Path(checkpoints_dir)

    # Check if any GGUF file already exists in the model dir
    model_dir = checkpoints_dir / lm_model_name
    if model_dir.exists():
        import glob
        existing = glob.glob(str(model_dir / "*.gguf"))
        if existing:
            return True, f"GGUF model available: {existing[0]}"

    return download_gguf_model(lm_model_name, quant, checkpoints_dir)


def print_model_list():
    """Print formatted list of available models."""
    print("\nAvailable Models for Download:")
    print("=" * 60)
    print("\nSupported Sources: HuggingFace Hub <-> ModelScope (auto-fallback)")

    print("\n[Main Model]")
    print(f"  main -> {MAIN_MODEL_REPO}")
    print("  Contains: vae, Qwen3-Embedding-0.6B, acestep-v15-turbo, acestep-5Hz-lm-1.7B")

    print("\n[Optional LM Models]")
    for name, repo in SUBMODEL_REGISTRY.items():
        if "lm" in name.lower():
            print(f"  {name} -> {repo}")

    print("\n[Optional DiT Models]")
    for name, repo in SUBMODEL_REGISTRY.items():
        if "lm" not in name.lower():
            print(f"  {name} -> {repo}")

    print("\n" + "=" * 60)


def main():
    """CLI entry point for model downloading."""
    parser = argparse.ArgumentParser(
        description="Download ACE-Step models with automatic fallback (HuggingFace <-> ModelScope)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  acestep-download                          # Download main model (includes LM 1.7B)
  acestep-download --all                    # Download all available models
  acestep-download --model acestep-v15-sft  # Download a specific model
  acestep-download --list                   # List all available models

Network Detection:
  Automatically detects network environment and chooses the best download source:
  - Google accessible -> HuggingFace (fallback to ModelScope)
  - Google blocked -> ModelScope (fallback to HuggingFace)

Alternative using huggingface-cli:
  huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir ./checkpoints
  huggingface-cli download ACE-Step/acestep-5Hz-lm-0.6B --local-dir ./checkpoints/acestep-5Hz-lm-0.6B
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Specific model to download (use --list to see available models)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all available models"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available models"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default=None,
        help="Custom checkpoints directory (default: ./checkpoints)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if model exists"
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=None,
        help="HuggingFace token for private repos"
    )
    parser.add_argument(
        "--skip-main",
        action="store_true",
        help="Skip downloading the main model (only download specified sub-model)"
    )
    parser.add_argument(
        "--gguf",
        type=str,
        nargs="?",
        const="acestep-5Hz-lm-4B",
        metavar="LM_MODEL",
        help="Download GGUF model for llama-cpp backend (default: acestep-5Hz-lm-4B)",
    )
    parser.add_argument(
        "--gguf-quant",
        type=str,
        default=None,
        help="GGUF quantization type (e.g. Q5_K_M, Q8_0). Default: best available.",
    )

    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        print_model_list()
        return 0
    
    # Handle --gguf
    if args.gguf:
        checkpoints_dir = get_checkpoints_dir(args.dir) if args.dir else get_checkpoints_dir()
        success, msg = download_gguf_model(args.gguf, args.gguf_quant, checkpoints_dir, args.force)
        print(msg)
        return 0 if success else 1

    # Get checkpoints directory
    checkpoints_dir = get_checkpoints_dir(args.dir) if args.dir else get_checkpoints_dir()
    print(f"Checkpoints directory: {checkpoints_dir}")
    
    # Handle --all
    if args.all:
        success, messages = download_all_models(checkpoints_dir, args.force, args.token)
        for msg in messages:
            print(msg)
        return 0 if success else 1
    
    # Handle --model
    if args.model:
        if args.model == "main":
            success, msg = download_main_model(checkpoints_dir, args.force, args.token)
        elif args.model in SUBMODEL_REGISTRY:
            # Download main model first if needed (unless --skip-main)
            if not args.skip_main and not check_main_model_exists(checkpoints_dir):
                print("Main model not found. Downloading main model first...")
                main_success, main_msg = download_main_model(checkpoints_dir, args.force, args.token)
                print(main_msg)
                if not main_success:
                    return 1
            
            success, msg = download_submodel(args.model, checkpoints_dir, args.force, args.token)
        else:
            print(f"Unknown model: {args.model}")
            print("Use --list to see available models")
            return 1
        
        print(msg)
        return 0 if success else 1
    
    # Default: download main model (includes default LM 1.7B)
    print("Downloading main model (includes vae, text encoder, DiT, and LM 1.7B)...")
    
    # Download main model
    success, msg = download_main_model(checkpoints_dir, args.force, args.token)
    print(msg)
    
    if success:
        print("\nDownload complete!")
        print(f"Models are available at: {checkpoints_dir}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
