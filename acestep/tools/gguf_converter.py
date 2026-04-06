"""Local GGUF conversion from safetensors models.

Orchestrates the full pipeline:
- Direct: safetensors → Q8_0 or BF16 via convert_hf_to_gguf.py
- Two-step: safetensors → BF16 → Q4_K_M/Q5_K_M/Q6_K via llama-quantize

The llama-quantize binary is auto-downloaded from llama.cpp GitHub releases
on first use.
"""

import glob
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Quantization types that convert_hf_to_gguf.py can produce directly
DIRECT_QUANTS = {"q8_0", "bf16", "f16"}

# Quantization types that require llama-quantize binary (two-step via BF16)
BINARY_QUANTS = {"Q4_K_M", "Q5_K_M", "Q6_K"}

# All supported quant types, ordered by size (smallest first)
ALL_QUANTS = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "BF16"]

# Approximate output sizes per quant type for the 4B model (for UI display)
QUANT_INFO = {
    "Q4_K_M": {"label": "Q4_K_M — smallest", "bpw": 4.5},
    "Q5_K_M": {"label": "Q5_K_M — best balance ⭐", "bpw": 5.8},
    "Q6_K":   {"label": "Q6_K — high quality", "bpw": 6.6},
    "Q8_0":   {"label": "Q8_0 — near-lossless", "bpw": 8.5},
    "BF16":   {"label": "BF16 — full precision", "bpw": 16.0},
}

# llama.cpp release info for binary download
LLAMACPP_RELEASE_TAG = "b8672"
LLAMACPP_BIN_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_RELEASE_TAG}"
    f"/llama-{LLAMACPP_RELEASE_TAG}-bin-win-vulkan-x64.zip"
)
# Minimal set of files needed for llama-quantize (CPU only)
REQUIRED_BINARIES = [
    "llama-quantize.exe",
    "llama.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cpu-haswell.dll",
    "ggml-cpu-alderlake.dll",
    "ggml-cpu-skylakex.dll",
    "ggml-cpu-x64.dll",
    "libomp140.x86_64.dll",
]


def _get_tools_dir() -> Path:
    """Get the acestep/tools directory."""
    return Path(__file__).parent


def _get_bin_dir() -> Path:
    """Get the binary tools directory (acestep/tools/bin/)."""
    return _get_tools_dir() / "bin"


def _get_converter_script() -> Path:
    """Get path to the bundled convert_hf_to_gguf.py."""
    return _get_tools_dir() / "convert_hf_to_gguf.py"


def _get_checkpoints_dir() -> Path:
    """Get the checkpoints directory."""
    project_root = os.environ.get("ACESTEP_PROJECT_ROOT", os.getcwd())
    return Path(project_root) / "checkpoints"


def ensure_quantize_binary(
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """Ensure llama-quantize binary is available, downloading if needed.

    Returns (success, message_or_path).
    """
    if platform.system() != "Windows":
        return False, "llama-quantize binary auto-download only supported on Windows"

    bin_dir = _get_bin_dir()
    quantize_exe = bin_dir / "llama-quantize.exe"

    if quantize_exe.exists():
        return True, str(quantize_exe)

    # Download from GitHub releases
    if progress_callback:
        progress_callback(f"Downloading llama-quantize from llama.cpp {LLAMACPP_RELEASE_TAG}...")

    bin_dir.mkdir(parents=True, exist_ok=True)

    try:
        if progress_callback:
            progress_callback(f"Fetching {LLAMACPP_BIN_URL}...")

        data = urllib.request.urlopen(LLAMACPP_BIN_URL).read()
        z = zipfile.ZipFile(BytesIO(data))

        extracted = []
        for member in z.namelist():
            basename = os.path.basename(member)
            if basename in REQUIRED_BINARIES:
                # Extract directly into bin_dir with flat structure
                target = bin_dir / basename
                with z.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                extracted.append(basename)
                if progress_callback:
                    progress_callback(f"  Extracted: {basename}")

        z.close()

        if not quantize_exe.exists():
            return False, f"llama-quantize.exe not found in release archive. Extracted: {extracted}"

        if progress_callback:
            progress_callback(f"✅ llama-quantize ready ({len(extracted)} files)")

        return True, str(quantize_exe)

    except Exception as e:
        return False, f"Failed to download llama-quantize: {e}"


def get_gguf_status(
    model_name: str,
    checkpoints_dir: Optional[Path] = None,
) -> Dict:
    """Get GGUF conversion status for a model.

    Returns dict with:
      safetensors_present: bool
      available_gguf: list of quant types that have GGUF files
      model_dir: str
    """
    if checkpoints_dir is None:
        checkpoints_dir = _get_checkpoints_dir()

    model_dir = checkpoints_dir / model_name
    result = {
        "safetensors_present": False,
        "available_gguf": [],
        "model_dir": str(model_dir),
    }

    if not model_dir.exists():
        return result

    # Check for safetensors
    safetensors = glob.glob(str(model_dir / "*.safetensors"))
    result["safetensors_present"] = len(safetensors) > 0

    # Check each quant type
    for quant in ALL_QUANTS:
        gguf_name = f"{model_name}-{quant}.gguf"
        if (model_dir / gguf_name).exists():
            result["available_gguf"].append(quant)

    return result


def _gguf_filename(model_name: str, quant: str) -> str:
    """Generate standardized GGUF filename."""
    return f"{model_name}-{quant}.gguf"


def convert_model(
    model_name: str,
    target_quant: str,
    checkpoints_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """Convert a safetensors model to GGUF with the specified quantization.

    Full pipeline:
    - For Q8_0, BF16: direct conversion via convert_hf_to_gguf.py
    - For Q4_K_M, Q5_K_M, Q6_K: convert to BF16 first, then quantize

    Args:
        model_name: e.g. "acestep-5Hz-lm-4B"
        target_quant: e.g. "Q5_K_M", "Q8_0", "BF16"
        checkpoints_dir: override checkpoints directory
        progress_callback: called with progress message strings

    Returns:
        (success, message)
    """
    if checkpoints_dir is None:
        checkpoints_dir = _get_checkpoints_dir()

    model_dir = checkpoints_dir / model_name
    target_file = model_dir / _gguf_filename(model_name, target_quant)

    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    # Check if already exists
    if target_file.exists() and target_file.stat().st_size > 0:
        log(f"✅ {target_quant} GGUF already exists: {target_file.name}")
        return True, f"Already exists: {target_file}"

    # Check safetensors exist
    safetensors = glob.glob(str(model_dir / "*.safetensors"))
    if not safetensors:
        msg = f"❌ No safetensors found in {model_dir}. Download the model first."
        log(msg)
        return False, msg

    converter_script = _get_converter_script()
    if not converter_script.exists():
        return False, f"❌ Converter script not found: {converter_script}"

    python_exe = sys.executable
    quant_upper = target_quant.upper()

    # Direct conversion path (Q8_0, BF16, F16)
    if quant_upper in ("Q8_0", "BF16", "F16"):
        outtype = quant_upper.lower()  # convert_hf_to_gguf uses lowercase
        log(f"Converting {model_name} → {quant_upper} (direct)...")

        cmd = [
            python_exe,
            str(converter_script),
            str(model_dir),
            "--outtype", outtype,
            "--outfile", str(target_file),
        ]

        success = _run_subprocess(cmd, log)
        if success and target_file.exists():
            size_gb = target_file.stat().st_size / (1024 ** 3)
            log(f"✅ Conversion complete: {target_file.name} ({size_gb:.1f} GB)")
            return True, str(target_file)
        else:
            return False, f"❌ Conversion failed for {model_name} → {quant_upper}"

    # Two-step path (Q4_K_M, Q5_K_M, Q6_K): need BF16 intermediate + llama-quantize
    elif quant_upper in ("Q4_K_M", "Q5_K_M", "Q6_K"):
        # Step 0: Ensure llama-quantize binary
        log("Ensuring llama-quantize binary is available...")
        bin_ok, bin_result = ensure_quantize_binary(progress_callback=log)
        if not bin_ok:
            return False, bin_result

        quantize_exe = bin_result

        # Step 1: Check if BF16 intermediate exists, create if not
        bf16_file = model_dir / _gguf_filename(model_name, "BF16")
        bf16_is_temp = False

        if bf16_file.exists() and bf16_file.stat().st_size > 0:
            log(f"Using existing BF16 intermediate: {bf16_file.name}")
        else:
            log(f"Step 1/2: Converting safetensors → BF16 intermediate...")
            bf16_is_temp = True

            cmd = [
                python_exe,
                str(converter_script),
                str(model_dir),
                "--outtype", "bf16",
                "--outfile", str(bf16_file),
            ]

            success = _run_subprocess(cmd, log)
            if not success or not bf16_file.exists():
                return False, f"❌ BF16 intermediate conversion failed"

            size_gb = bf16_file.stat().st_size / (1024 ** 3)
            log(f"BF16 intermediate ready ({size_gb:.1f} GB)")

        # Step 2: Quantize BF16 → target
        log(f"Step 2/2: Quantizing BF16 → {quant_upper}...")

        cmd = [
            quantize_exe,
            str(bf16_file),
            str(target_file),
            quant_upper,
        ]

        success = _run_subprocess(cmd, log)

        # Cleanup BF16 intermediate if it was temporary
        if bf16_is_temp and bf16_file.exists():
            log(f"Cleaning up BF16 intermediate...")
            try:
                bf16_file.unlink()
            except Exception as e:
                log(f"Warning: could not delete intermediate: {e}")

        if success and target_file.exists():
            size_gb = target_file.stat().st_size / (1024 ** 3)
            log(f"✅ Conversion complete: {target_file.name} ({size_gb:.1f} GB)")
            return True, str(target_file)
        else:
            return False, f"❌ Quantization failed for {model_name} → {quant_upper}"

    else:
        return False, f"❌ Unsupported quantization type: {target_quant}"


def _run_subprocess(
    cmd: List[str],
    log: Callable[[str], None],
) -> bool:
    """Run a subprocess, streaming each output line to the log callback.

    Returns True if exit code is 0.
    """
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            # Set cwd for DLL discovery (llama-quantize needs its DLLs)
            cwd=str(_get_bin_dir()) if "quantize" in str(cmd[0]).lower() else None,
        )

        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log(line)

        proc.wait()
        return proc.returncode == 0

    except Exception as e:
        log(f"❌ Subprocess error: {e}")
        return False


def convert_all_models(
    target_quant: str,
    checkpoints_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[Tuple[str, bool, str]]:
    """Convert all available LM models to the target quant.

    Returns list of (model_name, success, message).
    """
    if checkpoints_dir is None:
        checkpoints_dir = _get_checkpoints_dir()

    # Find all LM model directories
    lm_models = []
    for d in checkpoints_dir.iterdir():
        if d.is_dir() and d.name.startswith("acestep-5Hz-lm-"):
            # Check if it has safetensors
            if glob.glob(str(d / "*.safetensors")):
                lm_models.append(d.name)

    if not lm_models:
        if progress_callback:
            progress_callback("No LM models with safetensors found.")
        return []

    results = []
    for i, model in enumerate(sorted(lm_models), 1):
        if progress_callback:
            progress_callback(f"\n{'='*50}")
            progress_callback(f"Model {i}/{len(lm_models)}: {model}")
            progress_callback(f"{'='*50}")

        success, msg = convert_model(
            model, target_quant, checkpoints_dir, progress_callback
        )
        results.append((model, success, msg))

    return results


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert ACE-Step LM models to GGUF")
    parser.add_argument("model", nargs="?", help="Model name (e.g. acestep-5Hz-lm-4B)")
    parser.add_argument("--quant", default="Q8_0", choices=ALL_QUANTS,
                        help="Quantization type (default: Q8_0)")
    parser.add_argument("--all", action="store_true", help="Convert all available LM models")
    parser.add_argument("--status", action="store_true", help="Show GGUF status for all models")
    parser.add_argument("--checkpoints", type=Path, help="Override checkpoints directory")

    args = parser.parse_args()

    def print_progress(msg):
        print(msg, flush=True)

    if args.status:
        checkpoints = args.checkpoints or _get_checkpoints_dir()
        for d in sorted(checkpoints.iterdir()):
            if d.is_dir() and d.name.startswith("acestep-5Hz-lm-"):
                status = get_gguf_status(d.name, checkpoints)
                gguf_str = ", ".join(status["available_gguf"]) or "none"
                st_str = "✅" if status["safetensors_present"] else "❌"
                print(f"  {d.name}: safetensors={st_str}  gguf=[{gguf_str}]")
        sys.exit(0)

    if args.all:
        results = convert_all_models(args.quant, args.checkpoints, print_progress)
        successes = sum(1 for _, s, _ in results if s)
        print(f"\nDone: {successes}/{len(results)} models converted.")
        sys.exit(0 if successes == len(results) else 1)

    if not args.model:
        parser.error("Model name required (or use --all)")

    success, msg = convert_model(args.model, args.quant, args.checkpoints, print_progress)
    print(msg)
    sys.exit(0 if success else 1)
