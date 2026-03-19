import sys, re

with open("acestep/llm_inference.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace device checked backends
text = text.replace(
    'if backend == "vllm" and device != "cuda":',
    'if backend in ("vllm", "custom-vllm") and device != "cuda":'
)
text = text.replace(
    'f"[initialize] vllm backend requires CUDA, using PyTorch backend for device={device}."',
    'f"[initialize] {backend} backend requires CUDA, using PyTorch backend for device={device}."'
)

# Initializer routing
text = text.replace(
    'if backend == "vllm":\n                _warn_if_prerelease_python',
    'if backend in ("vllm", "custom-vllm"):\n                _warn_if_prerelease_python'
)

# Replace the block that calls _initialize_5hz_lm_vllm
old_init_call = """else:
                    status_msg = self._initialize_5hz_lm_vllm(
                        full_lm_model_path,
                        enforce_eager=enforce_eager_for_vllm,
                    )
                    logger.info(f"5Hz LM status message: {status_msg}")"""
new_init_call = """else:
                    if backend == "custom-vllm":
                        status_msg = self._initialize_5hz_lm_custom_vllm(
                            full_lm_model_path,
                            enforce_eager=enforce_eager_for_vllm,
                        )
                    else:
                        status_msg = self._initialize_5hz_lm_vllm(
                            full_lm_model_path,
                            enforce_eager=enforce_eager_for_vllm,
                        )
                    logger.info(f"5Hz LM status message: {status_msg}")"""
text = text.replace(old_init_call, new_init_call)

# Update _run_vllm to import SamplingParams correctly
old_import = "from nanovllm import SamplingParams"
new_import = """if self.llm_backend == "custom-vllm":
            from acestep.customized_vllm import SamplingParams
        else:
            from nanovllm import SamplingParams"""
text = text.replace(old_import, new_import)

# Extract _initialize_5hz_lm_vllm to create _initialize_5hz_lm_custom_vllm
import re
func_match = re.search(r'    def _initialize_5hz_lm_vllm\(.*?\n    def _run_vllm\(', text, re.DOTALL)
if func_match:
    original_func = func_match.group(0)[:-len("\n    def _run_vllm(")]
    
    # Create custom vllm version
    new_func = original_func.replace("_initialize_5hz_lm_vllm", "_initialize_5hz_lm_custom_vllm")
    new_func = new_func.replace("nanovllm", "acestep.customized_vllm")
    new_func = new_func.replace('self.llm_backend = "vllm"', 'self.llm_backend = "custom-vllm"')
    
    # Insert new function
    text = text.replace("    def _run_vllm(", new_func + "\n\n    def _run_vllm(")

with open("acestep/llm_inference.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Patch applied.")
