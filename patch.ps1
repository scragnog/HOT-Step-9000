import sys

with open('acestep/llm_inference.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update backend checks
content = content.replace(
    'if backend == "vllm" and device != "cuda":',
    'if backend in ("vllm", "custom-vllm") and device != "cuda":'
)
content = content.replace(
    'f"[initialize] vllm backend requires CUDA,',
    f'"[initialize] {backend} backend requires CUDA,'
) # Wait, this won't work perfectly inside an f-string if I do simple replace. Let's do regex or exact string.

with open('patch.py', 'w', encoding='utf-8') as f:
    f.write('''import sys, re

with open("acestep/llm_inference.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace(
    \\'if backend == "vllm" and device != "cuda":\\',
    \\'if backend in ("vllm", "custom-vllm") and device != "cuda":\\'
)
text = text.replace(
    \\'f"[initialize] vllm backend requires CUDA, using PyTorch backend for device={device}."\\',
    \\'f"[initialize] {backend} backend requires CUDA, using PyTorch backend for device={device}."\\'
)

text = text.replace(
    \\'if backend == "vllm":\\',
    \\'if backend in ("vllm", "custom-vllm"):\\',
    1 # Only the first occurrence after the backend variable check
)

match = re.search(r\\'else:\\n\\s+status_msg = self\\._initialize_5hz_lm_vllm\\(.*?\\n\\s+\\)\\n\\s+logger\\.info\\(f"5Hz LM status message: \\{status_msg\\}"\\)\\n\\s+if status_msg\\.startswith\\("❌"\\):\\', text, re.DOTALL)
if match:
    old_block = match.group(0)
    new_block = old_block.replace(
        "status_msg = self._initialize_5hz_lm_vllm(",
        \"\"\"if backend == "custom-vllm":
                        status_msg = self._initialize_5hz_lm_custom_vllm(
                            full_lm_model_path,
                            enforce_eager=enforce_eager_for_vllm,
                        )
                    else:
                        status_msg = self._initialize_5hz_lm_vllm(\"\"\"
    )
    text = text.replace(old_block, new_block)

text = text.replace(
    "from nanovllm import SamplingParams",
    \"\"\"if self.llm_backend == "custom-vllm":
            from acestep.customized_vllm import SamplingParams
        else:
            from nanovllm import SamplingParams\"\"\"
)

# Extract _initialize_5hz_lm_vllm
match_func = re.search(r\\'\\s+def _initialize_5hz_lm_vllm\\(.*?return f"❌ Error initializing 5Hz LM.*?\\n(.*?)(\\s+def _run_vllm)\\' , text, re.DOTALL)
if match_func:
    func_text = re.search(r\\'\\s+def _initialize_5hz_lm_vllm.*?(?:\\n\\s+def _run_vllm)\\', text, re.DOTALL).group(0)
    func_text = func_text.replace("def _run_vllm", "")
    new_func = func_text.replace("_initialize_5hz_lm_vllm", "_initialize_5hz_lm_custom_vllm")
    new_func = new_func.replace("nanovllm", "acestep.customized_vllm")
    new_func = new_func.replace("self.llm_backend = \\"vllm\\"", "self.llm_backend = \\"custom-vllm\\"")
    
    # insert before _run_vllm
    text = text.replace("    def _run_vllm(", new_func + "\\n    def _run_vllm(")

with open("acestep/llm_inference.py", "w", encoding="utf-8") as f:
    f.write(text)
print("Patched!")
''')
