from dataclasses import dataclass, field
from typing import List

@dataclass
class LoKRConfig:
    """Fallback LoKr configuration used when metadata reading fails or for LyCORIS fallback path."""
    target_modules: List[str] = field(default_factory=lambda: ["attn.qkv", "attn.proj", "ff.net.0", "ff.net.2"])
    linear_dim: int = 10000
    linear_alpha: int = 1
    factor: int = 16
    decompose_both: bool = False
    use_tucker: bool = False
    use_scalar: bool = False
    full_matrix: bool = False
    bypass_mode: bool = False
    rs_lora: bool = False
    unbalanced_factorization: bool = False
    weight_decompose: bool = False
