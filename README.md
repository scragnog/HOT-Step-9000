<div align="center">

# HOT-Step 9000

<img src="./images/hotstep9000-transparentlogo.webp" width="400px" alt="HOT-Step 9000 Logo">

**Local AI Music Generator for Windows**

*Summon bangers directly from your GPU.*

</div>

## About

**HOT-Step 9000** is a fully functional, open-source local AI music generation suite designed for Windows.

This project is a standalone frontend for [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5). It originally started as a fork of [sdbds/ACE-Step-1.5-for-windows](https://github.com/sdbds/ACE-Step-1.5-for-windows) but has since grown into its own distinct application with significant UI Overhauls, advanced features, and quality-of-life improvements.

### ⚠Work In Progress
This application is under active, ongoing development. You may occasionally encounter bugs or unoptimized features. We welcome bug reports and feature requests via GitHub Issues. 

While currently optimized and supported specifically for **Windows environments**, PRs contributing cross-platform support are welcome.

---

## Key Features

HOT-Step 9000 sits on top of the original ACE-Step backend but introduces a massive array of new tools and features for advanced AI music creation:

- **Auto-Mastering:** Every generated track can be automatically mastered with two options — a built-in 6-stage Processing Chain (EQ, saturation, stereo widening, compression, limiter) or reference-based mastering via [Matchering](https://github.com/sergree/matchering) with optional per-stem processing for superior tonal fidelity. Interactive console for real-time tweaks, persistent presets, and one-click remastering of existing tracks.
- **Advanced Multi-Adapter System:** Slot-based loading of up to 4 simultaneous LoRA/LoKr adapters with per-layer scaling via weight-space merging. Includes saveable scale presets and an optional **merge mode** that bakes adapter weights directly into the base model for zero runtime VRAM overhead.
- **Activation Steering (TADA):** Experimental zero-shot generation guidance by modifying model activations directly.
- **Advanced Guidance & Solvers:** 7 guidance modes (including PAG, APG, ADG) and 6 ODE solvers (Euler, Heun, DPM++ 2M, RK4, JKASS, JKASS Fast) with per-solver parameter tuning. Advanced Guidance sub-panel for independent text/lyric scales, APG tuning, and guidance decay scheduling.
- **JKASS Inference Solvers:** Purpose-built music diffusion solvers with frequency damping, beat stability, and temporal smoothing — configurable per-generation via dedicated UI controls.
- **Custom VLLM Backend:** Third LM backend option with bespoke KV-cache pooling and optimized two-phase generation. Hot-switchable between PyTorch, standard VLLM, and Custom VLLM without restart.
- **Anti-Autotune:** Spectral smoothing to reduce robotic vocal artifacts in AI-generated music, with a simple 0–1 intensity slider.
- **Vocoder Enhancement (HiFi-GAN):** Optional high-quality audio decode pass using ADaMoSHiFiGAN for improved timbre and clarity. Downloads separately (~206 MB).
- **Timestep Scheduler:** 6 pluggable timestep distributions (Linear, DDIM, SGM, Bong Tangent, Linear Quadratic, Composite 2-Stage) for controlling where denoising effort is concentrated.
- **Cover Mode Tools:** Tempo scaling (0.5x–2.0x) and pitch shifting (±12 semitones) for covers and repaints, applied before VAE encoding.
- **Stem Extraction & Separation:** Both generative extraction (via DiT) and deterministic separation (BS-RoFormer/Demucs) with a built-in multi-track mixer.
- **Upscale to HQ:** Re-run low-step preview audio at higher quality settings using precomputed audio codes.
- **Melodic Variation:** Adjust the language model's repetition penalty for varied or hypnotic melodic phrasing.
- **Quality Scoring:** Automatic PMI and DiT alignment scores on every generated track.
- **Synced Lyrics & Live Visualizer:** Real-time Web Audio API visualizers (10 presets, configurable pool, fullscreen mode) with synced `.lrc` lyrics overlays and section markers.
- **A/B Track Comparison:** Side-by-side simultaneous playback of any two tracks with instant toggle and parameter diff view.
- **Layer Ablation Lab:** Developer tool for automated layer sweeps to isolate adapter functional impact across 24 transformer layers.
- **One-Click Launcher:** Interactive loading screen with model selection dropdowns, auto-continue timer, and real-time service health polling.
- **Model Hot-Switching:** Change LM and DiT models instantly without restarting the server, with dynamic model discovery from the checkpoints directory. DiT quantization can also be hot-swapped on-the-fly from the UI.
- **JSON Export/Import:** Save and share complete generation configurations including adapter and steering state.
- **Creation Panel Overhaul:** Reorganized from a single monolithic panel into ~13 modular accordion sections with Simple vs Custom modes.
- **Audio Enhancement Studio (Legacy):** Per-stem DSP engine with 6 presets, reverb, echo, stereo widening, and optional Demucs stem separation.
- **UI & QoL:** Persistent settings, debug panel with GPU/RAM/CPU monitoring, waveform visualizer, track list pagination, job cancellation, queue management, bulk operations, and a simple one-click shutdown.
- **AI Cover Art:** Optional AI-generated album artwork using SDXL Turbo. When enabled in Settings, each generated song receives a unique 512×512 cover image based on its title, style, and lyrics — generated locally on your GPU after audio completes.
- **Redmond Mode:** One-click DPO quality refinement using the [AceStep_Refine_Redmond](https://huggingface.co/artificialguybr/AceStep_Refine_Redmond) adapter by [artificialguybr](https://huggingface.co/artificialguybr). Merges a trained quality adapter directly into the DiT decoder below the adapter slot system — improving musicality, arrangement, and vocal quality across all generations. Auto-downloads on first use (~132 MB), toggleable at runtime with adjustable scale.
- **Lyric Studio:** A complete integrated songwriting environment. Feed it your own lyrics, build an AI-powered stylistic profile of your writing, and generate new original songs that match your creative voice — then produce them as audio with one click. Features include multi-pass profiling (rhyme, structure, meter, vocabulary, themes), a 6-layer AI-slop detector, curated cross-album profiles, bulk operations, album presets with adapter & matchering config, and a persistent audio generation queue. → *[Full documentation](./docs/lyric-studio-documentation.md)*

*For a detailed, technical breakdown of every new feature, see [FEATURES.md](./docs/FEATURES.md).*

---

## Supported Models

HOT-Step 9000 supports both the standard **1.5B DiT** models and the newer **4B XL DiT** models from [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5). Models are auto-downloaded on first use, or can be pre-fetched via `install.bat` or the model downloader CLI.

### Standard Models (1.5B DiT)

| Model | Description | Steps |
|-------|-------------|-------|
| `acestep-v15-turbo` | Default. Fast generation with shift scheduling. | 8 |
| `acestep-v15-turbo-shift3` | Turbo variant with shift=3 (recommended default). | 8 |
| `acestep-v15-sft` | SFT-tuned — higher quality, more steps required. | 40+ |
| `acestep-v15-base` | Base model. Supports extract/lego/complete tasks. | 40+ |

### XL Models (4B DiT)

The XL variants are **twice the parameter count** (4B vs 1.5B) of the standard models, producing richer, more detailed audio. They require **≥12 GB VRAM** (16+ GB recommended).

| Model | Description | Steps | Size |
|-------|-------------|-------|------|
| `acestep-v15-xl-turbo` | Fastest XL variant — distilled for low-step generation. | 8 | ~10 GB |
| `acestep-v15-xl-sft` | XL SFT — highest quality XL variant. | 40+ | ~10 GB |
| `acestep-v15-xl-base` | XL base model. | 40+ | ~10 GB |
| `acestep-v15-merge-sft-turbo-xl-ta-0.5` | Community SFT+Turbo merge at α=0.5 by [jeankassio](https://huggingface.co/jeankassio). Blends SFT quality with Turbo speed. | 15–30 | ~20 GB |
| `acestep-v15-merge-base-turbo-xl-ta-0.5` | Community Base+Turbo merge at α=0.5 by [jeankassio](https://huggingface.co/jeankassio). | 15–30 | ~20 GB |
| `acestep-v15-merge-base-sft-xl-ta-0.5` | Community Base+SFT merge at α=0.5 by [jeankassio](https://huggingface.co/jeankassio). | 40+ | ~20 GB |

> **Note:** XL models support LoRA/LoKr adapters, but only those **trained specifically on the XL architecture** — standard 1.5B adapters are not compatible due to different layer dimensions. Multi-batch generation (`batch_size > 1`) is also not recommended for XL models at this time.

**Download via CLI:**
```bash
python -m acestep.model_downloader --model acestep-v15-xl-turbo --skip-main
```

### Language Models

| Model | Size | Notes |
|-------|------|-------|
| `acestep-5Hz-lm-1.7B` | ~3.4 GB | Default. Included in main download. |
| `acestep-5Hz-lm-0.6B` | ~1.2 GB | Lighter, faster. |
| `acestep-5Hz-lm-4B` | ~8 GB | Highest quality. GGUF quantization recommended. |

---

## VRAM Management

HOT-Step 9000 includes built-in tools to reduce GPU memory usage, making it accessible on hardware with as little as **16GB VRAM** — without sacrificing the full 4B parameter language model or advanced features like LoRA adapters.

### DiT Quantization

The DiT (Diffusion Transformer) is the largest model component. HOT-Step supports on-demand weight quantization via [torchao](https://github.com/pytorch/ao) to dramatically reduce its VRAM footprint:

| Setting | VRAM Saved | Quality Impact | LoRA Compatible |
|---------|-----------|----------------|-----------------|
| `none` (BF16) | Baseline | None | ✅ Full support |
| `int8_weight_only` | ~2.5 GB | Negligible | ✅ Full support |
| `nf4` | ~5.5 GB | Minor | ✅ Works via dequantize→merge→requantize pipeline |
| `int4_weight_only` | ~6.5 GB | Minor; experimental | ✅ Works via dequantized merge |

**Configuration:** Set `ACESTEP_QUANTIZATION` in your `.env` file:

```env
# Options: auto, none, int8_weight_only, int4_weight_only, nf4
ACESTEP_QUANTIZATION=auto
```

When set to `auto`, HOT-Step detects your GPU VRAM and applies the appropriate quantization level automatically. Quantization is applied at model load time — no model re-download required.

**Hot-swapping from the UI:** Quantization can also be changed on-the-fly from the model dropdown panel without restarting the server. Changing quantization will reload the DiT model with the new setting.

> **LoRA on quantized models:** Adapters work on both INT8 and INT4 quantized DiTs. Internally, base weights are dequantized for the merge computation, so adapter-modified layers run in BF16. VRAM increases slightly when LoRA is active, but remains well below the unquantized baseline.

### Local GGUF Conversion for Language Models

The language models (LMs) that generate structured music tokens can be converted to [GGUF format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) for use with `llama-cpp-python`. GGUF models use significantly less VRAM than the default vLLM/PyTorch backends and support partial CPU offloading.

**How to convert:**

The loading screen includes a built-in conversion panel. Select your model and desired quantization level, and click **Convert to GGUF**. Conversion progress is streamed live to the UI.

Alternatively, use the CLI:

```bash
python -m acestep.tools.gguf_converter acestep-5Hz-lm-4B --quant Q4_K_M
```

| Quant Level | LM VRAM | Quality | Speed |
|-------------|---------|---------|-------|
| BF16 | ~8 GB | Lossless | Fast |
| Q8_0 | ~4 GB | Near-lossless | Fast |
| Q4_K_M | ~2.5 GB | Good | Moderate |
| Q5_K_M | ~3 GB | Very good | Moderate |
| Q6_K | ~3.5 GB | Excellent | Moderate |

**What happens during conversion:**
1. For BF16/Q8_0: Direct conversion via `convert_hf_to_gguf.py` from [llama.cpp](https://github.com/ggml-org/llama.cpp)
2. For Q4/Q5/Q6: Two-step process — convert to BF16 intermediate, then quantize with `llama-quantize`
3. The `llama-quantize` binary and required DLLs are auto-downloaded from the llama.cpp GitHub releases on first use
4. Converted files are saved alongside the original model in the `checkpoints/` directory

**GPU offloading:** Configure how many transformer layers run on GPU via `ACESTEP_N_GPU_LAYERS` in `.env`:

```env
# -1 = all on GPU (fastest), 0 = all on CPU (no VRAM), N = partial offload
ACESTEP_N_GPU_LAYERS=-1
```

### Real-World VRAM Reference (RTX 3090 / 24GB)

These are measured values using the full **acestep-5Hz-lm-4B** model:

| Configuration | Peak VRAM | Notes |
|---------------|-----------|-------|
| No quantization (BF16 DiT + vLLM) | ~22.2 GB | Full quality, full speed |
| INT4 DiT only | ~19.8 GB | Minimal quality impact |
| INT4 DiT + Q4_K_M LM | ~15.6 GB | Fits 16GB GPUs with headroom |
| INT8 DiT + Q8_0 LM | ~17 GB | Best quality/VRAM balance |

> **Note:** GGUF-based LM inference disables CFG (classifier-free guidance) automatically to maintain reasonable generation speed. This has minimal impact on output quality, especially when using Thinking mode.

---

## Installation & Usage

*(Assuming basic Python/CUDA knowledge and a suitable Windows GPU environment)*

1. Clone the repository.
2. Run `install.bat` to install dependencies.
3. Download applicable models into the `checkpoints/` directory.
4. Run `LAUNCH.bat` to start the application with the interactive loading screen.
5. In the UI, set your generation parameters, enter a prompt, and bring forth the **bangers**.

---

## Lineage & Credits

HOT-Step 9000 exists thanks to the incredible open-source AI audio community:

- Core models and initial application framework by the **[ACE-Step 1.5 Team](https://github.com/ace-step/ACE-Step-1.5)**.
- Windows compatibility layer and upstream scaffolding by **[sdbds](https://github.com/sdbds/ACE-Step-1.5-for-windows)**.
- DPO quality refinement adapter (Redmond Mode) by **[artificialguybr](https://huggingface.co/artificialguybr)**.
- XL SFT+Turbo community merge model by **[jeankassio](https://huggingface.co/jeankassio)**.
- UI Overhaul, advanced tooling, and new features by **scragnog**.

## License

This project inherits the licensing of its upstream parents. See original repositories for detailed model and code licensing. Please use AI generation tools responsibly.
