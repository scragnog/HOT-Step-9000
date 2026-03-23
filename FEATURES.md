# HOT-Step 9000 — Features

This document details all features added on top of the upstream ACE-Step 1.5 codebase.

---

## Upstream Sync (v1.5.0-sync)

Synced 37 upstream commits from [sdbds/ACE-Step-1.5-for-windows](https://github.com/sdbds/ACE-Step-1.5-for-windows). Safely integrated core improvements while preserving all custom features (Guidance Mode, Auto-Mastering, Advanced Adapters).

### What's included

| File | Description |
|------|-------------|
| `acestep/api/http/release_task_request_builder.py` | Ported upstream fix for multi-seed `batch_size` parsing |
| `docs/*`, `.github/*` | Brought in VitePress documentation structure and issue templates |
| `Dockerfile.jetson`, `docker-compose.yaml` | Jetson Nano and Docker containerization support |
| `acestep/ui/gradio/i18n/*` | Chinese translation fixes for Gradio interfaces |
| `.claude/skills/*` | System prompt and skill script updates |
| `acestep/core/generation/handler/memory_utils.py` | Core ROCm `float32` default dtype fixes |

### How it works

1. 14 safe commits were cherry-picked directly from upstream.
2. The `release_task_request_builder.py` fix was manually ported to prevent overwriting the custom `guidance_mode` pass-through logic.
3. Upstream AceFlow UI commits that were added and reverted were skipped.
4. The remaining history was aligned with a Git merge.

---

## Melodic Variation

A slider that adds controlled melodic randomness to generation by adjusting the language model's repetition penalty. Higher values encourage the LM to explore less-repeated token sequences, introducing more melodic variety; lower values reinforce repetitive structure.

### What's included

| File | Description |
|------|-------------|
| `acestep/inference.py` | `lm_repetition_penalty` added to `GenerationParams` dataclass |
| `acestep/api_server.py` | `lm_repetition_penalty` added to request model, alias mapping, and passed to GenerationParams |
| `acestep/llm_inference.py` | `repetition_penalty` forwarded as float to both `vllm` and native `generate()` LLM handlers |
| `ace-step-ui` | Melodic Variation slider (0.0–2.0) in the Generation Settings accordion |

### How it works

1. The slider maps to `lm_repetition_penalty` (0.0–2.0, default 1.0 = no effect)
2. Values above 1.0 penalise tokens that have already appeared, nudging the LM away from repetitive melodic phrases
3. Values below 1.0 make repetition more likely — useful for looping, hypnotic, or structurally strict styles
4. The parameter is forwarded directly to the LM `generate()` call's `repetition_penalty` argument (both vLLM and native paths)

---

## Quality Scoring (PMI + DiT Alignment)

Automatic quality scoring computed at the end of each generation and displayed in the UI. Two complementary metrics assess different aspects of generation quality.

### What's included

| File | Description |
|------|-------------|
| `acestep/inference.py` | `get_scores` / `score_scale` wired through `GenerationParams`; scores returned in result dict |
| `acestep/api_server.py` | `get_scores` and `score_scale` added to request model and `_build_request()` mapping; scores included in response |
| `ace-step-ui/server/src/routes/generate.ts` | Forwards `getScores` / `scoreScale` to Python backend |
| `ace-step-ui/components/accordions/ScoreSystemAccordion.tsx` | Toggle and scale slider for the scoring system |
| `ace-step-ui/components/SongCard.tsx` | Score badges (PMI % and star ratings) on each song card |
| `ace-step-ui/components/RightSidebar.tsx` | Score badges in the Song Details panel |

### How it works

1. **PMI Score** — Pointwise Mutual Information between the lyric tokens and the generated audio codes. Measures how strongly the audio content correlates with the lyric semantics. Displayed as a percentage (0–100%).
2. **DiT Score** — Alignment between the DiT (diffusion transformer) output and the LM-guided audio codes. Measures how faithfully the audio synthesis followed the language model's musical intent. Displayed as a 1–5 star rating.
3. Scores are computed in the Python backend after generation and returned in the API response alongside the audio path.
4. Enable/disable via the **Score System** accordion in the Create panel. `score_scale` controls weighting of score-guided beam-search (if applicable).
5. Both scores appear as compact badges on each `SongCard` in the track list and in the Song Details sidebar.

---

## Job Cancellation & Queue Management

Cancel any running or queued generation job without restarting the server. Sends a cooperative stop signal to the Python backend that halts inference at the next safe checkpoint.

### What's included

| File | Description |
|------|-------------|
| `acestep/api_server.py` | `POST /v1/cancel/{job_id}` endpoint — sets a per-job cancellation flag read by the inference loop |
| `ace-step-ui/server/src/routes/generate.ts` | Express proxy for the cancel endpoint |
| `ace-step-ui/services/api.ts` | `cancelJob(jobId)` API client method |
| `ace-step-ui/components/SongList.tsx` | Cancel button visible on in-progress song cards |
| `ace-step-ui/components/SongCard.tsx` | Cancel button in compact view |

### How it works

1. Each generation job is assigned a unique `job_id` at submission time
2. During inference, the generation loop checks a shared cancellation flag keyed by `job_id` at each denoising step
3. When **Cancel** is clicked, `POST /v1/cancel/{job_id}` sets the flag — the running job exits cleanly at its next step boundary (no mid-tensor corruption)
4. Queued jobs (not yet started) are removed from the queue immediately on cancel
5. The cancelled song entry is removed from the track list

---

## Upscale to HQ

Re-run inference on a previously generated preview track at higher quality settings. Preserves the audio codes from the first pass to guide the upscale, producing a higher-fidelity version of the same generation without starting from scratch.

### What's included

| File | Description |
|------|-------------|
| `acestep/inference.py` | `lm_hints_path` field in `GenerationParams`; `precomputed_lm_hints_25Hz` tensor surfaced from `service_generate_outputs` and saved as `.pt` file alongside audio |
| `acestep/api_server.py` | `audio_code_string` field added to `GenerateMusicRequest`; generated `audio_codes` included in API response; `GET /v1/audio/{path}` serves `.pt` hint files |
| `acestep/core/generation/handler/service_generate_outputs.py` | Surfaces `precomputed_lm_hints_25Hz` from generation outputs |
| `ace-step-ui` | **Upscale to HQ** option in the `SongDropdownMenu`; HQ steps config in `SettingsModal`; `App.tsx` handler re-submits with stored `audioCodes` |

### How it works

1. **First pass (Preview):** Generate at low steps (e.g. 20–50). The Python backend saves the raw LM audio code tensor (`.pt` file) alongside the audio file. The API response includes the serialized `audio_codes` string.
2. **Upscale:** Click **Upscale to HQ** from the song's dropdown menu. The frontend re-submits the original generation parameters with:
   - The stored `audio_codes` from the first pass (bypasses LM re-generation)
   - `lm_hints_path` pointing to the saved `.pt` tensor file
   - The HQ step count from Settings (default: 150)
3. The diffusion model runs the higher-step denoising pass guided by the first-pass audio codes, producing a cleaner, higher-fidelity render of the same musical content
4. Configure the HQ step count in **Settings → Upscale**
5. Toast notification auto-dismisses after 5s when the upscale job is queued

---

## Tempo Scale & Pitch Shift (Cover Mode)

Pre-process source audio before VAE encoding with pitch-preserving tempo changes and speed-preserving pitch shifts. Enables changing the tempo of a cover independently from its key, or transposing a male vocal track into a female range (or vice versa) before generation.

### What's included

| File | Description |
|------|-------------|
| `acestep/core/generation/handler/generate_music_request.py` | Time-stretch via `torchaudio.functional.speed()` and pitch shift via `torchaudio.functional.pitch_shift()`, applied after `process_src_audio()` and before padding/VAE encoding |
| `acestep/core/generation/handler/generate_music.py` | `tempo_scale` and `pitch_shift` params threaded through `generate_music()` |
| `acestep/inference.py` | Added to `GenerationParams` dataclass |
| `acestep/api_server.py` | Alias mapping, request model, REST and Gradio handlers |
| `ace-step-ui/server/src/routes/generate.ts` | `GenerateBody` type + request body (gated to cover/repaint/a2a tasks) |
| `ace-step-ui/components/CreatePanel.tsx` | State variables, prop passing, generate request inclusion |
| `ace-step-ui/components/sections/CoverRepaintSettings.tsx` | Side-by-side Tempo Scale (0.5x–2.0x) and Pitch Shift (-12 to +12 semitones) sliders |
| `ace-step-ui/i18n/translations.ts` | Labels, help text, and tooltips for both controls |

### How it works

1. Both sliders appear in the **Cover Settings** section (hidden for text2music/extract modes)
2. **Tempo Scale** (0.5x–2.0x): Uses phase vocoder to change speed without affecting pitch. 1.3x = 30% faster output, 0.8x = 20% slower
3. **Pitch Shift** (-12 to +12 semitones): Transposes the source audio without changing speed. +4 shifts up ~a major third (e.g. male→female vocal range), -3 shifts down a minor third
4. Both transforms are applied to the source audio tensor *before* it enters the padding and VAE encoding pipeline, so the model generates in the new tempo/key space
5. The two can be combined — e.g. speed up 1.2x AND shift up 3 semitones simultaneously
6. Display format: Tempo shows as `1.30x`, Pitch shows as `+3 ♯` / `-2 ♭`

---

## Activation Steering (TADA)

Total integration of Task Adaptive Directional Activation (TADA), enabling zero-shot generation guidance by modifying model activations directly.  
> **Note:** This feature is currently in-progress, experimental, and may not yet work as intended.

### What's included

| File | Description |
|------|-------------|
| `acestep/compute_steering.py` | Core mathematical script to isolate mathematical delta vectors using contrastive decoding. |
| `acestep/steering_controller.py` | Handles PyTorch hook injection and logic for applying multiple chained concepts during auto-regressive generation. |
| `acestep/core/generation/handler/steering_mixin.py` | **[NEW]** Provides high-level handler methods for vector I/O, enabling, and UI configuration mapping. |
| `ace-step-ui/components/sections/ActivationSteeringSection.tsx` | **[NEW]** UI component for computing contrastive vectors, dynamically overriding base prompts, and applying multi-concept guidance. |
| `docs/en/Activation_Steering_Tutorial.md` | **[NEW]** Comprehensive guide on use-cases and terminology. |

### How it works

1. It isolates the explicit mathematical "essence" of a concept by decoding a neutral base prompt and then computing the targeted difference against an activated prompt.
2. The user submits lines of concept modifiers into the **Compute Queue** to generate and cache these arrays on disk as `.pkl` files.
3. Computed vectors can be hot-loaded directly into the model's memory map across targeted layers (`tf6`, `tf7`).
4. Scale (alpha) sliders fine-tune the absolute intensity of each loaded concept dynamically, including supporting negative steering constraints. 
5. Users can selectively unload or permanently **Delete** poor concepts through the UI via the Express backend.

---

## Advanced Guidance & Solver Modes

Total overhaul of the inference backend to support 7 distinct guidance modes and 4 ODE solver algorithms, complete with UI integrations and educational tooltips.

### What's included

| File | Description |
|------|-------------|
| `acestep/core/generation/guidance.py` | **[NEW]** Central registry for Guidance modes (Plain CFG, CFG++, Dynamic CFG, Rescaled CFG, APG, ADG, PAG) |
| `acestep/core/generation/solvers.py` | **[NEW]** Central registry for ODE Step Solvers (Euler, Heun, DPM++ 2M, RK4) |
| `patch_checkpoints.py` | Patches base models on-the-fly to hook into the new guidance/solver registries without touching upstream code |
| `ace-step-ui/components/CreatePanel.tsx` | Added Guidance dropdown, Inference Method dropdown, and conditional PAG detail sliders |
| `ace-step-ui/i18n/translations.ts` | 40+ localized educational tooltips explaining every generation parameter |

### How it works

1. **Guidance Modes:** Choose between strict mathematical CFG variants (Plain, CFG++, Dynamic, Rescaled) or specialized audio-flow projections (APG, ADG) to control how strongly the text guides the music. PAG (Perturbed Attention Guidance) adds structural clarity independently.
2. **Solvers:** Trade off speed vs. quality. Euler (1 eval/step) is fast. Heun (2 evals) and RK4 (4 evals) offer higher-quality numerical integration at the cost of generation speed. DPM++ 2M offers 2nd-order quality at 1 eval per step.
3. Hovering over any parameter reveals a localized tooltip explaining what it does.

---

## One-Click Launcher with Model Selection

Single-click launch experience with an interactive loading screen that lets you choose which models to load before the servers start.

### What's included

| File | Description |
|------|-------------|
| `LAUNCH.bat` | One-click launcher — scans `checkpoints/` for available models, writes `loading-config.js`, opens loading screen, starts Express first then Python API |
| `START.bat` | Alternative launcher without loading screen |
| `loading.html` | Animated loading page with model selection dropdowns, 5s auto-continue timer, and real-time service status checklist |
| `ace-step-ui/server/src/routes/models.ts` | `POST /api/models/update-env` — updates `.env` file on disk when the user changes model selection |
| `ace-step-ui/server/src/index.ts` | CORS fix to allow `file://` origin (loading screen runs from local file) |

### How it works

1. `LAUNCH.bat` scans `checkpoints/` for all `acestep-v15-*` (DiT) and `acestep-5Hz-lm-*` (LM) models
2. Writes the model list and current `.env` selections to `loading-config.js`
3. Opens `loading.html` in the browser — dropdowns are pre-populated and pre-selected
4. **Model selection:** If the user changes a dropdown, the loading screen calls `POST /api/models/update-env` (Express) to update `.env` on disk before the Python API reads it
5. **5-second auto-continue timer** — resets if the user interacts with the dropdowns
6. Express starts **before** the Python API (3s head start), allowing `.env` updates to take effect
7. The loading page polls three services:
   - **Python API** — `/v1/models/status` (waits for `active_model` to be non-null)
   - **Express backend** — `localhost:3001/health`
   - **Vite frontend** — `localhost:3000`
8. Auto-redirects to the app once all three are confirmed ready

### API changes

- Added `GET /v1/models/status` endpoint to `acestep/api_server.py` (no auth required, includes `lm_model` field)
- Added `POST /api/models/update-env` endpoint to Express for loading screen `.env` updates
- `ace-step-ui/start.bat` checks `ACESTEP_NO_BROWSER` env var to avoid opening a duplicate browser tab

---

## LM Model Hot-Switching

Fixes a bug where changing the Language Model (5Hz LM) in the UI had no effect — the original model loaded at startup was always used regardless of the user's selection. Now supports live switching between LM models during a session.

### What's included

| File | Description |
|------|-------------|
| `acestep/api_server.py` | `_ensure_llm_ready()` rewritten to detect model mismatch and hot-switch (unload old → load new). Added `app.state._llm_model_path` tracking. `/v1/models/status` now returns `lm_model`. |
| `ace-step-ui/server/src/routes/generate.ts` | `lm_model_path` is now always sent to the Python API (previously only sent when `thinking=true`) |
| `ace-step-ui/components/CreatePanel.tsx` | LM model dropdown syncs to the actually loaded model on initial page load via `/api/models/status` |

### How it works

1. Every generation request now includes the selected `lm_model_path` (regardless of thinking mode)
2. `_ensure_llm_ready()` compares the requested model against the currently loaded one (`app.state._llm_model_path`)
3. If they differ, the current LLM is unloaded and re-initialized with the new model (~5–15s blocking operation)
4. If they match, it's a no-op (instant return)
5. On initial UI load, `CreatePanel` fetches `/api/models/status` and syncs the LM dropdown to whichever model is actually loaded

---

## Enhanced Model Selector

Dynamic model discovery and hot-swap switching. The model dropdown auto-populates from all installed checkpoints and supports switching the active DiT model without restarting the server.

### What's included

| File | Description |
|------|-------------|
| `acestep/handler.py` | `switch_dit_model()` — hot-swaps DiT weights, handles LoRA unload, VRAM cleanup, attention fallback |
| `acestep/api_server.py` | `GET /v1/models/list` — scans `checkpoints/` for installed models; `POST /v1/models/switch` — triggers model swap |
| `ace-step-ui/server/src/routes/models.ts` | Express proxy routes for model list, status, and switch |
| `ace-step-ui/server/src/routes/generate.ts` | Updated proxy target to `/v1/models/list` |
| `ace-step-ui/services/api.ts` | `getModels()` / `switchModel()` frontend API methods |
| `ace-step-ui/components/CreatePanel.tsx` | Mismatch banner, switch button, dynamic dropdown |
| `LAUNCH.bat` | Added server TypeScript rebuild step before startup |

### How it works

1. On startup, the Python API scans `checkpoints/` for all `acestep-v15-*` directories
2. The frontend fetches the installed model list via `/api/generate/models`
3. If the selected model differs from the loaded model, an amber mismatch banner appears
4. Clicking "Switch" calls `POST /v1/models/switch` which hot-swaps the DiT model (unloads LoRA, frees VRAM, loads new weights)
5. A fallback list of the 6 default ACE-Step models is shown while the Python API starts

---

## Simple Shutdown

Quit button in the sidebar that gracefully shuts down all ACE-Step processes (Python API, Vite frontend, Express backend, and their hosting terminal windows).

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/server/src/index.ts` | `POST /api/shutdown` — snapshots the process table via `Get-CimInstance`, walks the process tree to find ancestor shells, kills everything |
| `ace-step-ui/components/Sidebar.tsx` | Power icon quit button (red styling, appears at sidebar bottom) |
| `ace-step-ui/App.tsx` | ConfirmDialog wiring + "ACE-Step has shut down" overlay |

### How it works

1. Click the red **Quit** button at the bottom of the sidebar
2. A confirmation dialog appears — "Are you sure you wish to shut down ACE-Step?"
3. On confirm, `POST /api/shutdown` is called:
   - Snapshots the entire Windows process table in one `Get-CimInstance Win32_Process` call (~200ms)
   - Finds PIDs on ports 8001 (Python API) and 3000 (Vite) via `netstat`
   - Walks UP the process tree to find ancestor CMD/PowerShell/conhost windows
   - Kills all collected PIDs with `taskkill /F /T`
   - Express process exits last
4. Browser shows a "You may now close this tab" overlay

---

## Persistent Settings

Toggleable localStorage persistence for all generation settings. Disabled by default — once enabled in Settings, every parameter survives page refresh.

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/hooks/usePersistedState.ts` | `usePersistedState` hook — drop-in `useState` replacement with auto-persistence gated by `ace-persist-enabled` flag |
| `ace-step-ui/components/SettingsModal.tsx` | "Persistent Settings" toggle section with enable/disable switch |
| `ace-step-ui/components/CreatePanel.tsx` | ~35 useState calls converted to usePersistedState |

### How it works

1. Open **Settings** → toggle **"Remember my settings"** ON
2. All generation parameters (style, lyrics, BPM, model, LoRA path, inference settings, etc.) are now auto-saved to localStorage
3. Page refresh → all settings restored
4. Toggle OFF → all saved settings cleared, page reloads with defaults
5. Future features: use `usePersistedState('key', default)` instead of `useState(default)` — one-line change

---

## Track List Updates

A collection of track list UX improvements, bug fixes, and a new bulk-delete feature.

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/components/WaveformVisualizer.tsx` | Replaced upstream fixed-width bar rendering with dynamic spacing so the waveform fills the full progress bar width. Includes shared `AudioContext`, LRU cache (30 entries), and `AbortController` for proper cleanup. |
| `ace-step-ui/server/src/routes/generate.ts` | Fixed generation progress display — parses tqdm-style `progress_text` from the Python API. Added per-job queue detection so queued jobs don't leak the running job's progress. |
| `ace-step-ui/components/CreatePanel.tsx` | "Queue Next" button now uses i18n key instead of hardcoded English text. |
| `ace-step-ui/i18n/translations.ts` | Added `queueNext`, `deleteAllTracks`, `deleteAllTracksConfirm`, `allTracksDeleted`, `deleteAllFailed` keys for all 4 languages (en, zh, ja, ko). |
| `ace-step-ui/components/SongList.tsx` | Removed `createdAt` DESC sort from `listItems` so songs maintain their order from state. Added `onDeleteAll` prop with a trash icon button in the header bar. |
| `ace-step-ui/App.tsx` | Removed the `refreshSongsList` sort that caused completed songs to jump to the top. Completion now does in-place merge instead of full list reload. Added `handleDeleteAll` with confirmation dialog. |
| `ace-step-ui/server/src/routes/songs.ts` | Added `DELETE /api/songs/all` endpoint — deletes all user songs and associated audio/cover files from storage. |
| `ace-step-ui/services/api.ts` | Added `deleteAllSongs()` API client method. |
| `LAUNCH.bat`, `START.bat`, `ace-step-ui/start.bat`, `ace-step-ui/start-all.bat` | Added `/min` flag to all `start` commands so spawned terminal windows launch minimized. |

### Changes in detail

- **Waveform alignment** — Bars now fill the entire progress bar width using dynamic step calculation instead of fixed `barWidth=2, gap=1`.
- **Generation progress** — The Express backend now parses tqdm output (`14%|##5| 27/200 [00:06<00:42, 4.10steps/s]`) to extract percentage, ETA, and step count.
- **Queue progress bleed fix** — The Python API maps both `queued` and `running` to status code `0` and shares a global `log_buffer.last_message`. The Express status handler now checks the per-job `stage` field in the result data to detect queued jobs and returns `status: 'queued'` with `progress: 0` instead of leaking the running job's tqdm output.
- **Track reordering fix** — Two sorts were causing completed songs to jump to the top: one in `refreshSongsList` (App.tsx) and one in `listItems` (SongList.tsx). Both removed. Completion now does an in-place merge that preserves existing order.
- **Delete All Tracks** — Trash icon button in the track list header (next to Select/Filter). Shows a confirmation dialog with the count of tracks. Calls `DELETE /api/songs/all` which deletes all audio/cover files from storage and removes all DB rows.
- **Minimized windows** — All spawned terminal windows (Python API, Express, Vite) now launch minimized via `/min` flag.

---

## Advanced Multi-Adapter System

Slot-based multi-adapter loading (up to 4 simultaneous LoRA/LoKr adapters) with per-slot scaling, per-module-group scaling (Self-Attn, Cross-Attn, MLP), and per-layer scaling (layers 0–23). Uses weight-space merging approach. Existing basic single-adapter UI is preserved — the advanced system is behind an opt-in "Advanced" checkbox.

### What's included

| File | Description |
|------|-------------|
| `acestep/core/generation/handler/lora/advanced_adapter_mixin.py` | **[NEW]** `AdvancedAdapterMixin` — delta extraction, weight-space merging (`base + Σ(scale × group_scale × layer_scale × delta)`), slot management, per-layer scale API |
| `acestep/core/generation/handler/lora_manager.py` | Import + export `AdvancedAdapterMixin` |
| `acestep/handler.py` | Added `AdvancedAdapterMixin` to MRO, init state (`_adapter_slots`, `_next_slot_id`, `_merged_dirty`, `lora_group_scales`) |
| `acestep/api_server.py` | 5 new endpoints, updated `load`/`unload` for slot param, new request models. `set_slot_layer_scales` and `set_slot_layer_scale` for per-layer control |
| `ace-step-ui/server/src/routes/lora.ts` | 5 new routes: `GET /list-files`, `POST /group-scales`, `POST /slot-group-scales`, `POST /slot-layer-scales`, `POST /slot-layer-scale` |
| `ace-step-ui/services/api.ts` | `listLoraFiles()`, `setGroupScales()`, `setSlotGroupScales()`, `setSlotLayerScales()`, updated `loadLora`/`unloadLora` for slot support |
| `ace-step-ui/components/CreatePanel.tsx` | Advanced toggle, folder browser, slot cards. Debounced layer scale changes (500ms) with batch API calls sending full layer state |
| `ace-step-ui/components/accordions/AdaptersAccordion.tsx` | Role Blend sliders (Voice/Style/Coherence), tooltip prop on EditableSlider, per-layer slider grid (0–23) |

### How it works

1. Open the **Adapters** panel and check **Advanced (Multi-Adapter)**
2. Enter an adapter folder path → click **Scan** → available `.safetensors` files appear
3. Click **Load** on any adapter → it's loaded into a slot, delta extracted via weight-space merging
4. Each slot card shows: adapter name, type badge (LoRA/LoKr), overall scale slider (0–2)
5. Expand **Groups** on a slot → independent Self-Attn, Cross-Attn, MLP sliders with inline descriptions and hover tooltips
6. **Role Blend** sliders (always visible, below groups): 🎤 Voice (layers 0–7), 🎸 Style (layers 8–15), 🔗 Coherence (layers 16–23) — sets all layers in each group simultaneously
7. Expand **Layers** on a slot → 24 individual layer sliders (0–23) for surgical control
8. Load additional adapters (up to 4) — all merge simultaneously: `decoder = base + Σ(slot_scale × group_scale × layer_scale × delta)`
9. Per-adapter group scale settings are persisted in localStorage by adapter filename
10. Uncheck "Advanced" → original basic single-adapter UI appears unchanged

### Role Blend sliders

Empirical layer ablation experiments on this adapter revealed three functional layer groups:

| Slider | Layers | What it controls |
|--------|--------|------------------|
| 🎤 **Voice** | 0–7 | Vocal timbre and singer identity (~60% of character) |
| 🎸 **Style** | 8–15 | Musical energy, genre character, song-section phrasing |
| 🔗 **Coherence** | 16–23 | Harmonic integration glue — binds voice and style into a coherent output |

All three groups must have non-zero coherence to avoid harmonic discordance. Boosting Voice beyond ~1.5 causes clipping artifacts.

### Architecture note

Basic mode uses PEFT runtime hooks (existing). Advanced mode uses **weight-space merging**: backs up base decoder to CPU (~1.5GB), extracts each adapter as a delta, applies `base + Σ(scaled deltas)` at inference. Re-merge takes ~1s on scale change. Layer scales are applied per-weight-key using `_extract_layer_index`, with non-layer weights using the average of the set layer scales.

---

## Layer Ablation Lab

Developer tool for systematically exploring what each adapter layer contributes to the generated audio. Accessible via Developer Mode in the Create panel. Runs an automated sweep — generating one track per layer with that layer zeroed — to identify functional roles of each of the 24 transformer layers.

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/components/accordions/LayerAblationPanel.tsx` | **[NEW]** Full ablation lab UI: layer scale sliders (0–23) with zero-selected/reset-all bulk ops, audio diff tool (compute RMS energy difference between two tracks), ablation sweep runner with progress + cancel |
| `ace-step-ui/components/CreatePanel.tsx` | `handleStartSweep` / `handleCancelSweep`, sweep state management, `waitForGenerationDone` promise, `handleBulkLayerScalesChange` batch API call |
| `acestep/api_server.py` | `compute_audio_diff` endpoint with `_resolve_audio_path` helper (handles HTTP URLs, relative paths, and API paths) |

### How it works

1. Enable **Developer Mode** → open the **Layer Ablation Lab** panel
2. **Manual control:** 24 per-layer scale sliders. "Zero Selected" sets checked layers to 0; "Reset All" restores all to 1.0. Changes are debounced 500ms and sent as a single batch API call
3. **Audio Diff:** Compare any two tracks using RMS energy — paste or pin track paths into the A/B fields and click Compute Diff to get an energy delta score
4. **Ablation Sweep:** Click **Run Sweep** — the tool iterates through all 24 layers, zeroing each in turn, triggering a generation and waiting for it to complete before moving to the next. Results are titled `"Layer N zeroed - [prompt]"`. Cancel at any time
5. The sweep output can be loaded into a spreadsheet to chart RMS energy vs. layer, revealing which layers have the most/least adapter influence

### Interpreting results

From empirical testing on a Green Day LoKr adapter:
- **Layers 0–7 (Voice):** High-impact on singer timbre. Zeroing these makes the output sound significantly less like the adapter artist
- **Layers 8–15 (Style):** Affects musical energy and genre coherence at the song-structure level. Coherent on their own; contribute to discordance when missing alongside other active groups  
- **Layers 16–23 (Coherence):** The binding layer group. Their absence causes harmonic discordance even when voice and style layers are active. Preserve at ≥0.5 to avoid artifacts

## Creation Panel Reorganization

Total UX reorganization and architectural refactoring of the Create panel to reduce cognitive overload and group related settings.

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/components/CreatePanel.tsx` | Massively refactored into a layout shell delegating to ~13 modular sub-components |
| `ace-step-ui/components/accordions/*` | **[NEW]** Generation Settings, Track Details, Adapters, Score System, Expert Controls, Guidance Settings, LmCot Accordions |
| `ace-step-ui/components/sections/*` | **[NEW]** Audio Selection, Lyrics, Style, Music Parameters, Cover Repaint Settings, Task Type, Simple Mode Settings, Audio Library Modal |

### How it works

1. **Categorized Settings:** Options are now grouped into logical accordions (Generation Settings, Expert Controls, Adapters, Score System) rather than a single massive scrolling list.
2. **Track Details Isolation:** Lyrics, Style, and Music Parameters are cleanly nested under a Track Details accordion in Custom Mode.
3. **Simple vs Custom Mode:** Simple mode presents a cleaner top-level interface while Custom mode exposes deep configuration.
4. **Tooltips & i18n:** Every single parameter now features a localized tooltip explaining its function.
5. **Maintainability:** The massive 4500-line CreatePanel was decomposed into specific, maintainable UI sections and components.

---

## JSON Export & Import

Export all generation parameters to a JSON file and import them later to reproduce exact configurations. Includes full adapter and steering parameter persistence in the Generation Parameters sidebar.

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/components/CreatePanel.tsx` | Export/Import buttons below Task Type selector. `handleExportJson` serializes all state to a downloadable `.json` file. `handleImportJson` parses the file and restores all state variables, auto-opening steering/adapter panels if applicable. |
| `ace-step-ui/App.tsx` | Replaced 65-line cherry-picked parameter allowlist in `handleGenerate` with `{ ...params }` spread — ensures all current and future parameters flow through to the API automatically. |
| `ace-step-ui/server/src/routes/generate.ts` | Simplified to `const params = req.body as GenerateBody` — stores the entire frontend payload verbatim in the SQLite `params` column. |
| `ace-step-ui/components/RightSidebar.tsx` | Added adapter slot display (name, type, scale) and steering concept display (concept, alpha) to the Generation Parameters sidebar. Removed redundant "Use ADG" entry (now covered by Guidance Mode). |
| `ace-step-ui/services/api.ts` | Synced `GenerationParams` interface with adapter and steering fields, updated `inferMethod` type union. |
| `ace-step-ui/types.ts` | Added `loraLoaded`, adapter, and steering properties to the shared `GenerationParams` interface. |

### How it works

1. **Export:** Click the **Export JSON** button → all generation parameters (including adapter slots, steering concepts, guidance mode, PAG settings, etc.) are serialized and downloaded as a timestamped `.json` file.
2. **Import:** Click the **Import JSON** button → select a previously exported file → all parameters are restored. If the config included loaded adapters or steering concepts, those panels auto-expand.
3. **Sidebar Display:** After generation, the right sidebar now shows:
   - **Basic LoRA** name and scale (only when not using advanced adapters)
   - **Advanced Adapter Slots** with name, type badge, and scale per slot
   - **Steering Concepts** with concept name and alpha value
4. **Future-proof API:** `App.tsx` now spreads params directly, so any new fields added to `CreatePanel` automatically flow through without needing to update the allowlist.

---

## Debug Panel & UI Polish

Live system monitoring panel and UI polish improvements: a collapsible debug panel showing GPU VRAM, RAM, and CPU usage alongside a real-time streaming API log, a resizable Create Panel, and a streamlined sidebar toggle.

### What's included

| File | Description |
|------|-------------|
| `acestep/api_server.py` | Extended `LogBuffer` with ring buffer + cursor. Added `GET /v1/system/metrics` (GPU, RAM, CPU) and `GET /v1/system/logs` (cursor-based). |
| `ace-step-ui/server/src/routes/system.ts` | **[NEW]** Express route: metrics proxy + SSE log stream at `/api/system/*`. |
| `ace-step-ui/server/src/index.ts` | Mounted system routes. |
| `ace-step-ui/components/DebugPanel.tsx` | **[NEW]** Fixed right-edge panel with progress bars, retro terminal log viewer, persisted state. |
| `ace-step-ui/App.tsx` | Integrated DebugPanel, content shift on open, resizable CreatePanel with drag handle. |
| `ace-step-ui/components/Sidebar.tsx` | Replaced logo + separate arrow with a single purple circle toggle (chevron arrow). |
| `requirements.txt` | Added `psutil>=5.9.0` for CPU/RAM metrics. |

### How it works

1. **Debug Panel:** A toggle tab on the right screen edge opens a 400px panel showing VRAM/RAM/CPU metrics (polled every 2s) and a streaming API log with color-coded levels (green text on black — retro terminal style). Panel state persists across sessions.
2. **Content Shift:** When the debug panel opens, the entire layout (including Song Details sidebar) smoothly slides left to keep everything visible.
3. **Resizable Create Panel:** Drag the right edge of the parameters panel to resize it (280–600px). Width persists across sessions. The track list absorbs the change.
4. **Sidebar Toggle:** The purple circle in the top-left now contains a chevron arrow that rotates to indicate expand/collapse. The separate arrow button has been removed.

---

## Stem Extraction (Extract Mode)

Full stem extraction workflow using ACE-Step's generative extract task. Select one or more instrument stems to isolate from a source audio file — each creates a separate queued job. Includes quality presets, style hints, and lyrics guidance for vocal tracks.

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/components/sections/ExtractTrackSelector.tsx` | Multi-select toggle chip UI for choosing stems (12 track types) |
| `ace-step-ui/components/CreatePanel.tsx` | Extract mode flow: quality presets, style hint, lyrics guidance, meta clearing, handleGenerate loop for multi-track jobs |
| `ace-step-ui/server/src/routes/generate.ts` | Passes `src_audio_path`, `reference_audio_path`, `track_name` to Python backend |
| `ace-step-ui/server/src/routes/referenceTrack.ts` | Extended upload whitelist for `.ogg`, `.opus`, `.webm` formats |
| `acestep/api_server.py` | Whitelisted project audio directory in `_validate_audio_path` |
| `acestep/core/generation/handler/io_audio.py` | Replaced `torchaudio.load` with `soundfile.read` for Windows compatibility |
| `ace-step-ui/i18n/translations.ts` | 20+ localized keys for extract UI (en, zh, ja, ko) |

### How it works

1. **Track Selection:** Toggle one or more stems from 12 available track types (Vocals, Backing Vocals, Drums, Bass, Guitar, Keyboard, Strings, Synth, Brass, Woodwinds, Percussion, FX). Each selected track queues a separate extraction job.
2. **Quality Presets:** Three one-click presets configure inference steps, solver, and guidance mode:
   - ⚡ **Low (Quick):** 20 steps, Euler, Dynamic CFG
   - ⚖️ **Medium:** 50 steps, Heun, Dynamic CFG
   - 💎 **High (Slow):** 200 steps, RK4, Dynamic CFG
3. **Style Hint (Optional):** A text field to describe the expected timbre/genre (e.g., "distorted electric guitar, heavy rock") — passed as the `style` parameter to guide generation quality.
4. **Lyrics Guidance (Optional):** For vocal/backing vocal tracks, paste lyrics to improve extraction accuracy. The `instrumental` flag is automatically set to `false` for vocal tracks.
5. **Meta Clearing:** BPM, key, and time signature are zeroed for extract mode so stale values from previous text2music sessions don't interfere — the model relies on the actual source audio.
6. **Title Format:** Extract jobs are titled `"Vocals - My Song.mp3"` instead of generic names, using the source audio filename.
7. **Windows Fix:** Replaced `torchaudio.load` with `soundfile.read` in the audio processing pipeline, resolving `torchcodec` dependency failures on Windows.

> **Note:** ACE-Step's extract is *generative*, not subtractive. Unlike traditional source separation tools (Demucs, BSRNN), the model re-generates what it thinks each stem sounds like based on the source audio and instruction. This means vocal tracks may occasionally hallucinate audio in silent sections.

---

## Server-Side Stem Separation

Professional-grade audio stem separation using BS-RoFormer and Demucs models, fully integrated into the Python API with a synchronized multi-track mixer UI. Replaces the old client-side demucs-web page.

### What's included

| File | Description |
|------|-------------|
| `acestep/stem_service.py` | **[NEW]** Core separation service — lazy-init `audio_separator`, 4 modes (vocals, multi-4, multi-6, two-pass), thread-safe singleton |
| `acestep/api_server.py` | 4 new endpoints: `/v1/stems/available`, `/v1/stems/separate`, `/v1/stems/{job_id}/progress` (SSE), `/v1/stems/{job_id}/download/{stem_type}`. Includes VRAM offloading (ACE-Step models → CPU during separation → GPU after). |
| `install_audio_separator.py` | **[NEW]** Standalone installer — handles PyTorch version detection, ONNX Runtime conflicts, platform-specific deps |
| `1、install-uv-qinglong.ps1` | Updated to run `install_audio_separator.py` during first-time setup |
| `ace-step-ui/components/StemSplitterModal.tsx` | **[NEW]** Self-managing modal with mode selection, SSE progress, and synchronized multi-track mixer |
| `ace-step-ui/components/SongList.tsx` | "Extract Stems" option in track context menu |
| `ace-step-ui/App.tsx` | StemSplitterModal mount point |

### Separation Modes

| Mode | Model | Stems | Quality |
|------|-------|-------|---------|
| **Vocals Only** | BS-RoFormer | 2 (vocals + instrumental) | Best vocal isolation (SDR 12.97) |
| **4-Stem** | htdemucs_ft | 4 (vocals, drums, bass, other) | Fast, good general split |
| **6-Stem** | htdemucs_6s | 6 (vocals, drums, bass, guitar, piano, other) | More instrument detail |
| **Two-Pass (Best)** | RoFormer → htdemucs_6s | 7 | RoFormer vocals + 6-stem instrumental split |

### Multi-Track Mixer

After separation completes, stems are displayed in a synchronized mixer with:
- **Master transport** — single play button syncs all stems, seekable progress bar with time display
- **Per-stem volume slider** — independent gain control for each stem
- **Mute/Solo** — mute individual stems or solo one to hear it in isolation
- **Download** — download individual stems as FLAC files
- Colour-coded channels by stem type (vocals=pink, drums=amber, bass=green, guitar=blue, piano=purple)

### VRAM Management

To prevent out-of-memory errors, ACE-Step's models (DiT, VAE, tokenizer) are automatically moved to CPU before stem separation runs, then restored to GPU afterwards. This frees ~10GB of VRAM for the separation models.

### Windows Compatibility

The `_float32_default_dtype()` context manager temporarily restores `torch.float32` as the default dtype during the entire separation pipeline. This is necessary because ACE-Step sets the global default dtype to `bfloat16` for GPU inference, but both BS-RoFormer and Demucs create internal tensors (via `torch.randn`, `torch.stft`) that inherit this dtype — causing MKL FFT crashes on Windows.

### Models

Models are lazy-downloaded on first use (~1.8 GB total) to `/tmp/audio-separator-models/`:
- `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (BS-RoFormer, ~1.3 GB)
- `htdemucs_6s.yaml` (Demucs 6-stem, ~55 MB — weights downloaded from Facebook servers)
- `htdemucs_ft.yaml` (Demucs fine-tuned, ~85 MB)

## Live Music Visualizer

Real-time audio-reactive visualizations powered by the Web Audio API. Repurposes the existing video generator's drawing engine into a shared module used by both live playback visualization and MP4 export. Features 10 presets, a Winamp/MilkDrop-inspired fullscreen mode, and an optional ambient background for the song list.

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/components/visualizerEngine.ts` | **[NEW]** Shared rendering engine extracted from VideoGeneratorModal — 10 drawing functions, particle system, album art, 13 post-processing effects, unified `renderVisualizerFrame()` entry point |
| `ace-step-ui/context/AudioAnalysisContext.tsx` | **[NEW]** React context providing a shared `AnalyserNode` connected to the main player's `HTMLAudioElement` via `createMediaElementSource` |
| `ace-step-ui/components/LiveVisualizer.tsx` | **[NEW]** Core canvas component with preset picker dropdown, Random mode (~30s auto-cycle), and dimmed mode for background use |
| `ace-step-ui/components/FullscreenVisualizer.tsx` | **[NEW]** Portal-based fullscreen overlay with auto-hiding HUD (song title, artist, progress bar, playback controls). Uses the Fullscreen API. |
| `ace-step-ui/components/VideoGeneratorModal.tsx` | Refactored to import drawing functions from `visualizerEngine.ts` — removed ~350 lines of duplicate code |
| `ace-step-ui/components/RightSidebar.tsx` | Cover art replaced by live visualizer when the displayed song is playing (fade-in animation). Preset picker and fullscreen button overlaid on canvas. |
| `ace-step-ui/components/SongList.tsx` | Optional dimmed background visualizer behind track rows (same preset as sidebar, 15% opacity) |
| `ace-step-ui/components/SettingsModal.tsx` | New "Visualizer" section with toggle for song list background |
| `ace-step-ui/App.tsx` | Wrapped in `AudioAnalysisProvider`, audio analysis connected on play, fullscreen state managed, visualizer bg setting wired from localStorage to SongList |

### How it works

1. **Audio Analysis:** On first playback, the `AudioAnalysisProvider` creates an `AudioContext` and connects an `AnalyserNode` to the player's `HTMLAudioElement`. This single analyser feeds frequency and time-domain data to all visualizer instances.
2. **Sidebar Visualizer:** When a song is playing and selected in the right sidebar, the static cover art fades out and a `LiveVisualizer` canvas fades in. The visualizer renders at the canvas's native resolution using `requestAnimationFrame`.
3. **Preset Picker:** A compact dropdown (Palette icon on the canvas) lets users choose from 10 presets: NCS Circle, Linear Bars, Dual Mirror, Center Wave, Orbital, Hexagon, Oscilloscope, Digital Rain, Shockwave, and Minimal. Selection persists to `localStorage`.
4. **Random Mode:** A "Random" option in the picker auto-cycles through presets every ~30 seconds, picking a different preset each time.
5. **Fullscreen Mode:** Click the Maximize icon → the Fullscreen API is activated, a portal renders the visualizer filling the screen. The HUD (song title, artist, progress bar, play/pause/skip controls) auto-hides after 3 seconds of mouse inactivity and reappears on movement. Keyboard shortcuts: Space (play/pause), Escape (exit), arrows (skip).
6. **Song List Background:** Toggleable in Settings → Visualizer → "Song list background". When enabled and music is playing, a dimmed (15% opacity) version of the same preset renders behind the track rows as an ambient visual.
7. **Shared Engine:** `visualizerEngine.ts` is imported by both `LiveVisualizer` (real-time) and `VideoGeneratorModal` (offline MP4 export), eliminating code duplication.

### Presets

| Preset | Style |
|--------|-------|
| NCS Circle | Radial frequency bars rotating around center point |
| Linear Bars | Classic horizontal spectrum analyzer |
| Dual Mirror | Mirrored horizontal bars from center |
| Center Wave | Concentric elliptical waves |
| Orbital | Animated arcs orbiting center |
| Hexagon | Pulsing hexagonal wireframe |
| Oscilloscope | Real-time waveform display |
| Digital Rain | Matrix-style falling characters |
| Shockwave | Expanding concentric ring pulses |
| Minimal | Clean particles-only |

---

## Synced Lyrics & Song Structure

Real-time synced lyrics display and song structure visualization. LRC lyrics files are automatically downloaded alongside generated audio, displayed as an overlay on the visualizer and a collapsible bar in the song list. Section markers from the LRC file (Verse, Chorus, Bridge, etc.) are shown as positioned labels above the player waveform.

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/server/src/routes/generate.ts` | Downloads `.lrc` file alongside audio during generation, saving it with matching UUID |
| `ace-step-ui/components/LyricsOverlay.tsx` | Synced lyrics overlay on the art box visualizer — fetches LRC directly, parses with `\r\n` handling, filters section markers |
| `ace-step-ui/components/LyricsBar.tsx` | **[NEW]** Collapsible bar at the bottom of the song list showing one lyric line at a time with smooth fade-up + blur animation |
| `ace-step-ui/components/SectionMarkers.tsx` | **[NEW]** Thin row above the player waveform with section labels (Verse, Chorus, Bridge, etc.) positioned proportionally by timestamp |
| `ace-step-ui/components/SongList.tsx` | Integrated LyricsBar, conditionally hidden during A/B comparison mode |
| `ace-step-ui/components/Player.tsx` | Integrated SectionMarkers above the waveform progress bar |
| `ace-step-ui/App.tsx` | Passes `currentTime` prop through to SongList for lyrics synchronization |

### How it works

1. **LRC Download:** When a song is generated, the Node.js server downloads the `.lrc` file from the Python API alongside the audio file, saving it as `/audio/{userId}/{songId}.lrc`.
2. **Lyrics Overlay:** The `LyricsOverlay` component fetches the `.lrc` file by swapping the audio file extension, parses timestamps, and displays lines synced to playback. Section markers (e.g., `[Verse 1]`) are filtered from display but preserved in parsed data.
3. **Lyrics Bar:** A collapsible bar at the bottom of the song list shows the current lyric line in large white text with a smooth fade-up + blur-to-sharp animation on each line change. Expanded by default, hidden during A/B comparison mode.
4. **Section Markers:** Section markers from the LRC file are parsed and displayed in a thin row above the player waveform. Labels like `[Chorus - Exciting]` are cleaned to just `Chorus`. Each marker is positioned at its proportional timestamp with a vertical tick at the section boundary.
5. **Windows Compatibility:** The LRC parser strips `\r` characters before splitting lines, handling Windows-style `\r\n` line endings that otherwise break the timestamp regex.

---

## Visualizer Preset Selection

Configurable pool of visualizer presets for random rotation. Users choose which presets are included via checkboxes in Settings. Multiple visualizer instances (art box, song list background, fullscreen) coordinate to never show the same preset simultaneously.

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/components/SettingsModal.tsx` | Checkbox grid in the Visualizer section for toggling presets, Select All/Clear buttons, stored in `localStorage` |
| `ace-step-ui/components/LiveVisualizer.tsx` | Reads enabled presets from `localStorage`, instance coordination via global registry, `instanceId` prop |
| `ace-step-ui/components/SongList.tsx` | `instanceId="songlist"` on background visualizer |
| `ace-step-ui/components/RightSidebar.tsx` | `instanceId="artbox"` on art box visualizer |
| `ace-step-ui/components/FullscreenVisualizer.tsx` | `instanceId="fullscreen"` on fullscreen visualizer |

### How it works

1. **Settings UI:** Open Settings → Visualizer → a 2-column checkbox grid shows all 10 presets with emoji labels. Toggle each on/off. Must keep at least 1 enabled.
2. **Enabled Pool:** The enabled preset list is stored in `localStorage` as `visualizer_enabled_presets`. Changes are broadcast via `StorageEvent` so all instances update immediately.
3. **Instance Coordination:** Each `LiveVisualizer` instance registers its current preset in a global `activePresets` map keyed by `instanceId`. When cycling in Random mode, each instance excludes presets currently shown by other instances.
4. **Defaults:** NCS Circle, Linear Bars, Dual Mirror, and Oscilloscope are enabled by default.

---

## A/B Track Comparison

Side-by-side A/B comparison of any two tracks in the song list. Both audio elements play simultaneously (one muted, one audible) with instant switching, synchronized positions, and full pause/resume control.

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/App.tsx` | `abTrackA/B`, `abActive`, `abAudioRef` state, `handleABPlay` (starts comparison), `handleABToggle` (swaps mute/unmute with pause-state respect), `handleABClear`, dual-audio `isPlaying` sync effect |
| `ace-step-ui/components/SongList.tsx` | A/B selection badges on song items, comparison bar as a flex footer at the bottom of the song list (outside the scroll container), `Play Comparison` / `A/B Toggle` / `Diff` / `Clear` controls |
| `ace-step-ui/components/SongDropdownMenu.tsx` | "Set as Track A" / "Set as Track B" options in the per-song context menu |
| `ace-step-ui/components/accordions/AdaptersAccordion.tsx` | Native folder picker for Browse buttons via `browseLoraFolder` API |
| `ace-step-ui/services/api.ts` | `browseLoraFolder()` API client method |
| `ace-step-ui/server/src/routes/lora.ts` | `GET /browse-folder` — opens native Windows folder picker dialog via PowerShell `FolderBrowserDialog` |

### How it works

1. Right-click any song → "Set as Track A". Right-click another → "Set as Track B". Both appear as `A` / `B` badges on the song items.
2. A comparison bar appears at the bottom of the song list with track labels, Play Comparison, Diff, and Clear buttons.
3. **Play Comparison** creates a secondary `HTMLAudioElement` for Track B, syncs position to Track A, and starts both. A is audible, B is muted.
4. **A/B Toggle** switches which track is audible by swapping mute states and syncing `currentTime`. Position stays perfectly aligned.
5. **Pause-state awareness** — if the user pauses during comparison (via the player bar), toggling A/B does NOT auto-resume. Both audio elements respect `isPlaying` state.
6. **Diff** opens a parameter comparison modal showing all generation settings side-by-side.
7. **Clear** removes the comparison selection and cleans up the secondary audio element.

### Native Folder Picker

The **Browse** button on both basic and advanced adapter panels now opens a native Windows folder picker dialog (PowerShell `FolderBrowserDialog`) instead of scanning an existing folder path. The selected folder path is written directly into the adapter folder input field.

---

<!-- 
## [Next Feature Name]

Brief description.

### What's included
- ...

### How it works
- ...
-->

---

## Timestep Scheduler

Pluggable timestep distribution system for the diffusion process. Controls *where* denoising steps are concentrated across the noise schedule, complementing the existing solver (which controls *how* each step is computed) and guidance mode (which controls *what direction* each step moves).

### What's included

| File | Description |
|------|-------------|
| `acestep/core/generation/schedulers.py` | **[NEW]** Registry + 6 schedule implementations including composite 2-stage |
| `acestep/models/sft/modeling_acestep_v15_base.py` | Integrated `get_schedule()` into `generate_audio()` (SFT variant) |
| `acestep/models/base/modeling_acestep_v15_base.py` | Integrated `get_schedule()` into `generate_audio()` (base variant) |
| `acestep/inference.py` | `scheduler` field in `GenerationParams` dataclass |
| Full API chain (4 files) | `scheduler` wired through request model, parser, builder, setup |
| Full handler chain (5 files) | Threaded through `diffusion` → `service_generate_execute` → `service_generate` → `generate_music_execute` → `generate_music` |
| `ace-step-ui` | Scheduler dropdown, composite sub-controls (Stage A/B, crossover, split), metadata display, i18n strings |

### Available schedulers

| Scheduler | Description |
|-----------|-------------|
| **Linear** (default) | Uniform spacing. Backward-compatible with all existing workflows. |
| **DDIM Uniform** | Log-SNR uniform in logit(t) space. S-shaped distribution — dense around t=0.5, balanced structure and detail. |
| **SGM Uniform** | Karras σ-ramp (ρ=7). Moderate front-loading for structural focus without starving detail. |
| **Bong Tangent** | Tangent-based front-loading. More budget for structural decisions. |
| **Linear Quadratic** | Linear start → quadratic end. More budget for fine detail refinement. |
| **Composite (2-Stage)** | Two-stage: pick different schedulers for the structural (high noise) and detail (low noise) phases. |

### How it works

1. Each scheduler maps the denoising step count to a sequence of timestep values between 0 and 1
2. The scheduler runs *after* the shift parameter is applied, so both compose naturally
3. Select a scheduler from the **Timestep Scheduler** dropdown in Generation Settings (below the Solver dropdown)
4. The scheduler name is logged alongside solver, guidance, and steps in the diffusion info line
5. Default is `linear` — existing generations are unaffected unless you explicitly change it

### Composite (2-Stage) scheduler

Inspired by ComfyUI's multi-pass denoising workflows, the **Composite** scheduler splits the diffusion trajectory into two phases at a configurable crossover timestep:

- **Stage A (Structure):** Handles the high-noise region (t=1 → crossover). Choose any scheduler for this phase — e.g., Bong Tangent for strong structural decisions.
- **Stage B (Detail):** Handles the low-noise region (crossover → 0). Choose a different scheduler — e.g., Linear Quadratic for fine detail refinement.
- **Crossover** (0.1–0.9): The timestep value where Stage A ends and Stage B begins. Default 0.5.
- **Step Split** (0.1–0.9): Fraction of total steps allocated to Stage A. Default 0.5 = equal split

---

## Auto-Mastering & Mastering Console

Automatic post-generation mastering that applies a professional mastering profile to every generated track. The system includes an interactive console for real-time adjustments, a reference-based matching mode using the Matchering library, persistent settings across sessions, and the ability to remaster previously generated tracks.

### What's included

| File | Description |
|------|-------------|
| `acestep/core/audio/mastering.py` | `MasteringEngine` class — applies a 6-stage processing chain via pedalboard, maps frontend parameters to DSP values |
| `acestep/core/audio/presets/*.json` | Bundled default presets (Preset 1, Preset 2) learned from professional reference mastering |
| `acestep/inference.py` | `auto_master` and `mastering_params` in `GenerationParams`; mastering hook runs between normalization and audio save. Added Matchering branch for reference-based mastering. |
| `acestep/api/http/mastering_routes.py` | **[NEW]** Provides the `/v1/mastering/remaster` endpoint to apply new settings to existing audio files |
| `ace-step-ui/components/MasteringConsoleModal.tsx` | **[NEW]** Interactive console with sliders for 5-band EQ, Exciter Drive, Stereo Width, Threshold, Ratio, Gain, and Ceiling |
| `ace-step-ui/components/sections/CoverRepaintSettings.tsx` | Interactive UI selector to choose between `Built-in` and `Matchering` mastering methods, along with an audio file uploader for the Matchering reference track. |
| `ace-step-ui/components/DownloadModal.tsx` | Upgraded to allow downloading 'original', 'mastered', or 'both' versions of a track |
| `ace-step-ui/App.tsx` | Global parameter persistence via `localStorage`, download version handling, console mounting |

### Processing Chains

**Built-in Mode (`MasteringEngine`)** applies 6 stages in professional gain-staging order:
1. **EQ shaping** — 5-Bands (Sub, Low, Mid, Presence, Air) 
2. **Harmonic saturation** — Soft-clip exciter drive for subtle warmth
3. **Stereo widening** — Mid/side processing to expand the stereo image
4. **Compression** — Dynamic range taming with configurable threshold and ratio
5. **Loudness push + limiter** — Gain boost through a brick-wall limiter with an absolute output ceiling
6. **Peak normalization** — Final safety normalize to -0.1 dBFS

**Matchering Mode** uses [sergree/matchering](https://github.com/sergree/matchering) (GPLv3) to apply reference-based EQ and loudness matching:
1. **Target Analysis** — Measures the frequency response and dynamics of your provided reference `.wav`/`.mp3`.
2. **Source Analysis** — Measures the generated ACE-Step audio.
3. **Matching** — Dynamically matches the EQ curve and overall RMS amplitude to match the target reference.

**Stem Matchering Mode** extends Matchering with per-stem processing for better tonal fidelity:
1. **Stem Separation** — Both reference and generated audio are split into 4 stem groups (vocals, drums, bass, other) using the built-in `StemService` (BS-RoFormer + Demucs). Guitar, piano, and other instruments are merged into the "other" group.
2. **Per-Stem Matching** — Each stem pair is individually matched via `matchering.process()`. Near-silent stems (RMS < 1e-6) are skipped to avoid amplifying noise.
3. **Recombination** — The 4 matched generation stems are summed back together.
4. **LUFS Normalization** — The recombined mix is loudness-matched to the reference track.
5. **Brickwall Limiting** — A Pedalboard limiter (ceiling -0.3 dBFS) catches any inter-stem clipping from the recombination.
6. **Final Polish** — A full-mix `matchering.process()` pass against the original reference for final tonal glue.

> **Trade-off:** Stem Matchering produces significantly better tonal fidelity than standard Matchering, but takes longer (~30-60s extra) due to the two stem separation passes.

### How it works

1. **Mode Selection**: Open the Output Processing accordion and turn on Auto-Master. Select either **Built-in** or **Matchering**.
2. **Built-in Interactive Console**: Click the sliders icon on any generated track to open the Mastering Console. You can tweak all the DSP parameters in real-time. Hover over parameter labels to view tooltips explaining how they affect the sound and warning against clipping.
3. **Matchering Reference Upload**: When Matchering is selected, click **Upload Reference** to attach a target song. The generated track will be matched to this target's sonic profile transparently.
4. **Stem Matchering Toggle**: When Matchering is selected and a reference file is uploaded, enable **Stem Matchering** for per-stem processing. This separates both tracks into stems, matches each individually, then recombines — better tonal fidelity at the cost of longer processing time.
5. **Persistent Settings**: Any tweaks made in the console or your selected mastering mode are saved automatically to `localStorage` and act as your default global mastering profile for all future generated tracks.
6. **Remastering**: Click **Remaster** in any track's dropdown menu. This opens the console loaded with the track's existing mastering settings. You can tweak the sliders, choose a different preset, and hit Apply — the backend re-runs the pipeline against the *original unmastered audio file* and creates a new master instantly.
7. **Download Choice**: When downloading an auto-mastered track, the Download Modal offers a sub-selection: download the final **Mastered** version, the raw uncompressed **Original** generated by diffusion, or **Both**.

### Dependencies

- **Required:** `pedalboard`, `numpy`, `matchering`
- The built-in engine lazy-imports `pedalboard` on first use — zero overhead when disabled.
- The Matchering engine leverages the new Python `matchering` library package dynamically if selected.

---

## Audio Enhancement Studio *(Legacy)*

> ⚠️ **Deprecated:** The [Auto-Mastering](#auto-mastering) feature above replaces this for most use cases. This tool remains available for users who want manual per-stem DSP control.

**Based on:** [ShmuelRonen/ComfyUI-Audio_Quality_Enhancer](https://github.com/ShmuelRonen/ComfyUI-Audio_Quality_Enhancer)

Post-processing engine for enhancing generated audio quality. Ported from the ComfyUI Audio Quality Enhancer's "AI Audio Enhancer Pro" node, adapted to run as a standalone backend service with a React modal UI. No external binaries required (SoX dependency removed — reverb/echo implemented purely in Python).

### What's included

| File | Description |
|------|-------------|
| `acestep/core/audio/enhancer.py` | Core DSP engine: multi-band EQ, compression, reverb, echo, stereo widening, per-stem enhancement, preset system |
| `acestep/api_server.py` | 4 new endpoints: `GET /v1/audio/enhance/available`, `POST /v1/audio/enhance`, `GET /v1/audio/enhance/{job_id}/progress` (SSE), `GET /v1/audio/enhance/{job_id}/download` |
| `ace-step-ui/components/AudioEnhancerModal.tsx` | Full modal UI with presets, grouped sliders, mode toggle, SSE progress, preview player, download |
| `ace-step-ui/components/SongDropdownMenu.tsx` | Added "Enhance Audio" menu item with Sparkles icon |
| `ace-step-ui/App.tsx` | Registered `<AudioEnhancerModal />` |
| `ace-step-ui/i18n/translations.ts` | `enhanceAudio` key in en/zh/ja/ko |
| `requirements.txt` | Added `pedalboard` dependency |

### How it works

**Two processing modes:**

1. **Simple mode** — Applies multi-band EQ and dynamics processing directly to the full mix:
   - **Warmth:** Low-shelf filter at ~100Hz (pedalboard `LowShelfFilter` or scipy Butterworth fallback)
   - **Clarity:** Peak filter at ~2.5kHz for vocal presence
   - **Air/Brilliance:** High-shelf filter at ~10kHz
   - **Dynamics:** Compressor + transient detection and boost

2. **Stem-Separation mode** — Uses Demucs (`htdemucs`) to split audio into vocals, drums, bass, and other, then applies targeted per-stem enhancement before remixing:
   - Vocals: Presence boost (3.5kHz), de-essing (7.5kHz cut), air shelf
   - Drums: Transient detection + boost, high-end air for cymbals
   - Bass: Low-shelf warmth, harmonic saturation for definition
   - Other: Balanced 3-band EQ (warmth + clarity + air)

**Effects (no SoX dependency):**
- **Reverb:** Convolution with synthetically generated impulse responses (early reflections + exponential decay noise). Configurable room size and damping
- **Echo:** Delay line with configurable delay time (0–0.5s) and feedback decay (4 repeats)
- **Stereo widening:** Mid/side decomposition, frequency-dependent width, Haas effect (small inter-channel delay), bass centering below 150Hz, subtle saturation for cohesion

**6 built-in presets:** Radio Ready, Warm & Rich, Bright & Clear, Club Master, Lo-Fi Chill, Cinematic — each sets all parameters to curated values.

### Dependencies

- **Required:** `numpy`, `scipy`, `soundfile` (already in requirements)
- **Recommended:** `pedalboard` (now in requirements — provides higher-quality EQ, compression, and limiting via Spotify's audio processing library)
- **Optional:** `demucs` (for stem-separation mode — falls back to simple mode if unavailable)

### VRAM offloading

When using stem-separation mode, ACE-Step's models (DiT, VAE, tokenizer) are automatically offloaded to CPU before Demucs runs, then restored to GPU afterwards — same pattern as the stem separation feature.

---

## JKASS Inference Solvers

Two additional ODE solver algorithms — **JKASS** and **JKASS Fast** — designed specifically for music diffusion models. Both provide frequency-dependent processing and optional post-step smoothing that the generic mathematical solvers (Euler, Heun, etc.) lack.

### What's included

| File | Description |
|------|-------------|
| `acestep/core/generation/solvers.py` | `jkass` and `jkass_fast` solver implementations with beat stability, frequency damping, and temporal smoothing |
| `acestep/models/base/modeling_acestep_v15_base.py` | `solver_state` dictionary wired into the denoising loop, passed to solver step functions |
| `acestep/inference.py` | `beat_stability`, `frequency_damping`, `temporal_smoothing` in `GenerationParams` |
| Full API chain (4 files) | Parameters wired through Pydantic model, request builder, generation setup |
| `ace-step-ui` | JKASS Fast Controls sub-panel (3 sliders with help text), reset button, conditional display when `jkass_fast` solver selected |

### How it works

1. **JKASS** — A 2nd-order Heun-class solver with frequency-aware momentum blending. After each Heun step, it applies spectral damping (exponential decay on high-frequency bins) and temporal smoothing (convolution kernel across the time axis). This reduces harsh overtones and temporal jitter without sacrificing musical structure.
2. **JKASS Fast** — A single-evaluation variant that achieves similar results at half the compute cost. Uses momentum blending with the previous step, spectral damping, and temporal smoothing — all configurable via the UI.
3. **JKASS Fast Controls** appear automatically when JKASS Fast is selected as the inference method:
   - **Beat Stability** (0–1, default 0.25): Momentum blending with the previous denoising step — smooths rhythmic elements
   - **Frequency Damping** (0–5, default 0.4): Exponential decay on high-frequency bins — tames harsh overtones
   - **Temporal Smoothing** (0–1, default 0.13): Smoothing kernel across the time axis — reduces temporal jitter
4. A **Reset** button restores all three parameters to their recommended defaults.

---

## Anti-Autotune System

Spectral smoothing for reducing robotic vocal artifacts that can occur in AI-generated music. Operates on the generated audio tensor before saving, applying a gentle spectral envelope smoothing pass that preserves natural timbre while reducing the "auto-tuned" quality of AI vocals.

### What's included

| File | Description |
|------|-------------|
| `acestep/inference.py` | `anti_autotune` field in `GenerationParams` (0.0–1.0); applied during audio post-processing |
| Full API chain (4 files) | Wired through Pydantic model, request builder, generation setup |
| `ace-step-ui` | Anti-Autotune slider in the Generation Settings accordion |

### How it works

1. The slider maps to `anti_autotune` (0.0 = off, 1.0 = full smoothing)
2. When enabled, a spectral envelope smoothing pass is applied to the generated audio before normalization and mastering
3. This reduces the unnaturally precise pitch quantization that gives AI vocals their characteristic "auto-tuned" sound
4. Best used at subtle values (0.1–0.3) for natural-sounding vocals; higher values can over-smooth

---

## Advanced Guidance Parameters

Extension of the existing [Advanced Guidance & Solver Modes](#advanced-guidance--solver-modes) with fine-grained control over individual guidance components, decay scheduling, and per-prompt-type guidance scaling.

### What's included

| File | Description |
|------|-------------|
| `acestep/inference.py` | 8 new fields in `GenerationParams`: `guidance_scale_text`, `guidance_scale_lyric`, `apg_momentum`, `apg_norm_threshold`, `omega_scale`, `erg_scale`, `guidance_interval_decay`, `min_guidance_scale` |
| Full API chain (4 files) | All parameters wired through Pydantic model, request builder, generation setup |
| `ace-step-ui/components/accordions/GuidanceSettingsAccordion.tsx` | Advanced Guidance sub-panel with text/lyric scale overrides, APG tuning, omega/ERG scales, and reset button |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Text Scale** | 0 (use main) | Independent guidance scale for the text/style prompt. When 0, uses the main guidance scale. |
| **Lyric Scale** | 0 (use main) | Independent guidance scale for lyrics. Allows stronger lyric adherence without affecting musical style. |
| **APG Momentum** | 0.75 | Adaptive Projected Gradient momentum — controls how much previous gradient direction influences the current step |
| **APG Norm Threshold** | 2.5 | Gradient clamping threshold for APG — prevents gradient explosion |
| **Omega Scale** | 1.0 | Omega guidance multiplier — scales the overall guidance signal |
| **ERG Scale** | 1.0 | Energy-Renormalized Guidance multiplier — preserves energy norms during guidance |
| **Guidance Decay** | 0 (off) | Linear decay of guidance scale over the diffusion trajectory. Stronger guidance early (structure), softer late (detail). |
| **Min Guidance** | 3.0 | Floor value when guidance decay is active — prevents guidance from dropping too low |

### How it works

1. Open the **Guidance Settings** accordion → the **Advanced Guidance** sub-panel appears below the main CFG scale slider
2. Set independent text/lyric scales to decouple prompt adherence from lyric fidelity
3. Tune APG momentum and norm threshold for smoother or more aggressive gradient-based guidance
4. Enable guidance decay for a "strong start, gentle finish" scheduling approach
5. A **Reset** button restores all parameters to model defaults

---

## Custom VLLM Backend

A third LM backend option alongside PyTorch (`pt`) and standard `vllm`. The Custom VLLM backend uses a bespoke vLLM integration with KV-cache pooling, custom memory management, and optimized token sampling — purpose-built for ACE-Step's two-phase generation (text tokens → audio codes).

### What's included

| File | Description |
|------|-------------|
| `acestep/llm_inference.py` | `custom-vllm` backend initialization, KV-cache pool management, active slot tracking, custom pipeline integration |
| `acestep/api/startup_llm_init.py` | `custom-vllm` added to valid backend set |
| `acestep/api/llm_readiness.py` | Readiness checks for `custom-vllm` |
| `acestep/api/http/model_switch_routes.py` | Hot-switch support between `pt`, `vllm`, and `custom-vllm` backends |
| `acestep/api/http/release_task_models.py` | `custom-vllm` added to `lm_backend` literal type |
| `ace-step-ui/components/accordions/LmCotAccordion.tsx` | "Custom VLLM" option in the LM Backend dropdown |
| `ace-step-ui/components/CreatePanel.tsx` | State management for the third backend option |

### How it works

1. Select **Custom VLLM** from the **LM Backend** dropdown in the LM / CoT accordion
2. The backend hot-switches from the current LM engine to the custom vLLM pipeline (~5–15s)
3. Custom VLLM uses a bespoke KV-cache pool with explicit slot management — memory is pre-allocated and reused across generations rather than allocated/freed each time
4. The custom pipeline handles ACE-Step's two-phase generation (text reasoning → audio code generation) with phase-specific sampling parameters
5. VRAM usage is comparable to standard VLLM (~9.2 GB) but with potentially better throughput on high-end GPUs due to cache pooling

### Switching backends

All three backends can be hot-switched at any time via the UI dropdown. The current model is unloaded, VRAM is freed, and the new backend is initialized with the same model weights. No server restart required.

---

## Vocoder Enhancement (HiFi-GAN)

Optional high-quality audio decode pass using ADaMoSHiFiGAN. After the standard VAE decoding, the generated waveform is re-encoded to a mel spectrogram and decoded through a dedicated vocoder model for improved timbre, clarity, and reduced artifacts.

### What's included

| File | Description |
|------|-------------|
| `acestep/core/audio/music_vocoder.py` | `ADaMoSHiFiGANV1` model class — HiFi-GAN architecture with mel-spectrogram encode/decode |
| `acestep/core/vocoder_service.py` | **[NEW]** `VocoderService` singleton — lazy model loading, checkpoint scanning, sample-rate resampling (48kHz ↔ 44.1kHz), shape normalization |
| `acestep/inference.py` | `vocoder_model` field in `GenerationParams`; vocoder hook runs between VAE decode and normalization |
| Full API chain (4 files) | `vocoder_model` wired through Pydantic model, request builder, generation setup |
| `ace-step-ui` | Vocoder Enhancement toggle with model dropdown, auto-populated from scanned checkpoints |

### How it works

1. **Model Installation:** Run `install.bat` and select "Yes" when prompted to download the vocoder model (~206 MB). The model is placed in `checkpoints/music_vocoder/`.
2. **Enable:** In the UI, open the Output Processing section and enable **Vocoder Enhancement**. The dropdown auto-populates with any vocoder models found in `checkpoints/`.
3. **Processing Pipeline:** After the standard VAE decode produces a waveform:
   - The stereo audio (`[2, time]`) is reshaped to `[2, 1, time]` (2 mono batches)
   - Resampled from 48kHz → 44.1kHz (HiFi-GAN's native rate)
   - Encoded to mel spectrogram via `model.encode()`
   - Decoded back to waveform via `model.decode()`
   - Resampled back to 48kHz and reshaped to `[2, time]`
4. The vocoder model is loaded lazily on first use and cached in memory for subsequent generations.
5. Set the dropdown to **None** to disable and skip the vocoder pass entirely.

### Model source

The ADaMoSHiFiGANV1 model is available from the official ACE-Step repository:
- **Repository:** [`ACE-Step/ACE-Step-v1-3.5B`](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/tree/main/music_vocoder)
- **Files:** `config.json` (940 B) + `diffusion_pytorch_model.safetensors` (206 MB)
- **Location:** `checkpoints/music_vocoder/`

