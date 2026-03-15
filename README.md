<div align="center">

# HOT-Step 9000

<img src="./images/hotstep9000-transparentlogo.webp" width="400px" alt="HOT-Step 9000 Logo">

**Local AI Music Generator for Windows**

*Summon bangers directly from your GPU.*

</div>

## About

**HOT-Step 9000** is a fully functional, open-source local AI music generation suite designed for Windows.

This project is a standalone evolution modification of [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5). It originally started as a fork of [sdbds/ACE-Step-1.5-for-windows](https://github.com/sdbds/ACE-Step-1.5-for-windows) but has since grown into its own distinct application with significant UI Overhauls, advanced features, and quality-of-life improvements.

### ⚠Work In Progress
This application is under active, ongoing development. You may occasionally encounter bugs or unoptimized features. We welcome bug reports and feature requests via GitHub Issues. 

While currently optimized and supported specifically for **Windows environments**, PRs contributing cross-platform support are welcome.

---

## Key Features

HOT-Step 9000 sits on top of the original ACE-Step backend but introduces a massive array of new tools and features for advanced AI music creation:

- **Advanced Multi-Adapter System:** Slot-based loading of up to 4 simultaneous LoRA/LoKr adapters with per-layer scaling via weight-space merging.
- **Activation Steering (TADA):** Experimental zero-shot generation guidance by modifying model activations directly.
- **Advanced Guidance & Solvers:** 7 guidance modes (including PAG, APG, ADG) and 4 ODE solvers (Euler, Heun, DPM++ 2M, RK4).
- **Stem Extraction:** Both generative extraction (via DiT) and deterministic separation (BS-RoFormer/Demucs) with a built-in multi-track mixer.
- **Upscale to HQ:** Re-run low-step preview audio at higher quality settings using precomputed audio codes.
- **Melodic Variation / Repetition Settings:** Adjust the language model's repetition penalty for varied or hypnotic melodic phrasing.
- **Quality Scoring:** Automatic PMI and DiT alignment scores on every generated track.
- **Synced Lyrics & Live Visualizer:** Real-time Web Audio API visualizers (10 presets) with synced `.lrc` lyrics overlays and section markers.
- **A/B Track Comparison:** Side-by-side simultaneous playback of any two tracks for critical listening.
- **Layer Ablation Lab:** Developer tool to run automated sweeps across adapter layers to isolate their functional impact.
- **One-Click Hot-Switching:** Change LM and DiT models instantly without restarting the server.
- **JSON Export/Import:** Save and share complete generation configurations.
- **UI & QoL:** Persistent settings, debug panel, queue management, job cancellation, tempo/pitch shifting in cover mode, and a simple one-click shutdown.

*For a detailed, technical breakdown of every new feature, see [FEATURES.md](./FEATURES.md).*

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
- UI Overhaul, advanced tooling, and new features by **scragnog**.

## License

This project inherits the licensing of its upstream parents. See original repositories for detailed model and code licensing. Please use AI generation tools responsibly.
