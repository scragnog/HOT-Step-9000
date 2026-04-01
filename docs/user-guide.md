# ACE-Step UI: End-User Guide

Welcome to the ACE-Step UI, a professional, Spotify-inspired interface for generating high-quality AI music locally. This guide will walk you through the user interface, the different types of generative tasks you can perform, and the advanced controls available to shape your musical creations.

## 1. The Interface Overview

The ACE-Step UI is designed to be intuitive while offering deep control over the generation process. The main layout consists of several key areas:

- **Sidebar (Left):** Access your library, playlists, liked songs, and settings. It also features a simple shutdown button to safely close the application.
- **Track List (Center):** Displays your generated songs, complete with cover art, titles, duration, and quality scores. You can play, pause, and manage your tracks from here.
- **Player (Bottom):** A full-featured audio player with a waveform visualizer, section markers, and synchronized lyrics display.
- **Create Panel (Right):** The heart of the application. This is where you configure all parameters for your next generation. It can be resized by dragging its left edge.
- **Debug Panel (Hidden/Right Edge):** Toggle this panel to view real-time system metrics (VRAM, RAM, CPU) and streaming API logs.

## 2. Generative Tasks (Task Types)

The application supports several distinct modes of operation, selectable from the **Task Type** dropdown in the Create Panel.

### Text to Music
This is the default mode for creating original songs from scratch.
- **Lyrics:** Enter your own lyrics or use structural tags (e.g., `[Verse]`, `[Chorus]`). If you leave this blank and toggle **Instrumental**, the model will generate a track without vocals.
- **Style:** Describe the genre, mood, instrumentation, and tempo (e.g., "upbeat synthwave with driving bass").
- **Music Parameters:** Explicitly set the BPM, Key/Scale, and Time Signature to force the model into a specific musical framework.

### Cover Mode (Audio to Audio)
Transform an existing audio file into a new style while preserving its core melody and structure.
- **Source Audio:** Upload or select a reference track.
- **Tempo Scale & Pitch Shift:** Adjust the speed (e.g., 1.2x faster) and transpose the pitch (e.g., +3 semitones) of the source audio *before* the model processes it.
- **Cover Strength:** Control how closely the generation adheres to the original audio versus the new style prompt.

### Repaint Mode
Regenerate a specific section of a track without altering the rest. This is useful for fixing a bad vocal take or changing a specific instrumental solo. You define the start and end times for the section you wish to "repaint."

### Extract Mode
A generative stem extraction tool. 
- Select one or more instruments (Vocals, Drums, Bass, Guitar, etc.) to isolate from a source audio file. 
- You can provide style hints or lyrics to guide the extraction, which is particularly helpful for vocal tracks.
- Note: This process is *generative*, meaning the model reconstructs the isolated stem based on the source, which can yield incredibly clean results.

## 3. Advanced Generation Controls

For users who want fine-grained control over the AI's output, the Create Panel offers several advanced accordions:

### Generation Settings & Solvers
- **Inference Steps:** Higher steps generally yield better quality but take longer.
- **Inference Method (Solver):** Choose the mathematical algorithm used for generation. *Euler* is fast, while *Heun* and *RK4* offer higher quality at the cost of speed.
- **Guidance Scale:** Determines how strictly the AI follows your text prompt.

### Guidance Modes
Choose how the text guides the music:
- **APG / ADG:** Specialized audio-flow projections.
- **PAG (Perturbed-Attention Guidance):** Enhances structural clarity and prevents the music from collapsing into noise over long generations.

### Activation Steering (TADA)
A cutting-edge feature that allows you to apply "concept vectors" directly to the model's brain. You can dial in specific musical traits (like "more acoustic" or "less distorted") using alpha sliders without changing your text prompt.

### Adapters (LoRA)
Load specialized mini-models (LoRAs) to mimic specific artists, instruments, or styles. The **Layer Ablation Lab** (in Developer Mode) even allows you to isolate and adjust the influence of specific neural network layers (e.g., isolating the "Voice" layers from the "Style" layers).

## 4. Lyric Studio (Lireek)

The Lyric Studio is a powerful, dedicated subsystem for fetching, profiling, and generating artist-accurate lyrics. It operates independently from the main generation panel but integrates directly into the audio workflow.

### Key Workflows in Lyric Studio
- **Fetch from Genius:** Search for any artist or specific album. The system will download up to 50 songs from Genius to build a dataset of their lyrical style.
- **Build Profile:** Once you have a lyrics set, the AI analyzes the artist's vocabulary, themes, sentence structures, and rhyming patterns to create a comprehensive "Profile."
- **Generate Lyrics:** Using the profile, the AI generates completely new lyrics that mimic the original artist's style. You can provide extra instructions (e.g., "Write a song about space travel") to guide the topic.
- **Refine:** If the generated lyrics aren't quite right, use the Refine tool to polish them. The system automatically enforces proper formatting, punctuation, and structural constraints (e.g., standardizing verse and chorus lengths).
- **Queue System:** For bulk operations, you can queue multiple profile builds or lyric generations to run sequentially in the background.

### Integration with Audio Generation
Once you are happy with a generated lyric sheet, you can seamlessly send it to the main Create Panel to generate the actual audio track, bridging the gap between text generation and music synthesis.

## 5. Output and Post-Processing

Once your track is generated, the UI provides tools to refine it further:

- **Quality Scoring:** Each track receives an automatic PMI score (semantic alignment) and a DiT star rating (audio fidelity) displayed on its card.
- **Upscale to HQ:** If you generated a quick preview at low steps, use the track menu to "Upscale to HQ." This re-runs the generation using the exact same latent codes but at a higher step count for pristine audio quality.
- **Auto-Mastering:** Enable the Mastering Console to apply professional EQ, compression, and normalization to your final track, ensuring it sounds loud and clear.

---
*Tip: Hover over any parameter in the Create Panel to view a helpful tooltip explaining its function.*
