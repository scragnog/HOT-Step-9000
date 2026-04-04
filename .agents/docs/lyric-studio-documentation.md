# Lyric Studio

> **Your lyrics. Your voice. Your sound.**
>
> Lyric Studio is HOT-Step 9000's integrated songwriting environment — a complete pipeline from raw lyrics to fully produced audio. Feed it your own writing, build a stylistic profile from your body of work, and generate new original lyrics that sound like *you*, not a generic AI.

---

## Overview

Lyric Studio turns your existing lyrics into a creative engine. The core workflow is:

1. **Organise** your lyrics into artists and albums
2. **Profile** your writing style using AI-powered analysis
3. **Generate** new original lyrics that match your voice
4. **Produce** — send generated lyrics directly to HOT-Step's audio engine with one click

Every step is designed around *your own work*. You create an artist identity, add your own songs, build a profile from your writing, and use that profile to generate new material that extends your creative voice.

---

## Core Concepts

### Artists

An **artist** is the top-level identity in Lyric Studio. This could be:

- **Your own artist/band name** — the primary use case
- **A project or persona** you're developing
- **A stylistic experiment** — try different writing approaches under different names

Each artist has:
- A display name
- An optional image URL (for visual identification in the grid)
- One or more **albums** (lyrics sets) containing source material
- Profiles and generated songs linked through those albums

### Albums (Lyrics Sets)

An **album** is a collection of source lyrics grouped under an artist. Think of it as a body of work that defines a particular style or era. You might have:

- "Early Material" — your first batch of songs
- "Dark Acoustic" — songs with a specific mood
- "Upbeat Pop" — a different stylistic direction

Each album can be profiled independently, giving you fine-grained control over which stylistic elements the generator draws from.

### Profiles

A **profile** is a detailed stylistic fingerprint built from analysing the lyrics in an album. It captures:

- **Themes & subjects** — what you write about
- **Tone & mood** — emotional character of your writing
- **Structural patterns** — how you organise verses, choruses, bridges
- **Rhyme schemes** — your preferred rhyming patterns (ABAB, AABB, free, etc.)
- **Vocabulary characteristics** — word choice complexity, contraction usage, type-token ratio
- **Meter & rhythm** — syllable patterns, line lengths
- **Narrative techniques** — perspective, storytelling devices
- **Imagery patterns** — recurring metaphors and sensory language
- **Signature devices** — unique stylistic fingerprints specific to your writing

### Generations

A **generation** is an original song created by the AI using a profile as its guide. Each generation includes:

- **Title** — AI-generated, editable
- **Subject** — the topic/theme, editable
- **Lyrics** — the actual song text, fully editable
- **Metadata** — BPM, key, duration, caption (style tags for audio generation)
- **Audio link** — direct integration with HOT-Step's audio engine

---

## Workflows

### 1. The Creative Workflow (Primary)

This is the recommended way to use Lyric Studio — building from your own material.

#### Step 1: Create Your Artist

Click **Add Artist** from the main grid. Enter your artist or project name and an optional image URL. This creates your workspace.

#### Step 2: Add Your Lyrics

Navigate into your artist, then create an album. You have two ways to add songs:

**Manual Entry** (recommended):
- Click **Add Song** within an album
- Enter a title and paste or type your lyrics
- Use section headers like `[Verse]`, `[Chorus]`, `[Bridge]` for best profiling results
- The profiler uses these structural markers to understand your song architecture

**Bulk Import**:
- Add multiple songs to build a representative corpus
- More source material = more accurate profiles
- Aim for 5–15 songs for a solid profile; even 3–4 can work

#### Step 3: Build a Profile

Switch to the **Profiles** tab and click **Build New Profile**. The system will:

1. **Analyse structure** — identify sections, count verse/chorus lines, map rhyme schemes
2. **Measure meter** — calculate syllable averages, standard deviation, words per line
3. **Assess vocabulary** — type-token ratio, unique word count, contraction frequency
4. **Detect rhyme quality** — perfect rhymes, slant rhymes, assonance patterns
5. **Deep analysis** (LLM-assisted) — themes, tone, imagery, narrative techniques, emotional arc

The profile streams in real-time so you can watch the analysis unfold.

#### Step 4: Generate New Lyrics

Switch to the **Written Songs** tab and click **Generate Lyrics**. The generator:

1. Creates a **structural blueprint** based on your profile's patterns
2. Plans **metadata** — title, subject, BPM, key, duration, style caption
3. Writes lyrics that match your voice — themes, vocabulary, rhyme patterns, structure
4. Runs **post-processing** to enforce structural rules (mandatory chorus, correct line counts)
5. Passes through the **AI-Slop Detector** to filter generic AI language

You can generate multiple songs at once using the count selector (1–10).

#### Step 5: Refine & Edit

Every field on a generated song is inline-editable:
- **Title, subject, BPM, key, duration** — click and edit directly
- **Caption** (style tags) — customise for audio generation
- **Lyrics** — full text editing in a monospace editor

Changes save automatically on blur.

#### Step 6: Generate Audio

Click **Generate Audio** or **Audio** on any written song to send it directly to HOT-Step's audio engine. The system:

1. Packages lyrics, caption, BPM, key, and duration
2. Applies any **album preset** (adapter, matchering reference)
3. Merges your current Create Panel settings (inference steps, scheduler, etc.)
4. Submits the job and tracks it in the audio generation queue
5. Links the resulting audio back to the lyrics for playback

#### Step 7: Listen & Download

Switch to the **Recordings** tab to see all generated audio grouped by their source lyrics. From here you can:

- **Play** any recording directly in the global player
- **Download** in your preferred format (WAV, MP3, FLAC, OGG/Opus) with configurable bitrate
- Download **mastered, unmastered, or both** versions
- Files are automatically named with the artist prefix (e.g. "Artist Name - Song Title")

---

### 2. Curated Profile Workflow

For artists with multiple albums, you can build a **curated profile** — cherry-picking specific songs from across different albums to create a focused stylistic fingerprint.

1. Click the **Curated Profile** button from the artist view
2. Expand each album to see its songs
3. **Select/deselect individual songs** or use **Select All** per album
4. The footer shows your total selection count across albums
5. Click **Build Profile** to analyse only the selected songs

This is powerful for:
- Focusing on your strongest material
- Blending styles from different eras
- Excluding outlier songs that don't represent your core voice

---

### 3. Reference Artist Workflow (Secondary)

Lyric Studio also supports fetching lyrics from **Genius** for reference and study purposes. This can be useful for:

- Understanding how a particular style or genre approaches structure
- Studying rhyme schemes and vocabulary in a specific tradition
- Building reference profiles to compare against your own writing

To use this, click **Fetch from Genius** within an album view. You can search for artists and fetch their catalogue. However, the primary intent of Lyric Studio is to amplify *your* creative voice, not to replicate existing artists.

> **Note**: Genius-fetched lyrics are cleaned and normalised automatically — annotations, contributor text, and formatting artifacts are stripped to leave clean, structured lyrics suitable for profiling.

---

### 4. Bulk Operations

The **Bulk Operations** panel (accessible from the sidebar) enables batch processing across your entire library:

#### Build Profiles
- Select multiple unprofiled albums
- Queue them all for sequential profile building
- The queue processes each album in order with real-time progress

#### Generate Lyrics
- Select multiple profiles
- Set the generation count per profile (1–20)
- Queue all generations — they execute sequentially

#### Assign Presets
- View preset status across all albums (complete / partial / missing)
- **Bulk-assign adapters** — set a LoRA/LoKR adapter path and scale for multiple albums at once
- **Bulk-assign matchering references** — set a reference audio file for mastering
- Fine-tune adapter group scales (self-attention, cross-attention, MLP, conditioning embedder)
- Smart selection helpers: "Select Missing", "Select Incomplete"

---

## Audio Generation Queue

When you generate audio from Lyric Studio, jobs enter a persistent queue that:

- **Survives page reloads and HMR** — queue state is persisted to localStorage
- **Artist-batched execution** — reorders pending jobs to minimise adapter switches (if two songs share the same adapter, they run back-to-back)
- **Deferred adapter loading** — adapters are loaded only when an item actually starts running, not at queue time
- **Progress tracking** — real-time status updates (loading adapter → generating → succeeded/failed)
- **Automatic linking** — audio results are linked back to Lireek generations for the Recordings tab

---

## Album Presets

Each album can have a **preset** that configures audio generation defaults:

### Adapter (LoRA/LoKR)
- **Adapter path** — path to a `.safetensors` file or adapter folder
- **Adapter scale** — overall strength (0.0–4.0)
- **Group scales** — fine-tuned control over adapter influence:
  - **Self-Attention** — temporal coherence (rhythmic patterns, melodic phrases)
  - **Cross-Attention** — how strongly the text prompt shapes output vs. the adapter
  - **MLP** — stored timbre, tonal texture, sonic character
  - **Conditioning Embedder** — how the adapter reshapes prompt interpretation

The adapter type (LoRA vs LoKR) is auto-detected and displayed as a badge.

### Matchering Reference
- **Reference audio path** — a `.wav`, `.mp3`, or `.flac` file
- Used for EQ and loudness matching during automated mastering

Both can be browsed for via the integrated file browser.

---

## System Prompts (Advanced)

The **Prompt Editor** gives full control over the system prompts used throughout the pipeline:

| Prompt | Controls |
|--------|----------|
| **Generation** | How the AI writes new lyrics from a profile |
| **Refinement** | How the AI revises/improves existing generations |
| **Metadata Planning** | How the AI determines title, subject, BPM, key, duration, caption |
| **Profile: Themes & Subjects** | First profiling pass — theme extraction |
| **Profile: Tone & Structure** | Second pass — tone, vocabulary, structural patterns |
| **Profile: Imagery & Summary** | Third pass — imagery, narrative techniques, emotional arc |
| **Profile: Song Subjects** | Fourth pass — per-song subject identification |

Each prompt can be:
- **Customised** — edit and save your own version
- **Reset** — restore to the built-in default
- Customised prompts are marked with a `custom` badge

This is the power-user layer — most users won't need to touch these, but they're fully exposed for those who want to fine-tune the AI's behaviour.

---

## Quality Assurance: The AI-Slop Detector

One of Lyric Studio's most distinctive features is its **6-layer defence system** against generic AI output:

### Layer 1: Blacklist Detection
A curated list of known AI clichés and overused phrases. Any generation containing these is flagged. Examples include hollow metaphors, generic emotional language, and tired imagery that LLMs tend to default to.

### Layer 2: Structure Heuristics
Checks for structural patterns common in low-quality AI output:
- Excessive use of em-dashes mid-line
- Parenthetical asides that break natural flow
- Overly theatrical stage directions

### Layer 3: Repetition Analysis
Detects unnatural repetition patterns:
- Repeated sentence starters across verses
- Identical phrasing patterns between sections
- Over-reliance on the same transition words

### Layer 4: Statistical Anomaly Detection
Measures statistical properties of the text against expected ranges:
- Vocabulary diversity (type-token ratio)
- Line length consistency
- Syllable distribution patterns

### Layer 5: Cliché Density
Calculates the ratio of flagged phrases to total content. A high density of stock phrases — even if no single one is on the blacklist — triggers a warning.

### Layer 6: Contextual Coherence
Evaluates whether the generated text maintains thematic consistency with the source profile's themes and subjects.

When the detector flags a generation, it can trigger automatic re-generation or provide warnings for manual review.

---

## Export System

Generated lyrics are automatically exported as structured files:

```
{export_dir}/{Artist}/Based on {Album}/
    {title}_{id}.json     — machine-readable, ACE-Step compatible
    {title}_{id}.txt      — human-readable with metadata header
```

### JSON Format
```json
{
  "title": "Song Title",
  "caption": "style tags for audio generation",
  "lyrics": "full lyrics text",
  "bpm": 120,
  "keyscale": "C minor",
  "duration": 180,
  "metadata": {
    "artist": "Your Artist Name",
    "album": "Album Name",
    "subject": "Theme of the song",
    "provider": "gemini",
    "model": "gemini-2.5-pro",
    "lireek_id": 42,
    "created_at": "2026-04-04T12:00:00"
  }
}
```

### TXT Format
```
Title: Song Title
Artist style: Your Artist Name (Based on Album Name)
BPM: 120 | Key: C minor
Subject: Theme of the song

Caption:
style tags for audio generation

---

[Verse 1]
Your generated lyrics here...

[Chorus]
...
```

---

## Profiling Engine: Technical Details

The profiler combines **rule-based heuristics** with **LLM-assisted deep analysis** across multiple passes:

### Rule-Based Analysis (Instant)
Computed locally without any LLM calls:

| Metric | What It Measures |
|--------|-----------------|
| **Section Counts** | Average verse lines, average chorus lines, section types used |
| **Rhyme Schemes** | Pattern detection (ABAB, AABB, ABCB, free) per section |
| **Rhyme Quality** | Perfect rhyme %, slant rhyme %, assonance % |
| **Meter Stats** | Avg syllables/line, syllable std deviation, avg words/line |
| **Vocabulary Stats** | Type-token ratio, total/unique word counts, contraction % |
| **Repetition Stats** | Chorus repetition %, repetition pattern classification |

### LLM Deep Analysis (4 Passes)
Each pass uses a dedicated system prompt and analyses the full corpus:

1. **Themes & Subjects** — extracts themes, common subjects, subject categories
2. **Tone & Structure** — analyses tone/mood, vocabulary notes, structural patterns, perspective
3. **Imagery & Summary** — identifies imagery patterns, signature devices, emotional arc, narrative techniques; produces a raw summary
4. **Song Subjects** — per-song subject mapping for the full catalogue

The results are merged into a single rich profile object with all metrics available for the generator.

---

## Generation Pipeline: Technical Details

### Blueprint Construction
Before generating, the system creates a structural blueprint based on the profile:

- Selects section types from the profile's observed patterns
- Determines line counts per section using profile averages (with slight randomisation)
- Ensures mandatory sections: at least one `[Chorus]` is always present
- Normalises section labels to valid types: Verse, Chorus, Bridge, Pre-Chorus, Outro, Intro

### Post-Processing Pipeline
After the LLM generates raw lyrics, a strict pipeline enforces quality:

1. **Section label validation** — unknown labels like `[X]` are converted to valid types
2. **Chorus enforcement** — if no chorus exists, the system identifies the most chorus-like section and relabels it
3. **Line count enforcement** — verses and choruses are trimmed or padded to match the profile's averages
4. **Structure validation** — ensures the song has a coherent beginning-to-end flow

### Metadata Planning
A separate LLM call plans the generation metadata:
- **Title** — creative, fitting the profile's style
- **Subject** — specific topic for this song
- **BPM** — appropriate tempo
- **Key** — musical key
- **Duration** — target length in seconds
- **Caption** — style tags for audio generation (genre, mood, instrumentation)

---

## LLM Provider Support

Lyric Studio supports multiple LLM providers for profiling and generation, configurable independently:

- **Profiling model** — used for style analysis (heavier, more analytical)
- **Generation model** — used for writing lyrics (creative, expressive)
- **Refinement model** — used for iterative improvement

Provider selection is managed through the sidebar's **Provider Selector** and persisted to localStorage.

---

## UI Architecture

### Main Navigation
The Lyric Studio V2 interface is organised as:

```
Artist Grid → Artist Detail → Album Detail
                                 ├── Source Lyrics (your input songs)
                                 ├── Profiles (built from source lyrics)
                                 ├── Written Songs (AI-generated lyrics)
                                 └── Recordings (generated audio)
```

### Sidebar
A persistent sidebar provides:
- **Artist list** — quick navigation between artists
- **Bulk Operations** — batch profile building, generation, preset assignment
- **Prompt Editor** — customise system prompts
- **Provider Selector** — choose LLM models for each pipeline stage
- **Global Scale Overrides** — override adapter scales across all generations

### Real-Time Streaming
Both profile building and lyric generation stream their output in real-time, so you can watch the AI work. The streaming panel shows:
- Current phase (e.g. "Analysing themes…", "Writing verse 2…")
- Raw LLM output as it arrives
- Completion status

---

## Key Files Reference

### Backend (Python — FastAPI)

| File | Purpose |
|------|---------|
| `acestep/api/lireek/profiler_service.py` | Lyrics analysis engine (rule-based + LLM) |
| `acestep/api/lireek/generation_service.py` | Lyrics generation and post-processing |
| `acestep/api/lireek/slop_detector.py` | 6-layer AI quality defence |
| `acestep/api/lireek/genius_service.py` | Genius lyrics fetching and cleaning |
| `acestep/api/lireek/export_service.py` | JSON/TXT export pipeline |
| `acestep/api/lireek/lireek_db.py` | SQLite database operations |
| `acestep/api/lireek/lireek_routes.py` | FastAPI route definitions |

### Frontend (TypeScript — React)

| File | Purpose |
|------|---------|
| `ace-step-ui/components/lyric-studio/v2/LyricStudioV2.tsx` | Main entry point, navigation, state management |
| `ace-step-ui/components/lyric-studio/v2/ProfilesTab.tsx` | Profile viewing and building |
| `ace-step-ui/components/lyric-studio/v2/WrittenSongsTab.tsx` | Generated lyrics management |
| `ace-step-ui/components/lyric-studio/v2/RecordingsTab.tsx` | Audio playback and download |
| `ace-step-ui/components/lyric-studio/v2/SourceLyricsTab.tsx` | Source lyrics viewing and editing |
| `ace-step-ui/components/lyric-studio/v2/AddArtistModal.tsx` | Artist creation |
| `ace-step-ui/components/lyric-studio/v2/AddSongModal.tsx` | Manual song entry |
| `ace-step-ui/components/lyric-studio/v2/CuratedProfileModal.tsx` | Cross-album song selection for curated profiles |
| `ace-step-ui/components/lyric-studio/v2/PresetSettingsModal.tsx` | Per-album adapter and matchering config |
| `ace-step-ui/components/lyric-studio/QueuePanel.tsx` | Bulk operations (profile, generate, presets) |
| `ace-step-ui/components/lyric-studio/PromptEditor.tsx` | System prompt customisation |
| `ace-step-ui/services/lyricStudioApi.ts` | Typed API client for all endpoints |
| `ace-step-ui/stores/streamingStore.ts` | SSE streaming state for profiles/generations |
| `ace-step-ui/stores/audioGenQueueStore.ts` | Persistent audio generation queue |

---

## Summary of Capabilities

| Capability | Description |
|-----------|-------------|
| **Manual lyrics input** | Add your own songs with structured section headers |
| **Stylistic profiling** | AI-powered analysis of writing style across multiple dimensions |
| **Curated profiles** | Cherry-pick songs across albums for focused analysis |
| **Original lyric generation** | AI writes new songs matching your established style |
| **AI-Slop Detection** | 6-layer defence against generic, clichéd AI output |
| **Inline editing** | Every field on generated songs is directly editable |
| **One-click audio** | Send lyrics to HOT-Step's audio engine instantly |
| **Album presets** | Per-album adapter and matchering reference configuration |
| **Bulk operations** | Batch profile building, generation, and preset assignment |
| **Persistent queue** | Audio generation queue survives reloads with progress tracking |
| **Artist-batched execution** | Minimises adapter switches for efficient bulk generation |
| **Multi-format download** | WAV, MP3, FLAC, OGG/Opus with configurable bitrate |
| **Structured export** | JSON + TXT export of all generated lyrics |
| **Custom system prompts** | Full control over AI behaviour at every pipeline stage |
| **Real-time streaming** | Watch profile analysis and lyric generation unfold live |
| **Genius integration** | Fetch reference lyrics for study and comparison |
| **Multi-provider LLM support** | Choose different models for profiling, generation, and refinement |
