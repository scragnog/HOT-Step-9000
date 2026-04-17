# Cover Studio — Design Document

**Date:** 2026-04-17
**Status:** Approved

## Goal

A new top-level page that lets users create AI covers of existing songs by combining:
a source audio file + lyrics from Genius + a target artist's voice/adapter settings from Lyric Studio.

## Layout

Three-panel workspace:
- **Left Panel**: Source audio upload (drag-drop/browse), song details (artist/title from metadata), audio analysis (BPM/key from Essentia)
- **Center Panel**: Editable lyrics textarea (fetched from Genius, user-editable)
- **Right Panel**: Target artist selector (from Lyric Studio DB), cover settings (strength, noise, tempo, pitch, BPM/key readout), generate button, recent covers list

## Key Decisions

1. **Cover generations are isolated** — tagged `source: 'cover-studio'`, filtered out of Library and Create page, only visible in Cover Studio's own "Recent Covers" panel
2. **Lyrics are editable** — fetched from Genius into an editable textarea
3. **Three-panel workspace** — all steps visible at once, no wizard/stepper

## Component Architecture

| Component | Purpose |
|---|---|
| `CoverStudio.tsx` | Main page — orchestrates state, layout, generation |
| `SourceAudioPanel.tsx` | Left panel: drag-drop upload, metadata extraction, Essentia analysis display |
| `SongIdentifier.tsx` | Artist/title fields (auto-filled from metadata), Genius search button |
| `ArtistSelector.tsx` | Right panel top: grid/list of Lyric Studio artists with their adapter configs |
| `CoverSettingsPanel.tsx` | Right panel: wraps CoverRepaintSettings with cover-specific defaults |
| `RecentCovers.tsx` | Right panel bottom: list of past cover-studio generations with playback |

## Server-Side Changes

| Change | Location | Details |
|---|---|---|
| Audio metadata extraction | `server/src/routes/analyze.ts` | New `POST /api/analyze/metadata` using `music-metadata` npm |
| Single-song lyrics fetch | Python Lireek server | New `POST /api/lireek/search-song-lyrics` |
| Source audio upload | Reuses `referenceTrack.ts` upload route | Already handles all needed formats |

## Frontend Changes

| Change | Location | Details |
|---|---|---|
| Add `'cover-studio'` to View type | `types.ts` | New view union member |
| Add nav item | `Sidebar.tsx` | New NavItem with icon |
| Add route handling | `App.tsx` | Render CoverStudio when currentView === 'cover-studio' |
| Filter cover-studio songs | `App.tsx` / `SongList.tsx` | Exclude source: 'cover-studio' from Library |
| New component directory | `components/cover-studio/` | 6 new components |
| Install music-metadata | `package.json` | Pure JS metadata reader |

## Reused Components

- `CoverRepaintSettings.tsx` — cover controls UI
- `EditableSlider` — sliders
- `FileBrowserModal` — file browsing
- Essentia `/api/analyze` route — BPM/key detection
- `generateApi.startGeneration()` — generation pipeline
- `audioGenQueueStore` — job queue + progress tracking
- Adapter loading flow from `useAudioGeneration.ts`
