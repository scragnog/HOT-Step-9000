/**
 * audioGenQueueStore.ts — Module-level singleton for sequential audio generation.
 *
 * Manages a queue of audio generation jobs with:
 * - Artist-batched execution (reorders pending items to minimize adapter switches)
 * - Deferred adapter/matchering loading (only when item starts running)
 * - Progress polling and status tracking
 *
 * Components subscribe via `useAudioGenQueue()` which uses `useSyncExternalStore`.
 */

import { useSyncExternalStore } from 'react';
import { lireekApi, Generation, AlbumPreset } from '../services/lyricStudioApi';
import { generateApi, GenerationJob } from '../services/api';

// ── Types ────────────────────────────────────────────────────────────────────

export type AudioQueueStatus = 'pending' | 'loading-adapter' | 'generating' | 'succeeded' | 'failed';

export interface AudioQueueItem {
  id: string;
  /** Lireek generation record */
  generation: Generation;
  /** Artist context for adapter batching */
  artistId: number;
  artistName: string;
  /** Preset resolved at queue time so it survives navigation */
  preset: AlbumPreset | null;
  /** Profile ID (for linking) */
  profileId: number;
  lyricsSetId: number;
  /** Runtime state */
  status: AudioQueueStatus;
  /** HOT-Step job ID once submitted */
  jobId?: string;
  progress?: number;   // 0–100
  stage?: string;
  elapsed?: number;    // seconds
  error?: string;
}

export interface AudioGenQueueState {
  items: AudioQueueItem[];
  /** Incremented whenever a gen completes — used as refreshKey by consumers */
  completionCounter: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function readPersisted(key: string): any {
  try {
    const raw = localStorage.getItem(key);
    return raw !== null ? JSON.parse(raw) : undefined;
  } catch { return undefined; }
}

function mergeCreatePanelSettings(params: Record<string, any>): void {
  const map: [string, string][] = [
    ['ace-inferenceSteps', 'inferenceSteps'],
    ['ace-inferMethod', 'inferMethod'],
    ['ace-scheduler', 'scheduler'],
    ['ace-guidanceScale', 'guidanceScale'],
    ['ace-shift', 'shift'],
    ['ace-guidanceMode', 'guidanceMode'],
    ['ace-audioFormat', 'audioFormat'],
    ['ace-randomSeed', 'randomSeed'],
    ['ace-enableNormalization', 'enableNormalization'],
    ['ace-normalizationDb', 'normalizationDb'],
    ['ace-vocalLanguage', 'vocalLanguage'],
    ['ace-vocalGender', 'vocalGender'],
    ['ace-lmTemperature', 'lmTemperature'],
    ['ace-lmCfgScale', 'lmCfgScale'],
    ['ace-lmTopK', 'lmTopK'],
    ['ace-lmTopP', 'lmTopP'],
    ['ace-lmModel', 'lmModel'],
    ['ace-useCotMetas', 'useCotMetas'],
    ['ace-useCotCaption', 'useCotCaption'],
    ['ace-useCotLanguage', 'useCotLanguage'],
    ['ace-vocoderModel', 'vocoderModel'],
    ['ace-thinking', 'thinking'],
  ];
  for (const [storageKey, paramKey] of map) {
    const val = readPersisted(storageKey);
    if (val !== undefined && val !== null && val !== '') {
      params[paramKey] = val;
    }
  }
  params.getLrc = true;
  params.generateCoverArt = localStorage.getItem('generate_cover_art') === 'true';
}

function getGlobalScaleOverride() {
  try {
    const enabled = JSON.parse(localStorage.getItem('ace-globalScaleOverride') || 'false');
    const overallScale = JSON.parse(localStorage.getItem('ace-globalOverallScale') || '1.0');
    const groupScales = JSON.parse(localStorage.getItem('ace-globalGroupScales') || 'null') || { self_attn: 1.0, cross_attn: 1.0, mlp: 1.0 };
    return { enabled: !!enabled, overallScale, groupScales };
  } catch {
    return { enabled: false, overallScale: 1.0, groupScales: { self_attn: 1.0, cross_attn: 1.0, mlp: 1.0 } };
  }
}

function applyTriggerWord(params: Record<string, any>, adapterPath: string): void {
  const useFilename = localStorage.getItem('ace-globalTriggerUseFilename') === 'true';
  const placement = (localStorage.getItem('ace-globalTriggerPlacement') as 'prepend' | 'append' | 'replace') || 'prepend';
  if (!useFilename) return;
  const fileName = adapterPath.replace(/\\/g, '/').split('/').pop() || '';
  const triggerWord = fileName.replace(/\.safetensors$/i, '');
  if (!triggerWord) return;
  const current = ((params.style as string) || '').trim();
  if (current.toLowerCase().includes(triggerWord.toLowerCase())) return;
  if (placement === 'replace') { params.style = triggerWord; }
  else if (placement === 'append') { params.style = current ? `${current}, ${triggerWord}` : triggerWord; }
  else { params.style = current ? `${triggerWord}, ${current}` : triggerWord; }
}

// ── Module-level singleton ───────────────────────────────────────────────────

let _state: AudioGenQueueState = {
  items: [],
  completionCounter: 0,
};

const _listeners = new Set<() => void>();

function _emit() {
  _state = { ..._state, items: [..._state.items] };
  _listeners.forEach(fn => fn());
}

function _getSnapshot(): AudioGenQueueState { return _state; }
function _subscribe(listener: () => void): () => void {
  _listeners.add(listener);
  return () => _listeners.delete(listener);
}

let _nextId = 0;
function _genId(): string { return `aq-${Date.now()}-${_nextId++}`; }

// ── Track which adapter is currently loaded (to skip reloads) ────────────

let _currentAdapterPath: string | null = null;

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * Add a generation to the audio queue.
 * Resolves the preset at queue time so we have adapter/matchering info.
 */
export async function enqueueAudioGen(
  gen: Generation,
  opts: {
    artistId: number;
    artistName: string;
    profileId: number;
    lyricsSetId: number;
  },
  token: string,
): Promise<void> {
  // Resolve preset now so it's captured
  let preset: AlbumPreset | null = null;
  try {
    const res = await lireekApi.getPreset(opts.lyricsSetId);
    preset = res.preset;
  } catch { /* no preset configured */ }

  const item: AudioQueueItem = {
    id: _genId(),
    generation: gen,
    artistId: opts.artistId,
    artistName: opts.artistName,
    preset,
    profileId: opts.profileId,
    lyricsSetId: opts.lyricsSetId,
    status: 'pending',
  };

  _state.items.push(item);
  _emit();
  _processQueue(token);
}

/** Remove a pending item from the queue. */
export function removeFromAudioQueue(id: string): void {
  _state.items = _state.items.filter(i => i.id !== id || i.status !== 'pending');
  _emit();
}

/** Clear all completed/failed items. */
export function clearFinishedFromAudioQueue(): void {
  _state.items = _state.items.filter(i => i.status !== 'succeeded' && i.status !== 'failed');
  _emit();
}

// ── Queue runner ─────────────────────────────────────────────────────────────

let _running = false;

async function _processQueue(token: string): Promise<void> {
  if (_running) return;
  _running = true;

  while (true) {
    // Find next item — artist-batched: prefer items matching current adapter
    const pending = _state.items.filter(i => i.status === 'pending');
    if (pending.length === 0) break;

    let next: AudioQueueItem;
    if (_currentAdapterPath) {
      // Find an item whose preset matches the currently loaded adapter
      const sameAdapter = pending.find(i =>
        i.preset?.adapter_path === _currentAdapterPath
      );
      next = sameAdapter || pending[0];
    } else {
      // Group by artist to batch: pick the artist with the most pending items
      const artistCounts = new Map<number, number>();
      for (const item of pending) {
        artistCounts.set(item.artistId, (artistCounts.get(item.artistId) || 0) + 1);
      }
      let bestArtist = pending[0].artistId;
      let bestCount = 0;
      for (const [aid, count] of artistCounts) {
        if (count > bestCount) { bestCount = count; bestArtist = aid; }
      }
      next = pending.find(i => i.artistId === bestArtist) || pending[0];
    }

    try {
      await _executeItem(next, token);
      next.status = 'succeeded';
      _state.completionCounter++;
    } catch (err) {
      next.status = 'failed';
      next.error = (err as Error).message;
    }
    _emit();
  }

  _running = false;
}

async function _executeItem(item: AudioQueueItem, token: string): Promise<void> {
  const gen = item.generation;
  const preset = item.preset;

  // 1) Build base params
  const params: Record<string, any> = {
    customMode: true,
    lyrics: gen.lyrics || '',
    style: gen.caption || '',
    title: gen.title || '',
    instrumental: false,
    duration: gen.duration || 180,
  };
  if (gen.bpm) params.bpm = gen.bpm;
  if (gen.subject) params.coverArtSubject = gen.subject;
  if (gen.key) params.keyScale = gen.key;

  // 2) Merge persisted CreatePanel settings
  mergeCreatePanelSettings(params);

  // 3) Load adapter (deferred — only now, when this item actually runs)
  if (preset?.adapter_path) {
    item.status = 'loading-adapter';
    item.stage = `Loading adapter for ${item.artistName}…`;
    _emit();

    const scaleOverride = getGlobalScaleOverride();
    const effectiveScale = scaleOverride.enabled ? scaleOverride.overallScale : (preset.adapter_scale ?? 1.0);
    const effectiveGroupScales = scaleOverride.enabled ? scaleOverride.groupScales : preset.adapter_group_scales;

    try {
      const loraStatus = await generateApi.getLoraStatus(token);
      const existingSlot = loraStatus?.advanced?.slots?.find(
        (s: any) => s.path === preset.adapter_path
      );

      if (existingSlot) {
        params.loraLoaded = true;
        params.loraPath = preset.adapter_path;
        params.loraScale = effectiveScale;
        if (effectiveGroupScales) {
          await generateApi.setSlotGroupScales({
            slot: existingSlot.slot,
            ...effectiveGroupScales,
          }, token);
        }
      } else {
        // Unload existing adapter and load the new one
        if (loraStatus?.advanced?.slots && loraStatus.advanced.slots.length > 0) {
          await generateApi.unloadLora(token);
        }
        await generateApi.loadLora({
          lora_path: preset.adapter_path,
          slot: 0,
          scale: effectiveScale,
          ...(effectiveGroupScales ? { group_scales: effectiveGroupScales } : {}),
        }, token);
        params.loraLoaded = true;
        params.loraPath = preset.adapter_path;
        params.loraScale = effectiveScale;
      }
      _currentAdapterPath = preset.adapter_path;
    } catch (loadErr) {
      console.warn('[AudioGenQueue] Adapter load failed:', loadErr);
      // Continue without adapter
    }

    // Trigger word
    applyTriggerWord(params, preset.adapter_path);
  }

  // 4) Matchering
  if (preset?.matchering_reference_path) {
    params.autoMaster = true;
    params.masteringParams = { mode: 'matchering', reference_file: preset.matchering_reference_path };
  }

  // 5) Mark source & submit
  params.source = 'lyric-studio';
  item.status = 'generating';
  item.stage = 'Submitting to audio engine…';
  _emit();

  const res = await generateApi.startGeneration(params as any, token);
  const jobId = res.jobId || (res as any).job_id;
  item.jobId = jobId;

  // 6) Link audio to Lireek generation
  if (jobId) {
    await lireekApi.linkAudio(gen.id, jobId);
  }

  // 7) Poll until done
  item.stage = 'Generating audio…';
  const startTime = Date.now();
  _emit();

  while (true) {
    await new Promise(r => setTimeout(r, 2500));
    try {
      const status = await generateApi.getStatus(jobId, token);
      item.progress = status.progress !== undefined
        ? Math.min(100, Math.max(0, (status.progress > 1 ? status.progress / 100 : status.progress) * 100))
        : undefined;
      item.stage = status.stage || 'Generating…';
      item.elapsed = Math.round((Date.now() - startTime) / 1000);
      _emit();

      if (status.status === 'succeeded') return;
      if (status.status === 'failed') throw new Error(status.error || 'Generation failed');
    } catch (e) {
      // If getStatus fails with a network error, don't break the loop immediately
      if ((e as Error).message.includes('failed') && !(e as Error).message.includes('fetch')) {
        throw e;
      }
    }
  }
}

// ── React hook ───────────────────────────────────────────────────────────────

export function useAudioGenQueue(): AudioGenQueueState {
  return useSyncExternalStore(_subscribe, _getSnapshot, _getSnapshot);
}
