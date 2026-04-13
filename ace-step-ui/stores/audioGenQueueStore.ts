/**
 * audioGenQueueStore.ts — Module-level singleton for sequential audio generation.
 *
 * Manages a queue of audio generation jobs with:
 * - Artist-batched execution (reorders pending items to minimize adapter switches)
 * - Deferred adapter/matchering loading (only when item starts running)
 * - Progress polling and status tracking
 * - **localStorage persistence** — queue survives page reloads / HMR
 * - **Resume on reload** — in-flight jobs resume polling automatically
 *
 * Components subscribe via `useAudioGenQueue()` which uses `useSyncExternalStore`.
 */

import { useSyncExternalStore, useEffect } from 'react';
import { lireekApi, Generation, AlbumPreset } from '../services/lyricStudioApi';
import { generateApi, songsApi, GenerationJob } from '../services/api';

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
  /** Resolved audio URL (set on success for inline playback) */
  audioUrl?: string;
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

// ── Persistence ──────────────────────────────────────────────────────────────

const STORAGE_KEY = 'lireek-audio-gen-queue';

function _persist(): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      items: _state.items,
      completionCounter: _state.completionCounter,
    }));
  } catch { /* quota exceeded etc */ }
}

function _restore(): AudioGenQueueState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { items: [], completionCounter: 0 };
    const parsed = JSON.parse(raw);
    const items: AudioQueueItem[] = (parsed.items || []).map((item: AudioQueueItem) => {
      // Items that were mid-load or submitted-but-not-yet-tracked when the page died → reset to pending.
      // 'loading-adapter': adapter load was interrupted
      // 'generating' without jobId: startGeneration hadn't returned yet, so the backend
      //   never received the request (or we lost the reference). Safe to re-submit.
      if (item.status === 'loading-adapter' ||
          (item.status === 'generating' && !item.jobId)) {
        return { ...item, status: 'pending' as AudioQueueStatus, stage: undefined, progress: undefined };
      }
      return item;
    });
    return {
      items,
      completionCounter: parsed.completionCounter || 0,
    };
  } catch {
    return { items: [], completionCounter: 0 };
  }
}

// ── Module-level singleton ───────────────────────────────────────────────────

let _state: AudioGenQueueState = _restore();

const _listeners = new Set<() => void>();

function _emit() {
  _state = { ..._state, items: [..._state.items] };
  _persist();
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

// ── Resume tracking (prevent double-polling after HMR) ──────────────────

const _resumedJobIds = new Set<string>();
let _resumeCalled = false;

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

/**
 * Resume the queue after a page reload.
 * - In-flight jobs (status 'generating' with a jobId) → resume polling
 * - Pending jobs → restart the queue runner AFTER in-flight jobs finish
 *
 * IMPORTANT: We must await ALL in-flight polling before starting the queue
 * runner for pending items. Otherwise the queue runner may load a different
 * adapter while an in-flight job is still using the GPU → device mismatch.
 *
 * Called by the useAudioGenQueue hook on mount.
 */
export function resumeQueue(token: string): void {
  if (_resumeCalled) return;
  _resumeCalled = true;

  // Defensive: reset any 'generating' items that have no jobId back to pending.
  // These were submitted-but-not-tracked and can never be resumed by polling.
  let didFix = false;
  for (const item of _state.items) {
    if (item.status === 'generating' && !item.jobId) {
      item.status = 'pending';
      item.stage = undefined;
      item.progress = undefined;
      didFix = true;
    }
  }
  if (didFix) _emit();

  const hasPending = _state.items.some(i => i.status === 'pending');
  const inFlight = _state.items.filter(i => i.status === 'generating' && i.jobId);

  // Resume polling for in-flight jobs — collect promises so we can await them
  const resumePromises: Promise<void>[] = [];
  for (const item of inFlight) {
    if (_resumedJobIds.has(item.jobId!)) continue;
    _resumedJobIds.add(item.jobId!);
    resumePromises.push(_resumePolling(item, token));
  }

  // Only start the queue runner for pending items AFTER all in-flight jobs finish
  if (hasPending) {
    if (resumePromises.length > 0) {
      // Wait for in-flight jobs to complete, then start queue runner
      Promise.all(resumePromises).then(() => _processQueue(token));
    } else {
      _processQueue(token);
    }
  }
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

  // 4) Reference Track — dual purpose: timbre conditioning + matchering
  if (preset?.reference_track_path) {
    // a) ACE-Step timbre conditioning (guides DiT toward target acoustics)
    params.referenceAudioUrl = preset.reference_track_path;
    params.audioCoverStrength = preset.audio_cover_strength ?? 0.5;
    // b) Post-processing matchering (polishes EQ/loudness)
    params.autoMaster = true;
    params.masteringParams = { mode: 'matchering', reference_file: preset.reference_track_path };
  }

  // 5) Mark source & artist for download naming
  params.source = 'lyric-studio';
  params.artistName = item.artistName;
  item.status = 'generating';
  item.stage = 'Submitting to audio engine…';
  _emit();

  const res = await generateApi.startGeneration(params as any, token);
  const jobId = res.jobId || (res as any).job_id;
  item.jobId = jobId;
  _emit(); // persist jobId immediately so it survives reload

  // 6) Link audio to Lireek generation
  if (jobId) {
    await lireekApi.linkAudio(gen.id, jobId);
  }

  // 7) Poll until done
  await _pollUntilDone(item, token);
}

/** Poll a job until it succeeds or fails. Shared between fresh submissions and resume. */
async function _pollUntilDone(item: AudioQueueItem, token: string): Promise<void> {
  const jobId = item.jobId!;
  item.stage = 'Generating audio…';
  const startTime = item.elapsed ? Date.now() - item.elapsed * 1000 : Date.now();
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

      if (status.status === 'succeeded') {
        // Resolve and persist audio URL + cover art to Lireek DB
        const audioUrl = status.result?.audioUrls?.[0];
        if (audioUrl) {
          item.audioUrl = audioUrl;
          _emit(); // persist audioUrl so it survives reload
        }
        if (audioUrl && jobId) {
          let coverUrl: string | undefined;
          try {
            const { songs: dbSongs } = await songsApi.getSongsByUrls([audioUrl], token);
            const db: any = dbSongs[0];
            coverUrl = db?.coverUrl || db?.cover_url || undefined;
          } catch { /* non-fatal */ }
          try {
            await lireekApi.resolveAudioGeneration(jobId, audioUrl, coverUrl);
          } catch { /* non-fatal */ }
        }
        return;
      }
      if (status.status === 'failed') throw new Error(status.error || 'Generation failed');
    } catch (e) {
      // If getStatus fails with a network error, don't break the loop immediately
      if ((e as Error).message.includes('failed') && !(e as Error).message.includes('fetch')) {
        throw e;
      }
    }
  }
}

/** Resume polling for a job that was in-flight when the page reloaded. */
async function _resumePolling(item: AudioQueueItem, token: string): Promise<void> {
  try {
    await _pollUntilDone(item, token);
    item.status = 'succeeded';
    _state.completionCounter++;
  } catch (err) {
    item.status = 'failed';
    item.error = (err as Error).message;
  }
  _emit();
}

// ── React hook ───────────────────────────────────────────────────────────────

export function useAudioGenQueue(token?: string): AudioGenQueueState {
  // On mount: resume any persisted in-flight or pending jobs
  useEffect(() => {
    if (token && _state.items.length > 0) {
      resumeQueue(token);
    }
  }, [token]);

  return useSyncExternalStore(_subscribe, _getSnapshot, _getSnapshot);
}
