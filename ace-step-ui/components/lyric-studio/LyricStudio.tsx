import React, { useState, useEffect, useCallback } from 'react';
import { usePersistedState } from '../../hooks/usePersistedState';
import {
  Music, Users, Disc3, FileText, Sparkles, ChevronRight, ChevronDown,
  Plus, Trash2, Download, RefreshCw, Loader2, Search, AlertTriangle, X, Wand2, Play, Settings2, Save, ListOrdered, Code2, Pencil, FolderSearch,
} from 'lucide-react';
import { lireekApi, Artist, LyricsSet, Profile, Generation, SongLyric, AlbumPreset } from '../../services/lyricStudioApi';
import { generateApi } from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { TripleProviderSelector, ModelSelections, loadSelections, saveSelections } from './ProviderSelector';
import { StreamingPanel } from './StreamingPanel';
import {
  useStreamingStore,
  startStreamBuildProfile,
  startStreamGenerate,
  startStreamRefine,
  doSkipThinking,
} from '../../stores/streamingStore';
import { QueuePanel } from './QueuePanel';
import { PromptEditor } from './PromptEditor';
import { EditableSlider } from '../EditableSlider';
import { FileBrowserModal } from '../FileBrowserModal';

// ── Helpers ──────────────────────────────────────────────────────────────────

function parseSongs(songs: SongLyric[] | string): SongLyric[] {
  if (typeof songs === 'string') {
    try { return JSON.parse(songs); } catch { return []; }
  }
  return songs || [];
}

function timeAgo(dateStr: string): string {
  const d = new Date(dateStr);
  const now = new Date();
  const diff = Math.floor((now.getTime() - d.getTime()) / 1000);
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

import { Song } from '../../types';

// ── Main Component ──────────────────────────────────────────────────────────

export const LyricStudio: React.FC<{ onPlaySong?: (song: Song) => void }> = ({ onPlaySong }) => {
  const { token } = useAuth();
  // ── Data state ──────────────────────────────────────────────────────────
  const [artists, setArtists] = useState<Artist[]>([]);
  const [lyricsSets, setLyricsSets] = useState<LyricsSet[]>([]);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [generations, setGenerations] = useState<Generation[]>([]);

  // ── UI state ────────────────────────────────────────────────────────────
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sidebarWidth, setSidebarWidth] = usePersistedState('lireek-sidebarWidth', 320);
  const [expandedArtists, setExpandedArtists] = useState<Set<number>>(new Set());
  const [expandedAlbums, setExpandedAlbums] = useState<Set<number>>(new Set());
  const [expandedProfiles, setExpandedProfiles] = useState<Set<number>>(new Set());

  // Selection
  const [selectedItem, setSelectedItem] = useState<{
    type: 'artist' | 'album' | 'profile' | 'generation' | 'fetch';
    id: number;
  } | null>(null);

  // Fetch panel
  const [fetchArtist, setFetchArtist] = useState('');
  const [fetchAlbum, setFetchAlbum] = useState('');
  const [fetchMaxSongs, setFetchMaxSongs] = useState(50);
  const [fetching, setFetching] = useState(false);

  // LLM provider selections (per-role)
  const [modelSelections, setModelSelections] = useState<ModelSelections>(loadSelections);

  // Expanded song index for lyrics preview
  const [expandedSong, setExpandedSong] = useState<string | null>(null);

  // Action state
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  // ── Streaming state (from global store — survives navigation) ──────
  const stream = useStreamingStore();
  // Queue modal
  const [queueOpen, setQueueOpen] = useState(false);
  // Prompt editor modal
  const [promptEditorOpen, setPromptEditorOpen] = useState(false);

  // Detail cache — full data fetched on-demand when an item is selected
  const [detailCache, setDetailCache] = useState<Record<string, any>>({});
  const [detailLoading, setDetailLoading] = useState(false);

  // Preset file browser state
  const [presetBrowserOpen, setPresetBrowserOpen] = useState(false);
  const [presetBrowserTarget, setPresetBrowserTarget] = useState<'adapter' | 'matchering'>('adapter');
  const [detectedAdapterType, setDetectedAdapterType] = useState<'lokr' | 'lora' | 'unknown' | null>(null);
  const [presetGroupsExpanded, setPresetGroupsExpanded] = useState(false);
  const [presetLoading, setPresetLoading] = useState(false);

  // Audio generations linked to lyric generations
  const [audioGens, setAudioGens] = useState<Record<number, Array<{ job_id: string; created_at: string; status?: string; audioUrl?: string }>>>({});

  // Album presets
  const [presets, setPresets] = useState<Record<number, AlbumPreset | null>>({});
  const [presetForm, setPresetForm] = useState<{
    adapter_path: string;
    adapter_scale: number;
    self_attn: number;
    cross_attn: number;
    mlp: number;
    matchering_reference_path: string;
  }>({
    adapter_path: '', adapter_scale: 1.0,
    self_attn: 1.0, cross_attn: 1.0, mlp: 1.0,
    matchering_reference_path: '',
  });

  // ── Load all data ───────────────────────────────────────────────────────
  const loadAll = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      setDetailCache({}); // Invalidate detail cache on refresh
      const [a, ls, p, g] = await Promise.all([
        lireekApi.listArtists(),
        lireekApi.listLyricsSets(),
        lireekApi.listProfiles(),
        lireekApi.listAllGenerations(),
      ]);
      setArtists(a.artists);
      setLyricsSets(ls.lyrics_sets);
      setProfiles(p.profiles);
      setGenerations(g.generations);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadAll(); }, [loadAll]);

  // ── Toast helper ────────────────────────────────────────────────────────
  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  };

  // ── Tree toggle helpers ─────────────────────────────────────────────────
  const toggleArtist = (id: number) => {
    setExpandedArtists(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };
  const toggleAlbum = (id: number) => {
    setExpandedAlbums(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };
  const toggleProfile = (id: number) => {
    setExpandedProfiles(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  // ── Actions ─────────────────────────────────────────────────────────────
  const handleFetch = async () => {
    if (!fetchArtist.trim()) return;
    setFetching(true);
    try {
      const res = await lireekApi.fetchLyrics({
        artist: fetchArtist.trim(),
        album: fetchAlbum.trim() || undefined,
        max_songs: fetchMaxSongs,
      });
      showToast(`Fetched ${res.songs_fetched} songs for ${res.artist.name}`);
      setFetchArtist('');
      setFetchAlbum('');
      await loadAll();
      // Auto-expand the artist
      setExpandedArtists(prev => new Set(prev).add(res.artist.id));
    } catch (err) {
      showToast(`Fetch failed: ${(err as Error).message}`);
    } finally {
      setFetching(false);
    }
  };

  const handleBuildProfile = async (lyricsSetId: number) => {
    setActionLoading(`profile-${lyricsSetId}`);
    try {
      const { provider, model } = modelSelections.profiling;
      await startStreamBuildProfile(
        lyricsSetId,
        { provider, model: model || undefined },
        () => { showToast('Profile built successfully'); loadAll(); },
      );
    } catch (err) {
      showToast(`Profile build failed: ${(err as Error).message}`);
    } finally {
      setActionLoading(null);
    }
  };

  const handleGenerate = async (profileId: number) => {
    setActionLoading(`generate-${profileId}`);
    try {
      const { provider, model } = modelSelections.generation;
      await startStreamGenerate(
        profileId,
        { profile_id: profileId, provider, model: model || undefined },
        () => { showToast('Lyrics generated successfully'); loadAll(); },
      );
    } catch (err) {
      showToast(`Generation failed: ${(err as Error).message}`);
    } finally {
      setActionLoading(null);
    }
  };

  const handleRefine = async (generationId: number) => {
    setActionLoading(`refine-${generationId}`);
    try {
      const { provider, model } = modelSelections.refinement;
      await startStreamRefine(
        generationId,
        { provider, model: model || undefined },
        () => { showToast('Lyrics refined successfully'); loadAll(); },
      );
    } catch (err) {
      showToast(`Refinement failed: ${(err as Error).message}`);
    } finally {
      setActionLoading(null);
    }
  };

  const handleExport = async (generationId: number) => {
    setActionLoading(`export-${generationId}`);
    try {
      const res = await lireekApi.exportGeneration(generationId);
      showToast(`Exported to ${res.path}`);
    } catch (err) {
      showToast(`Export failed: ${(err as Error).message}`);
    } finally {
      setActionLoading(null);
    }
  };

  const handleDeleteArtist = async (id: number) => {
    if (!confirm('Delete this artist and all their data?')) return;
    try {
      await lireekApi.deleteArtist(id);
      showToast('Artist deleted');
      if (selectedItem?.type === 'artist' && selectedItem.id === id) setSelectedItem(null);
      await loadAll();
    } catch (err) {
      showToast(`Delete failed: ${(err as Error).message}`);
    }
  };

  const handleDeleteLyricsSet = async (id: number) => {
    if (!confirm('Delete this album/lyrics set?')) return;
    try {
      await lireekApi.deleteLyricsSet(id);
      showToast('Lyrics set deleted');
      if (selectedItem?.type === 'album' && selectedItem.id === id) setSelectedItem(null);
      await loadAll();
    } catch (err) {
      showToast(`Delete failed: ${(err as Error).message}`);
    }
  };

  const handleDeleteProfile = async (id: number) => {
    if (!confirm('Delete this profile and all generations from it?')) return;
    try {
      await lireekApi.deleteProfile(id);
      showToast('Profile deleted');
      if (selectedItem?.type === 'profile' && selectedItem.id === id) setSelectedItem(null);
      await loadAll();
    } catch (err) {
      showToast(`Delete failed: ${(err as Error).message}`);
    }
  };

  const handleDeleteGeneration = async (id: number) => {
    if (!confirm('Delete this generation?')) return;
    try {
      await lireekApi.deleteGeneration(id);
      showToast('Generation deleted');
      if (selectedItem?.type === 'generation' && selectedItem.id === id) setSelectedItem(null);
      await loadAll();
    } catch (err) {
      showToast(`Delete failed: ${(err as Error).message}`);
    }
  };

  const handleRemoveSong = async (lyricsSetId: number, songIndex: number, songTitle: string) => {
    if (!confirm(`Remove "${songTitle}" from this album?`)) return;
    try {
      await lireekApi.removeSong(lyricsSetId, songIndex);
      showToast(`Removed "${songTitle}"`);
      await loadAll();
    } catch (err) {
      showToast(`Remove failed: ${(err as Error).message}`);
    }
  };

  const handleSaveGeneration = async (genId: number, field: string, value: string | number) => {
    try {
      const res = await fetch(`/api/lireek/generations/${genId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [field]: value }),
      });
      if (!res.ok) throw new Error(await res.text());
      showToast(`Updated ${field}`);
      await loadAll();
    } catch (err) {
      showToast(`Update failed: ${(err as Error).message}`);
    }
  };

  // ── Album Preset Handlers ───────────────────────────────────────────────
  const loadPreset = async (lyricsSetId: number) => {
    setPresetLoading(true);
    try {
      const res = await lireekApi.getPreset(lyricsSetId);
      setPresets(prev => ({ ...prev, [lyricsSetId]: res.preset }));
      if (res.preset) {
        setPresetForm({
          adapter_path: res.preset.adapter_path || '',
          adapter_scale: res.preset.adapter_scale ?? 1.0,
          self_attn: res.preset.adapter_group_scales?.self_attn ?? 1.0,
          cross_attn: res.preset.adapter_group_scales?.cross_attn ?? 1.0,
          mlp: res.preset.adapter_group_scales?.mlp ?? 1.0,
          matchering_reference_path: res.preset.matchering_reference_path || '',
        });
      } else {
        setPresetForm({ adapter_path: '', adapter_scale: 1.0, self_attn: 1.0, cross_attn: 1.0, mlp: 1.0, matchering_reference_path: '' });
      }
    } catch (err) {
      showToast(`Failed to load preset: ${(err as Error).message}`);
    } finally {
      setPresetLoading(false);
    }
  };

  const savePreset = async (lyricsSetId: number) => {
    setActionLoading(`preset-${lyricsSetId}`);
    try {
      await lireekApi.upsertPreset(lyricsSetId, {
        adapter_path: presetForm.adapter_path || undefined,
        adapter_scale: presetForm.adapter_scale,
        adapter_group_scales: { self_attn: presetForm.self_attn, cross_attn: presetForm.cross_attn, mlp: presetForm.mlp },
        matchering_reference_path: presetForm.matchering_reference_path || undefined,
      });
      showToast('Preset saved');
      await loadPreset(lyricsSetId);
    } catch (err) {
      showToast(`Save failed: ${(err as Error).message}`);
    } finally {
      setActionLoading(null);
    }
  };

  // Detect adapter type when preset adapter path changes
  useEffect(() => {
    if (!presetForm.adapter_path.trim() || !token) {
      setDetectedAdapterType(null);
      return;
    }
    let cancelled = false;
    generateApi.detectAdapterType(presetForm.adapter_path.trim(), token)
      .then(result => { if (!cancelled) setDetectedAdapterType(result.type); })
      .catch(() => { if (!cancelled) setDetectedAdapterType('unknown'); });
    return () => { cancelled = true; };
  }, [presetForm.adapter_path, token]);

  // ── Fetch linked audio generations for a lyric generation ──────────────
  const fetchAudioGens = useCallback(async (genId: number) => {
    if (!token) return;
    try {
      const res = await lireekApi.getAudioGenerations(genId);
      const entries = res.audio_generations || [];
      // Fetch status for each
      const withStatus = await Promise.all(entries.map(async (ag: any) => {
        try {
          const status = await generateApi.getStatus(ag.job_id || ag.hotstep_job_id, token);
          return {
            job_id: ag.job_id || ag.hotstep_job_id,
            created_at: ag.created_at,
            status: status.status,
            audioUrl: status.result?.audioUrls?.[0] || undefined,
          };
        } catch {
          return { job_id: ag.job_id || ag.hotstep_job_id, created_at: ag.created_at, status: 'unknown' };
        }
      }));
      setAudioGens(prev => ({ ...prev, [genId]: withStatus }));
    } catch { /* ignore */ }
  }, [token]);

  // ── Generate Audio from Lyrics ──────────────────────────────────────────
  const handleGenerateAudio = async (gen: Generation) => {
    if (!token) { showToast('Not authenticated'); return; }
    setActionLoading(`audio-${gen.id}`);
    try {
      // Find the album preset for this generation's profile -> lyrics_set
      const profile = profiles.find(p => p.id === gen.profile_id);
      let preset: AlbumPreset | null = null;
      if (profile) {
        const res = await lireekApi.getPreset(profile.lyrics_set_id);
        preset = res.preset;
      }

      const params: any = {
        customMode: true,
        lyrics: gen.lyrics || '',
        style: gen.caption || '',
        title: gen.title || '',
        instrumental: false,
        duration: gen.duration || 180,
      };
      if (gen.bpm) params.bpm = gen.bpm;
      if (gen.key) params.keyScale = gen.key;

      // ── Merge persisted CreatePanel generation engine settings ──
      // These are stored by usePersistedState in CreatePanel with 'ace-' prefix
      const readPersisted = (key: string) => {
        try {
          const raw = localStorage.getItem(key);
          return raw !== null ? JSON.parse(raw) : undefined;
        } catch { return undefined; }
      };
      // Generation engine
      const steps = readPersisted('ace-inferenceSteps');
      if (steps !== undefined) params.inferenceSteps = steps;
      const method = readPersisted('ace-inferMethod');
      if (method) params.inferMethod = method;
      const sched = readPersisted('ace-scheduler');
      if (sched) params.scheduler = sched;
      const cfg = readPersisted('ace-guidanceScale');
      if (cfg !== undefined) params.guidanceScale = cfg;
      const shiftVal = readPersisted('ace-shift');
      if (shiftVal !== undefined) params.shift = shiftVal;
      const gMode = readPersisted('ace-guidanceMode');
      if (gMode) params.guidanceMode = gMode;
      const fmt = readPersisted('ace-audioFormat');
      if (fmt) params.audioFormat = fmt;
      const rndSeed = readPersisted('ace-randomSeed');
      if (rndSeed !== undefined) params.randomSeed = rndSeed;
      const normEn = readPersisted('ace-enableNormalization');
      if (normEn !== undefined) params.enableNormalization = normEn;
      const normDb = readPersisted('ace-normalizationDb');
      if (normDb !== undefined) params.normalizationDb = normDb;
      const vocL = readPersisted('ace-vocalLanguage');
      if (vocL) params.vocalLanguage = vocL;
      const vocG = readPersisted('ace-vocalGender');
      if (vocG) params.vocalGender = vocG;
      // LM settings
      const lmTemp = readPersisted('ace-lmTemperature');
      if (lmTemp !== undefined) params.lmTemperature = lmTemp;
      const lmCfg = readPersisted('ace-lmCfgScale');
      if (lmCfg !== undefined) params.lmCfgScale = lmCfg;
      const lmTk = readPersisted('ace-lmTopK');
      if (lmTk !== undefined) params.lmTopK = lmTk;
      const lmTp = readPersisted('ace-lmTopP');
      if (lmTp !== undefined) params.lmTopP = lmTp;
      const lmModel = readPersisted('ace-lmModel');
      if (lmModel) params.lmModel = lmModel;
      // Advanced guidance
      const cotMeta = readPersisted('ace-useCotMetas');
      if (cotMeta !== undefined) params.useCotMetas = cotMeta;
      const cotCap = readPersisted('ace-useCotCaption');
      if (cotCap !== undefined) params.useCotCaption = cotCap;
      const cotLang = readPersisted('ace-useCotLanguage');
      if (cotLang !== undefined) params.useCotLanguage = cotLang;
      // Vocoder
      const vocoder = readPersisted('ace-vocoderModel');
      if (vocoder) params.vocoderModel = vocoder;
      // Thinking (enables LM audio code generation)
      const thinkVal = readPersisted('ace-thinking');
      if (thinkVal !== undefined) params.thinking = thinkVal;
      // getLrc is useState(true) in CreatePanel, not persisted — always enable for lyrics
      params.getLrc = true;
      // Cover art: read user's global setting (same as App.tsx handleGenerate)
      params.generateCoverArt = localStorage.getItem('generate_cover_art') === 'true';

      // ── Load adapter from album preset via API ──
      if (preset?.adapter_path) {
        try {
          // Check if the adapter is already loaded to avoid corrupting an active generation
          const loraStatus = await generateApi.getLoraStatus(token);
          const existingSlot = loraStatus?.advanced?.slots?.find(
            (s: any) => s.path === preset.adapter_path
          );
          if (existingSlot) {
            console.log('[LyricStudio] Adapter already loaded, skipping reload');
            params.loraLoaded = true;
            params.loraPath = preset.adapter_path;
            params.loraScale = preset.adapter_scale ?? 1.0;
            // Still apply group scales — they may differ from what's currently set
            if (preset.adapter_group_scales) {
              try {
                await generateApi.setSlotGroupScales({
                  slot: existingSlot.slot,
                  ...preset.adapter_group_scales,
                }, token);
              } catch { /* non-critical */ }
            }
          } else {
            showToast('Loading adapter...');
            await generateApi.loadLora({
              lora_path: preset.adapter_path,
              scale: preset.adapter_scale ?? 1.0,
              ...(preset.adapter_group_scales ? { group_scales: preset.adapter_group_scales } : {}),
            }, token);
            params.loraLoaded = true;
            params.loraPath = preset.adapter_path;
            params.loraScale = preset.adapter_scale ?? 1.0;
          }
          // Apply trigger word (derive from filename, same as CreatePanel global trigger logic)
          const useFilename = localStorage.getItem('ace-globalTriggerUseFilename') === 'true';
          const placement = (localStorage.getItem('ace-globalTriggerPlacement') as 'prepend' | 'append' | 'replace') || 'prepend';
          if (useFilename) {
            const fileName = preset.adapter_path.replace(/\\/g, '/').split('/').pop() || '';
            const triggerWord = fileName.replace(/\.safetensors$/i, '').replace(/_/g, ' ');
            if (triggerWord) {
              // Refresh status to get actual slot number after load
              const refreshStatus = existingSlot
                ? { advanced: { slots: [existingSlot] } }
                : await generateApi.getLoraStatus(token);
              const slot = refreshStatus?.advanced?.slots?.find(
                (s: any) => s.path === preset.adapter_path
              );
              if (slot) {
                try {
                  await generateApi.setSlotTriggerWord({
                    slot: slot.slot,
                    trigger_word: triggerWord,
                    tag_position: placement,
                  }, token);
                  console.log(`[LyricStudio] Trigger word '${triggerWord}' (${placement}) applied`);
                } catch { /* non-critical */ }
              }
            }
          }
        } catch (loadErr) {
          console.warn('[LyricStudio] Failed to load adapter, continuing without:', loadErr);
          showToast('Warning: adapter failed to load, generating without');
        }
      }
      // ── Matchering from album preset ──
      if (preset?.matchering_reference_path) {
        params.autoMaster = true;
        params.masteringParams = { mode: 'matchering', reference_file: preset.matchering_reference_path };
      }

      const res = await generateApi.startGeneration(params, token);
      const jobId = res.jobId || (res as any).job_id;
      showToast(`Audio job queued: ${jobId}`);

      // Immediately show a "running" entry in the UI before the link call completes
      if (jobId) {
        setAudioGens(prev => ({
          ...prev,
          [gen.id]: [...(prev[gen.id] || []), { job_id: jobId, created_at: new Date().toISOString(), status: 'running' }],
        }));
        await lireekApi.linkAudio(gen.id, jobId);
        // Re-fetch to get authoritative data
        fetchAudioGens(gen.id);
      }
    } catch (err) {
      showToast(`Audio generation failed: ${(err as Error).message}`);
    } finally {
      setActionLoading(null);
    }
  };

  // ── Send Generation to Create Page ─────────────────────────────────────
  const handleSendToCreate = async (gen: Generation) => {
    // Find the album preset for adapter + matchering data
    const profile = profiles.find(p => p.id === gen.profile_id);
    let preset: AlbumPreset | null = null;
    if (profile) {
      try {
        const res = await lireekApi.getPreset(profile.lyrics_set_id);
        preset = res.preset;
      } catch { /* ignore */ }
    }

    const importData: Record<string, any> = {
      title: gen.title || '',
      prompt: gen.lyrics || '',
      style: gen.caption || '',
      instrumental: false,
    };
    if (gen.bpm) importData.bpm = gen.bpm;
    if (gen.key) importData.keyScale = gen.key;
    if (gen.duration) importData.duration = gen.duration;
    if (preset?.adapter_path) {
      importData.loraPath = preset.adapter_path;
      importData.loraScale = preset.adapter_scale ?? 1.0;
      // NOTE: Do NOT include loraLoaded/advancedAdapters/adapterSlots here.
      // CreatePanel would try to auto-load them, corrupting tensors mid-generation.
    }
    if (preset?.matchering_reference_path) {
      importData.autoMaster = true;
      importData.masteringParams = { mode: 'matchering', reference_file: preset.matchering_reference_path };
    }

    // Store in localStorage for CreatePanel to pick up
    localStorage.setItem('hotstep_lireek_import', JSON.stringify(importData));
    // Navigate to Create page
    window.history.pushState({}, '', '/create');
    window.dispatchEvent(new PopStateEvent('popstate'));
  };

  // ── Get selected detail data (with lazy-load from cache) ───────────────
  const detailKey = selectedItem ? `${selectedItem.type}-${selectedItem.id}` : null;

  const getSelectedLyricsSet = () => {
    if (!selectedItem || selectedItem.type !== 'album') return undefined;
    const cached = detailCache[`album-${selectedItem.id}`];
    if (cached) return cached as LyricsSet;
    return lyricsSets.find(ls => ls.id === selectedItem.id);
  };
  const getSelectedProfile = () => {
    if (!selectedItem || selectedItem.type !== 'profile') return undefined;
    const cached = detailCache[`profile-${selectedItem.id}`];
    if (cached) return cached as Profile;
    return profiles.find(p => p.id === selectedItem.id);
  };
  const getSelectedGeneration = () => {
    if (!selectedItem || selectedItem.type !== 'generation') return undefined;
    const cached = detailCache[`generation-${selectedItem.id}`];
    if (cached) return cached as Generation;
    return generations.find(g => g.id === selectedItem.id);
  };

  // Fetch full detail when selection changes
  useEffect(() => {
    if (!selectedItem) return;
    const key = `${selectedItem.type}-${selectedItem.id}`;
    if (detailCache[key]) return; // already cached

    const fetchDetail = async () => {
      setDetailLoading(true);
      try {
        let data: any;
        if (selectedItem.type === 'album') {
          data = await lireekApi.getLyricsSet(selectedItem.id);
        } else if (selectedItem.type === 'profile') {
          data = await lireekApi.getProfile(selectedItem.id);
        } else if (selectedItem.type === 'generation') {
          data = await lireekApi.getGeneration(selectedItem.id);
        } else {
          return;
        }
        setDetailCache(prev => ({ ...prev, [key]: data }));
      } catch (err) {
        console.error('Failed to load detail:', err);
      } finally {
        setDetailLoading(false);
      }
    };
    fetchDetail();
  }, [selectedItem?.type, selectedItem?.id]);

  // ── Render ──────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-zinc-950 text-zinc-400">
        <Loader2 className="w-6 h-6 animate-spin mr-2" /> Loading Lyric Studio…
      </div>
    );
  }

  return (
    <div className="flex-1 flex h-full overflow-hidden bg-zinc-950">
      {/* ── Left Panel: Tree + Fetch ─────────────────────────────────────── */}
      <div className="flex-shrink-0 border-r border-white/5 flex flex-col h-full overflow-hidden" style={{ width: sidebarWidth }}>
        {/* Header */}
        <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Music className="w-5 h-5 text-pink-400" />
            <h2 className="text-base font-bold text-white">Lyric Studio</h2>
          </div>
          <div className="flex items-center gap-1">
            {/* Queue badge */}
            {(() => {
              const activeQ = stream.queue.filter(q => q.status === 'running' || q.status === 'pending');
              return activeQ.length > 0 ? (
                <button
                  onClick={() => setQueueOpen(true)}
                  className="relative p-1.5 rounded-lg hover:bg-white/5 text-pink-400 transition-colors"
                  title={`Queue: ${activeQ.length} active`}
                >
                  <ListOrdered className="w-4 h-4" />
                  <span className="absolute -top-0.5 -right-0.5 w-3.5 h-3.5 rounded-full bg-pink-500 text-[9px] text-white font-bold flex items-center justify-center">
                    {activeQ.length}
                  </span>
                </button>
              ) : null;
            })()}
            <button
              onClick={() => setPromptEditorOpen(true)}
              className="p-1.5 rounded-lg hover:bg-white/5 text-zinc-400 hover:text-white transition-colors"
              title="Edit System Prompts"
            >
              <Code2 className="w-4 h-4" />
            </button>
            <button
              onClick={() => setQueueOpen(true)}
              className="p-1.5 rounded-lg hover:bg-white/5 text-zinc-400 hover:text-white transition-colors"
              title="Bulk Queue"
            >
              <ListOrdered className="w-4 h-4" />
            </button>
            <button
              onClick={loadAll}
              className="p-1.5 rounded-lg hover:bg-white/5 text-zinc-400 hover:text-white transition-colors"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button
              onClick={() => setSelectedItem({ type: 'fetch', id: 0 })}
              className="p-1.5 rounded-lg hover:bg-pink-500/20 text-pink-400 hover:text-pink-300 transition-colors"
              title="Fetch Lyrics"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Tree */}
        <div className="flex-1 overflow-y-auto scrollbar-hide py-2">
          {error && (
            <div className="mx-3 mb-2 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs flex items-center gap-2">
              <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
              {error}
            </div>
          )}

          {artists.length === 0 && !error && (
            <div className="px-4 py-8 text-center text-zinc-500 text-sm">
              <Music className="w-8 h-8 mx-auto mb-2 opacity-30" />
              No artists yet.<br />
              Click <span className="text-pink-400">+</span> to fetch lyrics from Genius.
            </div>
          )}

          {artists.map(artist => {
            const artistSets = lyricsSets.filter(ls => ls.artist_id === artist.id);
            const isExpanded = expandedArtists.has(artist.id);
            return (
              <div key={artist.id} className="mb-0.5">
                {/* Artist row */}
                <div
                  className={`flex items-center gap-1.5 px-3 py-1.5 cursor-pointer group transition-colors ${
                    selectedItem?.type === 'artist' && selectedItem.id === artist.id
                      ? 'bg-white/10 text-white' : 'text-zinc-300 hover:bg-white/5 hover:text-white'
                  }`}
                  onClick={() => {
                    toggleArtist(artist.id);
                    setSelectedItem({ type: 'artist', id: artist.id });
                  }}
                >
                  {isExpanded ? <ChevronDown className="w-3.5 h-3.5 text-zinc-500" /> : <ChevronRight className="w-3.5 h-3.5 text-zinc-500" />}
                  <Users className="w-4 h-4 text-purple-400 flex-shrink-0" />
                  <span className="text-sm font-medium truncate flex-1">{artist.name}</span>
                  <span className="text-[10px] text-zinc-600 group-hover:text-zinc-400">{artistSets.length}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDeleteArtist(artist.id); }}
                    className="p-0.5 rounded opacity-0 group-hover:opacity-100 hover:bg-red-500/20 text-zinc-500 hover:text-red-400 transition-all"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>

                {/* Albums (Lyrics Sets) */}
                {isExpanded && artistSets.map(ls => {
                  const albumProfiles = profiles.filter(p => p.lyrics_set_id === ls.id);
                  const isAlbumExpanded = expandedAlbums.has(ls.id);
                  const songs = parseSongs(ls.songs);
                  return (
                    <div key={ls.id}>
                      <div
                        className={`flex items-center gap-1.5 pl-7 pr-3 py-1.5 cursor-pointer group transition-colors ${
                          selectedItem?.type === 'album' && selectedItem.id === ls.id
                            ? 'bg-white/10 text-white' : 'text-zinc-400 hover:bg-white/5 hover:text-white'
                        }`}
                        onClick={() => {
                          toggleAlbum(ls.id);
                          setSelectedItem({ type: 'album', id: ls.id });
                        }}
                      >
                        {isAlbumExpanded ? <ChevronDown className="w-3 h-3 text-zinc-600" /> : <ChevronRight className="w-3 h-3 text-zinc-600" />}
                        <Disc3 className="w-3.5 h-3.5 text-blue-400 flex-shrink-0" />
                        <span className="text-sm truncate flex-1">{ls.album || 'Unknown Album'}</span>
                        <span className="text-[10px] text-zinc-600">{songs.length}🎵</span>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleDeleteLyricsSet(ls.id); }}
                          className="p-0.5 rounded opacity-0 group-hover:opacity-100 hover:bg-red-500/20 text-zinc-500 hover:text-red-400 transition-all"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>

                      {/* Profiles */}
                      {isAlbumExpanded && (
                        <>
                          {albumProfiles.length === 0 && (
                            <div className="pl-14 pr-3 py-1 text-[11px] text-zinc-600 italic">
                              No profiles yet
                            </div>
                          )}
                          {albumProfiles.map(profile => {
                            const profileGens = generations.filter(g => g.profile_id === profile.id);
                            const isProfileExpanded = expandedProfiles.has(profile.id);
                            return (
                              <div key={profile.id}>
                                <div
                                  className={`flex items-center gap-1.5 pl-11 pr-3 py-1 cursor-pointer group transition-colors ${
                                    selectedItem?.type === 'profile' && selectedItem.id === profile.id
                                      ? 'bg-white/10 text-white' : 'text-zinc-400 hover:bg-white/5 hover:text-white'
                                  }`}
                                  onClick={() => {
                                    toggleProfile(profile.id);
                                    setSelectedItem({ type: 'profile', id: profile.id });
                                  }}
                                >
                                  {isProfileExpanded ? <ChevronDown className="w-3 h-3 text-zinc-600" /> : <ChevronRight className="w-3 h-3 text-zinc-600" />}
                                  <Sparkles className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
                                  <span className="text-xs truncate flex-1">Profile ({profile.provider})</span>
                                  <span className="text-[10px] text-zinc-600">{profileGens.length}</span>
                                  <button
                                    onClick={(e) => { e.stopPropagation(); handleDeleteProfile(profile.id); }}
                                    className="p-0.5 rounded opacity-0 group-hover:opacity-100 hover:bg-red-500/20 text-zinc-500 hover:text-red-400 transition-all"
                                  >
                                    <Trash2 className="w-3 h-3" />
                                  </button>
                                </div>

                                {/* Generations */}
                                {isProfileExpanded && profileGens.map(gen => (
                                  <div
                                    key={gen.id}
                                    className={`flex items-center gap-1.5 pl-16 pr-3 py-1 cursor-pointer group transition-colors ${
                                      selectedItem?.type === 'generation' && selectedItem.id === gen.id
                                        ? 'bg-white/10 text-white' : 'text-zinc-500 hover:bg-white/5 hover:text-white'
                                    }`}
                                    onClick={() => setSelectedItem({ type: 'generation', id: gen.id })}
                                  >
                                    <FileText className="w-3.5 h-3.5 text-green-400 flex-shrink-0" />
                                    <span className="text-xs truncate flex-1">
                                      {gen.title || 'Untitled'}{gen.parent_generation_id ? ' ✨' : ''}
                                    </span>
                                    <span className="text-[10px] text-zinc-600">{timeAgo(gen.created_at)}</span>
                                    <button
                                      onClick={(e) => { e.stopPropagation(); handleDeleteGeneration(gen.id); }}
                                      className="p-0.5 rounded opacity-0 group-hover:opacity-100 hover:bg-red-500/20 text-zinc-500 hover:text-red-400 transition-all"
                                    >
                                      <Trash2 className="w-3 h-3" />
                                    </button>
                                  </div>
                                ))}
                              </div>
                            );
                          })}
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>

        {/* Provider selectors at bottom of tree panel */}
        <div className="px-3 py-2 border-t border-white/5">
          <TripleProviderSelector
            selections={modelSelections}
            onSelectionsChange={(sel) => {
              setModelSelections(sel);
              saveSelections(sel);
            }}
          />
        </div>
      </div>

      {/* Resize handle */}
      <div
        className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors"
        onMouseDown={(e) => {
          e.preventDefault();
          const startX = e.clientX;
          const startW = sidebarWidth;
          const onMove = (ev: MouseEvent) => {
            const newW = Math.min(600, Math.max(240, startW + ev.clientX - startX));
            setSidebarWidth(newW);
          };
          const onUp = () => {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
          };
          document.body.style.cursor = 'col-resize';
          document.body.style.userSelect = 'none';
          document.addEventListener('mousemove', onMove);
          document.addEventListener('mouseup', onUp);
        }}
      >
        <div className="w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-pink-400 transition-colors" />
      </div>

      {/* ── Right Panel: Detail View ─────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto scrollbar-hide">
        {/* LLM Streaming Output */}
        <div className="p-4 pb-0">
          <StreamingPanel
            visible={stream.visible}
            streamText={stream.text}
            phase={stream.phase}
            done={stream.done}
            onSkipThinking={doSkipThinking}
          />
        </div>

        {selectedItem?.type === 'fetch' && (
          <FetchPanel
            fetchArtist={fetchArtist}
            fetchAlbum={fetchAlbum}
            fetchMaxSongs={fetchMaxSongs}
            fetching={fetching}
            onArtistChange={setFetchArtist}
            onAlbumChange={setFetchAlbum}
            onMaxSongsChange={setFetchMaxSongs}
            onFetch={handleFetch}
            onClose={() => setSelectedItem(null)}
          />
        )}

        {selectedItem?.type === 'album' && (() => {
          const ls = getSelectedLyricsSet();
          if (!ls) return detailLoading ? <div className="flex items-center justify-center p-12 text-zinc-500"><Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading album…</div> : <EmptyDetail />;
          const songs = parseSongs(ls.songs);
          const albumProfiles = profiles.filter(p => p.lyrics_set_id === ls.id);
          return (
            <div className="p-6">
              <div className="flex items-center gap-3 mb-6">
                <Disc3 className="w-6 h-6 text-blue-400" />
                <div>
                  <h2 className="text-xl font-bold text-white">{ls.album || 'Unknown Album'}</h2>
                  <p className="text-sm text-zinc-400">{ls.artist_name} · {songs.length} songs</p>
                </div>
              </div>

              {/* Action bar */}
              <div className="flex items-center gap-2 mb-6">
                <button
                  onClick={() => handleBuildProfile(ls.id)}
                  disabled={actionLoading === `profile-${ls.id}`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 text-sm font-medium transition-colors disabled:opacity-50"
                >
                  {actionLoading === `profile-${ls.id}` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
                  Build Profile
                </button>
                <span className="text-xs text-zinc-500">{albumProfiles.length} profile(s)</span>
              </div>

              {/* Album Preset Panel */}
              <details className="mb-6" onToggle={(e) => { if ((e.target as HTMLDetailsElement).open) loadPreset(ls.id); }}>
                <summary className="flex items-center gap-2 text-sm font-semibold text-zinc-400 uppercase tracking-wider cursor-pointer hover:text-zinc-300 transition-colors">
                  <Settings2 className="w-4 h-4" /> Album Preset
                  {presets[ls.id]?.adapter_path && (
                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-green-900/30 text-green-400 normal-case tracking-normal">
                      Configured
                    </span>
                  )}
                </summary>
                <div className="mt-3 p-4 rounded-xl bg-white/5 border border-white/5 space-y-4">
                  {presetLoading ? (
                    <div className="flex items-center justify-center py-6 text-zinc-500">
                      <Loader2 className="w-4 h-4 animate-spin mr-2" /> Loading preset…
                    </div>
                  ) : (<>

                  {/* ── Adapter Path ── */}
                  <div className="space-y-2">
                    <label className="text-xs font-medium text-zinc-400">Adapter Path</label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={presetForm.adapter_path}
                        onChange={e => setPresetForm(p => ({ ...p, adapter_path: e.target.value }))}
                        placeholder="Path to .safetensors file or adapter folder"
                        className="flex-1 bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-pink-500"
                      />
                      <button
                        onClick={() => { setPresetBrowserTarget('adapter'); setPresetBrowserOpen(true); }}
                        title="Browse for adapter files"
                        className="px-3 py-2 rounded-lg text-xs font-semibold bg-pink-900/20 text-pink-400 hover:bg-pink-900/30 transition-colors flex items-center gap-1.5 flex-shrink-0"
                      >
                        <FolderSearch size={14} />
                        Browse
                      </button>
                      {presetForm.adapter_path && (
                        <button
                          onClick={() => setPresetForm(p => ({ ...p, adapter_path: '' }))}
                          className="px-2 py-2 rounded-lg text-xs text-zinc-400 hover:text-red-400 transition-colors flex-shrink-0"
                          title="Clear selection"
                        >
                          ✕
                        </button>
                      )}
                    </div>
                    {presetForm.adapter_path && (() => {
                      const fileName = presetForm.adapter_path.split(/[\\/]/).pop() || '';
                      const tag = detectedAdapterType === 'lokr' ? 'LOKR' : detectedAdapterType === 'lora' ? 'LORA' : '...';
                      const tagColor = detectedAdapterType === 'lokr'
                        ? 'bg-purple-900/30 text-purple-400'
                        : 'bg-pink-900/30 text-pink-400';
                      return (
                        <div className="flex items-center gap-2">
                          <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${tagColor}`}>
                            {tag}
                          </span>
                          <span className="text-[10px] text-zinc-500 truncate" title={presetForm.adapter_path}>
                            {fileName}
                          </span>
                        </div>
                      );
                    })()}
                  </div>

                  {/* ── Adapter Scale ── */}
                  <EditableSlider
                    label="Adapter Scale"
                    value={presetForm.adapter_scale}
                    min={0}
                    max={2}
                    step={0.05}
                    onChange={(v) => setPresetForm(p => ({ ...p, adapter_scale: v }))}
                    formatDisplay={(v) => v.toFixed(2)}
                    helpText="Overall strength of the adapter's influence on the output"
                  />

                  {/* ── Group Scales (expandable) ── */}
                  <div className="space-y-2">
                    <button
                      onClick={() => setPresetGroupsExpanded(!presetGroupsExpanded)}
                      className="flex items-center gap-1.5 text-[10px] font-semibold text-zinc-500 hover:text-zinc-300 transition-colors uppercase tracking-wider"
                    >
                      {presetGroupsExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                      Group Scales
                    </button>
                    {presetGroupsExpanded && (
                      <div className="space-y-2 pl-3 border-l-2 border-pink-500/20">
                        <EditableSlider
                          label="Self-Attn"
                          value={presetForm.self_attn}
                          min={0}
                          max={2}
                          step={0.05}
                          onChange={(v) => setPresetForm(p => ({ ...p, self_attn: v }))}
                          formatDisplay={(v) => v.toFixed(2)}
                          helpText="Controls how audio frames relate to each other over time"
                          tooltip="Self-Attention: each audio frame attends to all other frames in the sequence. Controls internal temporal coherence — how rhythmic patterns, melodic phrases, and structural transitions hold together over time."
                        />
                        <EditableSlider
                          label="Cross-Attn"
                          value={presetForm.cross_attn}
                          min={0}
                          max={2}
                          step={0.05}
                          onChange={(v) => setPresetForm(p => ({ ...p, cross_attn: v }))}
                          formatDisplay={(v) => v.toFixed(2)}
                          helpText="How strongly the text prompt shapes the output vs. the adapter's character"
                          tooltip="Cross-Attention: audio frames attend to the text/style conditioning — the bridge between your prompt and the output. Lowering this lets the adapter's baked-in character dominate over explicit prompt instructions."
                        />
                        <EditableSlider
                          label="MLP"
                          value={presetForm.mlp}
                          min={0}
                          max={2}
                          step={0.05}
                          onChange={(v) => setPresetForm(p => ({ ...p, mlp: v }))}
                          formatDisplay={(v) => v.toFixed(2)}
                          helpText="Controls the adapter's stored timbre, tonal texture, and sonic character"
                          tooltip="Feed-Forward Network (MLP): per-frame feature transformation — the 'knowledge store' of learned audio patterns. Vocal timbre, tonal texture, and specific sonic character are thought to live primarily here."
                        />
                      </div>
                    )}
                  </div>

                  {/* ── Matchering Reference ── */}
                  <div className="space-y-2 pt-2 border-t border-white/5">
                    <label className="text-xs font-medium text-zinc-400">Matchering Reference</label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={presetForm.matchering_reference_path}
                        onChange={e => setPresetForm(p => ({ ...p, matchering_reference_path: e.target.value }))}
                        placeholder="Path to reference audio (.wav, .mp3, .flac)"
                        className="flex-1 bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-pink-500"
                      />
                      <button
                        onClick={() => { setPresetBrowserTarget('matchering'); setPresetBrowserOpen(true); }}
                        title="Browse for reference audio"
                        className="px-3 py-2 rounded-lg text-xs font-semibold bg-amber-900/20 text-amber-400 hover:bg-amber-900/30 transition-colors flex items-center gap-1.5 flex-shrink-0"
                      >
                        <FolderSearch size={14} />
                        Browse
                      </button>
                      {presetForm.matchering_reference_path && (
                        <button
                          onClick={() => setPresetForm(p => ({ ...p, matchering_reference_path: '' }))}
                          className="px-2 py-2 rounded-lg text-xs text-zinc-400 hover:text-red-400 transition-colors flex-shrink-0"
                          title="Clear selection"
                        >
                          ✕
                        </button>
                      )}
                    </div>
                    {presetForm.matchering_reference_path && (() => {
                      const refName = presetForm.matchering_reference_path.split(/[\\/]/).pop() || '';
                      return (
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-900/30 text-amber-400">REF</span>
                          <span className="text-[10px] text-zinc-500 truncate" title={presetForm.matchering_reference_path}>{refName}</span>
                        </div>
                      );
                    })()}
                    <p className="text-[10px] text-zinc-600">
                      Audio file to match EQ and loudness characteristics during mastering
                    </p>
                  </div>

                  {/* ── Save Button ── */}
                  <button onClick={() => savePreset(ls.id)} disabled={actionLoading === `preset-${ls.id}`}
                    className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-gradient-to-r from-pink-500/20 to-purple-500/20 text-pink-300 hover:from-pink-500/30 hover:to-purple-500/30 text-sm font-medium transition-all disabled:opacity-50 border border-pink-500/10">
                    {actionLoading === `preset-${ls.id}` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
                    Save Preset
                  </button>
                  </>)}
                </div>
              </details>

              {/* Songs list */}
              <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">Songs</h3>
              <div className="space-y-1">
              {songs.map((song, i) => {
                  const songKey = `${ls.id}-${i}`;
                  const isExpanded = expandedSong === songKey;
                  return (
                    <div key={i}>
                      <div
                        className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors group cursor-pointer"
                        onClick={() => setExpandedSong(isExpanded ? null : songKey)}
                      >
                        {isExpanded ? <ChevronDown className="w-3 h-3 text-zinc-500 flex-shrink-0" /> : <ChevronRight className="w-3 h-3 text-zinc-500 flex-shrink-0" />}
                        <span className="text-xs text-zinc-600 w-5 text-right">{i + 1}</span>
                        <span className="text-sm text-white truncate flex-1">{song.title}</span>
                        <span className="text-[10px] text-zinc-600">{(song as any).chars || song.lyrics?.length || 0} chars</span>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleRemoveSong(ls.id, i, song.title); }}
                          className="p-0.5 rounded opacity-0 group-hover:opacity-100 hover:bg-red-500/20 text-zinc-500 hover:text-red-400 transition-all"
                          title="Remove this song"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                      {isExpanded && song.lyrics && (
                        <pre className="mx-3 mt-1 mb-2 p-3 rounded-lg bg-black/40 border border-white/5 text-xs text-zinc-300 whitespace-pre-wrap font-mono leading-relaxed max-h-[40vh] overflow-y-auto scrollbar-hide">
                          {song.lyrics}
                        </pre>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })()}

        {selectedItem?.type === 'profile' && (() => {
          const profile = getSelectedProfile();
          if (!profile) return detailLoading ? <div className="flex items-center justify-center p-12 text-zinc-500"><Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading profile…</div> : <EmptyDetail />;
          if (!profile.profile_data) return <div className="flex items-center justify-center p-12 text-zinc-500"><Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading profile data…</div>;
          const pd = profile.profile_data || {};
          const profileGens = generations.filter(g => g.profile_id === profile.id);
          return (
            <div className="p-6">
              <div className="flex items-center gap-3 mb-6">
                <Sparkles className="w-6 h-6 text-amber-400" />
                <div>
                  <h2 className="text-xl font-bold text-white">{pd.artist || 'Profile'}</h2>
                  <p className="text-sm text-zinc-400">{profile.provider}/{profile.model} · {timeAgo(profile.created_at)}</p>
                </div>
              </div>

              {/* Action bar */}
              <div className="flex items-center gap-2 mb-6">
                <button
                  onClick={() => handleGenerate(profile.id)}
                  disabled={actionLoading === `generate-${profile.id}`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-green-500/20 text-green-300 hover:bg-green-500/30 text-sm font-medium transition-colors disabled:opacity-50"
                >
                  {actionLoading === `generate-${profile.id}` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Wand2 className="w-3.5 h-3.5" />}
                  Generate Lyrics
                </button>
                <span className="text-xs text-zinc-500">{profileGens.length} generation(s)</span>
              </div>

              {/* Profile analysis — full section layout */}
              <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">Profile Analysis</h3>

              {/* Tags row: Themes + Subjects */}
              {pd.themes?.length > 0 && (
                <div className="mb-4">
                  <span className="text-[10px] text-amber-400 uppercase tracking-wider font-semibold">Themes</span>
                  <div className="flex flex-wrap gap-1.5 mt-1">
                    {(Array.isArray(pd.themes) ? pd.themes : [pd.themes]).map((t: string, i: number) => (
                      <span key={i} className="px-2 py-0.5 rounded-md text-xs bg-amber-500/15 text-amber-300 border border-amber-500/20">{t}</span>
                    ))}
                  </div>
                </div>
              )}
              {pd.common_subjects?.length > 0 && (
                <div className="mb-4">
                  <span className="text-[10px] text-green-400 uppercase tracking-wider font-semibold">Common Subjects</span>
                  <div className="flex flex-wrap gap-1.5 mt-1">
                    {(Array.isArray(pd.common_subjects) ? pd.common_subjects : [pd.common_subjects]).map((s: string, i: number) => (
                      <span key={i} className="px-2 py-0.5 rounded-md text-xs bg-green-500/15 text-green-300 border border-green-500/20">{s}</span>
                    ))}
                  </div>
                </div>
              )}
              {pd.subject_categories?.length > 0 && (
                <div className="mb-4">
                  <span className="text-[10px] text-blue-400 uppercase tracking-wider font-semibold">Subject Categories</span>
                  <div className="flex flex-wrap gap-1.5 mt-1">
                    {(Array.isArray(pd.subject_categories) ? pd.subject_categories : [pd.subject_categories]).map((c: string, i: number) => (
                      <span key={i} className="px-2 py-0.5 rounded-md text-xs bg-blue-500/15 text-blue-300 border border-blue-500/20">{c}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Inline stats row */}
              <div className="flex flex-wrap gap-3 mb-4">
                {pd.avg_verse_lines > 0 && <MetaBadge label="Avg Verse" value={`${pd.avg_verse_lines} lines`} color="blue" />}
                {pd.avg_chorus_lines > 0 && <MetaBadge label="Avg Chorus" value={`${pd.avg_chorus_lines} lines`} color="pink" />}
                {pd.rhyme_schemes?.length > 0 && <MetaBadge label="Rhyme" value={pd.rhyme_schemes.slice(0, 3).join(', ')} color="purple" />}
                {pd.perspective && <MetaBadge label="Voice" value={typeof pd.perspective === 'string' ? pd.perspective.split('—')[0].trim() : ''} color="amber" />}
              </div>

              {/* Text sections */}
              {[  
                { label: 'Tone & Mood', value: pd.tone_and_mood, color: 'text-pink-400' },
                { label: 'Vocabulary', value: pd.vocabulary_notes, color: 'text-blue-400' },
                { label: 'Structural Patterns', value: pd.structural_patterns, color: 'text-purple-400' },
                { label: 'Narrative Techniques', value: pd.narrative_techniques, color: 'text-green-400' },
                { label: 'Imagery Patterns', value: pd.imagery_patterns, color: 'text-amber-400' },
                { label: 'Signature Devices', value: pd.signature_devices, color: 'text-cyan-400' },
                { label: 'Emotional Arc', value: pd.emotional_arc, color: 'text-rose-400' },
              ].filter(s => s.value).map((section, i) => (
                <div key={i} className="mb-4">
                  <span className={`text-[10px] uppercase tracking-wider font-semibold ${section.color}`}>{section.label}</span>
                  <p className="text-sm text-zinc-300 mt-1 leading-relaxed">{section.value}</p>
                </div>
              ))}

              {/* Song subjects */}
              {pd.song_subjects && Object.keys(pd.song_subjects).length > 0 && (
                <details className="mb-4">
                  <summary className="text-[10px] text-amber-400 uppercase tracking-wider font-semibold cursor-pointer hover:text-amber-300">
                    Song Subjects ({Object.keys(pd.song_subjects).length} songs)
                  </summary>
                  <div className="mt-2 space-y-1">
                    {Object.entries(pd.song_subjects).map(([title, subject]: [string, any]) => (
                      <div key={title} className="flex gap-2 text-xs">
                        <span className="text-zinc-400 font-medium shrink-0 w-32 truncate" title={title}>{title}</span>
                        <span className="text-zinc-500">{subject}</span>
                      </div>
                    ))}
                  </div>
                </details>
              )}

              {/* Stats sections */}
              {(pd.meter_stats || pd.vocabulary_stats || pd.repetition_stats || pd.rhyme_quality) && (
                <details className="mb-4">
                  <summary className="text-[10px] text-zinc-500 uppercase tracking-wider cursor-pointer hover:text-zinc-300">Detailed Stats</summary>
                  <div className="mt-2 grid grid-cols-2 gap-3">
                    {pd.meter_stats && (
                      <div className="p-3 rounded-lg bg-white/5 border border-white/5">
                        <span className="text-[10px] text-blue-400 uppercase tracking-wider font-semibold">Meter</span>
                        <div className="text-xs text-zinc-400 mt-1 space-y-0.5">
                          <div>Avg syllables: {pd.meter_stats.avg_syllables_per_line}/line</div>
                          <div>σ = {pd.meter_stats.syllable_std_dev}</div>
                          <div>Words: {pd.meter_stats.avg_words_per_line}/line</div>
                        </div>
                      </div>
                    )}
                    {pd.vocabulary_stats && (
                      <div className="p-3 rounded-lg bg-white/5 border border-white/5">
                        <span className="text-[10px] text-green-400 uppercase tracking-wider font-semibold">Vocabulary</span>
                        <div className="text-xs text-zinc-400 mt-1 space-y-0.5">
                          <div>TTR: {pd.vocabulary_stats.type_token_ratio}</div>
                          <div>{pd.vocabulary_stats.total_words} words ({pd.vocabulary_stats.unique_words} unique)</div>
                          <div>Contractions: {pd.vocabulary_stats.contraction_pct}%</div>
                        </div>
                      </div>
                    )}
                    {pd.repetition_stats && (
                      <div className="p-3 rounded-lg bg-white/5 border border-white/5">
                        <span className="text-[10px] text-pink-400 uppercase tracking-wider font-semibold">Repetition</span>
                        <div className="text-xs text-zinc-400 mt-1 space-y-0.5">
                          <div>Chorus: {pd.repetition_stats.chorus_repetition_pct}% repeated</div>
                          <div>Pattern: {pd.repetition_stats.pattern}</div>
                        </div>
                      </div>
                    )}
                    {pd.rhyme_quality && (
                      <div className="p-3 rounded-lg bg-white/5 border border-white/5">
                        <span className="text-[10px] text-purple-400 uppercase tracking-wider font-semibold">Rhyme Quality</span>
                        <div className="text-xs text-zinc-400 mt-1 space-y-0.5">
                          <div>Perfect: {pd.rhyme_quality.perfect}</div>
                          <div>Slant: {pd.rhyme_quality.slant}</div>
                          <div>Assonance: {pd.rhyme_quality.assonance}</div>
                        </div>
                      </div>
                    )}
                  </div>
                </details>
              )}

              {/* Raw summary */}
              {pd.raw_summary && (
                <details className="mb-4">
                  <summary className="text-[10px] text-zinc-500 uppercase tracking-wider cursor-pointer hover:text-zinc-300">Full Summary</summary>
                  <div className="mt-2 p-3 rounded-lg bg-black/40 border border-white/5 text-sm text-zinc-300 whitespace-pre-wrap leading-relaxed max-h-[40vh] overflow-y-auto scrollbar-hide">
                    {pd.raw_summary}
                  </div>
                </details>
              )}

              {/* Full JSON toggle */}
              <details className="mb-4">
                <summary className="text-xs text-zinc-500 cursor-pointer hover:text-zinc-300 transition-colors">Show raw profile data</summary>
                <pre className="mt-2 p-3 rounded-lg bg-black/40 border border-white/5 text-xs text-zinc-400 overflow-auto max-h-64 whitespace-pre-wrap">
                  {JSON.stringify(pd, null, 2)}
                </pre>
              </details>
            </div>
          );
        })()}

        {selectedItem?.type === 'generation' && (() => {
          const gen = getSelectedGeneration();
          if (!gen) return detailLoading ? <div className="flex items-center justify-center p-12 text-zinc-500"><Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading generation…</div> : <EmptyDetail />;
          if (gen.lyrics === undefined) return <div className="flex items-center justify-center p-12 text-zinc-500"><Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading lyrics…</div>;
          return (
            <div className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <FileText className="w-6 h-6 text-green-400" />
                <div className="flex-1">
                  <input
                    className="text-xl font-bold text-white bg-transparent border-b border-transparent hover:border-white/20 focus:border-pink-500/50 focus:outline-none w-full transition-colors"
                    defaultValue={gen.title || 'Untitled'}
                    onBlur={(e) => { if (e.target.value !== gen.title) handleSaveGeneration(gen.id, 'title', e.target.value); }}
                  />
                  <p className="text-sm text-zinc-400">
                    {gen.artist_name && `${gen.artist_name} · `}
                    {gen.provider}/{gen.model} · {timeAgo(gen.created_at)}
                    {gen.parent_generation_id && ' · ✨ Refined'}
                  </p>
                </div>
                <Pencil className="w-4 h-4 text-zinc-600" title="All fields are editable" />
              </div>

              {/* Editable metadata */}
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="px-3 py-2 rounded-lg bg-white/5 border border-white/5">
                  <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">Subject</label>
                  <input
                    className="w-full bg-transparent text-sm text-amber-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-amber-500/50 transition-colors"
                    defaultValue={gen.subject || ''}
                    onBlur={(e) => { if (e.target.value !== (gen.subject || '')) handleSaveGeneration(gen.id, 'subject', e.target.value); }}
                  />
                </div>
                <div className="px-3 py-2 rounded-lg bg-white/5 border border-white/5">
                  <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">BPM</label>
                  <input
                    type="number"
                    className="w-full bg-transparent text-sm text-pink-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-pink-500/50 transition-colors"
                    defaultValue={gen.bpm || 0}
                    onBlur={(e) => { const v = parseInt(e.target.value) || 0; if (v !== gen.bpm) handleSaveGeneration(gen.id, 'bpm', v); }}
                  />
                </div>
                <div className="px-3 py-2 rounded-lg bg-white/5 border border-white/5">
                  <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">Key</label>
                  <input
                    className="w-full bg-transparent text-sm text-blue-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-blue-500/50 transition-colors"
                    defaultValue={gen.key || ''}
                    onBlur={(e) => { if (e.target.value !== (gen.key || '')) handleSaveGeneration(gen.id, 'key', e.target.value); }}
                  />
                </div>
                <div className="px-3 py-2 rounded-lg bg-white/5 border border-white/5">
                  <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">Duration (seconds)</label>
                  <input
                    type="number"
                    className="w-full bg-transparent text-sm text-purple-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-purple-500/50 transition-colors"
                    defaultValue={gen.duration || 0}
                    onBlur={(e) => { const v = parseInt(e.target.value) || 0; if (v !== gen.duration) handleSaveGeneration(gen.id, 'duration', v); }}
                  />
                </div>
              </div>

              {/* Editable caption */}
              <div className="mb-4 px-3 py-2 rounded-lg bg-white/5 border border-white/5">
                <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">Caption</label>
                <textarea
                  className="w-full bg-transparent text-sm text-zinc-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-pink-500/50 transition-colors resize-none"
                  rows={2}
                  defaultValue={gen.caption || ''}
                  onBlur={(e) => { if (e.target.value !== (gen.caption || '')) handleSaveGeneration(gen.id, 'caption', e.target.value); }}
                />
              </div>

              {/* Actions */}
              <div className="flex items-center gap-2 mb-6">
                <button
                  onClick={() => handleRefine(gen.id)}
                  disabled={actionLoading === `refine-${gen.id}`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-purple-500/20 text-purple-300 hover:bg-purple-500/30 text-sm font-medium transition-colors disabled:opacity-50"
                >
                  {actionLoading === `refine-${gen.id}` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Wand2 className="w-3.5 h-3.5" />}
                  Refine
                </button>
                <button
                  onClick={() => handleExport(gen.id)}
                  disabled={actionLoading === `export-${gen.id}`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-500/20 text-blue-300 hover:bg-blue-500/30 text-sm font-medium transition-colors disabled:opacity-50"
                >
                  {actionLoading === `export-${gen.id}` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />}
                  Export
                </button>
                <button
                  onClick={() => handleGenerateAudio(gen)}
                  disabled={actionLoading === `audio-${gen.id}`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-pink-500/30 to-purple-500/30 text-white hover:from-pink-500/40 hover:to-purple-500/40 text-sm font-semibold transition-all disabled:opacity-50 border border-pink-500/20"
                >
                  {actionLoading === `audio-${gen.id}` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                  Generate Audio
                </button>
                <button
                  onClick={() => handleSendToCreate(gen)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 text-sm font-medium transition-colors border border-amber-500/10"
                  title="Send all generation data to the Create Page for further customization"
                >
                  <Wand2 className="w-3.5 h-3.5" />
                  Send to Create
                </button>
                <button
                  onClick={() => handleDeleteGeneration(gen.id)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 text-sm font-medium transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Editable lyrics */}
              <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">Lyrics</h3>
              <textarea
                className="w-full p-4 rounded-xl bg-black/40 border border-white/5 text-sm text-zinc-200 font-mono leading-relaxed focus:outline-none focus:border-pink-500/30 resize-y transition-colors scrollbar-hide"
                style={{ minHeight: '300px' }}
                defaultValue={gen.lyrics}
                onBlur={(e) => { if (e.target.value !== gen.lyrics) handleSaveGeneration(gen.id, 'lyrics', e.target.value); }}
              />

              {/* ── Linked Audio Generations ── */}
              <AudioGenSection genId={gen.id} audioGens={audioGens} fetchAudioGens={fetchAudioGens} token={token} onPlaySong={onPlaySong} genTitle={gen.title} genCaption={gen.caption} />
            </div>
          );
        })()}

        {selectedItem?.type === 'artist' && (() => {
          const artist = artists.find(a => a.id === selectedItem.id);
          if (!artist) return <EmptyDetail />;
          const artistSets = lyricsSets.filter(ls => ls.artist_id === artist.id);
          const totalSongs = artistSets.reduce((sum, ls) => sum + parseSongs(ls.songs).length, 0);
          const totalProfiles = profiles.filter(p => artistSets.some(ls => ls.id === p.lyrics_set_id)).length;
          const totalGens = generations.filter(g => {
            const p = profiles.find(pp => pp.id === g.profile_id);
            return p && artistSets.some(ls => ls.id === p.lyrics_set_id);
          }).length;
          return (
            <div className="p-6">
              <div className="flex items-center gap-3 mb-6">
                <Users className="w-6 h-6 text-purple-400" />
                <div>
                  <h2 className="text-xl font-bold text-white">{artist.name}</h2>
                  <p className="text-sm text-zinc-400">Added {timeAgo(artist.created_at)}</p>
                </div>
              </div>
              <div className="grid grid-cols-4 gap-3">
                <StatCard label="Albums" value={artistSets.length} color="blue" />
                <StatCard label="Songs" value={totalSongs} color="green" />
                <StatCard label="Profiles" value={totalProfiles} color="amber" />
                <StatCard label="Generations" value={totalGens} color="pink" />
              </div>
            </div>
          );
        })()}

        {!selectedItem && <EmptyDetail />}
      </div>

      {/* Toast */}
      {toast && (
        <div className="fixed bottom-20 left-1/2 -translate-x-1/2 z-50 px-4 py-2 rounded-xl bg-zinc-800 border border-white/10 text-sm text-white shadow-xl animate-in fade-in slide-in-from-bottom-4 duration-200">
          {toast}
        </div>
      )}

      {/* Queue Modal */}
      <QueuePanel
        open={queueOpen}
        onClose={() => setQueueOpen(false)}
        artists={artists}
        lyricsSets={lyricsSets}
        profiles={profiles}
        profilingModel={modelSelections.profiling}
        generationModel={modelSelections.generation}
        refinementModel={modelSelections.refinement}
      />

      {/* Prompt Editor Modal */}
      <PromptEditor
        open={promptEditorOpen}
        onClose={() => setPromptEditorOpen(false)}
      />

      {/* Preset File Browser */}
      <FileBrowserModal
        open={presetBrowserOpen}
        onClose={() => setPresetBrowserOpen(false)}
        onSelect={(path) => {
          if (presetBrowserTarget === 'adapter') {
            setPresetForm(p => ({ ...p, adapter_path: path }));
          } else {
            setPresetForm(p => ({ ...p, matchering_reference_path: path }));
          }
          setPresetBrowserOpen(false);
        }}
        mode="file"
        filter={presetBrowserTarget === 'matchering' ? 'audio' : 'adapters'}
        title={presetBrowserTarget === 'matchering' ? 'Select Reference Audio' : 'Select Adapter File'}
      />
    </div>
  );
};

// ── Sub-components ──────────────────────────────────────────────────────────

const EmptyDetail: React.FC = () => (
  <div className="flex-1 flex items-center justify-center h-full text-zinc-600">
    <div className="text-center">
      <Music className="w-12 h-12 mx-auto mb-3 opacity-20" />
      <p className="text-sm">Select an item from the tree<br />or click + to fetch lyrics</p>
    </div>
  </div>
);

const FetchPanel: React.FC<{
  fetchArtist: string;
  fetchAlbum: string;
  fetchMaxSongs: number;
  fetching: boolean;
  onArtistChange: (v: string) => void;
  onAlbumChange: (v: string) => void;
  onMaxSongsChange: (v: number) => void;
  onFetch: () => void;
  onClose: () => void;
}> = ({ fetchArtist, fetchAlbum, fetchMaxSongs, fetching, onArtistChange, onAlbumChange, onMaxSongsChange, onFetch, onClose }) => (
  <div className="p-6 max-w-lg">
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-2">
        <Search className="w-5 h-5 text-pink-400" />
        <h2 className="text-lg font-bold text-white">Fetch Lyrics from Genius</h2>
      </div>
      <button onClick={onClose} className="p-1 rounded-lg hover:bg-white/5 text-zinc-400 hover:text-white">
        <X className="w-4 h-4" />
      </button>
    </div>

    <div className="space-y-4">
      <div>
        <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider mb-1.5">Artist name or Genius URL</label>
        <input
          type="text"
          value={fetchArtist}
          onChange={e => onArtistChange(e.target.value)}
          placeholder="e.g. Blink-182 or https://genius.com/artists/..."
          className="w-full px-3 py-2 rounded-lg bg-zinc-800 border border-white/10 text-white text-sm placeholder-zinc-600 focus:outline-none focus:border-pink-500/50"
          onKeyDown={e => e.key === 'Enter' && onFetch()}
        />
      </div>
      <div>
        <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider mb-1.5">Album (optional)</label>
        <input
          type="text"
          value={fetchAlbum}
          onChange={e => onAlbumChange(e.target.value)}
          placeholder="e.g. Enema of the State"
          className="w-full px-3 py-2 rounded-lg bg-zinc-800 border border-white/10 text-white text-sm placeholder-zinc-600 focus:outline-none focus:border-pink-500/50"
          onKeyDown={e => e.key === 'Enter' && onFetch()}
        />
      </div>
      <div>
        <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider mb-1.5">Max songs</label>
        <input
          type="number"
          value={fetchMaxSongs}
          onChange={e => onMaxSongsChange(Math.max(1, Math.min(50, parseInt(e.target.value) || 10)))}
          min={1}
          max={50}
          className="w-24 px-3 py-2 rounded-lg bg-zinc-800 border border-white/10 text-white text-sm focus:outline-none focus:border-pink-500/50"
        />
      </div>
      <button
        onClick={onFetch}
        disabled={!fetchArtist.trim() || fetching}
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-pink-500 to-purple-600 text-white text-sm font-semibold hover:from-pink-600 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {fetching ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
        {fetching ? 'Fetching…' : 'Fetch Lyrics'}
      </button>
    </div>
  </div>
);

const ProfileCard: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="px-3 py-2 rounded-lg bg-white/5 border border-white/5">
    <span className="text-[10px] text-zinc-500 uppercase tracking-wider">{label}</span>
    <p className="text-sm text-zinc-200 mt-0.5 line-clamp-2">{value}</p>
  </div>
);

const MetaBadge: React.FC<{ label: string; value: string; color: string }> = ({ label, value, color }) => {
  const colors: Record<string, string> = {
    pink: 'bg-pink-500/20 text-pink-300 border-pink-500/20',
    blue: 'bg-blue-500/20 text-blue-300 border-blue-500/20',
    purple: 'bg-purple-500/20 text-purple-300 border-purple-500/20',
    green: 'bg-green-500/20 text-green-300 border-green-500/20',
    amber: 'bg-amber-500/20 text-amber-300 border-amber-500/20',
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium border ${colors[color] || colors.pink}`}>
      {label}: {value}
    </span>
  );
};

const StatCard: React.FC<{ label: string; value: number; color: string }> = ({ label, value, color }) => {
  const colors: Record<string, string> = {
    pink: 'text-pink-400',
    blue: 'text-blue-400',
    purple: 'text-purple-400',
    green: 'text-green-400',
    amber: 'text-amber-400',
  };
  return (
    <div className="px-4 py-3 rounded-xl bg-white/5 border border-white/5 text-center">
      <div className={`text-2xl font-bold ${colors[color] || 'text-white'}`}>{value}</div>
      <div className="text-[10px] text-zinc-500 uppercase tracking-wider mt-0.5">{label}</div>
    </div>
  );
};

// ── Audio Generation Section ─────────────────────────────────────────────
const AudioGenSection: React.FC<{
  genId: number;
  audioGens: Record<number, Array<{ job_id: string; created_at: string; status?: string; audioUrl?: string }>>;
  fetchAudioGens: (genId: number) => void;
  token: string;
  onPlaySong?: (song: Song) => void;
  genTitle?: string;
  genCaption?: string;
}> = ({ genId, audioGens, fetchAudioGens, token, onPlaySong, genTitle, genCaption }) => {
  const entries = audioGens[genId];
  const [loaded, setLoaded] = useState(false);

  // Auto-fetch on first render
  useEffect(() => {
    if (!loaded && token) {
      fetchAudioGens(genId);
      setLoaded(true);
    }
  }, [loaded, token, genId, fetchAudioGens]);

  // Poll running jobs every 5s
  useEffect(() => {
    const hasRunning = entries?.some(e => e.status === 'running' || e.status === 'pending' || e.status === 'queued');
    if (!hasRunning) return;
    const interval = setInterval(() => fetchAudioGens(genId), 5000);
    return () => clearInterval(interval);
  }, [entries, genId, fetchAudioGens]);

  if (!entries || entries.length === 0) return null;

  const statusBadge = (status?: string) => {
    const styles: Record<string, string> = {
      succeeded: 'bg-green-900/30 text-green-400',
      running: 'bg-blue-900/30 text-blue-400',
      pending: 'bg-yellow-900/30 text-yellow-400',
      queued: 'bg-yellow-900/30 text-yellow-400',
      failed: 'bg-red-900/30 text-red-400',
    };
    return (
      <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${styles[status || ''] || 'bg-zinc-800 text-zinc-400'}`}>
        {status || 'unknown'}
      </span>
    );
  };

  const handlePlay = (ag: { job_id: string; created_at: string; status?: string; audioUrl?: string }, index: number) => {
    if (!ag.audioUrl || !onPlaySong) return;
    const song: Song = {
      id: `lireek_${genId}_${ag.job_id}`,
      title: genTitle || `Generation ${index + 1}`,
      lyrics: '',
      style: genCaption || '',
      coverUrl: '',
      duration: '',
      createdAt: new Date(ag.created_at),
      tags: [],
      audioUrl: ag.audioUrl,
    };
    onPlaySong(song);
  };

  return (
    <div className="mt-4">
      <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-2 flex items-center gap-2">
        <Music className="w-3.5 h-3.5" /> Audio Generations ({entries.length})
        <button onClick={() => fetchAudioGens(genId)} className="ml-auto text-zinc-500 hover:text-zinc-300 transition-colors" title="Refresh">
          <RefreshCw className="w-3 h-3" />
        </button>
      </h3>
      <div className="space-y-2">
        {entries.map((ag, i) => (
          <div key={ag.job_id} className="rounded-lg bg-black/30 border border-white/5 p-3 space-y-2">
            <div className="flex items-center gap-2 text-xs">
              {statusBadge(ag.status)}
              <span className="text-zinc-500 font-mono truncate">{ag.job_id.slice(0, 8)}…</span>
              <span className="text-zinc-600 ml-auto">{new Date(ag.created_at).toLocaleTimeString()}</span>
            </div>
            {ag.status === 'running' && (
              <div className="flex items-center gap-2 text-xs text-blue-400">
                <Loader2 className="w-3 h-3 animate-spin" /> Generating…
              </div>
            )}
            {ag.status === 'succeeded' && ag.audioUrl && (
              <button
                onClick={() => handlePlay(ag, i)}
                className="flex items-center gap-2 w-full px-3 py-2 rounded-lg bg-green-500/10 hover:bg-green-500/20 text-green-400 text-sm font-medium transition-colors"
              >
                <Play className="w-4 h-4 fill-current" />
                Play in Player
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
