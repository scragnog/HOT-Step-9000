import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Upload, Music, Search, Play, Pause, Loader2, Guitar, Disc3, Zap, X, RefreshCw, Download, Trash2, Volume2, VolumeX, Sliders, ChevronDown, RotateCcw } from 'lucide-react';
import { lireekApi, Artist, AlbumPreset, LyricsSet } from '../../services/lyricStudioApi';
import { generateApi, songsApi } from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { Song } from '../../types';
import { EditableSlider } from '../EditableSlider';
import { DownloadModal, DownloadFormat, DownloadVersion } from '../DownloadModal';

// ── Types ────────────────────────────────────────────────────────────────────

interface AudioMetadata {
  artist: string;
  title: string;
  album: string;
  duration: number | null;
}

interface AudioAnalysis {
  bpm: number;
  key: string;
  scale?: string;
}

interface CoverStudioProps {
  onPlaySong: (song: Song, list?: Song[]) => void;
  isPlaying: boolean;
  currentSong: Song | null;
  currentTime: number;
}

// ── Persistence helpers (localStorage) ──────────────────────────────────────

const STORAGE_PREFIX = 'cover-studio-';
const TRACK_CACHE_KEY = 'cover-studio-trackCache';

function persist(key: string, value: any) {
  try { localStorage.setItem(STORAGE_PREFIX + key, JSON.stringify(value)); } catch {}
}

function restore<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(STORAGE_PREFIX + key);
    return raw !== null ? JSON.parse(raw) : fallback;
  } catch { return fallback; }
}

// ── Source track cache — remembers analysis + lyrics per filename ────────────

interface TrackCacheEntry {
  artist: string;
  title: string;
  lyrics: string;
  bpm: number;
  key: string;
  scale?: string;
  duration: number | null;
  album?: string;
}

function getTrackCache(): Record<string, TrackCacheEntry> {
  try {
    return JSON.parse(localStorage.getItem(TRACK_CACHE_KEY) || '{}');
  } catch { return {}; }
}

function saveTrackCacheEntry(filename: string, entry: Partial<TrackCacheEntry>) {
  try {
    const cache = getTrackCache();
    cache[filename] = { ...(cache[filename] || {}), ...entry } as TrackCacheEntry;
    localStorage.setItem(TRACK_CACHE_KEY, JSON.stringify(cache));
  } catch {}
}

// ── Helpers from useAudioGeneration (reused for param merging) ───────────────

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
  if (placement === 'replace') {
    params.style = triggerWord;
  } else if (placement === 'append') {
    params.style = current ? `${current}, ${triggerWord}` : triggerWord;
  } else {
    params.style = current ? `${triggerWord}, ${current}` : triggerWord;
  }
  console.log(`[CoverStudio] Trigger word '${triggerWord}' ${placement}ed → '${params.style}'`);
}

/** Transpose a key string (e.g. "D minor") by a number of semitones. */
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const NOTE_ALIASES: Record<string, number> = {
  'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'Fb': 4,
  'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11,
};

function transposeKey(keyStr: string, semitones: number): string {
  if (!keyStr || semitones === 0) return keyStr;
  // Parse "D minor" or "C# major" or "A"
  const parts = keyStr.trim().split(/\s+/);
  const notePart = parts[0];
  const quality = parts.slice(1).join(' ') || '';
  const noteIndex = NOTE_ALIASES[notePart];
  if (noteIndex === undefined) return keyStr;
  const newIndex = ((noteIndex + semitones) % 12 + 12) % 12;
  return quality ? `${NOTE_NAMES[newIndex]} ${quality}` : NOTE_NAMES[newIndex];
}

// ── Main Component ───────────────────────────────────────────────────────────

export const CoverStudio: React.FC<CoverStudioProps> = ({
  onPlaySong,
  isPlaying,
  currentSong,
}) => {
  const { token } = useAuth();

  // Source audio state (persisted)
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [sourceFileName, setSourceFileName] = useState(() => restore<string>('sourceFileName', ''));
  const [sourceAudioUrl, setSourceAudioUrl] = useState(() => restore<string>('sourceAudioUrl', ''));
  const [metadata, setMetadata] = useState<AudioMetadata | null>(() => restore('metadata', null));
  const [analysis, setAnalysis] = useState<AudioAnalysis | null>(() => restore('analysis', null));
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Song details state (persisted)
  const [songArtist, setSongArtist] = useState(() => restore<string>('songArtist', ''));
  const [songTitle, setSongTitle] = useState(() => restore<string>('songTitle', ''));
  const [lyrics, setLyrics] = useState(() => restore<string>('lyrics', ''));
  const [isSearchingLyrics, setIsSearchingLyrics] = useState(false);

  // Target artist state (persisted)
  const [artists, setArtists] = useState<Artist[]>([]);
  const [selectedArtistId, setSelectedArtistId] = useState<number | null>(() => restore('selectedArtistId', null));
  const [selectedPreset, setSelectedPreset] = useState<AlbumPreset | null>(() => restore('selectedPreset', null));
  const [artistCaption, setArtistCaption] = useState(() => restore<string>('artistCaption', ''));
  const [artistPresets, setArtistPresets] = useState<{ lsId: number; album: string; preset: AlbumPreset | null }[]>([]);
  const [isLoadingArtists, setIsLoadingArtists] = useState(false);

  // Cover settings (persisted)
  const [audioCoverStrength, setAudioCoverStrength] = useState(() => restore<number>('audioCoverStrength', 0.5));
  const [coverNoiseStrength, setCoverNoiseStrength] = useState(() => restore<number>('coverNoiseStrength', 0));
  const [tempoScale, setTempoScale] = useState(() => restore<number>('tempoScale', 1.0));
  const [pitchShift, setPitchShift] = useState(() => restore<number>('pitchShift', 0));
  // Advanced Mode (SuperSep stem separation)
  const [advancedMode, setAdvancedMode] = useState(() => restore<boolean>('advancedMode', false));
  const [sepLevel, setSepLevel] = useState(() => restore<string>('sepLevel', 'basic'));
  const [superSepStems, setSuperSepStems] = useState<any[]>([]);
  const [stemVolumes, setStemVolumes] = useState<Record<string, { volume: number; muted: boolean }>>({});
  const [isSeparating, setIsSeparating] = useState(false);
  const [sepProgress, setSepProgress] = useState(0);
  const [sepStage, setSepStage] = useState('');
  const [isRecombining, setIsRecombining] = useState(false);
  const [isPreviewPlaying, setIsPreviewPlaying] = useState(false);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const stemAudiosRef = useRef<{ el: HTMLAudioElement; gain: GainNode; source: MediaElementAudioSourceNode }[]>([]);

  // Generation
  const [isGenerating, setIsGenerating] = useState(false);
  const [toast, setToast] = useState('');
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [genProgress, setGenProgress] = useState(0);
  const [genStage, setGenStage] = useState('');
  const [filenamePrepend, setFilenamePrepend] = useState(() => restore<string>('filenamePrepend', ''));

  // Persist state changes
  useEffect(() => { persist('sourceFileName', sourceFileName); }, [sourceFileName]);
  useEffect(() => { persist('sourceAudioUrl', sourceAudioUrl); }, [sourceAudioUrl]);
  useEffect(() => { persist('metadata', metadata); }, [metadata]);
  useEffect(() => { persist('analysis', analysis); }, [analysis]);
  useEffect(() => { persist('songArtist', songArtist); }, [songArtist]);
  useEffect(() => { persist('songTitle', songTitle); }, [songTitle]);
  useEffect(() => { persist('lyrics', lyrics); }, [lyrics]);
  useEffect(() => { persist('selectedArtistId', selectedArtistId); }, [selectedArtistId]);
  useEffect(() => { persist('selectedPreset', selectedPreset); }, [selectedPreset]);
  useEffect(() => { persist('artistCaption', artistCaption); }, [artistCaption]);
  useEffect(() => { persist('audioCoverStrength', audioCoverStrength); }, [audioCoverStrength]);
  useEffect(() => { persist('coverNoiseStrength', coverNoiseStrength); }, [coverNoiseStrength]);
  useEffect(() => { persist('tempoScale', tempoScale); }, [tempoScale]);
  useEffect(() => { persist('pitchShift', pitchShift); }, [pitchShift]);
  useEffect(() => { persist('advancedMode', advancedMode); }, [advancedMode]);
  useEffect(() => { persist('sepLevel', sepLevel); }, [sepLevel]);
  useEffect(() => { persist('filenamePrepend', filenamePrepend); }, [filenamePrepend]);

  // Sync stem volume/mute to Web Audio gain nodes in real-time
  useEffect(() => {
    if (!isPreviewPlaying) return;
    stemAudiosRef.current.forEach(({ el, gain }, idx) => {
      const stem = superSepStems[idx];
      if (!stem) return;
      const sv = stemVolumes[stem.id] || { volume: 1.0, muted: false };
      gain.gain.value = sv.muted ? 0 : sv.volume;
    });
  }, [stemVolumes, isPreviewPlaying, superSepStems]);

  // Load artists on mount
  useEffect(() => {
    setIsLoadingArtists(true);
    lireekApi.listArtists()
      .then(res => {
        setArtists(res.artists);
        // Re-select previously selected artist's preset
        if (selectedArtistId && !selectedPreset) {
          const artist = res.artists.find(a => a.id === selectedArtistId);
          if (artist) loadArtistPresets(artist);
        }
      })
      .catch(() => showToast('Failed to load artists'))
      .finally(() => setIsLoadingArtists(false));
  }, []);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(''), 4000);
  };

  // ── File Upload + Metadata + Analysis ──────────────────────────────────

  const handleFileSelected = async (file: File) => {
    if (!token) { showToast('Please sign in first'); return; }

    setSourceFile(file);
    setSourceFileName(file.name);

    // ── Check track cache first ──
    const cached = getTrackCache()[file.name];
    if (cached) {
      console.log('[CoverStudio] Track cache hit for', file.name);
      showToast('Loaded from cache — instant recall!');
      if (cached.artist) setSongArtist(cached.artist);
      if (cached.title) setSongTitle(cached.title);
      if (cached.lyrics) setLyrics(cached.lyrics);
      setMetadata({ artist: cached.artist || '', title: cached.title || '', album: cached.album || '', duration: cached.duration });
      setAnalysis({ bpm: cached.bpm, key: cached.key, scale: cached.scale });

      // Still need to upload as reference track for the generation pipeline
      setIsUploading(true);
      try {
        const uploadFormData = new FormData();
        uploadFormData.append('audio', file);
        const uploadRes = await fetch('/api/reference-tracks', {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
          body: uploadFormData,
        });
        if (!uploadRes.ok) throw new Error('Upload failed');
        const uploadData = await uploadRes.json();
        setSourceAudioUrl(uploadData.track?.audio_url || '');
      } catch (err: any) {
        showToast(`Upload error: ${err.message}`);
      } finally {
        setIsUploading(false);
      }
      return;
    }

    // ── No cache — full analysis pipeline ──
    setIsUploading(true);
    try {
      // 1. Extract metadata
      const metaFormData = new FormData();
      metaFormData.append('audio', file);
      const metaRes = await fetch('/api/analyze/metadata', { method: 'POST', body: metaFormData });
      let extractedArtist = '';
      let extractedTitle = '';
      let extractedAlbum = '';
      let extractedDuration: number | null = null;
      if (metaRes.ok) {
        const meta: AudioMetadata = await metaRes.json();
        setMetadata(meta);
        extractedArtist = meta.artist || '';
        extractedTitle = meta.title || '';
        extractedAlbum = meta.album || '';
        extractedDuration = meta.duration;
        if (meta.artist) setSongArtist(meta.artist);
        if (meta.title) setSongTitle(meta.title);
      }

      // 2. Upload as reference track
      const uploadFormData = new FormData();
      uploadFormData.append('audio', file);
      const uploadRes = await fetch('/api/reference-tracks', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: uploadFormData,
      });
      if (!uploadRes.ok) throw new Error('Upload failed');
      const uploadData = await uploadRes.json();
      const audioUrl = uploadData.track?.audio_url || '';
      setSourceAudioUrl(audioUrl);

      // 3. Analyze with Essentia for BPM/key
      setIsUploading(false);
      setIsAnalyzing(true);
      let analyzedBpm = 120;
      let analyzedKey = 'C major';
      let analyzedScale: string | undefined;
      const analyzeRes = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audioUrl }),
      });
      if (analyzeRes.ok) {
        const data = await analyzeRes.json();
        analyzedBpm = data.bpm || 120;
        analyzedKey = `${data.key || 'C'} ${data.scale || 'major'}`;
        analyzedScale = data.scale;
        setAnalysis({ bpm: analyzedBpm, key: analyzedKey, scale: analyzedScale });
      }

      // 4. Save to track cache (lyrics added later after Genius search)
      saveTrackCacheEntry(file.name, {
        artist: extractedArtist,
        title: extractedTitle,
        album: extractedAlbum,
        duration: extractedDuration,
        bpm: analyzedBpm,
        key: analyzedKey,
        scale: analyzedScale,
      });
    } catch (err: any) {
      showToast(`Error: ${err.message}`);
    } finally {
      setIsUploading(false);
      setIsAnalyzing(false);
    }
  };

  // ── Genius Lyrics Search ───────────────────────────────────────────────

  const handleSearchLyrics = async () => {
    if (!songArtist.trim() || !songTitle.trim()) { showToast('Enter artist and title first'); return; }
    setIsSearchingLyrics(true);
    try {
      const result = await lireekApi.searchSongLyrics(songArtist.trim(), songTitle.trim());
      setLyrics(result.lyrics);
      if (result.title) setSongTitle(result.title);
      showToast('Lyrics found!');

      // Update track cache with lyrics (and any corrected artist/title)
      if (sourceFileName) {
        saveTrackCacheEntry(sourceFileName, {
          lyrics: result.lyrics,
          artist: songArtist.trim(),
          title: result.title || songTitle.trim(),
        });
      }
    } catch (err: any) {
      showToast(err.message || 'No lyrics found');
    } finally {
      setIsSearchingLyrics(false);
    }
  };

  // ── Artist Selection — loads ALL presets, picks the one with an adapter ─

  const loadArtistPresets = async (artist: Artist) => {
    try {
      const { lyrics_sets } = await lireekApi.listLyricsSets(artist.id);
      const presetResults: { lsId: number; album: string; preset: AlbumPreset | null }[] = [];

      for (const ls of lyrics_sets) {
        try {
          const { preset } = await lireekApi.getPreset(ls.id);
          presetResults.push({ lsId: ls.id, album: ls.album || ls.id.toString(), preset });
        } catch {
          presetResults.push({ lsId: ls.id, album: ls.album || ls.id.toString(), preset: null });
        }
      }
      setArtistPresets(presetResults);

      // Fetch caption from the first lyrics set that has generations with a caption
      let foundCaption = '';
      for (const ls of lyrics_sets) {
        try {
          const { generations } = await lireekApi.listGenerations(undefined, ls.id);
          console.log(`[CoverStudio] Lyrics set ${ls.id} (${ls.album}): ${generations.length} generations`);
          const withCaption = generations.find(g => g.caption && g.caption.trim().length > 0);
          if (withCaption?.caption) {
            foundCaption = withCaption.caption;
            console.log(`[CoverStudio] Found caption from ${ls.album}:`, foundCaption);
            break;
          }
        } catch (err) {
          console.warn(`[CoverStudio] Failed to fetch generations for lyrics set ${ls.id}:`, err);
        }
      }
      setArtistCaption(foundCaption);

      // Pick the first preset that has an adapter_path
      const withAdapter = presetResults.find(p => p.preset?.adapter_path);
      if (withAdapter?.preset) {
        setSelectedPreset(withAdapter.preset);
        if (withAdapter.preset.audio_cover_strength != null) {
          setAudioCoverStrength(withAdapter.preset.audio_cover_strength);
        }
      } else if (presetResults.length > 0 && presetResults[0].preset) {
        setSelectedPreset(presetResults[0].preset);
      } else {
        setSelectedPreset(null);
      }
    } catch {
      setSelectedPreset(null);
      setArtistPresets([]);
      setArtistCaption('');
    }
  };

  const handleSelectArtist = async (artist: Artist) => {
    setSelectedArtistId(artist.id);
    await loadArtistPresets(artist);
  };

  // ── SuperSep Advanced Mode ──────────────────────────────────────────

  const PYTHON_API = `http://${window.location.hostname}:8001`;

  const runSuperSep = async (audioUrl: string, level: string) => {
    setIsSeparating(true);
    setSepProgress(0);
    setSepStage('Starting SuperSep...');
    setSuperSepStems([]);
    setStemVolumes({});
    try {
      const resp = await fetch(`${PYTHON_API}/v1/supersep/separate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audio_path: audioUrl, level }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || `SuperSep failed: ${resp.status}`);
      }
      const { job_id } = await resp.json();

      // SSE progress
      return new Promise<any[]>((resolve, reject) => {
        const es = new EventSource(`${PYTHON_API}/v1/supersep/${job_id}/progress`);
        const timeout = setTimeout(() => { es.close(); reject(new Error('SuperSep timed out (10 min)')); }, 600_000);
        es.onmessage = (evt) => {
          try {
            const data = JSON.parse(evt.data);
            if (data.type === 'progress') {
              setSepProgress(Math.round((data.percent || 0) * 100));
              setSepStage(data.message || `Stage ${data.stage}`);
            } else if (data.type === 'complete') {
              clearTimeout(timeout);
              es.close();
              const stems = data.stems || [];
              setSuperSepStems(stems);
              // Initialize volumes
              const vols: Record<string, { volume: number; muted: boolean }> = {};
              stems.forEach((s: any) => { vols[s.id] = { volume: 1.0, muted: false }; });
              setStemVolumes(vols);
              setSepProgress(100);
              setSepStage(`Done — ${stems.length} stems`);
              resolve(stems);
            } else if (data.type === 'error') {
              clearTimeout(timeout);
              es.close();
              reject(new Error(data.message || 'SuperSep error'));
            }
          } catch { /* ignore */ }
        };
        es.onerror = () => { clearTimeout(timeout); es.close(); reject(new Error('SSE lost')); };
      });
    } catch (err: any) {
      showToast(`SuperSep failed: ${err.message}`);
      throw err;
    } finally {
      setIsSeparating(false);
    }
  };

  const recombineStems = async (): Promise<string> => {
    const stemsPayload = superSepStems.map(s => ({
      path: s.file_path,
      volume: stemVolumes[s.id]?.volume ?? 1.0,
      muted: stemVolumes[s.id]?.muted ?? false,
    }));
    const resp = await fetch(`${PYTHON_API}/v1/supersep/recombine`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stems: stemsPayload }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || 'Recombine failed');
    }
    const { mixed_path } = await resp.json();
    return mixed_path;
  };

  // ── Generation (mirrors Lyric Studio's useAudioGeneration flow) ────────

  const handleGenerate = async () => {
    if (!token) { showToast('Please sign in first'); return; }
    if (!sourceAudioUrl) { showToast('Upload a source audio file first'); return; }
    if (!lyrics.trim()) { showToast('Lyrics are required for cover generation'); return; }

    setIsGenerating(true);
    try {
      const selectedArtist = artists.find(a => a.id === selectedArtistId);

      // 0) Advanced Mode: recombine stems with adjusted volumes
      let effectiveSourceUrl = sourceAudioUrl;
      if (advancedMode && superSepStems.length > 0) {
        setGenStage('Recombining stems...');
        setGenProgress(5);
        try {
          effectiveSourceUrl = await recombineStems();
          console.log('[CoverStudio] Using recombined source:', effectiveSourceUrl);
        } catch (err: any) {
          showToast(`Stem recombine failed: ${err.message}. Using full mix.`);
        }
      }

      // 1) Build base params — BPM and key are the TARGET values (after adjustments)
      const sourceBpm = analysis?.bpm || 120;
      const sourceKey = analysis?.key || 'C major';
      const targetBpm = Math.round(sourceBpm * tempoScale);
      const targetKey = pitchShift !== 0 ? transposeKey(sourceKey, pitchShift) : sourceKey;

      // Derive trigger word from adapter filename (not full artist name)
      const adapterPath = selectedPreset?.adapter_path || '';
      const triggerWord = adapterPath
        ? adapterPath.replace(/\\/g, '/').split('/').pop()?.replace(/\.safetensors$/i, '') || ''
        : '';

      const params: Record<string, any> = {
        customMode: true,
        lyrics,
        // Style = artist caption (trigger word gets prepended by applyTriggerWord)
        style: artistCaption || '',
        title: `${songTitle || 'Cover'} (${selectedArtist?.name || 'Cover'})`,
        taskType: 'cover',
        sourceAudioUrl: effectiveSourceUrl,
        audioCoverStrength,
        coverNoiseStrength,
        bpm: targetBpm,
        keyScale: targetKey,
        duration: 0,
        instrumental: false,
        coverArtSubject: songTitle || 'cover',
        source: 'cover-studio',
        artistName: selectedArtist?.name || songArtist || '',
        sourceArtist: songArtist || '',
      };

      // Tempo/pitch params for source audio modification
      if (tempoScale !== 1.0) params.tempoScale = tempoScale;
      if (pitchShift !== 0) params.pitchShift = pitchShift;

      // 2) Merge persisted CreatePanel settings (scheduler, steps, guidance, etc.)
      mergeCreatePanelSettings(params);

      // 3) Load adapter from album preset (matches Lyric Studio flow exactly)
      if (selectedPreset?.adapter_path) {
        const scaleOverride = getGlobalScaleOverride();
        const effectiveScale = scaleOverride.enabled ? scaleOverride.overallScale : (selectedPreset.adapter_scale ?? 1.0);
        const effectiveGroupScales = scaleOverride.enabled ? scaleOverride.groupScales : selectedPreset.adapter_group_scales;

        try {
          const loraStatus = await generateApi.getLoraStatus(token);
          const existingSlot = loraStatus?.advanced?.slots?.find(
            (s: any) => s.path === selectedPreset.adapter_path
          );

          if (existingSlot) {
            console.log('[CoverStudio] Adapter already loaded, skipping reload');
            params.loraLoaded = true;
            params.loraPath = selectedPreset.adapter_path;
            params.loraScale = effectiveScale;
            if (effectiveGroupScales) {
              try {
                await generateApi.setSlotGroupScales({
                  slot: existingSlot.slot,
                  ...effectiveGroupScales,
                }, token);
              } catch (gsErr) {
                console.warn('[CoverStudio] Failed to apply group scales:', gsErr);
              }
            }
          } else {
            if (loraStatus?.advanced?.slots && loraStatus.advanced.slots.length > 0) {
              showToast('Switching adapter...');
              await generateApi.unloadLora(token);
            } else {
              showToast('Loading adapter...');
            }
            const loadPayload: any = {
              lora_path: selectedPreset.adapter_path,
              slot: 0,
              scale: effectiveScale,
              ...(effectiveGroupScales ? { group_scales: effectiveGroupScales } : {}),
            };
            console.log('[CoverStudio] Loading adapter:', JSON.stringify(loadPayload));
            await generateApi.loadLora(loadPayload, token);
            params.loraLoaded = true;
            params.loraPath = selectedPreset.adapter_path;
            params.loraScale = effectiveScale;
          }
        } catch (loadErr) {
          console.warn('[CoverStudio] Adapter load failed, continuing without:', loadErr);
          showToast('Warning: adapter failed to load');
        }

        // 4) Apply trigger word from adapter filename
        applyTriggerWord(params, selectedPreset.adapter_path);
      }

      // 5) Reference track + matchering (matches Lyric Studio flow)
      if (selectedPreset?.reference_track_path) {
        params.referenceAudioUrl = selectedPreset.reference_track_path;
        params.audioCoverStrength = selectedPreset.audio_cover_strength ?? audioCoverStrength;
        params.autoMaster = true;
        params.masteringParams = { mode: 'matchering', reference_file: selectedPreset.reference_track_path };
      }

      // 6) Start generation
      console.log('[CoverStudio] Starting generation:', JSON.stringify({
        title: params.title,
        taskType: params.taskType,
        source: params.source,
        loraPath: params.loraPath,
        referenceAudioUrl: params.referenceAudioUrl,
        matchering: !!params.masteringParams,
      }));
      const res = await generateApi.startGeneration(params as any, token);
      const jobId = res.jobId || (res as any).job_id;
      showToast(`Cover generation started! Job: ${jobId}`);
      setActiveJobId(jobId);

      // 7) Poll for completion
      if (jobId) {
        pollJob(jobId);
      }
    } catch (err: any) {
      showToast(`Generation failed: ${err.message}`);
      setIsGenerating(false);
    }
  };

  // ── Job polling ────────────────────────────────────────────────────────

  const pollJob = (jobId: string) => {
    if (!token) return;
    setGenProgress(0);
    setGenStage('Queued...');
    const interval = setInterval(async () => {
      try {
        const status = await generateApi.getStatus(jobId, token!);
        // Update progress
        if (status.progress != null) setGenProgress(Math.round(status.progress * 100));
        if (status.stage) setGenStage(status.stage);
        else if (status.status === 'running') setGenStage('Generating...');
        else if (status.status === 'queued') setGenStage('Queued...');

        if (status.status === 'succeeded') {
          clearInterval(interval);
          setGenProgress(100);
          setGenStage('Complete!');
          setIsGenerating(false);
          setActiveJobId(null);
          setRefreshTrigger(prev => prev + 1);
          showToast('Cover generated successfully!');
          // Reset progress after a moment
          setTimeout(() => { setGenProgress(0); setGenStage(''); }, 3000);
        } else if (status.status === 'failed') {
          clearInterval(interval);
          setGenProgress(0);
          setGenStage('');
          setIsGenerating(false);
          setActiveJobId(null);
          showToast(`Generation failed: ${status.error || 'Unknown error'}`);
        }
      } catch {
        // Transient poll error — keep polling
      }
    }, 2000);

    // Safety timeout: 30 minutes
    setTimeout(() => {
      clearInterval(interval);
      if (isGenerating) {
        setIsGenerating(false);
        setActiveJobId(null);
        setGenProgress(0);
        setGenStage('');
      }
    }, 1_800_000);
  };

  // ── Drag & Drop ────────────────────────────────────────────────────────

  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && /\.(mp3|wav|flac|ogg|m4a|opus)$/i.test(file.name)) {
      handleFileSelected(file);
    } else {
      showToast('Unsupported file format');
    }
  }, [token]);

  const selectedArtist = artists.find(a => a.id === selectedArtistId);
  const canGenerate = sourceAudioUrl && lyrics.trim() && !isGenerating;

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col w-full h-full bg-zinc-50 dark:bg-suno-panel overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-zinc-200 dark:border-white/5">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-500 to-teal-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Guitar className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-zinc-900 dark:text-white">Cover Studio</h1>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">Create AI covers of existing songs</p>
          </div>
        </div>
      </div>

      {/* Three-panel layout */}
      <div className="flex-1 flex overflow-hidden">

        {/* ── LEFT PANEL: Source Audio + Song Details ── */}
        <div className="w-[320px] flex-shrink-0 border-r border-zinc-200 dark:border-white/5 overflow-y-auto scrollbar-hide p-4 space-y-4">

          {/* Source Audio Upload */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
              <Upload className="w-4 h-4 text-cyan-400" />
              Source Audio
            </div>

            <div
              onDrop={handleDrop}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              className={`
                relative rounded-xl border-2 border-dashed transition-all duration-200 cursor-pointer
                ${isDragging
                  ? 'border-cyan-400 bg-cyan-500/10'
                  : (sourceFileName || sourceAudioUrl)
                    ? 'border-cyan-500/30 bg-cyan-500/5'
                    : 'border-zinc-300 dark:border-white/10 hover:border-cyan-400 hover:bg-cyan-500/5'
                }
              `}
            >
              <input
                type="file"
                accept=".mp3,.wav,.flac,.ogg,.m4a,.opus"
                className="absolute inset-0 opacity-0 cursor-pointer"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFileSelected(file);
                }}
              />
              <div className="p-6 text-center">
                {isUploading ? (
                  <div className="flex flex-col items-center gap-2">
                    <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
                    <span className="text-xs text-zinc-500">Uploading & analyzing...</span>
                  </div>
                ) : (sourceFileName || sourceAudioUrl) ? (
                  <div className="flex flex-col items-center gap-2">
                    <Music className="w-8 h-8 text-cyan-400" />
                    <span className="text-xs font-medium text-zinc-700 dark:text-zinc-300 truncate max-w-full">
                      {sourceFileName || 'Source loaded'}
                    </span>
                    {metadata?.duration && (
                      <span className="text-[10px] text-zinc-500">
                        {Math.floor(metadata.duration / 60)}:{String(Math.floor(metadata.duration % 60)).padStart(2, '0')}
                      </span>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-2">
                    <Upload className="w-8 h-8 text-zinc-400" />
                    <span className="text-xs text-zinc-500">Drop audio file here or click to browse</span>
                    <span className="text-[10px] text-zinc-400">MP3, WAV, FLAC, OGG</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Audio Analysis */}
          {(isAnalyzing || analysis) && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                <Disc3 className="w-4 h-4 text-amber-400" />
                Audio Analysis
              </div>
              {isAnalyzing ? (
                <div className="flex items-center gap-2 text-xs text-zinc-500">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Analyzing with Essentia...
                </div>
              ) : analysis && (
                <div className="flex gap-2">
                  <div className="flex-1 rounded-lg bg-black/5 dark:bg-white/5 px-3 py-2 text-center">
                    <div className="text-[10px] font-medium text-zinc-500 uppercase">BPM</div>
                    <div className="text-lg font-bold text-cyan-400">{Math.round(analysis.bpm)}</div>
                  </div>
                  <div className="flex-1 rounded-lg bg-black/5 dark:bg-white/5 px-3 py-2 text-center">
                    <div className="text-[10px] font-medium text-zinc-500 uppercase">Key</div>
                    <div className="text-lg font-bold text-amber-400">{analysis.key}</div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Advanced Mode (SuperSep) */}
          {sourceAudioUrl && (
            <div className="space-y-3">
              {/* Toggle + Level Dropdown */}
              <div className="flex items-center justify-between">
                <div
                  onClick={() => setAdvancedMode(v => !v)}
                  className="flex items-center gap-2 cursor-pointer group select-none"
                >
                  <div className={`relative w-9 h-5 rounded-full transition-colors duration-200 ${
                    advancedMode ? 'bg-pink-500' : 'bg-zinc-300 dark:bg-zinc-700'
                  }`}>
                    <div className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200 ${
                      advancedMode ? 'translate-x-4' : 'translate-x-0'
                    }`} />
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Sliders className="w-4 h-4 text-pink-400" />
                    <span className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 group-hover:text-pink-400 transition-colors">
                      Advanced Mode
                    </span>
                  </div>
                </div>
              </div>

              {advancedMode && (
                <div className="space-y-3">
                  {/* Separation Level Dropdown */}
                  <div className="flex items-center gap-2">
                    <label className="text-[10px] font-medium text-zinc-500 uppercase whitespace-nowrap">Separation</label>
                    <div className="relative flex-1">
                      <select
                        value={sepLevel}
                        onChange={e => setSepLevel(e.target.value)}
                        disabled={isSeparating}
                        className="w-full appearance-none rounded-lg bg-black/5 dark:bg-white/5 border border-zinc-200 dark:border-zinc-700 px-3 py-1.5 pr-8 text-xs text-zinc-700 dark:text-zinc-300 cursor-pointer disabled:opacity-50 focus:ring-2 focus:ring-pink-500/50 focus:outline-none"
                      >
                        <option value="basic">Basic — 6 stems (fastest)</option>
                        <option value="vocal-split">Vocal Split — 8 stems</option>
                        <option value="full">Full — 14 stems</option>
                        <option value="maximum">Maximum — 17 stems</option>
                      </select>
                      <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 text-zinc-400 pointer-events-none" />
                    </div>
                    <button
                      onClick={() => sourceAudioUrl && runSuperSep(sourceAudioUrl, sepLevel)}
                      disabled={isSeparating || !sourceAudioUrl}
                      className="px-3 py-1.5 rounded-lg bg-pink-500 hover:bg-pink-600 disabled:bg-zinc-600 disabled:cursor-not-allowed text-white text-xs font-semibold transition-colors whitespace-nowrap"
                    >
                      {isSeparating ? 'Separating...' : 'Split'}
                    </button>
                  </div>

                  {/* Progress Bar */}
                  {isSeparating && (
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 text-xs text-zinc-500">
                        <Loader2 className="w-3 h-3 animate-spin text-pink-400" />
                        <span className="truncate">{sepStage}</span>
                        <span className="ml-auto font-mono text-pink-400">{sepProgress}%</span>
                      </div>
                      <div className="w-full h-1.5 rounded-full bg-zinc-200 dark:bg-zinc-700 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-pink-500 to-purple-500 transition-all duration-300"
                          style={{ width: `${sepProgress}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Stem Mixer */}
                  {superSepStems.length > 0 && !isSeparating && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-medium text-zinc-500 uppercase">Stem Mixer ({superSepStems.length} stems)</span>
                        <button
                          onClick={() => {
                            const vols: Record<string, { volume: number; muted: boolean }> = {};
                            superSepStems.forEach((s: any) => { vols[s.id] = { volume: 1.0, muted: false }; });
                            setStemVolumes(vols);
                          }}
                          className="text-[10px] text-zinc-400 hover:text-pink-400 transition-colors flex items-center gap-1"
                        >
                          <RotateCcw className="w-3 h-3" /> Reset
                        </button>
                      </div>

                      {/* Group stems by category */}
                      {(['vocals', 'instruments', 'drums', 'other'] as const).map(cat => {
                        const catStems = superSepStems.filter((s: any) => s.category === cat);
                        if (catStems.length === 0) return null;
                        const catLabel = cat === 'vocals' ? '🎤 Vocals' : cat === 'instruments' ? '🎸 Instruments' : cat === 'drums' ? '🥁 Drums' : '✨ Other';
                        return (
                          <div key={cat} className="space-y-1">
                            <div className="text-[9px] font-bold text-zinc-400 uppercase tracking-wider">{catLabel}</div>
                            {catStems.map((stem: any) => {
                              const sv = stemVolumes[stem.id] || { volume: 1.0, muted: false };
                              return (
                                <div key={stem.id} className="flex items-center gap-2 py-0.5">
                                  <button
                                    onClick={() => setStemVolumes(prev => ({ ...prev, [stem.id]: { ...sv, muted: !sv.muted } }))}
                                    className={`flex-shrink-0 p-0.5 rounded transition-colors ${
                                      sv.muted ? 'text-red-400 hover:text-red-300' : 'text-zinc-400 hover:text-zinc-300'
                                    }`}
                                    title={sv.muted ? 'Unmute' : 'Mute'}
                                  >
                                    {sv.muted ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
                                  </button>
                                  <span className={`text-[10px] w-24 truncate ${
                                    sv.muted ? 'text-zinc-500 line-through' : 'text-zinc-300'
                                  }`} title={stem.stem_name}>
                                    {stem.stem_name}
                                  </span>
                                  <input
                                    type="range"
                                    min={0} max={200} step={1}
                                    value={Math.round(sv.volume * 100)}
                                    onChange={e => setStemVolumes(prev => ({
                                      ...prev,
                                      [stem.id]: { ...sv, volume: parseInt(e.target.value) / 100 }
                                    }))}
                                    disabled={sv.muted}
                                    className="flex-1 h-1 accent-pink-500 disabled:opacity-30"
                                  />
                                  <span className={`text-[9px] font-mono w-8 text-right ${
                                    sv.muted ? 'text-red-400' : sv.volume > 1 ? 'text-amber-400' : 'text-zinc-400'
                                  }`}>
                                    {sv.muted ? 'OFF' : `${Math.round(sv.volume * 100)}%`}
                                  </span>
                                </div>
                              );
                            })}
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {/* Real-time Preview Playback (multi-track) */}
                  {superSepStems.length > 0 && !isSeparating && (
                    <div className="flex items-center gap-2 pt-1">
                      <button
                        onClick={async () => {
                          if (isPreviewPlaying) {
                            // Stop all
                            stemAudiosRef.current.forEach(({ el }) => {
                              el.pause();
                              el.currentTime = 0;
                            });
                            setIsPreviewPlaying(false);
                            return;
                          }
                          if (isPreviewLoading) return;

                          // Create AudioContext if needed
                          if (!audioCtxRef.current) {
                            audioCtxRef.current = new AudioContext();
                          }
                          const ctx = audioCtxRef.current;
                          if (ctx.state === 'suspended') await ctx.resume();

                          // Tear down old nodes
                          stemAudiosRef.current.forEach(({ el, source }) => {
                            el.pause();
                            try { source.disconnect(); } catch {}
                          });
                          stemAudiosRef.current = [];

                          setIsPreviewLoading(true);

                          // Create all audio elements and wait for them to buffer
                          const nodes: typeof stemAudiosRef.current = [];
                          const bufferPromises: Promise<void>[] = [];

                          superSepStems.forEach((stem: any) => {
                            const sv = stemVolumes[stem.id] || { volume: 1.0, muted: false };
                            const url = `${PYTHON_API}/v1/supersep/serve?path=${encodeURIComponent(stem.file_path)}`;
                            const el = new Audio(url);
                            el.crossOrigin = 'anonymous';
                            el.preload = 'auto';
                            const source = ctx.createMediaElementSource(el);
                            const gain = ctx.createGain();
                            gain.gain.value = sv.muted ? 0 : sv.volume;
                            source.connect(gain).connect(ctx.destination);
                            nodes.push({ el, gain, source });

                            // Wait for this track to be fully buffered
                            bufferPromises.push(new Promise<void>((resolve) => {
                              if (el.readyState >= 4) { resolve(); return; }
                              el.addEventListener('canplaythrough', () => resolve(), { once: true });
                              el.addEventListener('error', () => resolve(), { once: true });
                              el.load();
                            }));
                          });

                          // Wait for ALL stems to buffer
                          await Promise.all(bufferPromises);
                          stemAudiosRef.current = nodes;
                          setIsPreviewLoading(false);

                          // Track end events
                          let endedCount = 0;
                          const totalActive = nodes.filter((_, i) => {
                            const stem = superSepStems[i];
                            return stem && !stemVolumes[stem.id]?.muted;
                          }).length;
                          nodes.forEach(({ el }) => {
                            el.onended = () => {
                              endedCount++;
                              if (endedCount >= Math.max(totalActive, 1)) setIsPreviewPlaying(false);
                            };
                          });

                          // Play all at exactly the same moment
                          nodes.forEach(({ el }) => el.play().catch(() => {}));
                          setIsPreviewPlaying(true);
                        }}
                        disabled={isPreviewLoading}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                          isPreviewPlaying
                            ? 'bg-pink-500 text-white hover:bg-pink-600'
                            : isPreviewLoading
                              ? 'bg-amber-500/20 text-amber-400'
                              : 'bg-black/5 dark:bg-white/5 text-zinc-600 dark:text-zinc-300 hover:bg-pink-500/20 hover:text-pink-400'
                        } disabled:cursor-wait`}
                      >
                        {isPreviewLoading ? (
                          <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Buffering...</>
                        ) : isPreviewPlaying ? (
                          <><Pause className="w-3.5 h-3.5" /> Stop Preview</>
                        ) : (
                          <><Play className="w-3.5 h-3.5" /> Preview Mix</>
                        )}
                      </button>
                      <span className="text-[9px] text-zinc-500">
                        {isPreviewLoading ? 'Pre-loading all stems...' : 'Real-time multi-track'}
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Song Details */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
              <Search className="w-4 h-4 text-purple-400" />
              Song Details
            </div>
            <div className="space-y-2">
              <div>
                <label className="text-[10px] font-medium text-zinc-500 uppercase">Artist</label>
                <input
                  type="text"
                  value={songArtist}
                  onChange={e => setSongArtist(e.target.value)}
                  placeholder="Song artist name"
                  className="w-full mt-1 bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg px-3 py-2 text-sm text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-cyan-500 transition-colors"
                />
              </div>
              <div>
                <label className="text-[10px] font-medium text-zinc-500 uppercase">Title</label>
                <input
                  type="text"
                  value={songTitle}
                  onChange={e => setSongTitle(e.target.value)}
                  placeholder="Song title"
                  className="w-full mt-1 bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg px-3 py-2 text-sm text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-cyan-500 transition-colors"
                />
              </div>
              <button
                onClick={handleSearchLyrics}
                disabled={isSearchingLyrics || !songArtist.trim() || !songTitle.trim()}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-purple-600 to-violet-600 hover:from-purple-500 hover:to-violet-500 text-white text-sm font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-purple-500/10"
              >
                {isSearchingLyrics ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /> Searching Genius...</>
                ) : (
                  <><Search className="w-4 h-4" /> Search Genius for Lyrics</>
                )}
              </button>
            </div>
          </div>

          {/* Filename Prepend */}
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
              <Download className="w-4 h-4 text-orange-400" />
              Download Settings
            </div>
            <div>
              <label className="text-[10px] font-medium text-zinc-500 uppercase">Filename Prepend</label>
              <input
                type="text"
                value={filenamePrepend}
                onChange={e => setFilenamePrepend(e.target.value)}
                placeholder="e.g. 01 - , My Album - "
                className="w-full mt-1 bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg px-3 py-2 text-sm text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-orange-500 transition-colors"
              />
              <p className="text-[9px] text-zinc-500 mt-1">Prepended to download filenames</p>
            </div>
          </div>
        </div>

        {/* ── CENTER PANEL: Lyrics Editor ── */}
        <div className="flex-1 flex flex-col overflow-hidden border-r border-zinc-200 dark:border-white/5">
          <div className="flex-shrink-0 px-4 py-3 border-b border-zinc-200 dark:border-white/5">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Lyrics</span>
              {lyrics && (
                <span className="text-[10px] text-zinc-500">{lyrics.split('\n').length} lines</span>
              )}
            </div>
          </div>
          <div className="flex-1 p-4">
            <textarea
              value={lyrics}
              onChange={e => setLyrics(e.target.value)}
              placeholder="Lyrics will appear here after searching Genius, or paste them manually..."
              className="w-full h-full resize-none bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-xl px-4 py-3 text-sm text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-cyan-500 transition-colors font-mono leading-relaxed"
            />
          </div>
        </div>

        {/* ── RIGHT PANEL: Artist Selector + Settings + Generate ── */}
        <div className="w-[640px] flex-shrink-0 overflow-y-auto scrollbar-hide p-4 space-y-4">

          {/* Target Artist */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
              <Guitar className="w-4 h-4 text-cyan-400" />
              Cover As Artist
            </div>

            {isLoadingArtists ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-5 h-5 text-zinc-500 animate-spin" />
              </div>
            ) : artists.length === 0 ? (
              <div className="text-center py-6">
                <p className="text-xs text-zinc-500">No artists in Lyric Studio yet.</p>
                <p className="text-[10px] text-zinc-400 mt-1">Add artists in Lyric Studio first.</p>
              </div>
            ) : (
              <div className="grid grid-cols-5 gap-2 max-h-[240px] overflow-y-auto scrollbar-hide">
                {artists.map(artist => (
                  <button
                    key={artist.id}
                    onClick={() => handleSelectArtist(artist)}
                    className={`
                      flex flex-col items-center gap-1.5 p-2 rounded-xl transition-all duration-200
                      ${selectedArtistId === artist.id
                        ? 'bg-cyan-500/20 ring-2 ring-cyan-400 shadow-lg shadow-cyan-500/10'
                        : 'bg-black/5 dark:bg-white/5 hover:bg-cyan-500/10 hover:ring-1 hover:ring-cyan-400/50'
                      }
                    `}
                  >
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-500 to-teal-600 flex items-center justify-center overflow-hidden flex-shrink-0">
                      {artist.image_url ? (
                        <img src={artist.image_url} alt={artist.name} className="w-full h-full object-cover" />
                      ) : (
                        <span className="text-white text-sm font-bold">{artist.name.charAt(0)}</span>
                      )}
                    </div>
                    <span className="text-[10px] font-medium text-zinc-700 dark:text-zinc-300 truncate w-full text-center">
                      {artist.name}
                    </span>
                  </button>
                ))}
              </div>
            )}

            {/* Album selector (when artist has multiple albums) */}
            {(() => {
              const presetsWithAdapters = artistPresets.filter(p => p.preset?.adapter_path);
              if (presetsWithAdapters.length <= 1) return null;
              return (
                <div className="flex items-center gap-2">
                  <label className="text-[10px] font-medium text-zinc-500 uppercase whitespace-nowrap">Album</label>
                  <div className="relative flex-1">
                    <select
                      value={artistPresets.findIndex(p => p.preset === selectedPreset)}
                      onChange={e => {
                        const idx = parseInt(e.target.value);
                        const chosen = artistPresets[idx];
                        if (chosen?.preset) {
                          setSelectedPreset(chosen.preset);
                          if (chosen.preset.audio_cover_strength != null) {
                            setAudioCoverStrength(chosen.preset.audio_cover_strength);
                          }
                        }
                      }}
                      className="w-full appearance-none rounded-lg bg-black/5 dark:bg-white/5 border border-zinc-200 dark:border-zinc-700 px-3 py-1.5 pr-8 text-xs text-zinc-700 dark:text-zinc-300 cursor-pointer focus:ring-2 focus:ring-cyan-500/50 focus:outline-none"
                    >
                      {artistPresets.filter(p => p.preset?.adapter_path).map(p => {
                        const idx = artistPresets.indexOf(p);
                        return (
                          <option key={p.lsId} value={idx}>
                            {p.album}
                          </option>
                        );
                      })}
                    </select>
                    <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 text-zinc-400 pointer-events-none" />
                  </div>
                </div>
              );
            })()}

            {/* Selected preset info */}
            {selectedPreset && (
              <div className="rounded-lg bg-cyan-500/5 border border-cyan-500/20 px-3 py-2 space-y-1">
                {selectedPreset.adapter_path && (
                  <div className="flex items-center gap-2">
                    <Zap className="w-3 h-3 text-pink-400" />
                    <span className="text-[10px] text-zinc-500 truncate flex-1">
                      {selectedPreset.adapter_path.split(/[\\/]/).pop()}
                    </span>
                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-pink-900/30 text-pink-400">
                      ADAPTER
                    </span>
                  </div>
                )}
                {selectedPreset.reference_track_path && (
                  <div className="flex items-center gap-2">
                    <Music className="w-3 h-3 text-amber-400" />
                    <span className="text-[10px] text-zinc-500 truncate flex-1">
                      {selectedPreset.reference_track_path.split(/[\\/]/).pop()}
                    </span>
                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-900/30 text-amber-400">
                      REF + MATCHERING
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Divider */}
          <div className="border-t border-zinc-200 dark:border-white/5" />

          {/* Cover Settings */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
              <Disc3 className="w-4 h-4 text-teal-400" />
              Cover Settings
            </div>

            <EditableSlider
              label="Structure Fidelity"
              value={audioCoverStrength}
              min={0} max={1} step={0.05}
              onChange={setAudioCoverStrength}
              formatDisplay={v => v.toFixed(2)}
              helpText="How closely the output follows the source song's arrangement. Higher = more faithful structure, lower = more creative freedom"
            />
            <EditableSlider
              label="Source Timbre"
              value={coverNoiseStrength}
              min={0} max={1} step={0.05}
              onChange={setCoverNoiseStrength}
              formatDisplay={v => v.toFixed(2)}
              helpText="How much of the original artist's sound character is preserved. Higher = more original artist, lower = more adapter artist"
            />


            <EditableSlider
              label="Tempo Scale"
              value={tempoScale}
              min={0.5} max={2.0} step={0.05}
              onChange={setTempoScale}
              formatDisplay={v => {
                const bpm = analysis?.bpm;
                return bpm ? `${v.toFixed(2)}x (${Math.round(bpm * v)} BPM)` : `${v.toFixed(2)}x`;
              }}
              helpText={`1.0 = original tempo${analysis?.bpm ? ` (${Math.round(analysis.bpm)} BPM)` : ''}, <1 = slower, >1 = faster`}
            />
            <EditableSlider
              label="Pitch Shift"
              value={pitchShift}
              min={-12} max={12} step={1}
              onChange={setPitchShift}
              formatDisplay={v => {
                const shifted = analysis?.key ? transposeKey(analysis.key, v) : null;
                const sign = v > 0 ? '+' : '';
                return shifted && v !== 0 ? `${sign}${v} st → ${shifted}` : `${sign}${v} st`;
              }}
              helpText={`Semitones to shift (-12 to +12)${analysis?.key ? `. Source: ${analysis.key}` : ''}`}
            />
          </div>

          {/* Divider */}
          <div className="border-t border-zinc-200 dark:border-white/5" />

          {/* Generate Button */}
          <button
            onClick={handleGenerate}
            disabled={!canGenerate}
            className={`
              w-full flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl text-sm font-bold transition-all duration-300 shadow-lg
              ${canGenerate
                ? 'bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-white shadow-cyan-500/20 hover:shadow-cyan-400/30 hover:scale-[1.02]'
                : 'bg-zinc-200 dark:bg-white/5 text-zinc-400 cursor-not-allowed shadow-none'
              }
            `}
          >
            {isGenerating ? (
              <><Loader2 className="w-5 h-5 animate-spin" /> Generating Cover...
              {genProgress > 0 && <span className="text-xs opacity-70">({genProgress}%)</span>}
              </>
            ) : (
              <><Guitar className="w-5 h-5" /> Generate Cover</>
            )}
          </button>

          {/* Progress bar */}
          {isGenerating && (
            <div className="space-y-1.5">
              <div className="w-full bg-black/20 rounded-full h-2 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan-500 to-teal-400 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${Math.max(genProgress, 2)}%` }}
                />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-zinc-500">{genStage || 'Starting...'}</span>
                <span className="text-[10px] text-zinc-500 font-mono">{genProgress}%</span>
              </div>
            </div>
          )}

          {/* Readiness checklist */}
          {!canGenerate && !isGenerating && (
            <div className="space-y-1.5">
              {!sourceAudioUrl && (
                <div className="flex items-center gap-2 text-[10px] text-zinc-500">
                  <div className="w-1.5 h-1.5 rounded-full bg-red-400" />
                  Upload source audio
                </div>
              )}
              {!lyrics.trim() && (
                <div className="flex items-center gap-2 text-[10px] text-zinc-500">
                  <div className="w-1.5 h-1.5 rounded-full bg-red-400" />
                  Add lyrics (search Genius or paste)
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── FOURTH PANEL: Recent Covers ── */}
        <div className="w-[320px] flex-shrink-0 border-l border-zinc-200 dark:border-white/5 overflow-y-auto scrollbar-hide p-4">
          <RecentCovers
            onPlaySong={onPlaySong}
            currentSong={currentSong}
            isPlaying={isPlaying}
            refreshTrigger={refreshTrigger}
          />
        </div>
      </div>

      {/* Toast */}
      {toast && (
        <div className="fixed bottom-20 left-1/2 -translate-x-1/2 z-50 px-4 py-2 rounded-xl bg-zinc-800 text-white text-sm font-medium shadow-2xl border border-white/10 animate-in slide-in-from-bottom-4 fade-in duration-300">
          {toast}
        </div>
      )}
    </div>
  );
};

// ── Recent Covers Sub-component ──────────────────────────────────────────────

interface RecentCoversProps {
  onPlaySong: (song: Song, list?: Song[]) => void;
  currentSong: Song | null;
  isPlaying: boolean;
  refreshTrigger: number;
}

const RecentCovers: React.FC<RecentCoversProps> = ({ onPlaySong, currentSong, isPlaying, refreshTrigger }) => {
  const { token } = useAuth();
  const [covers, setCovers] = useState<Song[]>([]);
  const [loading, setLoading] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [downloadSong, setDownloadSong] = useState<Song | null>(null);
  const [downloadModalOpen, setDownloadModalOpen] = useState(false);

  useEffect(() => {
    if (!token) return;
    setLoading(true);

    // Use source=cover-studio query param to get only cover studio songs
    fetch('/api/songs?source=cover-studio', {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(res => res.json())
      .then(data => {
        const allSongs = data.songs || [];
        const coverSongs = allSongs
          .map((s: any): Song => ({
            id: s.id,
            title: s.title || 'Untitled Cover',
            lyrics: s.lyrics || '',
            style: s.style || '',
            coverUrl: s.cover_url || '',
            duration: s.duration && s.duration > 0
              ? `${Math.floor(s.duration / 60)}:${String(Math.floor(s.duration % 60)).padStart(2, '0')}`
              : '0:00',
            createdAt: new Date(s.created_at),
            tags: s.tags || [],
            audioUrl: s.audio_url,
            isPublic: s.is_public,
            likeCount: s.like_count || 0,
            viewCount: s.view_count || 0,
            userId: s.user_id,
            creator: s.creator,
            generationParams: typeof s.generation_params === 'string'
              ? JSON.parse(s.generation_params || '{}')
              : (s.generation_params || {}),
          }))
          .sort((a: Song, b: Song) => b.createdAt.getTime() - a.createdAt.getTime())
          .slice(0, 20);
        setCovers(coverSongs);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [token, refreshTrigger]);

  // Clear all covers
  const handleClearAll = async () => {
    if (!token || covers.length === 0) return;
    if (!window.confirm(`Delete all ${covers.length} covers? This removes files from disk too.`)) return;
    setClearing(true);
    try {
      for (const cover of covers) {
        try { await songsApi.deleteSong(cover.id, token); } catch { /* continue */ }
      }
      setCovers([]);
    } finally {
      setClearing(false);
    }
  };

  // Download handler
  const handleDownloadClick = (e: React.MouseEvent, cover: Song) => {
    e.stopPropagation();
    setDownloadSong(cover);
    setDownloadModalOpen(true);
  };

  const handleDownload = (format: DownloadFormat, version: DownloadVersion) => {
    if (!downloadSong) return;
    const gp = (downloadSong as any).generationParams || {};
    const filenamePrepend = localStorage.getItem('cover-studio-filenamePrepend')?.replace(/^"|"$/g, '') || '';
    // Extract cover artist name and source artist
    const coverArtist = gp.artistName || '';
    // Title was stored as "SongTitle (ArtistName)" — extract the original song title
    const rawTitle = downloadSong.title || 'Untitled';
    const sourceArtist = gp.sourceArtist || localStorage.getItem('cover-studio-songArtist')?.replace(/^"|"$/g, '') || '';
    const rawDisplayTitle = `${filenamePrepend}${coverArtist ? coverArtist + ' - ' : ''}${rawTitle}${sourceArtist ? ` (${sourceArtist} Cover)` : ''}`;
    // Sanitize for filesystem: replace curly quotes with ASCII, strip illegal filename chars
    const displayTitle = rawDisplayTitle
      .replace(/[\u2018\u2019\u0060]/g, "'")   // curly single quotes → straight
      .replace(/[\u201C\u201D]/g, '"')           // curly double quotes → straight
      .replace(/[?*:<>|"]/g, '_')                // illegal filename chars → underscore
      .replace(/\//g, '-')                        // forward slash → dash
      .replace(/\\/g, '-');                       // backslash → dash

    const downloadSingleURL = (url: string, suffix: string) => {
      const targetUrl = new URL('/api/songs/download', window.location.origin);
      targetUrl.searchParams.set('audioUrl', url);
      targetUrl.searchParams.set('title', `${displayTitle}${suffix}`);
      targetUrl.searchParams.set('format', format);
      targetUrl.searchParams.set('songId', downloadSong.id);
      if (format === 'mp3') {
        const br = localStorage.getItem('mp3_export_bitrate');
        if (br) targetUrl.searchParams.set('mp3Bitrate', br);
      }
      if (format === 'opus') {
        const br = localStorage.getItem('opus_export_bitrate');
        if (br) targetUrl.searchParams.set('opusBitrate', br);
      }
      const link = document.createElement('a');
      link.href = targetUrl.toString();
      const ext = format === 'opus' ? 'ogg' : format;
      link.download = `${displayTitle}${suffix}.${ext}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };

    if (version === 'mastered' || version === 'both') {
      downloadSingleURL(downloadSong.audioUrl || '', '');
    }
    if (version === 'original' || version === 'both') {
      const origUrl = gp.originalAudioUrl;
      if (origUrl) {
        setTimeout(() => downloadSingleURL(origUrl, ' (Unmastered)'), version === 'both' ? 500 : 0);
      }
    }
  };

  const hasOriginal = downloadSong && (downloadSong as any).generationParams?.originalAudioUrl;

  return (
    <>
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
          <Music className="w-4 h-4 text-cyan-400" />
          Recent Covers
          {covers.length > 0 && <span className="text-[10px] text-zinc-500 font-normal">({covers.length})</span>}
        </div>
        {covers.length > 0 && (
          <button
            onClick={handleClearAll}
            disabled={clearing}
            className="p-1 rounded-md hover:bg-red-900/30 text-zinc-500 hover:text-red-400 transition-colors"
            title="Clear all covers"
          >
            {clearing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
          </button>
        )}
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-4">
          <Loader2 className="w-4 h-4 text-zinc-500 animate-spin" />
        </div>
      ) : covers.length === 0 ? (
        <p className="text-[10px] text-zinc-500 text-center py-4">No covers yet. Generate your first one!</p>
      ) : (
        <div className="space-y-1">
          {covers.map(cover => {
            const isCurrent = currentSong?.id === cover.id;
            return (
              <div
                key={cover.id}
                onClick={() => onPlaySong(cover, covers)}
                className={`
                  w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all duration-200 text-left cursor-pointer group relative
                  ${isCurrent
                    ? 'bg-cyan-500/20 ring-1 ring-cyan-400/50'
                    : 'hover:bg-white/5'
                  }
                `}
              >
                <div className="w-6 h-6 rounded-full bg-cyan-500/20 flex items-center justify-center flex-shrink-0">
                  {isCurrent && isPlaying ? (
                    <Pause className="w-3 h-3 text-cyan-400" />
                  ) : (
                    <Play className="w-3 h-3 text-cyan-400" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium text-zinc-700 dark:text-zinc-300 truncate">{cover.title}</div>
                  <div className="text-[10px] text-zinc-500">{cover.duration}</div>
                </div>
                {/* Hover actions */}
                <div className="absolute right-1 top-1/2 -translate-y-1/2 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={(e) => handleDownloadClick(e, cover)}
                    className="p-1.5 rounded-md bg-zinc-800/80 hover:bg-zinc-700 text-zinc-400 hover:text-white transition-colors"
                    title="Download"
                  >
                    <Download className="w-3 h-3" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>

    {/* Download Modal */}
    <DownloadModal
      isOpen={downloadModalOpen}
      onClose={() => { setDownloadModalOpen(false); setDownloadSong(null); }}
      onDownload={handleDownload}
      songTitle={downloadSong?.title}
      hasOriginal={!!hasOriginal}
    />
    </>
  );
};
