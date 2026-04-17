import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Upload, Music, Search, Play, Pause, Loader2, AlertCircle, Guitar, Disc3, Zap, ChevronDown, ChevronRight, X } from 'lucide-react';
import { lireekApi, Artist, AlbumPreset, LyricsSet } from '../../services/lyricStudioApi';
import { generateApi } from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { Song, GenerationParams } from '../../types';
import { EditableSlider } from '../EditableSlider';

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

// ── Main Component ───────────────────────────────────────────────────────────

export const CoverStudio: React.FC<CoverStudioProps> = ({
  onPlaySong,
  isPlaying,
  currentSong,
  currentTime,
}) => {
  const { token } = useAuth();

  // Source audio state
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [sourceAudioUrl, setSourceAudioUrl] = useState('');
  const [metadata, setMetadata] = useState<AudioMetadata | null>(null);
  const [analysis, setAnalysis] = useState<AudioAnalysis | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Song details state
  const [songArtist, setSongArtist] = useState('');
  const [songTitle, setSongTitle] = useState('');
  const [lyrics, setLyrics] = useState('');
  const [isSearchingLyrics, setIsSearchingLyrics] = useState(false);

  // Target artist state
  const [artists, setArtists] = useState<Artist[]>([]);
  const [selectedArtistId, setSelectedArtistId] = useState<number | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<AlbumPreset | null>(null);
  const [isLoadingArtists, setIsLoadingArtists] = useState(false);

  // Cover settings
  const [audioCoverStrength, setAudioCoverStrength] = useState(0.5);
  const [coverNoiseStrength, setCoverNoiseStrength] = useState(0);
  const [tempoScale, setTempoScale] = useState(1.0);
  const [pitchShift, setPitchShift] = useState(0);

  // Generation
  const [isGenerating, setIsGenerating] = useState(false);
  const [toast, setToast] = useState('');
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Load artists on mount
  useEffect(() => {
    setIsLoadingArtists(true);
    lireekApi.listArtists()
      .then(res => setArtists(res.artists))
      .catch(() => showToast('Failed to load artists'))
      .finally(() => setIsLoadingArtists(false));
  }, []);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(''), 4000);
  };

  // ── File Upload + Metadata + Analysis ──────────────────────────────────

  const handleFileSelected = async (file: File) => {
    if (!token) {
      showToast('Please sign in first');
      return;
    }

    setSourceFile(file);
    setIsUploading(true);

    try {
      // 1. Extract metadata
      const metaFormData = new FormData();
      metaFormData.append('audio', file);

      const metaRes = await fetch('/api/analyze/metadata', { method: 'POST', body: metaFormData });
      if (metaRes.ok) {
        const meta: AudioMetadata = await metaRes.json();
        setMetadata(meta);
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
      const analyzeRes = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audioUrl }),
      });
      if (analyzeRes.ok) {
        const analysisData = await analyzeRes.json();
        setAnalysis({
          bpm: analysisData.bpm || 120,
          key: `${analysisData.key || 'C'} ${analysisData.scale || 'major'}`,
          scale: analysisData.scale,
        });
      }
    } catch (err: any) {
      showToast(`Error: ${err.message}`);
    } finally {
      setIsUploading(false);
      setIsAnalyzing(false);
    }
  };

  // ── Genius Lyrics Search ───────────────────────────────────────────────

  const handleSearchLyrics = async () => {
    if (!songArtist.trim() || !songTitle.trim()) {
      showToast('Enter artist and title first');
      return;
    }
    setIsSearchingLyrics(true);
    try {
      const result = await lireekApi.searchSongLyrics(songArtist.trim(), songTitle.trim());
      setLyrics(result.lyrics);
      if (result.title) setSongTitle(result.title);
      showToast('Lyrics found!');
    } catch (err: any) {
      showToast(err.message || 'No lyrics found');
    } finally {
      setIsSearchingLyrics(false);
    }
  };

  // ── Artist Selection ───────────────────────────────────────────────────

  const handleSelectArtist = async (artist: Artist) => {
    setSelectedArtistId(artist.id);
    try {
      const { lyrics_sets } = await lireekApi.listLyricsSets(artist.id);
      if (lyrics_sets.length > 0) {
        const { preset } = await lireekApi.getPreset(lyrics_sets[0].id);
        setSelectedPreset(preset);
        if (preset?.audio_cover_strength != null) {
          setAudioCoverStrength(preset.audio_cover_strength);
        }
      } else {
        setSelectedPreset(null);
      }
    } catch {
      setSelectedPreset(null);
    }
  };

  // ── Generation ─────────────────────────────────────────────────────────

  const handleGenerate = async () => {
    if (!token) { showToast('Please sign in first'); return; }
    if (!sourceAudioUrl) { showToast('Upload a source audio file first'); return; }
    if (!lyrics.trim()) { showToast('Lyrics are required for cover generation'); return; }

    setIsGenerating(true);
    try {
      const selectedArtist = artists.find(a => a.id === selectedArtistId);

      const params: any = {
        customMode: true,
        lyrics,
        prompt: selectedArtist?.name || songArtist || 'cover',
        style: selectedArtist?.name || '',
        title: `${songTitle || 'Cover'} (${selectedArtist?.name || 'Cover'})`,
        taskType: 'cover',
        sourceAudioUrl,
        audioCoverStrength,
        coverNoiseStrength,
        bpm: analysis?.bpm || 120,
        keyScale: analysis?.key || 'C major',
        duration: 0,
        instrumental: false,
        vocalLanguage: 'en',
        inferenceSteps: 27,
        guidanceScale: 15,
        batchSize: 1,
        randomSeed: true,
        seed: -1,
        thinking: false,
        audioFormat: 'flac',
        inferMethod: 'euler',
        scheduler: 'linear',
        shift: 3.0,
        lmTemperature: 0.9,
        lmCfgScale: 1.0,
        lmTopK: 250,
        lmTopP: 0.95,
        lmNegativePrompt: '',
        loraPath: selectedPreset?.adapter_path,
        loraScale: selectedPreset?.adapter_scale || 1.0,
        referenceAudioUrl: selectedPreset?.reference_track_path,
        coverArtSubject: songTitle || 'cover',
        source: 'cover-studio',
      };

      // Apply tempo/pitch if changed from default
      if (tempoScale !== 1.0) params.tempoScale = tempoScale;
      if (pitchShift !== 0) params.pitchShift = pitchShift;

      const job = await generateApi.startGeneration(params, token);
      showToast(`Cover generation started! Job: ${job.jobId}`);
      setRefreshTrigger(prev => prev + 1);
    } catch (err: any) {
      showToast(`Generation failed: ${err.message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  // ── Drag & Drop ────────────────────────────────────────────────────────

  const [isDragging, setIsDragging] = useState(false);
  const dropRef = useRef<HTMLDivElement>(null);

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

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

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
              ref={dropRef}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={() => setIsDragging(false)}
              className={`
                relative rounded-xl border-2 border-dashed transition-all duration-200 cursor-pointer
                ${isDragging
                  ? 'border-cyan-400 bg-cyan-500/10'
                  : sourceFile
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
                ) : sourceFile ? (
                  <div className="flex flex-col items-center gap-2">
                    <Music className="w-8 h-8 text-cyan-400" />
                    <span className="text-xs font-medium text-zinc-700 dark:text-zinc-300 truncate max-w-full">{sourceFile.name}</span>
                    {metadata?.duration && (
                      <span className="text-[10px] text-zinc-500">{Math.floor(metadata.duration / 60)}:{String(Math.floor(metadata.duration % 60)).padStart(2, '0')}</span>
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
        <div className="w-[320px] flex-shrink-0 overflow-y-auto scrollbar-hide p-4 space-y-4">

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
              <div className="grid grid-cols-3 gap-2 max-h-[200px] overflow-y-auto scrollbar-hide">
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
                      REF
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
              label="Cover Strength"
              value={audioCoverStrength}
              min={0} max={1} step={0.05}
              onChange={setAudioCoverStrength}
              formatDisplay={v => v.toFixed(2)}
              helpText="How strongly the source audio influences the output structure"
            />
            <EditableSlider
              label="Noise Strength"
              value={coverNoiseStrength}
              min={0} max={1} step={0.05}
              onChange={setCoverNoiseStrength}
              formatDisplay={v => v.toFixed(2)}
              helpText="Amount of noise added to the cover for variation"
            />
            <EditableSlider
              label="Tempo Scale"
              value={tempoScale}
              min={0.5} max={2.0} step={0.05}
              onChange={setTempoScale}
              formatDisplay={v => `${v.toFixed(2)}x`}
              helpText="1.0 = original tempo, <1 = slower, >1 = faster"
            />
            <EditableSlider
              label="Pitch Shift"
              value={pitchShift}
              min={-12} max={12} step={1}
              onChange={setPitchShift}
              formatDisplay={v => `${v > 0 ? '+' : ''}${v} st`}
              helpText="Semitones to shift (-12 to +12)"
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
              <><Loader2 className="w-5 h-5 animate-spin" /> Generating Cover...</>
            ) : (
              <><Guitar className="w-5 h-5" /> Generate Cover</>
            )}
          </button>

          {/* Readiness checklist */}
          {!canGenerate && (
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
              {sourceAudioUrl && lyrics.trim() && (
                <div className="flex items-center gap-2 text-[10px] text-emerald-400">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                  Ready to generate
                </div>
              )}
            </div>
          )}

          {/* Divider */}
          <div className="border-t border-zinc-200 dark:border-white/5" />

          {/* Recent Covers */}
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

  useEffect(() => {
    if (!token) return;
    setLoading(true);

    fetch('/api/songs/mine', {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(res => res.json())
      .then(data => {
        const allSongs = data.songs || [];
        const coverSongs = allSongs
          .filter((s: any) => {
            const gp = typeof s.generationParams === 'string'
              ? JSON.parse(s.generationParams)
              : s.generationParams;
            return gp?.source === 'cover-studio';
          })
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
          }))
          .sort((a: Song, b: Song) => b.createdAt.getTime() - a.createdAt.getTime())
          .slice(0, 20);
        setCovers(coverSongs);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [token, refreshTrigger]);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
        <Music className="w-4 h-4 text-cyan-400" />
        Recent Covers
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
              <button
                key={cover.id}
                onClick={() => onPlaySong(cover, covers)}
                className={`
                  w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all duration-200 text-left
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
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};
