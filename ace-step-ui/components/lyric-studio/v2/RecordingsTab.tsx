import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Play, Trash2, Headphones, ChevronDown, ChevronRight, Loader2, Clock, X, Filter, Download, ListPlus, Check } from 'lucide-react';
import { lireekApi, Generation, AudioGeneration } from '../../../services/lyricStudioApi';
import { generateApi, songsApi } from '../../../services/api';
import { useAuth } from '../../../context/AuthContext';
import { Song } from '../../../types';
import { DownloadModal, DownloadFormat, DownloadVersion } from '../../DownloadModal';
import { usePlaylist } from './playlistStore';

interface SongGroup {
  generation: Generation;
  audioGens: AudioGeneration[];
  songs: Song[];  // resolved HOT-Step songs
}

interface RecordingsTabProps {
  generations: Generation[];
  onPlaySong: (song: Song) => void;
  showToast: (msg: string) => void;
  filterGenerationId?: number | null;
  onClearFilter?: () => void;
  onSongCountChange?: (count: number) => void;
  refreshKey?: number;
  artistName?: string;
}

export const RecordingsTab: React.FC<RecordingsTabProps> = ({
  generations, onPlaySong, showToast, filterGenerationId, onClearFilter, onSongCountChange, refreshKey = 0, artistName,
}) => {
  const { token } = useAuth();
  const [groups, setGroups] = useState<SongGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedGenId, setExpandedGenId] = useState<number | null>(null);
  const [localRefreshKey, setLocalRefreshKey] = useState(0);
  const [downloadSong, setDownloadSong] = useState<Song | null>(null);

  // Keep a ref to generations so the effect callback can read current data
  // without depending on the array reference
  const generationsRef = useRef(generations);
  generationsRef.current = generations;

  // Stable key: only re-fetch when the set of generation IDs, filter, or refreshKey changes
  const genKey = useMemo(() => {
    const ids = generations.map(g => g.id).sort().join(',');
    return `${ids}|${filterGenerationId ?? 'all'}|${refreshKey}|${localRefreshKey}`;
  }, [generations, filterGenerationId, refreshKey, localRefreshKey]);

  // Filtered view for rendering (cheap, no effect dependency)
  const filteredGenerations = useMemo(() =>
    filterGenerationId
      ? generations.filter(g => g.id === filterGenerationId)
      : generations,
    [generations, filterGenerationId]
  );

  // Load audio generations — depends on genKey (stable string) + token
  // No locks/guards — React deduplicates effects with identical string deps
  useEffect(() => {
    if (!token || genKey === '|all') {
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);

    const load = async () => {
      try {
        const gens = filterGenerationId
          ? generationsRef.current.filter(g => g.id === filterGenerationId)
          : generationsRef.current;

        console.log(`[RecordingsTab] Loading audio for ${gens.length} generations (key: ${genKey})`);

        // Get full job history for cross-referencing
        let jobHistory: Record<string, any> = {};
        try {
          const historyRes = await generateApi.getHistory(token);
          if (historyRes?.jobs) {
            for (const job of historyRes.jobs) {
              const id = job.jobId || job.id;
              if (id) jobHistory[id] = job;
            }
          }
          console.log(`[RecordingsTab] Job history has ${Object.keys(jobHistory).length} entries`);
        } catch { /* history not available */ }

        const results: SongGroup[] = [];
        for (const gen of gens) {
          try {
            console.log(`[RecordingsTab] Fetching audio gens for gen ${gen.id} "${gen.title}"...`);
            const res = await lireekApi.getAudioGenerations(gen.id);
            console.log(`[RecordingsTab] Gen ${gen.id} "${gen.title}" → ${res.audio_generations.length} audio gens`);
            if (res.audio_generations.length > 0) {
              const songs: Song[] = [];
              for (const ag of res.audio_generations) {
                try {
                  const jobRes = await generateApi.getStatus(ag.hotstep_job_id, token);
                  console.log(`[RecordingsTab] Job ${ag.hotstep_job_id}: status=${jobRes?.status}, audioUrls=${jobRes?.result?.audioUrls?.length ?? 0}`);
                  if (jobRes?.status === 'succeeded' && jobRes.result?.audioUrls) {
                    const originalPaths: string[] | undefined = jobRes.result.original_audio_paths;
                    for (let ai = 0; ai < jobRes.result.audioUrls.length; ai++) {
                      const audioUrl = jobRes.result.audioUrls[ai];
                      const genParams: Record<string, any> = {};
                      if (originalPaths?.[ai]) genParams.originalAudioUrl = originalPaths[ai];
                      songs.push({
                        id: ag.hotstep_job_id,
                        title: gen.title || 'Untitled',
                        style: gen.caption || '',
                        lyrics: gen.lyrics || '',
                        coverUrl: '',
                        duration: String(jobRes.result.duration || 0),
                        createdAt: new Date(ag.created_at),
                        tags: [],
                        audioUrl,
                        generationParams: Object.keys(genParams).length > 0 ? genParams : undefined,
                      });
                    }
                  } else {
                    // Fallback to history
                    const hj = jobHistory[ag.hotstep_job_id];
                    console.log(`[RecordingsTab] History fallback for ${ag.hotstep_job_id}: found=${!!hj}, status=${hj?.status}`);
                    if (hj?.status === 'succeeded' && hj.result?.audioUrls) {
                      for (const audioUrl of hj.result.audioUrls) {
                        songs.push({
                          id: ag.hotstep_job_id,
                          title: gen.title || 'Untitled',
                          style: gen.caption || '',
                          lyrics: gen.lyrics || '',
                          coverUrl: '',
                          duration: String(hj.result.duration || 0),
                          createdAt: new Date(ag.created_at),
                          tags: [],
                          audioUrl,
                        });
                      }
                    }
                  }
                } catch (err) {
                  console.warn(`[RecordingsTab] getStatus failed for ${ag.hotstep_job_id}:`, err);
                  const hj = jobHistory[ag.hotstep_job_id];
                  if (hj?.status === 'succeeded' && hj.result?.audioUrls) {
                    for (const audioUrl of hj.result.audioUrls) {
                      songs.push({
                        id: ag.hotstep_job_id,
                        title: gen.title || 'Untitled',
                        style: gen.caption || '',
                        lyrics: gen.lyrics || '',
                        coverUrl: '',
                        duration: String(hj.result.duration || 0),
                        createdAt: new Date(ag.created_at),
                        tags: [],
                        audioUrl,
                      });
                    }
                  } else {
                    console.warn(`[RecordingsTab] Could not resolve job ${ag.hotstep_job_id} — not in status or history`);
                  }
                }
              }
              console.log(`[RecordingsTab] Gen ${gen.id} resolved ${songs.length} playable songs`);
              results.push({ generation: gen, audioGens: res.audio_generations, songs });
            }
          } catch (err) {
            console.error(`[RecordingsTab] Failed to get audio gens for gen ${gen.id}:`, err);
          }
        }
        console.log(`[RecordingsTab] Total groups: ${results.length}, total songs: ${results.reduce((n, g) => n + g.songs.length, 0)}`);

        // Batch-fetch actual DB song records to get proper coverUrl, originalAudioUrl, etc.
        const allUrls = results.flatMap(g => g.songs.map(s => s.audioUrl).filter(Boolean));
        if (allUrls.length > 0 && token) {
          try {
            const { songs: dbSongs } = await songsApi.getSongsByUrls(allUrls, token);
            const dbMap = new Map(dbSongs.map(s => [s.audioUrl, s]));
            console.log(`[RecordingsTab] DB lookup returned ${dbSongs.length} songs for ${allUrls.length} URLs`);
            for (const group of results) {
              group.songs = group.songs.map(s => {
                const db = dbMap.get(s.audioUrl);
                if (db) {
                  return {
                    ...s,
                    id: db.id,
                    coverUrl: db.coverUrl || s.coverUrl,
                    generationParams: db.generationParams || s.generationParams,
                  };
                }
                return s;
              });
            }
          } catch (dbErr) {
            console.warn('[RecordingsTab] DB song lookup failed (non-fatal):', dbErr);
          }
        }

        if (!cancelled) {
          setGroups(results);
          const totalSongs = results.reduce((n, g) => n + g.songs.length, 0);
          onSongCountChange?.(totalSongs);
        }
      } catch (err) {
        console.error('[RecordingsTab] Failed to load:', err);
      } finally {
        // ALWAYS reset loading so UI never gets stuck
        setLoading(false);
      }
    };

    load();
    return () => { cancelled = true; };
  }, [genKey, token]);

  // Auto-expand when filtering to a single generation
  useEffect(() => {
    if (filterGenerationId && groups.length === 1) {
      setExpandedGenId(groups[0].generation.id);
    }
  }, [filterGenerationId, groups.length]);

  // ── Delete handler ──
  const handleDeleteAudioGen = useCallback(async (ag: AudioGeneration) => {
    if (!confirm('Delete this audio generation?')) return;
    try {
      await lireekApi.deleteAudioGeneration(ag.id);
      showToast('Audio generation deleted');
      setLocalRefreshKey(k => k + 1);
    } catch (err: any) {
      showToast(`Failed to delete: ${err.message}`);
    }
  }, [showToast]);

  // ── Download handler ──
  const handleDownload = useCallback((song: Song, format: DownloadFormat, version: DownloadVersion) => {
    if (!song.audioUrl) return;
    const filenamePrepend = localStorage.getItem('lireek-downloadFilenamePrepend')?.replace(/^"|"$/g, '') || '';
    const prefix = artistName ? `${artistName} - ` : '';
    const baseTitle = `${filenamePrepend}${prefix}${song.title || 'download'}`;

    const downloadSingleURL = (url: string, suffix: string) => {
      const targetUrl = new URL('/api/songs/download', window.location.origin);
      targetUrl.searchParams.set('audioUrl', url);
      targetUrl.searchParams.set('title', `${baseTitle}${suffix}`);
      targetUrl.searchParams.set('format', format);
      if (format === 'mp3') {
        const br = localStorage.getItem('mp3_export_bitrate');
        if (br) targetUrl.searchParams.set('mp3Bitrate', br);
      }
      if (format === 'opus') {
        const br = localStorage.getItem('opus_export_bitrate');
        if (br) targetUrl.searchParams.set('opusBitrate', br);
      }
      const a = document.createElement('a');
      a.href = targetUrl.toString();
      a.download = `${baseTitle}${suffix}.${format === 'opus' ? 'ogg' : format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    };

    if (version === 'mastered' || version === 'both') {
      downloadSingleURL(song.audioUrl, '');
    }
    if (version === 'original' || version === 'both') {
      const origUrl = song.generationParams?.originalAudioUrl || (song as any).originalAudioUrl;
      if (origUrl) {
        setTimeout(() => downloadSingleURL(origUrl, ' (Unmastered)'), version === 'both' ? 500 : 0);
      }
    }
    showToast(`Downloading ${format.toUpperCase()}...`);
  }, [showToast, artistName]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-6 h-6 text-zinc-500 animate-spin" />
      </div>
    );
  }

  return (
    <>
    <div className="p-4 space-y-2">
      {/* Filter indicator */}
      {filterGenerationId && onClearFilter && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-pink-500/10 border border-pink-500/20 mb-2">
          <Filter className="w-3.5 h-3.5 text-pink-400" />
          <span className="text-xs text-pink-300 flex-1">
            Showing songs from: <strong>{filteredGenerations[0]?.title || 'Untitled'}</strong>
          </span>
          <button
            onClick={onClearFilter}
            className="flex items-center gap-1 px-2 py-0.5 rounded text-xs text-zinc-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            <X className="w-3 h-3" />
            Clear
          </button>
        </div>
      )}

      {groups.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-center px-8">
          <div className="w-14 h-14 rounded-full bg-white/5 flex items-center justify-center mb-4">
            <Headphones className="w-7 h-7 text-zinc-600" />
          </div>
          <h3 className="text-base font-semibold text-zinc-400 mb-2">
            {filterGenerationId ? 'No songs generated from these lyrics yet' : 'No generated songs yet'}
          </h3>
          <p className="text-sm text-zinc-500 max-w-xs">
            Go to the Generated Lyrics tab and generate audio to see songs here.
          </p>
        </div>
      ) : (
        groups.map((group, idx) => {
          const isExpanded = expandedGenId === group.generation.id;
          return (
            <div
              key={group.generation.id}
              className={`rounded-xl border border-white/5 hover:border-white/10 overflow-hidden transition-colors ls2-card-in ls2-stagger-${Math.min(idx + 1, 11)}`}
            >
              {/* Group header */}
              <button
                className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-white/[0.02] transition-colors"
                onClick={() => setExpandedGenId(isExpanded ? null : group.generation.id)}
              >
                {isExpanded
                  ? <ChevronDown className="w-4 h-4 text-zinc-500 flex-shrink-0" />
                  : <ChevronRight className="w-4 h-4 text-zinc-500 flex-shrink-0" />
                }
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-zinc-200 truncate">
                    {group.generation.title || 'Untitled'}
                  </p>
                  <p className="text-xs text-zinc-500 mt-0.5">
                    {group.generation.subject || group.generation.caption?.slice(0, 60) || 'No caption'}
                  </p>
                </div>
                <span className="text-xs text-zinc-500 flex items-center gap-1">
                  <Headphones className="w-3 h-3" />
                  {group.songs.length} song{group.songs.length !== 1 ? 's' : ''}
                </span>
              </button>

              {/* Expanded: list recordings */}
              {isExpanded && (
                <div className="border-t border-white/5">
                  {group.songs.length === 0 ? (
                    <p className="px-4 py-6 text-sm text-zinc-500 text-center">
                      Audio generation is pending or failed. Check the queue.
                    </p>
                  ) : (
                    <div className="divide-y divide-white/5">
                      {group.songs.map((song, idx) => {
                        // Find the corresponding audio_generation record for delete
                        const ag = group.audioGens[idx];
                        return (
                          <div
                            key={idx}
                            className="flex items-center gap-3 px-4 py-2.5 hover:bg-white/[0.02] transition-colors"
                          >
                            <button
                              onClick={() => onPlaySong(song)}
                              className="w-8 h-8 rounded-full bg-pink-600/20 hover:bg-pink-600/30 flex items-center justify-center flex-shrink-0 transition-colors"
                            >
                              <Play className="w-3.5 h-3.5 text-pink-400 ml-0.5" />
                            </button>
                            <AddToPlaylistButton song={song} artistName={artistName} />
                            <div className="flex-1 min-w-0">
                              <p className="text-sm text-zinc-300 truncate">
                                {song.title || `Song ${idx + 1}`}
                              </p>
                              {song.duration && (
                                <p className="text-[11px] text-zinc-500 flex items-center gap-1">
                                  <Clock className="w-3 h-3" />
                                  {Math.floor(Number(song.duration) / 60)}:{String(Math.floor(Number(song.duration) % 60)).padStart(2, '0')}
                                </p>
                              )}
                            </div>
                            <button
                              onClick={() => setDownloadSong(song)}
                              className="p-1.5 rounded-lg text-zinc-500 hover:text-blue-400 hover:bg-blue-500/10 transition-colors"
                              title="Download"
                            >
                              <Download className="w-3.5 h-3.5" />
                            </button>
                            {ag && (
                              <button
                                onClick={() => handleDeleteAudioGen(ag)}
                                className="p-1.5 rounded-lg text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                                title="Delete"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })
      )}
    </div>

      {/* Download Modal */}
      <DownloadModal
        isOpen={!!downloadSong}
        onClose={() => setDownloadSong(null)}
        onDownload={(format, version) => {
          if (downloadSong) handleDownload(downloadSong, format, version);
        }}
        songTitle={downloadSong ? `${(localStorage.getItem('lireek-downloadFilenamePrepend')?.replace(/^"|"$/g, '') || '')}${artistName ? artistName + ' - ' : ''}${downloadSong.title || 'Untitled'}` : undefined}
        hasOriginal={!!(downloadSong?.generationParams?.originalAudioUrl || (downloadSong as any)?.originalAudioUrl)}
      />
    </>
  );
};

// ── Add-to-playlist helper ───────────────────────────────────────────────────

const AddToPlaylistButton: React.FC<{ song: Song; artistName?: string }> = ({ song, artistName }) => {
  const playlist = usePlaylist();
  const inPlaylist = playlist.isIn(song.id);

  const toggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (inPlaylist) {
      playlist.remove(song.id);
    } else {
      const dur = song.duration;
      let seconds = 0;
      if (typeof dur === 'string' && dur.includes(':')) {
        const [m, s] = dur.split(':').map(Number);
        seconds = (m || 0) * 60 + (s || 0);
      } else if (typeof dur === 'number') {
        seconds = dur;
      }
      playlist.add({
        id: song.id,
        title: song.title || 'Untitled',
        audioUrl: song.audioUrl || '',
        artistName: artistName || song.creator || '',
        coverUrl: song.coverUrl || (song as any).cover_url || '',
        duration: seconds,
        style: song.style || '',
      });
    }
  };

  return (
    <button
      onClick={toggle}
      className={`p-1 rounded-md transition-colors flex-shrink-0 ${
        inPlaylist
          ? 'text-pink-400 bg-pink-500/10 hover:bg-pink-500/20'
          : 'text-zinc-600 hover:text-pink-400 hover:bg-pink-500/10'
      }`}
      title={inPlaylist ? 'Remove from playlist' : 'Add to playlist'}
    >
      {inPlaylist ? <Check className="w-3.5 h-3.5" /> : <ListPlus className="w-3.5 h-3.5" />}
    </button>
  );
};
