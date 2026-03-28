import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Play, Trash2, Headphones, ChevronDown, ChevronRight, Loader2, Clock, X, Filter } from 'lucide-react';
import { lireekApi, Generation, AudioGeneration } from '../../../services/lyricStudioApi';
import { generateApi } from '../../../services/api';
import { useAuth } from '../../../context/AuthContext';
import { Song } from '../../../types';

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
}

export const RecordingsTab: React.FC<RecordingsTabProps> = ({
  generations, onPlaySong, showToast, filterGenerationId, onClearFilter,
}) => {
  const { token } = useAuth();
  const [groups, setGroups] = useState<SongGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedGenId, setExpandedGenId] = useState<number | null>(null);

  // Keep a ref to generations so the effect callback can read current data
  // without depending on the array reference
  const generationsRef = useRef(generations);
  generationsRef.current = generations;

  // Stable key: only re-fetch when the set of generation IDs or filter changes
  const genKey = useMemo(() => {
    const ids = generations.map(g => g.id).sort().join(',');
    return `${ids}|${filterGenerationId ?? 'all'}`;
  }, [generations, filterGenerationId]);

  // Filtered view for rendering (cheap, no effect dependency)
  const filteredGenerations = useMemo(() =>
    filterGenerationId
      ? generations.filter(g => g.id === filterGenerationId)
      : generations,
    [generations, filterGenerationId]
  );

  // Load audio generations — keyed only on genKey + token (stable strings)
  useEffect(() => {
    if (!token || genKey === '|all') {
      setLoading(false);
      return;
    }
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      try {
        const gens = filterGenerationId
          ? generationsRef.current.filter(g => g.id === filterGenerationId)
          : generationsRef.current;

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
        } catch { /* history not available */ }

        const results: SongGroup[] = [];
        for (const gen of gens) {
          if (cancelled) return;
          try {
            const res = await lireekApi.getAudioGenerations(gen.id);
            if (res.audio_generations.length > 0) {
              const songs: Song[] = [];
              for (const ag of res.audio_generations) {
                if (cancelled) return;
                try {
                  const jobRes = await generateApi.getStatus(ag.job_id, token);
                  if (jobRes?.status === 'succeeded' && jobRes.result?.audioUrls) {
                    for (const audioUrl of jobRes.result.audioUrls) {
                      songs.push({
                        id: ag.job_id,
                        title: gen.title || 'Untitled',
                        style: gen.caption || '',
                        lyrics: gen.lyrics || '',
                        coverUrl: '',
                        duration: String(jobRes.result.duration || 0),
                        createdAt: new Date(ag.created_at),
                        tags: [],
                        audioUrl,
                      });
                    }
                  } else {
                    // Fallback to history
                    const hj = jobHistory[ag.job_id];
                    if (hj?.status === 'succeeded' && hj.result?.audioUrls) {
                      for (const audioUrl of hj.result.audioUrls) {
                        songs.push({
                          id: ag.job_id,
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
                } catch {
                  const hj = jobHistory[ag.job_id];
                  if (hj?.status === 'succeeded' && hj.result?.audioUrls) {
                    for (const audioUrl of hj.result.audioUrls) {
                      songs.push({
                        id: ag.job_id,
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
                    console.warn(`[RecordingsTab] Could not resolve job ${ag.job_id}`);
                  }
                }
              }
              results.push({ generation: gen, audioGens: res.audio_generations, songs });
            }
          } catch { /* no audio gens */ }
        }
        if (!cancelled) setGroups(results);
      } catch (err) {
        console.error('[RecordingsTab] Failed to load:', err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();
    return () => { cancelled = true; };
  }, [genKey, token]); // ← stable string deps only!

  // Auto-expand when filtering to a single generation
  useEffect(() => {
    if (filterGenerationId && groups.length === 1) {
      setExpandedGenId(groups[0].generation.id);
    }
  }, [filterGenerationId, groups.length]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-6 h-6 text-zinc-500 animate-spin" />
      </div>
    );
  }

  return (
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
                      {group.songs.map((song, idx) => (
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
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })
      )}
    </div>
  );
};
