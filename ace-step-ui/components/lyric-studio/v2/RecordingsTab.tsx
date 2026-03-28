import React, { useState, useEffect, useCallback } from 'react';
import { Play, Trash2, Headphones, ChevronDown, ChevronRight, Loader2, Clock } from 'lucide-react';
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
}

export const RecordingsTab: React.FC<RecordingsTabProps> = ({
  generations, onPlaySong, showToast,
}) => {
  const { token } = useAuth();
  const [groups, setGroups] = useState<SongGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedGenId, setExpandedGenId] = useState<number | null>(null);

  // Load audio generations for ALL lyric generations
  const loadAudioGens = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      const results: SongGroup[] = [];
      for (const gen of generations) {
        try {
          const res = await lireekApi.getAudioGenerations(gen.id);
          if (res.audio_generations.length > 0) {
            // Resolve each job_id to HOT-Step audio via job status
            const songs: Song[] = [];
            for (const ag of res.audio_generations) {
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
                      audioUrl: audioUrl,
                    });
                  }
                }
              } catch { /* job may not exist yet */ }
            }
            results.push({
              generation: gen,
              audioGens: res.audio_generations,
              songs,
            });
          }
        } catch { /* no audio gens for this one */ }
      }
      setGroups(results);
    } catch (err) {
      console.error('[RecordingsTab] Failed to load:', err);
    } finally {
      setLoading(false);
    }
  }, [generations, token]);

  useEffect(() => { loadAudioGens(); }, [loadAudioGens]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-6 h-6 text-zinc-500 animate-spin" />
      </div>
    );
  }

  if (groups.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center px-8">
        <div className="w-14 h-14 rounded-full bg-white/5 flex items-center justify-center mb-4">
          <Headphones className="w-7 h-7 text-zinc-600" />
        </div>
        <h3 className="text-base font-semibold text-zinc-400 mb-2">No recordings yet</h3>
        <p className="text-sm text-zinc-500 max-w-xs">
          Go to the Written Songs tab and generate audio to see recordings here.
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-2">
      {groups.map((group) => {
        const isExpanded = expandedGenId === group.generation.id;
        return (
          <div
            key={group.generation.id}
            className="rounded-xl border border-white/5 hover:border-white/10 overflow-hidden transition-colors"
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
                <p className="text-xs text-zinc-500 truncate mt-0.5">
                  {group.generation.caption?.slice(0, 60) || 'No caption'}
                </p>
              </div>
              <span className="text-xs text-zinc-500 flex items-center gap-1">
                <Headphones className="w-3 h-3" />
                {group.songs.length} recording{group.songs.length !== 1 ? 's' : ''}
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
                            {song.title || `Recording ${idx + 1}`}
                          </p>
                          {song.duration && (
                            <p className="text-[11px] text-zinc-500 flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              {Math.floor(song.duration / 60)}:{String(Math.floor(song.duration % 60)).padStart(2, '0')}
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
      })}
    </div>
  );
};
