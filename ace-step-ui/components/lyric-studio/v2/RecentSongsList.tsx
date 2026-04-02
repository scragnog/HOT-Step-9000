/**
 * RecentSongsList.tsx — Shows recently generated songs across ALL Lireek artists.
 * Displays cover art thumbnails, artist name, title, and duration.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Play, Loader2, Clock, Music } from 'lucide-react';
import { lireekApi, RecentSong } from '../../../services/lyricStudioApi';
import { generateApi } from '../../../services/api';
import { useAuth } from '../../../context/AuthContext';
import { Song } from '../../../types';

interface RecentSongsListProps {
  onPlaySong: (song: Song) => void;
  refreshKey?: number;
}

interface ResolvedSong {
  recentSong: RecentSong;
  audioUrl: string;
  coverUrl: string;
  duration: number;
}

export const RecentSongsList: React.FC<RecentSongsListProps> = ({ onPlaySong, refreshKey = 0 }) => {
  const { token } = useAuth();
  const [songs, setSongs] = useState<ResolvedSong[]>([]);
  const [loading, setLoading] = useState(true);
  const prevKeyRef = useRef(`${refreshKey}`);

  const loadSongs = useCallback(async () => {
    if (!token) { setLoading(false); return; }
    setLoading(true);
    try {
      const res = await lireekApi.getRecentSongs(30);
      // Resolve HOT-Step job IDs to audio URLs
      const resolved: ResolvedSong[] = [];
      // Batch fetch — try job history first for speed
      let jobHistory: Record<string, any> = {};
      try {
        const historyRes = await generateApi.getHistory(token);
        if (historyRes?.jobs) {
          for (const job of historyRes.jobs) {
            const id = job.jobId || job.id;
            if (id) jobHistory[id] = job;
          }
        }
      } catch { /* no history */ }

      for (const rs of res.songs) {
        try {
          // Try status API first
          let audioUrl = '';
          let coverUrl = rs.album_image || rs.artist_image || '';
          let duration = rs.duration || 0;

          const hj = jobHistory[rs.hotstep_job_id];
          if (hj?.status === 'succeeded' && hj.result?.audioUrls?.[0]) {
            audioUrl = hj.result.audioUrls[0];
            if (hj.result.duration) duration = hj.result.duration;
          } else {
            try {
              const status = await generateApi.getStatus(rs.hotstep_job_id, token);
              if (status?.status === 'succeeded' && status.result?.audioUrls?.[0]) {
                audioUrl = status.result.audioUrls[0];
                if (status.result.duration) duration = status.result.duration;
              }
            } catch { /* job no longer exists */ }
          }

          if (audioUrl) {
            resolved.push({ recentSong: rs, audioUrl, coverUrl, duration });
          }
        } catch { /* skip failed resolution */ }
      }

      setSongs(resolved);
    } catch (err) {
      console.error('[RecentSongsList] Load failed:', err);
    } finally {
      setLoading(false);
    }
  }, [token]);

  // Load on mount and when refreshKey changes
  useEffect(() => {
    const key = `${refreshKey}`;
    if (key !== prevKeyRef.current || songs.length === 0) {
      prevKeyRef.current = key;
      loadSongs();
    }
  }, [refreshKey, loadSongs]);

  const handlePlay = useCallback((rs: ResolvedSong) => {
    const song: Song = {
      id: rs.recentSong.hotstep_job_id,
      title: rs.recentSong.song_title || 'Untitled',
      style: rs.recentSong.caption || '',
      lyrics: rs.recentSong.lyrics || '',
      coverUrl: rs.coverUrl,
      duration: String(rs.duration || 0),
      createdAt: new Date(rs.recentSong.ag_created_at),
      tags: [],
      audioUrl: rs.audioUrl,
    };
    onPlaySong(song);
  }, [onPlaySong]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-4 h-4 text-zinc-500 animate-spin" />
      </div>
    );
  }

  if (songs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-center px-4">
        <Music className="w-5 h-5 text-zinc-600 mb-2" />
        <p className="text-xs text-zinc-500">No recent generations yet</p>
      </div>
    );
  }

  return (
    <div className="space-y-0.5">
      {songs.map((rs, idx) => {
        const mins = Math.floor(rs.duration / 60);
        const secs = String(Math.floor(rs.duration % 60)).padStart(2, '0');
        return (
          <button
            key={rs.recentSong.ag_id}
            onClick={() => handlePlay(rs)}
            className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg hover:bg-white/[0.04] transition-colors text-left group"
          >
            {/* Cover art thumbnail */}
            <div className="w-8 h-8 rounded-md flex-shrink-0 overflow-hidden bg-zinc-800 relative">
              {rs.coverUrl ? (
                <img src={rs.coverUrl} alt="" className="w-full h-full object-cover" />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <Music className="w-3.5 h-3.5 text-zinc-600" />
                </div>
              )}
              {/* Play overlay */}
              <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                <Play className="w-3 h-3 text-white ml-0.5" />
              </div>
            </div>
            {/* Info */}
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-zinc-200 truncate">
                {rs.recentSong.song_title || 'Untitled'}
              </p>
              <p className="text-[10px] text-zinc-500 truncate">
                {rs.recentSong.artist_name}{rs.recentSong.album ? ` · ${rs.recentSong.album}` : ''}
              </p>
            </div>
            {/* Duration */}
            {rs.duration > 0 && (
              <span className="text-[10px] text-zinc-600 flex items-center gap-0.5 flex-shrink-0">
                <Clock className="w-2.5 h-2.5" />
                {mins}:{secs}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
};
