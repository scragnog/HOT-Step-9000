/**
 * RecentSongsList.tsx — Shows recently generated songs across ALL Lireek artists.
 *
 * Uses a MODULE-LEVEL CACHE so data survives component unmount/remount.
 * On mount: shows cached data instantly. Refreshes in background only when
 * refreshKey changes (new generation completed).
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Play, Loader2, Clock, Music } from 'lucide-react';
import { lireekApi, RecentSong } from '../../../services/lyricStudioApi';
import { generateApi, songsApi } from '../../../services/api';
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

// ── Module-level cache — survives unmount/remount ────────────────────────────

let _cachedSongs: ResolvedSong[] = [];
let _cachedRefreshKey = -1;
let _fetchInFlight = false;
let _jobHistoryCache: Record<string, any> = {};
let _jobHistoryLoaded = false;

async function _resolveRecentSongs(token: string): Promise<ResolvedSong[]> {
  const res = await lireekApi.getRecentSongs(30);

  // Load job history once and cache it
  if (!_jobHistoryLoaded) {
    try {
      const historyRes = await generateApi.getHistory(token);
      if (historyRes?.jobs) {
        for (const job of historyRes.jobs) {
          const id = job.jobId || job.id;
          if (id) _jobHistoryCache[id] = job;
        }
      }
      _jobHistoryLoaded = true;
    } catch { /* no history */ }
  }

  const resolved: ResolvedSong[] = [];

  for (const rs of res.songs) {
    try {
      let audioUrl = '';
      let coverUrl = rs.album_image || rs.artist_image || '';
      let duration = rs.duration || 0;

      const hj = _jobHistoryCache[rs.hotstep_job_id];
      if (hj?.status === 'succeeded' && hj.result?.audioUrls?.[0]) {
        audioUrl = hj.result.audioUrls[0];
        if (hj.result.duration) duration = hj.result.duration;
      } else {
        // Only call status API for jobs not in cache — skip failures quickly
        try {
          const status = await generateApi.getStatus(rs.hotstep_job_id, token);
          // Cache the result for next time
          _jobHistoryCache[rs.hotstep_job_id] = status;
          if (status?.status === 'succeeded' && status.result?.audioUrls?.[0]) {
            audioUrl = status.result.audioUrls[0];
            if (status.result.duration) duration = status.result.duration;
          }
        } catch { /* job no longer exists */ }
      }

      if (audioUrl) {
        resolved.push({ recentSong: rs, audioUrl, coverUrl, duration });
      }
    } catch { /* skip */ }
  }

  // Batch-fetch actual DB song records for real cover art
  const allUrls = resolved.map(r => r.audioUrl).filter(Boolean);
  if (allUrls.length > 0) {
    try {
      const { songs: dbSongs } = await songsApi.getSongsByUrls(allUrls, token);
      const dbMap = new Map(dbSongs.map(s => [s.audioUrl || s.audio_url, s]));
      for (const item of resolved) {
        const db: any = dbMap.get(item.audioUrl);
        if (db?.coverUrl || db?.cover_url) {
          item.coverUrl = db.coverUrl || db.cover_url;
        }
        if (db?.duration) {
          item.duration = Number(db.duration) || item.duration;
        }
      }
    } catch { /* non-fatal */ }
  }

  return resolved;
}

// ── Component ────────────────────────────────────────────────────────────────

export const RecentSongsList: React.FC<RecentSongsListProps> = ({ onPlaySong, refreshKey = 0 }) => {
  const { token } = useAuth();
  // Init from cache — instant render with no loading state
  const [songs, setSongs] = useState<ResolvedSong[]>(_cachedSongs);
  const [loading, setLoading] = useState(_cachedSongs.length === 0);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  // Refresh: only when refreshKey actually changes, or first load
  useEffect(() => {
    if (!token) { setLoading(false); return; }

    // If we already have cached data for this refreshKey, skip
    if (_cachedRefreshKey === refreshKey && _cachedSongs.length > 0) return;

    // If another fetch is already in flight, skip
    if (_fetchInFlight) return;

    // Only show spinner if cache is empty
    if (_cachedSongs.length === 0) setLoading(true);

    _fetchInFlight = true;

    _resolveRecentSongs(token).then(resolved => {
      _cachedSongs = resolved;
      _cachedRefreshKey = refreshKey;
      _fetchInFlight = false;
      if (mountedRef.current) {
        setSongs(resolved);
        setLoading(false);
      }
    }).catch(() => {
      _fetchInFlight = false;
      if (mountedRef.current) setLoading(false);
    });
  }, [refreshKey, token]);

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

  if (loading && songs.length === 0) {
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
    <div className="grid grid-cols-2 grid-rows-4 gap-1 px-2 py-1.5 h-full">
      {songs.slice(0, 8).map((rs) => {
        const mins = Math.floor(rs.duration / 60);
        const secs = String(Math.floor(rs.duration % 60)).padStart(2, '0');
        return (
          <button
            key={rs.recentSong.ag_id}
            onClick={() => handlePlay(rs)}
            className="flex items-center gap-2.5 rounded-lg hover:bg-white/[0.06] transition-colors text-left group px-2 overflow-hidden"
          >
            {/* Cover art */}
            <div className="w-14 h-14 rounded-md flex-shrink-0 overflow-hidden bg-zinc-800 relative">
              {rs.coverUrl ? (
                <img src={rs.coverUrl} alt="" className="w-full h-full object-cover" />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <Music className="w-5 h-5 text-zinc-600" />
                </div>
              )}
              {/* Play overlay */}
              <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                <Play className="w-4 h-4 text-white ml-0.5" />
              </div>
            </div>
            {/* Info */}
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-zinc-200 truncate leading-snug">
                {rs.recentSong.song_title || 'Untitled'}
              </p>
              <p className="text-[10px] text-zinc-500 truncate leading-snug">
                {rs.recentSong.artist_name}
              </p>
              {rs.duration > 0 && (
                <p className="text-[10px] text-zinc-600 font-mono mt-0.5">
                  {mins}:{secs}
                </p>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
};
