/**
 * RecentSongsList.tsx — Shows recently generated songs across ALL Lireek artists.
 *
 * Data flow (fast path):
 *   1. GET /api/lireek/recent-songs returns rows with pre-resolved audio_url + cover_url
 *   2. We render directly — no job-history lookups, no songs-DB lookups
 *
 * MODULE-LEVEL CACHE ensures instant render on navigation.
 * Background refresh only on refreshKey change (new generation completed).
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Play, Loader2, Music, Download, Trash2, ListPlus, Check } from 'lucide-react';
import { lireekApi, RecentSong } from '../../../services/lyricStudioApi';
import { songsApi } from '../../../services/api';
import { useAuth } from '../../../context/AuthContext';
import { Song } from '../../../types';
import { DownloadModal, DownloadFormat, DownloadVersion } from '../../DownloadModal';
import { usePlaylist } from './playlistStore';

interface RecentSongsListProps {
  onPlaySong: (song: Song) => void;
  showToast: (msg: string, type?: 'success' | 'error') => void;
  refreshKey?: number;
}

// ── Module-level cache ───────────────────────────────────────────────────────

let _cachedSongs: RecentSong[] = [];
let _cachedRefreshKey = -1;
let _fetchInFlight = false;

async function _loadRecentSongs(): Promise<RecentSong[]> {
  const res = await lireekApi.getRecentSongs(50);
  // Only keep songs that have pre-resolved audio URLs
  return (res.songs || []).filter(s => !!s.audio_url);
}

// ── Component ────────────────────────────────────────────────────────────────

export const RecentSongsList: React.FC<RecentSongsListProps> = ({ onPlaySong, showToast, refreshKey = 0 }) => {
  const { token } = useAuth();
  const [songs, setSongs] = useState<RecentSong[]>(_cachedSongs);
  const [loading, setLoading] = useState(_cachedSongs.length === 0);
  const mountedRef = useRef(true);

  // Download modal state
  const [downloadSong, setDownloadSong] = useState<{ rs: RecentSong; dbSong: any } | null>(null);
  const [downloadModalOpen, setDownloadModalOpen] = useState(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  useEffect(() => {
    if (!token) { setLoading(false); return; }
    if (_cachedRefreshKey === refreshKey && _cachedSongs.length > 0) return;
    if (_fetchInFlight) return;
    if (_cachedSongs.length === 0) setLoading(true);

    _fetchInFlight = true;
    _loadRecentSongs().then(resolved => {
      _cachedSongs = resolved;
      _cachedRefreshKey = refreshKey;
      _fetchInFlight = false;
      if (mountedRef.current) { setSongs(resolved); setLoading(false); }
    }).catch(() => {
      _fetchInFlight = false;
      if (mountedRef.current) setLoading(false);
    });
  }, [refreshKey, token]);

  const handlePlay = useCallback(async (rs: RecentSong) => {
    const audioUrl = rs.audio_url || '';
    const song: Song = {
      id: rs.hotstep_job_id,
      title: rs.song_title || 'Untitled',
      style: rs.caption || '',
      lyrics: rs.lyrics || '',
      coverUrl: rs.cover_url || rs.album_image || rs.artist_image || '',
      duration: String(rs.duration || 0),
      createdAt: new Date(rs.ag_created_at),
      tags: [],
      audioUrl,
    };

    // Look up full DB record for generationParams (originalAudioUrl for M/O toggle)
    if (audioUrl && token) {
      try {
        const { songs: dbSongs } = await songsApi.getSongsByUrls([audioUrl], token);
        const db: any = dbSongs[0];
        if (db) {
          if (db.coverUrl || db.cover_url) song.coverUrl = db.coverUrl || db.cover_url;
          if (db.generationParams) (song as any).generationParams = db.generationParams;
          if (db.duration) song.duration = String(db.duration);
        }
      } catch { /* non-fatal */ }
    }

    // Ensure artistName + source are always set for Lireek songs (older DB records may lack artistName)
    const gp = (song as any).generationParams || {};
    gp.source = gp.source || 'lyric-studio';
    gp.artistName = gp.artistName || rs.artist_name;
    (song as any).generationParams = gp;

    onPlaySong(song);
  }, [onPlaySong, token]);

  // ── Delete handler ──────────────────────────────────────────────────────
  const handleDelete = useCallback(async (e: React.MouseEvent, rs: RecentSong) => {
    e.stopPropagation();
    if (!token || !rs.audio_url) return;

    try {
      // Find and delete from songs DB
      const { songs: dbSongs } = await songsApi.getSongsByUrls([rs.audio_url], token);
      const db: any = dbSongs[0];
      if (db?.id) {
        await songsApi.deleteSong(db.id, token);
      }
      // Remove from local list
      setSongs(prev => {
        const updated = prev.filter(s => s.ag_id !== rs.ag_id);
        _cachedSongs = updated;
        return updated;
      });
      showToast('Song deleted');
    } catch (err) {
      showToast('Failed to delete song', 'error');
    }
  }, [token, showToast]);

  // ── Download handler ────────────────────────────────────────────────────
  const handleDownloadClick = useCallback(async (e: React.MouseEvent, rs: RecentSong) => {
    e.stopPropagation();
    if (!token || !rs.audio_url) return;

    // Fetch full DB record for originalAudioUrl
    let dbSong: any = null;
    try {
      const { songs: dbSongs } = await songsApi.getSongsByUrls([rs.audio_url], token);
      dbSong = dbSongs[0] || null;
    } catch { /* non-fatal */ }

    setDownloadSong({ rs, dbSong });
    setDownloadModalOpen(true);
  }, [token]);

  const handleDownload = useCallback((format: DownloadFormat, version: DownloadVersion) => {
    if (!downloadSong) return;
    const { rs, dbSong } = downloadSong;
    const audioUrl = rs.audio_url || '';
    const filenamePrepend = localStorage.getItem('lireek-downloadFilenamePrepend')?.replace(/^"|"$/g, '') || '';
    const artistPrefix = rs.artist_name ? `${rs.artist_name} - ` : '';
    const displayTitle = `${filenamePrepend}${artistPrefix}${rs.song_title || 'Untitled'}`;

    const downloadSingleURL = (url: string, suffix: string) => {
      const targetUrl = new URL('/api/songs/download', window.location.origin);
      targetUrl.searchParams.set('audioUrl', url);
      targetUrl.searchParams.set('title', `${displayTitle}${suffix}`);
      targetUrl.searchParams.set('format', format);
      if (dbSong?.id) targetUrl.searchParams.set('songId', dbSong.id);
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
      downloadSingleURL(audioUrl, '');
    }
    if (version === 'original' || version === 'both') {
      const origUrl = dbSong?.generationParams?.originalAudioUrl || dbSong?.originalAudioUrl;
      if (origUrl) {
        setTimeout(() => downloadSingleURL(origUrl, ' (Unmastered)'), version === 'both' ? 500 : 0);
      }
    }
  }, [downloadSong]);

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

  const hasOriginal = downloadSong?.dbSong?.generationParams?.originalAudioUrl ||
    downloadSong?.dbSong?.originalAudioUrl;

  return (
    <>
      <div className="flex flex-col gap-1 px-2 py-1.5 overflow-y-auto scrollbar-hide" style={{ maxHeight: '100%' }}>
        {songs.slice(0, 50).map((rs) => {
          const dur = rs.duration || 0;
          const mins = Math.floor(dur / 60);
          const secs = String(Math.floor(dur % 60)).padStart(2, '0');
          const coverUrl = rs.cover_url || rs.album_image || rs.artist_image || '';
          return (
            <div
              key={rs.ag_id}
              className="flex items-center gap-2.5 rounded-lg hover:bg-white/[0.06] transition-colors text-left group px-2 overflow-hidden relative cursor-pointer"
              onClick={() => handlePlay(rs)}
            >
              {/* Cover art */}
              <div className="w-14 h-14 rounded-md flex-shrink-0 overflow-hidden bg-zinc-800 relative">
                {coverUrl ? (
                  <img src={coverUrl} alt="" className="w-full h-full object-cover" />
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
                  {rs.song_title || 'Untitled'}
                </p>
                <p className="text-[10px] text-zinc-500 truncate leading-snug">
                  {rs.artist_name}
                </p>
                {dur > 0 && (
                  <p className="text-[10px] text-zinc-600 font-mono mt-0.5">
                    {mins}:{secs}
                  </p>
                )}
              </div>
              {/* Action buttons — visible on hover */}
              <div className="absolute right-1 top-1/2 -translate-y-1/2 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <AddToPlaylistBtn rs={rs} />
                <button
                  onClick={(e) => handleDownloadClick(e, rs)}
                  className="p-1.5 rounded-md bg-zinc-800/80 hover:bg-zinc-700 text-zinc-400 hover:text-white transition-colors"
                  title="Download"
                >
                  <Download className="w-3 h-3" />
                </button>
                <button
                  onClick={(e) => handleDelete(e, rs)}
                  className="p-1.5 rounded-md bg-zinc-800/80 hover:bg-red-900/60 text-zinc-400 hover:text-red-400 transition-colors"
                  title="Delete"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Download Modal */}
      <DownloadModal
        isOpen={downloadModalOpen}
        onClose={() => { setDownloadModalOpen(false); setDownloadSong(null); }}
        onDownload={handleDownload}
        songTitle={downloadSong ? `${(localStorage.getItem('lireek-downloadFilenamePrepend')?.replace(/^"|"$/g, '') || '')}${downloadSong.rs.artist_name ? downloadSong.rs.artist_name + ' - ' : ''}${downloadSong.rs.song_title || 'Untitled'}` : undefined}
        hasOriginal={!!hasOriginal}
      />
    </>
  );
};

// ── Add-to-playlist helper ───────────────────────────────────────────────────

const AddToPlaylistBtn: React.FC<{ rs: RecentSong }> = ({ rs }) => {
  const playlist = usePlaylist();
  const itemId = String(rs.ag_id) || `recent-${rs.song_title}`;
  const inPlaylist = playlist.isIn(itemId);

  const toggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (inPlaylist) {
      playlist.remove(itemId);
    } else {
      playlist.add({
        id: itemId,
        title: rs.song_title || 'Untitled',
        audioUrl: rs.audio_url || '',
        artistName: rs.artist_name || '',
        coverUrl: rs.cover_url || rs.album_image || rs.artist_image || '',
        duration: rs.duration || 0,
      });
    }
  };

  return (
    <button
      onClick={toggle}
      className={`p-1.5 rounded-md transition-colors ${
        inPlaylist
          ? 'bg-pink-500/20 text-pink-400 hover:bg-pink-500/30'
          : 'bg-zinc-800/80 hover:bg-zinc-700 text-zinc-400 hover:text-pink-400'
      }`}
      title={inPlaylist ? 'Remove from playlist' : 'Add to playlist'}
    >
      {inPlaylist ? <Check className="w-3 h-3" /> : <ListPlus className="w-3 h-3" />}
    </button>
  );
};
