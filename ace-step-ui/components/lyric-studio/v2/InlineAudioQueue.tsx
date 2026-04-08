/**
 * InlineAudioQueue.tsx — Inline audio generation queue display.
 * Shows active, pending, and completed audio generation jobs from the audioGenQueueStore.
 *
 * Order: Active (generating/loading) → Queued (pending) → Completed (succeeded/failed, newest first)
 * Completed items play through the main player (onPlaySong) so they drive the
 * playbar, visualisations, and cover art background like every other playback path.
 */

import React, { useCallback } from 'react';
import { Loader2, CheckCircle2, XCircle, X, Music, Clock, Play, Square, ListPlus, Check } from 'lucide-react';
import {
  useAudioGenQueue,
  removeFromAudioQueue,
  clearFinishedFromAudioQueue,
  AudioQueueItem,
} from '../../../stores/audioGenQueueStore';
import { usePlaylist } from './playlistStore';
import { Song } from '../../../types';

interface InlineAudioQueueProps {
  /** Route playback through the main player (App → Player → visualisations etc.) */
  onPlaySong?: (song: Song) => void;
  /** ID of the song currently playing in the main player — used for highlight state */
  currentSongId?: string | null;
}

export const InlineAudioQueue: React.FC<InlineAudioQueueProps> = ({ onPlaySong, currentSongId }) => {
  const { items } = useAudioGenQueue();

  // ── Group & sort ──────────────────────────────────────────────────────
  const active = items.filter(
    i => i.status === 'loading-adapter' || i.status === 'generating',
  );
  const queued = items.filter(i => i.status === 'pending');
  const finished = items
    .filter(i => i.status === 'succeeded' || i.status === 'failed')
    // Newest completed first  (items are pushed chronologically, so reverse)
    .slice()
    .reverse();

  const pendingCount = queued.length;
  const finishedCount = finished.length;

  // ── Play via main player ──────────────────────────────────────────────
  const handlePlay = useCallback((item: AudioQueueItem) => {
    if (!item.audioUrl || !onPlaySong) return;

    // Build a Song object the main player understands
    const song: Song = {
      id: item.id,
      title: item.generation.title || 'Untitled',
      lyrics: '',
      style: item.generation.style || '',
      audioUrl: item.audioUrl,
      coverUrl: '',
      duration: '0:00',
      createdAt: new Date(),
      tags: [],
      isPublic: false,
      likeCount: 0,
      viewCount: 0,
      userId: '',
      creator: item.artistName || '',
    };

    onPlaySong(song);
  }, [onPlaySong]);

  if (items.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-center px-4">
        <Music className="w-5 h-5 text-zinc-600 mb-2" />
        <p className="text-xs text-zinc-500">No queued generations</p>
      </div>
    );
  }

  return (
    <div className="space-y-1 px-2">
      {/* Header bar */}
      <div className="flex items-center justify-between px-1 py-1">
        <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">
          {pendingCount > 0 && `${pendingCount} pending`}
          {pendingCount > 0 && finishedCount > 0 && ' · '}
          {finishedCount > 0 && `${finishedCount} done`}
        </span>
        {finishedCount > 0 && (
          <button
            onClick={clearFinishedFromAudioQueue}
            className="text-[10px] text-zinc-500 hover:text-red-400 transition-colors"
          >
            Clear Done
          </button>
        )}
      </div>

      {/* ── Active ────────────────────────────────────────────────── */}
      {active.length > 0 && (
        <>
          <GroupLabel label="Active" color="text-pink-400" />
          {active.map(item => (
            <QueueItemRow key={item.id} item={item} isPlayingInMain={currentSongId === item.id} onPlay={handlePlay} />
          ))}
        </>
      )}

      {/* ── Queued ────────────────────────────────────────────────── */}
      {queued.length > 0 && (
        <>
          <GroupLabel label="Queued" color="text-zinc-400" />
          {queued.map(item => (
            <QueueItemRow key={item.id} item={item} isPlayingInMain={currentSongId === item.id} onPlay={handlePlay} />
          ))}
        </>
      )}

      {/* ── Completed ─────────────────────────────────────────────── */}
      {finished.length > 0 && (
        <>
          <GroupLabel label="Completed" color="text-green-400" />
          {finished.map(item => (
            <QueueItemRow key={item.id} item={item} isPlayingInMain={currentSongId === item.id} onPlay={handlePlay} />
          ))}
        </>
      )}
    </div>
  );
};

// ── Group label ──────────────────────────────────────────────────────────────

const GroupLabel: React.FC<{ label: string; color: string }> = ({ label, color }) => (
  <p className={`text-[9px] font-bold uppercase tracking-widest ${color} px-1 pt-2 pb-0.5`}>
    {label}
  </p>
);

// ── Item Row ─────────────────────────────────────────────────────────────────

interface QueueItemRowProps {
  item: AudioQueueItem;
  isPlayingInMain: boolean;
  onPlay: (item: AudioQueueItem) => void;
}

const QueueItemRow: React.FC<QueueItemRowProps> = ({ item, isPlayingInMain, onPlay }) => {
  const isRunning = item.status === 'loading-adapter' || item.status === 'generating';
  const isSucceeded = item.status === 'succeeded';
  const isFailed = item.status === 'failed';
  const isPending = item.status === 'pending';

  const elapsed = item.elapsed || 0;
  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;
  const timeStr = elapsed > 0
    ? mins > 0 ? `${mins}:${String(secs).padStart(2, '0')}` : `${secs}s`
    : '';

  const borderColor = isSucceeded
    ? 'border-green-500/20'
    : isFailed
      ? 'border-red-500/20'
      : isRunning
        ? 'border-pink-500/20'
        : 'border-white/5';

  return (
    <div className={`rounded-lg border ${borderColor} bg-white/[0.02] px-3 py-2 transition-all ${isPlayingInMain ? 'ring-1 ring-pink-500/40 bg-pink-500/5' : ''}`}>
      <div className="flex items-center gap-2">
        {/* Status icon / Play button */}
        <div className="flex-shrink-0">
          {isPending && <div className="w-2 h-2 rounded-full bg-zinc-500" />}
          {isRunning && <Loader2 className="w-3.5 h-3.5 text-pink-400 animate-spin" />}
          {isSucceeded && item.audioUrl ? (
            <button
              onClick={() => onPlay(item)}
              className={`p-0.5 rounded-full transition-colors ${
                isPlayingInMain
                  ? 'bg-green-500/20 text-green-300 hover:bg-green-500/30'
                  : 'text-green-400 hover:bg-green-500/20 hover:text-green-300'
              }`}
              title={isPlayingInMain ? 'Playing' : 'Play'}
            >
              {isPlayingInMain
                ? <Square className="w-3 h-3" />
                : <Play className="w-3 h-3" />
              }
            </button>
          ) : isSucceeded ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
          ) : null}
          {isFailed && <XCircle className="w-3.5 h-3.5 text-red-400" />}
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <p className={`text-xs font-medium truncate ${isPlayingInMain ? 'text-pink-300' : 'text-zinc-200'}`}>
            {item.generation.title || 'Untitled'}
          </p>
          <p className="text-[10px] text-zinc-500 truncate">
            {item.artistName}
          </p>
        </div>

        {/* Time / Actions */}
        <div className="flex items-center gap-1 flex-shrink-0">
          {timeStr && (
            <span className="text-[10px] text-zinc-500 font-mono flex items-center gap-0.5">
              <Clock className="w-2.5 h-2.5" />
              {timeStr}
            </span>
          )}
          {isPending && (
            <button
              onClick={() => removeFromAudioQueue(item.id)}
              className="p-0.5 rounded hover:bg-red-500/20 text-zinc-500 hover:text-red-400 transition-colors"
              title="Remove from queue"
            >
              <X className="w-3 h-3" />
            </button>
          )}
          {isSucceeded && item.audioUrl && (
            <QueueAddToPlaylistBtn item={item} />
          )}
        </div>
      </div>

      {/* Progress bar (running items) */}
      {isRunning && (
        <div className="mt-1.5 space-y-1">
          <div className="h-1 rounded-full bg-white/10 overflow-hidden">
            <div
              className={`h-full bg-gradient-to-r from-pink-500 to-purple-600 transition-all duration-500 ${
                !item.progress ? 'animate-pulse opacity-40 w-full' : ''
              }`}
              style={item.progress ? { width: `${item.progress}%` } : undefined}
            />
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[9px] text-zinc-500">
              {item.stage || 'Processing…'}
            </span>
            {item.progress !== undefined && item.progress > 0 && (
              <span className="text-[9px] font-bold text-pink-400">
                {Math.round(item.progress)}%
              </span>
            )}
          </div>
        </div>
      )}

      {/* Error message */}
      {isFailed && item.error && (
        <p className="mt-1 text-[9px] text-red-400 truncate">{item.error}</p>
      )}
    </div>
  );
};

// ── Add-to-playlist helper ───────────────────────────────────────────────────

const QueueAddToPlaylistBtn: React.FC<{ item: AudioQueueItem }> = ({ item }) => {
  const playlist = usePlaylist();
  const inPlaylist = playlist.isIn(item.id);

  const toggle = () => {
    if (inPlaylist) {
      playlist.remove(item.id);
    } else {
      playlist.add({
        id: item.id,
        title: item.generation.title || 'Untitled',
        audioUrl: item.audioUrl || '',
        artistName: item.artistName || '',
        coverUrl: '',
        duration: 0,
      });
    }
  };

  return (
    <button
      onClick={toggle}
      className={`p-0.5 rounded transition-colors ${
        inPlaylist
          ? 'text-pink-400 bg-pink-500/10 hover:bg-pink-500/20'
          : 'text-zinc-600 hover:text-pink-400 hover:bg-pink-500/10'
      }`}
      title={inPlaylist ? 'Remove from playlist' : 'Add to playlist'}
    >
      {inPlaylist ? <Check className="w-3 h-3" /> : <ListPlus className="w-3 h-3" />}
    </button>
  );
};
