/**
 * InlineAudioQueue.tsx — Inline audio generation queue display.
 * Shows pending, running, and completed audio generation jobs from the audioGenQueueStore.
 */

import React from 'react';
import { Loader2, CheckCircle2, XCircle, X, Music, Trash2, Clock } from 'lucide-react';
import {
  useAudioGenQueue,
  removeFromAudioQueue,
  clearFinishedFromAudioQueue,
  AudioQueueItem,
} from '../../../stores/audioGenQueueStore';

export const InlineAudioQueue: React.FC = () => {
  const { items } = useAudioGenQueue();

  const pendingCount = items.filter(i => i.status === 'pending').length;
  const runningItem = items.find(i => i.status === 'loading-adapter' || i.status === 'generating');
  const finishedCount = items.filter(i => i.status === 'succeeded' || i.status === 'failed').length;

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

      {/* Items */}
      {items.map(item => (
        <QueueItemRow key={item.id} item={item} />
      ))}
    </div>
  );
};

// ── Item Row ─────────────────────────────────────────────────────────────────

const QueueItemRow: React.FC<{ item: AudioQueueItem }> = ({ item }) => {
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
    <div className={`rounded-lg border ${borderColor} bg-white/[0.02] px-3 py-2 transition-all`}>
      <div className="flex items-center gap-2">
        {/* Status icon */}
        <div className="flex-shrink-0">
          {isPending && <div className="w-2 h-2 rounded-full bg-zinc-500" />}
          {isRunning && <Loader2 className="w-3.5 h-3.5 text-pink-400 animate-spin" />}
          {isSucceeded && <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />}
          {isFailed && <XCircle className="w-3.5 h-3.5 text-red-400" />}
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-zinc-200 truncate">
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
