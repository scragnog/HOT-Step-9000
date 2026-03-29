import React, { useState } from 'react';
import { X, Loader2, CheckCircle, AlertCircle, ListOrdered, Sparkles, Wand2, RefreshCw, Trash2 } from 'lucide-react';
import {
  useStreamingStore,
  addBulkToQueue,
  removeFromQueue,
  clearQueue,
  QueueItem,
  QueueItemType,
} from '../../stores/streamingStore';
import { Artist, LyricsSet, Profile } from '../../services/lyricStudioApi';

interface QueuePanelProps {
  open: boolean;
  onClose: () => void;
  artists: Artist[];
  lyricsSets: LyricsSet[];
  profiles: Profile[];
  profilingModel: { provider: string; model?: string };
  generationModel: { provider: string; model?: string };
  refinementModel: { provider: string; model?: string };
}

type QueueMode = 'profile' | 'generate';

export const QueuePanel: React.FC<QueuePanelProps> = ({
  open, onClose, artists, lyricsSets, profiles,
  profilingModel, generationModel, refinementModel,
}) => {
  const stream = useStreamingStore();
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [mode, setMode] = useState<QueueMode>('profile');
  const [genCount, setGenCount] = useState(4);

  if (!open) return null;

  const toggleItem = (id: number) => {
    setSelected(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const selectAll = () => {
    if (mode === 'profile') {
      const profiledSetIds = new Set(profiles.map(p => p.lyrics_set_id));
      const ids = lyricsSets.filter(ls => !profiledSetIds.has(ls.id)).map(ls => ls.id);
      setSelected(new Set(ids));
    } else {
      setSelected(new Set(profiles.map(p => p.id)));
    }
  };

  const handleQueue = () => {
    if (selected.size === 0) return;

    if (mode === 'profile') {
      const items = Array.from(selected).map((lsId: number) => {
        const ls = lyricsSets.find(l => l.id === lsId);
        return {
          type: 'profile' as QueueItemType,
          targetId: lsId,
          label: `Profile: ${ls?.artist_name || '?'} — ${ls?.album || 'Unknown'}`,
          provider: profilingModel.provider,
          model: profilingModel.model,
        };
      });
      addBulkToQueue(items);
    } else {
      const items = Array.from(selected).map((profileId: number) => {
        const profile = profiles.find(p => p.id === profileId);
        const ls = lyricsSets.find(l => l.id === profile?.lyrics_set_id);
        return {
          type: 'generate' as QueueItemType,
          targetId: profileId,
          label: `Generate: ${ls?.artist_name || '?'} — ${ls?.album || 'Unknown'}`,
          provider: generationModel.provider,
          model: generationModel.model,
          count: genCount,
        };
      });
      addBulkToQueue(items);
    }

    setSelected(new Set());
    // Don't close; show queue progress
  };

  const queueItems = stream.queue;
  const pendingCount = queueItems.filter(q => q.status === 'pending').length;
  const runningItem = queueItems.find(q => q.status === 'running');
  const doneCount = queueItems.filter(q => q.status === 'done').length;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-zinc-900 rounded-2xl border border-white/10 shadow-2xl w-[600px] max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
          <div className="flex items-center gap-2">
            <ListOrdered className="w-5 h-5 text-pink-400" />
            <h2 className="text-lg font-bold text-white">Bulk Queue</h2>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/10 text-zinc-400 hover:text-white transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Mode tabs */}
        <div className="flex items-center gap-2 px-6 pt-4">
          <button
            onClick={() => { setMode('profile'); setSelected(new Set()); }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              mode === 'profile' ? 'bg-amber-500/20 text-amber-300' : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" /> Build Profiles
          </button>
          <button
            onClick={() => { setMode('generate'); setSelected(new Set()); }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              mode === 'generate' ? 'bg-green-500/20 text-green-300' : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'
            }`}
          >
            <Wand2 className="w-3.5 h-3.5" /> Generate Lyrics
          </button>
        </div>

        {/* Selection list */}
        <div className="flex-1 overflow-y-auto px-6 py-3 space-y-1 scrollbar-hide" style={{ maxHeight: '300px' }}>
          {mode === 'profile' ? (() => {
            // Only show albums that don't already have a profile
            const profiledSetIds = new Set(profiles.map(p => p.lyrics_set_id));
            const unprofiled = lyricsSets.filter(ls => !profiledSetIds.has(ls.id));
            return unprofiled.length === 0 ? (
              <p className="text-zinc-500 text-sm text-center py-4">
                {lyricsSets.length === 0 ? 'No albums available' : 'All albums already have profiles ✓'}
              </p>
            ) : (
              <>{unprofiled.map(ls => (
                <label
                  key={ls.id}
                  className={`flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                    selected.has(ls.id) ? 'bg-amber-500/10 border border-amber-500/20' : 'bg-white/5 border border-transparent hover:bg-white/10'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selected.has(ls.id)}
                    onChange={() => toggleItem(ls.id)}
                    className="accent-amber-500"
                  />
                  <div className="flex-1 min-w-0">
                    <span className="text-sm text-white truncate block">{ls.album || 'Unknown Album'}</span>
                    <span className="text-[10px] text-zinc-500">{ls.artist_name}</span>
                  </div>
                </label>
              ))}</>
            );
          })() : (
            profiles.length === 0 ? (
              <p className="text-zinc-500 text-sm text-center py-4">No profiles available — build profiles first</p>
            ) : (
              profiles.map(profile => {
                const ls = lyricsSets.find(l => l.id === profile.lyrics_set_id);
                return (
                  <label
                    key={profile.id}
                    className={`flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                      selected.has(profile.id) ? 'bg-green-500/10 border border-green-500/20' : 'bg-white/5 border border-transparent hover:bg-white/10'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selected.has(profile.id)}
                      onChange={() => toggleItem(profile.id)}
                      className="accent-green-500"
                    />
                    <div className="flex-1 min-w-0">
                      <span className="text-sm text-white truncate block">
                        {ls?.album || 'Unknown Album'} — {ls?.artist_name || '?'}
                      </span>
                      <span className="text-[10px] text-zinc-500">
                        {profile.provider}/{profile.model} · {new Date(profile.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </label>
                );
              })
            )
          )}
        </div>

        {/* Generation count (only for generate mode) */}
        {mode === 'generate' && (
          <div className="px-6 py-2 flex items-center gap-3">
            <span className="text-xs text-zinc-400">Generations per profile:</span>
            <input
              type="number"
              min={1}
              max={20}
              value={genCount}
              onChange={(e) => setGenCount(Math.max(1, Math.min(20, parseInt(e.target.value) || 1)))}
              className="w-16 px-2 py-1 rounded-lg bg-zinc-800 border border-white/10 text-sm text-white text-center focus:outline-none focus:border-green-500/50"
            />
          </div>
        )}

        {/* Action bar */}
        <div className="px-6 py-3 border-t border-white/5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button
              onClick={selectAll}
              className="px-3 py-1.5 rounded-lg text-xs text-zinc-400 hover:text-zinc-200 hover:bg-white/5 transition-colors"
            >
              Select All
            </button>
            <button
              onClick={() => setSelected(new Set())}
              className="px-3 py-1.5 rounded-lg text-xs text-zinc-400 hover:text-zinc-200 hover:bg-white/5 transition-colors"
            >
              Clear
            </button>
          </div>
          <button
            onClick={handleQueue}
            disabled={selected.size === 0}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold transition-all disabled:opacity-30 ${
              mode === 'profile'
                ? 'bg-amber-500 text-black hover:bg-amber-400'
                : 'bg-green-500 text-black hover:bg-green-400'
            }`}
          >
            <ListOrdered className="w-4 h-4" />
            Queue {selected.size} {mode === 'profile' ? 'Profile Build' : 'Generation Run'}{selected.size !== 1 ? 's' : ''}
          </button>
        </div>

        {/* Queue status */}
        {queueItems.length > 0 && (
          <div className="px-6 py-3 border-t border-white/5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Queue Progress</span>
              <button
                onClick={clearQueue}
                className="text-[10px] text-zinc-500 hover:text-red-400 transition-colors"
              >
                Clear Finished
              </button>
            </div>
            <div className="space-y-1 max-h-32 overflow-y-auto scrollbar-hide">
              {queueItems.map((item) => (
                <div key={item.id} className="flex items-center gap-2 px-2 py-1 rounded-lg bg-white/5 text-xs">
                  {item.status === 'pending' && <div className="w-2 h-2 rounded-full bg-zinc-500" />}
                  {item.status === 'running' && <Loader2 className="w-3 h-3 animate-spin text-pink-400" />}
                  {item.status === 'done' && <CheckCircle className="w-3 h-3 text-green-400" />}
                  {item.status === 'error' && <AlertCircle className="w-3 h-3 text-red-400" />}
                  <span className="text-zinc-300 flex-1 truncate">{item.label}</span>
                  {item.count && item.count > 1 && (
                    <span className="text-[10px] text-zinc-500">{item.countCompleted || 0}/{item.count}</span>
                  )}
                  {item.status === 'pending' && (
                    <button
                      onClick={() => removeFromQueue(item.id)}
                      className="p-0.5 rounded hover:bg-red-500/20 text-zinc-500 hover:text-red-400 transition-colors"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  )}
                </div>
              ))}
            </div>
            {(pendingCount > 0 || runningItem) && (
              <div className="mt-2 text-[10px] text-zinc-500">
                {runningItem ? `Running: ${runningItem.label}` : ''} 
                {pendingCount > 0 ? ` · ${pendingCount} pending` : ''}
                {doneCount > 0 ? ` · ${doneCount} done` : ''}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
