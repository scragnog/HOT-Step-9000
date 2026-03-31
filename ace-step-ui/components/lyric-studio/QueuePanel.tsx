import React, { useState, useEffect, useCallback } from 'react';
import {
  X, Loader2, CheckCircle, AlertCircle, ListOrdered, Sparkles,
  Wand2, Settings2, FolderSearch, ChevronDown, ChevronRight,
  RefreshCw, Zap, Music,
} from 'lucide-react';
import {
  useStreamingStore,
  addBulkToQueue,
  removeFromQueue,
  clearQueue,
  QueueItem,
  QueueItemType,
} from '../../stores/streamingStore';
import { Artist, LyricsSet, Profile, AlbumPreset, lireekApi } from '../../services/lyricStudioApi';
import { generateApi } from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { FileBrowserModal } from '../FileBrowserModal';
import { EditableSlider } from '../EditableSlider';

interface QueuePanelProps {
  open: boolean;
  onClose: () => void;
  artists: Artist[];
  lyricsSets: LyricsSet[];
  profiles: Profile[];
  profilingModel: { provider: string; model?: string };
  generationModel: { provider: string; model?: string };
  refinementModel: { provider: string; model?: string };
  showToast?: (msg: string) => void;
}

type QueueMode = 'profile' | 'generate' | 'presets';

// ── Preset status helpers ─────────────────────────────────────────────────────

type PresetStatus = 'complete' | 'partial' | 'missing';

function getPresetStatus(preset?: AlbumPreset | null): PresetStatus {
  if (!preset) return 'missing';
  const hasAdapter = !!preset.adapter_path;
  const hasRef = !!preset.matchering_reference_path;
  if (hasAdapter && hasRef) return 'complete';
  if (hasAdapter || hasRef) return 'partial';
  return 'missing';
}

const STATUS_BADGE: Record<PresetStatus, { label: string; color: string; icon: string }> = {
  complete: { label: 'PRESET', color: 'bg-green-900/30 text-green-400', icon: '✓' },
  partial:  { label: 'PARTIAL', color: 'bg-amber-900/30 text-amber-400', icon: '⚠' },
  missing:  { label: 'NONE', color: 'bg-red-900/30 text-red-400', icon: '✕' },
};

// ── Component ────────────────────────────────────────────────────────────────

export const QueuePanel: React.FC<QueuePanelProps> = ({
  open, onClose, artists, lyricsSets, profiles,
  profilingModel, generationModel, refinementModel, showToast,
}) => {
  const stream = useStreamingStore();
  const { token } = useAuth();
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [mode, setMode] = useState<QueueMode>('profile');
  const [genCount, setGenCount] = useState(4);

  // ── Presets state ──
  const [presetMap, setPresetMap] = useState<Map<number, AlbumPreset>>(new Map());
  const [presetsLoading, setPresetsLoading] = useState(false);
  const [applying, setApplying] = useState(false);

  // Preset form fields (what to bulk-apply)
  const [adapterPath, setAdapterPath] = useState('');
  const [matcheringPath, setMatcheringPath] = useState('');
  const [adapterScale, setAdapterScale] = useState(1.0);
  const [selfAttn, setSelfAttn] = useState(1.0);
  const [crossAttn, setCrossAttn] = useState(1.0);
  const [mlp, setMlp] = useState(1.0);
  const [groupsExpanded, setGroupsExpanded] = useState(false);

  // File browser
  const [browserOpen, setBrowserOpen] = useState(false);
  const [browserTarget, setBrowserTarget] = useState<'adapter' | 'matchering'>('adapter');

  // Adapter type detection
  const [detectedType, setDetectedType] = useState<'lokr' | 'lora' | 'unknown' | null>(null);

  if (!open) return null;

  // ── Load presets when tab switches ──
  const loadPresets = async () => {
    setPresetsLoading(true);
    try {
      const res = await lireekApi.listAllPresets();
      const map = new Map<number, AlbumPreset>();
      for (const p of res.presets) {
        map.set(p.lyrics_set_id, p);
      }
      setPresetMap(map);
    } catch (err) {
      console.error('[QueuePanel] Failed to load presets:', err);
    } finally {
      setPresetsLoading(false);
    }
  };

  // Detect adapter type on path change
  // eslint-disable-next-line react-hooks/rules-of-hooks
  useEffect(() => {
    if (!adapterPath.trim() || !token) {
      setDetectedType(null);
      return;
    }
    let cancelled = false;
    generateApi.detectAdapterType(adapterPath.trim(), token)
      .then(result => { if (!cancelled) setDetectedType(result.type); })
      .catch(() => { if (!cancelled) setDetectedType('unknown'); });
    return () => { cancelled = true; };
  }, [adapterPath, token]);

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
    } else if (mode === 'generate') {
      setSelected(new Set(profiles.map(p => p.id)));
    } else {
      // presets — select all albums
      setSelected(new Set(lyricsSets.map(ls => ls.id)));
    }
  };

  const selectMissing = () => {
    const ids = lyricsSets
      .filter(ls => getPresetStatus(presetMap.get(ls.id)) === 'missing')
      .map(ls => ls.id);
    setSelected(new Set(ids));
  };

  const selectIncomplete = () => {
    const ids = lyricsSets
      .filter(ls => {
        const status = getPresetStatus(presetMap.get(ls.id));
        return status === 'missing' || status === 'partial';
      })
      .map(ls => ls.id);
    setSelected(new Set(ids));
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
  };

  const handleApplyPresets = async () => {
    if (selected.size === 0) return;
    const hasAdapter = adapterPath.trim().length > 0;
    const hasRef = matcheringPath.trim().length > 0;
    if (!hasAdapter && !hasRef) {
      showToast?.('Set at least one path (adapter or reference) to apply');
      return;
    }

    setApplying(true);
    let success = 0;
    let failed = 0;

    const albumIds: number[] = Array.from(selected);
    for (const lsId of albumIds) {
      try {
        // Build partial update — only send fields that have values
        const params: any = {};
        if (hasAdapter) {
          params.adapter_path = adapterPath.trim();
          params.adapter_scale = adapterScale;
          params.adapter_group_scales = { self_attn: selfAttn, cross_attn: crossAttn, mlp };
        }
        if (hasRef) {
          params.matchering_reference_path = matcheringPath.trim();
        }

        // If we have adapter but no explicit scale fields, still send defaults
        if (hasAdapter && !params.adapter_scale) {
          params.adapter_scale = 1.0;
          params.adapter_group_scales = { self_attn: 1.0, cross_attn: 1.0, mlp: 1.0 };
        }

        await lireekApi.upsertPreset(lsId, params);
        success++;
      } catch (err) {
        failed++;
        console.error(`[QueuePanel] Failed to upsert preset for ls_id=${lsId}:`, err);
      }
    }

    showToast?.(
      failed === 0
        ? `Applied presets to ${success} album${success !== 1 ? 's' : ''}`
        : `Applied to ${success}, failed ${failed}`
    );
    setSelected(new Set());
    // Refresh preset map
    await loadPresets();
    setApplying(false);
  };

  const queueItems = stream.queue;
  const pendingCount = queueItems.filter(q => q.status === 'pending').length;
  const runningItem = queueItems.find(q => q.status === 'running');
  const doneCount = queueItems.filter(q => q.status === 'done').length;

  const adapterFileName = adapterPath ? adapterPath.split(/[\\/]/).pop() || '' : '';
  const matchFileName = matcheringPath ? matcheringPath.split(/[\\/]/).pop() || '' : '';
  const adapterTag = detectedType === 'lokr' ? 'LOKR' : detectedType === 'lora' ? 'LORA' : detectedType ? '...' : null;
  const adapterTagColor = detectedType === 'lokr' ? 'bg-purple-900/30 text-purple-400' : 'bg-pink-900/30 text-pink-400';

  // Stats for preset mode
  const presetStats = {
    complete: lyricsSets.filter(ls => getPresetStatus(presetMap.get(ls.id)) === 'complete').length,
    partial: lyricsSets.filter(ls => getPresetStatus(presetMap.get(ls.id)) === 'partial').length,
    missing: lyricsSets.filter(ls => getPresetStatus(presetMap.get(ls.id)) === 'missing').length,
  };

  return (
    <>
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
        <div className="bg-zinc-900 rounded-2xl border border-white/10 shadow-2xl w-[680px] max-h-[85vh] flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
            <div className="flex items-center gap-2">
              <ListOrdered className="w-5 h-5 text-pink-400" />
              <h2 className="text-lg font-bold text-white">Bulk Operations</h2>
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
            <button
              onClick={() => {
                setMode('presets');
                setSelected(new Set());
                loadPresets();
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                mode === 'presets' ? 'bg-pink-500/20 text-pink-300' : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'
              }`}
            >
              <Settings2 className="w-3.5 h-3.5" /> Assign Presets
            </button>
          </div>

          {/* ══════════ Presets mode config panel ══════════ */}
          {mode === 'presets' && (
            <div className="px-6 pt-3 pb-2 space-y-3 border-b border-white/5">
              {/* Summary badges */}
              <div className="flex items-center gap-3 text-[10px] font-semibold">
                <span className="text-green-400">{presetStats.complete} ✓ complete</span>
                <span className="text-amber-400">{presetStats.partial} ⚠ partial</span>
                <span className="text-red-400">{presetStats.missing} ✕ missing</span>
              </div>

              {/* Adapter path */}
              <div className="space-y-1.5">
                <div className="flex items-center gap-2 text-xs font-semibold text-zinc-300">
                  <Zap className="w-3.5 h-3.5 text-pink-400" />
                  Adapter to Apply
                </div>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={adapterPath}
                    onChange={e => setAdapterPath(e.target.value)}
                    placeholder="Path to .safetensors adapter file or folder"
                    className="flex-1 bg-black/20 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-pink-500 transition-colors"
                  />
                  <button
                    onClick={() => { setBrowserTarget('adapter'); setBrowserOpen(true); }}
                    className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-pink-900/20 text-pink-400 hover:bg-pink-900/30 transition-colors flex items-center gap-1 flex-shrink-0"
                  >
                    <FolderSearch size={12} /> Browse
                  </button>
                </div>
                {adapterPath && adapterTag && (
                  <div className="flex items-center gap-2">
                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${adapterTagColor}`}>{adapterTag}</span>
                    <span className="text-[10px] text-zinc-500 truncate" title={adapterPath}>{adapterFileName}</span>
                  </div>
                )}
              </div>

              {/* Adapter scale */}
              {adapterPath && (
                <div className="space-y-1">
                  <EditableSlider
                    label="Adapter Scale"
                    value={adapterScale}
                    min={0} max={2} step={0.05}
                    onChange={setAdapterScale}
                    formatDisplay={(v) => v.toFixed(2)}
                  />
                  <button
                    onClick={() => setGroupsExpanded(!groupsExpanded)}
                    className="flex items-center gap-1.5 text-[10px] font-semibold text-zinc-500 hover:text-zinc-300 transition-colors uppercase tracking-wider"
                  >
                    {groupsExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                    Group Scales
                  </button>
                  {groupsExpanded && (
                    <div className="space-y-1 pl-3 border-l-2 border-pink-500/20">
                      <EditableSlider
                        label="Self-Attn"
                        value={selfAttn}
                        min={0} max={2} step={0.05}
                        onChange={setSelfAttn}
                        formatDisplay={(v) => v.toFixed(2)}
                      />
                      <EditableSlider
                        label="Cross-Attn"
                        value={crossAttn}
                        min={0} max={2} step={0.05}
                        onChange={setCrossAttn}
                        formatDisplay={(v) => v.toFixed(2)}
                      />
                      <EditableSlider
                        label="MLP"
                        value={mlp}
                        min={0} max={2} step={0.05}
                        onChange={setMlp}
                        formatDisplay={(v) => v.toFixed(2)}
                      />
                    </div>
                  )}
                </div>
              )}

              {/* Matchering reference */}
              <div className="space-y-1.5">
                <div className="flex items-center gap-2 text-xs font-semibold text-zinc-300">
                  <Music className="w-3.5 h-3.5 text-amber-400" />
                  Matchering Reference to Apply
                </div>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={matcheringPath}
                    onChange={e => setMatcheringPath(e.target.value)}
                    placeholder="Path to reference audio (.wav, .mp3, .flac)"
                    className="flex-1 bg-black/20 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-amber-500 transition-colors"
                  />
                  <button
                    onClick={() => { setBrowserTarget('matchering'); setBrowserOpen(true); }}
                    className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-amber-900/20 text-amber-400 hover:bg-amber-900/30 transition-colors flex items-center gap-1 flex-shrink-0"
                  >
                    <FolderSearch size={12} /> Browse
                  </button>
                </div>
                {matcheringPath && (
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-900/30 text-amber-400">REF</span>
                    <span className="text-[10px] text-zinc-500 truncate" title={matcheringPath}>{matchFileName}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Selection list */}
          <div className="flex-1 overflow-y-auto px-6 py-3 space-y-1 scrollbar-hide" style={{ maxHeight: mode === 'presets' ? '250px' : '300px' }}>
            {mode === 'profile' ? (() => {
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
            })() : mode === 'generate' ? (
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
            ) : (
              /* ── Presets mode album list ── */
              presetsLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-5 h-5 text-zinc-500 animate-spin" />
                </div>
              ) : lyricsSets.length === 0 ? (
                <p className="text-zinc-500 text-sm text-center py-4">No albums available</p>
              ) : (
                lyricsSets.map(ls => {
                  const preset = presetMap.get(ls.id);
                  const status = getPresetStatus(preset);
                  const badge = STATUS_BADGE[status];
                  return (
                    <label
                      key={ls.id}
                      className={`flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                        selected.has(ls.id) ? 'bg-pink-500/10 border border-pink-500/20' : 'bg-white/5 border border-transparent hover:bg-white/10'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={selected.has(ls.id)}
                        onChange={() => toggleItem(ls.id)}
                        className="accent-pink-500"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-white truncate">{ls.album || 'Top Songs'}</span>
                          <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${badge.color}`}>
                            {badge.icon} {badge.label}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 mt-0.5">
                          <span className="text-[10px] text-zinc-500">{ls.artist_name}</span>
                          {preset?.adapter_path && (
                            <span className="text-[9px] text-zinc-600 truncate max-w-[120px]" title={preset.adapter_path}>
                              🔌 {preset.adapter_path.split(/[\\/]/).pop()}
                            </span>
                          )}
                          {preset?.matchering_reference_path && (
                            <span className="text-[9px] text-zinc-600 truncate max-w-[120px]" title={preset.matchering_reference_path}>
                              🎵 {preset.matchering_reference_path.split(/[\\/]/).pop()}
                            </span>
                          )}
                        </div>
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
              {mode === 'presets' && (
                <>
                  <button
                    onClick={selectMissing}
                    className="px-3 py-1.5 rounded-lg text-xs text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-colors"
                  >
                    Select Missing
                  </button>
                  <button
                    onClick={selectIncomplete}
                    className="px-3 py-1.5 rounded-lg text-xs text-amber-400 hover:text-amber-300 hover:bg-amber-500/10 transition-colors"
                  >
                    Select Incomplete
                  </button>
                </>
              )}
              <button
                onClick={() => setSelected(new Set())}
                className="px-3 py-1.5 rounded-lg text-xs text-zinc-400 hover:text-zinc-200 hover:bg-white/5 transition-colors"
              >
                Clear
              </button>
            </div>
            {mode === 'presets' ? (
              <div className="flex items-center gap-2">
                <button
                  onClick={loadPresets}
                  disabled={presetsLoading}
                  className="p-2 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-white/5 transition-colors disabled:opacity-50"
                  title="Refresh preset data"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${presetsLoading ? 'animate-spin' : ''}`} />
                </button>
                <button
                  onClick={handleApplyPresets}
                  disabled={selected.size === 0 || applying || (!adapterPath.trim() && !matcheringPath.trim())}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold transition-all disabled:opacity-30 bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 text-white shadow-lg shadow-pink-500/10"
                >
                  {applying ? <Loader2 className="w-4 h-4 animate-spin" /> : <Settings2 className="w-4 h-4" />}
                  Apply to {selected.size} Album{selected.size !== 1 ? 's' : ''}
                </button>
              </div>
            ) : (
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
            )}
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

      {/* File Browser sub-modal */}
      <FileBrowserModal
        open={browserOpen}
        onClose={() => setBrowserOpen(false)}
        onSelect={(path) => {
          if (browserTarget === 'adapter') {
            setAdapterPath(path);
          } else {
            setMatcheringPath(path);
          }
          setBrowserOpen(false);
        }}
        mode="file"
        filter={browserTarget === 'matchering' ? 'audio' : 'adapters'}
        title={browserTarget === 'matchering' ? 'Select Reference Audio' : 'Select Adapter File'}
      />
    </>
  );
};
