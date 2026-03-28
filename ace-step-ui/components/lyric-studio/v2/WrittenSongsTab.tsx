import React, { useState } from 'react';
import { Plus, Trash2, Pencil, Music2, Wand2, Play, Loader2, ChevronDown, ChevronRight } from 'lucide-react';
import { lireekApi, Generation, Profile } from '../../../services/lyricStudioApi';

interface WrittenSongsTabProps {
  generations: Generation[];
  profiles: Profile[];
  onRefresh: () => void;
  onGenerateAudio: (gen: Generation) => void;
  showToast: (msg: string) => void;
}

export const WrittenSongsTab: React.FC<WrittenSongsTabProps> = ({
  generations, profiles, onRefresh, onGenerateAudio, showToast,
}) => {
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editValues, setEditValues] = useState<Partial<Generation>>({});
  const [generating, setGenerating] = useState(false);
  const [genCount, setGenCount] = useState(1);

  const handleQuickGenerate = async () => {
    if (profiles.length === 0) {
      showToast('Build a profile first');
      return;
    }
    setGenerating(true);
    try {
      const profile = profiles[0]; // Use most recent profile
      for (let i = 0; i < genCount; i++) {
        await lireekApi.generateLyrics(profile.id, {
          profile_id: profile.id,
          provider: profile.provider,
          model: profile.model,
        });
      }
      showToast(`Generated ${genCount} new song${genCount > 1 ? 's' : ''}`);
      onRefresh();
    } catch (err: any) {
      showToast(`Failed: ${err.message}`);
    } finally {
      setGenerating(false);
    }
  };

  const handleDelete = async (gen: Generation) => {
    if (!confirm(`Delete "${gen.title || 'Untitled'}"?`)) return;
    try {
      await lireekApi.deleteGeneration(gen.id);
      showToast('Deleted');
      onRefresh();
    } catch (err: any) {
      showToast(`Failed: ${err.message}`);
    }
  };

  const startEdit = (gen: Generation) => {
    setEditingId(gen.id);
    setEditValues({
      title: gen.title,
      caption: gen.caption,
      bpm: gen.bpm,
      key: gen.key,
      duration: gen.duration,
      subject: gen.subject,
    });
  };

  const saveEdit = async () => {
    if (!editingId) return;
    try {
      await lireekApi.updateMetadata(editingId, editValues);
      showToast('Saved');
      setEditingId(null);
      onRefresh();
    } catch (err: any) {
      showToast(`Failed: ${err.message}`);
    }
  };

  return (
    <div className="p-4 space-y-4">
      {/* Generate controls */}
      <div className="flex items-center gap-3 flex-wrap">
        <button
          onClick={handleQuickGenerate}
          disabled={generating || profiles.length === 0}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-sm font-semibold transition-all"
        >
          {generating ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Wand2 className="w-4 h-4" />
              Generate Songs
            </>
          )}
        </button>
        <div className="flex items-center gap-2">
          <label className="text-xs text-zinc-500">Count:</label>
          <select
            value={genCount}
            onChange={(e) => setGenCount(parseInt(e.target.value))}
            className="px-2 py-1.5 rounded-lg bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-emerald-500/50"
          >
            {[1, 2, 3, 4, 5, 8, 10].map(n => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </div>
        {profiles.length === 0 && (
          <span className="text-xs text-amber-400/60">Build a profile first</span>
        )}
      </div>

      {/* Generations list */}
      {generations.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="w-14 h-14 rounded-full bg-white/5 flex items-center justify-center mb-4">
            <Music2 className="w-7 h-7 text-zinc-600" />
          </div>
          <h3 className="text-base font-semibold text-zinc-400 mb-2">No written songs yet</h3>
          <p className="text-sm text-zinc-500 max-w-xs">
            Generate lyrics from a profile to see them here.
          </p>
        </div>
      ) : (
        <div className="space-y-1">
          {generations.map((gen) => {
            const isExpanded = expandedId === gen.id;
            const isEditing = editingId === gen.id;

            return (
              <div
                key={gen.id}
                className="rounded-xl border border-white/5 hover:border-white/10 overflow-hidden transition-colors"
              >
                {/* Header */}
                <button
                  className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-white/[0.02] transition-colors"
                  onClick={() => setExpandedId(isExpanded ? null : gen.id)}
                >
                  {isExpanded
                    ? <ChevronDown className="w-4 h-4 text-zinc-500 flex-shrink-0" />
                    : <ChevronRight className="w-4 h-4 text-zinc-500 flex-shrink-0" />
                  }
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-zinc-200 truncate">
                      {gen.title || 'Untitled'}
                    </p>
                    <p className="text-xs text-zinc-500 truncate mt-0.5">
                      {gen.caption?.slice(0, 80) || 'No caption'}
                    </p>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {gen.bpm ? (
                      <span className="text-[11px] text-zinc-500 font-mono">{gen.bpm} BPM</span>
                    ) : null}
                    {gen.key ? (
                      <span className="text-[11px] text-zinc-500 font-mono">{gen.key}</span>
                    ) : null}
                  </div>
                </button>

                {/* Expanded content */}
                {isExpanded && (
                  <div className="border-t border-white/5">
                    {isEditing ? (
                      /* Edit form */
                      <div className="p-4 space-y-3">
                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <label className="block text-xs text-zinc-500 mb-1">Title</label>
                            <input
                              value={editValues.title || ''}
                              onChange={(e) => setEditValues(v => ({ ...v, title: e.target.value }))}
                              className="w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-pink-500/50"
                            />
                          </div>
                          <div>
                            <label className="block text-xs text-zinc-500 mb-1">Subject</label>
                            <input
                              value={editValues.subject || ''}
                              onChange={(e) => setEditValues(v => ({ ...v, subject: e.target.value }))}
                              className="w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-pink-500/50"
                            />
                          </div>
                        </div>
                        <div>
                          <label className="block text-xs text-zinc-500 mb-1">Caption</label>
                          <textarea
                            value={editValues.caption || ''}
                            onChange={(e) => setEditValues(v => ({ ...v, caption: e.target.value }))}
                            rows={2}
                            className="w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-pink-500/50 resize-none"
                          />
                        </div>
                        <div className="grid grid-cols-3 gap-3">
                          <div>
                            <label className="block text-xs text-zinc-500 mb-1">BPM</label>
                            <input
                              type="number"
                              value={editValues.bpm || ''}
                              onChange={(e) => setEditValues(v => ({ ...v, bpm: parseInt(e.target.value) || undefined }))}
                              className="w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-sm text-white font-mono focus:outline-none focus:border-pink-500/50"
                            />
                          </div>
                          <div>
                            <label className="block text-xs text-zinc-500 mb-1">Key</label>
                            <input
                              value={editValues.key || ''}
                              onChange={(e) => setEditValues(v => ({ ...v, key: e.target.value }))}
                              className="w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-sm text-white font-mono focus:outline-none focus:border-pink-500/50"
                            />
                          </div>
                          <div>
                            <label className="block text-xs text-zinc-500 mb-1">Duration</label>
                            <input
                              type="number"
                              value={editValues.duration || ''}
                              onChange={(e) => setEditValues(v => ({ ...v, duration: parseInt(e.target.value) || undefined }))}
                              className="w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-sm text-white font-mono focus:outline-none focus:border-pink-500/50"
                            />
                          </div>
                        </div>
                        <div className="flex items-center gap-2 pt-2">
                          <button
                            onClick={saveEdit}
                            className="px-4 py-2 rounded-lg bg-pink-600 hover:bg-pink-500 text-white text-sm font-medium transition-colors"
                          >
                            Save
                          </button>
                          <button
                            onClick={() => setEditingId(null)}
                            className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-zinc-300 text-sm font-medium transition-colors"
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      /* Read view */
                      <div>
                        <pre className="px-4 py-3 text-sm text-zinc-300 whitespace-pre-wrap font-sans leading-relaxed max-h-80 overflow-y-auto">
                          {gen.lyrics}
                        </pre>
                        <div className="flex items-center gap-2 px-4 py-2 border-t border-white/5 bg-white/[0.01]">
                          <button
                            onClick={() => startEdit(gen)}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-zinc-300 hover:bg-white/5 transition-colors"
                          >
                            <Pencil className="w-3 h-3" />
                            Edit
                          </button>
                          <button
                            onClick={() => onGenerateAudio(gen)}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-emerald-400 hover:bg-emerald-500/10 transition-colors"
                          >
                            <Play className="w-3 h-3" />
                            Generate Audio
                          </button>
                          <div className="flex-1" />
                          <button
                            onClick={() => handleDelete(gen)}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-red-400 hover:bg-red-500/10 transition-colors"
                          >
                            <Trash2 className="w-3 h-3" />
                            Delete
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
