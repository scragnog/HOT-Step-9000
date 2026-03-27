import React, { useState, useEffect, useCallback } from 'react';
import {
  Music, Users, Disc3, FileText, Sparkles, ChevronRight, ChevronDown,
  Plus, Trash2, Download, RefreshCw, Loader2, Search, AlertTriangle, X, Wand2, Play, Settings2, Save,
} from 'lucide-react';
import { lireekApi, Artist, LyricsSet, Profile, Generation, SongLyric, AlbumPreset } from '../../services/lyricStudioApi';
import { ProviderSelector } from './ProviderSelector';

// ── Helpers ──────────────────────────────────────────────────────────────────

function parseSongs(songs: SongLyric[] | string): SongLyric[] {
  if (typeof songs === 'string') {
    try { return JSON.parse(songs); } catch { return []; }
  }
  return songs || [];
}

function timeAgo(dateStr: string): string {
  const d = new Date(dateStr);
  const now = new Date();
  const diff = Math.floor((now.getTime() - d.getTime()) / 1000);
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

// ── Main Component ──────────────────────────────────────────────────────────

export const LyricStudio: React.FC = () => {
  // ── Data state ──────────────────────────────────────────────────────────
  const [artists, setArtists] = useState<Artist[]>([]);
  const [lyricsSets, setLyricsSets] = useState<LyricsSet[]>([]);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [generations, setGenerations] = useState<Generation[]>([]);

  // ── UI state ────────────────────────────────────────────────────────────
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedArtists, setExpandedArtists] = useState<Set<number>>(new Set());
  const [expandedAlbums, setExpandedAlbums] = useState<Set<number>>(new Set());
  const [expandedProfiles, setExpandedProfiles] = useState<Set<number>>(new Set());

  // Selection
  const [selectedItem, setSelectedItem] = useState<{
    type: 'artist' | 'album' | 'profile' | 'generation' | 'fetch';
    id: number;
  } | null>(null);

  // Fetch panel
  const [fetchArtist, setFetchArtist] = useState('');
  const [fetchAlbum, setFetchAlbum] = useState('');
  const [fetchMaxSongs, setFetchMaxSongs] = useState(10);
  const [fetching, setFetching] = useState(false);

  // LLM provider
  const [provider, setProvider] = useState('gemini');
  const [model, setModel] = useState('');

  // Action state
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  // Album presets
  const [presets, setPresets] = useState<Record<number, AlbumPreset | null>>({});
  const [presetForm, setPresetForm] = useState<{
    adapter_path: string;
    adapter_scale: number;
    self_attn: number;
    cross_attn: number;
    mlp: number;
    matchering_reference_path: string;
  }>({
    adapter_path: '', adapter_scale: 1.0,
    self_attn: 1.0, cross_attn: 1.0, mlp: 1.0,
    matchering_reference_path: '',
  });

  // ── Load all data ───────────────────────────────────────────────────────
  const loadAll = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [a, ls, p, g] = await Promise.all([
        lireekApi.listArtists(),
        lireekApi.listLyricsSets(),
        lireekApi.listProfiles(),
        lireekApi.listAllGenerations(),
      ]);
      setArtists(a.artists);
      setLyricsSets(ls.lyrics_sets);
      setProfiles(p.profiles);
      setGenerations(g.generations);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadAll(); }, [loadAll]);

  // ── Toast helper ────────────────────────────────────────────────────────
  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  };

  // ── Tree toggle helpers ─────────────────────────────────────────────────
  const toggleArtist = (id: number) => {
    setExpandedArtists(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };
  const toggleAlbum = (id: number) => {
    setExpandedAlbums(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };
  const toggleProfile = (id: number) => {
    setExpandedProfiles(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  // ── Actions ─────────────────────────────────────────────────────────────
  const handleFetch = async () => {
    if (!fetchArtist.trim()) return;
    setFetching(true);
    try {
      const res = await lireekApi.fetchLyrics({
        artist: fetchArtist.trim(),
        album: fetchAlbum.trim() || undefined,
        max_songs: fetchMaxSongs,
      });
      showToast(`Fetched ${res.songs_fetched} songs for ${res.artist.name}`);
      setFetchArtist('');
      setFetchAlbum('');
      await loadAll();
      // Auto-expand the artist
      setExpandedArtists(prev => new Set(prev).add(res.artist.id));
    } catch (err) {
      showToast(`Fetch failed: ${(err as Error).message}`);
    } finally {
      setFetching(false);
    }
  };

  const handleBuildProfile = async (lyricsSetId: number) => {
    setActionLoading(`profile-${lyricsSetId}`);
    try {
      await lireekApi.buildProfile(lyricsSetId, { provider, model: model || undefined });
      showToast('Profile built successfully');
      await loadAll();
    } catch (err) {
      showToast(`Profile build failed: ${(err as Error).message}`);
    } finally {
      setActionLoading(null);
    }
  };

  const handleGenerate = async (profileId: number) => {
    setActionLoading(`generate-${profileId}`);
    try {
      await lireekApi.generateLyrics(profileId, {
        profile_id: profileId,
        provider,
        model: model || undefined,
      });
      showToast('Lyrics generated successfully');
      await loadAll();
    } catch (err) {
      showToast(`Generation failed: ${(err as Error).message}`);
    } finally {
      setActionLoading(null);
    }
  };

  const handleRefine = async (generationId: number) => {
    setActionLoading(`refine-${generationId}`);
    try {
      await lireekApi.refineLyrics(generationId, { provider, model: model || undefined });
      showToast('Lyrics refined successfully');
      await loadAll();
    } catch (err) {
      showToast(`Refinement failed: ${(err as Error).message}`);
    } finally {
      setActionLoading(null);
    }
  };

  const handleExport = async (generationId: number) => {
    setActionLoading(`export-${generationId}`);
    try {
      const res = await lireekApi.exportGeneration(generationId);
      showToast(`Exported to ${res.path}`);
    } catch (err) {
      showToast(`Export failed: ${(err as Error).message}`);
    } finally {
      setActionLoading(null);
    }
  };

  const handleDeleteArtist = async (id: number) => {
    if (!confirm('Delete this artist and all their data?')) return;
    try {
      await lireekApi.deleteArtist(id);
      showToast('Artist deleted');
      if (selectedItem?.type === 'artist' && selectedItem.id === id) setSelectedItem(null);
      await loadAll();
    } catch (err) {
      showToast(`Delete failed: ${(err as Error).message}`);
    }
  };

  const handleDeleteLyricsSet = async (id: number) => {
    if (!confirm('Delete this album/lyrics set?')) return;
    try {
      await lireekApi.deleteLyricsSet(id);
      showToast('Lyrics set deleted');
      if (selectedItem?.type === 'album' && selectedItem.id === id) setSelectedItem(null);
      await loadAll();
    } catch (err) {
      showToast(`Delete failed: ${(err as Error).message}`);
    }
  };

  const handleDeleteProfile = async (id: number) => {
    if (!confirm('Delete this profile and all generations from it?')) return;
    try {
      await lireekApi.deleteProfile(id);
      showToast('Profile deleted');
      if (selectedItem?.type === 'profile' && selectedItem.id === id) setSelectedItem(null);
      await loadAll();
    } catch (err) {
      showToast(`Delete failed: ${(err as Error).message}`);
    }
  };

  const handleDeleteGeneration = async (id: number) => {
    if (!confirm('Delete this generation?')) return;
    try {
      await lireekApi.deleteGeneration(id);
      showToast('Generation deleted');
      if (selectedItem?.type === 'generation' && selectedItem.id === id) setSelectedItem(null);
      await loadAll();
    } catch (err) {
      showToast(`Delete failed: ${(err as Error).message}`);
    }
  };

  // ── Album Preset Handlers ───────────────────────────────────────────────
  const loadPreset = async (lyricsSetId: number) => {
    try {
      const res = await lireekApi.getPreset(lyricsSetId);
      setPresets(prev => ({ ...prev, [lyricsSetId]: res.preset }));
      if (res.preset) {
        setPresetForm({
          adapter_path: res.preset.adapter_path || '',
          adapter_scale: res.preset.adapter_scale ?? 1.0,
          self_attn: res.preset.adapter_group_scales?.self_attn ?? 1.0,
          cross_attn: res.preset.adapter_group_scales?.cross_attn ?? 1.0,
          mlp: res.preset.adapter_group_scales?.mlp ?? 1.0,
          matchering_reference_path: res.preset.matchering_reference_path || '',
        });
      } else {
        setPresetForm({ adapter_path: '', adapter_scale: 1.0, self_attn: 1.0, cross_attn: 1.0, mlp: 1.0, matchering_reference_path: '' });
      }
    } catch (err) {
      showToast(`Failed to load preset: ${(err as Error).message}`);
    }
  };

  const savePreset = async (lyricsSetId: number) => {
    setActionLoading(`preset-${lyricsSetId}`);
    try {
      await lireekApi.upsertPreset(lyricsSetId, {
        adapter_path: presetForm.adapter_path || undefined,
        adapter_scale: presetForm.adapter_scale,
        adapter_group_scales: { self_attn: presetForm.self_attn, cross_attn: presetForm.cross_attn, mlp: presetForm.mlp },
        matchering_reference_path: presetForm.matchering_reference_path || undefined,
      });
      showToast('Preset saved');
      await loadPreset(lyricsSetId);
    } catch (err) {
      showToast(`Save failed: ${(err as Error).message}`);
    } finally {
      setActionLoading(null);
    }
  };

  // ── Generate Audio from Lyrics ──────────────────────────────────────────
  const handleGenerateAudio = async (gen: Generation) => {
    setActionLoading(`audio-${gen.id}`);
    try {
      // Find the album preset for this generation's profile -> lyrics_set
      const profile = profiles.find(p => p.id === gen.profile_id);
      let preset: AlbumPreset | null = null;
      if (profile) {
        const res = await lireekApi.getPreset(profile.lyrics_set_id);
        preset = res.preset;
      }

      const payload: any = {
        lyrics: gen.lyrics,
        prompt: gen.caption || '',
        audio_duration: gen.duration || 180,
      };
      if (gen.bpm) payload.bpm = gen.bpm;
      if (gen.key) payload.key_scale = gen.key;
      if (preset?.adapter_path) {
        payload.lireek_adapter_path = preset.adapter_path;
        payload.lireek_adapter_scale = preset.adapter_scale;
        if (preset.adapter_group_scales) {
          payload.lireek_group_scales = preset.adapter_group_scales;
        }
      }
      if (preset?.matchering_reference_path) {
        payload.mastering_params = { mode: 'matchering', reference_file: preset.matchering_reference_path };
      }

      const res = await lireekApi.submitAudioGeneration(payload);
      showToast(`Audio job queued: ${res.job_id}`);
      // Link the audio generation
      await lireekApi.linkAudio(gen.id, res.job_id);
    } catch (err) {
      showToast(`Audio generation failed: ${(err as Error).message}`);
    } finally {
      setActionLoading(null);
    }
  };

  // ── Get selected detail data ────────────────────────────────────────────
  const getSelectedLyricsSet = () => lyricsSets.find(ls => ls.id === selectedItem?.id);
  const getSelectedProfile = () => profiles.find(p => p.id === selectedItem?.id);
  const getSelectedGeneration = () => generations.find(g => g.id === selectedItem?.id);

  // ── Render ──────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-zinc-950 text-zinc-400">
        <Loader2 className="w-6 h-6 animate-spin mr-2" /> Loading Lyric Studio…
      </div>
    );
  }

  return (
    <div className="flex-1 flex h-full overflow-hidden bg-zinc-950">
      {/* ── Left Panel: Tree + Fetch ─────────────────────────────────────── */}
      <div className="w-80 flex-shrink-0 border-r border-white/5 flex flex-col h-full overflow-hidden">
        {/* Header */}
        <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Music className="w-5 h-5 text-pink-400" />
            <h2 className="text-base font-bold text-white">Lyric Studio</h2>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={loadAll}
              className="p-1.5 rounded-lg hover:bg-white/5 text-zinc-400 hover:text-white transition-colors"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button
              onClick={() => setSelectedItem({ type: 'fetch', id: 0 })}
              className="p-1.5 rounded-lg hover:bg-pink-500/20 text-pink-400 hover:text-pink-300 transition-colors"
              title="Fetch Lyrics"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Tree */}
        <div className="flex-1 overflow-y-auto scrollbar-hide py-2">
          {error && (
            <div className="mx-3 mb-2 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs flex items-center gap-2">
              <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
              {error}
            </div>
          )}

          {artists.length === 0 && !error && (
            <div className="px-4 py-8 text-center text-zinc-500 text-sm">
              <Music className="w-8 h-8 mx-auto mb-2 opacity-30" />
              No artists yet.<br />
              Click <span className="text-pink-400">+</span> to fetch lyrics from Genius.
            </div>
          )}

          {artists.map(artist => {
            const artistSets = lyricsSets.filter(ls => ls.artist_id === artist.id);
            const isExpanded = expandedArtists.has(artist.id);
            return (
              <div key={artist.id} className="mb-0.5">
                {/* Artist row */}
                <div
                  className={`flex items-center gap-1.5 px-3 py-1.5 cursor-pointer group transition-colors ${
                    selectedItem?.type === 'artist' && selectedItem.id === artist.id
                      ? 'bg-white/10 text-white' : 'text-zinc-300 hover:bg-white/5 hover:text-white'
                  }`}
                  onClick={() => {
                    toggleArtist(artist.id);
                    setSelectedItem({ type: 'artist', id: artist.id });
                  }}
                >
                  {isExpanded ? <ChevronDown className="w-3.5 h-3.5 text-zinc-500" /> : <ChevronRight className="w-3.5 h-3.5 text-zinc-500" />}
                  <Users className="w-4 h-4 text-purple-400 flex-shrink-0" />
                  <span className="text-sm font-medium truncate flex-1">{artist.name}</span>
                  <span className="text-[10px] text-zinc-600 group-hover:text-zinc-400">{artistSets.length}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDeleteArtist(artist.id); }}
                    className="p-0.5 rounded opacity-0 group-hover:opacity-100 hover:bg-red-500/20 text-zinc-500 hover:text-red-400 transition-all"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>

                {/* Albums (Lyrics Sets) */}
                {isExpanded && artistSets.map(ls => {
                  const albumProfiles = profiles.filter(p => p.lyrics_set_id === ls.id);
                  const isAlbumExpanded = expandedAlbums.has(ls.id);
                  const songs = parseSongs(ls.songs);
                  return (
                    <div key={ls.id}>
                      <div
                        className={`flex items-center gap-1.5 pl-7 pr-3 py-1.5 cursor-pointer group transition-colors ${
                          selectedItem?.type === 'album' && selectedItem.id === ls.id
                            ? 'bg-white/10 text-white' : 'text-zinc-400 hover:bg-white/5 hover:text-white'
                        }`}
                        onClick={() => {
                          toggleAlbum(ls.id);
                          setSelectedItem({ type: 'album', id: ls.id });
                        }}
                      >
                        {isAlbumExpanded ? <ChevronDown className="w-3 h-3 text-zinc-600" /> : <ChevronRight className="w-3 h-3 text-zinc-600" />}
                        <Disc3 className="w-3.5 h-3.5 text-blue-400 flex-shrink-0" />
                        <span className="text-sm truncate flex-1">{ls.album || 'Unknown Album'}</span>
                        <span className="text-[10px] text-zinc-600">{songs.length}🎵</span>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleDeleteLyricsSet(ls.id); }}
                          className="p-0.5 rounded opacity-0 group-hover:opacity-100 hover:bg-red-500/20 text-zinc-500 hover:text-red-400 transition-all"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>

                      {/* Profiles */}
                      {isAlbumExpanded && (
                        <>
                          {albumProfiles.length === 0 && (
                            <div className="pl-14 pr-3 py-1 text-[11px] text-zinc-600 italic">
                              No profiles yet
                            </div>
                          )}
                          {albumProfiles.map(profile => {
                            const profileGens = generations.filter(g => g.profile_id === profile.id);
                            const isProfileExpanded = expandedProfiles.has(profile.id);
                            return (
                              <div key={profile.id}>
                                <div
                                  className={`flex items-center gap-1.5 pl-11 pr-3 py-1 cursor-pointer group transition-colors ${
                                    selectedItem?.type === 'profile' && selectedItem.id === profile.id
                                      ? 'bg-white/10 text-white' : 'text-zinc-400 hover:bg-white/5 hover:text-white'
                                  }`}
                                  onClick={() => {
                                    toggleProfile(profile.id);
                                    setSelectedItem({ type: 'profile', id: profile.id });
                                  }}
                                >
                                  {isProfileExpanded ? <ChevronDown className="w-3 h-3 text-zinc-600" /> : <ChevronRight className="w-3 h-3 text-zinc-600" />}
                                  <Sparkles className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
                                  <span className="text-xs truncate flex-1">Profile ({profile.provider})</span>
                                  <span className="text-[10px] text-zinc-600">{profileGens.length}</span>
                                  <button
                                    onClick={(e) => { e.stopPropagation(); handleDeleteProfile(profile.id); }}
                                    className="p-0.5 rounded opacity-0 group-hover:opacity-100 hover:bg-red-500/20 text-zinc-500 hover:text-red-400 transition-all"
                                  >
                                    <Trash2 className="w-3 h-3" />
                                  </button>
                                </div>

                                {/* Generations */}
                                {isProfileExpanded && profileGens.map(gen => (
                                  <div
                                    key={gen.id}
                                    className={`flex items-center gap-1.5 pl-16 pr-3 py-1 cursor-pointer group transition-colors ${
                                      selectedItem?.type === 'generation' && selectedItem.id === gen.id
                                        ? 'bg-white/10 text-white' : 'text-zinc-500 hover:bg-white/5 hover:text-white'
                                    }`}
                                    onClick={() => setSelectedItem({ type: 'generation', id: gen.id })}
                                  >
                                    <FileText className="w-3.5 h-3.5 text-green-400 flex-shrink-0" />
                                    <span className="text-xs truncate flex-1">
                                      {gen.title || 'Untitled'}{gen.parent_generation_id ? ' ✨' : ''}
                                    </span>
                                    <span className="text-[10px] text-zinc-600">{timeAgo(gen.created_at)}</span>
                                    <button
                                      onClick={(e) => { e.stopPropagation(); handleDeleteGeneration(gen.id); }}
                                      className="p-0.5 rounded opacity-0 group-hover:opacity-100 hover:bg-red-500/20 text-zinc-500 hover:text-red-400 transition-all"
                                    >
                                      <Trash2 className="w-3 h-3" />
                                    </button>
                                  </div>
                                ))}
                              </div>
                            );
                          })}
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>

        {/* Provider selector at bottom of tree panel */}
        <div className="px-3 py-2 border-t border-white/5">
          <ProviderSelector
            selectedProvider={provider}
            selectedModel={model}
            onProviderChange={setProvider}
            onModelChange={setModel}
            compact
          />
        </div>
      </div>

      {/* ── Right Panel: Detail View ─────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto scrollbar-hide">
        {selectedItem?.type === 'fetch' && (
          <FetchPanel
            fetchArtist={fetchArtist}
            fetchAlbum={fetchAlbum}
            fetchMaxSongs={fetchMaxSongs}
            fetching={fetching}
            onArtistChange={setFetchArtist}
            onAlbumChange={setFetchAlbum}
            onMaxSongsChange={setFetchMaxSongs}
            onFetch={handleFetch}
            onClose={() => setSelectedItem(null)}
          />
        )}

        {selectedItem?.type === 'album' && (() => {
          const ls = getSelectedLyricsSet();
          if (!ls) return <EmptyDetail />;
          const songs = parseSongs(ls.songs);
          const albumProfiles = profiles.filter(p => p.lyrics_set_id === ls.id);
          return (
            <div className="p-6">
              <div className="flex items-center gap-3 mb-6">
                <Disc3 className="w-6 h-6 text-blue-400" />
                <div>
                  <h2 className="text-xl font-bold text-white">{ls.album || 'Unknown Album'}</h2>
                  <p className="text-sm text-zinc-400">{ls.artist_name} · {songs.length} songs</p>
                </div>
              </div>

              {/* Action bar */}
              <div className="flex items-center gap-2 mb-6">
                <button
                  onClick={() => handleBuildProfile(ls.id)}
                  disabled={actionLoading === `profile-${ls.id}`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 text-sm font-medium transition-colors disabled:opacity-50"
                >
                  {actionLoading === `profile-${ls.id}` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
                  Build Profile
                </button>
                <span className="text-xs text-zinc-500">{albumProfiles.length} profile(s)</span>
              </div>

              {/* Album Preset Panel */}
              <details className="mb-6" onToggle={(e) => { if ((e.target as HTMLDetailsElement).open) loadPreset(ls.id); }}>
                <summary className="flex items-center gap-2 text-sm font-semibold text-zinc-400 uppercase tracking-wider cursor-pointer hover:text-zinc-300 transition-colors">
                  <Settings2 className="w-4 h-4" /> Album Preset
                </summary>
                <div className="mt-3 p-4 rounded-xl bg-white/5 border border-white/5 space-y-3">
                  <div>
                    <label className="block text-[10px] text-zinc-500 uppercase tracking-wider mb-1">Adapter Path</label>
                    <input type="text" value={presetForm.adapter_path} onChange={e => setPresetForm(p => ({ ...p, adapter_path: e.target.value }))}
                      placeholder="D:\\path\\to\\adapter.safetensors" className="w-full px-2.5 py-1.5 rounded-lg bg-zinc-800 border border-white/10 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-pink-500/50" />
                  </div>
                  <div className="grid grid-cols-4 gap-2">
                    <div>
                      <label className="block text-[10px] text-zinc-500 uppercase mb-1">Scale</label>
                      <input type="number" step={0.1} min={0} max={2} value={presetForm.adapter_scale} onChange={e => setPresetForm(p => ({ ...p, adapter_scale: parseFloat(e.target.value) || 1 }))}
                        className="w-full px-2 py-1.5 rounded-lg bg-zinc-800 border border-white/10 text-sm text-white focus:outline-none focus:border-pink-500/50" />
                    </div>
                    <div>
                      <label className="block text-[10px] text-zinc-500 uppercase mb-1">Self-Attn</label>
                      <input type="number" step={0.1} min={0} max={2} value={presetForm.self_attn} onChange={e => setPresetForm(p => ({ ...p, self_attn: parseFloat(e.target.value) || 1 }))}
                        className="w-full px-2 py-1.5 rounded-lg bg-zinc-800 border border-white/10 text-sm text-white focus:outline-none focus:border-pink-500/50" />
                    </div>
                    <div>
                      <label className="block text-[10px] text-zinc-500 uppercase mb-1">Cross-Attn</label>
                      <input type="number" step={0.1} min={0} max={2} value={presetForm.cross_attn} onChange={e => setPresetForm(p => ({ ...p, cross_attn: parseFloat(e.target.value) || 1 }))}
                        className="w-full px-2 py-1.5 rounded-lg bg-zinc-800 border border-white/10 text-sm text-white focus:outline-none focus:border-pink-500/50" />
                    </div>
                    <div>
                      <label className="block text-[10px] text-zinc-500 uppercase mb-1">MLP</label>
                      <input type="number" step={0.1} min={0} max={2} value={presetForm.mlp} onChange={e => setPresetForm(p => ({ ...p, mlp: parseFloat(e.target.value) || 1 }))}
                        className="w-full px-2 py-1.5 rounded-lg bg-zinc-800 border border-white/10 text-sm text-white focus:outline-none focus:border-pink-500/50" />
                    </div>
                  </div>
                  <div>
                    <label className="block text-[10px] text-zinc-500 uppercase tracking-wider mb-1">Matchering Reference</label>
                    <input type="text" value={presetForm.matchering_reference_path} onChange={e => setPresetForm(p => ({ ...p, matchering_reference_path: e.target.value }))}
                      placeholder="D:\\path\\to\\reference.wav" className="w-full px-2.5 py-1.5 rounded-lg bg-zinc-800 border border-white/10 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-pink-500/50" />
                  </div>
                  <button onClick={() => savePreset(ls.id)} disabled={actionLoading === `preset-${ls.id}`}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-pink-500/20 text-pink-300 hover:bg-pink-500/30 text-sm font-medium transition-colors disabled:opacity-50">
                    {actionLoading === `preset-${ls.id}` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
                    Save Preset
                  </button>
                </div>
              </details>

              {/* Songs list */}
              <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">Songs</h3>
              <div className="space-y-1.5">
                {songs.map((song, i) => (
                  <div key={i} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors group">
                    <span className="text-xs text-zinc-600 w-5 text-right">{i + 1}</span>
                    <span className="text-sm text-white truncate flex-1">{song.title}</span>
                    <span className="text-[10px] text-zinc-600">{song.lyrics?.length || 0} chars</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })()}

        {selectedItem?.type === 'profile' && (() => {
          const profile = getSelectedProfile();
          if (!profile) return <EmptyDetail />;
          const pd = profile.profile_data || {};
          const profileGens = generations.filter(g => g.profile_id === profile.id);
          return (
            <div className="p-6">
              <div className="flex items-center gap-3 mb-6">
                <Sparkles className="w-6 h-6 text-amber-400" />
                <div>
                  <h2 className="text-xl font-bold text-white">{pd.artist || 'Profile'}</h2>
                  <p className="text-sm text-zinc-400">{profile.provider}/{profile.model} · {timeAgo(profile.created_at)}</p>
                </div>
              </div>

              {/* Action bar */}
              <div className="flex items-center gap-2 mb-6">
                <button
                  onClick={() => handleGenerate(profile.id)}
                  disabled={actionLoading === `generate-${profile.id}`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-green-500/20 text-green-300 hover:bg-green-500/30 text-sm font-medium transition-colors disabled:opacity-50"
                >
                  {actionLoading === `generate-${profile.id}` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Wand2 className="w-3.5 h-3.5" />}
                  Generate Lyrics
                </button>
                <span className="text-xs text-zinc-500">{profileGens.length} generation(s)</span>
              </div>

              {/* Profile highlights */}
              <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">Profile Analysis</h3>
              <div className="grid grid-cols-2 gap-3 mb-6">
                {pd.genres && (
                  <ProfileCard label="Genres" value={Array.isArray(pd.genres) ? pd.genres.join(', ') : pd.genres} />
                )}
                {pd.themes && (
                  <ProfileCard label="Themes" value={Array.isArray(pd.themes) ? pd.themes.join(', ') : pd.themes} />
                )}
                {pd.vocal_style && (
                  <ProfileCard label="Vocal Style" value={pd.vocal_style} />
                )}
                {pd.song_structures && (
                  <ProfileCard label="Song Structures" value={Array.isArray(pd.song_structures) ? pd.song_structures.join(', ') : pd.song_structures} />
                )}
                {pd.bpm_range && (
                  <ProfileCard label="BPM Range" value={pd.bpm_range} />
                )}
                {pd.key_preferences && (
                  <ProfileCard label="Key Preferences" value={Array.isArray(pd.key_preferences) ? pd.key_preferences.join(', ') : pd.key_preferences} />
                )}
              </div>

              {/* Full JSON toggle */}
              <details className="mb-4">
                <summary className="text-xs text-zinc-500 cursor-pointer hover:text-zinc-300 transition-colors">Show raw profile data</summary>
                <pre className="mt-2 p-3 rounded-lg bg-black/40 border border-white/5 text-xs text-zinc-400 overflow-auto max-h-64 whitespace-pre-wrap">
                  {JSON.stringify(pd, null, 2)}
                </pre>
              </details>
            </div>
          );
        })()}

        {selectedItem?.type === 'generation' && (() => {
          const gen = getSelectedGeneration();
          if (!gen) return <EmptyDetail />;
          return (
            <div className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <FileText className="w-6 h-6 text-green-400" />
                <div className="flex-1">
                  <h2 className="text-xl font-bold text-white">{gen.title || 'Untitled'}</h2>
                  <p className="text-sm text-zinc-400">
                    {gen.artist_name && `${gen.artist_name} · `}
                    {gen.provider}/{gen.model} · {timeAgo(gen.created_at)}
                    {gen.parent_generation_id && ' · ✨ Refined'}
                  </p>
                </div>
              </div>

              {/* Metadata badges */}
              <div className="flex flex-wrap gap-2 mb-4">
                {gen.bpm && <MetaBadge label="BPM" value={String(gen.bpm)} color="pink" />}
                {gen.key && <MetaBadge label="Key" value={gen.key} color="blue" />}
                {gen.duration && <MetaBadge label="Duration" value={`${gen.duration}s`} color="purple" />}
              </div>

              {/* Caption */}
              {gen.caption && (
                <div className="mb-4 px-3 py-2 rounded-lg bg-white/5 border border-white/5">
                  <span className="text-[10px] text-zinc-500 uppercase tracking-wider">Caption</span>
                  <p className="text-sm text-zinc-300 mt-0.5">{gen.caption}</p>
                </div>
              )}

              {/* Actions */}
              <div className="flex items-center gap-2 mb-6">
                <button
                  onClick={() => handleRefine(gen.id)}
                  disabled={actionLoading === `refine-${gen.id}`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-purple-500/20 text-purple-300 hover:bg-purple-500/30 text-sm font-medium transition-colors disabled:opacity-50"
                >
                  {actionLoading === `refine-${gen.id}` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Wand2 className="w-3.5 h-3.5" />}
                  Refine
                </button>
                <button
                  onClick={() => handleExport(gen.id)}
                  disabled={actionLoading === `export-${gen.id}`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-500/20 text-blue-300 hover:bg-blue-500/30 text-sm font-medium transition-colors disabled:opacity-50"
                >
                  {actionLoading === `export-${gen.id}` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />}
                  Export
                </button>
                <button
                  onClick={() => handleGenerateAudio(gen)}
                  disabled={actionLoading === `audio-${gen.id}`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-pink-500/30 to-purple-500/30 text-white hover:from-pink-500/40 hover:to-purple-500/40 text-sm font-semibold transition-all disabled:opacity-50 border border-pink-500/20"
                >
                  {actionLoading === `audio-${gen.id}` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                  Generate Audio
                </button>
                <button
                  onClick={() => handleDeleteGeneration(gen.id)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 text-sm font-medium transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Lyrics */}
              <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">Lyrics</h3>
              <pre className="p-4 rounded-xl bg-black/40 border border-white/5 text-sm text-zinc-200 whitespace-pre-wrap font-mono leading-relaxed max-h-[60vh] overflow-y-auto scrollbar-hide">
                {gen.lyrics}
              </pre>
            </div>
          );
        })()}

        {selectedItem?.type === 'artist' && (() => {
          const artist = artists.find(a => a.id === selectedItem.id);
          if (!artist) return <EmptyDetail />;
          const artistSets = lyricsSets.filter(ls => ls.artist_id === artist.id);
          const totalSongs = artistSets.reduce((sum, ls) => sum + parseSongs(ls.songs).length, 0);
          const totalProfiles = profiles.filter(p => artistSets.some(ls => ls.id === p.lyrics_set_id)).length;
          const totalGens = generations.filter(g => {
            const p = profiles.find(pp => pp.id === g.profile_id);
            return p && artistSets.some(ls => ls.id === p.lyrics_set_id);
          }).length;
          return (
            <div className="p-6">
              <div className="flex items-center gap-3 mb-6">
                <Users className="w-6 h-6 text-purple-400" />
                <div>
                  <h2 className="text-xl font-bold text-white">{artist.name}</h2>
                  <p className="text-sm text-zinc-400">Added {timeAgo(artist.created_at)}</p>
                </div>
              </div>
              <div className="grid grid-cols-4 gap-3">
                <StatCard label="Albums" value={artistSets.length} color="blue" />
                <StatCard label="Songs" value={totalSongs} color="green" />
                <StatCard label="Profiles" value={totalProfiles} color="amber" />
                <StatCard label="Generations" value={totalGens} color="pink" />
              </div>
            </div>
          );
        })()}

        {!selectedItem && <EmptyDetail />}
      </div>

      {/* Toast */}
      {toast && (
        <div className="fixed bottom-20 left-1/2 -translate-x-1/2 z-50 px-4 py-2 rounded-xl bg-zinc-800 border border-white/10 text-sm text-white shadow-xl animate-in fade-in slide-in-from-bottom-4 duration-200">
          {toast}
        </div>
      )}
    </div>
  );
};

// ── Sub-components ──────────────────────────────────────────────────────────

const EmptyDetail: React.FC = () => (
  <div className="flex-1 flex items-center justify-center h-full text-zinc-600">
    <div className="text-center">
      <Music className="w-12 h-12 mx-auto mb-3 opacity-20" />
      <p className="text-sm">Select an item from the tree<br />or click + to fetch lyrics</p>
    </div>
  </div>
);

const FetchPanel: React.FC<{
  fetchArtist: string;
  fetchAlbum: string;
  fetchMaxSongs: number;
  fetching: boolean;
  onArtistChange: (v: string) => void;
  onAlbumChange: (v: string) => void;
  onMaxSongsChange: (v: number) => void;
  onFetch: () => void;
  onClose: () => void;
}> = ({ fetchArtist, fetchAlbum, fetchMaxSongs, fetching, onArtistChange, onAlbumChange, onMaxSongsChange, onFetch, onClose }) => (
  <div className="p-6 max-w-lg">
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-2">
        <Search className="w-5 h-5 text-pink-400" />
        <h2 className="text-lg font-bold text-white">Fetch Lyrics from Genius</h2>
      </div>
      <button onClick={onClose} className="p-1 rounded-lg hover:bg-white/5 text-zinc-400 hover:text-white">
        <X className="w-4 h-4" />
      </button>
    </div>

    <div className="space-y-4">
      <div>
        <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider mb-1.5">Artist name or Genius URL</label>
        <input
          type="text"
          value={fetchArtist}
          onChange={e => onArtistChange(e.target.value)}
          placeholder="e.g. Blink-182 or https://genius.com/artists/..."
          className="w-full px-3 py-2 rounded-lg bg-zinc-800 border border-white/10 text-white text-sm placeholder-zinc-600 focus:outline-none focus:border-pink-500/50"
          onKeyDown={e => e.key === 'Enter' && onFetch()}
        />
      </div>
      <div>
        <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider mb-1.5">Album (optional)</label>
        <input
          type="text"
          value={fetchAlbum}
          onChange={e => onAlbumChange(e.target.value)}
          placeholder="e.g. Enema of the State"
          className="w-full px-3 py-2 rounded-lg bg-zinc-800 border border-white/10 text-white text-sm placeholder-zinc-600 focus:outline-none focus:border-pink-500/50"
          onKeyDown={e => e.key === 'Enter' && onFetch()}
        />
      </div>
      <div>
        <label className="block text-xs font-medium text-zinc-400 uppercase tracking-wider mb-1.5">Max songs</label>
        <input
          type="number"
          value={fetchMaxSongs}
          onChange={e => onMaxSongsChange(Math.max(1, Math.min(50, parseInt(e.target.value) || 10)))}
          min={1}
          max={50}
          className="w-24 px-3 py-2 rounded-lg bg-zinc-800 border border-white/10 text-white text-sm focus:outline-none focus:border-pink-500/50"
        />
      </div>
      <button
        onClick={onFetch}
        disabled={!fetchArtist.trim() || fetching}
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-pink-500 to-purple-600 text-white text-sm font-semibold hover:from-pink-600 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {fetching ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
        {fetching ? 'Fetching…' : 'Fetch Lyrics'}
      </button>
    </div>
  </div>
);

const ProfileCard: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="px-3 py-2 rounded-lg bg-white/5 border border-white/5">
    <span className="text-[10px] text-zinc-500 uppercase tracking-wider">{label}</span>
    <p className="text-sm text-zinc-200 mt-0.5 line-clamp-2">{value}</p>
  </div>
);

const MetaBadge: React.FC<{ label: string; value: string; color: string }> = ({ label, value, color }) => {
  const colors: Record<string, string> = {
    pink: 'bg-pink-500/20 text-pink-300 border-pink-500/20',
    blue: 'bg-blue-500/20 text-blue-300 border-blue-500/20',
    purple: 'bg-purple-500/20 text-purple-300 border-purple-500/20',
    green: 'bg-green-500/20 text-green-300 border-green-500/20',
    amber: 'bg-amber-500/20 text-amber-300 border-amber-500/20',
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium border ${colors[color] || colors.pink}`}>
      {label}: {value}
    </span>
  );
};

const StatCard: React.FC<{ label: string; value: number; color: string }> = ({ label, value, color }) => {
  const colors: Record<string, string> = {
    pink: 'text-pink-400',
    blue: 'text-blue-400',
    purple: 'text-purple-400',
    green: 'text-green-400',
    amber: 'text-amber-400',
  };
  return (
    <div className="px-4 py-3 rounded-xl bg-white/5 border border-white/5 text-center">
      <div className={`text-2xl font-bold ${colors[color] || 'text-white'}`}>{value}</div>
      <div className="text-[10px] text-zinc-500 uppercase tracking-wider mt-0.5">{label}</div>
    </div>
  );
};
