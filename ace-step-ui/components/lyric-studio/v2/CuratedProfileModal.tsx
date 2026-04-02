import React, { useState, useEffect, useCallback } from 'react';
import { X, Loader2, Sparkles, ChevronDown, ChevronRight, Check, Disc3, FileText, CheckSquare, Square } from 'lucide-react';
import { lireekApi, LyricsSet, SongLyric, Profile } from '../../../services/lyricStudioApi';
import { streamBuildCuratedProfile, StreamCallbacks } from '../../../services/lyricStudioApi';
import { loadSelections } from '../ProviderSelector';

function parseSongs(songs: SongLyric[] | string): SongLyric[] {
  if (typeof songs === 'string') {
    try { return JSON.parse(songs); } catch { return []; }
  }
  return songs || [];
}

interface CuratedProfileModalProps {
  isOpen: boolean;
  onClose: () => void;
  artistId: number;
  artistName: string;
  albums: LyricsSet[];
  showToast: (msg: string) => void;
  onComplete: (lyricsSet: LyricsSet, profile: Profile) => void;
}

// Track selected songs per album
type SelectionMap = Record<number, Set<number>>;

export const CuratedProfileModal: React.FC<CuratedProfileModalProps> = ({
  isOpen, onClose, artistId, artistName, albums, showToast, onComplete,
}) => {
  const [expandedAlbumId, setExpandedAlbumId] = useState<number | null>(null);
  const [selections, setSelections] = useState<SelectionMap>({});
  const [building, setBuilding] = useState(false);
  const [streamPhase, setStreamPhase] = useState('');
  const [streamText, setStreamText] = useState('');

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setSelections({});
      setExpandedAlbumId(null);
      setBuilding(false);
      setStreamPhase('');
      setStreamText('');
    }
  }, [isOpen]);

  // Load full album data for songs when modal opens
  const [fullAlbums, setFullAlbums] = useState<LyricsSet[]>([]);
  const [loadingAlbums, setLoadingAlbums] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    const load = async () => {
      setLoadingAlbums(true);
      try {
        const results = await Promise.all(
          albums.map(a => lireekApi.getLyricsSet(a.id))
        );
        setFullAlbums(results);
      } catch (err) {
        console.error('Failed to load album details:', err);
      } finally {
        setLoadingAlbums(false);
      }
    };
    load();
  }, [isOpen, albums]);

  const toggleSong = useCallback((albumId: number, songIdx: number) => {
    setSelections(prev => {
      const next = { ...prev };
      const set = new Set(next[albumId] || []);
      if (set.has(songIdx)) {
        set.delete(songIdx);
      } else {
        set.add(songIdx);
      }
      next[albumId] = set;
      return next;
    });
  }, []);

  const toggleAlbum = useCallback((albumId: number) => {
    const album = fullAlbums.find(a => a.id === albumId);
    if (!album) return;
    const songs = parseSongs(album.songs);
    setSelections(prev => {
      const next = { ...prev };
      const currentSet = next[albumId] || new Set();
      if (currentSet.size === songs.length) {
        // Deselect all
        next[albumId] = new Set();
      } else {
        // Select all
        next[albumId] = new Set(songs.map((_, i) => i));
      }
      return next;
    });
  }, [fullAlbums]);

  const totalSelected = Object.values(selections).reduce((sum, set) => sum + set.size, 0);

  const handleBuild = useCallback(async () => {
    if (totalSelected === 0) return;
    setBuilding(true);
    setStreamPhase('Preparing curated selection…');
    setStreamText('');

    const selectionList = Object.entries(selections)
      .filter(([_, set]) => set.size > 0)
      .map(([albumId, set]) => ({
        lyrics_set_id: Number(albumId),
        song_indices: Array.from(set).sort((a, b) => a - b),
      }));

    const { profiling } = loadSelections();

    try {
      await streamBuildCuratedProfile(
        artistId,
        {
          selections: selectionList,
          provider: profiling.provider,
          model: profiling.model || undefined,
        },
        {
          onChunk: (text) => setStreamText(prev => prev + text),
          onPhase: (phase) => {
            setStreamPhase(phase);
            setStreamText('');
          },
          onResult: (data) => {
            setBuilding(false);
            showToast('Curated profile built successfully!');
            onComplete(data.lyrics_set, data.profile);
            onClose();
          },
          onError: (message) => {
            setBuilding(false);
            showToast(`Profile build failed: ${message}`);
          },
        } as StreamCallbacks
      );
    } catch (err: any) {
      setBuilding(false);
      showToast(`Build failed: ${err.message}`);
    }
  }, [totalSelected, selections, artistId, showToast, onComplete, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div
        className="bg-zinc-900 border border-white/10 rounded-2xl shadow-2xl w-full max-w-2xl max-h-[85vh] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/5 flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-purple-500/10 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-purple-400" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Build Curated Profile</h2>
              <p className="text-xs text-zinc-500">{artistName} — Pick songs from any album</p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={building}
            className="p-1.5 rounded-lg hover:bg-white/5 text-zinc-400 hover:text-white transition-colors disabled:opacity-50"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Album list with songs */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {loadingAlbums ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-6 h-6 text-zinc-500 animate-spin" />
            </div>
          ) : fullAlbums.length === 0 ? (
            <div className="text-center py-12 text-zinc-500">
              No albums found for this artist.
            </div>
          ) : (
            fullAlbums.map(album => {
              const songs = parseSongs(album.songs);
              const isExpanded = expandedAlbumId === album.id;
              const selectedInAlbum = selections[album.id]?.size || 0;
              const allSelected = selectedInAlbum === songs.length && songs.length > 0;

              return (
                <div key={album.id} className="rounded-xl border border-white/5 overflow-hidden">
                  {/* Album header */}
                  <div
                    className="flex items-center gap-3 px-4 py-3 hover:bg-white/[0.02] cursor-pointer transition-colors"
                    onClick={() => setExpandedAlbumId(isExpanded ? null : album.id)}
                  >
                    {isExpanded
                      ? <ChevronDown className="w-4 h-4 text-zinc-500 flex-shrink-0" />
                      : <ChevronRight className="w-4 h-4 text-zinc-500 flex-shrink-0" />
                    }
                    <Disc3 className="w-4 h-4 text-zinc-400 flex-shrink-0" />
                    <span className="flex-1 text-sm font-medium text-zinc-200 truncate">
                      {album.album || 'Top Songs'}
                    </span>
                    {selectedInAlbum > 0 && (
                      <span className="px-2 py-0.5 rounded-full text-[11px] font-bold bg-purple-500/20 text-purple-300">
                        {selectedInAlbum}/{songs.length}
                      </span>
                    )}
                    <span className="text-xs text-zinc-500">{songs.length} songs</span>
                  </div>

                  {/* Expanded songs */}
                  {isExpanded && (
                    <div className="border-t border-white/5">
                      {/* Select/deselect all */}
                      <button
                        className="w-full flex items-center gap-2 px-4 py-2 text-xs text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.02] transition-colors"
                        onClick={(e) => { e.stopPropagation(); toggleAlbum(album.id); }}
                      >
                        {allSelected
                          ? <CheckSquare className="w-3.5 h-3.5 text-purple-400" />
                          : <Square className="w-3.5 h-3.5" />
                        }
                        {allSelected ? 'Deselect All' : 'Select All'}
                      </button>

                      {songs.map((song, idx) => {
                        const isSelected = selections[album.id]?.has(idx) || false;
                        return (
                          <button
                            key={idx}
                            className={`w-full flex items-center gap-3 px-4 py-2 text-left hover:bg-white/[0.02] transition-colors ${
                              isSelected ? 'bg-purple-500/5' : ''
                            }`}
                            onClick={() => toggleSong(album.id, idx)}
                          >
                            {isSelected
                              ? <Check className="w-3.5 h-3.5 text-purple-400 flex-shrink-0" />
                              : <div className="w-3.5 h-3.5 rounded border border-white/20 flex-shrink-0" />
                            }
                            <FileText className="w-3 h-3 text-zinc-500 flex-shrink-0" />
                            <span className={`text-sm truncate ${isSelected ? 'text-white' : 'text-zinc-400'}`}>
                              {song.title}
                            </span>
                            <span className="text-[11px] text-zinc-600 ml-auto flex-shrink-0">
                              {(song.lyrics || '').split('\n').length} lines
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })
          )}

          {/* Streaming progress */}
          {building && (
            <div className="rounded-xl border border-purple-500/20 bg-purple-500/5 p-4 mt-4">
              <div className="flex items-center gap-2 mb-2">
                <Loader2 className="w-4 h-4 text-purple-400 animate-spin" />
                <span className="text-sm font-medium text-purple-300">{streamPhase}</span>
              </div>
              {streamText && (
                <pre className="text-xs text-zinc-400 whitespace-pre-wrap max-h-32 overflow-y-auto font-mono leading-relaxed">
                  {streamText}
                </pre>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-white/5 bg-zinc-900/80 flex-shrink-0">
          <span className="text-sm text-zinc-400">
            {totalSelected > 0 ? (
              <><strong className="text-white">{totalSelected}</strong> songs selected across{' '}
                <strong className="text-white">
                  {Object.values(selections).filter(s => s.size > 0).length}
                </strong> albums</>
            ) : (
              'Select songs to build a curated profile'
            )}
          </span>
          <button
            onClick={handleBuild}
            disabled={totalSelected === 0 || building}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-purple-600 hover:bg-purple-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-sm font-semibold transition-all"
          >
            {building ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Building…
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                Build Profile
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
