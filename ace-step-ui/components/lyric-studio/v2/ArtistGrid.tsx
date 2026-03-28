import React, { useState } from 'react';
import { Music, Plus, RefreshCw, Trash2, ImageOff, MoreVertical } from 'lucide-react';
import { Artist } from '../../../services/lyricStudioApi';

interface ArtistGridProps {
  artists: Artist[];
  loading: boolean;
  onSelectArtist: (artist: Artist) => void;
  onAddNew: () => void;
  onDelete: (artist: Artist) => void;
  onRefreshImage: (artist: Artist) => void;
}

export const ArtistGrid: React.FC<ArtistGridProps> = ({
  artists, loading, onSelectArtist, onAddNew, onDelete, onRefreshImage,
}) => {
  const [menuOpenId, setMenuOpenId] = useState<number | null>(null);
  const [imageErrors, setImageErrors] = useState<Set<number>>(new Set());

  const gradient = (name: string) => {
    const hash = name.split('').reduce((a, c) => ((a << 5) - a + c.charCodeAt(0)) | 0, 0);
    const h1 = Math.abs(hash) % 360;
    const h2 = (h1 + 40) % 360;
    return `linear-gradient(135deg, hsl(${h1}, 70%, 35%), hsl(${h2}, 60%, 25%))`;
  };

  if (loading) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4 p-6">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="aspect-[3/4] rounded-2xl bg-white/5 animate-pulse" />
        ))}
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <Music className="w-7 h-7 text-pink-400" />
          Lyric Studio
        </h1>
        <button
          onClick={onAddNew}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-pink-600 hover:bg-pink-500 text-white text-sm font-semibold transition-all hover:scale-105 shadow-lg shadow-pink-600/20"
        >
          <Plus className="w-4 h-4" />
          Fetch Lyrics
        </button>
      </div>

      {artists.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-4">
            <Music className="w-10 h-10 text-zinc-600" />
          </div>
          <h2 className="text-lg font-semibold text-zinc-400 mb-2">No artists yet</h2>
          <p className="text-sm text-zinc-500 max-w-sm mb-6">
            Start by fetching lyrics from Genius to build your artist library.
          </p>
          <button
            onClick={onAddNew}
            className="flex items-center gap-2 px-5 py-3 rounded-xl bg-pink-600 hover:bg-pink-500 text-white text-sm font-semibold transition-all"
          >
            <Plus className="w-4 h-4" />
            Fetch Your First Album
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
          {artists.map((artist) => (
            <div
              key={artist.id}
              className="group relative aspect-[3/4] rounded-2xl overflow-hidden cursor-pointer transition-all duration-300 hover:scale-[1.03] hover:shadow-2xl hover:shadow-pink-500/10"
              onClick={() => onSelectArtist(artist)}
            >
              {/* Background image or gradient */}
              {artist.image_url && !imageErrors.has(artist.id) ? (
                <img
                  src={artist.image_url}
                  alt={artist.name}
                  className="absolute inset-0 w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                  onError={() => setImageErrors(prev => new Set(prev).add(artist.id))}
                />
              ) : (
                <div
                  className="absolute inset-0 flex items-center justify-center"
                  style={{ background: gradient(artist.name) }}
                >
                  <span className="text-5xl font-black text-white/20 select-none">
                    {artist.name.charAt(0).toUpperCase()}
                  </span>
                </div>
              )}

              {/* Dark overlay gradient */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/30 to-transparent" />

              {/* Content */}
              <div className="absolute inset-x-0 bottom-0 p-4">
                <h3 className="text-base font-bold text-white truncate mb-1 drop-shadow-lg">
                  {artist.name}
                </h3>
                <p className="text-xs text-zinc-300/80">
                  {artist.lyrics_set_count ?? 0} album{(artist.lyrics_set_count ?? 0) !== 1 ? 's' : ''}
                </p>
              </div>

              {/* Hover glow ring */}
              <div className="absolute inset-0 rounded-2xl ring-1 ring-white/10 group-hover:ring-pink-500/40 transition-all duration-300" />

              {/* Context menu button */}
              <button
                className="absolute top-2 right-2 p-1.5 rounded-lg bg-black/50 text-white/60 hover:text-white hover:bg-black/70 opacity-0 group-hover:opacity-100 transition-all z-10"
                onClick={(e) => {
                  e.stopPropagation();
                  setMenuOpenId(menuOpenId === artist.id ? null : artist.id);
                }}
              >
                <MoreVertical className="w-4 h-4" />
              </button>

              {/* Context menu */}
              {menuOpenId === artist.id && (
                <div
                  className="absolute top-10 right-2 z-20 min-w-[160px] rounded-xl bg-zinc-900 border border-white/10 shadow-2xl py-1 animate-in fade-in slide-in-from-top-1"
                  onClick={(e) => e.stopPropagation()}
                >
                  <button
                    className="w-full flex items-center gap-2 px-3 py-2 text-sm text-zinc-300 hover:bg-white/5 hover:text-white transition-colors"
                    onClick={() => { onRefreshImage(artist); setMenuOpenId(null); }}
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    Refresh Image
                  </button>
                  <div className="border-t border-white/5 my-1" />
                  <button
                    className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 hover:text-red-300 transition-colors"
                    onClick={() => { onDelete(artist); setMenuOpenId(null); }}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    Delete Artist
                  </button>
                </div>
              )}
            </div>
          ))}

          {/* Add new card */}
          <div
            className="aspect-[3/4] rounded-2xl border-2 border-dashed border-white/10 hover:border-pink-500/30 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 hover:bg-white/[0.02] group"
            onClick={onAddNew}
          >
            <div className="w-12 h-12 rounded-full bg-white/5 group-hover:bg-pink-500/10 flex items-center justify-center mb-3 transition-colors">
              <Plus className="w-6 h-6 text-zinc-500 group-hover:text-pink-400 transition-colors" />
            </div>
            <span className="text-sm text-zinc-500 group-hover:text-zinc-300 font-medium transition-colors">
              Add Artist
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
