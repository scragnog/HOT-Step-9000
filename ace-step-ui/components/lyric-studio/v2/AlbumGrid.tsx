import React from 'react';
import { Disc3, Plus, FileText, Users } from 'lucide-react';
import { LyricsSet, SongLyric } from '../../../services/lyricStudioApi';

function parseSongs(songs: SongLyric[] | string): SongLyric[] {
  if (typeof songs === 'string') {
    try { return JSON.parse(songs); } catch { return []; }
  }
  return songs || [];
}

interface AlbumGridProps {
  albums: LyricsSet[];
  loading: boolean;
  artistName: string;
  onSelectAlbum: (album: LyricsSet) => void;
  onAddAlbum: () => void;
  onDeleteAlbum: (album: LyricsSet) => void;
}

export const AlbumGrid: React.FC<AlbumGridProps> = ({
  albums, loading, artistName, onSelectAlbum, onAddAlbum, onDeleteAlbum,
}) => {
  const gradient = (name: string) => {
    const hash = (name || 'album').split('').reduce((a, c) => ((a << 5) - a + c.charCodeAt(0)) | 0, 0);
    const h1 = Math.abs(hash) % 360;
    const h2 = (h1 + 30) % 360;
    return `linear-gradient(135deg, hsl(${h1}, 50%, 25%), hsl(${h2}, 40%, 18%))`;
  };

  if (loading) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 p-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="aspect-square rounded-2xl bg-white/5 animate-pulse" />
        ))}
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-white">Albums</h2>
          <p className="text-sm text-zinc-400 mt-0.5">{artistName}</p>
        </div>
        <button
          onClick={onAddAlbum}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-pink-600 hover:bg-pink-500 text-white text-sm font-semibold transition-all hover:scale-105 shadow-lg shadow-pink-600/20"
        >
          <Plus className="w-4 h-4" />
          Fetch Album
        </button>
      </div>

      {albums.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mb-4">
            <Disc3 className="w-8 h-8 text-zinc-600" />
          </div>
          <h3 className="text-base font-semibold text-zinc-400 mb-2">No albums yet</h3>
          <p className="text-sm text-zinc-500 max-w-xs mb-4">
            Fetch lyrics for an album to get started.
          </p>
          <button
            onClick={onAddAlbum}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-pink-600 hover:bg-pink-500 text-white text-sm font-semibold transition-all"
          >
            <Plus className="w-4 h-4" />
            Fetch Album
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
          {albums.map((album) => {
            const songs = parseSongs(album.songs);
            return (
              <div
                key={album.id}
                className="group relative aspect-square rounded-2xl overflow-hidden cursor-pointer transition-all duration-300 hover:scale-[1.03] hover:shadow-2xl hover:shadow-indigo-500/10"
                onClick={() => onSelectAlbum(album)}
              >
                {/* Album gradient background */}
                <div
                  className="absolute inset-0"
                  style={{ background: gradient(album.album || String(album.id)) }}
                />

                {/* Decorative vinyl record */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[60%] h-[60%] rounded-full border border-white/5 opacity-20">
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[30%] h-[30%] rounded-full border border-white/10" />
                </div>

                {/* Overlay */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent" />

                {/* Content */}
                <div className="absolute inset-x-0 bottom-0 p-4">
                  <h3 className="text-sm font-bold text-white truncate mb-2 drop-shadow-lg">
                    {album.album || 'Top Songs'}
                  </h3>
                  <div className="flex items-center gap-3 text-xs text-zinc-300/70">
                    <span className="flex items-center gap-1">
                      <FileText className="w-3 h-3" />
                      {songs.length} songs
                    </span>
                  </div>
                </div>

                {/* Hover ring */}
                <div className="absolute inset-0 rounded-2xl ring-1 ring-white/10 group-hover:ring-indigo-500/40 transition-all duration-300" />
              </div>
            );
          })}

          {/* Add new album card */}
          <div
            className="aspect-square rounded-2xl border-2 border-dashed border-white/10 hover:border-pink-500/30 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 hover:bg-white/[0.02] group"
            onClick={onAddAlbum}
          >
            <div className="w-10 h-10 rounded-full bg-white/5 group-hover:bg-pink-500/10 flex items-center justify-center mb-2 transition-colors">
              <Plus className="w-5 h-5 text-zinc-500 group-hover:text-pink-400 transition-colors" />
            </div>
            <span className="text-xs text-zinc-500 group-hover:text-zinc-300 font-medium transition-colors">
              Add Album
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
