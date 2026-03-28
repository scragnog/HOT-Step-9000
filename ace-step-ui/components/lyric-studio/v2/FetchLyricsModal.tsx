import React, { useState } from 'react';
import { X, Search, Loader2, Music, Disc3 } from 'lucide-react';

interface FetchLyricsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onFetch: (artist: string, album: string, maxSongs: number) => Promise<void>;
  prefillArtist?: string;
}

export const FetchLyricsModal: React.FC<FetchLyricsModalProps> = ({
  isOpen, onClose, onFetch, prefillArtist,
}) => {
  const [artist, setArtist] = useState(prefillArtist || '');
  const [album, setAlbum] = useState('');
  const [maxSongs, setMaxSongs] = useState(15);
  const [fetching, setFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset form when modal opens with a new prefill
  React.useEffect(() => {
    if (isOpen) {
      setArtist(prefillArtist || '');
      setAlbum('');
      setError(null);
    }
  }, [isOpen, prefillArtist]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!artist.trim()) return;
    setFetching(true);
    setError(null);
    try {
      await onFetch(artist.trim(), album.trim(), maxSongs);
      onClose();
    } catch (err: any) {
      setError(err?.message || 'Failed to fetch lyrics');
    } finally {
      setFetching(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div
        className="bg-zinc-900 border border-white/10 rounded-2xl shadow-2xl w-full max-w-md overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-pink-500/10 flex items-center justify-center">
              <Search className="w-4 h-4 text-pink-400" />
            </div>
            <h2 className="text-lg font-bold text-white">Fetch Lyrics</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-white/5 text-zinc-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-5">
          <div>
            <label className="block text-sm font-medium text-zinc-300 mb-1.5">
              <span className="flex items-center gap-1.5">
                <Music className="w-3.5 h-3.5 text-zinc-400" />
                Artist Name
              </span>
            </label>
            <input
              type="text"
              value={artist}
              onChange={(e) => setArtist(e.target.value)}
              placeholder="e.g. Steel Panther"
              disabled={!!prefillArtist || fetching}
              className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white placeholder:text-zinc-500 focus:outline-none focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 transition-all disabled:opacity-50"
              autoFocus={!prefillArtist}
            />
            <p className="text-xs text-zinc-500 mt-1">
              Supports artist names or Genius URLs
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-zinc-300 mb-1.5">
              <span className="flex items-center gap-1.5">
                <Disc3 className="w-3.5 h-3.5 text-zinc-400" />
                Album Name
              </span>
            </label>
            <input
              type="text"
              value={album}
              onChange={(e) => setAlbum(e.target.value)}
              placeholder="e.g. Feel the Steel (or leave empty for top songs)"
              disabled={fetching}
              className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white placeholder:text-zinc-500 focus:outline-none focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 transition-all disabled:opacity-50"
              autoFocus={!!prefillArtist}
            />
            <p className="text-xs text-zinc-500 mt-1">
              Also supports Genius album URLs
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-zinc-300 mb-1.5">
              Max Songs
            </label>
            <input
              type="number"
              value={maxSongs}
              onChange={(e) => setMaxSongs(Math.max(1, Math.min(50, parseInt(e.target.value) || 15)))}
              min={1}
              max={50}
              disabled={fetching}
              className="w-24 px-3 py-2 rounded-xl bg-white/5 border border-white/10 text-white text-center font-mono focus:outline-none focus:border-pink-500/50 transition-all disabled:opacity-50"
            />
          </div>

          {error && (
            <div className="px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20 text-sm text-red-400">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={!artist.trim() || fetching}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-pink-600 hover:bg-pink-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white font-semibold transition-all"
          >
            {fetching ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Fetching from Genius...
              </>
            ) : (
              <>
                <Search className="w-4 h-4" />
                Fetch Lyrics
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
};
