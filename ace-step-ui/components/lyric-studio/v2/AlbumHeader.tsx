import React from 'react';
import { ChevronLeft, Settings2, Disc3, Music, FileText, Users, Music2, Headphones } from 'lucide-react';
import { Artist, LyricsSet, SongLyric } from '../../../services/lyricStudioApi';
import { TripleProviderSelector, ModelSelections, loadSelections, saveSelections } from '../ProviderSelector';

function parseSongs(songs: SongLyric[] | string): SongLyric[] {
  if (typeof songs === 'string') {
    try { return JSON.parse(songs); } catch { return []; }
  }
  return songs || [];
}

interface AlbumHeaderProps {
  artist: Artist;
  album: LyricsSet;
  onBack: () => void;
  onOpenPreset: () => void;
  profileCount?: number;
  generationCount?: number;
  songCount?: number;
}

export const AlbumHeader: React.FC<AlbumHeaderProps> = ({
  artist, album, onBack, onOpenPreset, profileCount = 0, generationCount = 0, songCount = 0,
}) => {
  const [imageError, setImageError] = React.useState(false);
  const [modelSelections, setModelSelections] = React.useState<ModelSelections>(loadSelections);
  const songs = parseSongs(album.songs);

  const gradient = (name: string) => {
    const hash = name.split('').reduce((a, c) => ((a << 5) - a + c.charCodeAt(0)) | 0, 0);
    const h1 = Math.abs(hash) % 360;
    const h2 = (h1 + 40) % 360;
    return `linear-gradient(180deg, hsl(${h1}, 50%, 20%) 0%, hsl(${h2}, 40%, 12%) 100%)`;
  };

  return (
    <div className="flex flex-col h-full bg-zinc-950/50">
      {/* Back button */}
      <button
        onClick={onBack}
        className="flex items-center gap-2 px-4 py-3 text-sm text-zinc-400 hover:text-white hover:bg-white/5 border-b border-white/5 transition-colors"
      >
        <ChevronLeft className="w-4 h-4" />
        Albums
      </button>

      {/* Artist image / gradient */}
      <div className="relative">
        {artist.image_url && !imageError ? (
          <img
            src={artist.image_url}
            alt={artist.name}
            className="w-full aspect-video object-cover"
            onError={() => setImageError(true)}
          />
        ) : (
          <div className="w-full aspect-video" style={{ background: gradient(artist.name) }} />
        )}
        <div className="absolute inset-0 bg-gradient-to-t from-zinc-950 via-transparent to-transparent" />
      </div>

      {/* Info */}
      <div className="px-4 py-4 -mt-8 relative z-10">
        <p className="text-xs text-zinc-400 uppercase tracking-wider mb-1 font-semibold">
          {artist.name}
        </p>
        <h2 className="text-lg font-bold text-white leading-tight mb-3">
          {album.album || 'Top Songs'}
        </h2>

        <div className="space-y-2 text-sm">
          <div className="flex items-center gap-2 text-zinc-400">
            <FileText className="w-3.5 h-3.5" />
            <span>{songs.length} source lyrics</span>
          </div>
          {profileCount > 0 && (
            <div className="flex items-center gap-2 text-zinc-400">
              <Users className="w-3.5 h-3.5" />
              <span>{profileCount} profile{profileCount !== 1 ? 's' : ''}</span>
            </div>
          )}
          {generationCount > 0 && (
            <div className="flex items-center gap-2 text-zinc-400">
              <Music2 className="w-3.5 h-3.5" />
              <span>{generationCount} generated lyric{generationCount !== 1 ? 's' : ''}</span>
            </div>
          )}
          {songCount > 0 && (
            <div className="flex items-center gap-2 text-zinc-400">
              <Headphones className="w-3.5 h-3.5" />
              <span>{songCount} generated song{songCount !== 1 ? 's' : ''}</span>
            </div>
          )}
        </div>
      </div>

      {/* LLM Model Selector */}
      <div className="px-4 pb-3">
        <p className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold mb-2">LLM Models</p>
        <TripleProviderSelector
          selections={modelSelections}
          onSelectionsChange={(sel) => {
            setModelSelections(sel);
            saveSelections(sel);
          }}
        />
      </div>

      {/* Preset button */}
      <div className="px-4 pb-4">
        <button
          onClick={onOpenPreset}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-sm text-zinc-300 hover:text-white font-medium transition-all"
        >
          <Settings2 className="w-4 h-4" />
          Album Preset
        </button>
      </div>
    </div>
  );
};
