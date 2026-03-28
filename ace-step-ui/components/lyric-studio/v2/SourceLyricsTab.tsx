import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Pencil, Trash2, Save, X, FileText } from 'lucide-react';
import { LyricsSet, SongLyric } from '../../../services/lyricStudioApi';

function parseSongs(songs: SongLyric[] | string): SongLyric[] {
  if (typeof songs === 'string') {
    try { return JSON.parse(songs); } catch { return []; }
  }
  return songs || [];
}

interface SourceLyricsTabProps {
  album: LyricsSet;
  onDeleteSong: (index: number) => void;
}

export const SourceLyricsTab: React.FC<SourceLyricsTabProps> = ({ album, onDeleteSong }) => {
  const songs = parseSongs(album.songs);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  if (songs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center px-8">
        <div className="w-14 h-14 rounded-full bg-white/5 flex items-center justify-center mb-4">
          <FileText className="w-7 h-7 text-zinc-600" />
        </div>
        <h3 className="text-base font-semibold text-zinc-400 mb-2">No source lyrics</h3>
        <p className="text-sm text-zinc-500 max-w-xs">
          This album has no fetched lyrics. Try fetching again from Genius.
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-1">
      {songs.map((song, idx) => {
        const isExpanded = expandedIdx === idx;
        return (
          <div
            key={idx}
            className="rounded-xl border border-white/5 overflow-hidden transition-colors hover:border-white/10"
          >
            {/* Song header */}
            <button
              className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-white/[0.02] transition-colors"
              onClick={() => setExpandedIdx(isExpanded ? null : idx)}
            >
              {isExpanded
                ? <ChevronDown className="w-4 h-4 text-zinc-500 flex-shrink-0" />
                : <ChevronRight className="w-4 h-4 text-zinc-500 flex-shrink-0" />
              }
              <span className="flex-1 text-sm font-medium text-zinc-200 truncate">
                {song.title}
              </span>
              <span className="text-xs text-zinc-500">
                {song.lyrics.split('\n').length} lines
              </span>
            </button>

            {/* Expanded content */}
            {isExpanded && (
              <div className="border-t border-white/5">
                <div className="px-4 py-3">
                  <pre className="text-sm text-zinc-300 whitespace-pre-wrap font-sans leading-relaxed max-h-96 overflow-y-auto">
                    {song.lyrics}
                  </pre>
                </div>
                <div className="flex items-center gap-2 px-4 py-2 border-t border-white/5 bg-white/[0.01]">
                  <button
                    onClick={() => {
                      if (confirm(`Delete "${song.title}" from this album?`)) {
                        onDeleteSong(idx);
                      }
                    }}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-red-400 hover:bg-red-500/10 transition-colors"
                  >
                    <Trash2 className="w-3 h-3" />
                    Delete
                  </button>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
