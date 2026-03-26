import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronRight, FolderOpen, Search, Music, Loader2 } from 'lucide-react';
import { usePersistedState } from '../hooks/usePersistedState';

interface Track {
    title: string;
    caption: string;
    lyrics: string;
    bpm: number;
    keyscale: string;
    duration: number;
    filename: string;
    is_refined?: boolean;
    subject?: string;
}

interface Album {
    name: string;
    tracks: Track[];
}

interface Artist {
    name: string;
    albums: Album[];
}

interface Props {
    setStyle: (v: string) => void;
    setLyrics: (v: string) => void;
    setBpm: (v: number) => void;
    setKeyScale: (v: string) => void;
    setTitle: (v: string) => void;
    setDuration: (v: number) => void;
    setCoverArtSubject?: (v: string) => void;
}

export function LyricsLibrary({ setStyle, setLyrics, setBpm, setKeyScale, setTitle, setDuration, setCoverArtSubject }: Props) {
    const [libraryPath, setLibraryPath] = usePersistedState('ace-lyrics-library-path', '');
    const [artists, setArtists] = useState<Artist[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [isOpen, setIsOpen] = usePersistedState('ace-lyrics-library-open', false);
    const [expandedArtists, setExpandedArtists] = useState<Set<string>>(new Set());
    const [expandedAlbums, setExpandedAlbums] = useState<Set<string>>(new Set());
    const [appliedTrack, setAppliedTrack] = useState('');
    const [useLibraryDuration, setUseLibraryDuration] = usePersistedState('ace-lyrics-use-duration', false);

    const handleScan = async () => {
        if (!libraryPath.trim()) return;
        setLoading(true);
        setError('');
        try {
            const res = await fetch(`/api/lyrics-library/scan?path=${encodeURIComponent(libraryPath)}`);
            const data = await res.json();
            if (data.error) {
                setError(data.error);
            }
            setArtists(data.artists || []);
            // Auto-expand all artists on scan
            if (data.artists?.length) {
                setExpandedArtists(new Set(data.artists.map((a: Artist) => a.name)));
            }
        } catch {
            setError('Failed to scan directory');
        } finally {
            setLoading(false);
        }
    };

    // Auto-scan on mount if path is set
    useEffect(() => {
        if (libraryPath.trim() && isOpen) {
            handleScan();
        }
    }, []);

    const toggleArtist = (name: string) => {
        setExpandedArtists(prev => {
            const next = new Set(prev);
            if (next.has(name)) next.delete(name);
            else next.add(name);
            return next;
        });
    };

    const toggleAlbum = (key: string) => {
        setExpandedAlbums(prev => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key);
            else next.add(key);
            return next;
        });
    };

    const applyTrack = (track: Track, artistName: string) => {
        if (track.caption) setStyle(track.caption);
        if (track.lyrics) setLyrics(track.lyrics);
        if (track.bpm > 0) setBpm(track.bpm);
        if (track.keyscale) {
            // Normalize: ACE-Step expects "C# minor" not "C# Minor"
            const parts = track.keyscale.trim().split(/\s+/);
            if (parts.length === 2) {
                setKeyScale(`${parts[0]} ${parts[1].toLowerCase()}`);
            } else {
                setKeyScale(track.keyscale);
            }
        }
        if (track.title) setTitle(artistName ? `${artistName} - ${track.title}` : track.title);
        if (track.duration > 0 && useLibraryDuration) {
            setDuration(track.duration);
        }
        if (setCoverArtSubject) setCoverArtSubject(track.subject || '');
        setAppliedTrack(track.filename);
        setTimeout(() => setAppliedTrack(''), 2000);
    };

    const totalTracks = artists.reduce((sum, a) => sum + a.albums.reduce((s, al) => s + al.tracks.length, 0), 0);

    return (
        <div className="border border-white/10 rounded-xl overflow-hidden">
            {/* Header toggle */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center gap-2 px-4 py-3 text-left hover:bg-white/5 transition-colors"
            >
                {isOpen ? <ChevronDown size={14} className="text-zinc-400" /> : <ChevronRight size={14} className="text-zinc-400" />}
                <FolderOpen size={14} className="text-purple-400" />
                <span className="text-sm font-medium text-zinc-200">Lyrics Library</span>
                {totalTracks > 0 && (
                    <span className="text-xs text-zinc-500 ml-auto">{totalTracks} tracks</span>
                )}
            </button>

            {isOpen && (
                <div className="border-t border-white/10 px-4 py-3 space-y-3">
                    {/* Path + Scan */}
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={libraryPath}
                            onChange={(e) => setLibraryPath(e.target.value)}
                            placeholder="D:\Ace-Step-Latest\Lyrics"
                            className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-zinc-200 outline-none focus:border-purple-500/50 transition-colors"
                        />
                        <button
                            onClick={handleScan}
                            disabled={loading || !libraryPath.trim()}
                            className="px-3 py-2 rounded-lg bg-purple-500/20 text-purple-300 text-sm font-medium hover:bg-purple-500/30 disabled:opacity-40 transition-colors flex items-center gap-1"
                        >
                            {loading ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />}
                            Scan
                        </button>
                    </div>

                    {error && (
                        <div className="text-xs text-red-400 bg-red-500/10 rounded-lg px-3 py-2">{error}</div>
                    )}

                    {/* Tree */}
                    {artists.length > 0 && (
                        <div className="space-y-1 max-h-80 overflow-y-auto">
                            {artists.map(artist => (
                                <div key={artist.name}>
                                    {/* Artist */}
                                    <button
                                        onClick={() => toggleArtist(artist.name)}
                                        className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-white/5 transition-colors"
                                    >
                                        {expandedArtists.has(artist.name) ? <ChevronDown size={12} className="text-zinc-500" /> : <ChevronRight size={12} className="text-zinc-500" />}
                                        <span className="text-sm font-semibold text-zinc-300">{artist.name}</span>
                                        <span className="text-xs text-zinc-600 ml-auto">
                                            {artist.albums.reduce((s, a) => s + a.tracks.length, 0)}
                                        </span>
                                    </button>

                                    {expandedArtists.has(artist.name) && artist.albums.map(album => {
                                        const albumKey = `${artist.name}/${album.name}`;
                                        return (
                                            <div key={albumKey} className="ml-4">
                                                {/* Album */}
                                                <button
                                                    onClick={() => toggleAlbum(albumKey)}
                                                    className="w-full flex items-center gap-2 px-2 py-1 rounded-lg hover:bg-white/5 transition-colors"
                                                >
                                                    {expandedAlbums.has(albumKey) ? <ChevronDown size={10} className="text-zinc-600" /> : <ChevronRight size={10} className="text-zinc-600" />}
                                                    <span className="text-xs text-zinc-400">{album.name}</span>
                                                    <span className="text-xs text-zinc-600 ml-auto">{album.tracks.length}</span>
                                                </button>

                                                {expandedAlbums.has(albumKey) && (
                                                    <div className="ml-4 space-y-0.5">
                                                        {album.tracks.map(track => (
                                                            <button
                                                                key={track.filename}
                                                                onClick={() => applyTrack(track, artist.name)}
                                                                className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-left transition-all ${appliedTrack === track.filename ? 'bg-green-500/20 text-green-300' : 'hover:bg-white/5'}`}
                                                                title={`${track.caption}\n\nBPM: ${track.bpm} | Key: ${track.keyscale}${track.duration > 0 ? ` | ~${Math.floor(track.duration / 60)}:${String(track.duration % 60).padStart(2, '0')}` : ''}`}
                                                            >
                                                                <Music size={10} className={appliedTrack === track.filename ? 'text-green-400' : 'text-zinc-600'} />
                                                                <span className={`text-xs truncate ${appliedTrack === track.filename ? 'text-green-300' : 'text-zinc-300'}`}>
                                                                    {track.title || track.filename}
                                                                </span>
                                                                {track.is_refined && (
                                                                    <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded-full bg-amber-500/20 text-amber-300 shrink-0">✨ Refined</span>
                                                                )}
                                                                {track.bpm > 0 && (
                                                                    <span className="text-[10px] text-zinc-600 ml-auto shrink-0">{track.bpm}bpm</span>
                                                                )}
                                                            </button>
                                                        ))}
                                                    </div>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
                            ))}
                        </div>
                    )}

                    {artists.length === 0 && !loading && libraryPath && (
                        <div className="text-xs text-zinc-500 text-center py-2">
                            No tracks found. Click Scan to search for lyrics files.
                        </div>
                    )}

                    {/* Use Library Duration toggle */}
                    {totalTracks > 0 && (
                        <div className="flex items-center justify-between pt-2 mt-2 border-t border-white/10">
                            <div>
                                <div className="text-xs font-medium text-zinc-300">Use Library Duration</div>
                                <div className="text-[10px] text-zinc-500">Set duration from track data so audio codes match song length</div>
                            </div>
                            <button
                                onClick={() => setUseLibraryDuration(!useLibraryDuration)}
                                className={`relative w-9 h-5 rounded-full transition-colors ${
                                    useLibraryDuration ? 'bg-purple-500' : 'bg-zinc-700'
                                }`}
                            >
                                <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
                                    useLibraryDuration ? 'translate-x-4' : 'translate-x-0'
                                }`} />
                            </button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
