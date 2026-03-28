import React, { useState, useEffect, useCallback } from 'react';
import { lireekApi, Artist, LyricsSet, Profile, Generation, AlbumPreset } from '../../../services/lyricStudioApi';
import { ArtistGrid } from './ArtistGrid';
import { ArtistSidebar } from './ArtistSidebar';
import { AlbumGrid } from './AlbumGrid';
import { AlbumHeader } from './AlbumHeader';
import { FetchLyricsModal } from './FetchLyricsModal';
import { Song } from '../../../types';

// ── Navigation state ─────────────────────────────────────────────────────────

type NavLevel = 'artists' | 'albums' | 'album-detail';

interface NavState {
  level: NavLevel;
  selectedArtist: Artist | null;
  selectedAlbum: LyricsSet | null;
}

// ── Main Component ──────────────────────────────────────────────────────────

export const LyricStudioV2: React.FC<{ onPlaySong?: (song: Song) => void }> = ({ onPlaySong }) => {
  // ── Navigation ──
  const [nav, setNav] = useState<NavState>({
    level: 'artists',
    selectedArtist: null,
    selectedAlbum: null,
  });

  // ── Data ──
  const [artists, setArtists] = useState<Artist[]>([]);
  const [albums, setAlbums] = useState<LyricsSet[]>([]);
  const [artistsLoading, setArtistsLoading] = useState(true);
  const [albumsLoading, setAlbumsLoading] = useState(false);

  // ── Modals ──
  const [fetchModalOpen, setFetchModalOpen] = useState(false);
  const [fetchModalPrefill, setFetchModalPrefill] = useState<string | undefined>();

  // ── Toast ──
  const [toast, setToast] = useState<string | null>(null);
  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3500);
  };

  // ── Load artists ──
  const loadArtists = useCallback(async () => {
    setArtistsLoading(true);
    try {
      const res = await lireekApi.listArtists();
      setArtists(res.artists);
    } catch (err) {
      console.error('[LyricStudioV2] Failed to load artists:', err);
    } finally {
      setArtistsLoading(false);
    }
  }, []);

  useEffect(() => { loadArtists(); }, [loadArtists]);

  // ── Load albums for selected artist ──
  const loadAlbums = useCallback(async (artistId: number) => {
    setAlbumsLoading(true);
    try {
      const res = await lireekApi.listLyricsSets(artistId);
      setAlbums(res.lyrics_sets);
    } catch (err) {
      console.error('[LyricStudioV2] Failed to load albums:', err);
    } finally {
      setAlbumsLoading(false);
    }
  }, []);

  // ── Navigation handlers ──
  const handleSelectArtist = useCallback((artist: Artist) => {
    setNav({ level: 'albums', selectedArtist: artist, selectedAlbum: null });
    loadAlbums(artist.id);
  }, [loadAlbums]);

  const handleSelectAlbum = useCallback((album: LyricsSet) => {
    setNav(prev => ({ ...prev, level: 'album-detail', selectedAlbum: album }));
  }, []);

  const handleBackToArtists = useCallback(() => {
    setNav({ level: 'artists', selectedArtist: null, selectedAlbum: null });
    setAlbums([]);
    loadArtists(); // Refresh counts
  }, [loadArtists]);

  const handleBackToAlbums = useCallback(() => {
    setNav(prev => ({ ...prev, level: 'albums', selectedAlbum: null }));
  }, []);

  // ── Actions ──
  const handleDeleteArtist = useCallback(async (artist: Artist) => {
    if (!confirm(`Delete ${artist.name} and ALL their albums, profiles, and generations?`)) return;
    try {
      await lireekApi.deleteArtist(artist.id);
      showToast(`Deleted ${artist.name}`);
      loadArtists();
    } catch (err: any) {
      showToast(`Failed to delete: ${err.message}`);
    }
  }, [loadArtists]);

  const handleRefreshImage = useCallback(async (artist: Artist) => {
    try {
      showToast(`Refreshing image for ${artist.name}...`);
      const res = await lireekApi.refreshArtistImage(artist.id);
      setArtists(prev => prev.map(a => a.id === artist.id ? { ...a, image_url: res.image_url } : a));
      showToast(`Updated image for ${artist.name}`);
    } catch (err: any) {
      showToast(`Couldn't find image: ${err.message}`);
    }
  }, []);

  const handleDeleteAlbum = useCallback(async (album: LyricsSet) => {
    if (!confirm(`Delete album "${album.album || 'Top Songs'}" and all associated data?`)) return;
    try {
      await lireekApi.deleteLyricsSet(album.id);
      showToast('Album deleted');
      if (nav.selectedArtist) loadAlbums(nav.selectedArtist.id);
    } catch (err: any) {
      showToast(`Failed to delete: ${err.message}`);
    }
  }, [nav.selectedArtist, loadAlbums]);

  const handleFetchLyrics = useCallback(async (artist: string, album: string, maxSongs: number) => {
    const res = await lireekApi.fetchLyrics({ artist, album: album || undefined, max_songs: maxSongs });
    showToast(`Fetched ${res.songs_fetched} songs`);
    await loadArtists();
    // If we're in album view for this artist, reload albums too
    if (nav.selectedArtist && res.artist.id === nav.selectedArtist.id) {
      await loadAlbums(nav.selectedArtist.id);
    }
    // Navigate to the new artist's albums if we were on artist grid
    if (nav.level === 'artists') {
      handleSelectArtist(res.artist);
    }
  }, [loadArtists, loadAlbums, nav.selectedArtist, nav.level, handleSelectArtist]);

  const openFetchForArtist = useCallback(() => {
    setFetchModalPrefill(nav.selectedArtist?.name);
    setFetchModalOpen(true);
  }, [nav.selectedArtist]);

  const openFetchNew = useCallback(() => {
    setFetchModalPrefill(undefined);
    setFetchModalOpen(true);
  }, []);

  // ── Keyboard shortcuts ──
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (fetchModalOpen) {
          setFetchModalOpen(false);
        } else if (nav.level === 'album-detail') {
          handleBackToAlbums();
        } else if (nav.level === 'albums') {
          handleBackToArtists();
        }
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [nav.level, fetchModalOpen, handleBackToAlbums, handleBackToArtists]);

  // ── Render ──
  return (
    <div className="h-full flex flex-col relative">
      {/* Toast */}
      {toast && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 px-5 py-2.5 rounded-xl bg-zinc-800 border border-white/10 text-sm text-white shadow-2xl animate-in fade-in slide-in-from-top-2">
          {toast}
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 overflow-hidden">
        {nav.level === 'artists' && (
          <div className="h-full overflow-y-auto">
            <ArtistGrid
              artists={artists}
              loading={artistsLoading}
              onSelectArtist={handleSelectArtist}
              onAddNew={openFetchNew}
              onDelete={handleDeleteArtist}
              onRefreshImage={handleRefreshImage}
            />
          </div>
        )}

        {nav.level === 'albums' && nav.selectedArtist && (
          <div className="h-full flex">
            {/* Left: artist sidebar */}
            <div className="w-56 flex-shrink-0 border-r border-white/5 overflow-hidden">
              <ArtistSidebar
                artists={artists}
                selectedArtistId={nav.selectedArtist.id}
                onSelectArtist={handleSelectArtist}
                onBack={handleBackToArtists}
              />
            </div>
            {/* Right: album grid */}
            <div className="flex-1 overflow-y-auto">
              <AlbumGrid
                albums={albums}
                loading={albumsLoading}
                artistName={nav.selectedArtist.name}
                onSelectAlbum={handleSelectAlbum}
                onAddAlbum={openFetchForArtist}
                onDeleteAlbum={handleDeleteAlbum}
              />
            </div>
          </div>
        )}

        {nav.level === 'album-detail' && nav.selectedArtist && nav.selectedAlbum && (
          <div className="h-full flex">
            {/* Left: album header */}
            <div className="w-64 flex-shrink-0 border-r border-white/5 overflow-hidden">
              <AlbumHeader
                artist={nav.selectedArtist}
                album={nav.selectedAlbum}
                onBack={handleBackToAlbums}
                onOpenPreset={() => showToast('Preset settings coming in Phase 2')}
              />
            </div>
            {/* Right: tabbed content (placeholder for Phase 3) */}
            <div className="flex-1 overflow-y-auto">
              <div className="flex flex-col items-center justify-center h-full text-center p-8">
                <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mb-4">
                  <span className="text-2xl">🚧</span>
                </div>
                <h3 className="text-lg font-semibold text-zinc-300 mb-2">
                  Content Tabs Coming Soon
                </h3>
                <p className="text-sm text-zinc-500 max-w-sm">
                  Source Lyrics, Profiles, Written Songs, and Recordings tabs will appear here.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Fetch modal */}
      <FetchLyricsModal
        isOpen={fetchModalOpen}
        onClose={() => setFetchModalOpen(false)}
        onFetch={handleFetchLyrics}
        prefillArtist={fetchModalPrefill}
      />
    </div>
  );
};
