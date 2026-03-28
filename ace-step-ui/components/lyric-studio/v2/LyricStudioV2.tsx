import React, { useState, useEffect, useCallback } from 'react';
import { lireekApi, Artist, LyricsSet, Profile, Generation, AlbumPreset, SongLyric } from '../../../services/lyricStudioApi';
import { ArtistGrid } from './ArtistGrid';
import { ArtistSidebar } from './ArtistSidebar';
import { AlbumGrid } from './AlbumGrid';
import { AlbumHeader } from './AlbumHeader';
import { FetchLyricsModal } from './FetchLyricsModal';
import { ContentTabs, TabId } from './ContentTabs';
import { SourceLyricsTab } from './SourceLyricsTab';
import { ProfilesTab } from './ProfilesTab';
import { WrittenSongsTab } from './WrittenSongsTab';
import { RecordingsTab } from './RecordingsTab';
import { Song } from '../../../types';

function parseSongs(songs: SongLyric[] | string): SongLyric[] {
  if (typeof songs === 'string') {
    try { return JSON.parse(songs); } catch { return []; }
  }
  return songs || [];
}

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
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [generations, setGenerations] = useState<Generation[]>([]);
  const [artistsLoading, setArtistsLoading] = useState(true);
  const [albumsLoading, setAlbumsLoading] = useState(false);

  // ── Tabs ──
  const [activeTab, setActiveTab] = useState<TabId>('source-lyrics');

  // ── Modals ──
  const [fetchModalOpen, setFetchModalOpen] = useState(false);
  const [fetchModalPrefill, setFetchModalPrefill] = useState<string | undefined>();

  // ── Toast ──
  const [toast, setToast] = useState<string | null>(null);
  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3500);
  }, []);

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

  // ── Load album detail data ──
  const loadAlbumData = useCallback(async (albumId: number) => {
    try {
      const profileRes = await lireekApi.listProfiles(albumId);
      setProfiles(profileRes.profiles);

      // Load generations from all profiles under this album
      const allGens: Generation[] = [];
      for (const p of profileRes.profiles) {
        try {
          const genRes = await lireekApi.listGenerations(p.id);
          allGens.push(...genRes.generations);
        } catch { /* no generations for this profile */ }
      }
      setGenerations(allGens);
    } catch (err) {
      console.error('[LyricStudioV2] Failed to load album data:', err);
    }
  }, []);

  // ── Navigation handlers ──
  const handleSelectArtist = useCallback((artist: Artist) => {
    setNav({ level: 'albums', selectedArtist: artist, selectedAlbum: null });
    loadAlbums(artist.id);
  }, [loadAlbums]);

  const handleSelectAlbum = useCallback((album: LyricsSet) => {
    setNav(prev => ({ ...prev, level: 'album-detail', selectedAlbum: album }));
    setActiveTab('source-lyrics');
    loadAlbumData(album.id);
  }, [loadAlbumData]);

  const handleBackToArtists = useCallback(() => {
    setNav({ level: 'artists', selectedArtist: null, selectedAlbum: null });
    setAlbums([]);
    setProfiles([]);
    setGenerations([]);
    loadArtists();
  }, [loadArtists]);

  const handleBackToAlbums = useCallback(() => {
    setNav(prev => ({ ...prev, level: 'albums', selectedAlbum: null }));
    setProfiles([]);
    setGenerations([]);
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
  }, [loadArtists, showToast]);

  const handleRefreshImage = useCallback(async (artist: Artist) => {
    try {
      showToast(`Refreshing image for ${artist.name}...`);
      const res = await lireekApi.refreshArtistImage(artist.id);
      setArtists(prev => prev.map(a => a.id === artist.id ? { ...a, image_url: res.image_url } : a));
      showToast(`Updated image for ${artist.name}`);
    } catch (err: any) {
      showToast(`Couldn't find image: ${err.message}`);
    }
  }, [showToast]);

  const handleDeleteAlbum = useCallback(async (album: LyricsSet) => {
    if (!confirm(`Delete album "${album.album || 'Top Songs'}" and all associated data?`)) return;
    try {
      await lireekApi.deleteLyricsSet(album.id);
      showToast('Album deleted');
      if (nav.selectedArtist) loadAlbums(nav.selectedArtist.id);
    } catch (err: any) {
      showToast(`Failed to delete: ${err.message}`);
    }
  }, [nav.selectedArtist, loadAlbums, showToast]);

  const handleDeleteSong = useCallback(async (index: number) => {
    if (!nav.selectedAlbum) return;
    try {
      await lireekApi.removeSong(nav.selectedAlbum.id, index);
      showToast('Song removed');
      // Reload the album to get updated songs
      const updated = await lireekApi.getLyricsSet(nav.selectedAlbum.id);
      setNav(prev => ({ ...prev, selectedAlbum: updated }));
    } catch (err: any) {
      showToast(`Failed: ${err.message}`);
    }
  }, [nav.selectedAlbum, showToast]);

  const handleFetchLyrics = useCallback(async (artist: string, album: string, maxSongs: number) => {
    const res = await lireekApi.fetchLyrics({ artist, album: album || undefined, max_songs: maxSongs });
    showToast(`Fetched ${res.songs_fetched} songs`);
    await loadArtists();
    if (nav.selectedArtist && res.artist.id === nav.selectedArtist.id) {
      await loadAlbums(nav.selectedArtist.id);
    }
    if (nav.level === 'artists') {
      handleSelectArtist(res.artist);
    }
  }, [loadArtists, loadAlbums, nav.selectedArtist, nav.level, handleSelectArtist, showToast]);

  const handleGenerateAudio = useCallback((gen: Generation) => {
    showToast(`Audio generation for "${gen.title}" — use the V1 view for now`);
    // TODO: Port handleGenerateAudio from V1 in Phase 4
  }, [showToast]);

  const handlePlaySong = useCallback((song: Song) => {
    onPlaySong?.(song);
  }, [onPlaySong]);

  const openFetchForArtist = useCallback(() => {
    setFetchModalPrefill(nav.selectedArtist?.name);
    setFetchModalOpen(true);
  }, [nav.selectedArtist]);

  const openFetchNew = useCallback(() => {
    setFetchModalPrefill(undefined);
    setFetchModalOpen(true);
  }, []);

  const refreshAlbumData = useCallback(() => {
    if (nav.selectedAlbum) {
      loadAlbumData(nav.selectedAlbum.id);
    }
  }, [nav.selectedAlbum, loadAlbumData]);

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

  // ── Render helpers ──
  const sourceLyricsCount = nav.selectedAlbum ? parseSongs(nav.selectedAlbum.songs).length : 0;

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
            <div className="w-56 flex-shrink-0 border-r border-white/5 overflow-hidden">
              <ArtistSidebar
                artists={artists}
                selectedArtistId={nav.selectedArtist.id}
                onSelectArtist={handleSelectArtist}
                onBack={handleBackToArtists}
              />
            </div>
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
                onOpenPreset={() => showToast('Preset settings coming next')}
              />
            </div>
            {/* Right: tabbed content */}
            <div className="flex-1 overflow-hidden">
              <ContentTabs
                activeTab={activeTab}
                onTabChange={setActiveTab}
                sourceLyricsCount={sourceLyricsCount}
                profilesCount={profiles.length}
                writtenSongsCount={generations.length}
                recordingsCount={0}
              >
                {activeTab === 'source-lyrics' && (
                  <SourceLyricsTab
                    album={nav.selectedAlbum}
                    onDeleteSong={handleDeleteSong}
                  />
                )}
                {activeTab === 'profiles' && (
                  <ProfilesTab
                    lyricsSetId={nav.selectedAlbum.id}
                    profiles={profiles}
                    onRefresh={refreshAlbumData}
                    showToast={showToast}
                  />
                )}
                {activeTab === 'written-songs' && (
                  <WrittenSongsTab
                    generations={generations}
                    profiles={profiles}
                    onRefresh={refreshAlbumData}
                    onGenerateAudio={handleGenerateAudio}
                    showToast={showToast}
                  />
                )}
                {activeTab === 'recordings' && (
                  <RecordingsTab
                    generations={generations}
                    onPlaySong={handlePlaySong}
                    showToast={showToast}
                  />
                )}
              </ContentTabs>
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
