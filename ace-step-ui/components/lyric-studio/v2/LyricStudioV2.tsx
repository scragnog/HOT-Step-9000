import React, { useState, useEffect, useCallback } from 'react';
import { lireekApi, Artist, LyricsSet, Profile, Generation, AlbumPreset, SongLyric } from '../../../services/lyricStudioApi';
import { ArtistGrid } from './ArtistGrid';
import { ArtistSidebar } from './ArtistSidebar';
import { AlbumGrid } from './AlbumGrid';
import { AlbumHeader } from './AlbumHeader';
import { FetchLyricsModal } from './FetchLyricsModal';
import { PresetSettingsModal } from './PresetSettingsModal';
import { ContentTabs, TabId } from './ContentTabs';
import { SourceLyricsTab } from './SourceLyricsTab';
import { ProfilesTab } from './ProfilesTab';
import { WrittenSongsTab } from './WrittenSongsTab';
import { RecordingsTab } from './RecordingsTab';
import { useAudioGeneration } from './useAudioGeneration';
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
  const [presetModalOpen, setPresetModalOpen] = useState(false);

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
      setArtistsLoading(false);

      // Auto-fetch missing images in the background (non-blocking)
      const missing = res.artists.filter(a => !a.image_url);
      if (missing.length > 0) {
        console.log(`[LyricStudioV2] Fetching images for ${missing.length} artists...`);
        const batchSize = 3;
        for (let i = 0; i < missing.length; i += batchSize) {
          const batch = missing.slice(i, i + batchSize);
          const results = await Promise.allSettled(
            batch.map(a => lireekApi.refreshArtistImage(a.id))
          );
          setArtists(prev => prev.map(artist => {
            const idx = batch.findIndex(b => b.id === artist.id);
            if (idx === -1) return artist;
            const result = results[idx];
            if (result.status === 'fulfilled' && result.value.image_url) {
              return { ...artist, image_url: result.value.image_url };
            }
            return artist;
          }));
        }
        console.log('[LyricStudioV2] Image refresh complete');
      }
    } catch (err) {
      console.error('[LyricStudioV2] Failed to load artists:', err);
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
      setAlbumsLoading(false);

      // Auto-fetch missing album cover art in the background
      const missing = res.lyrics_sets.filter(a => !a.image_url && a.album);
      if (missing.length > 0) {
        console.log(`[LyricStudioV2] Fetching cover art for ${missing.length} albums...`);
        const batchSize = 2;
        for (let i = 0; i < missing.length; i += batchSize) {
          const batch = missing.slice(i, i + batchSize);
          const results = await Promise.allSettled(
            batch.map(a => lireekApi.refreshAlbumImage(a.id))
          );
          setAlbums(prev => prev.map(album => {
            const idx = batch.findIndex(b => b.id === album.id);
            if (idx === -1) return album;
            const result = results[idx];
            if (result.status === 'fulfilled' && result.value.image_url) {
              return { ...album, image_url: result.value.image_url };
            }
            return album;
          }));
        }
      }
    } catch (err) {
      console.error('[LyricStudioV2] Failed to load albums:', err);
      setAlbumsLoading(false);
    }
  }, []);

  // ── Load album detail data ──
  const loadAlbumData = useCallback(async (albumId: number) => {
    try {
      // Fetch the full lyrics set (with actual lyrics text for SourceLyricsTab)
      const fullAlbum = await lireekApi.getLyricsSet(albumId);
      setNav(prev => ({ ...prev, selectedAlbum: fullAlbum }));

      // Fetch profiles with full profile_data
      const profileRes = await lireekApi.listProfiles(albumId, true);
      setProfiles(profileRes.profiles);

      // Load generations with full lyrics from all profiles under this album
      const allGens: Generation[] = [];
      for (const p of profileRes.profiles) {
        try {
          const genRes = await lireekApi.listGenerations(p.id, true);
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
    // Set the album immediately (with stripped data), then loadAlbumData will replace with full data
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

  const handleSetImage = useCallback(async (artist: Artist, url: string) => {
    try {
      const res = await lireekApi.setArtistImage(artist.id, url);
      setArtists(prev => prev.map(a => a.id === artist.id ? { ...a, image_url: res.image_url } : a));
      showToast(`Image updated for ${artist.name}`);
    } catch (err: any) {
      showToast(`Failed: ${err.message}`);
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
      const updated = await lireekApi.getLyricsSet(nav.selectedAlbum.id);
      setNav(prev => ({ ...prev, selectedAlbum: updated }));
    } catch (err: any) {
      showToast(`Failed: ${err.message}`);
    }
  }, [nav.selectedAlbum, showToast]);

  const handleEditSong = useCallback(async (index: number, lyrics: string) => {
    if (!nav.selectedAlbum) return;
    try {
      const updated = await lireekApi.editSong(nav.selectedAlbum.id, index, lyrics);
      setNav(prev => ({ ...prev, selectedAlbum: updated }));
      showToast('Lyrics updated');
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

  const refreshAlbumData = useCallback(() => {
    if (nav.selectedAlbum) {
      loadAlbumData(nav.selectedAlbum.id);
    }
  }, [nav.selectedAlbum, loadAlbumData]);

  // ── Audio generation hook ──
  const { generateAudio, sendToCreate } = useAudioGeneration({
    profiles,
    showToast,
    onJobLinked: (genId, jobId) => {
      console.log(`[LyricStudioV2] Audio job ${jobId} linked to generation ${genId}`);
    },
  });

  const handleGenerateAudio = useCallback(async (gen: Generation) => {
    await generateAudio(gen);
    refreshAlbumData();
  }, [generateAudio, refreshAlbumData]);

  const handleSendToCreate = useCallback(async (gen: Generation) => {
    await sendToCreate(gen);
  }, [sendToCreate]);

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
    <div className="h-full w-full flex flex-col relative bg-zinc-950">
      {/* Toast */}
      {toast && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 px-5 py-2.5 rounded-xl bg-zinc-800/90 backdrop-blur-sm border border-white/10 text-sm text-white shadow-2xl ls2-slide-up">
          {toast}
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 overflow-hidden">
        {nav.level === 'artists' && (
          <div className="h-full overflow-y-auto ls2-fade-in">
            <ArtistGrid
              artists={artists}
              loading={artistsLoading}
              onSelectArtist={handleSelectArtist}
              onAddNew={openFetchNew}
              onDelete={handleDeleteArtist}
              onRefreshImage={handleRefreshImage}
              onSetImage={handleSetImage}
            />
          </div>
        )}

        {nav.level === 'albums' && nav.selectedArtist && (
          <div className="h-full flex ls2-fade-in">
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
          <div className="h-full flex ls2-fade-in">
            {/* Left: album header */}
            <div className="w-64 flex-shrink-0 border-r border-white/5 overflow-hidden">
              <AlbumHeader
                artist={nav.selectedArtist}
                album={nav.selectedAlbum}
                onBack={handleBackToAlbums}
                onOpenPreset={() => setPresetModalOpen(true)}
                profileCount={profiles.length}
                generationCount={generations.length}
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
                    onEditSong={handleEditSong}
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
                    onSendToCreate={handleSendToCreate}
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

      {/* Preset modal */}
      {nav.selectedAlbum && (
        <PresetSettingsModal
          isOpen={presetModalOpen}
          lyricsSetId={nav.selectedAlbum.id}
          albumName={nav.selectedAlbum.album || 'Top Songs'}
          onClose={() => setPresetModalOpen(false)}
          showToast={showToast}
        />
      )}
    </div>
  );
};
