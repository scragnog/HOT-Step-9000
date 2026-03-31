import React, { useState, useEffect, useCallback, useRef } from 'react';
import { lireekApi, Artist, LyricsSet, Profile, Generation, AlbumPreset, SongLyric } from '../../../services/lyricStudioApi';
import { ArtistGrid } from './ArtistGrid';
import { ArtistSidebar } from './ArtistSidebar';
import { ArtistPageSidebar } from './ArtistPageSidebar';
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
import { AudioJobProgress, ActiveJob } from './AudioJobProgress';
import { LiveVisualizer } from '../../LiveVisualizer';
import { LyricsBar } from '../../LyricsBar';
import { Song } from '../../../types';
import { QueuePanel } from '../QueuePanel';
import { PromptEditor } from '../PromptEditor';
import { useStreamingStore } from '../../../stores/streamingStore';
import { loadSelections } from '../ProviderSelector';

// ── URL helpers ──────────────────────────────────────────────────────────────

const LS_BASE = '/lyric-studio';

function buildUrl(artistId?: number, albumId?: number, tab?: TabId): string {
  if (artistId && albumId && tab) return `${LS_BASE}/artist/${artistId}/album/${albumId}/${tab}`;
  if (artistId && albumId) return `${LS_BASE}/artist/${artistId}/album/${albumId}`;
  if (artistId) return `${LS_BASE}/artist/${artistId}`;
  return LS_BASE;
}

function parseUrl(path: string): { artistId?: number; albumId?: number; tab?: TabId } {
  // /lyric-studio/artist/:id/album/:albumId/:tab?
  const m = path.match(/\/lyric-studio\/artist\/(\d+)(?:\/album\/(\d+)(?:\/(source-lyrics|profiles|written-songs))?)?/);
  if (!m) return {};
  return {
    artistId: Number(m[1]),
    albumId: m[2] ? Number(m[2]) : undefined,
    tab: (m[3] as TabId) || undefined,
  };
}

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

interface LyricStudioV2Props {
  onPlaySong?: (song: Song) => void;
  isPlaying?: boolean;
  currentSong?: Song | null;
  currentTime?: number;
}

export const LyricStudioV2: React.FC<LyricStudioV2Props> = ({ onPlaySong, isPlaying = false, currentSong = null, currentTime = 0 }) => {
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
  const [recordingsFilter, setRecordingsFilter] = useState<number | null>(null);
  const [songCount, setSongCount] = useState(0);
  const [recordingsRefreshKey, setRecordingsRefreshKey] = useState(0);

  // ── URL routing flag to prevent pushState during initial restore ──
  const isRestoringUrl = useRef(false);

  // ── Modals ──
  const [fetchModalOpen, setFetchModalOpen] = useState(false);
  const [fetchModalPrefill, setFetchModalPrefill] = useState<string | undefined>();
  const [presetModalOpen, setPresetModalOpen] = useState(false);
  const [queueOpen, setQueueOpen] = useState(false);
  const [promptEditorOpen, setPromptEditorOpen] = useState(false);

  // ── Bulk queue data (all artists, loaded when queue opens) ──
  const [allLyricsSets, setAllLyricsSets] = useState<LyricsSet[]>([]);
  const [allProfiles, setAllProfiles] = useState<Profile[]>([]);
  const stream = useStreamingStore();

  // ── Audio generation jobs ──
  const [activeJobs, setActiveJobs] = useState<ActiveJob[]>([]);

  // ── Toast ──
  const [toast, setToast] = useState<string | null>(null);
  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3500);
  }, []);

  // ── Load artists ──
  const loadArtists = useCallback(async (): Promise<Artist[]> => {
    setArtistsLoading(true);
    let artistsList: typeof artists = [];
    try {
      const res = await lireekApi.listArtists();
      artistsList = res.artists;
      setArtists(res.artists);
    } catch (err) {
      console.error('[LyricStudioV2] Failed to load artists:', err);
    } finally {
      setArtistsLoading(false);
    }

    // Fire-and-forget: fetch missing artist images one at a time
    const missing = artistsList.filter(a => !a.image_url);
    if (missing.length > 0) {
      console.log(`[LyricStudioV2] Background: fetching images for ${missing.length} artists...`);
      const fetchNext = (idx: number) => {
        if (idx >= missing.length) return;
        lireekApi.refreshArtistImage(missing[idx].id)
          .then(result => {
            if (result.image_url) {
              setArtists(prev => prev.map(a => a.id === missing[idx].id ? { ...a, image_url: result.image_url } : a));
            }
          })
          .catch(() => {})
          .finally(() => setTimeout(() => fetchNext(idx + 1), 500));
      };
      fetchNext(0);
    }
    return artistsList;
  }, []);

  // ── Initial load: artists + URL restore ──
  useEffect(() => {
    const init = async () => {
      const artistsList = await loadArtists();
      // Restore navigation from URL
      const parsed = parseUrl(window.location.pathname);
      if (parsed.artistId) {
        isRestoringUrl.current = true;
        try {
          // Use the already-loaded artists list (no redundant fetch)
          const artist = artistsList.find(a => a.id === parsed.artistId);
          if (!artist) return;

          // Load albums
          const albumRes = await lireekApi.listLyricsSets(artist.id);
          setAlbums(albumRes.lyrics_sets);

          if (parsed.albumId) {
            const album = albumRes.lyrics_sets.find(a => a.id === parsed.albumId);
            if (!album) {
              setNav({ level: 'albums', selectedArtist: artist, selectedAlbum: null });
              return;
            }
            setNav({ level: 'album-detail', selectedArtist: artist, selectedAlbum: album });
            if (parsed.tab) setActiveTab(parsed.tab);
            // Load album data in parallel
            await loadAlbumData(album.id);
          } else {
            setNav({ level: 'albums', selectedArtist: artist, selectedAlbum: null });
          }
        } finally {
          isRestoringUrl.current = false;
        }
      }
    };
    init();
  }, [loadArtists]);

  // ── Load albums for selected artist ──
  const loadAlbums = useCallback(async (artistId: number) => {
    setAlbumsLoading(true);
    let albumsList: typeof albums = [];
    try {
      const res = await lireekApi.listLyricsSets(artistId);
      albumsList = res.lyrics_sets;
      setAlbums(res.lyrics_sets);
    } catch (err) {
      console.error('[LyricStudioV2] Failed to load albums:', err);
    } finally {
      setAlbumsLoading(false);
    }

    // Fire-and-forget: fetch missing album cover art one at a time
    const missing = albumsList.filter(a => !a.image_url && a.album);
    if (missing.length > 0) {
      console.log(`[LyricStudioV2] Background: fetching cover art for ${missing.length} albums...`);
      const fetchNext = (idx: number) => {
        if (idx >= missing.length) return;
        lireekApi.refreshAlbumImage(missing[idx].id)
          .then(result => {
            if (result.image_url) {
              setAlbums(prev => prev.map(a => a.id === missing[idx].id ? { ...a, image_url: result.image_url } : a));
            }
          })
          .catch(() => {})
          .finally(() => setTimeout(() => fetchNext(idx + 1), 500));
      };
      fetchNext(0);
    }
  }, []);

  // ── Load album detail data ──
  const loadAlbumData = useCallback(async (albumId: number) => {
    try {
      // Fetch album details and profiles in parallel
      const [fullAlbum, profileRes] = await Promise.all([
        lireekApi.getLyricsSet(albumId),
        lireekApi.listProfiles(albumId, true),
      ]);
      setNav(prev => ({ ...prev, selectedAlbum: fullAlbum }));
      setProfiles(profileRes.profiles);

      // Load all profile generations in parallel
      const genResults = await Promise.allSettled(
        profileRes.profiles.map(p => lireekApi.listGenerations(p.id, true))
      );
      const allGens: Generation[] = [];
      for (const result of genResults) {
        if (result.status === 'fulfilled') {
          allGens.push(...result.value.generations);
        }
      }
      setGenerations(allGens);
    } catch (err) {
      console.error('[LyricStudioV2] Failed to load album data:', err);
    }
  }, []);

  // ── Navigation handlers ──
  // ── URL push helper ──
  const pushUrl = useCallback((artistId?: number, albumId?: number, tab?: TabId) => {
    if (isRestoringUrl.current) return;
    const url = buildUrl(artistId, albumId, tab);
    if (window.location.pathname !== url) {
      window.history.pushState({}, '', url);
    }
  }, []);

  const handleSelectArtist = useCallback((artist: Artist) => {
    setNav({ level: 'albums', selectedArtist: artist, selectedAlbum: null });
    loadAlbums(artist.id);
    pushUrl(artist.id);
  }, [loadAlbums, pushUrl]);

  const handleSelectAlbum = useCallback((album: LyricsSet) => {
    setNav(prev => ({ ...prev, level: 'album-detail', selectedAlbum: album }));
    setActiveTab('source-lyrics');
    loadAlbumData(album.id);
    pushUrl(nav.selectedArtist?.id, album.id, 'source-lyrics');
  }, [loadAlbumData, pushUrl, nav.selectedArtist]);

  const handleBackToArtists = useCallback(() => {
    setNav({ level: 'artists', selectedArtist: null, selectedAlbum: null });
    setAlbums([]);
    setProfiles([]);
    setGenerations([]);
    setSongCount(0);
    loadArtists();
    pushUrl();
  }, [loadArtists, pushUrl]);

  const handleBackToAlbums = useCallback(() => {
    setNav(prev => ({ ...prev, level: 'albums', selectedAlbum: null }));
    setProfiles([]);
    setGenerations([]);
    setSongCount(0);
    pushUrl(nav.selectedArtist?.id);
  }, [pushUrl, nav.selectedArtist]);

  // ── Tab change with URL update ──
  const handleTabChange = useCallback((tab: TabId) => {
    setActiveTab(tab);
    pushUrl(nav.selectedArtist?.id, nav.selectedAlbum?.id, tab);
  }, [pushUrl, nav.selectedArtist, nav.selectedAlbum]);

  // ── popstate: restore from browser back/forward ──
  useEffect(() => {
    const handlePopState = async () => {
      const parsed = parseUrl(window.location.pathname);
      if (!parsed.artistId) {
        // Back to artist grid
        setNav({ level: 'artists', selectedArtist: null, selectedAlbum: null });
        setAlbums([]);
        setProfiles([]);
        setGenerations([]);
        setSongCount(0);
        return;
      }
      // Find artist from current list
      const artist = artists.find(a => a.id === parsed.artistId);
      if (!artist) return;

      if (!parsed.albumId) {
        // Back to albums
        setNav({ level: 'albums', selectedArtist: artist, selectedAlbum: null });
        setProfiles([]);
        setGenerations([]);
        setSongCount(0);
        loadAlbums(artist.id);
        return;
      }
      // Album detail
      const album = albums.find(a => a.id === parsed.albumId);
      if (album) {
        setNav({ level: 'album-detail', selectedArtist: artist, selectedAlbum: album });
        if (parsed.tab) setActiveTab(parsed.tab);
        loadAlbumData(album.id);
      }
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [artists, albums, loadAlbums, loadAlbumData]);

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
    showToast(`Fetching lyrics for ${artist}${album ? ` — ${album}` : ''}…`);
    try {
      const res = await lireekApi.fetchLyrics({ artist, album: album || undefined, max_songs: maxSongs });
      showToast(`Fetched ${res.songs_fetched} songs`);
      await loadArtists();
      if (nav.selectedArtist && res.artist.id === nav.selectedArtist.id) {
        await loadAlbums(nav.selectedArtist.id);
      }
      if (nav.level === 'artists') {
        handleSelectArtist(res.artist);
      }
    } catch (err: any) {
      showToast(`Fetch failed: ${err.message}`);
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
      const gen = generations.find(g => g.id === genId);
      setActiveJobs(prev => [...prev, {
        jobId,
        title: gen?.title || 'Untitled',
        generationId: genId,
      }]);
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
          <div className="h-full flex ls2-fade-in">
            {/* Settings sidebar */}
            <div className="w-60 flex-shrink-0 border-r border-white/5 overflow-hidden">
              <ArtistPageSidebar
                onOpenQueue={async () => {
                  try {
                    const [lsRes, pRes] = await Promise.all([
                      lireekApi.listLyricsSets(),
                      lireekApi.listProfiles(),
                    ]);
                    setAllLyricsSets(lsRes.lyrics_sets);
                    setAllProfiles(pRes.profiles);
                  } catch (err) {
                    console.error('[LyricStudioV2] Failed to load queue data:', err);
                  }
                  setQueueOpen(true);
                }}
                onOpenPromptEditor={() => setPromptEditorOpen(true)}
              />
            </div>
            {/* Artist grid */}
            <div className="flex-1 overflow-y-auto">
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
          </div>
        )}

        {nav.level === 'albums' && nav.selectedArtist && (
          <div className="h-full flex ls2-fade-in">
            {/* Left: artist quick-switch list */}
            <div className="w-48 flex-shrink-0 border-r border-white/5 overflow-hidden">
              <ArtistSidebar
                artists={artists}
                selectedArtistId={nav.selectedArtist.id}
                onSelectArtist={handleSelectArtist}
                onBack={handleBackToArtists}
              />
            </div>
            {/* Settings sidebar */}
            <div className="w-60 flex-shrink-0 border-r border-white/5 overflow-hidden">
              <ArtistPageSidebar
                artist={nav.selectedArtist}
                albumCount={albums.length}
                onOpenQueue={async () => {
                  // Load full dataset for QueuePanel
                  try {
                    const [lsRes, pRes] = await Promise.all([
                      lireekApi.listLyricsSets(),
                      lireekApi.listProfiles(),
                    ]);
                    setAllLyricsSets(lsRes.lyrics_sets);
                    setAllProfiles(pRes.profiles);
                  } catch (err) {
                    console.error('[LyricStudioV2] Failed to load queue data:', err);
                  }
                  setQueueOpen(true);
                }}
                onOpenPromptEditor={() => setPromptEditorOpen(true)}
              />
            </div>
            {/* Album grid */}
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
          <div className="h-full flex flex-col ls2-fade-in">
            <div className="flex-1 flex min-h-0">
            {/* Left: album header — relative for visualizer bg */}
            <div className="w-64 flex-shrink-0 border-r border-white/5 overflow-hidden relative">
              {/* Visualizer background */}
              {isPlaying && (
                <div className="absolute inset-0 z-0 pointer-events-none">
                  <LiveVisualizer isPlaying={true} className="w-full h-full" dimmed={true} showControls={false} instanceId="ls-sidebar" />
                </div>
              )}
              <div className="relative z-[1] h-full">
              <AlbumHeader
                artist={nav.selectedArtist}
                album={nav.selectedAlbum}
                onBack={handleBackToAlbums}
                onOpenPreset={() => setPresetModalOpen(true)}
                profileCount={profiles.length}
                generationCount={generations.length}
                songCount={songCount}
              />
              </div>
            </div>
            {/* Middle: tabbed content — relative for cover art backdrop */}
            <div className="flex-1 overflow-hidden relative">
              {/* Cover art backdrop */}
              {isPlaying && currentSong?.coverUrl && (
                <>
                  <style>
                    {`
                      @keyframes ls-random-zoom {
                        0%, 100% { scale: 1.4; }
                        50%      { scale: 1.6; }
                      }
                      @keyframes ls-random-rotate {
                        0%, 100% { rotate: -5deg; }
                        25%      { rotate: 15deg; }
                        50%      { rotate: 2deg; }
                        75%      { rotate: -15deg; }
                      }
                      @keyframes ls-random-pan {
                        0%, 100% { translate: 0% 0%; }
                        20%      { translate: -5% 4%; }
                        40%      { translate: 6% -5%; }
                        60%      { translate: -4% -6%; }
                        80%      { translate: 5% 5%; }
                      }
                      .ls-dynamic-backdrop {
                        animation: 
                          ls-random-zoom 47s ease-in-out infinite,
                          ls-random-rotate 61s ease-in-out infinite,
                          ls-random-pan 53s ease-in-out infinite;
                      }
                    `}
                  </style>
                  <div
                    className="absolute inset-0 z-0 pointer-events-none transition-[background-image] duration-700 ls-dynamic-backdrop"
                    style={{
                      backgroundImage: `url(${currentSong.coverUrl})`,
                      backgroundSize: 'cover',
                      backgroundPosition: 'center',
                      filter: 'brightness(0.15) blur(2px) saturate(1.4)',
                    }}
                  />
                </>
              )}
              <div className="relative z-[1] h-full">
              <ContentTabs
                activeTab={activeTab}
                onTabChange={handleTabChange}
                sourceLyricsCount={sourceLyricsCount}
                profilesCount={profiles.length}
                writtenSongsCount={generations.length}
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
                    profilingModel={loadSelections().profiling}
                  />
                )}
                {activeTab === 'written-songs' && (
                  <WrittenSongsTab
                    generations={generations}
                    profiles={profiles}
                    onRefresh={refreshAlbumData}
                    onGenerateAudio={handleGenerateAudio}
                    onSendToCreate={handleSendToCreate}
                    onViewRecordings={(genId) => {
                      setRecordingsFilter(genId);
                    }}
                    showToast={showToast}
                  />
                )}
              </ContentTabs>
              </div>
            </div>
            {/* Right: Generated Songs — relative for visualizer bg */}
            <div className="w-[30%] min-w-[280px] flex-shrink-0 border-l border-white/5 overflow-hidden flex flex-col relative">
              {/* Visualizer background */}
              {isPlaying && (
                <div className="absolute inset-0 z-0 pointer-events-none">
                  <LiveVisualizer isPlaying={true} className="w-full h-full" dimmed={true} showControls={false} instanceId="ls-recordings" />
                </div>
              )}
              <div className="relative z-[1] flex-shrink-0 px-4 py-3 border-b border-white/5 bg-zinc-950/30">
                <h3 className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                  <span>🎧</span> Generated Songs
                  {songCount > 0 && (
                    <span className="min-w-[20px] h-5 px-1.5 rounded-full text-[11px] font-bold flex items-center justify-center bg-pink-500/20 text-pink-300">
                      {songCount}
                    </span>
                  )}
                </h3>
              </div>
              <div className="relative z-[1] flex-1 overflow-y-auto">
                <RecordingsTab
                  generations={generations}
                  onPlaySong={handlePlaySong}
                  showToast={showToast}
                  filterGenerationId={recordingsFilter}
                  onClearFilter={() => setRecordingsFilter(null)}
                  onSongCountChange={setSongCount}
                  refreshKey={recordingsRefreshKey}
                  artistName={nav.selectedArtist?.name}
                />
              </div>
            </div>
            </div>
            {/* Lyrics Bar — streaming lyric line above the player */}
            {isPlaying && currentSong && (
              <LyricsBar
                audioUrl={currentSong.audioUrl}
                currentTime={currentTime}
                isPlaying={isPlaying}
              />
            )}
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

      {/* Queue modal */}
      <QueuePanel
        open={queueOpen}
        onClose={() => setQueueOpen(false)}
        artists={artists}
        lyricsSets={allLyricsSets}
        profiles={allProfiles}
        profilingModel={loadSelections().profiling}
        generationModel={loadSelections().generation}
        refinementModel={loadSelections().refinement}
        showToast={showToast}
      />

      {/* Prompt Editor modal */}
      <PromptEditor
        open={promptEditorOpen}
        onClose={() => setPromptEditorOpen(false)}
      />

      {/* Audio generation progress bar */}
      <AudioJobProgress
        activeJobs={activeJobs}
        onJobComplete={(jobId) => {
          refreshAlbumData();
          setRecordingsRefreshKey(k => k + 1);
        }}
        onJobRemove={(jobId) => {
          setActiveJobs(prev => prev.filter(j => j.jobId !== jobId));
        }}
      />
    </div>
  );
};
