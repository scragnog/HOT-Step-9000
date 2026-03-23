import React, { useState, useEffect, useRef, useCallback, Component, PropsWithChildren } from 'react';
import { Sidebar } from './components/Sidebar';
import { CreatePanel } from './components/CreatePanel';
import { SongList } from './components/SongList';
import { RightSidebar } from './components/RightSidebar';
import { Player } from './components/Player';
import { LibraryView } from './components/LibraryView';
import { CreatePlaylistModal, AddToPlaylistModal } from './components/PlaylistModals';
import { VideoGeneratorModal } from './components/VideoGeneratorModal';
import { UsernameModal } from './components/UsernameModal';
import { DownloadModal, DownloadFormat } from './components/DownloadModal';
import { UserProfile } from './components/UserProfile';
import { SettingsModal } from './components/SettingsModal';
import { SongProfile } from './components/SongProfile';
import { Song, GenerationParams, View, Playlist } from './types';
import { generateApi, songsApi, playlistsApi, getAudioUrl } from './services/api';
import { useAuth } from './context/AuthContext';
import { useResponsive } from './context/ResponsiveContext';
import { I18nProvider, useI18n } from './context/I18nContext';
import { List } from 'lucide-react';
import { PlaylistDetail } from './components/PlaylistDetail';
import { Toast, ToastType } from './components/Toast';
import { StemSplitterModal } from './components/StemSplitterModal';
import { AudioEnhancerModal, openAudioEnhancer } from './components/AudioEnhancerModal';
import { RefineModal } from './components/RefineModal';
import { MasteringConsoleModal, MasteringParams as MasteringParamsType } from './components/MasteringConsoleModal';
import { SearchPage } from './components/SearchPage';
import { ConfirmDialog } from './components/ConfirmDialog';
import DebugPanel from './components/DebugPanel';
import { usePersistedState } from './hooks/usePersistedState';
import { AudioAnalysisProvider, useAudioAnalysis } from './context/AudioAnalysisContext';
import { FullscreenVisualizer } from './components/FullscreenVisualizer';
import { ABCompareModal } from './components/ABCompareModal';


type AppErrorBoundaryProps = PropsWithChildren<{}>;
type AppErrorBoundaryState = { error: Error | null };

class AppErrorBoundary extends Component<AppErrorBoundaryProps, AppErrorBoundaryState> {
  state: AppErrorBoundaryState = { error: null };
  declare props: Readonly<AppErrorBoundaryProps>;
  declare setState: (
    state:
      | AppErrorBoundaryState
      | null
      | ((
        prevState: Readonly<AppErrorBoundaryState>,
        props: Readonly<AppErrorBoundaryProps>
      ) => AppErrorBoundaryState | null)
  ) => void;

  static getDerivedStateFromError(error: Error): AppErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error) {
    console.error('App crashed:', error);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="w-full h-full min-h-screen flex items-center justify-center bg-black text-zinc-200 p-6">
          <div className="max-w-2xl w-full bg-zinc-900/60 border border-white/10 rounded-2xl p-6 space-y-4">
            <div className="text-lg font-bold">UI crashed</div>
            <div className="text-sm text-zinc-300 break-words whitespace-pre-wrap">{this.state.error.message}</div>
            {this.state.error.stack ? (
              <pre className="text-xs text-zinc-400 break-words whitespace-pre-wrap bg-black/40 border border-white/10 rounded-lg p-3 overflow-auto max-h-72">
                {this.state.error.stack}
              </pre>
            ) : null}
            <div className="flex items-center gap-3">
              <button
                onClick={() => window.location.reload()}
                className="px-4 py-2 rounded-lg bg-white text-black font-semibold"
              >
                Reload
              </button>
              <button
                onClick={() => this.setState({ error: null })}
                className="px-4 py-2 rounded-lg bg-zinc-800 text-zinc-200 font-semibold"
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

function AppContent() {
  // i18n
  const { t } = useI18n();

  // Responsive
  const { isMobile, isDesktop } = useResponsive();

  // Debug panel state (App.tsx owns it, passes to DebugPanel as props)
  const [debugPanelOpen, setDebugPanelOpen] = usePersistedState('ace-debugPanelOpen', false);

  // Create panel resizable width (persisted)
  const [createPanelWidth, setCreatePanelWidth] = usePersistedState('ace-createPanelWidth', 490);

  // Right sidebar resizable width (persisted)
  const [rightSidebarWidth, setRightSidebarWidth] = usePersistedState('ace-rightSidebarWidth', 360);

  // Auth
  const { user, token, isAuthenticated, isLoading: authLoading, setupUser, logout } = useAuth();
  const [showUsernameModal, setShowUsernameModal] = useState(false);
  // Track multiple concurrent generation jobs
  const activeJobsRef = useRef<Map<string, { tempId: string; pollInterval: ReturnType<typeof setInterval> }>>(new Map());
  const [activeJobCount, setActiveJobCount] = useState(0);

  // Theme State
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const stored = localStorage.getItem('theme');
    if (stored === 'dark' || stored === 'light') return stored;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  // Navigation State - default to create view
  const [currentView, setCurrentView] = useState<View>('create');

  // Content State
  const [songs, setSongs] = useState<Song[]>([]);
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [likedSongIds, setLikedSongIds] = useState<Set<string>>(new Set());
  const [referenceTracks, setReferenceTracks] = useState<ReferenceTrack[]>([]);
  const [playQueue, setPlayQueue] = useState<Song[]>([]);
  const [queueIndex, setQueueIndex] = useState(-1);

  // Selection State
  const [currentSong, setCurrentSong] = useState<Song | null>(null);
  const [selectedSong, setSelectedSong] = useState<Song | null>(null);
  const [selectedPlaylist, setSelectedPlaylist] = useState<Playlist | null>(null);

  // Player State
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(() => {
    const stored = localStorage.getItem('volume');
    return stored ? parseFloat(stored) : 0.8;
  });
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [isShuffle, setIsShuffle] = useState(false);
  const [repeatMode, setRepeatMode] = useState<'none' | 'all' | 'one'>('none');

  // UI State
  const [isGenerating, setIsGenerating] = useState(false);
  const [showRightSidebar, setShowRightSidebar] = useState(true);
  const [showLeftSidebar, setShowLeftSidebar] = useState(false);
  const [pendingAudioSelection, setPendingAudioSelection] = useState<{ target: 'reference' | 'source'; url: string; title?: string } | null>(null);

  // Mobile UI Toggle
  const [mobileShowList, setMobileShowList] = useState(false);

  // Modals
  const [isCreatePlaylistModalOpen, setIsCreatePlaylistModalOpen] = useState(false);
  const [isAddToPlaylistModalOpen, setIsAddToPlaylistModalOpen] = useState(false);
  const [songToAddToPlaylist, setSongToAddToPlaylist] = useState<Song | null>(null);

  // Video Modal
  const [isVideoModalOpen, setIsVideoModalOpen] = useState(false);
  const [songForVideo, setSongForVideo] = useState<Song | null>(null);

  // Download Modal
  const [isDownloadModalOpen, setIsDownloadModalOpen] = useState(false);
  const [songToDownload, setSongToDownload] = useState<Song | null>(null);
  const [bulkDownloadSongs, setBulkDownloadSongs] = useState<Song[]>([]);

  // Re-master Modal
  const [showRemasterConsole, setShowRemasterConsole] = useState(false);
  const [remasterSong, setRemasterSong] = useState<Song | null>(null);

  // M/O (Mastered/Original) toggle — shared between Player and SongList
  const [playingOriginal, setPlayingOriginal] = useState(false);

  // Settings Modal
  const [showSettingsModal, setShowSettingsModal] = useState(false);

  // Profile View
  const [viewingUsername, setViewingUsername] = useState<string | null>(null);

  // Song View
  const [viewingSongId, setViewingSongId] = useState<string | null>(null);

  // Playlist View
  const [viewingPlaylistId, setViewingPlaylistId] = useState<string | null>(null);

  // Reuse State
  const [reuseData, setReuseData] = useState<{ song: Song, timestamp: number } | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const altAudioRef = useRef<HTMLAudioElement | null>(null);
  const currentSongIdRef = useRef<string | null>(null);
  const pendingSeekRef = useRef<number | null>(null);
  const playNextRef = useRef<() => void>(() => { });

  // Mobile Details Modal State
  const [showMobileDetails, setShowMobileDetails] = useState(false);

  // Fullscreen Visualizer State
  const [showFullscreenVisualizer, setShowFullscreenVisualizer] = useState(false);

  // A/B Comparison State
  const [abTrackA, setAbTrackA] = useState<Song | null>(null);
  const [abTrackB, setAbTrackB] = useState<Song | null>(null);
  const [abActive, setAbActive] = useState<'A' | 'B' | null>(null);
  const [showABCompare, setShowABCompare] = useState(false);
  const abAudioRef = useRef<HTMLAudioElement | null>(null);

  // Ablation diff pin state — shared between CreatePanel (labels in ablation panel)
  // and SongList (highlights [A]/[B] on cards). Pinning A twice = toggle off.
  const [diffPinnedA, setDiffPinnedA] = useState<Song | null>(null);
  const [diffPinnedB, setDiffPinnedB] = useState<Song | null>(null);

  // Dev mode (synced from localStorage so SongCard buttons light up correctly)
  const [devMode, setDevMode] = useState(() => {
    try { return localStorage.getItem('ace_dev_mode') === 'true'; } catch { return false; }
  });
  useEffect(() => {
    const check = () => setDevMode(localStorage.getItem('ace_dev_mode') === 'true');
    const id = setInterval(check, 500);
    return () => clearInterval(id);
  }, []);

  // A/B Comparison Handlers
  const handleABPlay = useCallback(() => {
    if (!abTrackA || !abTrackB) return;
    const mainAudio = audioRef.current;
    if (!mainAudio) return;

    // Create secondary audio element for track B
    if (abAudioRef.current) {
      abAudioRef.current.pause();
      abAudioRef.current = null;
    }
    const secondAudio = new Audio();
    secondAudio.crossOrigin = 'anonymous';
    secondAudio.volume = volume;
    secondAudio.playbackRate = playbackRate;
    abAudioRef.current = secondAudio;

    // Load track A into main audio, track B into secondary
    currentSongIdRef.current = abTrackA.id;
    mainAudio.src = abTrackA.audioUrl;
    mainAudio.muted = false;
    mainAudio.load();

    secondAudio.src = abTrackB.audioUrl;
    secondAudio.muted = true;
    secondAudio.load();

    // Play both simultaneously
    const playBoth = () => {
      mainAudio.play().catch(() => { });
      secondAudio.play().catch(() => { });
    };
    mainAudio.addEventListener('canplay', playBoth, { once: true });

    // Sync ended event
    const onEnded = () => {
      secondAudio.pause();
      secondAudio.currentTime = 0;
      playNextRef.current();
    };
    mainAudio.addEventListener('ended', onEnded, { once: true });

    setCurrentSong(abTrackA);
    setAbActive('A');
    setIsPlaying(true);
  }, [abTrackA, abTrackB, volume, playbackRate]);

  const handleABToggle = useCallback(() => {
    if (!abTrackA || !abTrackB || !abActive) return;
    const mainAudio = audioRef.current;
    const secondAudio = abAudioRef.current;
    if (!mainAudio || !secondAudio) return;

    const next = abActive === 'A' ? 'B' : 'A';

    // Check if the active audio is actually paused — respect pause state
    const activeAudio = abActive === 'A' ? mainAudio : secondAudio;
    const wasPaused = activeAudio.paused;
    if (wasPaused) setIsPlaying(false);

    // Sync position: bring inactive to active's currentTime
    if (next === 'B') {
      secondAudio.currentTime = mainAudio.currentTime;
      mainAudio.muted = true;
      secondAudio.muted = false;
      setCurrentSong(abTrackB);
      currentSongIdRef.current = abTrackB.id;
    } else {
      mainAudio.currentTime = secondAudio.currentTime;
      secondAudio.muted = true;
      mainAudio.muted = false;
      setCurrentSong(abTrackA);
      currentSongIdRef.current = abTrackA.id;
    }

    setAbActive(next);
  }, [abTrackA, abTrackB, abActive]);

  const handleABClear = useCallback(() => {
    if (abAudioRef.current) {
      abAudioRef.current.pause();
      abAudioRef.current = null;
    }
    // Unmute main audio if it was muted
    if (audioRef.current) audioRef.current.muted = false;
    setAbTrackA(null);
    setAbTrackB(null);
    setAbActive(null);
  }, []);

  // Keep secondary audio volume/rate/playstate in sync
  useEffect(() => {
    if (abAudioRef.current) {
      abAudioRef.current.volume = volume;
      abAudioRef.current.playbackRate = playbackRate;
      // Sync play/pause state with main player
      if (abActive) {
        if (isPlaying) {
          abAudioRef.current.play().catch(() => { });
        } else {
          abAudioRef.current.pause();
        }
      }
    }
  }, [volume, playbackRate, isPlaying, abActive]);

  // Visualizer songlist background setting
  const [showVisualizerBg, setShowVisualizerBg] = useState(() => localStorage.getItem('visualizer_songlist_bg') === 'true');

  // Listen for storage changes (when setting is toggled in SettingsModal)
  useEffect(() => {
    const handleStorage = () => {
      setShowVisualizerBg(localStorage.getItem('visualizer_songlist_bg') === 'true');
    };
    window.addEventListener('storage', handleStorage);
    // Also poll on interval to catch same-tab changes
    const interval = setInterval(handleStorage, 500);
    return () => {
      window.removeEventListener('storage', handleStorage);
      clearInterval(interval);
    };
  }, []);

  // Audio Analysis
  const { connect: connectAudioAnalysis } = useAudioAnalysis();

  // Toast State
  const [toast, setToast] = useState<{ message: string; type: ToastType; isVisible: boolean }>({
    message: '',
    type: 'success',
    isVisible: false,
  });

  // Confirm Dialog State
  const [confirmDialog, setConfirmDialog] = useState<{
    title: string;
    message: string;
    confirmLabel?: string;
    danger?: boolean;
    onConfirm: () => void;
  } | null>(null);
  const [isShutdown, setIsShutdown] = useState(false);

  interface ReferenceTrack {
    id: string;
    filename: string;
    storage_key: string;
    duration: number | null;
    file_size_bytes: number | null;
    tags: string[] | null;
    created_at: string;
    audio_url: string;
  }

  const showToast = (message: string, type: ToastType = 'success') => {
    setToast({ message, type, isVisible: true });
  };

  const closeToast = () => {
    setToast(prev => ({ ...prev, isVisible: false }));
  };

  // Show username modal if not authenticated and not loading
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      setShowUsernameModal(true);
    }
  }, [authLoading, isAuthenticated]);

  // Load Playlists
  useEffect(() => {
    if (token) {
      playlistsApi.getMyPlaylists(token)
        .then(res => setPlaylists(res.playlists))
        .catch(err => console.error('Failed to load playlists', err));
    } else {
      setPlaylists([]);
    }
  }, [token]);

  // Cleanup active jobs on unmount
  useEffect(() => {
    return () => {
      // Clear all polling intervals when component unmounts
      activeJobsRef.current.forEach(({ pollInterval }) => {
        clearInterval(pollInterval);
      });
      activeJobsRef.current.clear();
    };
  }, []);

  const handleShowDetails = (song: Song) => {
    setSelectedSong(song);
    setShowMobileDetails(true);
  };

  // Reuse Handler
  const handleReuse = (song: Song) => {
    setReuseData({ song, timestamp: Date.now() });
    setCurrentView('create');
    setMobileShowList(false);
  };

  // Upscale to HQ Handler — regenerate with same params, higher steps, no re-thinking
  const handleUpscaleToHQ = (song: Song) => {
    const gp = song.generationParams;
    if (!gp) {
      showToast('No generation parameters saved for this song — cannot upscale.', 'error');
      return;
    }

    const hqSteps = (() => {
      const saved = localStorage.getItem('hq_upscale_steps');
      return saved ? parseInt(saved, 10) : 160;
    })();

    const upscaleParams: GenerationParams = {
      // Start with ALL original params — ensures nothing is missed
      ...gp,

      // Upscale-specific overrides
      title: `[HQ] ${song.title || 'Untitled'}`,
      inferenceSteps: hqSteps,
      thinking: false,       // Don't re-run LM — reuse stored audio codes
      randomSeed: false,     // Use the exact same seed
      batchSize: 1,          // Only one HQ version
      getScores: false,      // Skip scoring for upscale
      audioFormat: gp.audioFormat ?? 'flac', // Default to FLAC for HQ
    };

    handleGenerate(upscaleParams);
    showToast(`Upscaling "${song.title}" to HQ (${hqSteps} steps)...`, 'success');
  };

  // Song Update Handler
  const handleSongUpdate = (updatedSong: Song) => {
    setSongs(prev => prev.map(s => s.id === updatedSong.id ? updatedSong : s));
    if (currentSong?.id === updatedSong.id) {
      setCurrentSong(updatedSong);
    }
    if (selectedSong?.id === updatedSong.id) {
      setSelectedSong(updatedSong);
    }
  };

  // Navigate to Profile Handler
  const handleNavigateToProfile = (username: string) => {
    setViewingUsername(username);
    setCurrentView('profile');
    window.history.pushState({}, '', `/@${username}`);
  };

  // Back from Profile Handler
  const handleBackFromProfile = () => {
    setViewingUsername(null);
    setCurrentView('create');
    window.history.pushState({}, '', '/');
  };

  // Navigate to Song Handler
  const handleNavigateToSong = (songId: string) => {
    setViewingSongId(songId);
    setCurrentView('song');
    window.history.pushState({}, '', `/song/${songId}`);
  };

  // Back from Song Handler
  const handleBackFromSong = () => {
    setViewingSongId(null);
    setCurrentView('create');
    window.history.pushState({}, '', '/');
  };

  // Theme Effect
  useEffect(() => {
    localStorage.setItem('theme', theme);
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  // URL Routing Effect
  useEffect(() => {
    const handleUrlChange = () => {
      const path = window.location.pathname;
      const params = new URLSearchParams(window.location.search);

      // Handle ?song= query parameter
      const songParam = params.get('song');
      if (songParam) {
        setViewingSongId(songParam);
        setCurrentView('song');
        window.history.replaceState({}, '', `/song/${songParam}`);
        return;
      }

      if (path === '/create' || path === '/') {
        setCurrentView('create');
        setMobileShowList(false);
      } else if (path === '/library') {
        setCurrentView('library');
      } else if (path.startsWith('/@')) {
        const username = path.substring(2);
        if (username) {
          setViewingUsername(username);
          setCurrentView('profile');
        }
      } else if (path.startsWith('/song/')) {
        const songId = path.substring(6);
        if (songId) {
          setViewingSongId(songId);
          setCurrentView('song');
        }
      } else if (path.startsWith('/playlist/')) {
        const playlistId = path.substring(10);
        if (playlistId) {
          setViewingPlaylistId(playlistId);
          setCurrentView('playlist');
        }
      } else if (path === '/search') {
        setCurrentView('search');
      }
    };

    handleUrlChange();

    window.addEventListener('popstate', handleUrlChange);
    return () => window.removeEventListener('popstate', handleUrlChange);
  }, []);

  const normalizeGenerationParams = useCallback((s: any) => {
    try {
      const normalizeObject = (gp: any) => {
        if (!gp || typeof gp !== 'object') return undefined;

        const bpm = gp.bpm ?? s?.bpm ?? s?.metas?.bpm;
        const duration = gp.duration ?? s?.duration ?? s?.metas?.duration;
        const keyScale = gp.keyScale ?? gp.key_scale ?? gp.keyscale ?? s?.key_scale ?? s?.metas?.keyscale;
        const normalizeTimeSignature = (v: unknown) => {
          if (v == null) return undefined;
          if (typeof v === 'string') {
            const str = v.trim();
            if (!str) return undefined;
            if (str.includes('/')) return str;
            const n = Number(str);
            return Number.isFinite(n) ? `${n}/4` : str;
          }
          if (typeof v === 'number' && Number.isFinite(v)) return `${v}/4`;
          const str = String(v);
          return str.includes('/') ? str : str;
        };
        const timeSignature = normalizeTimeSignature(
          gp.timeSignature ?? gp.time_signature ?? gp.timesignature ?? s?.time_signature ?? s?.metas?.timesignature
        );
        const ditModel = gp.ditModel ?? gp.dit_model ?? s?.dit_model ?? s?.ditModel;
        const lmModel = gp.lmModel ?? gp.lm_model ?? s?.lm_model;
        const genres = gp.genres ?? s?.metas?.genres;

        const generationInfo = gp.generationInfo ?? gp.generation_info ?? s?.generation_info;
        const inferenceSteps = (() => {
          if (gp.inferenceSteps != null) return gp.inferenceSteps;
          if (gp.inference_steps != null) return gp.inference_steps;
          const gi = generationInfo;
          if (typeof gi !== 'string') return undefined;
          const m = gi.match(/Steps:\s*(\d+)/i);
          if (!m?.[1]) return undefined;
          const n = Number(m[1]);
          return Number.isFinite(n) ? n : undefined;
        })();

        const seedText = gp.seedText ?? gp.seed_text ?? (s?.seed_value != null ? String(s.seed_value) : undefined);
        const seed = (() => {
          if (gp.seed != null) return gp.seed;
          if (!seedText) return undefined;
          const first = String(seedText).split(',')[0]?.trim();
          if (!first) return undefined;
          const n = Number(first);
          return Number.isFinite(n) ? n : undefined;
        })();

        const merged = { ...gp };
        if (bpm != null) merged.bpm = bpm;
        if (duration != null) merged.duration = duration;
        if (keyScale != null) merged.keyScale = keyScale;
        if (timeSignature != null) merged.timeSignature = timeSignature;
        if (ditModel != null) merged.ditModel = ditModel;
        if (lmModel != null) merged.lmModel = lmModel;
        if (genres != null) merged.genres = genres;
        if (seed != null) merged.seed = seed;
        if (seedText != null) merged.seedText = seedText;
        if (inferenceSteps != null) merged.inferenceSteps = inferenceSteps;
        if (generationInfo != null) merged.generationInfo = generationInfo;

        return merged;
      };

      if (s?.generation_params) {
        const parsed = typeof s.generation_params === 'string' ? JSON.parse(s.generation_params) : s.generation_params;
        const normalized = normalizeObject(parsed);
        if (normalized) return normalized;
      }

      if (s?.metas) {
        return normalizeObject({});
      }

      if (s?.bpm != null || s?.key_scale != null || s?.time_signature != null) {
        return normalizeObject({});
      }

      return undefined;
    } catch {
      return undefined;
    }
  }, []);

  // Load Songs Effect
  useEffect(() => {
    if (!isAuthenticated || !token) return;

    const loadSongs = async () => {
      try {
        const [mySongsRes, likedSongsRes] = await Promise.all([
          songsApi.getMySongs(token),
          songsApi.getLikedSongs(token)
        ]);

        const mapSong = (s: any): Song => ({
          id: s.id,
          title: s.title,
          lyrics: s.lyrics,
          style: s.style,
          coverUrl: `https://picsum.photos/seed/${s.id}/400/400`,
          duration: s.duration && s.duration > 0 ? `${Math.floor(s.duration / 60)}:${String(Math.floor(s.duration % 60)).padStart(2, '0')}` : '0:00',
          createdAt: new Date(s.created_at || s.createdAt),
          tags: s.tags || [],
          audioUrl: getAudioUrl(s.audio_url, s.id),
          isPublic: s.is_public,
          likeCount: s.like_count || 0,
          viewCount: s.view_count || 0,
          userId: s.user_id,
          creator: s.creator,
          ditModel: s.ditModel,
          generationParams: normalizeGenerationParams(s),
        });

        const mySongs = mySongsRes.songs.map(mapSong);
        const likedSongs = likedSongsRes.songs.map(mapSong);

        const songsMap = new Map<string, Song>();
        [...mySongs, ...likedSongs].forEach(s => songsMap.set(s.id, s));

        // Merge: keep generating songs + loaded songs (SongList sorts by date)
        setSongs(prev => {
          const generatingSongs = prev.filter(s => s.isGenerating);
          const loadedSongs = Array.from(songsMap.values());
          const loadedById = new Map(loadedSongs.map(s => [s.id, s]));
          generatingSongs.forEach(g => loadedById.delete(g.id));
          return [...generatingSongs, ...loadedById.values()];
        });

        const likedIds = new Set(likedSongs.map(s => s.id));
        setLikedSongIds(likedIds);

      } catch (error) {
        console.error('Failed to load songs:', error);
      }
    };

    loadSongs();
  }, [isAuthenticated, token]);

  const loadReferenceTracks = useCallback(async () => {
    if (!isAuthenticated || !token) return;
    try {
      const response = await fetch('/api/reference-tracks', {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (!response.ok) return;
      const data = await response.json();
      setReferenceTracks(data.tracks || []);
    } catch (error) {
      console.error('Failed to load reference tracks:', error);
    }
  }, [isAuthenticated, token]);

  // Load reference tracks for Library
  useEffect(() => {
    loadReferenceTracks();
  }, [loadReferenceTracks]);

  useEffect(() => {
    if (currentView === 'library') {
      loadReferenceTracks();
    }
  }, [currentView, loadReferenceTracks]);

  // Player Logic
  const getActiveQueue = (song?: Song) => {
    if (playQueue.length > 0) return playQueue;
    if (song && songs.some(s => s.id === song.id)) return songs;
    return songs;
  };

  const playNext = useCallback(() => {
    if (!currentSong) return;
    const queue = getActiveQueue(currentSong);
    if (queue.length === 0) return;

    const currentIndex = queueIndex >= 0 && queue[queueIndex]?.id === currentSong.id
      ? queueIndex
      : queue.findIndex(s => s.id === currentSong.id);
    if (currentIndex === -1) return;

    if (repeatMode === 'one') {
      if (audioRef.current) {
        audioRef.current.currentTime = 0;
        audioRef.current.play();
      }
      return;
    }

    let nextIndex;
    if (isShuffle) {
      do {
        nextIndex = Math.floor(Math.random() * queue.length);
      } while (queue.length > 1 && nextIndex === currentIndex);
    } else {
      nextIndex = (currentIndex + 1) % queue.length;
    }

    const nextSong = queue[nextIndex];
    setQueueIndex(nextIndex);
    setCurrentSong(nextSong);
    setIsPlaying(true);
  }, [currentSong, queueIndex, isShuffle, repeatMode, playQueue, songs]);

  const playPrevious = useCallback(() => {
    if (!currentSong) return;
    const queue = getActiveQueue(currentSong);
    if (queue.length === 0) return;

    const currentIndex = queueIndex >= 0 && queue[queueIndex]?.id === currentSong.id
      ? queueIndex
      : queue.findIndex(s => s.id === currentSong.id);
    if (currentIndex === -1) return;

    if (currentTime > 3) {
      if (audioRef.current) audioRef.current.currentTime = 0;
      return;
    }

    let prevIndex = (currentIndex - 1 + queue.length) % queue.length;
    if (isShuffle) {
      prevIndex = Math.floor(Math.random() * queue.length);
    }

    const prevSong = queue[prevIndex];
    setQueueIndex(prevIndex);
    setCurrentSong(prevSong);
    setIsPlaying(true);
  }, [currentSong, queueIndex, currentTime, isShuffle, playQueue, songs]);

  useEffect(() => {
    playNextRef.current = playNext;
  }, [playNext]);

  // Audio Setup — dual elements for seamless M/O toggle
  useEffect(() => {
    // Primary audio (mastered)
    audioRef.current = new Audio();
    audioRef.current.crossOrigin = "anonymous";
    const audio = audioRef.current;
    audio.volume = volume;

    // Alternate audio (original) — plays simultaneously, muted until toggled
    altAudioRef.current = new Audio();
    altAudioRef.current.crossOrigin = "anonymous";
    const altAudio = altAudioRef.current;
    altAudio.volume = 0;

    const onTimeUpdate = () => setCurrentTime(audio.currentTime);
    const applyPendingSeek = () => {
      if (pendingSeekRef.current === null) return;
      if (audio.seekable.length === 0) return;
      const target = pendingSeekRef.current;
      const safeTarget = Number.isFinite(audio.duration)
        ? Math.min(Math.max(target, 0), audio.duration)
        : Math.max(target, 0);
      audio.currentTime = safeTarget;
      if (altAudio.src && altAudio.readyState >= 1) {
        altAudio.currentTime = safeTarget;
      }
      setCurrentTime(safeTarget);
      pendingSeekRef.current = null;
    };

    const onLoadedMetadata = () => {
      setDuration(audio.duration);
      applyPendingSeek();
    };

    const onCanPlay = () => {
      applyPendingSeek();
    };

    const onProgress = () => {
      applyPendingSeek();
    };

    const onEnded = () => {
      playNextRef.current();
    };

    const onError = (e: Event) => {
      if (audio.error && audio.error.code !== 1) {
        console.error("Audio playback error:", audio.error);
        if (audio.error.code === 4) {
          showToast(t('songNotAvailable'), 'error');
        } else {
          showToast(t('unableToPlay'), 'error');
        }
      }
      setIsPlaying(false);
    };

    // Alt audio time sync: keep alt in sync with primary via periodic correction
    const altTimeUpdate = () => {
      if (audio.readyState >= 1 && altAudio.readyState >= 1) {
        const drift = Math.abs(audio.currentTime - altAudio.currentTime);
        if (drift > 0.15) {
          altAudio.currentTime = audio.currentTime;
        }
      }
    };

    audio.addEventListener('timeupdate', onTimeUpdate);
    audio.addEventListener('loadedmetadata', onLoadedMetadata);
    audio.addEventListener('canplay', onCanPlay);
    audio.addEventListener('progress', onProgress);
    audio.addEventListener('ended', onEnded);
    audio.addEventListener('error', onError);
    altAudio.addEventListener('timeupdate', altTimeUpdate);
    // Suppress alt audio ended (primary controls advancement)
    altAudio.addEventListener('ended', () => { altAudio.pause(); });

    return () => {
      audio.pause();
      altAudio.pause();
      audio.removeEventListener('timeupdate', onTimeUpdate);
      audio.removeEventListener('loadedmetadata', onLoadedMetadata);
      audio.removeEventListener('canplay', onCanPlay);
      audio.removeEventListener('progress', onProgress);
      audio.removeEventListener('ended', onEnded);
      audio.removeEventListener('error', onError);
      altAudio.removeEventListener('timeupdate', altTimeUpdate);
    };
  }, []);

  // Handle Playback State — dual-audio with volume-based M/O switching
  useEffect(() => {
    const audio = audioRef.current;
    const altAudio = altAudioRef.current;
    if (!audio || !altAudio || !currentSong?.audioUrl) return;

    const playBoth = async () => {
      try {
        await audio.play();
      } catch (err) {
        if (err instanceof Error && err.name !== 'AbortError') {
          console.error("Playback failed:", err);
          if (err.name === 'NotSupportedError') {
            showToast(t('songNotAvailable'), 'error');
          }
          setIsPlaying(false);
        }
      }
      // Play alt audio silently alongside primary (if loaded)
      if (altAudio.src && altAudio.readyState >= 1) {
        try { await altAudio.play(); } catch { /* alt is optional */ }
      }
    };

    const pauseBoth = () => {
      audio.pause();
      if (altAudio.src) altAudio.pause();
    };

    const masteredUrl = currentSong.audioUrl;
    const originalUrl = currentSong.generationParams?.originalAudioUrl || null;
    const isSongChange = currentSongIdRef.current !== currentSong.id;

    if (isSongChange) {
      // --- NEW SONG: load both sources, start from 0 ---
      currentSongIdRef.current = currentSong.id;

      audio.src = masteredUrl;
      connectAudioAnalysis(audio);
      audio.load();

      // Pre-load original into alt element (if available)
      if (originalUrl) {
        altAudio.src = originalUrl;
        altAudio.load();
      } else {
        altAudio.removeAttribute('src');
      }

      // Set volumes: mastered audible, original silent (playingOriginal is reset on new song)
      audio.volume = volume;
      altAudio.volume = 0;

      const onCanPlay = () => {
        audio.currentTime = 0;
        if (isPlaying) playBoth();
        audio.removeEventListener('canplay', onCanPlay);
      };
      audio.addEventListener('canplay', onCanPlay);

      // Start alt audio when it's ready too
      if (originalUrl) {
        const onAltCanPlay = () => {
          // Sync alt to primary's position
          if (audio.readyState >= 1) altAudio.currentTime = audio.currentTime;
          if (isPlaying && !audio.paused) {
            try { altAudio.play(); } catch { /* optional */ }
          }
          altAudio.removeEventListener('canplay', onAltCanPlay);
        };
        altAudio.addEventListener('canplay', onAltCanPlay);
      }
    } else {
      // --- SAME SONG: M/O toggle or play/pause ---
      // Swap volumes for instant M/O switching (no src change!)
      if (playingOriginal && originalUrl) {
        audio.volume = 0;
        altAudio.volume = volume;
      } else {
        audio.volume = volume;
        altAudio.volume = 0;
      }

      if (isPlaying) playBoth();
      else pauseBoth();
    }
  }, [currentSong, isPlaying, playingOriginal]);

  // Handle Volume — apply to active element only
  useEffect(() => {
    if (playingOriginal && altAudioRef.current?.src) {
      if (audioRef.current) audioRef.current.volume = 0;
      if (altAudioRef.current) altAudioRef.current.volume = volume;
    } else {
      if (audioRef.current) audioRef.current.volume = volume;
      if (altAudioRef.current) altAudioRef.current.volume = 0;
    }
  }, [volume, playingOriginal]);

  // Handle Playback Rate — apply to both elements so they stay in sync
  useEffect(() => {
    if (audioRef.current) audioRef.current.playbackRate = playbackRate;
    if (altAudioRef.current) altAudioRef.current.playbackRate = playbackRate;
  }, [playbackRate]);

  // Helper to cleanup a job and check if all jobs are done
  const cleanupJob = useCallback((jobId: string, tempId: string) => {
    const jobData = activeJobsRef.current.get(jobId);
    if (jobData) {
      clearInterval(jobData.pollInterval);
      activeJobsRef.current.delete(jobId);
    }

    // Remove temp song
    setSongs(prev => prev.filter(s => s.id !== tempId));

    // Update active job count
    setActiveJobCount(activeJobsRef.current.size);

    // If no more active jobs, set isGenerating to false
    if (activeJobsRef.current.size === 0) {
      setIsGenerating(false);
    }
  }, []);

  // Refresh songs list (called when any job completes successfully)
  const refreshSongsList = useCallback(async () => {
    if (!token) return;
    try {
      const response = await songsApi.getMySongs(token);
      const loadedSongs: Song[] = response.songs.map(s => ({
        id: s.id,
        title: s.title,
        lyrics: s.lyrics,
        style: s.style,
        coverUrl: `https://picsum.photos/seed/${s.id}/400/400`,
        duration: s.duration && s.duration > 0 ? `${Math.floor(s.duration / 60)}:${String(Math.floor(s.duration % 60)).padStart(2, '0')}` : '0:00',
        createdAt: new Date(s.created_at),
        tags: s.tags || [],
        audioUrl: getAudioUrl(s.audio_url, s.id),
        isPublic: s.is_public,
        likeCount: s.like_count || 0,
        viewCount: s.view_count || 0,
        userId: s.user_id,
        creator: s.creator,
        ditModel: s.ditModel,
        generationParams: normalizeGenerationParams(s),
      }));

      // Preserve only generating songs that still exist in the database
      // Preserve any generating songs (they have temp IDs that won't be in the database)
      // Merge: keep generating songs + loaded songs (SongList sorts by date)
      setSongs(prev => {
        const generatingSongs = prev.filter(s => s.isGenerating);
        const loadedById = new Map(loadedSongs.map(s => [s.id, s]));
        // Don't duplicate: drop loaded songs that share an ID with generating ones
        generatingSongs.forEach(g => loadedById.delete(g.id));
        return [...generatingSongs, ...loadedById.values()];
      });
      // If the current selection was a temp/generating song, replace it with newest real song
      if (selectedSong?.isGenerating || (selectedSong && !loadedSongs.some(s => s.id === selectedSong.id))) {
        setSelectedSong(loadedSongs[0] ?? null);
      }
    } catch (error) {
      console.error('Failed to refresh songs:', error);
    }
  }, [token]);

  const beginPollingJob = useCallback((jobId: string, tempId: string) => {
    if (!token) return;
    if (activeJobsRef.current.has(jobId)) return;

    // Track consecutive poll failures — tolerate transient errors instead of
    // immediately killing queued jobs that the backend is still processing.
    let consecutiveErrors = 0;
    const MAX_CONSECUTIVE_ERRORS = 30; // ~60s of continuous failure at 2s intervals

    const pollInterval = setInterval(async () => {
      try {
        const status = await generateApi.getStatus(jobId, token);

        // Successful poll — reset error counter
        consecutiveErrors = 0;

        // When job is queued/pending, don't show progress from another running job
        const isQueued = status.status === 'queued' || status.status === 'pending';
        const normalizedProgress = isQueued
          ? 0
          : (Number.isFinite(Number(status.progress))
            ? (Number(status.progress) > 1 ? Number(status.progress) / 100 : Number(status.progress))
            : undefined);

        setSongs(prev => prev.map(s => {
          if (s.id === tempId) {
            return {
              ...s,
              queuePosition: isQueued ? status.queuePosition : undefined,
              progress: normalizedProgress ?? s.progress,
              stage: isQueued ? (status.queuePosition ? `Queued #${status.queuePosition}` : 'Queued') : (status.stage ?? s.stage),
            };
          }
          return s;
        }));

        if (status.status === 'succeeded' && status.result) {
          cleanupJob(jobId, tempId);

          // Replace temp song in-place with real song data from DB
          // This preserves position instead of reloading the full list
          try {
            const response = await songsApi.getMySongs(token);
            const loadedSongs: Song[] = response.songs.map((s: any) => ({
              id: s.id,
              title: s.title,
              lyrics: s.lyrics,
              style: s.style,
              coverUrl: `https://picsum.photos/seed/${s.id}/400/400`,
              duration: s.duration && s.duration > 0 ? `${Math.floor(s.duration / 60)}:${String(Math.floor(s.duration % 60)).padStart(2, '0')}` : '0:00',
              createdAt: new Date(s.created_at || s.createdAt),
              tags: s.tags || [],
              audioUrl: s.audioUrl || getAudioUrl(s.audio_url, s.id),
              isPublic: s.is_public !== undefined ? s.is_public : s.isPublic,
              likeCount: s.like_count || s.likeCount || 0,
              viewCount: s.view_count || s.viewCount || 0,
              userId: s.user_id || s.userId,
              creator: s.creator,
              ditModel: s.ditModel,
              generationParams: s.generationParams ?? normalizeGenerationParams(s),
            }));
            setSongs(prev => {
              const generatingSongs = prev.filter(s => s.isGenerating);
              const loadedById = new Map(loadedSongs.map(s => [s.id, s]));
              generatingSongs.forEach(g => loadedById.delete(g.id));
              return [...generatingSongs, ...loadedById.values()];
            });
          } catch (refreshErr) {
            console.error('Failed to refresh after completion:', refreshErr);
            // Fallback: just do a full refresh
            await refreshSongsList();
          }

          if (window.innerWidth < 768) {
            setMobileShowList(true);
          }
        } else if (status.status === 'failed') {
          cleanupJob(jobId, tempId);
          console.error(`Job ${jobId} failed:`, status.error);
          showToast(`Generation failed: ${status.error || 'Unknown error'}`, 'error');
        }
      } catch (pollError) {
        consecutiveErrors++;
        console.warn(`Polling error for job ${jobId} (${consecutiveErrors}/${MAX_CONSECUTIVE_ERRORS}):`, pollError);

        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
          console.error(`Job ${jobId}: too many consecutive poll failures, giving up`);
          cleanupJob(jobId, tempId);
          showToast('Generation lost contact with backend — check if it completed in your library', 'error');
        }
        // Otherwise keep polling — transient errors are normal when the backend is busy
      }
    }, 2000);

    activeJobsRef.current.set(jobId, { tempId, pollInterval });
    setActiveJobCount(activeJobsRef.current.size);

    // Hard safety timeout: 1 hour accommodates deep queues (e.g. 10+ songs)
    setTimeout(() => {
      if (activeJobsRef.current.has(jobId)) {
        console.warn(`Job ${jobId} timed out after 1 hour`);
        cleanupJob(jobId, tempId);
        showToast('Generation timed out — check if it completed in your library', 'error');
      }
    }, 3_600_000);
  }, [token, cleanupJob, refreshSongsList]);

  const buildTempSongFromParams = (params: GenerationParams, tempId: string, createdAt?: string) => ({
    id: tempId,
    title: params.title || 'Generating...',
    lyrics: '',
    style: params.style || params.songDescription || '',
    coverUrl: 'https://picsum.photos/200/200?blur=10',
    duration: '--:--',
    createdAt: createdAt ? new Date(createdAt) : new Date(),
    isGenerating: true,
    tags: params.customMode ? ['custom'] : ['simple'],
    isPublic: true,
  });

  // Handlers
  const handleGenerate = async (params: GenerationParams) => {
    if (!isAuthenticated || !token) {
      setShowUsernameModal(true);
      return;
    }

    setIsGenerating(true);
    setCurrentView('create');
    setMobileShowList(false);

    try {
      const storedGlobalParams = localStorage.getItem('globalMasteringParams');
      const globalMasteringParams = storedGlobalParams ? JSON.parse(storedGlobalParams) : undefined;

      // Merge mastering params: global defaults from MasteringConsole are the base,
      // per-job params from Cover Settings (e.g. newly uploaded reference file) override.
      const mergedMasteringParams = (globalMasteringParams || params.masteringParams)
        ? { ...(globalMasteringParams || {}), ...(params.masteringParams || {}) }
        : undefined;

      const job = await generateApi.startGeneration({
        ...params,
        // Normalize duration: treat 0 or negative as undefined
        duration: params.duration && params.duration > 0 ? params.duration : undefined,
        ...(mergedMasteringParams ? { masteringParams: mergedMasteringParams } : {})
      }, token);

      const tempId = `job_${job.jobId}`;
      const tempSong: Song = {
        ...buildTempSongFromParams(params, tempId),
        queuePosition: job.status === 'queued' ? job.queuePosition : undefined,
        stage: job.status,
      };

      setSongs(prev => [tempSong, ...prev]);
      setSelectedSong(tempSong);
      setShowRightSidebar(true);

      beginPollingJob(job.jobId, tempId);

    } catch (e) {
      console.error('Generation error:', e);
      // Only set isGenerating to false if no other jobs are running
      if (activeJobsRef.current.size === 0) {
        setIsGenerating(false);
      }

      const msg = e instanceof Error ? e.message : '';
      const userMsg = msg ? `${t('generationFailed')}: ${msg}` : t('generationFailed');
      showToast(userMsg, 'error');
    }
  };

  // Resume active jobs on refresh so progress keeps updating
  useEffect(() => {
    if (!isAuthenticated || !token) return;

    const resumeJobs = async () => {
      try {
        const history = await generateApi.getHistory(token);
        const jobs = Array.isArray(history.jobs) ? history.jobs : [];

        const activeStatuses = new Set(['pending', 'queued', 'running']);
        const jobsToResume = jobs.filter((job: any) => activeStatuses.has(job.status));

        if (jobsToResume.length === 0) return;

        setSongs(prev => {
          const existingIds = new Set(prev.map(s => s.id));
          const next = [...prev];

          for (const job of jobsToResume) {
            const jobId = job.id || job.jobId;
            if (!jobId) continue;
            const tempId = `job_${jobId}`;
            if (existingIds.has(tempId)) continue;

            const params = (() => {
              try {
                if (!job.params) return {};
                return typeof job.params === 'string' ? JSON.parse(job.params) : job.params;
              } catch {
                return {};
              }
            })();

            next.unshift(buildTempSongFromParams(params, tempId, job.created_at));
            existingIds.add(tempId);
          }
          return next;
        });

        for (const job of jobsToResume) {
          const jobId = job.id || job.jobId;
          if (!jobId) continue;
          const tempId = `job_${jobId}`;
          beginPollingJob(jobId, tempId);
        }
      } catch (error) {
        console.error('Failed to resume jobs:', error);
      }
    };

    resumeJobs();
  }, [isAuthenticated, token, beginPollingJob]);

  const togglePlay = () => {
    if (!currentSong) return;
    setIsPlaying(!isPlaying);
  };

  const normalizeSongForState = useCallback((raw: any): Song => {
    const durationText = (() => {
      const d = raw?.duration;
      if (typeof d === 'string') return d;
      if (typeof d === 'number' && Number.isFinite(d) && d > 0) {
        return `${Math.floor(d / 60)}:${String(Math.floor(d % 60)).padStart(2, '0')}`;
      }
      return '0:00';
    })();

    const createdAt = (() => {
      const v = raw?.createdAt ?? raw?.created_at;
      const d = v ? new Date(v) : new Date();
      return Number.isNaN(d.getTime()) ? new Date() : d;
    })();

    const audioUrl = raw?.audioUrl ?? raw?.audio_url;

    return {
      id: String(raw?.id ?? ''),
      title: String(raw?.title ?? ''),
      lyrics: String(raw?.lyrics ?? ''),
      style: String(raw?.style ?? ''),
      coverUrl: raw?.coverUrl || raw?.cover_url || `https://picsum.photos/seed/${raw?.id}/400/400`,
      duration: durationText,
      createdAt,
      tags: Array.isArray(raw?.tags) ? raw.tags : [],
      audioUrl: getAudioUrl(audioUrl, raw?.id),
      isPublic: raw?.isPublic ?? raw?.is_public,
      likeCount: raw?.likeCount ?? raw?.like_count,
      viewCount: raw?.viewCount ?? raw?.view_count,
      userId: raw?.userId ?? raw?.user_id,
      creator: raw?.creator,
      creator_avatar: raw?.creator_avatar,
      ditModel: raw?.ditModel ?? raw?.dit_model,
      isGenerating: raw?.isGenerating,
      queuePosition: raw?.queuePosition,
      progress: raw?.progress,
      stage: raw?.stage,
      generationParams: raw?.generationParams ?? normalizeGenerationParams(raw),
    };
  }, [normalizeGenerationParams, getAudioUrl]);

  const playSong = (song: Song, list?: Song[]) => {
    const normalizedSong = normalizeSongForState(song);
    const normalizedList = Array.isArray(list) ? list.map(normalizeSongForState) : undefined;

    const nextQueue = normalizedList && normalizedList.length > 0
      ? normalizedList
      : (playQueue.length > 0 && playQueue.some(s => s.id === normalizedSong.id))
        ? playQueue
        : (songs.some(s => s.id === normalizedSong.id) ? songs : [normalizedSong]);
    const nextIndex = nextQueue.findIndex(s => s.id === normalizedSong.id);
    setPlayQueue(nextQueue.map(normalizeSongForState));
    setQueueIndex(nextIndex);

    if (currentSong?.id !== normalizedSong.id) {
      const updatedSong = { ...normalizedSong, viewCount: (normalizedSong.viewCount || 0) + 1 };
      setCurrentSong(updatedSong);
      setSelectedSong(updatedSong);
      setIsPlaying(true);
      setPlayingOriginal(false); // Reset M/O toggle for new song
      setSongs(prev => prev.map(s => s.id === updatedSong.id ? { ...s, ...updatedSong } : s));
      songsApi.trackPlay(updatedSong.id, token).catch(err => console.error('Failed to track play:', err));
    } else {
      togglePlay();
    }
    if (currentSong?.id === normalizedSong.id) {
      setSelectedSong(normalizedSong);
    }
    setShowRightSidebar(true);
  };

  const handleSeek = (time: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    if (Number.isNaN(audio.duration) || audio.readyState < 1 || audio.seekable.length === 0) {
      pendingSeekRef.current = time;
      return;
    }
    audio.currentTime = time;
    // Keep alt audio in sync for seamless M/O toggle
    const altAudio = altAudioRef.current;
    if (altAudio?.src && altAudio.readyState >= 1) {
      altAudio.currentTime = time;
    }
    setCurrentTime(time);
  };

  const toggleLike = async (songId: string) => {
    if (!token) return;

    const isLiked = likedSongIds.has(songId);

    // Optimistic update
    setLikedSongIds(prev => {
      const next = new Set(prev);
      if (isLiked) next.delete(songId);
      else next.add(songId);
      return next;
    });

    setSongs(prev => prev.map(s => {
      if (s.id === songId) {
        const newCount = (s.likeCount || 0) + (isLiked ? -1 : 1);
        return { ...s, likeCount: Math.max(0, newCount) };
      }
      return s;
    }));

    if (selectedSong?.id === songId) {
      setSelectedSong(prev => prev ? {
        ...prev,
        likeCount: Math.max(0, (prev.likeCount || 0) + (isLiked ? -1 : 1))
      } : null);
    }

    // Persist to database
    try {
      await songsApi.toggleLike(songId, token);
    } catch (error) {
      console.error('Failed to toggle like:', error);
      // Revert on error
      setLikedSongIds(prev => {
        const next = new Set(prev);
        if (isLiked) next.add(songId);
        else next.delete(songId);
        return next;
      });
    }
  };

  const handleDeleteSong = (song: Song) => {
    handleDeleteSongs([song]);
  };

  const handleDeleteSongs = (songsToDelete: Song[], onSuccess?: () => void) => {
    if (!token || songsToDelete.length === 0) return;

    const isSingle = songsToDelete.length === 1;
    const title = isSingle ? t('confirmDeleteTitle') : t('confirmDeleteManyTitle');
    const message = isSingle
      ? t('deleteSongConfirm').replace('{title}', songsToDelete[0].title)
      : t('deleteSongsConfirm').replace('{count}', String(songsToDelete.length));

    setConfirmDialog({
      title,
      message,
      onConfirm: async () => {
        setConfirmDialog(null);

        const succeeded: string[] = [];
        const failed: string[] = [];
        const hardRemoveIds = new Set<string>();

        const cancelPollingForTempId = (tempId: string) => {
          for (const [jobId, jobData] of activeJobsRef.current.entries()) {
            if (jobData.tempId === tempId) {
              cleanupJob(jobId, tempId);
              break;
            }
          }
        };

        for (const song of songsToDelete) {
          if (song.id.startsWith('temp_')) {
            cancelPollingForTempId(song.id);
            hardRemoveIds.add(song.id);
            succeeded.push(song.id);
            continue;
          }

          if (song.id.startsWith('job_')) {
            cancelPollingForTempId(song.id);
            hardRemoveIds.add(song.id);
            succeeded.push(song.id);

            const jobId = song.id.slice('job_'.length);
            try {
              await generateApi.deleteJob(jobId, token!);
            } catch (error) {
              const msg = error instanceof Error ? error.message : String(error);
              if (!msg.startsWith('404:')) {
                console.error('Failed to delete generation job:', error);
              }
            }
            continue;
          }

          try {
            await songsApi.deleteSong(song.id, token!);
            hardRemoveIds.add(song.id);
            succeeded.push(song.id);
          } catch (error) {
            console.error('Failed to delete song:', error);
            failed.push(song.id);
          }
        }

        if (hardRemoveIds.size > 0) {
          setSongs(prev => prev.filter(s => !hardRemoveIds.has(s.id)));

          setLikedSongIds(prev => {
            const next = new Set(prev);
            succeeded.forEach(id => next.delete(id));
            return next;
          });

          if (selectedSong?.id && succeeded.includes(selectedSong.id)) {
            setSelectedSong(null);
          }

          if (currentSong?.id && succeeded.includes(currentSong.id)) {
            setCurrentSong(null);
            setIsPlaying(false);
            if (audioRef.current) {
              audioRef.current.pause();
              audioRef.current.src = '';
            }
          }

          setPlayQueue(prev => prev.filter(s => !hardRemoveIds.has(s.id)));

          // Call success callback (e.g., to exit selection mode)
          onSuccess?.();
        }

        if (failed.length > 0) {
          showToast(t('songsDeletedPartial').replace('{succeeded}', String(succeeded.length)).replace('{total}', String(songsToDelete.length)), 'error');
        } else if (isSingle) {
          showToast(t('songDeleted'));
        } else {
          showToast(t('songsDeletedSuccess'));
        }
      },
    });
  };

  const handleDeleteAll = () => {
    if (!token) return;
    const songCount = songs.filter(s => !s.isGenerating).length;
    if (songCount === 0) return;

    setConfirmDialog({
      title: t('deleteAllTracks'),
      message: t('deleteAllTracksConfirm').replace('{count}', String(songCount)),
      onConfirm: async () => {
        setConfirmDialog(null);
        try {
          await Promise.all([
            songsApi.deleteAllSongs(token!),
            songsApi.deleteAllReferenceTracks(token!),
          ]);

          // Stop playback
          if (audioRef.current) {
            audioRef.current.pause();
            audioRef.current.src = '';
          }
          setCurrentSong(null);
          setIsPlaying(false);
          setSelectedSong(null);
          setPlayQueue([]);
          setLikedSongIds(new Set());
          setReferenceTracks([]);

          // Keep only generating songs (temp/job IDs)
          setSongs(prev => prev.filter(s => s.isGenerating));

          showToast(t('allTracksDeleted'));
        } catch (error) {
          console.error('Failed to delete all songs:', error);
          showToast(t('deleteAllFailed'), 'error');
        }
      },
    });
  };

  const handleCancelJob = async (song: Song) => {
    if (!token || !song.isGenerating) return;
    // Generating placeholder IDs are 'job_<dbJobId>' — extract the DB job ID
    const jobId = song.id.startsWith('job_') ? song.id.slice(4) : song.id;
    try {
      await generateApi.cancelJob(jobId, token);
      // Remove the generating placeholder from the song list
      setSongs(prev => prev.filter(s => s.id !== song.id));
      showToast('Generation cancelled');
    } catch (error) {
      console.error('Cancel job error:', error);
      showToast('Failed to cancel generation', 'error');
    }
  };

  const handleCancelAll = async () => {
    if (!token) return;
    try {
      const result = await generateApi.cancelAllJobs(token);
      // Remove all generating placeholders from the song list
      setSongs(prev => prev.filter(s => !s.isGenerating));
      showToast(`Cleared ${result.cancelled || 0} stuck job(s)`);
    } catch (error) {
      console.error('Cancel all error:', error);
      showToast('Failed to clear queue', 'error');
    }
  };

  const handleDeleteReferenceTrack = (trackId: string) => {
    if (!token) return;

    setConfirmDialog({
      title: t('delete'),
      message: t('deleteUploadConfirm'),
      onConfirm: async () => {
        setConfirmDialog(null);
        try {
          const response = await fetch(`/api/reference-tracks/${trackId}`, {
            method: 'DELETE',
            headers: { Authorization: `Bearer ${token!}` }
          });
          if (!response.ok) {
            throw new Error('Failed to delete upload');
          }
          setReferenceTracks(prev => prev.filter(track => track.id !== trackId));
          showToast(t('songDeleted'));
        } catch (error) {
          console.error('Failed to delete upload:', error);
          showToast(t('failedToDeleteSong'), 'error');
        }
      },
    });
  };

  const createPlaylist = async (name: string, description: string) => {
    if (!token) return;
    try {
      const res = await playlistsApi.create(name, description, true, token);
      setPlaylists(prev => [res.playlist, ...prev]);

      if (songToAddToPlaylist) {
        await playlistsApi.addSong(res.playlist.id, songToAddToPlaylist.id, token);
        setSongToAddToPlaylist(null);
        playlistsApi.getMyPlaylists(token).then(r => setPlaylists(r.playlists));
      }
      showToast(t('playlistCreated'));
    } catch (error) {
      console.error('Create playlist error:', error);
      showToast(t('failedToCreatePlaylist'), 'error');
    }
  };

  const openAddToPlaylistModal = (song: Song) => {
    setSongToAddToPlaylist(song);
    setIsAddToPlaylistModalOpen(true);
  };

  const addSongToPlaylist = async (playlistId: string) => {
    if (!songToAddToPlaylist || !token) return;
    try {
      await playlistsApi.addSong(playlistId, songToAddToPlaylist.id, token);
      setSongToAddToPlaylist(null);
      showToast(t('songAddedToPlaylist'));
      playlistsApi.getMyPlaylists(token).then(r => setPlaylists(r.playlists));
    } catch (error) {
      console.error('Add song error:', error);
      showToast(t('failedToAddSong'), 'error');
    }
  };

  const handleNavigateToPlaylist = (playlistId: string) => {
    setViewingPlaylistId(playlistId);
    setCurrentView('playlist');
    window.history.pushState({}, '', `/playlist/${playlistId}`);
  };

  const handleUseAsReference = (song: Song) => {
    if (!song.audioUrl) return;
    setPendingAudioSelection({ target: 'reference', url: song.audioUrl, title: song.title });
    setCurrentView('create');
    setMobileShowList(false);
  };

  const handleCoverSong = (song: Song) => {
    if (!song.audioUrl) return;
    setPendingAudioSelection({ target: 'source', url: song.audioUrl, title: song.title });
    setCurrentView('create');
    setMobileShowList(false);
  };

  const handleUseUploadAsReference = (track: { audio_url: string; filename: string }) => {
    setPendingAudioSelection({
      target: 'reference',
      url: track.audio_url,
      title: track.filename.replace(/\.[^/.]+$/, ''),
    });
    setCurrentView('create');
    setMobileShowList(false);
  };

  const handleCoverUpload = (track: { audio_url: string; filename: string }) => {
    setPendingAudioSelection({
      target: 'source',
      url: track.audio_url,
      title: track.filename.replace(/\.[^/.]+$/, ''),
    });
    setCurrentView('create');
    setMobileShowList(false);
  };

  const handleBackFromPlaylist = () => {
    setViewingPlaylistId(null);
    setCurrentView('library');
    window.history.pushState({}, '', '/library');
  };

  const openDownloadModal = (song: Song) => {
    setBulkDownloadSongs([]);
    setSongToDownload(song);
    setIsDownloadModalOpen(true);
  };

  const openBulkDownloadModal = (songs: Song[]) => {
    setBulkDownloadSongs(songs);
    setSongToDownload(null);
    setIsDownloadModalOpen(true);
  };

  const handleDownloadFormat = async (format: DownloadFormat, version: 'mastered' | 'original' | 'both' = 'mastered') => {
    if (!songToDownload?.audioUrl) return;

    try {
      const downloadSingleURL = (url: string, suffix: string) => {
        const targetUrl = new URL('/api/songs/download', window.location.origin);
        targetUrl.searchParams.set('audioUrl', url);
        targetUrl.searchParams.set('title', `${songToDownload.title || 'song'}${suffix}`);
        targetUrl.searchParams.set('format', format);
        if (songToDownload.id) {
          targetUrl.searchParams.set('songId', songToDownload.id);
        }
        if (format === 'mp3') {
          const br = localStorage.getItem('mp3_export_bitrate');
          if (br) targetUrl.searchParams.set('mp3Bitrate', br);
        }
        if (format === 'opus') {
          const br = localStorage.getItem('opus_export_bitrate');
          if (br) targetUrl.searchParams.set('opusBitrate', br);
        }

        const link = document.createElement('a');
        link.href = targetUrl.toString();
        const ext = format === 'opus' ? 'ogg' : format;
        link.download = `${songToDownload.title || 'song'}${suffix}.${ext}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      };

      if (version === 'mastered' || version === 'both') {
        downloadSingleURL(songToDownload.audioUrl, '');
      }
      
      if (version === 'original' || version === 'both') {
        const origUrl = songToDownload.generationParams?.originalAudioUrl || (songToDownload as any).originalAudioUrl;
        if (origUrl) {
          setTimeout(() => downloadSingleURL(origUrl, ' (Unmastered)'), version === 'both' ? 500 : 0);
        }
      }

    } catch (error) {
      console.error('Download failed:', error);
      showToast(`Failed to download ${format.toUpperCase()}`, 'error');
    }
  };

  const handleBulkDownloadFormat = async (format: DownloadFormat, version: 'mastered' | 'original' | 'both' = 'mastered') => {
    if (bulkDownloadSongs.length === 0) return;

    const downloadOneSong = (song: Song, url: string, suffix: string) => {
      const targetUrl = new URL('/api/songs/download', window.location.origin);
      targetUrl.searchParams.set('audioUrl', url);
      targetUrl.searchParams.set('title', `${song.title || 'song'}${suffix}`);
      targetUrl.searchParams.set('format', format);
      if (song.id) targetUrl.searchParams.set('songId', song.id);
      if (format === 'mp3') {
        const br = localStorage.getItem('mp3_export_bitrate');
        if (br) targetUrl.searchParams.set('mp3Bitrate', br);
      }
      if (format === 'opus') {
        const br = localStorage.getItem('opus_export_bitrate');
        if (br) targetUrl.searchParams.set('opusBitrate', br);
      }
      const link = document.createElement('a');
      link.href = targetUrl.toString();
      const ext = format === 'opus' ? 'ogg' : format;
      link.download = `${song.title || 'song'}${suffix}.${ext}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };

    try {
      let delay = 0;
      for (const song of bulkDownloadSongs) {
        if (!song.audioUrl) continue;
        if (version === 'mastered' || version === 'both') {
          setTimeout(() => downloadOneSong(song, song.audioUrl, ''), delay);
          delay += 400;
        }
        if (version === 'original' || version === 'both') {
          const origUrl = song.generationParams?.originalAudioUrl || (song as any).originalAudioUrl;
          if (origUrl) {
            setTimeout(() => downloadOneSong(song, origUrl, ' (Unmastered)'), delay);
            delay += 400;
          }
        }
      }
      showToast(`Downloading ${bulkDownloadSongs.length} tracks as ${format.toUpperCase()}`, 'success');
    } catch (error) {
      console.error('Bulk download failed:', error);
      showToast(`Bulk download failed`, 'error');
    }
  };

  const openVideoGenerator = (song: Song) => {
    if (isPlaying) {
      setIsPlaying(false);
      if (audioRef.current) audioRef.current.pause();
    }
    setSongForVideo(song);
    setIsVideoModalOpen(true);
  };

  // Handle username setup
  const handleUsernameSubmit = async (username: string) => {
    await setupUser(username);
    setShowUsernameModal(false);
  };

  // Render Layout Logic
  const renderContent = () => {
    switch (currentView) {
      case 'library': {
        const allSongs = user ? songs.filter(s => s.userId === user.id) : [];
        return (
          <LibraryView
            allSongs={allSongs}
            likedSongs={songs.filter(s => likedSongIds.has(s.id))}
            playlists={playlists}
            referenceTracks={referenceTracks}
            onPlaySong={playSong}
            onCreatePlaylist={() => {
              setSongToAddToPlaylist(null);
              setIsCreatePlaylistModalOpen(true);
            }}
            onSelectPlaylist={(p) => handleNavigateToPlaylist(p.id)}
            onAddToPlaylist={openAddToPlaylistModal}
            onOpenVideo={openVideoGenerator}
            onReusePrompt={handleReuse}
            onDeleteSong={handleDeleteSong}
            onDeleteReferenceTrack={handleDeleteReferenceTrack}
          />
        );
      }

      case 'profile':
        if (!viewingUsername) {
          return (
            <div className="flex flex-col items-center justify-center h-full gap-4 bg-black">
              <div className="text-zinc-400">{t('userNotFound')}</div>
              <button
                onClick={() => {
                  setCurrentView('library');
                  window.history.pushState({}, '', '/library');
                }}
                className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-white"
              >
                {t('goBack')}
              </button>
            </div>
          );
        }
        return (
          <UserProfile
            username={viewingUsername}
            onBack={handleBackFromProfile}
            onPlaySong={playSong}
            onNavigateToProfile={handleNavigateToProfile}
            onNavigateToPlaylist={handleNavigateToPlaylist}
            currentSong={currentSong}
            isPlaying={isPlaying}
            likedSongIds={likedSongIds}
            onToggleLike={toggleLike}
          />
        );

      case 'playlist':
        if (!viewingPlaylistId) {
          // Auto-navigate back to library if no playlist is selected
          return (
            <div className="flex flex-col items-center justify-center h-full gap-4 bg-black">
              <div className="text-zinc-400">{t('playlistNotFound')}</div>
              <button
                onClick={() => {
                  setCurrentView('library');
                  window.history.pushState({}, '', '/library');
                }}
                className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-white"
              >
                {t('goBack')}
              </button>
            </div>
          );
        }
        return (
          <PlaylistDetail
            playlistId={viewingPlaylistId}
            onBack={handleBackFromPlaylist}
            onPlaySong={playSong}
            onSelect={(s) => {
              setSelectedSong(normalizeSongForState(s));
              setShowRightSidebar(true);
            }}
            onNavigateToProfile={handleNavigateToProfile}
          />
        );

      case 'song':
        if (!viewingSongId) {
          return (
            <div className="flex flex-col items-center justify-center h-full gap-4 bg-black">
              <div className="text-zinc-400">{t('songNotFound')}</div>
              <button
                onClick={() => {
                  setCurrentView('library');
                  window.history.pushState({}, '', '/library');
                }}
                className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-white"
              >
                {t('goBack')}
              </button>
            </div>
          );
        }
        return (
          <SongProfile
            songId={viewingSongId}
            onBack={handleBackFromSong}
            onPlay={playSong}
            onNavigateToProfile={handleNavigateToProfile}
            currentSong={currentSong}
            isPlaying={isPlaying}
            likedSongIds={likedSongIds}
            onToggleLike={toggleLike}
            onDelete={handleDeleteSong}
          />
        );

      case 'search':
        return (
          <SearchPage
            onPlaySong={playSong}
            currentSong={currentSong}
            isPlaying={isPlaying}
            onNavigateToProfile={handleNavigateToProfile}
            onNavigateToSong={handleNavigateToSong}
            onNavigateToPlaylist={handleNavigateToPlaylist}
          />
        );


      case 'create':
      default:
        return (
          <div className="flex h-full overflow-hidden relative w-full bg-zinc-50 dark:bg-suno-panel">
            {/* Create Panel — resizable */}
            <div
              className="w-full md:w-auto flex-shrink-0 h-full border-r border-zinc-200 dark:border-white/5 z-10"
              style={{ width: isDesktop ? createPanelWidth : undefined }}
            >
              <CreatePanel
                onGenerate={handleGenerate}
                isGenerating={isGenerating}
                activeJobCount={activeJobCount}
                initialData={reuseData}
                createdSongs={songs}
                pendingAudioSelection={pendingAudioSelection}
                onAudioSelectionApplied={() => setPendingAudioSelection(null)}
                diffPinnedA={diffPinnedA}
                diffPinnedB={diffPinnedB}
                onClearDiffA={() => setDiffPinnedA(null)}
                onClearDiffB={() => setDiffPinnedB(null)}
              />
            </div>
            {/* Left resize handle — outside scrollable area so it doesn't overlap the scrollbar */}
            <div
              className="hidden md:flex flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors"
              onMouseDown={(e) => {
                e.preventDefault();
                const startX = e.clientX;
                const startW = createPanelWidth;
                const onMove = (ev: MouseEvent) => {
                  const newW = Math.min(700, Math.max(490, startW + ev.clientX - startX));
                  setCreatePanelWidth(newW);
                };
                const onUp = () => {
                  document.removeEventListener('mousemove', onMove);
                  document.removeEventListener('mouseup', onUp);
                  document.body.style.cursor = '';
                  document.body.style.userSelect = '';
                };
                document.body.style.cursor = 'col-resize';
                document.body.style.userSelect = 'none';
                document.addEventListener('mousemove', onMove);
                document.addEventListener('mouseup', onUp);
              }}
            >
              <div className="w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-pink-400 transition-colors" />
            </div>

            {/* Song List */}
            <div className="hidden md:flex flex-1 flex-col h-full overflow-hidden bg-white dark:bg-suno-DEFAULT">
              <SongList
                songs={songs}
                currentSong={currentSong}
                selectedSong={selectedSong}
                likedSongIds={likedSongIds}
                isPlaying={isPlaying}
                referenceTracks={referenceTracks}
                onPlay={playSong}
                onSelect={(s) => {
                  setSelectedSong(s);
                  setShowRightSidebar(true);
                }}
                onToggleLike={toggleLike}
                onAddToPlaylist={openAddToPlaylistModal}
                onOpenVideo={openVideoGenerator}
                onShowDetails={handleShowDetails}
                onNavigateToProfile={handleNavigateToProfile}
                onReusePrompt={handleReuse}
                onDelete={handleDeleteSong}
                onDeleteMany={handleDeleteSongs}
                onBulkDownload={openBulkDownloadModal}
                onDeleteAll={handleDeleteAll}
                onUseAsReference={handleUseAsReference}
                onCoverSong={handleCoverSong}
                onUpscaleToHQ={handleUpscaleToHQ}
                onDownloadFormat={openDownloadModal}
                onOpenRemaster={(song) => {
                    setRemasterSong(song);
                    setShowRemasterConsole(true);
                }}
                onToggleMastering={() => setPlayingOriginal(prev => !prev)}
                playingOriginal={playingOriginal}
                onUseUploadAsReference={handleUseUploadAsReference}
                onCoverUpload={handleCoverUpload}
                onSongUpdate={handleSongUpdate}
                onDeleteUpload={handleDeleteReferenceTrack}
                showVisualizerBg={showVisualizerBg}
                currentTime={currentTime}
                onSetAsTrackA={(song: Song) => {
                  // If already A → toggle off
                  if (abTrackA?.id === song.id) {
                    handleABClear();
                    showToast('Comparison cleared');
                    return;
                  }
                  // If A already exists → assign as B instead
                  if (abTrackA) {
                    setAbTrackB(song);
                    showToast(`Track B: ${song.title}`);
                  } else {
                    setAbTrackA(song);
                    showToast(`Track A: ${song.title}`);
                  }
                }}
                onSetAsTrackB={(song: Song) => {
                  // If already B → toggle off
                  if (abTrackB?.id === song.id) {
                    setAbTrackB(null);
                    if (abActive) handleABClear();
                    showToast('Track B cleared');
                    return;
                  }
                  setAbTrackB(song);
                  showToast(`Track B: ${song.title}`);
                }}
                abTrackA={abTrackA}
                abTrackB={abTrackB}
                abActive={abActive}
                onABCompare={() => setShowABCompare(true)}
                onABPlay={handleABPlay}
                onABClear={handleABClear}
                onABToggle={handleABToggle}
                // Ablation diff pins
                devMode={devMode}
                diffPinnedA={diffPinnedA}
                diffPinnedB={diffPinnedB}
                onPinDiffA={(song: Song) => setDiffPinnedA(prev => prev?.id === song.id ? null : song)}
                onPinDiffB={(song: Song) => setDiffPinnedB(prev => prev?.id === song.id ? null : song)}
                onCancelJob={handleCancelJob}
                onCancelAll={handleCancelAll}
              />
            </div>

            {/* Right Sidebar — resizable */}
            {showRightSidebar && (
              <>
                {/* Right resize handle — outside scrollable area */}
                <div
                  className="hidden xl:flex flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors"
                  onMouseDown={(e) => {
                    e.preventDefault();
                    const startX = e.clientX;
                    const startW = rightSidebarWidth;
                    const onMove = (ev: MouseEvent) => {
                      // Dragging left = wider (startX - ev.clientX)
                      const newW = Math.min(600, Math.max(280, startW + startX - ev.clientX));
                      setRightSidebarWidth(newW);
                    };
                    const onUp = () => {
                      document.removeEventListener('mousemove', onMove);
                      document.removeEventListener('mouseup', onUp);
                      document.body.style.cursor = '';
                      document.body.style.userSelect = '';
                    };
                    document.body.style.cursor = 'col-resize';
                    document.body.style.userSelect = 'none';
                    document.addEventListener('mousemove', onMove);
                    document.addEventListener('mouseup', onUp);
                  }}
                >
                  <div className="w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-pink-400 transition-colors" />
                </div>
                <div
                  className="hidden xl:block flex-shrink-0 h-full bg-zinc-50 dark:bg-suno-panel z-10 border-l border-zinc-200 dark:border-white/5"
                  style={{ width: rightSidebarWidth }}
                >
                  <RightSidebar
                    song={selectedSong}
                    onClose={() => setShowRightSidebar(false)}
                    onOpenVideo={() => selectedSong && openVideoGenerator(selectedSong)}
                    onReuse={handleReuse}
                    onSongUpdate={handleSongUpdate}
                    onNavigateToProfile={handleNavigateToProfile}
                    onNavigateToSong={handleNavigateToSong}
                    isLiked={selectedSong ? likedSongIds.has(selectedSong.id) : false}
                    onToggleLike={toggleLike}
                    onDelete={handleDeleteSong}
                    onPlay={playSong}
                    isPlaying={isPlaying && currentSong?.id === selectedSong?.id}
                    currentSong={currentSong}
                    currentTime={currentTime}
                    onFullscreenVisualizer={() => setShowFullscreenVisualizer(true)}
                  />
                </div>
              </>
            )}
          </div>
        );
    }
  };

  return (
    <div
      className="flex flex-col h-screen bg-white dark:bg-suno-DEFAULT text-zinc-900 dark:text-white font-sans antialiased selection:bg-pink-500/30 transition-all duration-300"
      style={{ paddingRight: debugPanelOpen ? 400 : 0 }}
    >
      <div className="flex-1 flex overflow-hidden">
        <Sidebar
          currentView={currentView}
          onNavigate={(v) => {
            // Clear viewing states when navigating away from detail pages
            if (v !== 'playlist') {
              setViewingPlaylistId(null);
            }
            if (v !== 'profile') {
              setViewingUsername(null);
            }
            if (v !== 'song') {
              setViewingSongId(null);
            }
            setCurrentView(v);
            if (v === 'create') {
              setMobileShowList(false);
              window.history.pushState({}, '', '/');
            } else if (v === 'library') {
              window.history.pushState({}, '', '/library');
            } else if (v === 'search') {
              window.history.pushState({}, '', '/search');
            }
            if (isMobile) setShowLeftSidebar(false);
          }}
          theme={theme}
          onToggleTheme={toggleTheme}
          user={user}
          onLogin={() => setShowUsernameModal(true)}
          onLogout={logout}
          onOpenSettings={() => setShowSettingsModal(true)}
          isOpen={showLeftSidebar}
          onToggle={() => setShowLeftSidebar(!showLeftSidebar)}
          onQuit={() => {
            setConfirmDialog({
              title: 'Quit HOT-Step 9000',
              message: 'Are you sure you wish to shut down HOT-Step 9000? This will stop the Python API and all servers.',
              confirmLabel: 'Shut Down',
              danger: true,
              onConfirm: async () => {
                setConfirmDialog(null);
                try {
                  await fetch('/api/shutdown', { method: 'POST' });
                } catch { /* server is shutting down */ }
                setIsShutdown(true);
              },
            });
          }}
        />

        <main className="flex-1 flex overflow-hidden relative">
          {renderContent()}
        </main>
      </div>

      <Player
        currentSong={currentSong}
        isPlaying={isPlaying}
        onTogglePlay={togglePlay}
        currentTime={currentTime}
        duration={duration}
        onSeek={handleSeek}
        onNext={playNext}
        onPrevious={playPrevious}
        volume={volume}
        onVolumeChange={setVolume}
        playbackRate={playbackRate}
        onPlaybackRateChange={setPlaybackRate}
        audioRef={audioRef}
        isShuffle={isShuffle}
        onToggleShuffle={() => setIsShuffle(!isShuffle)}
        repeatMode={repeatMode}
        onToggleRepeat={() => setRepeatMode(prev => prev === 'none' ? 'all' : prev === 'all' ? 'one' : 'none')}
        isLiked={currentSong ? likedSongIds.has(currentSong.id) : false}
        onToggleLike={() => currentSong && toggleLike(currentSong.id)}
        onNavigateToSong={handleNavigateToSong}
        onOpenVideo={() => currentSong && openVideoGenerator(currentSong)}
        onReusePrompt={() => currentSong && handleReuse(currentSong)}
        onDownloadFormat={() => currentSong && openDownloadModal(currentSong)}
        onAddToPlaylist={() => currentSong && openAddToPlaylistModal(currentSong)}
        onDelete={() => currentSong && handleDeleteSong(currentSong)}
        onOpenRemaster={(song) => {
          setRemasterSong(song);
          setShowRemasterConsole(true);
        }}
        playingOriginal={playingOriginal}
        onToggleMastering={() => setPlayingOriginal(prev => !prev)}
      />

      <CreatePlaylistModal
        isOpen={isCreatePlaylistModalOpen}
        onClose={() => setIsCreatePlaylistModalOpen(false)}
        onCreate={createPlaylist}
      />
      <AddToPlaylistModal
        isOpen={isAddToPlaylistModalOpen}
        onClose={() => setIsAddToPlaylistModalOpen(false)}
        playlists={playlists}
        onSelect={addSongToPlaylist}
        onCreateNew={() => {
          setIsAddToPlaylistModalOpen(false);
          setIsCreatePlaylistModalOpen(true);
        }}
      />
      <Toast
        message={toast.message}
        type={toast.type}
        isVisible={toast.isVisible}
        onClose={closeToast}
      />
      <VideoGeneratorModal
        isOpen={isVideoModalOpen}
        onClose={() => setIsVideoModalOpen(false)}
        song={songForVideo}
      />
      <UsernameModal
        isOpen={showUsernameModal}
        onSubmit={handleUsernameSubmit}
      />
      <DownloadModal
        isOpen={isDownloadModalOpen}
        onClose={() => { setIsDownloadModalOpen(false); setBulkDownloadSongs([]); }}
        onDownload={bulkDownloadSongs.length > 0 ? handleBulkDownloadFormat : handleDownloadFormat}
        songTitle={songToDownload?.title}
        songCount={bulkDownloadSongs.length > 0 ? bulkDownloadSongs.length : undefined}
        hasOriginal={bulkDownloadSongs.length > 0
          ? bulkDownloadSongs.some(s => !!(s.generationParams?.originalAudioUrl || (s as any).originalAudioUrl))
          : !!(songToDownload?.generationParams?.originalAudioUrl || (songToDownload as any)?.originalAudioUrl)
        }
      />
      <SettingsModal
        isOpen={showSettingsModal}
        onClose={() => setShowSettingsModal(false)}
        theme={theme}
        onToggleTheme={toggleTheme}
        onNavigateToProfile={handleNavigateToProfile}
      />
      <StemSplitterModal />
      <AudioEnhancerModal />
      <RefineModal />
      {/* Re-master via Mastering Console */}
      <MasteringConsoleModal
        isOpen={showRemasterConsole}
        onClose={() => { setShowRemasterConsole(false); setRemasterSong(null); }}
        currentParams={remasterSong?.generationParams?.masteringParams || null}
        onParamsChange={async (params) => {
          if (!remasterSong) return;
          const originalUrl = remasterSong.generationParams?.originalAudioUrl;
          if (!originalUrl) { showToast('No original audio available for re-mastering', 'error'); return; }
          try {
            showToast('Re-mastering...', 'info');
            const PYTHON_API = `http://${window.location.hostname}:8001`;
            const resp = await fetch(`${PYTHON_API}/v1/mastering/apply`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ audio_path: originalUrl, mastering_params: params }),
            });
            if (!resp.ok) throw new Error(`Re-master failed: ${resp.status}`);
            const data = await resp.json();

            // Create a new song entry from the re-mastered file
            const outputPath = data.output_path as string;
            // Build audio URL for the node server
            const audioUrl = `/audio/${outputPath.split(/\/audio\/|\\audio\\|\\audio\//).pop() || ''}`;
            if (token) {
              const newSong = await songsApi.createSong({
                title: `${remasterSong.title} (Remastered)`,
                lyrics: remasterSong.lyrics,
                style: remasterSong.style,
                tags: remasterSong.tags,
                audioUrl,
                coverUrl: remasterSong.coverUrl,
                duration: remasterSong.duration,
                isPublic: false,
                generationParams: {
                  ...remasterSong.generationParams,
                  originalAudioUrl: originalUrl,
                  masteringParams: params,
                },
              } as any, token);
              showToast(`Re-mastered! New track: ${newSong.song.title}`, 'success');
            } else {
              showToast(`Re-mastered! Output: ${outputPath}`, 'success');
            }
            refreshSongsList();
            setShowRemasterConsole(false);
            setRemasterSong(null);
          } catch (err) {
            console.error('[Remaster] Failed:', err);
            showToast(`Re-master failed: ${(err as Error).message}`, 'error');
          }
        }}
      />
      <FullscreenVisualizer
        isOpen={showFullscreenVisualizer}
        onClose={() => setShowFullscreenVisualizer(false)}
        song={currentSong}
        isPlaying={isPlaying}
        currentTime={currentTime}
        duration={duration}
        onTogglePlay={togglePlay}
        onNext={playNext}
        onPrevious={playPrevious}
        onSeek={handleSeek}
      />

      <ABCompareModal
        isOpen={showABCompare}
        onClose={() => setShowABCompare(false)}
        trackA={abTrackA}
        trackB={abTrackB}
      />

      {/* Mobile Details Modal */}
      {showMobileDetails && selectedSong && (
        <div className="fixed inset-0 z-[60] flex justify-end xl:hidden">
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-in fade-in"
            onClick={() => setShowMobileDetails(false)}
          />
          <div className="relative w-full max-w-md h-full bg-zinc-50 dark:bg-suno-panel shadow-2xl animate-in slide-in-from-right duration-300 border-l border-white/10">
            <RightSidebar
              song={selectedSong}
              onClose={() => setShowMobileDetails(false)}
              onOpenVideo={() => selectedSong && openVideoGenerator(selectedSong)}
              onReuse={handleReuse}
              onSongUpdate={handleSongUpdate}
              onNavigateToProfile={handleNavigateToProfile}
              onNavigateToSong={handleNavigateToSong}
              isLiked={selectedSong ? likedSongIds.has(selectedSong.id) : false}
              onToggleLike={toggleLike}
              onDelete={handleDeleteSong}
              onPlay={playSong}
              isPlaying={isPlaying && currentSong?.id === selectedSong?.id}
              currentSong={currentSong}
              currentTime={currentTime}
              onFullscreenVisualizer={() => setShowFullscreenVisualizer(true)}
            />
          </div>
        </div>
      )}

      <ConfirmDialog
        isOpen={confirmDialog !== null}
        title={confirmDialog?.title ?? ''}
        message={confirmDialog?.message ?? ''}
        confirmLabel={confirmDialog?.confirmLabel}
        danger={confirmDialog?.danger}
        onConfirm={() => confirmDialog?.onConfirm()}
        onCancel={() => setConfirmDialog(null)}
      />

      {/* Debug Panel */}
      <DebugPanel isOpen={debugPanelOpen} onToggle={() => setDebugPanelOpen(prev => !prev)} />

      {/* Shutdown overlay */}
      {isShutdown && (
        <div className="fixed inset-0 z-[200] flex flex-col items-center justify-center bg-black text-white">
          <div className="text-6xl mb-6">👋</div>
          <h1 className="text-2xl font-bold mb-2">HOT-Step 9000 has shut down</h1>
          <p className="text-zinc-400">You may now close this browser tab.</p>
        </div>
      )}
    </div>
  );
}

export default function App() {
  return (
    <I18nProvider>
      <AudioAnalysisProvider>
        <AppErrorBoundary>
          <AppContent />
        </AppErrorBoundary>
      </AudioAnalysisProvider>
    </I18nProvider>
  );
}
