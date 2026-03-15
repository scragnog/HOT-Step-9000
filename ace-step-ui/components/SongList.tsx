import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { Song } from '../types';
import { Play, MoreHorizontal, Heart, ThumbsDown, ListPlus, Pause, Search, Filter, Check, Globe, Lock, Loader2, ThumbsUp, Share2, Video, Info, Clock, ChevronLeft, ChevronRight, Trash2, LayoutList, LayoutGrid, List, ArrowLeftRight, X, Download } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useI18n } from '../context/I18nContext';
import { SongDropdownMenu } from './SongDropdownMenu';
import { ShareModal } from './ShareModal';
import { AlbumCover } from './AlbumCover';
import { songsApi } from '../services/api';
import { LiveVisualizer } from './LiveVisualizer';
import { LyricsBar } from './LyricsBar';
import { SongCard } from './SongCard';
import { SongItemCompact } from './SongItemCompact';
import { MasteringToggle } from './MasteringToggle';
import { usePersistedState } from '../hooks/usePersistedState';

type ViewMode = 'list' | 'grid' | 'compact';

interface SongListProps {
    songs: Song[];
    currentSong: Song | null;
    selectedSong: Song | null;
    likedSongIds: Set<string>;
    isPlaying: boolean;
    referenceTracks?: { id: string; filename: string; audio_url: string; duration?: number | null; created_at?: string }[];
    onPlay: (song: Song) => void;
    onSelect: (song: Song) => void;
    onToggleLike: (songId: string) => void;
    onAddToPlaylist: (song: Song) => void;
    onOpenVideo?: (song: Song) => void;
    onShowDetails?: (song: Song) => void;
    onNavigateToProfile?: (username: string) => void;
    onReusePrompt?: (song: Song) => void;
    onDelete?: (song: Song) => void;
    onSongUpdate?: (updatedSong: Song) => void;
    onDeleteMany?: (songs: Song[], onSuccess?: () => void) => void;
    onBulkDownload?: (songs: Song[]) => void;
    onDeleteAll?: () => void;
    onUseAsReference?: (song: Song) => void;
    onCoverSong?: (song: Song) => void;
    onUpscaleToHQ?: (song: Song) => void;
    onUseUploadAsReference?: (track: { audio_url: string; filename: string }) => void;
    onCoverUpload?: (track: { audio_url: string; filename: string }) => void;
    onDownloadFormat?: (song: Song) => void;
    onOpenRemaster?: (song: Song) => void;
    onToggleMastering?: () => void;
    playingOriginal?: boolean;
    onDeleteUpload?: (trackId: string) => void;
    showVisualizerBg?: boolean;
    currentTime?: number;
    onSetAsTrackA?: (song: Song) => void;
    onSetAsTrackB?: (song: Song) => void;
    abTrackA?: Song | null;
    abTrackB?: Song | null;
    onABCompare?: () => void;
    onABPlay?: () => void;
    onABClear?: () => void;
    abActive?: 'A' | 'B' | null;
    onABToggle?: () => void;
    // Ablation diff pins
    devMode?: boolean;
    diffPinnedA?: Song | null;
    diffPinnedB?: Song | null;
    onPinDiffA?: (song: Song) => void;
    onPinDiffB?: (song: Song) => void;
    // Cancel a queued/running generation
    onCancelJob?: (song: Song) => void;
    onCancelAll?: () => void;
}

// ... existing code ...



// Define Filter Types
type FilterType = 'liked' | 'public' | 'private' | 'generating';

// Map model ID to short display name
const getModelDisplayName = (modelId?: string): string => {
    if (!modelId) return 'v1.5';

    const mapping: Record<string, string> = {
        'acestep-v15-base': '1.5B',
        'acestep-v15-sft': '1.5S',
        'acestep-v15-turbo-shift1': '1.5TS1',
        'acestep-v15-turbo-shift3': '1.5TS3',
        'acestep-v15-turbo-continuous': '1.5TC',
        'acestep-v15-turbo': '1.5T',
    };
    return mapping[modelId] || 'v1.5';
};

const createDragPreview = (element: HTMLElement) => {
    const clone = element.cloneNode(true) as HTMLElement;
    clone.style.width = `${element.offsetWidth}px`;
    clone.style.position = 'fixed';
    clone.style.top = '-1000px';
    clone.style.left = '-1000px';
    clone.style.pointerEvents = 'none';
    clone.style.opacity = '0.95';

    const badge = document.createElement('div');
    badge.textContent = '+';
    badge.style.position = 'absolute';
    badge.style.left = '8px';
    badge.style.bottom = '8px';
    badge.style.width = '24px';
    badge.style.height = '24px';
    badge.style.display = 'flex';
    badge.style.alignItems = 'center';
    badge.style.justifyContent = 'center';
    badge.style.borderRadius = '9999px';
    badge.style.background = '#22c55e';
    badge.style.color = 'white';
    badge.style.boxShadow = '0 6px 16px rgba(0,0,0,0.25)';
    badge.style.fontSize = '16px';
    badge.style.lineHeight = '1';
    clone.style.position = 'relative';
    clone.appendChild(badge);

    document.body.appendChild(clone);
    return clone;
};

export const SongList: React.FC<SongListProps> = ({
    songs,
    currentSong,
    selectedSong,
    likedSongIds,
    isPlaying,
    referenceTracks = [],
    onPlay,
    onSelect,
    onToggleLike,
    onAddToPlaylist,
    onOpenVideo,
    onShowDetails,
    onNavigateToProfile,
    onReusePrompt,
    onDelete,
    onSongUpdate,
    onDeleteMany,
    onBulkDownload,
    onDeleteAll,
    onUseAsReference,
    onCoverSong,
    onUpscaleToHQ,
    onUseUploadAsReference,
    onCoverUpload,
    onDownloadFormat,
    onOpenRemaster,
    onToggleMastering,
    playingOriginal,
    onDeleteUpload,
    showVisualizerBg,
    currentTime,
    onSetAsTrackA,
    onSetAsTrackB,
    abTrackA,
    abTrackB,
    onABCompare,
    onABPlay,
    onABClear,
    abActive,
    onABToggle,
    devMode,
    diffPinnedA,
    diffPinnedB,
    onPinDiffA,
    onPinDiffB,
    onCancelJob,
    onCancelAll,
}) => {
    const { user } = useAuth();
    const { t } = useI18n();
    const [searchQuery, setSearchQuery] = useState('');
    const [activeFilters, setActiveFilters] = useState<Set<FilterType>>(new Set());
    const [isFilterOpen, setIsFilterOpen] = useState(false);
    const [isSelecting, setIsSelecting] = useState(false);
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const filterRef = useRef<HTMLDivElement>(null);
    const [pageSize, setPageSize] = useState(30);
    const [currentPage, setCurrentPage] = useState(1);
    const [viewMode, setViewMode] = usePersistedState<ViewMode>('ace-songListViewMode', 'list');

    const FILTERS: { id: FilterType; label: string; icon: React.ReactNode }[] = [
        { id: 'liked', label: t('liked'), icon: <ThumbsUp size={16} /> },
        { id: 'public', label: t('public'), icon: <Globe size={16} /> },
        { id: 'private', label: t('private'), icon: <Lock size={16} /> },
        { id: 'generating', label: t('generatingStatus'), icon: <Loader2 size={16} /> }
    ];

    // Close filter dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (filterRef.current && !filterRef.current.contains(event.target as Node)) {
                setIsFilterOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    useEffect(() => {
        setSelectedIds(prev => {
            if (prev.size === 0) return prev;
            const validIds = new Set(songs.map(song => song.id));
            const next = new Set<string>();
            prev.forEach(id => {
                if (validIds.has(id)) next.add(id);
            });
            return next;
        });
    }, [songs]);

    const toggleFilter = (filterId: FilterType) => {
        setActiveFilters(prev => {
            const newFilters = new Set(prev);
            if (newFilters.has(filterId)) {
                newFilters.delete(filterId);
            } else {
                newFilters.add(filterId);
            }
            return newFilters;
        });
    };

    const filteredSongs = useMemo(() => {
        return songs.filter(song => {
            // 1. Search Logic
            const matchesSearch =
                song.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                song.style.toLowerCase().includes(searchQuery.toLowerCase()) ||
                song.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));

            if (!matchesSearch) return false;

            // 2. Filter Logic
            if (activeFilters.size === 0) return true;

            if (activeFilters.has('liked') && !likedSongIds.has(song.id)) return false;
            if (activeFilters.has('public') && !song.isPublic) return false;
            if (activeFilters.has('private') && song.isPublic) return false;
            if (activeFilters.has('generating') && !song.isGenerating) return false;

            return true;
        });
    }, [songs, searchQuery, activeFilters, likedSongIds]);

    const filteredUploads = useMemo(() => {
        if (activeFilters.size > 0) return [];
        if (!referenceTracks.length) return [];
        return referenceTracks.filter(track => {
            const title = track.filename.replace(/\.[^/.]+$/, '');
            return title.toLowerCase().includes(searchQuery.toLowerCase());
        });
    }, [referenceTracks, searchQuery, activeFilters]);

    const listItems = useMemo(() => {
        const songItems = filteredSongs.map(song => ({
            type: 'song' as const,
            id: song.id,
            createdAt: song.createdAt,
            song
        }));
        const uploadItems = filteredUploads.map(track => ({
            type: 'upload' as const,
            id: track.id,
            createdAt: new Date(track.created_at || Date.now()),
            track
        }));
        // Always sort newest first by creation date
        return [...songItems, ...uploadItems].sort((a, b) => {
            const dateA = a.createdAt instanceof Date ? a.createdAt.getTime() : new Date(a.createdAt).getTime();
            const dateB = b.createdAt instanceof Date ? b.createdAt.getTime() : new Date(b.createdAt).getTime();
            return dateB - dateA;
        });
    }, [filteredSongs, filteredUploads]);

    // Reset to page 1 when list changes
    useEffect(() => {
        setCurrentPage(1);
    }, [searchQuery, activeFilters, songs.length]);

    // Pagination
    const totalItems = listItems.length;
    const totalPages = pageSize === 0 ? 1 : Math.max(1, Math.ceil(totalItems / pageSize));
    const safePage = Math.min(currentPage, totalPages);

    const paginatedItems = useMemo(() => {
        if (pageSize === 0) return listItems; // "All"
        const start = (safePage - 1) * pageSize;
        return listItems.slice(start, start + pageSize);
    }, [listItems, safePage, pageSize]);

    const selectableSongs = useMemo(
        () => filteredSongs,
        [filteredSongs]
    );

    const allSelected = selectableSongs.length > 0 && selectableSongs.every(song => selectedIds.has(song.id));
    const selectedSongs = selectableSongs.filter(song => selectedIds.has(song.id));

    return (
        <div className="relative flex-1 flex flex-col h-full bg-white dark:bg-black transition-colors duration-300">
            {/* HOT-Step 9000 Branded Background — always visible unless visualizer is playing */}
            {!(showVisualizerBg && isPlaying) && (
                <div 
                    className="absolute inset-0 pointer-events-none opacity-[0.04] dark:opacity-[0.08] dark:invert-0 invert transition-opacity duration-700 bg-center bg-no-repeat bg-contain"
                    style={{ backgroundImage: 'url(/hotstep-bg.webp)' }}
                />
            )}
            
            <div className="relative flex-1 overflow-y-auto custom-scrollbar p-6 pb-32">
                {/* Background Visualizer — dimmed, behind content */}
                {showVisualizerBg && isPlaying && (
                    <div className="absolute inset-0 z-0 pointer-events-none">
                        <LiveVisualizer
                            isPlaying={true}
                            className="w-full h-full"
                            dimmed={true}
                            showControls={false}
                            instanceId="songlist"
                        />
                    </div>
                )}
                <div className="relative z-[1] max-w-5xl mx-auto w-full"> {/* Container constraint */}

                    {/* Header */}
                    <div className="flex flex-col gap-6 mb-8">
                        <div className="flex items-center gap-2 text-sm text-zinc-500 dark:text-zinc-400">
                            <span className="hover:text-black dark:hover:text-white cursor-pointer transition-colors">{t('workspaces')}</span>
                            <span className="text-zinc-400 dark:text-zinc-600">›</span>
                            <span className="text-zinc-900 dark:text-white font-medium">{t('myWorkspace')}</span>
                        </div>

                        <div className="flex items-center gap-3">
                            <div className="relative group flex-1">
                                <input
                                    type="text"
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    placeholder={t('searchYourSongs')}
                                    className="w-full bg-zinc-100 dark:bg-[#121214] border border-zinc-200 dark:border-white/10 rounded-lg pl-10 pr-4 py-2.5 text-sm text-zinc-900 dark:text-white focus:outline-none focus:border-zinc-400 dark:focus:border-white/20 placeholder-zinc-500 dark:placeholder-zinc-600 transition-colors"
                                />
                                <Search className="w-4 h-4 text-zinc-500 absolute left-3 top-3 group-focus-within:text-black dark:group-focus-within:text-white transition-colors" />
                            </div>

                            <div className="relative" ref={filterRef}>
                                <button
                                    onClick={() => setIsFilterOpen(!isFilterOpen)}
                                    className={`
                        border text-xs font-bold px-4 py-2.5 rounded-lg flex items-center gap-2 transition-all select-none
                        ${isFilterOpen || activeFilters.size > 0
                                            ? 'bg-zinc-900 dark:bg-white text-white dark:text-black border-transparent'
                                            : 'bg-zinc-100 dark:bg-[#121214] hover:bg-zinc-200 dark:hover:bg-white/5 border-zinc-200 dark:border-white/10 text-zinc-700 dark:text-white'
                                        }
                    `}
                                >
                                    <Filter size={14} fill={activeFilters.size > 0 ? "currentColor" : "none"} />
                                    <span>{t('filters')} {activeFilters.size > 0 && `(${activeFilters.size})`}</span>
                                </button>

                                {/* Filter Dropdown */}
                                {isFilterOpen && (
                                    <div className="absolute right-0 top-full mt-2 w-56 bg-white dark:bg-[#18181b] border border-zinc-200 dark:border-white/10 rounded-xl shadow-2xl overflow-hidden py-1 z-50 animate-in fade-in zoom-in-95 duration-100 origin-top-right">
                                        <div className="px-3 py-2 text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
                                            {t('refineBy')}
                                        </div>
                                        {FILTERS.map(filter => (
                                            <button
                                                key={filter.id}
                                                onClick={() => toggleFilter(filter.id)}
                                                className="w-full text-left px-4 py-2.5 flex items-center justify-between hover:bg-zinc-100 dark:hover:bg-white/5 transition-colors group"
                                            >
                                                <div className="flex items-center gap-3 text-sm font-medium text-zinc-700 dark:text-zinc-300 group-hover:text-black dark:group-hover:text-white">
                                                    <span className="text-zinc-400 dark:text-zinc-500 group-hover:text-zinc-600 dark:group-hover:text-zinc-300 transition-colors">
                                                        {filter.icon}
                                                    </span>
                                                    {filter.label}
                                                </div>
                                                <div className={`
                                     w-4 h-4 rounded border flex items-center justify-center transition-all
                                     ${activeFilters.has(filter.id)
                                                        ? 'bg-pink-600 border-pink-600'
                                                        : 'border-zinc-300 dark:border-zinc-600 group-hover:border-zinc-400 dark:group-hover:border-zinc-500'
                                                    }
                                 `}>
                                                    {activeFilters.has(filter.id) && <Check size={10} className="text-white" strokeWidth={4} />}
                                                </div>
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>

                            <button
                                onClick={() => {
                                    setIsSelecting(prev => !prev);
                                    setSelectedIds(new Set());
                                }}
                                className={`border text-xs font-bold px-4 py-2.5 rounded-lg flex items-center gap-2 transition-all select-none ${isSelecting
                                    ? 'bg-zinc-900 dark:bg-white text-white dark:text-black border-transparent'
                                    : 'bg-zinc-100 dark:bg-[#121214] hover:bg-zinc-200 dark:hover:bg-white/5 border-zinc-200 dark:border-white/10 text-zinc-700 dark:text-white'
                                    }`}
                            >
                                {t('select')}
                            </button>

                            {onDeleteAll && songs.length > 0 && (
                                <button
                                    onClick={onDeleteAll}
                                    title={t('deleteAllTracks')}
                                    className="border border-zinc-200 dark:border-white/10 bg-zinc-100 dark:bg-[#121214] hover:bg-red-50 dark:hover:bg-red-500/10 hover:border-red-300 dark:hover:border-red-500/30 text-zinc-500 dark:text-zinc-400 hover:text-red-600 dark:hover:text-red-400 p-2.5 rounded-lg transition-all"
                                >
                                    <Trash2 size={14} />
                                </button>
                            )}

                            {/* Clear stuck / generating jobs */}
                            {onCancelAll && songs.some(s => s.isGenerating) && (
                                <button
                                    onClick={onCancelAll}
                                    title="Clear all queued/running jobs"
                                    className="border border-amber-300 dark:border-amber-500/40 bg-amber-50 dark:bg-amber-500/10 hover:bg-amber-100 dark:hover:bg-amber-500/20 text-amber-600 dark:text-amber-400 text-xs font-semibold px-3 py-2.5 rounded-lg flex items-center gap-1.5 transition-all"
                                >
                                    <X size={12} /> Clear Queue
                                </button>
                            )}

                            {/* View mode toggle */}
                            <div className="flex items-center bg-zinc-100 dark:bg-[#121214] border border-zinc-200 dark:border-white/10 rounded-lg overflow-hidden">
                                <button
                                    onClick={() => setViewMode('list')}
                                    title={t('viewList') || 'List view'}
                                    className={`p-2.5 transition-colors ${viewMode === 'list' ? 'bg-zinc-900 dark:bg-white text-white dark:text-black' : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-700 dark:hover:text-white'}`}
                                >
                                    <LayoutList size={14} />
                                </button>
                                <button
                                    onClick={() => setViewMode('grid')}
                                    title={t('viewGrid') || 'Grid view'}
                                    className={`p-2.5 transition-colors ${viewMode === 'grid' ? 'bg-zinc-900 dark:bg-white text-white dark:text-black' : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-700 dark:hover:text-white'}`}
                                >
                                    <LayoutGrid size={14} />
                                </button>
                                <button
                                    onClick={() => setViewMode('compact')}
                                    title={t('viewCompact') || 'Compact view'}
                                    className={`p-2.5 transition-colors ${viewMode === 'compact' ? 'bg-zinc-900 dark:bg-white text-white dark:text-black' : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-700 dark:hover:text-white'}`}
                                >
                                    <List size={14} />
                                </button>
                            </div>
                        </div>

                        {isSelecting && (
                            <div className="flex items-center justify-between gap-3 rounded-xl border border-zinc-200 dark:border-white/10 bg-zinc-50 dark:bg-white/5 px-4 py-3">
                                <div className="text-sm text-zinc-600 dark:text-zinc-300">
                                    {t('selectedCount').replace('{count}', String(selectedSongs.length))}
                                </div>
                                <div className="flex items-center gap-2">
                                    <button
                                        onClick={() => {
                                            const next = new Set<string>();
                                            if (!allSelected) {
                                                selectableSongs.forEach(song => next.add(song.id));
                                            }
                                            setSelectedIds(next);
                                        }}
                                        className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-zinc-200 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:border-zinc-300 dark:hover:border-white/20"
                                    >
                                        {allSelected ? t('clearAll') : t('selectAll')}
                                    </button>
                                    <button
                                        onClick={() => {
                                            if (!selectedSongs.length || !onDeleteMany) return;
                                            onDeleteMany(selectedSongs, () => {
                                                setIsSelecting(false);
                                            });
                                        }}
                                        className={`px-3 py-1.5 rounded-lg text-xs font-semibold border ${selectedSongs.length
                                            ? 'border-red-500 text-red-600 hover:bg-red-50 dark:hover:bg-red-500/10'
                                            : 'border-zinc-200 dark:border-white/10 text-zinc-400 cursor-not-allowed'
                                            }`}
                                        disabled={!selectedSongs.length}
                                    >
                                        {t('delete')}
                                    </button>
                                    <button
                                        onClick={() => {
                                            if (!selectedSongs.length || !onBulkDownload) return;
                                            onBulkDownload(selectedSongs);
                                        }}
                                        className={`px-3 py-1.5 rounded-lg text-xs font-semibold border flex items-center gap-1.5 ${selectedSongs.length
                                            ? 'border-pink-500 text-pink-600 hover:bg-pink-50 dark:hover:bg-pink-500/10'
                                            : 'border-zinc-200 dark:border-white/10 text-zinc-400 cursor-not-allowed'
                                            }`}
                                        disabled={!selectedSongs.length}
                                    >
                                        <Download size={12} />
                                        {t('download') || 'Download'}
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Song items — view mode dependent */}
                    <div className={viewMode === 'grid' ? 'grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-2 2xl:grid-cols-3 gap-3' : viewMode === 'compact' ? 'space-y-0.5' : 'space-y-2'}>
                        {listItems.length === 0 ? (
                            <div className={`flex flex-col items-center justify-center h-64 text-zinc-500 space-y-4 border border-dashed border-zinc-200 dark:border-white/5 rounded-2xl bg-zinc-50 dark:bg-white/[0.02] ${viewMode === 'grid' ? 'col-span-full' : ''}`}>
                                <div className="w-16 h-16 rounded-full bg-zinc-100 dark:bg-white/5 flex items-center justify-center">
                                    <Filter size={32} />
                                </div>
                                <p className="font-medium">{t('noSongsMatchFilters')}</p>
                                <button
                                    onClick={() => { setActiveFilters(new Set()); setSearchQuery(''); }}
                                    className="text-pink-600 dark:text-pink-500 text-sm font-bold hover:underline"
                                >
                                    {t('clearFilters')}
                                </button>
                            </div>
                        ) : (
                            paginatedItems.map((item) => {
                                if (item.type === 'upload') {
                                    return (
                                        <UploadItem
                                            key={`upload_${item.id}`}
                                            track={item.track}
                                            viewMode={viewMode}
                                            onPlay={(audioUrl, title) => {
                                                onPlay({
                                                    id: `upload_${item.id}`,
                                                    title,
                                                    lyrics: '',
                                                    style: 'Upload',
                                                    coverUrl: '',
                                                    duration: '0:00',
                                                    createdAt: item.createdAt,
                                                    tags: [],
                                                    audioUrl,
                                                    isPublic: false,
                                                } as Song);
                                            }}
                                            onUseAsReference={() => onUseUploadAsReference?.(item.track)}
                                            onCoverSong={() => onCoverUpload?.(item.track)}
                                            onDelete={onDeleteUpload ? () => onDeleteUpload(item.id) : undefined}
                                        />
                                    );
                                }

                                const songProps = {
                                    key: item.id,
                                    song: item.song,
                                    isCurrent: currentSong?.id === item.song.id,
                                    isSelected: selectedSong?.id === item.song.id,
                                    isSelectionMode: isSelecting,
                                    isChecked: selectedIds.has(item.song.id),
                                    isLiked: likedSongIds.has(item.song.id),
                                    isPlaying: isPlaying,
                                    isOwner: user?.id === item.song.userId,
                                    onPlay: () => onPlay(item.song),
                                    onSelect: () => onSelect(item.song),
                                    onToggleSelect: () => {
                                        setSelectedIds(prev => {
                                            const next = new Set(prev);
                                            if (next.has(item.song.id)) next.delete(item.song.id);
                                            else next.add(item.song.id);
                                            return next;
                                        });
                                    },
                                    onToggleLike: () => onToggleLike(item.song.id),
                                    onAddToPlaylist: () => onAddToPlaylist(item.song),
                                    onOpenVideo: () => onOpenVideo && onOpenVideo(item.song),
                                    onShowDetails: () => onShowDetails && onShowDetails(item.song),
                                    onNavigateToProfile: onNavigateToProfile,
                                    onReusePrompt: () => onReusePrompt?.(item.song),
                                    onDelete: () => onDelete?.(item.song),
                                    onSongUpdate: onSongUpdate,
                                    onUseAsReference: () => onUseAsReference?.(item.song),
                                    onCoverSong: () => onCoverSong?.(item.song),
                                    onUpscaleToHQ: onUpscaleToHQ ? () => onUpscaleToHQ(item.song) : undefined,
                                    onDownloadFormat: () => onDownloadFormat?.(item.song),
                                    onOpenRemaster: () => onOpenRemaster?.(item.song),
                                    onToggleMastering: currentSong?.id === item.song.id && item.song.generationParams?.originalAudioUrl ? onToggleMastering : undefined,
                                    playingOriginal: currentSong?.id === item.song.id ? playingOriginal : false,
                                    onSetAsTrackA: () => onSetAsTrackA?.(item.song),
                                    onSetAsTrackB: () => onSetAsTrackB?.(item.song),
                                    isTrackA: abTrackA?.id === item.song.id,
                                    isTrackB: abTrackB?.id === item.song.id,
                                    // Ablation diff pins
                                    devMode,
                                    isDiffPinnedA: diffPinnedA?.id === item.song.id,
                                    isDiffPinnedB: diffPinnedB?.id === item.song.id,
                                    onPinDiffA: () => onPinDiffA?.(item.song),
                                    onPinDiffB: () => onPinDiffB?.(item.song),
                                    // Cancel generating songs
                                    onCancel: item.song.isGenerating && onCancelJob ? () => onCancelJob(item.song) : undefined,
                                };

                                if (viewMode === 'grid') return <SongCard {...songProps} />;
                                if (viewMode === 'compact') return <SongItemCompact {...songProps} />;
                                return <SongItem {...songProps} />;
                            })
                        )}
                    </div>

                    {/* Pagination Controls */}
                    {totalItems > 0 && (
                        <div className="flex items-center justify-between mt-6 pt-4 border-t border-zinc-200 dark:border-white/10">
                            {/* Page size selector */}
                            <div className="flex items-center gap-1">
                                {[30, 60, 120, 0].map(size => (
                                    <button
                                        key={size}
                                        onClick={() => { setPageSize(size); setCurrentPage(1); }}
                                        className={`px-2.5 py-1 rounded-md text-xs font-semibold transition-all ${pageSize === size
                                            ? 'bg-zinc-900 dark:bg-white text-white dark:text-black'
                                            : 'text-zinc-500 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-white/5'
                                            }`}
                                    >
                                        {size === 0 ? t('all') || 'All' : size}
                                    </button>
                                ))}
                            </div>

                            {/* Page navigation */}
                            {pageSize > 0 && totalPages > 1 && (
                                <div className="flex items-center gap-2">
                                    <span className="text-xs text-zinc-500 dark:text-zinc-400 font-medium">
                                        {(safePage - 1) * pageSize + 1}–{Math.min(safePage * pageSize, totalItems)} of {totalItems}
                                    </span>
                                    <button
                                        onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                                        disabled={safePage <= 1}
                                        className="p-1.5 rounded-md hover:bg-zinc-100 dark:hover:bg-white/5 text-zinc-500 dark:text-zinc-400 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                                    >
                                        <ChevronLeft size={16} />
                                    </button>
                                    <button
                                        onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                                        disabled={safePage >= totalPages}
                                        className="p-1.5 rounded-md hover:bg-zinc-100 dark:hover:bg-white/5 text-zinc-500 dark:text-zinc-400 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                                    >
                                        <ChevronRight size={16} />
                                    </button>
                                </div>
                            )}
                        </div>
                    )}
                </div> {/* End container */}
            </div> {/* End scroll area */}

            {/* Lyrics Bar — shown when playing and NOT in comparison mode, hidden for cover/repaint */}
            {!(abTrackA && abTrackB) && isPlaying && currentSong && currentSong.generationParams?.taskType !== 'cover' && currentSong.generationParams?.taskType !== 'repaint' && (
                <LyricsBar
                    audioUrl={currentSong.audioUrl}
                    currentTime={currentTime ?? 0}
                    isPlaying={isPlaying}
                />
            )}

            {/* A/B Comparison Bar — fixed footer below scroll area */}
            {abTrackA && abTrackB && (
                <div className="flex-shrink-0 bg-zinc-100 dark:bg-zinc-900 border-t border-zinc-200 dark:border-white/10 px-6 py-2.5 z-30">
                    <div className="flex items-center gap-3">
                        {/* Track labels */}
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold border ${abActive === 'A' ? 'bg-blue-500/20 text-blue-600 dark:text-blue-400 border-blue-500/30 ring-1 ring-blue-500/20' : 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20'}`}>
                                A
                            </span>
                            <span className="text-xs text-zinc-700 dark:text-zinc-300 truncate">{abTrackA.title}</span>
                            <span className="text-zinc-400 text-xs">vs</span>
                            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold border ${abActive === 'B' ? 'bg-orange-500/20 text-orange-600 dark:text-orange-400 border-orange-500/30 ring-1 ring-orange-500/20' : 'bg-orange-500/10 text-orange-600 dark:text-orange-400 border-orange-500/20'}`}>
                                B
                            </span>
                            <span className="text-xs text-zinc-700 dark:text-zinc-300 truncate">{abTrackB.title}</span>
                        </div>

                        {/* Controls */}
                        <div className="flex items-center gap-2 flex-shrink-0">
                            {/* A/B Toggle */}
                            {abActive && (
                                <div className="flex items-center rounded-lg border border-zinc-200 dark:border-white/10 overflow-hidden">
                                    <button
                                        onClick={onABToggle}
                                        className={`px-3 py-1.5 text-xs font-bold transition-all ${abActive === 'A' ? 'bg-blue-500 text-white' : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500 hover:text-blue-600'}`}
                                    >
                                        A
                                    </button>
                                    <button
                                        onClick={onABToggle}
                                        className={`px-3 py-1.5 text-xs font-bold transition-all ${abActive === 'B' ? 'bg-orange-500 text-white' : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500 hover:text-orange-600'}`}
                                    >
                                        B
                                    </button>
                                </div>
                            )}

                            {/* Play Comparison */}
                            {!abActive && (
                                <button
                                    onClick={onABPlay}
                                    className="flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-bold bg-gradient-to-r from-blue-500 to-orange-500 text-white hover:from-blue-600 hover:to-orange-600 shadow-lg shadow-blue-500/20 transition-all"
                                >
                                    <Play size={14} fill="currentColor" />
                                    Play Comparison
                                </button>
                            )}

                            {/* Compare Params */}
                            <button
                                onClick={onABCompare}
                                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
                                title="Compare parameters"
                            >
                                <ArrowLeftRight size={14} />
                                Diff
                            </button>

                            {/* Clear */}
                            <button
                                onClick={onABClear}
                                className="p-1.5 rounded-lg text-zinc-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                                title="Clear comparison"
                            >
                                <X size={16} />
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

interface SongItemProps {
    song: Song;
    isCurrent: boolean;
    isSelected: boolean;
    isSelectionMode: boolean;
    isChecked: boolean;
    isLiked: boolean;
    isPlaying: boolean;
    isOwner: boolean;
    onPlay: () => void;
    onSelect: () => void;
    onToggleSelect: () => void;
    onToggleLike: () => void;
    onAddToPlaylist: () => void;
    onOpenVideo?: () => void;
    onShowDetails?: () => void;
    onNavigateToProfile?: (username: string) => void;
    onReusePrompt?: () => void;
    onDelete?: () => void;
    onSongUpdate?: (updatedSong: Song) => void;
    onUseAsReference?: () => void;
    onCoverSong?: () => void;
    onUpscaleToHQ?: () => void;
    onDownloadFormat?: () => void;
    onOpenRemaster?: () => void;
    onToggleMastering?: () => void;
    playingOriginal?: boolean;
    onSetAsTrackA?: () => void;
    onSetAsTrackB?: () => void;
    isTrackA?: boolean;
    isTrackB?: boolean;
    onCancel?: () => void;
}

const SongItem: React.FC<SongItemProps> = ({
    song,
    isCurrent,
    isSelected,
    isSelectionMode,
    isChecked,
    isLiked,
    isPlaying,
    isOwner,
    onPlay,
    onSelect,
    onToggleSelect,
    onToggleLike,
    onAddToPlaylist,
    onOpenVideo,
    onShowDetails,
    onNavigateToProfile,
    onReusePrompt,
    onDelete,
    onSongUpdate,
    onUseAsReference,
    onCoverSong,
    onUpscaleToHQ,
    onDownloadFormat,
    onOpenRemaster,
    onToggleMastering,
    playingOriginal,
    onSetAsTrackA,
    onSetAsTrackB,
    isTrackA,
    isTrackB,
    onCancel
}) => {
    const { token } = useAuth();
    const { t } = useI18n();
    const [showDropdown, setShowDropdown] = useState(false);
    const [shareModalOpen, setShareModalOpen] = useState(false);
    const [imageError, setImageError] = useState(false);
    const [isEditingTitle, setIsEditingTitle] = useState(false);
    const [editedTitle, setEditedTitle] = useState(song.title);
    const titleInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (isEditingTitle && titleInputRef.current) {
            titleInputRef.current.focus();
            titleInputRef.current.select();
        }
    }, [isEditingTitle]);

    const handleSaveTitle = async () => {
        if (!token || !isOwner || !editedTitle.trim() || editedTitle === song.title) {
            setIsEditingTitle(false);
            setEditedTitle(song.title);
            return;
        }

        try {
            const response = await songsApi.updateSong(song.id, { title: editedTitle.trim() }, token);
            setIsEditingTitle(false);
            // 更新父组件的歌曲列表
            if (onSongUpdate && response.song) {
                onSongUpdate(response.song);
            }
        } catch (error) {
            console.error('Failed to update title:', error);
            setEditedTitle(song.title);
            setIsEditingTitle(false);
        }
    };

    const handleTitleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
            handleSaveTitle();
        } else if (e.key === 'Escape') {
            setEditedTitle(song.title);
            setIsEditingTitle(false);
        }
    };

    return (
        <>
            <div
                onClick={onSelect}
                draggable={Boolean(song.audioUrl) && !song.isGenerating}
                onDragStart={(e) => {
                    if (!song.audioUrl || song.isGenerating) return;
                    e.dataTransfer.effectAllowed = 'copy';
                    e.dataTransfer.setData('application/x-ace-audio', JSON.stringify({
                        url: song.audioUrl,
                        title: song.title || 'Untitled',
                        source: 'song',
                    }));
                    const preview = createDragPreview(e.currentTarget);
                    const rect = e.currentTarget.getBoundingClientRect();
                    const offsetX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
                    const offsetY = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
                    e.dataTransfer.setDragImage(preview, offsetX, offsetY);
                    setTimeout(() => {
                        try {
                            preview.remove();
                        } catch {
                            // ignore
                        }
                    }, 0);
                }}
                className={`group flex items-center gap-4 p-2 rounded-lg hover:bg-zinc-100 dark:hover:bg-[#18181b] transition-all cursor-pointer border ${isSelected ? 'bg-zinc-100 dark:bg-[#18181b] border-zinc-200 dark:border-white/10' : 'border-transparent bg-transparent'} ${song.audioUrl && !song.isGenerating ? 'cursor-grab active:cursor-grabbing' : ''}`}
            >
                {isSelectionMode && (
                    <button
                        type="button"
                        onClick={(e) => {
                            e.stopPropagation();
                            onToggleSelect();
                        }}
                        className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${isChecked
                            ? 'bg-pink-600 border-pink-600 text-white'
                            : 'border-zinc-300 dark:border-zinc-600 text-transparent hover:border-zinc-400 dark:hover:border-zinc-500'
                            }`}
                        aria-pressed={isChecked}
                    >
                        <Check size={12} strokeWidth={3} className={isChecked ? 'text-white' : 'text-transparent'} />
                    </button>
                )}

                {/* Cover Art - Reduced size */}
                <div className="relative w-16 h-16 flex-shrink-0 rounded-md bg-zinc-200 dark:bg-zinc-800 overflow-hidden shadow-sm group/image">
                    {/* Use gradient fallback if no coverUrl or image fails to load */}
                    {(!song.coverUrl || imageError) ? (
                        <AlbumCover seed={song.id || song.title} size="full" className={`w-full h-full ${song.isGenerating ? 'opacity-20 blur-sm' : 'opacity-100'}`} />
                    ) : (
                        <img
                            src={song.coverUrl}
                            alt={song.title}
                            className={`w-full h-full object-cover transition-opacity ${song.isGenerating ? 'opacity-20 blur-sm' : 'opacity-100'}`}
                            onError={() => setImageError(true)}
                        />
                    )}

                    {song.isGenerating ? (
                        <div className="absolute inset-0 bg-black/40 flex flex-col items-center justify-center gap-1">
                            {song.queuePosition ? (
                                /* Queue indicator */
                                <>
                                    <div className="w-8 h-8 rounded-full bg-amber-500/20 flex items-center justify-center">
                                        <Clock size={16} className="text-amber-400" />
                                    </div>
                                    <span className="text-[10px] font-medium text-amber-400">{t('queuePosition')}{song.queuePosition}</span>
                                </>
                            ) : (
                                /* Generating - Music Waveform Animation */
                                <div className="flex items-end gap-1 h-6">
                                    <div className="w-1 bg-pink-500 rounded-full music-bar-anim" style={{ animationDelay: '0.0s' }}></div>
                                    <div className="w-1 bg-pink-500 rounded-full music-bar-anim" style={{ animationDelay: '0.2s' }}></div>
                                    <div className="w-1 bg-pink-500 rounded-full music-bar-anim" style={{ animationDelay: '0.4s' }}></div>
                                    <div className="w-1 bg-pink-500 rounded-full music-bar-anim" style={{ animationDelay: '0.1s' }}></div>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div
                            className={`absolute inset-0 bg-black/40 flex items-center justify-center backdrop-blur-[1px] cursor-pointer transition-opacity duration-200 ${isCurrent ? 'opacity-100' : 'opacity-0 group-hover/image:opacity-100'}`}
                            onClick={(e) => {
                                e.stopPropagation();
                                onPlay();
                            }}
                        >
                            <div className="w-10 h-10 rounded-full bg-white flex items-center justify-center shadow-lg transform transition-transform hover:scale-105">
                                {isCurrent && isPlaying ? (
                                    <Pause fill="black" className="text-black w-5 h-5" />
                                ) : (
                                    <Play fill="black" className="text-black ml-1 w-5 h-5" />
                                )}
                            </div>
                        </div>
                    )}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0 flex flex-col justify-between py-1">
                    <div className="space-y-1">
                        <div className="flex items-center gap-2">
                            {isEditingTitle && isOwner ? (
                                <input
                                    ref={titleInputRef}
                                    type="text"
                                    value={editedTitle}
                                    onChange={(e) => setEditedTitle(e.target.value)}
                                    onBlur={handleSaveTitle}
                                    onKeyDown={handleTitleKeyDown}
                                    onClick={(e) => e.stopPropagation()}
                                    className="font-bold text-lg bg-zinc-100 dark:bg-zinc-800 px-2 py-0.5 rounded border border-pink-500 focus:outline-none text-zinc-900 dark:text-white min-w-0 flex-1"
                                />
                            ) : (
                                <h3
                                    className={`font-bold text-lg truncate ${isCurrent ? 'text-pink-600 dark:text-pink-500' : 'text-zinc-900 dark:text-white'} ${isOwner && !song.isGenerating ? 'cursor-pointer hover:underline' : ''}`}
                                    onClick={(e) => {
                                        if (isOwner && !song.isGenerating) {
                                            e.stopPropagation();
                                            setIsEditingTitle(true);
                                        }
                                    }}
                                >
                                    {song.title || (song.isGenerating ? (song.queuePosition ? "Queued..." : "Creating...") : "Untitled")}
                                </h3>
                            )}
                            <span className="inline-flex items-center justify-center text-[9px] font-bold text-white bg-gradient-to-r from-pink-500 to-purple-500 px-1.5 py-0.5 rounded-sm shadow-sm" title={`DiT model: ${song.ditModel || 'undefined'}`}>
                                {getModelDisplayName(song.ditModel)}
                            </span>
                            {song.isPublic === false && (
                                <Lock size={12} className="text-zinc-400 dark:text-zinc-500" />
                            )}
                        </div>
                        <div className="flex items-center gap-2">
                            <div
                                className="flex items-center gap-1.5 cursor-pointer hover:opacity-80 transition-opacity"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    if (song.creator && onNavigateToProfile) {
                                        onNavigateToProfile(song.creator);
                                    }
                                }}
                            >
                                <div className="w-4 h-4 rounded-full bg-purple-500 text-[8px] flex items-center justify-center font-bold text-white">
                                    {(song.creator?.[0] || 'U').toUpperCase()}
                                </div>
                                <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 transition-colors hover:underline">
                                    {song.creator || 'Unknown'}
                                </span>
                            </div>
                        </div>
                        <p className="text-xs text-zinc-500 dark:text-zinc-500 line-clamp-2 pt-1 font-medium max-w-2xl">
                            {song.style}
                        </p>
                        {song.isGenerating && (
                            <div className="pt-2 space-y-1">
                                <div className="h-1 rounded-full bg-zinc-200/70 dark:bg-white/10 overflow-hidden">
                                    <div
                                        className={`h-full bg-gradient-to-r from-pink-500 to-purple-600 transition-all ${song.progress === undefined ? 'opacity-40' : ''}`}
                                        style={{
                                            width: `${Math.min(
                                                100,
                                                Math.max(0, ((song.progress ?? 0) > 1 ? (song.progress ?? 0) / 100 : (song.progress ?? 0)) * 100)
                                            )}%`,
                                        }}
                                    />
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-[10px] font-medium text-zinc-400 dark:text-zinc-500">
                                        {song.queuePosition
                                            ? `Queue #${song.queuePosition}`
                                            : song.stage
                                                ? song.stage
                                                : 'Generating…'}
                                    </span>
                                    {song.progress !== undefined && !song.queuePosition && (
                                        <span className="text-[10px] font-bold text-pink-500 dark:text-pink-400">
                                            {Math.round((song.progress > 1 ? song.progress / 100 : song.progress) * 100)}%
                                        </span>
                                    )}
                                </div>
                                {onCancel && (
                                    <button
                                        type="button"
                                        onClick={(e) => { e.stopPropagation(); onCancel(); }}
                                        className="text-[10px] font-medium text-red-400 hover:text-red-500 transition-colors"
                                    >
                                        Cancel
                                    </button>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Actions Row - Hidden while generating */}
                    {!song.isGenerating && (
                        <div className="flex items-center gap-1 pt-2">
                            <button
                                className={`flex items-center gap-1 px-3 py-1.5 rounded-full hover:bg-white/5 transition-colors ${isLiked ? 'text-pink-600 dark:text-pink-500 bg-pink-100 dark:bg-pink-500/10' : 'text-zinc-400 hover:text-black dark:hover:text-white'}`}
                                onClick={(e) => { e.stopPropagation(); onToggleLike(); }}
                            >
                                <ThumbsUp size={16} fill={isLiked ? "currentColor" : "none"} />
                                {(song.likeCount || 0) > 0 && (
                                    <span className="text-xs font-bold">{song.likeCount}</span>
                                )}
                            </button>

                            <button
                                className="p-2 rounded-full hover:bg-zinc-200 dark:hover:bg-white/5 text-zinc-400 hover:text-black dark:hover:text-white transition-colors"
                                onClick={(e) => { e.stopPropagation(); }}
                            >
                                <ThumbsDown size={16} />
                            </button>

                            <button
                                className="p-2 rounded-full hover:bg-zinc-200 dark:hover:bg-white/5 text-zinc-400 hover:text-black dark:hover:text-white transition-colors"
                                onClick={(e) => { e.stopPropagation(); setShareModalOpen(true); }}
                                title="Share"
                            >
                                <Share2 size={16} />
                            </button>

                            <button
                                className="p-2 rounded-full hover:bg-zinc-200 dark:hover:bg-white/5 text-zinc-400 hover:text-black dark:hover:text-white transition-colors"
                                onClick={(e) => { e.stopPropagation(); if (onOpenVideo) onOpenVideo(); }}
                                title="Create Video"
                            >
                                <Video size={16} />
                            </button>

                            <button
                                className="p-2 rounded-full hover:bg-zinc-200 dark:hover:bg-white/5 text-zinc-400 hover:text-black dark:hover:text-white transition-colors ml-auto"
                                onClick={(e) => { e.stopPropagation(); onAddToPlaylist(); }}
                                title="Add to Playlist"
                            >
                                <ListPlus size={16} />
                            </button>

                            {/* Compare toggle button */}
                            <button
                                className={`relative p-2 rounded-full transition-colors ${isTrackA
                                    ? 'bg-blue-500/15 text-blue-600 dark:text-blue-400 ring-1 ring-blue-500/30'
                                    : isTrackB
                                        ? 'bg-orange-500/15 text-orange-600 dark:text-orange-400 ring-1 ring-orange-500/30'
                                        : 'hover:bg-zinc-200 dark:hover:bg-white/5 text-zinc-400 hover:text-black dark:hover:text-white'
                                    }`}
                                onClick={(e) => {
                                    e.stopPropagation();
                                    if (isTrackA) {
                                        // Already A → clear A
                                        onSetAsTrackA?.(); // App.tsx will toggle off
                                    } else if (isTrackB) {
                                        // Already B → clear B
                                        onSetAsTrackB?.(); // App.tsx will toggle off
                                    } else {
                                        // Not assigned → auto-pick A or B
                                        onSetAsTrackA?.(); // App.tsx will decide: if A exists, set as B instead
                                    }
                                }}
                                title={isTrackA ? 'Compare: Track A (click to clear)' : isTrackB ? 'Compare: Track B (click to clear)' : 'Add to comparison'}
                            >
                                <ArrowLeftRight size={16} />
                                {(isTrackA || isTrackB) && (
                                    <span className={`absolute -top-1 -right-1 text-[8px] font-black px-1 rounded-full ${isTrackA ? 'bg-blue-500 text-white' : 'bg-orange-500 text-white'}`}>
                                        {isTrackA ? 'A' : 'B'}
                                    </span>
                                )}
                            </button>

                            {/* Mastering toggle + remaster */}
                            {song.generationParams?.originalAudioUrl && (
                                onToggleMastering ? (
                                    <MasteringToggle
                                        isOriginal={!!playingOriginal}
                                        onToggle={onToggleMastering}
                                        size="sm"
                                    />
                                ) : (
                                    <span
                                        className="inline-flex items-center text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-gradient-to-r from-amber-500 to-orange-500 text-white shadow-sm"
                                        title="Mastered track"
                                    >
                                        M
                                    </span>
                                )
                            )}
                            {song.generationParams?.originalAudioUrl && onOpenRemaster && (
                                <button
                                    className="p-2 rounded-full hover:bg-zinc-200 dark:hover:bg-white/5 text-zinc-400 hover:text-amber-500 transition-colors"
                                    onClick={(e) => { e.stopPropagation(); onOpenRemaster(); }}
                                    title="Re-master this track"
                                >
                                    <span className="text-sm">🎛️</span>
                                </button>
                            )}

                            {/* Info Button - Visible only on small/medium screens where sidebar is hidden */}
                            <button
                                className="p-2 rounded-full hover:bg-zinc-200 dark:hover:bg-white/5 text-zinc-400 hover:text-black dark:hover:text-white transition-colors xl:hidden"
                                onClick={(e) => { e.stopPropagation(); if (onShowDetails) onShowDetails(); }}
                                title="Song Details"
                            >
                                <Info size={16} />
                            </button>

                            <div className="relative">
                                <button
                                    className="p-2 rounded-full hover:bg-zinc-200 dark:hover:bg-white/5 text-zinc-400 hover:text-black dark:hover:text-white transition-colors"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        setShowDropdown(!showDropdown);
                                    }}
                                >
                                    <MoreHorizontal size={16} />
                                </button>
                                <SongDropdownMenu
                                    song={song}
                                    isOpen={showDropdown}
                                    onClose={() => setShowDropdown(false)}
                                    isOwner={isOwner}
                                    onCreateVideo={() => onOpenVideo?.(song)}
                                    onReusePrompt={onReusePrompt ? () => onReusePrompt?.(song) : undefined}
                                    onAddToPlaylist={() => onAddToPlaylist?.(song)}
                                    onDelete={() => onDelete?.(song)}
                                    onShare={() => setShareModalOpen(true)}
                                    onUseAsReference={() => onUseAsReference?.()}
                                    onCoverSong={() => onCoverSong?.()}
                                    onUpscaleToHQ={onUpscaleToHQ}
                                    onDownloadFormat={() => onDownloadFormat?.()}
                                />
                            </div>
                        </div>
                    )}
                </div>

                {/* Duration & Date */}
                <div className="text-right self-start pt-1 flex-shrink-0">
                    <div className="text-xs font-mono text-zinc-500 dark:text-zinc-600">
                        {song.isGenerating ? (
                            <span className={song.queuePosition ? 'text-amber-500' : 'text-pink-500'}>
                                {song.queuePosition ? `#${song.queuePosition}` : 'Creating...'}
                            </span>
                        ) : song.duration}
                    </div>
                    {!song.isGenerating && song.createdAt && (
                        <div className="text-[10px] text-zinc-400 dark:text-zinc-600 mt-0.5">
                            {(() => {
                                const d = song.createdAt instanceof Date ? song.createdAt : new Date(song.createdAt);
                                if (isNaN(d.getTime())) return null;
                                const now = new Date();
                                const isToday = d.toDateString() === now.toDateString();
                                const yesterday = new Date(now); yesterday.setDate(now.getDate() - 1);
                                const isYesterday = d.toDateString() === yesterday.toDateString();
                                const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                                if (isToday) return time;
                                if (isYesterday) return `Yesterday ${time}`;
                                return `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${time}`;
                            })()}
                        </div>
                    )}
                </div>
            </div>

            <ShareModal
                isOpen={shareModalOpen}
                onClose={() => setShareModalOpen(false)}
                song={song}
            />
        </>
    );
};

const UploadItem: React.FC<{
    track: { id: string; filename: string; audio_url: string; duration?: number | null };
    viewMode?: ViewMode;
    onPlay: (audioUrl: string, title: string) => void;
    onUseAsReference?: () => void;
    onCoverSong?: () => void;
    onDelete?: () => void;
}> = ({ track, viewMode = 'list', onPlay, onUseAsReference, onCoverSong, onDelete }) => {
    const title = track.filename.replace(/\.[^/.]+$/, '');
    const duration = track.duration
        ? `${Math.floor(track.duration / 60)}:${String(Math.floor(track.duration % 60)).padStart(2, '0')}`
        : '--:--';
    const songData = {
        id: `upload_${track.id}`,
        title,
        lyrics: '',
        style: 'Upload',
        coverUrl: '',
        duration,
        createdAt: new Date(),
        tags: [],
        audioUrl: track.audio_url,
        isPublic: false,
    } as Song;
    const commonProps = {
        song: songData,
        isCurrent: false,
        isSelected: false,
        isSelectionMode: false,
        isChecked: false,
        isLiked: false,
        isPlaying: false,
        isOwner: true,
        onPlay: () => onPlay(track.audio_url, title),
        onSelect: () => onPlay(track.audio_url, title),
        onToggleSelect: () => undefined,
        onToggleLike: () => undefined,
        onAddToPlaylist: () => undefined,
        onOpenVideo: () => undefined,
        onShowDetails: () => undefined,
        onNavigateToProfile: () => undefined,
        onReusePrompt: undefined,
        onDelete: onDelete,
        onUseAsReference: onUseAsReference,
        onCoverSong: onCoverSong,
    } as const;

    if (viewMode === 'grid') return <SongCard {...commonProps} />;
    if (viewMode === 'compact') return <SongItemCompact {...commonProps} />;
    return <SongItem {...commonProps} />;
};
