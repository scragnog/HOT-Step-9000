import React, { useState } from 'react';
import { Song } from '../types';
import { Play, Pause, ThumbsUp, MoreHorizontal, Lock, Clock, X } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useI18n } from '../context/I18nContext';
import { SongDropdownMenu } from './SongDropdownMenu';
import { ShareModal } from './ShareModal';
import { AlbumCover } from './AlbumCover';
import { MasteringToggle } from './MasteringToggle';

// Map model ID to short display name
const getModelDisplayName = (modelId?: string): string => {
    if (!modelId) return 'v1.5';
    const mapping: Record<string, string> = {
        'acestep-v15-base': '1.5B',
        'acestep-v15-sft': '1.5S',
        'acestep-v15-merge-base-sft-0.5': '1.5M',
        'acestep-v15-turbo-shift3': '1.5TS3',
        'acestep-v15-turbo-continuous': '1.5TC',
        'acestep-v15-turbo': '1.5T',
    };
    return mapping[modelId] || 'v1.5';
};

export interface SongItemCompactProps {
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
    onCancel?: () => void;
}

export const SongItemCompact: React.FC<SongItemCompactProps> = ({
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
    onReusePrompt,
    onDelete,
    onUseAsReference,
    onCoverSong,
    onUpscaleToHQ,
    onDownloadFormat,
    onOpenRemaster,
    onToggleMastering,
    playingOriginal,
    onCancel
}) => {
    const { t } = useI18n();
    const [showDropdown, setShowDropdown] = useState(false);
    const [shareModalOpen, setShareModalOpen] = useState(false);
    const [imageError, setImageError] = useState(false);

    const formatDate = () => {
        if (!song.createdAt) return null;
        const d = song.createdAt instanceof Date ? song.createdAt : new Date(song.createdAt);
        if (isNaN(d.getTime())) return null;
        const now = new Date();
        const isToday = d.toDateString() === now.toDateString();
        const yesterday = new Date(now); yesterday.setDate(now.getDate() - 1);
        const isYesterday = d.toDateString() === yesterday.toDateString();
        const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        if (isToday) return time;
        if (isYesterday) return `Yest.`;
        return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
    };

    return (
        <>
            <div
                onClick={onSelect}
                className={`group flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-zinc-100 dark:hover:bg-[#18181b] transition-all cursor-pointer border ${isSelected
                    ? 'bg-zinc-100 dark:bg-[#18181b] border-zinc-200 dark:border-white/10'
                    : 'border-transparent'
                    }`}
            >
                {/* Selection checkbox */}
                {isSelectionMode && (
                    <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); onToggleSelect(); }}
                        className={`w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 transition-colors ${isChecked ? 'bg-pink-600 border-pink-600 text-white' : 'border-zinc-300 dark:border-zinc-600'
                            }`}
                    >
                        {isChecked && <span className="text-[8px] font-bold">✓</span>}
                    </button>
                )}

                {/* Small album art */}
                <div className="relative w-8 h-8 flex-shrink-0 rounded overflow-hidden bg-zinc-200 dark:bg-zinc-800 group/image">
                    {(!song.coverUrl || imageError) ? (
                        <AlbumCover seed={song.id || song.title} size="full" className={`w-full h-full ${song.isGenerating ? 'opacity-20' : ''}`} />
                    ) : (
                        <img
                            src={song.coverUrl}
                            alt={song.title}
                            className={`w-full h-full object-cover ${song.isGenerating ? 'opacity-20' : ''}`}
                            onError={() => setImageError(true)}
                        />
                    )}
                    {song.isGenerating ? (
                        <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                            {song.queuePosition ? (
                                <Clock size={12} className="text-amber-400" />
                            ) : (
                                <div className="flex items-end gap-px h-3">
                                    <div className="w-0.5 bg-pink-500 rounded-full music-bar-anim" style={{ animationDelay: '0s' }} />
                                    <div className="w-0.5 bg-pink-500 rounded-full music-bar-anim" style={{ animationDelay: '0.2s' }} />
                                    <div className="w-0.5 bg-pink-500 rounded-full music-bar-anim" style={{ animationDelay: '0.4s' }} />
                                </div>
                            )}
                        </div>
                    ) : (
                        <div
                            className={`absolute inset-0 bg-black/50 flex items-center justify-center cursor-pointer transition-opacity ${isCurrent ? 'opacity-100' : 'opacity-0 group-hover/image:opacity-100'
                                }`}
                            onClick={(e) => { e.stopPropagation(); onPlay(); }}
                        >
                            {isCurrent && isPlaying ? (
                                <Pause fill="white" className="text-white w-3 h-3" />
                            ) : (
                                <Play fill="white" className="text-white ml-0.5 w-3 h-3" />
                            )}
                        </div>
                    )}
                </div>

                {/* Title + creator */}
                <div className="flex-1 min-w-0 flex items-center gap-2">
                    <span className={`text-sm font-medium truncate ${isCurrent ? 'text-pink-600 dark:text-pink-500' : 'text-zinc-900 dark:text-white'}`}>
                        {song.title || (song.isGenerating ? 'Creating...' : 'Untitled')}
                    </span>
                    <span className="text-[11px] text-zinc-400 dark:text-zinc-600 truncate flex-shrink-0">
                        {song.creator || 'Unknown'}
                    </span>
                </div>

                {/* Model tag */}
                <span className="inline-flex items-center text-[8px] font-bold text-white bg-gradient-to-r from-pink-500 to-purple-500 px-1 py-0.5 rounded-sm flex-shrink-0">
                    {getModelDisplayName(song.ditModel)}
                </span>

                {/* Like button */}
                {!song.isGenerating && (
                    <button
                        className={`flex-shrink-0 p-1 rounded transition-colors ${isLiked ? 'text-pink-600 dark:text-pink-500' : 'text-zinc-400 opacity-0 group-hover:opacity-100 hover:text-pink-600'}`}
                        onClick={(e) => { e.stopPropagation(); onToggleLike(); }}
                    >
                        <ThumbsUp size={12} fill={isLiked ? 'currentColor' : 'none'} />
                    </button>
                )}

                {/* Private icon */}
                {song.isPublic === false && (
                    <Lock size={10} className="text-zinc-400 dark:text-zinc-600 flex-shrink-0" />
                )}

                {/* Duration */}
                <span className="text-[11px] font-mono text-zinc-500 dark:text-zinc-600 flex-shrink-0 w-10 text-right">
                    {song.isGenerating ? (
                        <span className={song.queuePosition ? 'text-amber-500' : 'text-pink-500'}>
                            {song.queuePosition ? `#${song.queuePosition}` : '...'}
                        </span>
                    ) : song.duration}
                </span>

                {/* Date */}
                <span className="text-[10px] text-zinc-400 dark:text-zinc-600 flex-shrink-0 w-12 text-right hidden lg:block">
                    {formatDate()}
                </span>

                {/* Stop button for generating songs */}
                {song.isGenerating && onCancel && (
                    <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); onCancel(); }}
                        title="Cancel generation"
                        className="flex-shrink-0 p-1 rounded text-red-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                    >
                        <X size={14} />
                    </button>
                )}

                {/* Mastering toggle + remaster */}
                {!song.isGenerating && song.generationParams?.originalAudioUrl && (
                    <>
                        {onToggleMastering ? (
                            <MasteringToggle
                                isOriginal={!!playingOriginal}
                                onToggle={onToggleMastering}
                                size="sm"
                            />
                        ) : (
                            <span
                                className="inline-flex items-center text-[8px] font-bold px-1 py-0.5 rounded-full bg-gradient-to-r from-amber-500 to-orange-500 text-white flex-shrink-0"
                                title="Mastered track"
                            >
                                M
                            </span>
                        )}
                        {onOpenRemaster && (
                            <button
                                className="flex-shrink-0 p-1 rounded text-zinc-400 hover:text-amber-500 transition-colors opacity-0 group-hover:opacity-100"
                                onClick={(e) => { e.stopPropagation(); onOpenRemaster(); }}
                                title="Re-master this track"
                            >
                                <span className="text-[10px]">🎛️</span>
                            </button>
                        )}
                    </>
                )}

                {/* Dropdown menu */}
                {!song.isGenerating && (
                    <div className="relative flex-shrink-0">
                        <button
                            className="p-1 rounded text-zinc-400 hover:text-zinc-700 dark:hover:text-white transition-colors opacity-0 group-hover:opacity-100"
                            onClick={(e) => { e.stopPropagation(); setShowDropdown(!showDropdown); }}
                        >
                            <MoreHorizontal size={14} />
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
                            onRemaster={onOpenRemaster}
                        />
                    </div>
                )}
            </div>

            <ShareModal
                isOpen={shareModalOpen}
                onClose={() => setShareModalOpen(false)}
                song={song}
            />
        </>
    );
};
