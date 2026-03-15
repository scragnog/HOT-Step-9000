import React, { useState, useEffect } from 'react';
import { Song } from '../types';
import { Heart, Share2, Play, Pause, MoreHorizontal, X, Copy, Wand2, MoreVertical, Download, Repeat, Video, Music, Link as LinkIcon, Sparkles, Globe, Lock, Trash2, Edit3, Layers, Activity, Zap, Settings2, Hash, Clock, Gauge, Radio, Disc, Volume2, SlidersHorizontal, Cpu, Terminal, Music2, Mic2, AudioWaveform, Box } from 'lucide-react';
import { songsApi } from '../services/api';
import { useAuth } from '../context/AuthContext';
import { useI18n } from '../context/I18nContext';
import { SongDropdownMenu } from './SongDropdownMenu';
import { ShareModal } from './ShareModal';
import { openStemSplitter } from './StemSplitterModal';
import { AlbumCover } from './AlbumCover';
import { LiveVisualizer } from './LiveVisualizer';
import { LyricsOverlay } from './LyricsOverlay';

interface RightSidebarProps {
    song: Song | null;
    onClose?: () => void;
    onOpenVideo?: () => void;
    onReuse?: (song: Song) => void;
    onSongUpdate?: (song: Song) => void;
    onNavigateToProfile?: (username: string) => void;
    onNavigateToSong?: (songId: string) => void;
    isLiked?: boolean;
    onToggleLike?: (songId: string) => void;
    onDelete?: (song: Song) => void;
    onAddToPlaylist?: (song: Song) => void;
    onPlay?: (song: Song) => void;
    isPlaying?: boolean;
    currentSong?: Song | null;
    currentTime?: number;
    onFullscreenVisualizer?: () => void;
}

export const RightSidebar: React.FC<RightSidebarProps> = ({ song, onClose, onOpenVideo, onReuse, onSongUpdate, onNavigateToProfile, onNavigateToSong, isLiked, onToggleLike, onDelete, onAddToPlaylist, onPlay, isPlaying, currentSong, currentTime, onFullscreenVisualizer }) => {
    const { token, user } = useAuth();
    const { t } = useI18n();
    const [showMenu, setShowMenu] = useState(false);
    const [isOwner, setIsOwner] = useState(false);
    const [tagsExpanded, setTagsExpanded] = useState(false);
    const [shareModalOpen, setShareModalOpen] = useState(false);
    const [copiedStyle, setCopiedStyle] = useState(false);
    const [copiedLyrics, setCopiedLyrics] = useState(false);
    const [isEditingTitle, setIsEditingTitle] = useState(false);
    const [titleDraft, setTitleDraft] = useState('');
    const [titleError, setTitleError] = useState<string | null>(null);
    const [isSavingTitle, setIsSavingTitle] = useState(false);
    const [genParamsExpanded, setGenParamsExpanded] = useState(false);

    useEffect(() => {
        if (song) {
            setIsOwner(user?.id === song.userId);
        }
    }, [song, user]);

    useEffect(() => {
        if (song) {
            setTitleDraft(song.title || '');
            setIsEditingTitle(false);
            setTitleError(null);
            setIsSavingTitle(false);
        }
    }, [song?.id]);

    const startTitleEdit = () => {
        if (!song || !isOwner) return;
        setTitleDraft(song.title || '');
        setTitleError(null);
        setIsEditingTitle(true);
    };

    const cancelTitleEdit = () => {
        if (!song) return;
        setTitleDraft(song.title || '');
        setTitleError(null);
        setIsEditingTitle(false);
    };

    const saveTitleEdit = async () => {
        if (!song) return;
        if (!token) {
            setTitleError('Please sign in to rename.');
            return;
        }
        const trimmed = titleDraft.trim();
        if (!trimmed) {
            setTitleError('Title cannot be empty.');
            return;
        }
        if (trimmed === song.title) {
            setIsEditingTitle(false);
            return;
        }
        setIsSavingTitle(true);
        setTitleError(null);
        try {
            await songsApi.updateSong(song.id, { title: trimmed }, token);
            onSongUpdate?.({ ...song, title: trimmed });
            setIsEditingTitle(false);
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Rename failed';
            setTitleError(message);
        } finally {
            setIsSavingTitle(false);
        }
    };

    const getSourceLabel = (url?: string) => {
        if (!url) return 'None';
        try {
            const parsed = new URL(url, window.location.origin);
            const name = decodeURIComponent(parsed.pathname.split('/').pop() || url);
            return name.replace(/\.[^/.]+$/, '') || name;
        } catch {
            const parts = url.split('/');
            const name = decodeURIComponent(parts[parts.length - 1] || url);
            return name.replace(/\.[^/.]+$/, '') || name;
        }
    };

    const openSource = (url?: string) => {
        if (!url) return;
        const resolved = url.startsWith('http') ? url : `${window.location.origin}${url}`;
        window.open(resolved, '_blank');
    };

    if (!song) return (
        <div className="w-full h-full bg-zinc-50 dark:bg-suno-panel border-l border-zinc-200 dark:border-white/5 flex items-center justify-center text-zinc-400 dark:text-zinc-500 text-sm transition-colors duration-300">
            <div className="flex flex-col items-center gap-2">
                <Music size={40} className="text-zinc-300 dark:text-zinc-700" />
                <p>{t('selectSongToView')}</p>
            </div>
        </div>
    );

    return (
        <div className="w-full h-full bg-zinc-50 dark:bg-suno-panel flex flex-col border-l border-zinc-200 dark:border-white/5 relative transition-colors duration-300">

            {/* Header */}
            <div className="h-14 flex items-center justify-between px-4 border-b border-zinc-200 dark:border-white/5 flex-shrink-0 bg-zinc-50/50 dark:bg-suno-panel/50 backdrop-blur-md z-10">
                <span className="font-semibold text-sm text-zinc-900 dark:text-white">{t('songDetails')}</span>
                <button
                    onClick={onClose}
                    className="p-1.5 hover:bg-zinc-200 dark:hover:bg-white/10 rounded-full text-zinc-500 dark:text-zinc-400 transition-colors"
                >
                    <X size={18} />
                </button>
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar">
                <div className="p-5 pb-24 lg:pb-32 space-y-6">

                    {/* Cover Art / Live Visualizer */}
                    <div
                        className="group relative aspect-square w-full rounded-xl overflow-hidden shadow-2xl bg-zinc-200 dark:bg-zinc-800 ring-1 ring-black/5 dark:ring-white/10 cursor-pointer"
                        onClick={() => onPlay?.(song)}
                    >
                        {/* Live Visualizer — shown when this song is playing */}
                        {isPlaying && currentSong?.id === song.id ? (
                            <div className="absolute inset-0 z-[1] animate-in fade-in duration-500">
                                <LiveVisualizer
                                    isPlaying={true}
                                    className="w-full h-full"
                                    showControls={true}
                                    onFullscreen={onFullscreenVisualizer}
                                    instanceId="artbox"
                                />
                                {/* Synced lyrics overlay — hidden for cover/repaint (timestamps don't match actual output) */}
                                {song.generationParams?.taskType !== 'cover' && song.generationParams?.taskType !== 'repaint' && (
                                    <LyricsOverlay
                                        audioUrl={song.audioUrl}
                                        currentTime={currentTime ?? 0}
                                        isPlaying={true}
                                    />
                                )}
                            </div>
                        ) : (
                            <>
                                {song.coverUrl ? (
                                    <img src={song.coverUrl} alt={song.title} className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105" onError={(e) => { e.currentTarget.style.display = 'none'; }} />
                                ) : null}
                                {!song.coverUrl && <AlbumCover seed={song.id || song.title} size="full" className="w-full h-full" />}
                            </>
                        )}

                        {/* Overlay Gradient */}
                        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-60 z-[2] pointer-events-none"></div>

                        {/* Play Button Overlay — only when not visualizing */}
                        {!(isPlaying && currentSong?.id === song.id) && (
                            <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-[3]">
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        onPlay?.(song);
                                    }}
                                    className="w-16 h-16 rounded-full bg-white/95 dark:bg-white text-black flex items-center justify-center shadow-2xl hover:scale-110 transition-transform"
                                >
                                    <Play size={28} fill="currentColor" className="ml-1" />
                                </button>
                            </div>
                        )}

                        <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between z-[3]">
                            <div className="flex items-center gap-2 text-white">
                                <Play size={16} fill="currentColor" />
                                <span className="text-xs font-bold font-mono">{song.viewCount || 0}</span>
                            </div>
                            <span className="text-[10px] font-bold text-black bg-white/90 px-1.5 py-0.5 rounded backdrop-blur-sm">
                                {song.duration}
                            </span>
                        </div>
                    </div>

                    {/* Title & Artist Block */}
                    <div className="space-y-3">
                        <div className="flex justify-between items-start gap-2">
                            <div className="flex items-center gap-2 flex-1">
                                {!isEditingTitle ? (
                                    <h2
                                        onClick={() => onNavigateToSong?.(song.id)}
                                        className="text-2xl font-bold text-zinc-900 dark:text-white leading-tight tracking-tight cursor-pointer hover:underline"
                                    >
                                        {song.title}
                                    </h2>
                                ) : (
                                    <div className="w-full">
                                        <input
                                            value={titleDraft}
                                            onChange={(e) => setTitleDraft(e.target.value)}
                                            onKeyDown={(e) => {
                                                if (e.key === 'Enter') {
                                                    e.preventDefault();
                                                    void saveTitleEdit();
                                                }
                                                if (e.key === 'Escape') {
                                                    e.preventDefault();
                                                    cancelTitleEdit();
                                                }
                                            }}
                                            className="w-full text-xl font-bold text-zinc-900 dark:text-white bg-white dark:bg-black/30 border border-zinc-200 dark:border-white/10 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-pink-500/40"
                                            maxLength={120}
                                            autoFocus
                                        />
                                        <div className="flex items-center gap-2 mt-2">
                                            <button
                                                onClick={() => void saveTitleEdit()}
                                                disabled={isSavingTitle}
                                                className="px-3 py-1.5 rounded-md text-xs font-semibold bg-pink-600 text-white hover:bg-pink-700 disabled:opacity-60"
                                            >
                                                {isSavingTitle ? t('saving') : t('save')}
                                            </button>
                                            <button
                                                onClick={cancelTitleEdit}
                                                disabled={isSavingTitle}
                                                className="px-3 py-1.5 rounded-md text-xs font-semibold bg-zinc-200 text-zinc-700 hover:bg-zinc-300 dark:bg-white/10 dark:text-zinc-200 dark:hover:bg-white/20 disabled:opacity-60"
                                            >
                                                {t('cancel')}
                                            </button>
                                            {titleError && (
                                                <span className="text-xs text-red-500">{titleError}</span>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                            <div className="relative">
                                {isOwner && !isEditingTitle && (
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            startTitleEdit();
                                        }}
                                        className="text-zinc-400 hover:text-black dark:hover:text-white p-1 mr-1"
                                        title="Rename song"
                                    >
                                        <Edit3 size={18} />
                                    </button>
                                )}
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        setShowMenu(!showMenu);
                                    }}
                                    className="text-zinc-400 hover:text-black dark:hover:text-white p-1"
                                >
                                    <MoreVertical size={20} />
                                </button>
                                <SongDropdownMenu
                                    song={song}
                                    isOpen={showMenu}
                                    onClose={() => setShowMenu(false)}
                                    isOwner={isOwner}
                                    onCreateVideo={onOpenVideo}
                                    onReusePrompt={() => onReuse?.(song)}
                                    onDelete={() => onDelete?.(song)}
                                    onAddToPlaylist={() => onAddToPlaylist?.(song)}
                                    onShare={() => setShareModalOpen(true)}
                                />
                            </div>
                        </div>

                        {/* Quality Scores */}
                        {(() => {
                            const gi = song.generationParams?.generationInfo;
                            if (!gi || typeof gi !== 'string' || !gi.includes('Quality Scores')) return null;

                            // Parse PMI: new format "PMI: 57%" or old "PMI Global: 0.5670"
                            const pmiNewMatch = gi.match(/PMI:\s*(\d+)%/);
                            const pmiOldMatch = !pmiNewMatch ? gi.match(/PMI Global:\s*([\d.]+)/) : null;
                            const pmiPct = pmiNewMatch ? pmiNewMatch[1] : pmiOldMatch ? Math.round(parseFloat(pmiOldMatch[1]) * 100).toString() : null;

                            // Parse DiT: new format "Lyric Alignment (X): ★★★ Good (0.081)" or old "DiT Alignment (X): 0.081"
                            const ditLabel = (v: number) => {
                                if (v >= 0.25) return '★★★★★';
                                if (v >= 0.15) return '★★★★';
                                if (v >= 0.08) return '★★★';
                                if (v >= 0.04) return '★★';
                                return '★';
                            };
                            const ditFullLabel = (v: number) => {
                                if (v >= 0.25) return '★★★★★ Excellent';
                                if (v >= 0.15) return '★★★★ Great';
                                if (v >= 0.08) return '★★★ Good';
                                if (v >= 0.04) return '★★ Fair';
                                return '★ Low';
                            };

                            const lmNewMatch = gi.match(/Lyric Alignment \(LM\):\s*(★[^(]+)\(([\d.]+)\)/);
                            const lmOldMatch = !lmNewMatch ? gi.match(/DiT Alignment \(LM\):\s*([\d.]+)/) : null;
                            const lmStars = lmNewMatch ? lmNewMatch[1].trim() : lmOldMatch ? ditFullLabel(parseFloat(lmOldMatch[1])) : null;

                            const ditNewMatch = gi.match(/Lyric Alignment \(DiT\):\s*(★[^(]+)\(([\d.]+)\)/);
                            const ditOldMatch = !ditNewMatch ? gi.match(/DiT Alignment \(DiT\):\s*([\d.]+)/) : null;
                            const ditStars = ditNewMatch ? ditNewMatch[1].trim() : ditOldMatch ? ditFullLabel(parseFloat(ditOldMatch[1])) : null;

                            if (!pmiPct && !lmStars && !ditStars) return null;
                            return (
                                <div className="flex flex-wrap items-center gap-1.5 mt-1">
                                    {pmiPct && (
                                        <span
                                            title="PMI (Pointwise Mutual Information): How well audio codes match your prompt. 50% = neutral, higher = better. Only with Thinking ON."
                                            className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-1 rounded-lg bg-gradient-to-r from-emerald-50 to-green-50 dark:from-emerald-900/30 dark:to-green-900/30 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-700/50 shadow-sm"
                                        >
                                            <Activity size={11} /> PMI {pmiPct}%
                                        </span>
                                    )}
                                    {lmStars && (
                                        <span
                                            title="Lyric Alignment (LM): How well lyrics sync with audio via the Language Model cross-attention. Uses coverage, monotonicity and path confidence."
                                            className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-1 rounded-lg bg-gradient-to-r from-sky-50 to-blue-50 dark:from-sky-900/30 dark:to-blue-900/30 text-sky-700 dark:text-sky-400 border border-sky-200 dark:border-sky-700/50 shadow-sm"
                                        >
                                            <AudioWaveform size={11} /> LM {lmStars}
                                        </span>
                                    )}
                                    {ditStars && (
                                        <span
                                            title="Lyric Alignment (DiT): How well lyrics sync with audio via the Diffusion Transformer cross-attention. Uses coverage, monotonicity and path confidence."
                                            className="inline-flex items-center gap-1 text-[10px] font-bold px-2 py-1 rounded-lg bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/30 dark:to-purple-900/30 text-violet-700 dark:text-violet-400 border border-violet-200 dark:border-violet-700/50 shadow-sm"
                                        >
                                            <Zap size={11} /> DiT {ditStars}
                                        </span>
                                    )}
                                </div>
                            );
                        })()}

                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-xs font-bold text-white shadow-sm ring-2 ring-white dark:ring-black">
                                {song.creator ? song.creator[0].toUpperCase() : 'A'}
                            </div>
                            <div className="flex flex-col">
                                <span
                                    onClick={() => song.creator && onNavigateToProfile?.(song.creator)}
                                    className="text-sm font-semibold text-zinc-900 dark:text-white hover:underline cursor-pointer"
                                >
                                    {song.creator || t('anonymous')}
                                </span>
                                <p className="text-xs text-zinc-500">{t('created')} {new Date(song.createdAt).toLocaleDateString()}</p>
                            </div>
                        </div>
                    </div>

                    {/* Main Actions */}
                    <div className="flex items-center justify-between px-2 py-2 bg-gradient-to-r from-zinc-100 to-zinc-50 dark:from-white/5 dark:to-white/[0.02] backdrop-blur-sm rounded-2xl border border-zinc-200 dark:border-white/10 shadow-sm">
                        <button
                            onClick={onOpenVideo}
                            title={t('createVideo')}
                            className="group flex flex-col items-center gap-1 p-2.5 text-zinc-500 hover:text-rose-600 dark:text-zinc-400 dark:hover:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-500/10 rounded-xl transition-all duration-200"
                        >
                            <Video size={18} strokeWidth={1.5} className="group-hover:scale-110 transition-transform" />
                            <span className="text-[9px] font-medium">Video</span>
                        </button>
                        <button
                            onClick={() => {
                                if (!song?.audioUrl) return;
                                const audioUrl = song.audioUrl.startsWith('http') ? song.audioUrl : `${window.location.origin}${song.audioUrl}`;
                                window.open(`/editor?audioUrl=${encodeURIComponent(audioUrl)}`, '_blank');
                            }}
                            title={t('openInEditor')}
                            className="group flex flex-col items-center gap-1 p-2.5 text-zinc-500 hover:text-amber-600 dark:text-zinc-400 dark:hover:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-500/10 rounded-xl transition-all duration-200"
                        >
                            <Edit3 size={18} strokeWidth={1.5} className="group-hover:scale-110 transition-transform" />
                            <span className="text-[9px] font-medium">Editor</span>
                        </button>
                        <button
                            onClick={() => onReuse && onReuse(song)}
                            title={t('reusePrompt')}
                            className="group flex flex-col items-center gap-1 p-2.5 text-zinc-500 hover:text-emerald-600 dark:text-zinc-400 dark:hover:text-emerald-400 hover:bg-emerald-50 dark:hover:bg-emerald-500/10 rounded-xl transition-all duration-200"
                        >
                            <Repeat size={18} strokeWidth={1.5} className="group-hover:scale-110 transition-transform" />
                            <span className="text-[9px] font-medium">Reuse</span>
                        </button>
                        <button
                            onClick={() => {
                                if (!song?.audioUrl) return;
                                openStemSplitter(song.audioUrl, song.title);
                            }}
                            title={t('extractStems')}
                            className="group flex flex-col items-center gap-1 p-2.5 text-zinc-500 hover:text-violet-600 dark:text-zinc-400 dark:hover:text-violet-400 hover:bg-violet-50 dark:hover:bg-violet-500/10 rounded-xl transition-all duration-200"
                        >
                            <Layers size={18} strokeWidth={1.5} className="group-hover:scale-110 transition-transform" />
                            <span className="text-[9px] font-medium">Stems</span>
                        </button>
                    </div>

                    {/* Icon Actions Row */}
                    <div className="flex items-center justify-center px-4 py-3 bg-gradient-to-r from-zinc-50/80 to-zinc-100/80 dark:from-white/[0.03] dark:to-white/[0.05] backdrop-blur-sm rounded-2xl border border-zinc-200/60 dark:border-white/10">
                        <div className="flex items-center gap-4">
                            <ActionButton
                                icon={<Heart size={20} fill={isLiked ? 'currentColor' : 'none'} />}
                                label={String(song.likeCount || 0)}
                                active={isLiked}
                                onClick={() => onToggleLike?.(song.id)}
                                variant="pink"
                            />
                            <div className="w-px h-6 bg-zinc-200 dark:bg-white/10" />
                            <ActionButton
                                icon={<Share2 size={20} />}
                                label={t('share')}
                                onClick={() => setShareModalOpen(true)}
                                variant="blue"
                            />
                            <div className="w-px h-6 bg-zinc-200 dark:bg-white/10" />
                            <ActionButton
                                icon={<Download size={20} />}
                                label={t('download')}
                                onClick={async () => {
                                    if (!song.audioUrl) return;
                                    try {
                                        const response = await fetch(song.audioUrl);
                                        const blob = await response.blob();
                                        const url = URL.createObjectURL(blob);
                                        const link = document.createElement('a');
                                        link.href = url;
                                        link.download = `${song.title || 'song'}.mp3`;
                                        document.body.appendChild(link);
                                        link.click();
                                        document.body.removeChild(link);
                                        URL.revokeObjectURL(url);
                                    } catch (error) {
                                        console.error('Download failed:', error);
                                    }
                                }}
                                variant="emerald"
                            />
                        </div>
                    </div>

                    {(song.generationParams?.referenceAudioUrl || song.generationParams?.sourceAudioUrl) && (
                        <div className="space-y-3">
                            <div className="flex items-center gap-2">
                                <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-500/20 dark:from-cyan-400/20 dark:to-blue-400/20 border border-cyan-200 dark:border-cyan-500/30 flex items-center justify-center">
                                    <LinkIcon size={14} className="text-cyan-600 dark:text-cyan-400" />
                                </div>
                                <h3 className="text-xs font-bold text-zinc-800 dark:text-zinc-200 uppercase tracking-wider">{t('sources')}</h3>
                            </div>
                            <div className="space-y-2">
                                {song.generationParams?.referenceAudioUrl && (
                                    <div className="group flex items-center justify-between gap-3 rounded-xl border border-zinc-200 dark:border-white/10 bg-gradient-to-r from-white to-zinc-50 dark:from-zinc-900/60 dark:to-black/40 px-3 py-2.5 hover:border-cyan-300 dark:hover:border-cyan-500/40 hover:shadow-md transition-all duration-200">
                                        <div className="flex items-center gap-2.5 min-w-0">
                                            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-100 to-blue-100 dark:from-cyan-500/20 dark:to-blue-500/20 flex items-center justify-center flex-shrink-0">
                                                <Music size={14} className="text-cyan-600 dark:text-cyan-400" />
                                            </div>
                                            <div className="min-w-0">
                                                <div className="text-[10px] font-bold text-cyan-600 dark:text-cyan-400 uppercase tracking-wider">{t('referenceAudio')}</div>
                                                <div className="text-xs font-bold text-zinc-800 dark:text-zinc-200 truncate">
                                                    {song.generationParams?.referenceAudioTitle || getSourceLabel(song.generationParams?.referenceAudioUrl)}
                                                </div>
                                            </div>
                                        </div>
                                        <button
                                            className="flex-shrink-0 text-[10px] font-bold px-3 py-1.5 rounded-full bg-gradient-to-r from-cyan-500 to-blue-500 text-white hover:from-cyan-600 hover:to-blue-600 shadow-sm hover:shadow-md hover:scale-105 transition-all duration-200"
                                            onClick={() => {
                                                if (!song.generationParams?.referenceAudioUrl || !onPlay) return;
                                                const previewSong = {
                                                    id: `ref_${song.id}`,
                                                    title: song.generationParams?.referenceAudioTitle || getSourceLabel(song.generationParams?.referenceAudioUrl),
                                                    lyrics: '',
                                                    style: 'Reference',
                                                    coverUrl: song.coverUrl,
                                                    duration: '0:00',
                                                    createdAt: new Date(),
                                                    tags: [],
                                                    audioUrl: song.generationParams?.referenceAudioUrl,
                                                    isPublic: false,
                                                    userId: song.userId,
                                                    creator: song.creator,
                                                };
                                                onPlay(previewSong);
                                            }}
                                        >
                                            Play
                                        </button>
                                    </div>
                                )}
                                {song.generationParams?.sourceAudioUrl && (
                                    <div className="group flex items-center justify-between gap-3 rounded-xl border border-zinc-200 dark:border-white/10 bg-gradient-to-r from-white to-zinc-50 dark:from-zinc-900/60 dark:to-black/40 px-3 py-2.5 hover:border-violet-300 dark:hover:border-violet-500/40 hover:shadow-md transition-all duration-200">
                                        <div className="flex items-center gap-2.5 min-w-0">
                                            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-100 to-purple-100 dark:from-violet-500/20 dark:to-purple-500/20 flex items-center justify-center flex-shrink-0">
                                                <Layers size={14} className="text-violet-600 dark:text-violet-400" />
                                            </div>
                                            <div className="min-w-0">
                                                <div className="text-[10px] font-bold text-violet-600 dark:text-violet-400 uppercase tracking-wider">{t('coverAudio')}</div>
                                                <div className="text-xs font-bold text-zinc-800 dark:text-zinc-200 truncate">
                                                    {song.generationParams?.sourceAudioTitle || getSourceLabel(song.generationParams?.sourceAudioUrl)}
                                                </div>
                                            </div>
                                        </div>
                                        <button
                                            className="flex-shrink-0 text-[10px] font-bold px-3 py-1.5 rounded-full bg-gradient-to-r from-violet-500 to-purple-500 text-white hover:from-violet-600 hover:to-purple-600 shadow-sm hover:shadow-md hover:scale-105 transition-all duration-200"
                                            onClick={() => {
                                                if (!song.generationParams?.sourceAudioUrl || !onPlay) return;
                                                const previewSong = {
                                                    id: `cover_${song.id}`,
                                                    title: song.generationParams?.sourceAudioTitle || getSourceLabel(song.generationParams?.sourceAudioUrl),
                                                    lyrics: '',
                                                    style: 'Cover',
                                                    coverUrl: song.coverUrl,
                                                    duration: '0:00',
                                                    createdAt: new Date(),
                                                    tags: [],
                                                    audioUrl: song.generationParams?.sourceAudioUrl,
                                                    isPublic: false,
                                                    userId: song.userId,
                                                    creator: song.creator,
                                                };
                                                onPlay(previewSong);
                                            }}
                                        >
                                            Play
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    <div className="h-px bg-zinc-200 dark:bg-white/5 w-full"></div>

                    {/* Tags / Style */}
                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-pink-500/20 to-rose-500/20 dark:from-pink-400/20 dark:to-rose-400/20 border border-pink-200 dark:border-pink-500/30 flex items-center justify-center">
                                    <Sparkles size={14} className="text-pink-600 dark:text-pink-400" />
                                </div>
                                <h2 className="text-xs font-bold text-zinc-800 dark:text-zinc-200 uppercase tracking-wider">{t('songDetails')}</h2>
                            </div>
                            <button
                                onClick={async (e) => {
                                    e.stopPropagation();
                                    try {
                                        const allTags = Array.isArray(song.tags) && song.tags.length > 0
                                            ? song.tags.join(', ')
                                            : (song.style ?? '');
                                        if (!allTags) return;
                                        await navigator.clipboard.writeText(allTags);
                                        setCopiedStyle(true);
                                        setTimeout(() => setCopiedStyle(false), 2000);
                                    } catch (error) {
                                        console.error('Failed to copy style tags:', error);
                                    }
                                }}
                                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-semibold transition-all duration-200 cursor-pointer ${copiedStyle ? 'bg-green-100 dark:bg-green-500/20 text-green-600 dark:text-green-400' : 'bg-zinc-100 dark:bg-white/10 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-white/15 hover:text-zinc-900 dark:hover:text-white'}`}
                                title={t('copyAllTags')}
                            >
                                <Copy size={11} />
                                {copiedStyle ? t('copied') : t('copy')}
                            </button>
                        </div>
                        <div
                            className={`flex flex-wrap gap-2 relative ${!tagsExpanded ? 'max-h-[28px] overflow-hidden' : ''}`}
                        >
                            {(Array.isArray(song.tags) && song.tags.length > 0
                                ? song.tags.filter(tag => tag && tag.trim())
                                : song.style.split(',').filter(tag => tag && tag.trim())
                            ).map((tag, idx) => (
                                <span
                                    key={idx}
                                    onClick={async (e) => {
                                        e.stopPropagation();
                                        try {
                                            await navigator.clipboard.writeText(tag.trim());
                                            setCopiedStyle(true);
                                            setTimeout(() => setCopiedStyle(false), 2000);
                                        } catch (error) {
                                            console.error('Failed to copy tag:', error);
                                        }
                                    }}
                                    className="px-3 py-1.5 bg-gradient-to-r from-zinc-100 to-zinc-50 dark:from-white/10 dark:to-white/5 hover:from-pink-50 hover:to-rose-50 dark:hover:from-pink-500/20 dark:hover:to-rose-500/20 border border-zinc-200 dark:border-white/10 hover:border-pink-300 dark:hover:border-pink-500/40 rounded-lg text-[11px] font-semibold text-zinc-700 dark:text-zinc-300 hover:text-pink-700 dark:hover:text-pink-300 transition-all duration-200 shadow-sm hover:shadow-md cursor-pointer"
                                    style={{ animationDelay: `${idx * 50}ms` }}
                                    title={t('copy')}
                                >
                                    {tag.trim()}
                                </span>
                            ))}
                            {!tagsExpanded && (
                                <button
                                    onClick={() => setTagsExpanded(true)}
                                    className="absolute right-0 top-0 px-3 py-1.5 bg-gradient-to-r from-zinc-200 to-zinc-300 dark:from-zinc-700 dark:to-zinc-600 hover:from-zinc-300 hover:to-zinc-400 dark:hover:from-zinc-600 dark:hover:to-zinc-500 rounded-lg text-[11px] font-bold text-zinc-700 dark:text-zinc-200 shadow-sm border border-zinc-300 dark:border-white/10 transition-all duration-200"
                                >
                                    +{t('more')}
                                </button>
                            )}
                            {tagsExpanded && (
                                <button
                                    onClick={() => setTagsExpanded(false)}
                                    className="px-3 py-1.5 bg-gradient-to-r from-zinc-200 to-zinc-300 dark:from-zinc-700 dark:to-zinc-600 hover:from-zinc-300 hover:to-zinc-400 dark:hover:from-zinc-600 dark:hover:to-zinc-500 rounded-lg text-[11px] font-bold text-zinc-700 dark:text-zinc-200 shadow-sm border border-zinc-300 dark:border-white/10 transition-all duration-200"
                                >
                                    {t('collapse')}
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Lyrics Section */}
                    <div className="bg-gradient-to-b from-white to-zinc-50/50 dark:from-black/30 dark:to-black/10 rounded-2xl border border-zinc-200 dark:border-white/10 overflow-hidden shadow-sm">
                        <div className="px-4 py-3.5 border-b border-zinc-200 dark:border-white/10 flex items-center justify-between bg-gradient-to-r from-zinc-50 to-zinc-100/50 dark:from-white/5 dark:to-transparent">
                            <div className="flex items-center gap-2">
                                <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-amber-500/20 to-yellow-500/20 dark:from-amber-400/20 dark:to-yellow-400/20 border border-amber-200 dark:border-amber-500/30 flex items-center justify-center">
                                    <Mic2 size={12} className="text-amber-600 dark:text-amber-400" />
                                </div>
                                <h3 className="text-xs font-bold text-zinc-700 dark:text-zinc-300 uppercase tracking-wider">{t('lyricsSection')}</h3>
                            </div>
                            <button
                                onClick={async (e) => {
                                    e.stopPropagation();
                                    try {
                                        if (song.lyrics) {
                                            await navigator.clipboard.writeText(song.lyrics);
                                            setCopiedLyrics(true);
                                            setTimeout(() => setCopiedLyrics(false), 2000);
                                        }
                                    } catch (error) {
                                        console.error('Failed to copy lyrics:', error);
                                    }
                                }}
                                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-semibold transition-all duration-200 cursor-pointer ${copiedLyrics ? 'bg-green-100 dark:bg-green-500/20 text-green-600 dark:text-green-400' : 'bg-white dark:bg-white/10 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-white/15 hover:text-zinc-900 dark:hover:text-white border border-zinc-200 dark:border-white/10'}`}
                            >
                                <Copy size={11} />
                                {copiedLyrics ? t('copied') : t('copy')}
                            </button>
                        </div>
                        <div className="p-4 max-h-[280px] overflow-y-auto custom-scrollbar">
                            <div className="text-[13px] text-zinc-700 dark:text-zinc-300 leading-relaxed opacity-95">
                                {song.lyrics ? (
                                    <div className="space-y-3">
                                        {song.lyrics.split('\n').map((line, idx) => (
                                            line.trim() === '' ? (
                                                <div key={idx} className="h-2" />
                                            ) : (
                                                <p key={idx} className="text-zinc-600 dark:text-zinc-300 hover:text-zinc-900 dark:hover:text-white transition-colors">
                                                    {line}
                                                </p>
                                            )
                                        ))}
                                    </div>
                                ) : (
                                    <div className="flex flex-col items-center justify-center py-10 text-center">
                                        <div className="w-12 h-12 rounded-full bg-zinc-100 dark:bg-white/5 flex items-center justify-center mb-3">
                                            <Music2 size={20} className="text-zinc-400 dark:text-zinc-600" />
                                        </div>
                                        <div className="text-zinc-500 dark:text-zinc-500 font-medium">{t('instrumentalTrack')}</div>
                                        <div className="text-zinc-400 dark:text-zinc-600 text-xs mt-1">{t('noLyricsGenerated')}</div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Generation Parameters Metadata */}
                    {song.generationParams && (
                        <div className="space-y-3">
                            {/* Section Header */}
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500/20 to-purple-500/20 dark:from-indigo-400/20 dark:to-purple-400/20 border border-indigo-200 dark:border-indigo-500/30 flex items-center justify-center">
                                        <SlidersHorizontal size={14} className="text-indigo-600 dark:text-indigo-400" />
                                    </div>
                                    <h3 className="text-xs font-bold text-zinc-800 dark:text-zinc-200 uppercase tracking-wider">
                                        {t('generationParameters')}
                                    </h3>
                                </div>
                                <button
                                    onClick={() => setGenParamsExpanded(v => !v)}
                                    className="text-[10px] font-semibold px-2.5 py-1 rounded-full bg-zinc-100 dark:bg-white/10 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-white/15 hover:text-zinc-900 dark:hover:text-white transition-all duration-200"
                                >
                                    {genParamsExpanded ? t('less') : t('more')}
                                </button>
                            </div>

                            {/* Parameters Grid */}
                            <div className="grid grid-cols-2 gap-2">
                                {(() => {
                                    const gp = song.generationParams;
                                    const coreItems: { label: string; value: string; icon: React.ReactNode; gradient: string; }[] = [];
                                    const moreItems: { label: string; value: string; icon: React.ReactNode; gradient: string; }[] = [];

                                    // Helper to get gradient based on param type
                                    const getGradient = (type: string) => {
                                        const gradients: Record<string, string> = {
                                            model: 'from-blue-500/10 to-cyan-500/10 border-blue-200 dark:border-blue-500/30 text-blue-600 dark:text-blue-400',
                                            audio: 'from-emerald-500/10 to-teal-500/10 border-emerald-200 dark:border-emerald-500/30 text-emerald-600 dark:text-emerald-400',
                                            timing: 'from-amber-500/10 to-orange-500/10 border-amber-200 dark:border-amber-500/30 text-amber-600 dark:text-amber-400',
                                            music: 'from-rose-500/10 to-pink-500/10 border-rose-200 dark:border-rose-500/30 text-rose-600 dark:text-rose-400',
                                            settings: 'from-violet-500/10 to-purple-500/10 border-violet-200 dark:border-violet-500/30 text-violet-600 dark:text-violet-400',
                                            tech: 'from-slate-500/10 to-zinc-500/10 border-slate-200 dark:border-slate-500/30 text-slate-600 dark:text-slate-400',
                                        };
                                        return gradients[type] || gradients.settings;
                                    };

                                    // Helper to get model short name — handles both bare IDs and full filesystem paths
                                    const getModelShortName = (modelId: string): string => {
                                        // Extract just the basename (last path segment), handling / and \ separators
                                        const base = modelId.split(/[\\/]/).filter(Boolean).pop() || modelId;
                                        return base.replace(/^acestep-/, '');
                                    };

                                    if (gp.ditModel || song.ditModel) {
                                        const modelId = gp.ditModel || song.ditModel || '';
                                        coreItems.push({
                                            label: t('metaModel'),
                                            value: getModelShortName(modelId),
                                            icon: <Cpu size={12} />,
                                            gradient: getGradient('model')
                                        });
                                    }
                                    if (gp.lmModel) {
                                        coreItems.push({
                                            label: t('metaLmModel'),
                                            value: getModelShortName(gp.lmModel),
                                            icon: <Terminal size={12} />,
                                            gradient: getGradient('model')
                                        });
                                    }
                                    if (gp.genres && gp.genres !== 'N/A') {
                                        coreItems.push({
                                            label: t('metaGenres'),
                                            value: gp.genres,
                                            icon: <Music2 size={12} />,
                                            gradient: getGradient('music')
                                        });
                                    }
                                    if (gp.bpm && gp.bpm > 0) {
                                        coreItems.push({
                                            label: t('metaBpm'),
                                            value: String(gp.bpm),
                                            icon: <Activity size={12} />,
                                            gradient: getGradient('timing')
                                        });
                                    }
                                    if (gp.keyScale) {
                                        coreItems.push({
                                            label: t('metaKey'),
                                            value: gp.keyScale,
                                            icon: <AudioWaveform size={12} />,
                                            gradient: getGradient('music')
                                        });
                                    }
                                    if (gp.timeSignature) {
                                        coreItems.push({
                                            label: t('metaTimeSignature'),
                                            value: String(gp.timeSignature),
                                            icon: <Clock size={12} />,
                                            gradient: getGradient('timing')
                                        });
                                    }
                                    if (gp.duration != null && gp.duration > 0) {
                                        const mins = Math.floor(gp.duration / 60);
                                        const secs = gp.duration % 60;
                                        coreItems.push({
                                            label: t('metaDuration'),
                                            value: mins > 0 ? `${mins}m ${secs}s` : `${secs}s`,
                                            icon: <Disc size={12} />,
                                            gradient: getGradient('audio')
                                        });
                                    }
                                    if (gp.audioFormat) {
                                        coreItems.push({
                                            label: t('metaAudioFormat'),
                                            value: gp.audioFormat.toUpperCase(),
                                            icon: <Volume2 size={12} />,
                                            gradient: getGradient('audio')
                                        });
                                    }
                                    if (gp.inferenceSteps) {
                                        coreItems.push({
                                            label: t('metaInferenceSteps'),
                                            value: String(gp.inferenceSteps),
                                            icon: <Gauge size={12} />,
                                            gradient: getGradient('tech')
                                        });
                                    }
                                    if (gp.guidanceScale != null) {
                                        coreItems.push({
                                            label: t('metaCfg'),
                                            value: String(gp.guidanceScale),
                                            icon: <Settings2 size={12} />,
                                            gradient: getGradient('tech')
                                        });
                                    }
                                    if (gp.seedText != null && String(gp.seedText).length > 0) {
                                        coreItems.push({
                                            label: t('metaSeed'),
                                            value: String(gp.seedText).substring(0, 12) + (String(gp.seedText).length > 12 ? '...' : ''),
                                            icon: <Hash size={12} />,
                                            gradient: getGradient('tech')
                                        });
                                    } else if (gp.seed != null && String(gp.seed).length > 0) {
                                        coreItems.push({
                                            label: t('metaSeed'),
                                            value: String(gp.seed),
                                            icon: <Hash size={12} />,
                                            gradient: getGradient('tech')
                                        });
                                    }
                                    if (gp.inferMethod) {
                                        coreItems.push({
                                            label: t('metaInferMethod'),
                                            value: gp.inferMethod.toUpperCase(),
                                            icon: <Zap size={12} />,
                                            gradient: getGradient('tech')
                                        });
                                    }
                                    if (gp.scheduler && gp.scheduler !== 'linear') {
                                        coreItems.push({
                                            label: t('metaScheduler'),
                                            value: gp.scheduler.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
                                            icon: <Clock size={12} />,
                                            gradient: getGradient('tech')
                                        });
                                    }
                                    if (gp.shift != null) {
                                        coreItems.push({
                                            label: t('metaShift'),
                                            value: String(gp.shift),
                                            icon: <Radio size={12} />,
                                            gradient: getGradient('settings')
                                        });
                                    }

                                    // Advanced params
                                    if (gp.lmBackend) {
                                        moreItems.push({
                                            label: t('lmBackendLabel'),
                                            value: String(gp.lmBackend),
                                            icon: <Terminal size={12} />,
                                            gradient: getGradient('model')
                                        });
                                    }
                                    if (gp.thinking != null) {
                                        moreItems.push({
                                            label: t('thinking'),
                                            value: gp.thinking ? 'ON' : 'OFF',
                                            icon: <Zap size={12} />,
                                            gradient: getGradient('settings')
                                        });
                                    }

                                    if (gp.batchSize != null) {
                                        moreItems.push({
                                            label: t('batchSize'),
                                            value: String(gp.batchSize),
                                            icon: <Layers size={12} />,
                                            gradient: getGradient('tech')
                                        });
                                    }
                                    if (gp.guidanceMode) {
                                        moreItems.push({
                                            label: t('guidanceMode'),
                                            value: String(gp.guidanceMode).toUpperCase(),
                                            icon: <Settings2 size={12} />,
                                            gradient: getGradient('settings')
                                        });
                                    }
                                    if (gp.usePag != null && gp.usePag) {
                                        moreItems.push({
                                            label: t('usePag'),
                                            value: `Scale: ${gp.pagScale || 'Default'}`,
                                            icon: <Settings2 size={12} />,
                                            gradient: getGradient('settings')
                                        });
                                    }
                                    if (gp.allowLmBatch != null) {
                                        moreItems.push({
                                            label: t('allowLmBatch'),
                                            value: gp.allowLmBatch ? 'ON' : 'OFF',
                                            icon: <Zap size={12} />,
                                            gradient: getGradient('settings')
                                        });
                                    }
                                    if (gp.useCotMetas != null) {
                                        moreItems.push({
                                            label: t('useCotMetas'),
                                            value: gp.useCotMetas ? 'ON' : 'OFF',
                                            icon: <Settings2 size={12} />,
                                            gradient: getGradient('settings')
                                        });
                                    }
                                    if (gp.useCotCaption != null) {
                                        moreItems.push({
                                            label: t('useCotCaption'),
                                            value: gp.useCotCaption ? 'ON' : 'OFF',
                                            icon: <Settings2 size={12} />,
                                            gradient: getGradient('settings')
                                        });
                                    }
                                    if (gp.useCotLanguage != null) {
                                        moreItems.push({
                                            label: t('useCotLanguage'),
                                            value: gp.useCotLanguage ? 'ON' : 'OFF',
                                            icon: <Settings2 size={12} />,
                                            gradient: getGradient('settings')
                                        });
                                    }
                                    if (gp.autogen != null) {
                                        moreItems.push({
                                            label: t('autogen'),
                                            value: gp.autogen ? 'ON' : 'OFF',
                                            icon: <Sparkles size={12} />,
                                            gradient: getGradient('settings')
                                        });
                                    }
                                    if (gp.constrainedDecodingDebug != null) {
                                        moreItems.push({
                                            label: t('constrainedDecodingDebug'),
                                            value: gp.constrainedDecodingDebug ? 'ON' : 'OFF',
                                            icon: <Terminal size={12} />,
                                            gradient: getGradient('tech')
                                        });
                                    }
                                    if (gp.isFormatCaption != null) {
                                        moreItems.push({
                                            label: t('formatCaption'),
                                            value: gp.isFormatCaption ? 'ON' : 'OFF',
                                            icon: <Settings2 size={12} />,
                                            gradient: getGradient('settings')
                                        });
                                    }
                                    if (gp.getScores != null) {
                                        moreItems.push({
                                            label: t('getScores'),
                                            value: gp.getScores ? 'ON' : 'OFF',
                                            icon: <Gauge size={12} />,
                                            gradient: getGradient('tech')
                                        });
                                    }
                                    if (gp.getLrc != null) {
                                        moreItems.push({
                                            label: t('getLrcLyrics'),
                                            value: gp.getLrc ? 'ON' : 'OFF',
                                            icon: <Mic2 size={12} />,
                                            gradient: getGradient('settings')
                                        });
                                    }

                                    // Adapters — only show basic LoRA if NOT using advanced multi-adapter
                                    if (gp.loraLoaded && gp.loraPath && !gp.advancedAdapters) {
                                        moreItems.push({
                                            label: 'Basic LoRA',
                                            value: `${gp.loraPath.split('/').pop()} (Scale: ${gp.loraScale || 1.0})`,
                                            icon: <Box size={12} />,
                                            gradient: getGradient('model')
                                        });
                                    }
                                    if (gp.advancedAdapters && Array.isArray(gp.adapterSlots)) {
                                        gp.adapterSlots.forEach((slot: any, idx: number) => {
                                            moreItems.push({
                                                label: `Adapter Slot ${slot.slot}`,
                                                value: `${slot.name || slot.path.split('/').pop()} (${slot.type || 'unknown'}, Scale: ${slot.scale || 1.0})`,
                                                icon: <Box size={12} />,
                                                gradient: getGradient('model')
                                            });
                                        });
                                    }

                                    // Steering
                                    if (gp.steeringEnabled && gp.steeringAlphas) {
                                        Object.entries(gp.steeringAlphas).forEach(([concept, alpha]) => {
                                            moreItems.push({
                                                label: `Steering Concept`,
                                                value: `${concept} (Alpha: ${alpha})`,
                                                icon: <Wand2 size={12} />,
                                                gradient: getGradient('tech')
                                            });
                                        });
                                    }

                                    const all = genParamsExpanded ? [...coreItems, ...moreItems] : coreItems;
                                    if (all.length === 0) return null;

                                    return all.map((item, idx) => (
                                        <div
                                            key={idx}
                                            className={`group relative flex items-center gap-2 px-3 py-2.5 bg-gradient-to-br ${item.gradient} rounded-xl border hover:shadow-md hover:scale-[1.02] transition-all duration-200 cursor-default`}
                                        >
                                            <div className="flex-shrink-0 w-5 h-5 rounded-md bg-white/60 dark:bg-black/30 flex items-center justify-center">
                                                {item.icon}
                                            </div>
                                            <div className="flex-1 min-w-0">
                                                <div className="text-[9px] font-bold uppercase tracking-wider text-zinc-500 dark:text-zinc-400 mb-0.5">
                                                    {item.label}
                                                </div>
                                                <div
                                                    className="text-xs font-bold text-zinc-800 dark:text-zinc-100 truncate font-mono"
                                                    title={item.value}
                                                >
                                                    {item.value}
                                                </div>
                                            </div>
                                        </div>
                                    ));
                                })()}
                            </div>
                        </div>
                    )}

                </div>
            </div>

            {song && (
                <ShareModal
                    isOpen={shareModalOpen}
                    onClose={() => setShareModalOpen(false)}
                    song={song}
                />
            )}
        </div>
    );
};

const ActionButton: React.FC<{ icon: React.ReactNode; label?: string; active?: boolean; variant?: 'pink' | 'blue' | 'emerald' | 'default'; onClick?: () => void }> = ({ icon, label, active, variant = 'default', onClick }) => {
    const variantClasses = {
        pink: active
            ? 'text-pink-600 dark:text-pink-500 bg-pink-50 dark:bg-pink-500/10 border-pink-200 dark:border-pink-500/30'
            : 'text-zinc-500 dark:text-zinc-400 hover:text-pink-600 dark:hover:text-pink-400 hover:bg-pink-50 dark:hover:bg-pink-500/10 border-transparent hover:border-pink-200 dark:hover:border-pink-500/30',
        blue: 'text-zinc-500 dark:text-zinc-400 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-500/10 border-transparent hover:border-blue-200 dark:hover:border-blue-500/30',
        emerald: 'text-zinc-500 dark:text-zinc-400 hover:text-emerald-600 dark:hover:text-emerald-400 hover:bg-emerald-50 dark:hover:bg-emerald-500/10 border-transparent hover:border-emerald-200 dark:hover:border-emerald-500/30',
        default: active
            ? 'text-pink-600 dark:text-pink-500'
            : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white'
    };

    return (
        <button
            onClick={onClick}
            className={`flex items-center gap-2 px-3 py-2 rounded-xl border transition-all duration-200 ${variantClasses[variant]}`}
        >
            {icon}
            {label && <span className="text-xs font-semibold">{label}</span>}
        </button>
    );
};
