import React, { useState, useEffect } from 'react';
import { Song } from '../types';
import { songsApi, getAudioUrl } from '../services/api';
import { useAuth } from '../context/AuthContext';
import { useI18n } from '../context/I18nContext';
import { ArrowLeft, Play, Pause, Heart, Share2, MoreHorizontal, ThumbsDown, Music as MusicIcon, Edit3, Eye, SlidersHorizontal, Cpu, Terminal, Music2, Activity, AudioWaveform, Clock, Disc, Volume2, Gauge } from 'lucide-react';
import { ShareModal } from './ShareModal';
import { SongDropdownMenu } from './SongDropdownMenu';

interface SongProfileProps {
    songId: string;
    onBack: () => void;
    onPlay: (song: Song, list?: Song[]) => void;
    onNavigateToProfile: (username: string) => void;
    currentSong?: Song | null;
    isPlaying?: boolean;
    likedSongIds?: Set<string>;
    onToggleLike?: (songId: string) => void;
    onDelete?: (song: Song) => void;
}

const updateMetaTags = (song: Song) => {
    const baseUrl = window.location.origin;
    const songUrl = `${baseUrl}/song/${song.id}`;
    const title = `${song.title} by ${song.creator || 'Unknown Artist'} | ACE-Step UI`;
    const description = `Listen to "${song.title}" - ${song.style}. ${song.viewCount || 0} plays, ${song.likeCount || 0} likes. Create your own AI music with ACE-Step UI.`;

    document.title = title;

    const updateOrCreateMeta = (selector: string, attribute: string, value: string) => {
        let element = document.querySelector(selector) as HTMLMetaElement;
        if (!element) {
            element = document.createElement('meta');
            const [attr, attrValue] = selector.replace(/[\[\]'"]/g, '').split('=');
            if (attr === 'property') element.setAttribute('property', attrValue);
            else if (attr === 'name') element.setAttribute('name', attrValue);
            document.head.appendChild(element);
        }
        element.setAttribute(attribute, value);
    };

    updateOrCreateMeta('meta[name="description"]', 'content', description);
    updateOrCreateMeta('meta[name="title"]', 'content', title);

    updateOrCreateMeta('meta[property="og:type"]', 'content', 'music.song');
    updateOrCreateMeta('meta[property="og:url"]', 'content', songUrl);
    updateOrCreateMeta('meta[property="og:title"]', 'content', title);
    updateOrCreateMeta('meta[property="og:description"]', 'content', description);
    updateOrCreateMeta('meta[property="og:image"]', 'content', song.coverUrl);
    updateOrCreateMeta('meta[property="og:image:width"]', 'content', '400');
    updateOrCreateMeta('meta[property="og:image:height"]', 'content', '400');
    updateOrCreateMeta('meta[property="og:audio"]', 'content', song.audioUrl || '');
    updateOrCreateMeta('meta[property="og:audio:type"]', 'content', 'audio/mpeg');

    updateOrCreateMeta('meta[name="twitter:card"]', 'content', 'summary_large_image');
    updateOrCreateMeta('meta[name="twitter:url"]', 'content', songUrl);
    updateOrCreateMeta('meta[name="twitter:title"]', 'content', title);
    updateOrCreateMeta('meta[name="twitter:description"]', 'content', description);
    updateOrCreateMeta('meta[name="twitter:image"]', 'content', song.coverUrl);

    updateOrCreateMeta('meta[property="music:duration"]', 'content', String(song.duration || 0));
    updateOrCreateMeta('meta[property="music:musician"]', 'content', song.creator || 'Unknown Artist');
};

const resetMetaTags = () => {
    document.title = 'ACE-Step UI - Local AI Music Generator';
    const defaultDescription = 'Create original music with AI locally. Generate songs in any style with custom lyrics and professional quality using ACE-Step.';
    const defaultImage = '/og-image.png';

    const updateMeta = (selector: string, content: string) => {
        const element = document.querySelector(selector) as HTMLMetaElement;
        if (element) element.setAttribute('content', content);
    };

    updateMeta('meta[name="description"]', defaultDescription);
    updateMeta('meta[property="og:title"]', 'ACE-Step UI - Local AI Music Generator');
    updateMeta('meta[property="og:description"]', defaultDescription);
    updateMeta('meta[property="og:image"]', defaultImage);
    updateMeta('meta[property="og:type"]', 'website');
    updateMeta('meta[name="twitter:title"]', 'ACE-Step UI - Local AI Music Generator');
    updateMeta('meta[name="twitter:description"]', defaultDescription);
    updateMeta('meta[name="twitter:image"]', defaultImage);
};

export const SongProfile: React.FC<SongProfileProps> = ({ songId, onBack, onPlay, onNavigateToProfile, currentSong, isPlaying, likedSongIds = new Set(), onToggleLike, onDelete }) => {
    const { user, token } = useAuth();
    const { t } = useI18n();
    const [song, setSong] = useState<Song | null>(null);
    const [loading, setLoading] = useState(true);
    const [shareModalOpen, setShareModalOpen] = useState(false);
    const [showDropdown, setShowDropdown] = useState(false);
    const [genParamsExpanded, setGenParamsExpanded] = useState(false);

    const isCurrentSong = song && currentSong?.id === song.id;
    const isCurrentlyPlaying = isCurrentSong && isPlaying;
    const isLiked = song ? likedSongIds.has(song.id) : false;

    useEffect(() => {
        loadSongData();
        return () => resetMetaTags();
    }, [songId]);

    useEffect(() => {
        if (song) {
            updateMetaTags(song);
        }
    }, [song]);

    const loadSongData = async () => {
        setLoading(true);
        try {
            const response = await songsApi.getFullSong(songId, token);

            const generationParams = (() => {
                try {
                    const s: any = response.song as any;
                    const parsed = (() => {
                        const gp = s.generation_params;
                        if (!gp) return undefined;
                        return typeof gp === 'string' ? JSON.parse(gp) : gp;
                    })();

                    const gp: any = (parsed && typeof parsed === 'object') ? { ...parsed } : {};

                    const bpm = gp.bpm ?? s.bpm;
                    const duration = gp.duration ?? s.duration;
                    const keyScale = gp.keyScale ?? gp.key_scale ?? gp.keyscale ?? s.key_scale;
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
                    const timeSignature = normalizeTimeSignature(gp.timeSignature ?? gp.time_signature ?? gp.timesignature ?? s.time_signature);
                    const ditModel = gp.ditModel ?? gp.dit_model ?? s.dit_model ?? s.ditModel;
                    const lmModel = gp.lmModel ?? gp.lm_model ?? s.lm_model;

                    if (bpm != null) gp.bpm = bpm;
                    if (duration != null) gp.duration = duration;
                    if (keyScale != null) gp.keyScale = keyScale;
                    if (timeSignature != null) gp.timeSignature = timeSignature;
                    if (ditModel != null) gp.ditModel = ditModel;
                    if (lmModel != null) gp.lmModel = lmModel;

                    return Object.keys(gp).length > 0 ? gp : undefined;
                } catch {
                    return undefined;
                }
            })();

            const transformedSong: Song = {
                id: response.song.id,
                title: response.song.title,
                lyrics: response.song.lyrics,
                style: response.song.style,
                coverUrl: `https://picsum.photos/seed/${response.song.id}/400/400`,
                duration: response.song.duration
                    ? `${Math.floor(response.song.duration / 60)}:${String(Math.floor(response.song.duration % 60)).padStart(2, '0')}`
                    : '0:00',
                createdAt: new Date(response.song.created_at),
                tags: response.song.tags || [],
                audioUrl: getAudioUrl(response.song.audio_url, response.song.id),
                isPublic: response.song.is_public,
                likeCount: response.song.like_count || 0,
                viewCount: response.song.view_count || 0,
                userId: response.song.user_id,
                creator: response.song.creator,
                creator_avatar: response.song.creator_avatar,
                generationParams,
            };

            setSong(transformedSong);
        } catch (error) {
            console.error('Failed to load song:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-full bg-zinc-50 dark:bg-black">
                <div className="text-zinc-500 dark:text-zinc-400 flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-zinc-400 border-t-transparent rounded-full animate-spin" />
                    {t('loadingSong')}
                </div>
            </div>
        );
    }

    if (!song) {
        return (
            <div className="flex flex-col items-center justify-center h-full gap-4 bg-zinc-50 dark:bg-black">
                <div className="text-zinc-500 dark:text-zinc-400">{t('songNotFound')}</div>
                <button onClick={onBack} className="px-4 py-2 bg-zinc-200 dark:bg-zinc-800 hover:bg-zinc-300 dark:hover:bg-zinc-700 rounded-lg text-zinc-900 dark:text-white transition-colors">
                    {t('goBack')}
                </button>
            </div>
        );
    }

    return (
        <div className="w-full h-full flex flex-col bg-zinc-50 dark:bg-black overflow-hidden">
            {/* Header */}
            <div className="border-b border-zinc-200 dark:border-zinc-800 px-4 md:px-6 py-4 flex-shrink-0">
                <button
                    onClick={onBack}
                    className="flex items-center gap-2 text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white mb-4 transition-colors"
                >
                    <ArrowLeft size={20} />
                    <span>{t('back')}</span>
                </button>

                <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
                    <div className="flex-1">
                        <h1 className="text-2xl md:text-3xl font-bold text-zinc-900 dark:text-white mb-2">{song.title}</h1>
                        <div className="flex items-center gap-3 mb-3">
                            <div
                                onClick={() => song.creator && onNavigateToProfile(song.creator)}
                                className="flex items-center gap-2 cursor-pointer hover:underline"
                            >
                                <div className="w-6 h-6 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-xs font-bold text-white overflow-hidden">
                                    {song.creator_avatar ? (
                                        <img src={song.creator_avatar} alt={song.creator || 'Creator'} className="w-full h-full object-cover" />
                                    ) : (
                                        song.creator ? song.creator[0].toUpperCase() : 'A'
                                    )}
                                </div>
                                <span className="text-zinc-900 dark:text-white font-semibold">{song.creator || 'Anonymous'}</span>
                            </div>
                        </div>

                        {/* Tags */}
                        <div className="flex flex-wrap gap-2 mb-2">
                            {song.style.split(',').slice(0, 4).map((tag, i) => (
                                <span key={i} className="px-2 py-1 bg-zinc-200 dark:bg-zinc-800 rounded text-xs text-zinc-600 dark:text-zinc-300">
                                    {tag.trim()}
                                </span>
                            ))}
                        </div>

                        <div className="text-xs text-zinc-500">
                            {new Date(song.createdAt).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })} at {new Date(song.createdAt).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                            {!song.isPublic && song.userId === user?.id && (
                                <span className="ml-2 px-2 py-0.5 bg-zinc-200 dark:bg-zinc-800 rounded text-xs text-zinc-600 dark:text-zinc-400">{t('private')}</span>
                            )}
                        </div>
                    </div>

                    {/* Related Songs Tab - Hidden on mobile */}
                    <div className="hidden md:flex items-center gap-2">
                        <button className="px-4 py-2 bg-zinc-900 dark:bg-white text-white dark:text-black rounded-full text-sm font-semibold">
                            Similar
                        </button>
                        <button
                            onClick={() => song.creator && onNavigateToProfile(song.creator)}
                            className="px-4 py-2 text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white text-sm font-semibold transition-colors"
                        >
                            By {song.creator || 'Artist'}
                        </button>
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto">
                <div className="max-w-6xl mx-auto px-4 md:px-6 lg:px-8 py-6 md:py-8 pb-24 lg:pb-32">
                    
                    {/* Two Column Layout */}
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8">
                        
                        {/* Left Column: Cover Art & Actions (4 cols) */}
                        <div className="lg:col-span-4 space-y-6">
                            {/* Cover Art */}
                            <div className="relative aspect-square max-w-sm mx-auto lg:max-w-none rounded-2xl overflow-hidden shadow-2xl ring-1 ring-zinc-200 dark:ring-white/10">
                                <img src={song.coverUrl} alt={song.title} className={`w-full h-full object-cover transition-transform duration-700 ${isCurrentlyPlaying ? 'scale-105' : ''}`} />
                                <button
                                    onClick={() => onPlay(song)}
                                    className={`absolute inset-0 transition-all duration-300 flex items-center justify-center group ${isCurrentSong ? 'bg-black/40' : 'bg-black/20 hover:bg-black/40'}`}
                                >
                                    <div className="w-20 h-20 rounded-full bg-white/95 group-hover:bg-white group-hover:scale-110 transition-all duration-300 flex items-center justify-center shadow-2xl backdrop-blur-sm">
                                        {isCurrentlyPlaying ? (
                                            <Pause size={32} className="text-black fill-black" />
                                        ) : (
                                            <Play size={32} className="text-black fill-black ml-1" />
                                        )}
                                    </div>
                                </button>
                                {isCurrentlyPlaying && (
                                    <div className="absolute bottom-4 left-4 flex items-end gap-1">
                                        <span className="w-1.5 h-5 bg-pink-500 rounded-full animate-pulse" style={{ animationDelay: '0ms' }} />
                                        <span className="w-1.5 h-8 bg-pink-500 rounded-full animate-pulse" style={{ animationDelay: '150ms' }} />
                                        <span className="w-1.5 h-4 bg-pink-500 rounded-full animate-pulse" style={{ animationDelay: '300ms' }} />
                                        <span className="w-1.5 h-9 bg-pink-500 rounded-full animate-pulse" style={{ animationDelay: '450ms' }} />
                                    </div>
                                )}
                            </div>

                            {/* Action Buttons */}
                            <div className="flex items-center justify-center lg:justify-start gap-2">
                                <button
                                    onClick={() => onToggleLike?.(song.id)}
                                    className={`flex items-center gap-1.5 px-3 py-2 rounded-full text-sm font-semibold transition-all duration-200 ${isLiked ? 'bg-pink-500 text-white shadow-lg shadow-pink-500/25' : 'bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 hover:border-pink-300 dark:hover:border-pink-500/50 text-zinc-900 dark:text-white shadow-sm'}`}
                                >
                                    <Heart size={16} className={isLiked ? 'fill-current' : ''} />
                                    <span>{song.likeCount || 0}</span>
                                </button>
                                
                                <div className="flex items-center gap-1.5 px-3 py-2 rounded-full bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-sm text-zinc-700 dark:text-zinc-300">
                                    <Eye size={14} />
                                    <span className="font-medium">{song.viewCount || 0}</span>
                                </div>



                                
                                <button
                                    onClick={() => setShareModalOpen(true)}
                                    className="p-2 bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 hover:border-zinc-300 dark:hover:border-zinc-600 rounded-full transition-all duration-200 shadow-sm"
                                >
                                    <Share2 size={16} className="text-zinc-700 dark:text-zinc-300" />
                                </button>
                                
                                <div className="relative">
                                    <button
                                        onClick={() => setShowDropdown(!showDropdown)}
                                        className="p-2 bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 hover:border-zinc-300 dark:hover:border-zinc-600 rounded-full transition-all duration-200 shadow-sm"
                                    >
                                        <MoreHorizontal size={16} className="text-zinc-700 dark:text-zinc-300" />
                                    </button>
                                    {song && (
                                        <SongDropdownMenu
                                            song={song}
                                            isOpen={showDropdown}
                                            onClose={() => setShowDropdown(false)}
                                            isOwner={user?.id === song.userId}
                                            onReusePrompt={() => {}}
                                            onAddToPlaylist={() => {}}
                                            onDelete={() => onDelete?.(song)}
                                            onShare={() => setShareModalOpen(true)}
                                        />
                                    )}
                                </div>
                            </div>

                            {/* Generation Parameters - Below cover on desktop */}
                            {song.generationParams && (
                                <div className="hidden lg:block bg-white dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 rounded-xl p-4">
                                    {/* Section Header */}
                                    <div className="flex items-center justify-between mb-3">
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
                                            className="text-xs font-semibold px-2.5 py-1 rounded-full bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-700 hover:text-zinc-900 dark:hover:text-white transition-all duration-200"
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

                                        // Helper to get model short name (remove 'acestep-' prefix)
                                        const getModelShortName = (modelId: string): string => {
                                            return modelId.replace(/^acestep-/, '');
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
                                            moreItems.push({ 
                                                label: t('metaAudioFormat'), 
                                                value: gp.audioFormat.toUpperCase(),
                                                icon: <Volume2 size={12} />,
                                                gradient: getGradient('audio')
                                            });
                                        }
                                        if (gp.inferenceSteps) {
                                            moreItems.push({ 
                                                label: t('metaInferenceSteps'), 
                                                value: String(gp.inferenceSteps),
                                                icon: <Gauge size={12} />,
                                                gradient: getGradient('tech')
                                            });
                                        }

                                        const displayItems = genParamsExpanded 
                                            ? [...coreItems, ...moreItems] 
                                            : coreItems.slice(0, 6);

                                        return displayItems.map((item, idx) => (
                                            <div 
                                                key={idx} 
                                                className={`group relative flex items-center gap-2 px-3 py-2.5 bg-gradient-to-br ${item.gradient} rounded-xl border hover:shadow-md hover:scale-[1.02] transition-all duration-200`}
                                            >
                                                <div className="flex-shrink-0 opacity-70 group-hover:opacity-100 transition-opacity">
                                                    {item.icon}
                                                </div>
                                                <div className="min-w-0 flex-1">
                                                    <div className="text-[10px] opacity-70 leading-tight truncate">{item.label}</div>
                                                    <div className="text-xs font-bold leading-tight truncate">{item.value}</div>
                                                </div>
                                            </div>
                                        ));
                                    })()}
                                </div>
                            </div>
                        )}
                        </div>

                        {/* Right Column: Lyrics & Mobile GenParams (8 cols) */}
                        <div className="lg:col-span-8 space-y-6">
                            {/* Lyrics */}
                            {song.lyrics && (
                                <div className="bg-white dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-5 md:p-6">
                                    <div className="flex items-center gap-2 mb-4">
                                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-rose-500/20 to-pink-500/20 dark:from-rose-400/20 dark:to-pink-400/20 border border-rose-200 dark:border-rose-500/30 flex items-center justify-center">
                                            <MusicIcon size={16} className="text-rose-600 dark:text-rose-400" />
                                        </div>
                                        <h3 className="text-sm font-bold text-zinc-800 dark:text-zinc-200 uppercase tracking-wider">
                                            {t('lyrics')}
                                        </h3>
                                    </div>
                                    <div className="text-sm md:text-base text-zinc-700 dark:text-zinc-300 whitespace-pre-line leading-relaxed max-h-[60vh] overflow-y-auto pr-2 lyrics-scrollbar">
                                        {song.lyrics}
                                    </div>
                                </div>
                            )}

                            {/* Generation Parameters - Mobile only (shown below lyrics) */}
                            {song.generationParams && (
                                <div className="lg:hidden bg-white dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 rounded-xl p-4">
                                    {/* Section Header */}
                                    <div className="flex items-center justify-between mb-3">
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
                                            className="text-xs font-semibold px-2.5 py-1 rounded-full bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-700 hover:text-zinc-900 dark:hover:text-white transition-all duration-200"
                                        >
                                            {genParamsExpanded ? t('less') : t('more')}
                                        </button>
                                    </div>

                                    {/* Parameters Grid */}
                                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                                        {(() => {
                                            const gp = song.generationParams;
                                            const coreItems: { label: string; value: string; icon: React.ReactNode; gradient: string; }[] = [];
                                            const moreItems: { label: string; value: string; icon: React.ReactNode; gradient: string; }[] = [];

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

                                            const getModelShortName = (modelId: string): string => {
                                                return modelId.replace(/^acestep-/, '');
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
                                                moreItems.push({ 
                                                    label: t('metaAudioFormat'), 
                                                    value: gp.audioFormat.toUpperCase(),
                                                    icon: <Volume2 size={12} />,
                                                    gradient: getGradient('audio')
                                                });
                                            }
                                            if (gp.inferenceSteps) {
                                                moreItems.push({ 
                                                    label: t('metaInferenceSteps'), 
                                                    value: String(gp.inferenceSteps),
                                                    icon: <Gauge size={12} />,
                                                    gradient: getGradient('tech')
                                                });
                                            }

                                            const displayItems = genParamsExpanded 
                                                ? [...coreItems, ...moreItems] 
                                                : coreItems.slice(0, 6);

                                            return displayItems.map((item, idx) => (
                                                <div 
                                                    key={idx} 
                                                    className={`group relative flex items-center gap-2 px-3 py-2.5 bg-gradient-to-br ${item.gradient} rounded-xl border hover:shadow-md hover:scale-[1.02] transition-all duration-200`}
                                                >
                                                    <div className="flex-shrink-0 opacity-70 group-hover:opacity-100 transition-opacity">
                                                        {item.icon}
                                                    </div>
                                                    <div className="min-w-0 flex-1">
                                                        <div className="text-[10px] opacity-70 leading-tight truncate">{item.label}</div>
                                                        <div className="text-xs font-bold leading-tight truncate">{item.value}</div>
                                                    </div>
                                                </div>
                                            ));
                                        })()}
                                    </div>
                                </div>
                            )}
                        </div>

                    </div>
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
