// FullscreenVisualizer.tsx — Full-screen Winamp-style visualizer with auto-hiding HUD
import React, { useRef, useEffect, useState, useCallback } from 'react';
import ReactDOM from 'react-dom';
import { Play, Pause, SkipBack, SkipForward, X, Minimize2 } from 'lucide-react';
import { Song } from '../types';
import { LiveVisualizer } from './LiveVisualizer';
import { LyricsOverlay } from './LyricsOverlay';

interface FullscreenVisualizerProps {
    isOpen: boolean;
    onClose: () => void;
    song: Song | null;
    isPlaying: boolean;
    currentTime: number;
    duration: number;
    onTogglePlay: () => void;
    onNext: () => void;
    onPrevious: () => void;
    onSeek: (time: number) => void;
}

const formatTime = (time: number) => {
    if (!Number.isFinite(time) || time < 0) return '0:00';
    const mins = Math.floor(time / 60);
    const secs = Math.floor(time % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
};

export const FullscreenVisualizer: React.FC<FullscreenVisualizerProps> = ({
    isOpen, onClose, song,
    isPlaying, currentTime, duration,
    onTogglePlay, onNext, onPrevious, onSeek,
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [showHud, setShowHud] = useState(true);
    const hideTimerRef = useRef<ReturnType<typeof setTimeout>>();
    const progressRef = useRef<HTMLDivElement>(null);

    // Auto-hide HUD after 3 seconds of inactivity
    const resetHideTimer = useCallback(() => {
        setShowHud(true);
        if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
        hideTimerRef.current = setTimeout(() => setShowHud(false), 3000);
    }, []);

    // Mouse movement shows HUD
    useEffect(() => {
        if (!isOpen) return;
        const handleMouseMove = () => resetHideTimer();
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                onClose();
            } else if (e.key === ' ') {
                e.preventDefault();
                onTogglePlay();
                resetHideTimer();
            } else if (e.key === 'ArrowRight') {
                onNext();
                resetHideTimer();
            } else if (e.key === 'ArrowLeft') {
                onPrevious();
                resetHideTimer();
            }
        };

        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('keydown', handleKeyDown);
        resetHideTimer();

        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('keydown', handleKeyDown);
            if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
        };
    }, [isOpen, onClose, onTogglePlay, onNext, onPrevious, resetHideTimer]);

    // Request actual fullscreen API
    useEffect(() => {
        if (!isOpen || !containerRef.current) return;

        const el = containerRef.current;
        if (document.fullscreenElement !== el) {
            el.requestFullscreen?.().catch(console.error);
        }

        const onFullscreenChange = () => {
            if (!document.fullscreenElement) {
                onClose();
            }
        };

        document.addEventListener('fullscreenchange', onFullscreenChange);
        return () => {
            document.removeEventListener('fullscreenchange', onFullscreenChange);
            if (document.fullscreenElement) {
                document.exitFullscreen?.().catch(console.error);
            }
        };
    }, [isOpen, onClose]);

    const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (!progressRef.current || !duration) return;
        const rect = progressRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percent = Math.max(0, Math.min(1, x / rect.width));
        onSeek(percent * duration);
    };

    if (!isOpen || !song) return null;

    const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

    return ReactDOM.createPortal(
        <div
            ref={containerRef}
            className="fixed inset-0 z-[100] bg-black cursor-none"
            style={{ cursor: showHud ? 'default' : 'none' }}
            onClick={resetHideTimer}
        >
            {/* Visualizer canvas */}
            <LiveVisualizer
                isPlaying={isPlaying}
                className="absolute inset-0 w-full h-full"
                showControls={false}
                instanceId="fullscreen"
            />

            {/* Synced lyrics overlay — always visible, not tied to HUD; hidden for cover/repaint */}
            {song.generationParams?.taskType !== 'cover' && song.generationParams?.taskType !== 'repaint' && (
                <LyricsOverlay
                    audioUrl={song.audioUrl}
                    currentTime={currentTime}
                    isPlaying={isPlaying}
                />
            )}

            {/* HUD Overlay */}
            <div
                className={`absolute inset-0 flex flex-col justify-end transition-opacity duration-500 ${showHud ? 'opacity-100' : 'opacity-0 pointer-events-none'
                    }`}
            >
                {/* Close button */}
                <button
                    onClick={onClose}
                    className="absolute top-6 right-6 p-3 rounded-full bg-black/40 hover:bg-black/60 text-white/70 hover:text-white transition-all backdrop-blur-sm"
                >
                    <Minimize2 size={20} />
                </button>

                {/* Bottom HUD */}
                <div className="bg-gradient-to-t from-black/80 via-black/40 to-transparent pt-24 pb-8 px-8">
                    {/* Song info */}
                    <div className="mb-4">
                        <h2 className="text-2xl font-bold text-white drop-shadow-lg truncate">
                            {song.title}
                        </h2>
                        <p className="text-sm text-white/60 font-medium mt-0.5">
                            {song.creator || 'Unknown Artist'} • {song.style}
                        </p>
                    </div>

                    {/* Progress bar */}
                    <div className="flex items-center gap-3 mb-4">
                        <span className="text-xs text-white/50 font-mono w-10 text-right">
                            {formatTime(currentTime)}
                        </span>
                        <div
                            ref={progressRef}
                            className="flex-1 h-1.5 bg-white/20 rounded-full cursor-pointer group relative"
                            onClick={handleProgressClick}
                        >
                            <div
                                className="h-full bg-gradient-to-r from-pink-500 to-rose-400 rounded-full relative transition-all"
                                style={{ width: `${progress}%` }}
                            >
                                <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3.5 h-3.5 rounded-full bg-white shadow-lg opacity-0 group-hover:opacity-100 transition-opacity" />
                            </div>
                        </div>
                        <span className="text-xs text-white/50 font-mono w-10">
                            {formatTime(duration)}
                        </span>
                    </div>

                    {/* Playback controls */}
                    <div className="flex items-center justify-center gap-6">
                        <button
                            onClick={onPrevious}
                            className="p-2 text-white/60 hover:text-white transition-colors"
                        >
                            <SkipBack size={24} fill="currentColor" />
                        </button>
                        <button
                            onClick={onTogglePlay}
                            className="w-14 h-14 rounded-full bg-white/10 hover:bg-white/20 backdrop-blur-sm flex items-center justify-center text-white transition-all hover:scale-105"
                        >
                            {isPlaying ? (
                                <Pause size={28} fill="currentColor" />
                            ) : (
                                <Play size={28} fill="currentColor" className="ml-1" />
                            )}
                        </button>
                        <button
                            onClick={onNext}
                            className="p-2 text-white/60 hover:text-white transition-colors"
                        >
                            <SkipForward size={24} fill="currentColor" />
                        </button>
                    </div>
                </div>
            </div>
        </div>,
        document.body
    );
};
