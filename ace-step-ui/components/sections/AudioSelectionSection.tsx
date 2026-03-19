import React from 'react';
import { useI18n } from '../../context/I18nContext';

interface AudioBoxProps {
    /** Which audio target this box controls */
    target: 'reference' | 'source';
    /** The loaded audio URL (empty = no audio loaded) */
    audioUrl: string | null;
    audioTitle: string;
    playing: boolean;
    duration: number;
    time: number;
    audioRef: React.RefObject<HTMLAudioElement>;
    setAudioUrl: (val: string) => void;
    setAudioTitle: (val: string) => void;
    setPlaying: (val: boolean) => void;
    setTime: (val: number) => void;
    setDuration: (val: number) => void;
    toggleAudio: (type: 'reference' | 'source') => void;
    openAudioModal: (tab: 'reference' | 'source', initialTab: 'library' | 'uploads') => void;
    inputRef: React.RefObject<HTMLInputElement>;
    handleDrop: (e: React.DragEvent, type: 'reference' | 'source') => void;
    handleDragOver: (e: React.DragEvent) => void;
    formatTime: (time: number) => string;
    getAudioLabel: (url: string) => string;
    /** Analyze button for cover audio (BPM/key detection) */
    onAnalyze?: () => void;
    isAnalyzing?: boolean;
}

/** Self-contained audio box for a single target (reference OR source) */
const AudioBox: React.FC<AudioBoxProps> = ({
    target,
    audioUrl,
    audioTitle,
    playing,
    duration,
    time,
    audioRef,
    setAudioUrl,
    setAudioTitle,
    setPlaying,
    setTime,
    setDuration,
    toggleAudio,
    openAudioModal,
    inputRef,
    handleDrop,
    handleDragOver,
    formatTime,
    getAudioLabel,
    onAnalyze,
    isAnalyzing,
}) => {
    const { t } = useI18n();
    const isSource = target === 'source';

    // Color scheme differs per target
    const gradient = isSource
        ? 'from-emerald-500 to-teal-600'
        : 'from-pink-500 to-purple-600';
    const shadow = isSource ? 'shadow-emerald-500/20' : 'shadow-pink-500/20';
    const progressGradient = isSource
        ? 'from-emerald-500 to-teal-500'
        : 'from-pink-500 to-purple-500';

    return (
        <div
            onDrop={(e) => handleDrop(e, target)}
            onDragOver={handleDragOver}
            className="bg-white dark:bg-[#1a1a1f] rounded-xl border border-zinc-200 dark:border-white/5 overflow-hidden"
        >
            {/* Header */}
            <div className="px-3 py-2.5 border-b border-zinc-100 dark:border-white/5 bg-zinc-50 dark:bg-white/[0.02]">
                <span className="text-xs font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wide">
                    {isSource ? t('coverAudio') : t('referenceAudio')}
                </span>
            </div>

            {/* Content */}
            <div className="p-3 space-y-2">
                {/* Audio player (when audio is loaded) */}
                {audioUrl && (
                    <div className="flex items-center gap-3 p-2 rounded-lg bg-zinc-50 dark:bg-white/[0.03] border border-zinc-100 dark:border-white/5">
                        <button
                            type="button"
                            onClick={() => toggleAudio(target)}
                            className={`relative flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br ${gradient} text-white flex items-center justify-center shadow-lg ${shadow} hover:scale-105 transition-transform`}
                        >
                            {playing ? (
                                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" /></svg>
                            ) : (
                                <svg className="w-4 h-4 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
                            )}
                            <span className="absolute -bottom-1 -right-1 text-[8px] font-bold bg-zinc-900 text-white px-1 py-0.5 rounded">
                                {formatTime(duration)}
                            </span>
                        </button>
                        <div className="flex-1 min-w-0">
                            <div className="text-xs font-medium text-zinc-800 dark:text-zinc-200 truncate mb-1.5">
                                {audioTitle || getAudioLabel(audioUrl)}
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-[10px] text-zinc-400 tabular-nums">{formatTime(time)}</span>
                                <div
                                    className="flex-1 h-1.5 rounded-full bg-zinc-200 dark:bg-white/10 cursor-pointer group/seek"
                                    onClick={(e) => {
                                        if (audioRef.current && duration > 0) {
                                            const rect = e.currentTarget.getBoundingClientRect();
                                            const percent = (e.clientX - rect.left) / rect.width;
                                            audioRef.current.currentTime = percent * duration;
                                        }
                                    }}
                                >
                                    <div
                                        className={`h-full bg-gradient-to-r ${progressGradient} rounded-full transition-all relative`}
                                        style={{ width: duration ? `${Math.min(100, (time / duration) * 100)}%` : '0%' }}
                                    >
                                        <div className="absolute right-0 top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-white shadow-md opacity-0 group-hover/seek:opacity-100 transition-opacity" />
                                    </div>
                                </div>
                                <span className="text-[10px] text-zinc-400 tabular-nums">{formatTime(duration)}</span>
                            </div>
                        </div>
                        {/* Clear button */}
                        <button
                            type="button"
                            onClick={() => { setAudioUrl(''); setAudioTitle(''); setPlaying(false); setTime(0); setDuration(0); }}
                            className="p-1.5 rounded-full hover:bg-zinc-200 dark:hover:bg-white/10 text-zinc-400 hover:text-zinc-600 dark:hover:text-white transition-colors"
                        >
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                        </button>
                        {/* Analyze button — source only */}
                        {isSource && onAnalyze && (
                            <div className="flex items-center gap-2">
                                {isAnalyzing && (
                                    <span className="text-[10px] text-amber-600 dark:text-amber-400 font-medium animate-pulse ml-2">
                                        {t('analyzingAudio') !== 'analyzingAudio' ? t('analyzingAudio') : 'Analyzing key and BPM...'}
                                    </span>
                                )}
                                <button
                                    type="button"
                                    onClick={onAnalyze}
                                    disabled={isAnalyzing}
                                    title={t('analyzeSource')}
                                    className={`p-1.5 rounded-full transition-colors ${isAnalyzing
                                        ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400 cursor-wait'
                                        : 'hover:bg-emerald-100 dark:hover:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300'
                                        }`}
                                >
                                    {isAnalyzing ? (
                                        <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                        </svg>
                                    ) : (
                                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                        </svg>
                                    )}
                                </button>
                            </div>
                        )}
                    </div>
                )}

                {/* Action buttons */}
                <div className="flex gap-2">
                    <button
                        type="button"
                        onClick={() => openAudioModal(target, 'uploads')}
                        className="flex-1 flex items-center justify-center gap-1.5 rounded-lg bg-zinc-100 dark:bg-white/5 hover:bg-zinc-200 dark:hover:bg-white/10 text-zinc-700 dark:text-zinc-300 px-3 py-2 text-xs font-medium transition-colors border border-zinc-200 dark:border-white/5"
                    >
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                        </svg>
                        {t('fromLibrary')}
                    </button>
                    <button
                        type="button"
                        onClick={() => inputRef.current?.click()}
                        className="flex-1 flex items-center justify-center gap-1.5 rounded-lg bg-zinc-100 dark:bg-white/5 hover:bg-zinc-200 dark:hover:bg-white/10 text-zinc-700 dark:text-zinc-300 px-3 py-2 text-xs font-medium transition-colors border border-zinc-200 dark:border-white/5"
                    >
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                        {t('upload')}
                    </button>
                </div>
            </div>
        </div>
    );
};

/* ─────────────────────────────────────────────────── */

interface AudioSelectionSectionProps {
    useReferenceAudio: boolean;
    setUseReferenceAudio: (val: boolean) => void;
    referenceAsCover?: boolean;
    setReferenceAsCover?: (val: boolean) => void;
    taskType: string;
    // Reference audio state
    referenceAudioUrl: string | null;
    referenceAudioTitle: string;
    referencePlaying: boolean;
    toggleAudio: (type: 'reference' | 'source') => void;
    referenceDuration: number;
    referenceTime: number;
    referenceAudioRef: React.RefObject<HTMLAudioElement>;
    setReferenceAudioUrl: (val: string) => void;
    setReferenceAudioTitle: (val: string) => void;
    setReferencePlaying: (val: boolean) => void;
    setReferenceTime: (val: number) => void;
    setReferenceDuration: (val: number) => void;
    // Source/cover audio state
    sourceAudioUrl: string | null;
    sourceAudioTitle: string;
    sourcePlaying: boolean;
    sourceDuration: number;
    sourceTime: number;
    sourceAudioRef: React.RefObject<HTMLAudioElement>;
    setSourceAudioUrl: (val: string) => void;
    setSourceAudioTitle: (val: string) => void;
    setSourcePlaying: (val: boolean) => void;
    setSourceTime: (val: number) => void;
    setSourceDuration: (val: number) => void;
    openAudioModal: (tab: 'reference' | 'source', initialTab: 'library' | 'uploads') => void;
    referenceInputRef: React.RefObject<HTMLInputElement>;
    sourceInputRef: React.RefObject<HTMLInputElement>;
    handleDrop: (e: React.DragEvent, type: 'reference' | 'source') => void;
    handleDragOver: (e: React.DragEvent) => void;
    formatTime: (time: number) => string;
    getAudioLabel: (url: string) => string;
    onAnalyzeSource?: () => void;
    isAnalyzing?: boolean;
}

export const AudioSelectionSection: React.FC<AudioSelectionSectionProps> = ({
    useReferenceAudio,
    setUseReferenceAudio,
    referenceAsCover,
    setReferenceAsCover,
    taskType,
    referenceAudioUrl,
    referenceAudioTitle,
    referencePlaying,
    toggleAudio,
    referenceDuration,
    referenceTime,
    referenceAudioRef,
    setReferenceAudioUrl,
    setReferenceAudioTitle,
    setReferencePlaying,
    setReferenceTime,
    setReferenceDuration,
    sourceAudioUrl,
    sourceAudioTitle,
    sourcePlaying,
    sourceDuration,
    sourceTime,
    sourceAudioRef,
    setSourceAudioUrl,
    setSourceAudioTitle,
    setSourcePlaying,
    setSourceTime,
    setSourceDuration,
    openAudioModal,
    referenceInputRef,
    sourceInputRef,
    handleDrop,
    handleDragOver,
    formatTime,
    getAudioLabel,
    onAnalyzeSource,
    isAnalyzing
}) => {
    const { t } = useI18n();

    return (
        <div className="space-y-5">
            {/* ── Cover / Source Audio box ── always visible for non-text2music */}
            {taskType !== 'text2music' && (
                <AudioBox
                    target="source"
                    audioUrl={sourceAudioUrl}
                    audioTitle={sourceAudioTitle}
                    playing={sourcePlaying}
                    duration={sourceDuration}
                    time={sourceTime}
                    audioRef={sourceAudioRef}
                    setAudioUrl={setSourceAudioUrl}
                    setAudioTitle={setSourceAudioTitle}
                    setPlaying={setSourcePlaying}
                    setTime={setSourceTime}
                    setDuration={setSourceDuration}
                    toggleAudio={toggleAudio}
                    openAudioModal={openAudioModal}
                    inputRef={sourceInputRef}
                    handleDrop={handleDrop}
                    handleDragOver={handleDragOver}
                    formatTime={formatTime}
                    getAudioLabel={getAudioLabel}
                    onAnalyze={onAnalyzeSource}
                    isAnalyzing={isAnalyzing}
                />
            )}

            {/* ── Use Reference Audio toggle ── hidden in extract mode */}
            {taskType !== 'extract' && (
                <div className="flex items-center justify-between px-2">
                    <div>
                        <span className="text-xs font-bold text-zinc-600 dark:text-zinc-300 uppercase tracking-wide">{t('useReferenceAudio')}</span>
                        <p className="text-[11px] text-zinc-400 dark:text-zinc-500">{t('useReferenceAudioTooltip')}</p>
                    </div>
                    <button
                        onClick={() => {
                            const newValue = !useReferenceAudio;
                            setUseReferenceAudio(newValue);
                            if (!newValue) {
                                setReferenceAudioUrl('');
                                setReferenceAudioTitle('');
                                setReferencePlaying(false);
                                setReferenceTime(0);
                                setReferenceDuration(0);
                            }
                        }}
                        className={`w-10 h-5 rounded-full flex items-center transition-colors duration-200 px-0.5 border border-zinc-200 dark:border-white/5 ${useReferenceAudio ? 'bg-pink-600' : 'bg-zinc-300 dark:bg-black/40'} cursor-pointer`}
                    >
                        <div className={`w-4 h-4 rounded-full bg-white transform transition-transform duration-200 shadow-sm ${useReferenceAudio ? 'translate-x-5' : 'translate-x-0'}`} />
                    </button>
                </div>
            )}

            {/* ── Reference As Cover toggle ── only when reference is active */}
            {useReferenceAudio && taskType !== 'extract' && setReferenceAsCover && (
                <div className="flex items-center justify-between px-2 pl-4 py-1 border-l-2 border-pink-500/20 ml-2">
                    <div>
                        <span className="text-xs font-bold text-zinc-600 dark:text-zinc-300">Reference As Cover</span>
                        <p className="text-[10px] text-zinc-400 dark:text-zinc-500">Treats reference audio as structural cover input instead of style transfer</p>
                    </div>
                    <button
                        onClick={() => setReferenceAsCover(!referenceAsCover)}
                        className={`w-8 h-4 rounded-full flex items-center transition-colors duration-200 px-0.5 border border-zinc-200 dark:border-white/5 ${referenceAsCover ? 'bg-pink-600' : 'bg-zinc-300 dark:bg-black/40'} cursor-pointer`}
                    >
                        <div className={`w-3 h-3 rounded-full bg-white transform transition-transform duration-200 shadow-sm ${referenceAsCover ? 'translate-x-4' : 'translate-x-0'}`} />
                    </button>
                </div>
            )}

            {/* ── Reference Audio box ── only when toggle is ON and not extract mode */}
            {useReferenceAudio && taskType !== 'extract' && (
                <AudioBox
                    target="reference"
                    audioUrl={referenceAudioUrl}
                    audioTitle={referenceAudioTitle}
                    playing={referencePlaying}
                    duration={referenceDuration}
                    time={referenceTime}
                    audioRef={referenceAudioRef}
                    setAudioUrl={setReferenceAudioUrl}
                    setAudioTitle={setReferenceAudioTitle}
                    setPlaying={setReferencePlaying}
                    setTime={setReferenceTime}
                    setDuration={setReferenceDuration}
                    toggleAudio={toggleAudio}
                    openAudioModal={openAudioModal}
                    inputRef={referenceInputRef}
                    handleDrop={handleDrop}
                    handleDragOver={handleDragOver}
                    formatTime={formatTime}
                    getAudioLabel={getAudioLabel}
                />
            )}
        </div>
    );
};

export default AudioSelectionSection;
