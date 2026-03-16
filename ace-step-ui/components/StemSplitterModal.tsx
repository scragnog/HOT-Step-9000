import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom';
import { X, Download, Play, Pause, Layers, Loader2, Volume2, VolumeX, Archive } from 'lucide-react';

// ---- Types ----

interface StemResult {
    id: string;
    stem_type: string;
    file_path: string;
    file_name: string;
    duration: number;
}

// ---- Global open hook ----
// This lets any component trigger the modal without prop-drilling or lifecycle issues.

type OpenFn = (audioUrl: string, songTitle: string) => void;
let _globalOpen: OpenFn | null = null;

/** Call from anywhere to open the stem splitter modal. */
export function openStemSplitter(audioUrl: string, songTitle?: string) {
    _globalOpen?.(audioUrl, songTitle || 'Untitled');
}

// ---- Constants ----

const PYTHON_API = (() => {
    if (typeof window !== 'undefined') {
        const host = window.location.hostname;
        return `http://${host}:8001`;
    }
    return 'http://localhost:8001';
})();

const EXPRESS_API = (() => {
    if (typeof window !== 'undefined') {
        const host = window.location.hostname;
        return `http://${host}:3001`;
    }
    return 'http://localhost:3001';
})();

const MODES = [
    { id: 'vocals', label: 'Vocals / Instrumental', desc: 'BS-RoFormer — best quality vocal isolation', stems: 2 },
    { id: 'every-stem', label: '6-Stem Separation', desc: 'Vocals, Drums, Bass, Guitar, Piano, Other', stems: 6 },
];

const STEM_ICONS: Record<string, string> = {
    vocals: '🎤', drums: '🥁', bass: '🎸', guitar: '🎸',
    piano: '🎹', instrumental: '🎵', other: '🎵',
};

const STEM_COLORS: Record<string, string> = {
    vocals: 'from-pink-500/20 to-rose-500/20 border-pink-500/30',
    drums: 'from-amber-500/20 to-orange-500/20 border-amber-500/30',
    bass: 'from-emerald-500/20 to-teal-500/20 border-emerald-500/30',
    guitar: 'from-blue-500/20 to-indigo-500/20 border-blue-500/30',
    piano: 'from-purple-500/20 to-violet-500/20 border-purple-500/30',
    instrumental: 'from-cyan-500/20 to-sky-500/20 border-cyan-500/30',
    other: 'from-zinc-500/20 to-slate-500/20 border-zinc-500/30',
};

const STEM_ACCENT: Record<string, string> = {
    vocals: 'bg-pink-500', drums: 'bg-amber-500', bass: 'bg-emerald-500',
    guitar: 'bg-blue-500', piano: 'bg-purple-500', instrumental: 'bg-cyan-500',
    other: 'bg-zinc-500',
};

// ---- Stem Mixer (multi-track synchronized player) ----

const formatTime = (s: number) => {
    if (!isFinite(s) || s < 0) return '0:00';
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
};

const StemMixer: React.FC<{ stems: StemResult[]; jobId: string }> = ({ stems, jobId }) => {
    const audioRefs = useRef<(HTMLAudioElement | null)[]>([]);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [volumes, setVolumes] = useState<number[]>(() => stems.map(() => 1));
    const [muted, setMuted] = useState<boolean[]>(() => stems.map(() => false));
    const [solo, setSolo] = useState<string | null>(null);
    const [loaded, setLoaded] = useState(0);
    const animRef = useRef<number>(0);

    // Build audio URLs
    const audioUrls = stems.map(s =>
        `${PYTHON_API}/v1/stems/${jobId}/download/${encodeURIComponent(s.stem_type)}`
    );

    // Sync time display via requestAnimationFrame
    useEffect(() => {
        const tick = () => {
            const master = audioRefs.current[0];
            if (master && isPlaying) {
                setCurrentTime(master.currentTime);
            }
            animRef.current = requestAnimationFrame(tick);
        };
        animRef.current = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(animRef.current);
    }, [isPlaying]);

    // Update volumes/mutes reactively
    useEffect(() => {
        audioRefs.current.forEach((el, i) => {
            if (!el) return;
            const isMuted = muted[i] || (solo !== null && stems[i].stem_type !== solo);
            el.volume = isMuted ? 0 : volumes[i];
        });
    }, [volumes, muted, solo, stems]);

    const onLoadedMetadata = (i: number) => {
        const el = audioRefs.current[i];
        if (el && el.duration > duration) setDuration(el.duration);
        setLoaded(prev => prev + 1);
    };

    const playAll = () => {
        audioRefs.current.forEach(el => el?.play());
        setIsPlaying(true);
    };

    const pauseAll = () => {
        audioRefs.current.forEach(el => el?.pause());
        setIsPlaying(false);
    };

    const togglePlay = () => (isPlaying ? pauseAll : playAll)();

    const seekTo = (pct: number) => {
        const t = pct * duration;
        audioRefs.current.forEach(el => { if (el) el.currentTime = t; });
        setCurrentTime(t);
    };

    const onEnded = () => {
        setIsPlaying(false);
        setCurrentTime(0);
        audioRefs.current.forEach(el => { if (el) el.currentTime = 0; });
    };

    const toggleMute = (i: number) => {
        setMuted(prev => { const n = [...prev]; n[i] = !n[i]; return n; });
    };

    const toggleSolo = (stemType: string) => {
        setSolo(prev => prev === stemType ? null : stemType);
    };

    const setVolume = (i: number, v: number) => {
        setVolumes(prev => { const n = [...prev]; n[i] = v; return n; });
    };

    const downloadStem = async (stem: StemResult) => {
        try {
            const url = `${PYTHON_API}/v1/stems/${jobId}/download/${encodeURIComponent(stem.stem_type)}`;
            const resp = await fetch(url);
            const blob = await resp.blob();
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = stem.file_name;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        } catch (err) { console.error('Download failed:', err); }
    };

    const allLoaded = loaded >= stems.length;
    const progressPct = duration > 0 ? (currentTime / duration) * 100 : 0;

    return (
        <div className="space-y-3">
            {/* Hidden audio elements */}
            {stems.map((stem, i) => (
                <audio
                    key={stem.id}
                    ref={el => { audioRefs.current[i] = el; }}
                    src={audioUrls[i]}
                    preload="auto"
                    onLoadedMetadata={() => onLoadedMetadata(i)}
                    onEnded={i === 0 ? onEnded : undefined}
                />
            ))}

            {/* Master transport */}
            <div className="flex items-center gap-3 p-3 rounded-xl bg-zinc-100 dark:bg-zinc-800/50 border border-zinc-200 dark:border-white/10">
                <button
                    onClick={togglePlay}
                    disabled={!allLoaded}
                    className="w-11 h-11 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-lg hover:scale-105 transition-transform disabled:opacity-40 flex-shrink-0"
                >
                    {isPlaying
                        ? <Pause size={18} className="text-white" fill="white" />
                        : <Play size={18} className="text-white ml-0.5" fill="white" />}
                </button>
                <div className="flex-1 min-w-0">
                    <div
                        className="w-full h-2 bg-zinc-300 dark:bg-zinc-700 rounded-full cursor-pointer relative group"
                        onClick={(e) => {
                            const rect = e.currentTarget.getBoundingClientRect();
                            seekTo((e.clientX - rect.left) / rect.width);
                        }}
                    >
                        <div
                            className="h-full bg-gradient-to-r from-violet-500 to-purple-500 rounded-full transition-[width] duration-100"
                            style={{ width: `${progressPct}%` }}
                        />
                        <div
                            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow-md border-2 border-violet-500 opacity-0 group-hover:opacity-100 transition-opacity"
                            style={{ left: `${progressPct}%`, marginLeft: '-6px' }}
                        />
                    </div>
                    <div className="flex justify-between mt-1">
                        <span className="text-[10px] text-zinc-500 font-mono">{formatTime(currentTime)}</span>
                        <span className="text-[10px] text-zinc-500 font-mono">{formatTime(duration)}</span>
                    </div>
                </div>
            </div>

            {/* Per-stem mixer channels */}
            {stems.map((stem, i) => {
                const colorClass = STEM_COLORS[stem.stem_type] || STEM_COLORS.other;
                const accentClass = STEM_ACCENT[stem.stem_type] || STEM_ACCENT.other;
                const icon = STEM_ICONS[stem.stem_type] || '🎵';
                const isMuted = muted[i] || (solo !== null && stem.stem_type !== solo);
                const isSoloed = solo === stem.stem_type;

                return (
                    <div key={stem.id}
                        className={`flex items-center gap-2.5 p-2.5 rounded-xl border bg-gradient-to-r ${colorClass} transition-all ${isMuted && !isSoloed ? 'opacity-40' : ''}`}
                    >
                        {/* Stem info */}
                        <div className="flex items-center gap-1.5 w-24 flex-shrink-0">
                            <span className="text-base">{icon}</span>
                            <span className="text-xs font-bold text-zinc-900 dark:text-white capitalize truncate">{stem.stem_type}</span>
                        </div>

                        {/* Volume slider */}
                        <div className="flex-1 flex items-center gap-2">
                            <div className={`w-1.5 h-6 rounded-full ${accentClass} flex-shrink-0`} />
                            <input
                                type="range"
                                min={0} max={1} step={0.01}
                                value={volumes[i]}
                                onChange={(e) => setVolume(i, parseFloat(e.target.value))}
                                className="flex-1 h-1.5 accent-violet-500 cursor-pointer"
                            />
                        </div>

                        {/* Mute button */}
                        <button
                            onClick={() => toggleMute(i)}
                            className={`p-1.5 rounded-lg text-xs font-bold transition-colors ${muted[i]
                                ? 'bg-red-500/20 text-red-500'
                                : 'hover:bg-black/10 dark:hover:bg-white/10 text-zinc-600 dark:text-zinc-400'
                                }`}
                            title={muted[i] ? 'Unmute' : 'Mute'}
                        >
                            {muted[i]
                                ? <VolumeX size={14} />
                                : <Volume2 size={14} />}
                        </button>

                        {/* Solo button */}
                        <button
                            onClick={() => toggleSolo(stem.stem_type)}
                            className={`px-2 py-1 rounded-lg text-[10px] font-black uppercase tracking-wider transition-colors ${isSoloed
                                ? 'bg-amber-500/30 text-amber-500 ring-1 ring-amber-500/50'
                                : 'hover:bg-black/10 dark:hover:bg-white/10 text-zinc-500 dark:text-zinc-400'
                                }`}
                            title={isSoloed ? 'Unsolo' : 'Solo'}
                        >
                            S
                        </button>

                        {/* Download */}
                        <button
                            onClick={() => downloadStem(stem)}
                            className="p-1.5 rounded-lg hover:bg-black/10 dark:hover:bg-white/10 transition-colors flex-shrink-0"
                            title="Download stem"
                        >
                            <Download size={14} className="text-zinc-600 dark:text-zinc-400" />
                        </button>
                    </div>
                );
            })}
        </div>
    );
};

// ---- Main Modal (self-managing — render once in App.tsx) ----

export const StemSplitterModal: React.FC = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [audioUrl, setAudioUrl] = useState('');
    const [songTitle, setSongTitle] = useState('');
    const [mode, setMode] = useState('vocals');
    const [status, setStatus] = useState<'idle' | 'running' | 'complete' | 'error'>('idle');
    const [progress, setProgress] = useState(0);
    const [message, setMessage] = useState('');
    const [stems, setStems] = useState<StemResult[]>([]);
    const [jobId, setJobId] = useState('');
    const [error, setError] = useState('');
    const [available, setAvailable] = useState<boolean | null>(null);
    const eventSourceRef = useRef<EventSource | null>(null);

    // Register global open function
    useEffect(() => {
        _globalOpen = (url: string, title: string) => {
            setAudioUrl(url);
            setSongTitle(title);
            setStatus('idle');
            setProgress(0);
            setMessage('');
            setStems([]);
            setJobId('');
            setError('');
            setIsOpen(true);
        };
        return () => { _globalOpen = null; };
    }, []);

    // Check availability when modal opens
    useEffect(() => {
        if (!isOpen) return;
        fetch(`${PYTHON_API}/v1/stems/available`)
            .then(r => r.json())
            .then(data => setAvailable(data.available))
            .catch(() => setAvailable(false));
    }, [isOpen]);

    // Cleanup SSE on unmount
    useEffect(() => {
        return () => { eventSourceRef.current?.close(); };
    }, []);

    const onClose = useCallback(() => {
        eventSourceRef.current?.close();
        setIsOpen(false);
    }, []);

    const resolveAudioPath = (): string => {
        // Express URL: /audio/songId/fileId.flac → pass as-is, Python API will resolve
        // Python API URL: /v1/audio?path=xxx → extract path param
        if (audioUrl.includes('/v1/audio?path=')) {
            try {
                const url = new URL(audioUrl, window.location.origin);
                return decodeURIComponent(url.searchParams.get('path') || audioUrl);
            } catch { /* fall through */ }
        }
        // Relative URL (e.g. /audio/...) or absolute path — pass as-is
        return audioUrl;
    };

    const startSeparation = async () => {
        setStatus('running');
        setProgress(0);
        setMessage('Starting…');
        setError('');
        setStems([]);

        try {
            const audioPath = resolveAudioPath();
            const resp = await fetch(`${PYTHON_API}/v1/stems/separate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ audio_path: audioPath, mode }),
            });

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({ detail: resp.statusText }));
                throw new Error(errData.detail || `Server error ${resp.status}`);
            }

            const { job_id } = await resp.json();
            setJobId(job_id);

            // SSE progress stream
            const es = new EventSource(`${PYTHON_API}/v1/stems/${job_id}/progress`);
            eventSourceRef.current = es;

            es.onmessage = (evt) => {
                try {
                    const data = JSON.parse(evt.data);
                    if (data.type === 'progress') {
                        setProgress(data.percent);
                        setMessage(data.message || '');
                    } else if (data.type === 'complete') {
                        setStatus('complete');
                        setStems(data.stems || []);
                        setProgress(1);
                        setMessage('Done!');
                        es.close();
                    } else if (data.type === 'error') {
                        setStatus('error');
                        setError(data.message || 'Unknown error');
                        es.close();
                    }
                } catch { /* ignore parse errors */ }
            };

            es.onerror = () => { es.close(); };
        } catch (err) {
            setStatus('error');
            setError(err instanceof Error ? err.message : 'Failed to start separation');
        }
    };

    if (!isOpen) return null;

    const modalContent = (
        <div
            className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 dark:bg-black/80 backdrop-blur-sm p-4"
            onClick={(e) => e.target === e.currentTarget && onClose()}
        >
            <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-white/10 rounded-2xl w-full max-w-lg shadow-2xl animate-in fade-in zoom-in-95 duration-200 max-h-[85vh] flex flex-col overflow-hidden">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-white/10">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-lg">
                            <Layers size={20} className="text-white" />
                        </div>
                        <div>
                            <h2 className="text-lg font-bold text-zinc-900 dark:text-white">Extract Stems</h2>
                            {songTitle && <p className="text-xs text-zinc-500 truncate max-w-[200px]">{songTitle}</p>}
                        </div>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-zinc-100 dark:hover:bg-white/10 rounded-lg transition-colors">
                        <X size={20} className="text-zinc-500" />
                    </button>
                </div>

                {/* Body */}
                <div className="flex-1 overflow-y-auto p-6 space-y-5">
                    {available === false && (
                        <div className="p-4 rounded-xl bg-red-50 dark:bg-red-500/10 border border-red-200 dark:border-red-500/30 text-sm text-red-700 dark:text-red-300">
                            <strong>audio-separator not installed.</strong> Run{' '}
                            <code className="bg-red-100 dark:bg-red-500/20 px-1.5 py-0.5 rounded text-xs">
                                python install_audio_separator.py
                            </code>{' '}from the project root.
                        </div>
                    )}

                    {status === 'idle' && (
                        <div className="space-y-3">
                            <label className="text-sm font-bold text-zinc-700 dark:text-zinc-300 uppercase tracking-wider">
                                Separation Mode
                            </label>
                            <div className="grid grid-cols-2 gap-2">
                                {MODES.map((m) => (
                                    <button key={m.id} onClick={() => setMode(m.id)}
                                        className={`p-3 rounded-xl border text-left transition-all ${mode === m.id
                                            ? 'border-violet-500 bg-violet-50 dark:bg-violet-500/10 ring-1 ring-violet-500/50'
                                            : 'border-zinc-200 dark:border-white/10 hover:border-zinc-300 dark:hover:border-white/20'
                                            }`}>
                                        <div className="text-sm font-bold text-zinc-900 dark:text-white">{m.label}</div>
                                        <div className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-0.5">{m.desc}</div>
                                        <div className="text-[10px] text-violet-600 dark:text-violet-400 font-mono mt-1">{m.stems} stems</div>
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {status === 'running' && (
                        <div className="space-y-4">
                            <div className="flex items-center gap-3">
                                <Loader2 size={20} className="text-violet-500 animate-spin" />
                                <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">{message}</span>
                            </div>
                            <div className="w-full h-3 bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden">
                                <div className="h-full bg-gradient-to-r from-violet-500 to-purple-500 rounded-full transition-all duration-500 ease-out"
                                    style={{ width: `${Math.max(progress * 100, 2)}%` }} />
                            </div>
                            <p className="text-xs text-zinc-500 text-center font-mono">{(progress * 100).toFixed(0)}%</p>
                        </div>
                    )}

                    {status === 'error' && (
                        <div className="p-4 rounded-xl bg-red-50 dark:bg-red-500/10 border border-red-200 dark:border-red-500/30">
                            <p className="text-sm font-medium text-red-700 dark:text-red-300">Separation failed</p>
                            <p className="text-xs text-red-500 dark:text-red-400 mt-1">{error}</p>
                            <button onClick={() => setStatus('idle')}
                                className="mt-3 px-4 py-1.5 text-xs font-bold rounded-lg bg-red-100 dark:bg-red-500/20 text-red-700 dark:text-red-300 hover:bg-red-200 dark:hover:bg-red-500/30 transition-colors">
                                Try Again
                            </button>
                        </div>
                    )}

                    {status === 'complete' && stems.length > 0 && (
                        <div className="space-y-3">
                            <div className="flex items-center justify-between">
                                <span className="text-sm font-bold text-zinc-700 dark:text-zinc-300 uppercase tracking-wider">
                                    {stems.length} Stems Extracted
                                </span>
                                <button onClick={() => setStatus('idle')}
                                    className="text-xs font-bold text-violet-600 dark:text-violet-400 hover:underline">
                                    Extract Again
                                </button>
                            </div>
                            <StemMixer stems={stems} jobId={jobId} />
                            {/* Download All as ZIP */}
                            <button
                                onClick={async () => {
                                    try {
                                        const resp = await fetch(`${PYTHON_API}/v1/stems/${jobId}/download_all`);
                                        if (!resp.ok) throw new Error('Download failed');
                                        const blob = await resp.blob();
                                        const a = document.createElement('a');
                                        a.href = URL.createObjectURL(blob);
                                        a.download = `${songTitle || 'stems'}_all_stems.zip`;
                                        document.body.appendChild(a);
                                        a.click();
                                        document.body.removeChild(a);
                                        URL.revokeObjectURL(a.href);
                                    } catch (err) { console.error('ZIP download failed:', err); }
                                }}
                                className="w-full py-2.5 px-4 rounded-xl border border-violet-500/30 bg-violet-50 dark:bg-violet-500/10 hover:bg-violet-100 dark:hover:bg-violet-500/20 flex items-center justify-center gap-2 transition-colors"
                            >
                                <Archive size={16} className="text-violet-600 dark:text-violet-400" />
                                <span className="text-sm font-bold text-violet-700 dark:text-violet-300">Download All Stems as ZIP</span>
                            </button>
                        </div>
                    )}
                </div>

                {status === 'idle' && (
                    <div className="px-6 py-4 border-t border-zinc-200 dark:border-white/10 bg-zinc-50 dark:bg-black/30">
                        <button onClick={startSeparation} disabled={available === false}
                            className="w-full py-3 rounded-xl bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 text-white font-bold text-sm shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed">
                            Extract Stems
                        </button>
                    </div>
                )}
            </div>
        </div>
    );

    return ReactDOM.createPortal(modalContent, document.body);
};
