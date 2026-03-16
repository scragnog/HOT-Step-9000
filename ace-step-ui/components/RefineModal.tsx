import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom';
import { X, Wand2, Loader2, Play, Pause, ChevronDown } from 'lucide-react';
import { Song } from '../types';

// ---- Global open hook ----
type OpenFn = (song: Song) => void;
let _globalOpen: OpenFn | null = null;

/** Call from anywhere to open the refine modal for a given song. */
export function openRefineModal(song: Song) {
    _globalOpen?.(song);
}

// ---- Constants ----
const PASS_OPTIONS = [
    { value: 1, label: '1 Pass', desc: 'Quick cleanup — subtle improvement' },
    { value: 2, label: '2 Passes', desc: 'Balanced — best quality/speed ratio' },
    { value: 3, label: '3 Passes', desc: 'Deep refine — diminishing returns' },
];

const STRENGTH_PRESETS = [
    { value: 0.15, label: 'Light', desc: 'Subtle polish' },
    { value: 0.30, label: 'Medium', desc: 'Recommended' },
    { value: 0.50, label: 'Strong', desc: 'More restructuring' },
];

// ---- Main Modal ----
export const RefineModal: React.FC = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [song, setSong] = useState<Song | null>(null);
    const [passes, setPasses] = useState(1);
    const [strength, setStrength] = useState(0.30);
    const [status, setStatus] = useState<'idle' | 'refining' | 'complete' | 'error'>('idle');
    const [progress, setProgress] = useState(0);
    const [stage, setStage] = useState('');
    const [error, setError] = useState('');
    const [resultJobId, setResultJobId] = useState('');
    const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

    // Register global open function
    useEffect(() => {
        _globalOpen = (s: Song) => {
            setSong(s);
            setPasses(1);
            setStrength(0.30);
            setStatus('idle');
            setProgress(0);
            setStage('');
            setError('');
            setResultJobId('');
            setIsOpen(true);
        };
        return () => { _globalOpen = null; };
    }, []);

    // Cleanup polling on unmount
    useEffect(() => {
        return () => { if (pollRef.current) clearInterval(pollRef.current); };
    }, []);

    const onClose = useCallback(() => {
        if (pollRef.current) clearInterval(pollRef.current);
        setIsOpen(false);
    }, []);

    const startRefine = async () => {
        if (!song) return;
        setStatus('refining');
        setProgress(0);
        setStage('Submitting refinement job...');
        setError('');

        try {
            // Build generation request from original song params
            const gp = song.generationParams || {};
            const body: Record<string, any> = {
                customMode: true,
                title: `${song.title} (Refined)`,
                lyrics: gp.lyrics || song.lyrics || '',
                style: gp.style || gp.prompt || song.style || '',
                instrumental: gp.instrumental || false,
                vocalLanguage: gp.vocalLanguage || 'en',
                duration: gp.duration || undefined,
                bpm: gp.bpm || undefined,
                keyScale: gp.keyScale || '',
                timeSignature: gp.timeSignature || '',
                inferenceSteps: gp.inferenceSteps || 32,
                guidanceScale: gp.guidanceScale || 7.0,
                batchSize: 1,
                randomSeed: true,
                thinking: gp.thinking || false,
                audioFormat: gp.audioFormat || 'flac',
                inferMethod: gp.inferMethod || 'ode',
                scheduler: gp.scheduler || 'linear',
                shift: gp.shift,
                taskType: gp.taskType || 'text2music',
                // Refinement fields
                refinePasses: passes,
                refineStrength: strength,
                // Preserve original generation settings
                useAdg: gp.useAdg || false,
                guidanceMode: gp.guidanceMode || '',
                cfgIntervalStart: gp.cfgIntervalStart || 0.0,
                cfgIntervalEnd: gp.cfgIntervalEnd || 1.0,
                enableNormalization: gp.enableNormalization !== false,
                normalizationDb: gp.normalizationDb ?? -1.0,
                autoMaster: gp.autoMaster !== false,
                latentShift: gp.latentShift || 0.0,
                latentRescale: gp.latentRescale || 1.0,
                // LM params
                lmTemperature: gp.lmTemperature,
                lmCfgScale: gp.lmCfgScale,
                lmTopK: gp.lmTopK,
                lmTopP: gp.lmTopP,
                lmNegativePrompt: gp.lmNegativePrompt,
                // PAG
                usePag: gp.usePag || false,
                pagStart: gp.pagStart,
                pagEnd: gp.pagEnd,
                pagScale: gp.pagScale,
                // Audio codes (reuse from original if available)
                audioCodes: gp.audioCodes || '',
                // Steering
                steeringEnabled: gp.steeringEnabled || false,
                steeringLoaded: gp.steeringLoaded || [],
                steeringAlphas: gp.steeringAlphas || {},
                // DiT model
                ditModel: gp.ditModel || song.ditModel,
            };

            // Use original seed if available (for reproducibility + refinement)
            if (gp.seed && gp.seed > 0) {
                body.randomSeed = false;
                body.seed = gp.seed;
            }

            const resp = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({ error: resp.statusText }));
                throw new Error(errData.error || `Server error ${resp.status}`);
            }

            const { jobId } = await resp.json();
            setResultJobId(jobId);
            setStage('Refinement in progress...');

            // Poll for completion
            pollRef.current = setInterval(async () => {
                try {
                    const statusResp = await fetch(`/api/generate/status/${jobId}`);
                    if (!statusResp.ok) return;
                    const statusData = await statusResp.json();

                    if (statusData.progress) {
                        setProgress(statusData.progress);
                    }
                    if (statusData.stage) {
                        setStage(statusData.stage);
                    }

                    if (statusData.status === 'succeeded') {
                        if (pollRef.current) clearInterval(pollRef.current);
                        setStatus('complete');
                        setProgress(1);
                        setStage('Refinement complete!');
                        // Reload the page to see the new song in the library
                        setTimeout(() => {
                            window.location.reload();
                        }, 1500);
                    } else if (statusData.status === 'failed') {
                        if (pollRef.current) clearInterval(pollRef.current);
                        setStatus('error');
                        setError(statusData.error || 'Refinement failed');
                    }
                } catch {
                    // Ignore transient poll errors
                }
            }, 2000);
        } catch (err) {
            setStatus('error');
            setError(err instanceof Error ? err.message : 'Failed to start refinement');
        }
    };

    if (!isOpen || !song) return null;

    const modalContent = (
        <div
            className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 dark:bg-black/80 backdrop-blur-sm p-4"
            onClick={(e) => e.target === e.currentTarget && status !== 'refining' && onClose()}
        >
            <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-white/10 rounded-2xl w-full max-w-lg shadow-2xl animate-in fade-in zoom-in-95 duration-200 max-h-[85vh] flex flex-col overflow-hidden">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-white/10">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center shadow-lg">
                            <Wand2 size={20} className="text-white" />
                        </div>
                        <div>
                            <h2 className="text-lg font-bold text-zinc-900 dark:text-white">Refine Audio</h2>
                            <p className="text-xs text-zinc-500 truncate max-w-[250px]">{song.title}</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        disabled={status === 'refining'}
                        className="p-2 hover:bg-zinc-100 dark:hover:bg-white/10 rounded-lg transition-colors disabled:opacity-40"
                    >
                        <X size={20} className="text-zinc-500" />
                    </button>
                </div>

                {/* Body */}
                <div className="flex-1 overflow-y-auto p-6 space-y-5">
                    {status === 'idle' && (
                        <>
                            {/* Explanation */}
                            <div className="p-3 rounded-xl bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/20">
                                <p className="text-xs text-amber-800 dark:text-amber-300 leading-relaxed">
                                    Refinement re-denoises your audio to clean up fine detail — reducing fizziness and metallic artifacts 
                                    while preserving the melody, rhythm, and lyrics. A new song will be created for A/B comparison.
                                </p>
                            </div>

                            {/* Passes */}
                            <div className="space-y-2">
                                <label className="text-sm font-bold text-zinc-700 dark:text-zinc-300 uppercase tracking-wider">
                                    Refinement Passes
                                </label>
                                <div className="grid grid-cols-3 gap-2">
                                    {PASS_OPTIONS.map((opt) => (
                                        <button
                                            key={opt.value}
                                            onClick={() => setPasses(opt.value)}
                                            className={`p-3 rounded-xl border text-left transition-all ${passes === opt.value
                                                ? 'border-amber-500 bg-amber-50 dark:bg-amber-500/10 ring-1 ring-amber-500/50'
                                                : 'border-zinc-200 dark:border-white/10 hover:border-zinc-300 dark:hover:border-white/20'
                                            }`}
                                        >
                                            <div className="text-sm font-bold text-zinc-900 dark:text-white">{opt.label}</div>
                                            <div className="text-[10px] text-zinc-500 dark:text-zinc-400 mt-0.5">{opt.desc}</div>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Strength */}
                            <div className="space-y-2">
                                <div className="flex items-center justify-between">
                                    <label className="text-sm font-bold text-zinc-700 dark:text-zinc-300 uppercase tracking-wider">
                                        Refine Strength
                                    </label>
                                    <span className="text-xs font-mono text-amber-600 dark:text-amber-400">{(strength * 100).toFixed(0)}%</span>
                                </div>
                                <div className="grid grid-cols-3 gap-2">
                                    {STRENGTH_PRESETS.map((preset) => (
                                        <button
                                            key={preset.value}
                                            onClick={() => setStrength(preset.value)}
                                            className={`p-2 rounded-lg border text-center transition-all ${Math.abs(strength - preset.value) < 0.01
                                                ? 'border-amber-500 bg-amber-50 dark:bg-amber-500/10 ring-1 ring-amber-500/50'
                                                : 'border-zinc-200 dark:border-white/10 hover:border-zinc-300 dark:hover:border-white/20'
                                            }`}
                                        >
                                            <div className="text-xs font-bold text-zinc-900 dark:text-white">{preset.label}</div>
                                            <div className="text-[9px] text-zinc-500 dark:text-zinc-400">{preset.desc}</div>
                                        </button>
                                    ))}
                                </div>
                                <input
                                    type="range"
                                    min={5} max={70} step={1}
                                    value={strength * 100}
                                    onChange={(e) => setStrength(parseInt(e.target.value) / 100)}
                                    className="w-full h-1.5 accent-amber-500 cursor-pointer"
                                />
                                <div className="flex justify-between text-[9px] text-zinc-400 font-mono">
                                    <span>5% (subtle)</span>
                                    <span>70% (aggressive)</span>
                                </div>
                            </div>

                            {/* Time estimate */}
                            <div className="text-center text-xs text-zinc-500 dark:text-zinc-400">
                                Estimated extra time: ~{Math.round(strength * 100 * passes)}% of original generation
                            </div>
                        </>
                    )}

                    {status === 'refining' && (
                        <div className="space-y-4 py-4">
                            <div className="flex items-center justify-center gap-3">
                                <Loader2 size={24} className="text-amber-500 animate-spin" />
                                <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">{stage}</span>
                            </div>
                            <div className="w-full h-3 bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-amber-500 to-orange-500 rounded-full transition-all duration-500 ease-out"
                                    style={{ width: `${Math.max(progress * 100, 2)}%` }}
                                />
                            </div>
                            <p className="text-xs text-zinc-500 text-center font-mono">{(progress * 100).toFixed(0)}%</p>
                            <p className="text-[10px] text-zinc-400 text-center">
                                {passes} pass{passes > 1 ? 'es' : ''} × {(strength * 100).toFixed(0)}% strength
                            </p>
                        </div>
                    )}

                    {status === 'error' && (
                        <div className="p-4 rounded-xl bg-red-50 dark:bg-red-500/10 border border-red-200 dark:border-red-500/30">
                            <p className="text-sm font-medium text-red-700 dark:text-red-300">Refinement failed</p>
                            <p className="text-xs text-red-500 dark:text-red-400 mt-1">{error}</p>
                            <button
                                onClick={() => setStatus('idle')}
                                className="mt-3 px-4 py-1.5 text-xs font-bold rounded-lg bg-red-100 dark:bg-red-500/20 text-red-700 dark:text-red-300 hover:bg-red-200 dark:hover:bg-red-500/30 transition-colors"
                            >
                                Try Again
                            </button>
                        </div>
                    )}

                    {status === 'complete' && (
                        <div className="space-y-4 py-4 text-center">
                            <div className="w-16 h-16 mx-auto rounded-full bg-gradient-to-br from-green-500/20 to-emerald-500/20 border border-green-500/30 flex items-center justify-center">
                                <Wand2 size={28} className="text-green-500" />
                            </div>
                            <div>
                                <p className="text-lg font-bold text-zinc-900 dark:text-white">Refinement Complete!</p>
                                <p className="text-sm text-zinc-500 mt-1">
                                    "{song.title} (Refined)" has been added to your library.
                                </p>
                            </div>
                            <p className="text-xs text-zinc-400">Reloading library...</p>
                        </div>
                    )}
                </div>

                {/* Footer */}
                {status === 'idle' && (
                    <div className="px-6 py-4 border-t border-zinc-200 dark:border-white/10 bg-zinc-50 dark:bg-black/30">
                        <button
                            onClick={startRefine}
                            className="w-full py-3 rounded-xl bg-gradient-to-r from-amber-500 to-orange-600 hover:from-amber-600 hover:to-orange-700 text-white font-bold text-sm shadow-lg hover:shadow-xl transition-all"
                        >
                            Refine Audio
                        </button>
                    </div>
                )}
            </div>
        </div>
    );

    return ReactDOM.createPortal(modalContent, document.body);
};
