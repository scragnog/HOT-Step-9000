import React, { useState, useEffect } from 'react';
import { ChevronDown, FlaskConical, Play, X } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { generateApi } from '../../services/api';
import { Song } from '../../types';

interface LayerAblationPanelProps {
    customMode: boolean;
    hasLoadedAdapters: boolean;
    onLayerScaleChange?: (slot: number, layer: number, scale: number) => void;
    onBulkLayerScalesChange?: (slot: number, layerScales: Record<number, number>) => void;
    // Ablation sweep
    onRunSweep?: () => void;
    isSweepRunning?: boolean;
    sweepProgress?: { current: number; total: number } | null;
    onCancelSweep?: () => void;
    isGenerating?: boolean;
    // Diff pin
    diffPinnedA?: Song | null;
    diffPinnedB?: Song | null;
    onClearDiffA?: () => void;
    onClearDiffB?: () => void;
}

export const LayerAblationPanel: React.FC<LayerAblationPanelProps> = ({
    customMode,
    hasLoadedAdapters,
    onLayerScaleChange,
    onBulkLayerScalesChange,
    onRunSweep,
    isSweepRunning,
    sweepProgress,
    onCancelSweep,
    isGenerating,
    diffPinnedA,
    diffPinnedB,
    onClearDiffA,
    onClearDiffB,
}) => {
    const { token } = useAuth();
    const [devMode, setDevMode] = useState(() => {
        try { return localStorage.getItem('ace_dev_mode') === 'true'; } catch { return false; }
    });
    const [isOpen, setIsOpen] = useState(false);
    const [selectedLayers, setSelectedLayers] = useState<Set<number>>(new Set());
    const [diffAmplify, setDiffAmplify] = useState(3.0);
    // Manual path fields — used as fallback when no song is pinned
    const [referencePath, setReferencePath] = useState('');
    const [ablatedPath, setAblatedPath] = useState('');
    const [diffResult, setDiffResult] = useState<{
        output_path: string;
        rms_energy: number;
        peak: number;
    } | null>(null);
    const [isDiffing, setIsDiffing] = useState(false);
    const [diffError, setDiffError] = useState<string | null>(null);

    useEffect(() => {
        try { localStorage.setItem('ace_dev_mode', String(devMode)); } catch { }
    }, [devMode]);

    if (!customMode) return null;

    const toggleLayer = (layer: number) => {
        setSelectedLayers(prev => {
            const next = new Set(prev);
            if (next.has(layer)) next.delete(layer); else next.add(layer);
            return next;
        });
    };

    const selectAll = () => setSelectedLayers(new Set(Array.from({ length: 24 }, (_, i) => i)));
    const selectNone = () => setSelectedLayers(new Set());

    // Resolve paths: prefer pinned song's audioUrl, fall back to manual input
    const resolvedReferencePath = diffPinnedA?.audioUrl ?? referencePath;
    const resolvedAblatedPath = diffPinnedB?.audioUrl ?? ablatedPath;

    const handleComputeDiff = async () => {
        if (!token || !resolvedReferencePath || !resolvedAblatedPath) return;
        setIsDiffing(true);
        setDiffError(null);
        setDiffResult(null);
        try {
            const result = await generateApi.computeAudioDiff({
                reference_path: resolvedReferencePath,
                ablated_path: resolvedAblatedPath,
                amplify: diffAmplify,
            }, token);
            setDiffResult(result);
        } catch (err: any) {
            setDiffError(err.message || 'Failed to compute diff');
        } finally {
            setIsDiffing(false);
        }
    };

    const handleZeroSelectedLayers = async () => {
        if (!token || selectedLayers.size === 0) return;
        const layerScales: Record<number, number> = {};
        for (const layer of selectedLayers) {
            layerScales[layer] = 0.0;
        }
        if (onBulkLayerScalesChange) {
            onBulkLayerScalesChange(0, layerScales);
        } else {
            try {
                await generateApi.setSlotLayerScales({ slot: 0, layer_scales: layerScales }, token);
            } catch (err) {
                console.error('Failed to set layer scales:', err);
            }
        }
    };

    const handleResetAllLayers = async () => {
        if (!token) return;
        const layerScales: Record<number, number> = {};
        for (let i = 0; i < 24; i++) {
            layerScales[i] = 1.0;
        }
        if (onBulkLayerScalesChange) {
            onBulkLayerScalesChange(0, layerScales);
        } else {
            try {
                await generateApi.setSlotLayerScales({ slot: 0, layer_scales: layerScales }, token);
            } catch (err) {
                console.error('Failed to set layer scales:', err);
            }
        }
    };

    const sweepDisabled = !hasLoadedAdapters || !!isGenerating || !!isSweepRunning;

    const Toggle: React.FC<{ on: boolean; onClick: () => void; disabled?: boolean }> = ({ on, onClick, disabled }) => (
        <button
            type="button"
            onClick={onClick}
            disabled={disabled}
            className={`w-10 h-5 rounded-full flex items-center transition-colors duration-200 px-0.5 border border-zinc-200 dark:border-white/5 ${on ? 'bg-purple-600' : 'bg-zinc-300 dark:bg-black/40'} ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
        >
            <div className={`w-4 h-4 rounded-full bg-white transform transition-transform duration-200 shadow-sm ${on ? 'translate-x-5' : 'translate-x-0'}`} />
        </button>
    );

    return (
        <div>
            {/* Dev Mode Toggle */}
            <div className="flex items-center justify-between px-1 py-1">
                <div>
                    <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400 flex items-center gap-1.5">
                        <FlaskConical size={12} />
                        Developer Mode
                    </span>
                    <p className="text-[10px] text-zinc-500">Layer ablation lab and audio diff tools for adapter analysis</p>
                </div>
                <Toggle on={devMode} onClick={() => setDevMode(!devMode)} />
            </div>

            {devMode && (
                <div>
                    <button
                        onClick={() => setIsOpen(!isOpen)}
                        className={`w-full flex items-center justify-between px-4 py-3 bg-white dark:bg-suno-card border border-zinc-200 dark:border-white/5 text-sm font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors ${isOpen ? 'rounded-t-xl rounded-b-none border-b-0' : 'rounded-xl'}`}
                    >
                        <div className="flex items-center gap-2">
                            <FlaskConical size={16} className="text-purple-500" />
                            <span>Layer Ablation Lab</span>
                            {isSweepRunning && sweepProgress && (
                                <span className="text-[10px] bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 px-1.5 py-0.5 rounded-full font-semibold">
                                    {sweepProgress.current}/{sweepProgress.total}
                                </span>
                            )}
                        </div>
                        <ChevronDown size={18} className={`text-purple-500 chevron-icon ${isOpen ? 'rotated' : ''}`} />
                    </button>

                    {isOpen && (
                        <div className="bg-white dark:bg-suno-card rounded-b-xl rounded-t-none border border-t-0 border-zinc-200 dark:border-white/5 p-4 space-y-4">
                            {!hasLoadedAdapters ? (
                                <div className="text-xs text-zinc-400 dark:text-zinc-600 text-center py-4">
                                    Load an adapter in Advanced Mode to use the ablation lab
                                </div>
                            ) : (
                                <>
                                    {/* Workflow explanation */}
                                    <div className="text-[10px] text-zinc-500 dark:text-zinc-400 leading-relaxed bg-purple-50 dark:bg-purple-900/10 rounded-lg px-3 py-2 border border-purple-200 dark:border-purple-800/30">
                                        <strong>How to use:</strong> Generate a reference track (full adapter, fixed seed). Then zero specific layers, regenerate with the same seed, and use Audio Diff to hear what changed. Higher RMS energy = bigger layer impact.
                                    </div>

                                    {/* Layer Selection Grid */}
                                    <div className="space-y-2">
                                        <div className="flex items-center justify-between">
                                            <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">Layers to Test</span>
                                            <div className="flex gap-2">
                                                <button onClick={selectAll} className="text-[10px] text-purple-500 hover:text-purple-400">All</button>
                                                <button onClick={selectNone} className="text-[10px] text-zinc-400 hover:text-zinc-300">None</button>
                                            </div>
                                        </div>
                                        <div className="grid grid-cols-8 gap-1">
                                            {Array.from({ length: 24 }, (_, i) => (
                                                <button
                                                    key={i}
                                                    onClick={() => toggleLayer(i)}
                                                    className={`px-1 py-1 rounded text-[10px] font-mono font-semibold border transition-colors ${selectedLayers.has(i)
                                                        ? 'bg-purple-500 text-white border-purple-600'
                                                        : 'bg-zinc-100 dark:bg-black/20 text-zinc-500 dark:text-zinc-400 border-zinc-200 dark:border-white/10 hover:border-purple-300'
                                                        }`}
                                                >
                                                    {i}
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Quick Actions */}
                                    <div className="flex gap-1.5">
                                        <button
                                            onClick={handleZeroSelectedLayers}
                                            disabled={selectedLayers.size === 0}
                                            className="flex-1 px-2 py-1.5 rounded-lg text-[10px] font-semibold bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 border border-red-200 dark:border-red-800/30 hover:bg-red-100 dark:hover:bg-red-900/30 disabled:opacity-40 transition-colors"
                                        >
                                            Zero Selected ({selectedLayers.size})
                                        </button>
                                        <button
                                            onClick={handleResetAllLayers}
                                            className="flex-1 px-2 py-1.5 rounded-lg text-[10px] font-semibold bg-emerald-50 dark:bg-emerald-900/20 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800/30 hover:bg-emerald-100 dark:hover:bg-emerald-900/30 transition-colors"
                                        >
                                            Reset All to 1.0
                                        </button>
                                    </div>

                                    {/* Ablation Sweep */}
                                    <div className="space-y-2 pt-2 border-t border-zinc-200 dark:border-white/5">
                                        <div className="flex items-center justify-between">
                                            <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">Ablation Sweep</span>
                                        </div>
                                        <p className="text-[10px] text-zinc-400 dark:text-zinc-500 leading-tight">
                                            Auto-generates 24 tracks, zeroing one layer at a time. Uses the current prompt + fixed seed. Tracks are named <em>Title - layer00</em> through <em>layer23</em>.
                                        </p>

                                        {isSweepRunning && sweepProgress ? (
                                            <div className="space-y-1.5">
                                                {/* Progress bar */}
                                                <div className="flex items-center justify-between">
                                                    <span className="text-[10px] text-purple-500 font-semibold">
                                                        Layer {sweepProgress.current} / {sweepProgress.total}
                                                    </span>
                                                    <span className="text-[10px] text-zinc-400">
                                                        {Math.round((sweepProgress.current / sweepProgress.total) * 100)}%
                                                    </span>
                                                </div>
                                                <div className="h-1.5 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full bg-gradient-to-r from-purple-500 to-indigo-500 rounded-full transition-all duration-500"
                                                        style={{ width: `${(sweepProgress.current / sweepProgress.total) * 100}%` }}
                                                    />
                                                </div>
                                                <button
                                                    onClick={onCancelSweep}
                                                    className="w-full px-2 py-1.5 rounded-lg text-[10px] font-semibold bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 border border-zinc-200 dark:border-white/10 hover:bg-red-50 dark:hover:bg-red-900/20 hover:text-red-600 dark:hover:text-red-400 hover:border-red-200 dark:hover:border-red-800/30 transition-colors"
                                                >
                                                    Cancel (finishes current gen)
                                                </button>
                                            </div>
                                        ) : (
                                            <button
                                                onClick={onRunSweep}
                                                disabled={sweepDisabled}
                                                title={!hasLoadedAdapters ? 'Load an adapter first' : isGenerating ? 'Wait for current generation to finish' : ''}
                                                className="w-full px-3 py-2 rounded-lg text-xs font-semibold bg-gradient-to-r from-purple-500 to-indigo-600 text-white hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
                                            >
                                                <Play size={12} fill="currentColor" />
                                                Run Ablation Sweep (24 layers)
                                            </button>
                                        )}
                                    </div>

                                    {/* Audio Diff Section */}
                                    <div className="space-y-2 pt-2 border-t border-zinc-200 dark:border-white/5">
                                        <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">Audio Diff</span>
                                        <p className="text-[10px] text-zinc-400 dark:text-zinc-500 leading-tight">
                                            Pin tracks using [A] / [B] buttons on song cards, or paste paths manually.
                                        </p>
                                        <div className="space-y-1.5">
                                            {/* Reference (A) */}
                                            {diffPinnedA ? (
                                                <div className="flex items-center gap-1.5 bg-blue-50 dark:bg-blue-900/10 border border-blue-200 dark:border-blue-800/30 rounded-lg px-2.5 py-1.5">
                                                    <span className="text-[10px] font-bold text-blue-600 dark:text-blue-400 flex-shrink-0">A</span>
                                                    <span className="text-[11px] text-zinc-700 dark:text-zinc-300 truncate flex-1">{diffPinnedA.title || 'Untitled'}</span>
                                                    <button onClick={onClearDiffA} className="text-zinc-400 hover:text-red-500 flex-shrink-0 transition-colors">
                                                        <X size={12} />
                                                    </button>
                                                </div>
                                            ) : (
                                                <input
                                                    type="text"
                                                    value={referencePath}
                                                    onChange={(e) => setReferencePath(e.target.value)}
                                                    placeholder="[A] Reference audio path (or pin from song list)"
                                                    className="w-full bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg px-3 py-1.5 text-[11px] text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-purple-500"
                                                />
                                            )}

                                            {/* Ablated (B) */}
                                            {diffPinnedB ? (
                                                <div className="flex items-center gap-1.5 bg-orange-50 dark:bg-orange-900/10 border border-orange-200 dark:border-orange-800/30 rounded-lg px-2.5 py-1.5">
                                                    <span className="text-[10px] font-bold text-orange-600 dark:text-orange-400 flex-shrink-0">B</span>
                                                    <span className="text-[11px] text-zinc-700 dark:text-zinc-300 truncate flex-1">{diffPinnedB.title || 'Untitled'}</span>
                                                    <button onClick={onClearDiffB} className="text-zinc-400 hover:text-red-500 flex-shrink-0 transition-colors">
                                                        <X size={12} />
                                                    </button>
                                                </div>
                                            ) : (
                                                <input
                                                    type="text"
                                                    value={ablatedPath}
                                                    onChange={(e) => setAblatedPath(e.target.value)}
                                                    placeholder="[B] Ablated audio path (or pin from song list)"
                                                    className="w-full bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg px-3 py-1.5 text-[11px] text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-purple-500"
                                                />
                                            )}

                                            <div className="flex items-center gap-2">
                                                <label className="text-[10px] text-zinc-500 whitespace-nowrap">Amplify:</label>
                                                <input
                                                    type="range"
                                                    min={1}
                                                    max={20}
                                                    step={0.5}
                                                    value={diffAmplify}
                                                    onChange={(e) => setDiffAmplify(parseFloat(e.target.value))}
                                                    className="flex-1 accent-purple-500"
                                                />
                                                <span className="text-[10px] text-zinc-500 font-mono w-8">{diffAmplify}x</span>
                                            </div>
                                            <button
                                                onClick={handleComputeDiff}
                                                disabled={isDiffing || !resolvedReferencePath || !resolvedAblatedPath}
                                                className="w-full px-3 py-2 rounded-lg text-xs font-semibold bg-gradient-to-r from-purple-500 to-indigo-600 text-white hover:brightness-110 disabled:opacity-40 transition-all flex items-center justify-center gap-2"
                                            >
                                                {isDiffing ? (
                                                    <><span className="inline-block w-3 h-3 border-2 border-white/60 border-t-transparent rounded-full animate-spin" />Computing...</>
                                                ) : (
                                                    <>🔬 Compute Diff</>
                                                )}
                                            </button>
                                        </div>

                                        {/* Diff Error */}
                                        {diffError && (
                                            <div className="text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 px-2 py-1 rounded">
                                                {diffError}
                                            </div>
                                        )}

                                        {/* Diff Results */}
                                        {diffResult && (
                                            <div className="bg-purple-50 dark:bg-purple-900/10 rounded-lg p-3 space-y-2 border border-purple-200 dark:border-purple-800/30">
                                                <div className="flex items-center justify-between">
                                                    <span className="text-xs font-semibold text-purple-600 dark:text-purple-400">Diff Result</span>
                                                    <span className="text-[10px] text-zinc-500">{(diffResult as any).duration_match ? '✓ Length match' : '⚠ Length mismatch'}</span>
                                                </div>
                                                <div className="grid grid-cols-2 gap-2 text-[10px]">
                                                    <div>
                                                        <span className="text-zinc-500">RMS Energy:</span>
                                                        <span className={`ml-1 font-bold ${diffResult.rms_energy > 0.05 ? 'text-red-500' : diffResult.rms_energy > 0.01 ? 'text-amber-500' : 'text-emerald-500'}`}>
                                                            {diffResult.rms_energy.toFixed(4)}
                                                        </span>
                                                    </div>
                                                    <div>
                                                        <span className="text-zinc-500">Peak:</span>
                                                        <span className={`ml-1 font-bold ${diffResult.peak > 0.3 ? 'text-red-500' : diffResult.peak > 0.1 ? 'text-amber-500' : 'text-emerald-500'}`}>
                                                            {diffResult.peak.toFixed(4)}
                                                        </span>
                                                    </div>
                                                </div>
                                                {/* Impact bar */}
                                                <div className="h-2 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full bg-gradient-to-r from-emerald-500 via-amber-500 to-red-500 rounded-full transition-all"
                                                        style={{ width: `${Math.min(diffResult.rms_energy * 1000, 100)}%` }}
                                                    />
                                                </div>
                                                <p className="text-[9px] text-zinc-400 dark:text-zinc-500 truncate" title={diffResult.output_path}>
                                                    Saved: {diffResult.output_path}
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                </>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default LayerAblationPanel;
