import React, { useState, useEffect } from 'react';
import { Sliders, ChevronDown, FolderSearch } from 'lucide-react';
import { useI18n } from '../../context/I18nContext';
import { useAuth } from '../../context/AuthContext';
import { generateApi } from '../../services/api';
import { EditableSlider } from '../EditableSlider';
import { FileBrowserModal } from '../FileBrowserModal';

export interface AdapterSlot {
    slot: number;
    type: 'peft_lora' | 'lokr';
    name: string;
    path: string;
    scale: number;
    group_scales: Record<string, number>;
    layer_scales?: Record<number, number>;
}

export interface AdapterFile {
    path: string;
    name: string;
    type: string;
    size: number;
}

interface AdaptersAccordionProps {
    // Mode/Visibility
    customMode: boolean;
    isOpen: boolean;
    onToggle: () => void;

    // Basic Mode
    advancedAdapters: boolean;
    onAdvancedAdaptersChange: (val: boolean) => void;
    loraPath: string;
    onLoraPathChange: (val: string) => void;
    loraLoaded: boolean;
    isLoraLoading: boolean;
    onLoraToggle: () => void;
    loraError: string | null;
    loraScale: number;
    onLoraScaleChange: (val: number) => void;

    // Advanced Mode (Multi-slot)
    adapterFolder: string;
    onAdapterFolderChange: (val: string) => void;
    onScanFolder: () => void;
    adapterFiles: AdapterFile[];
    adapterSlots: AdapterSlot[];
    loadingAdapterPath: string | null;
    adapterLoadingMessage: string | null;
    expandedSlots: Set<number>;
    setExpandedSlots: React.Dispatch<React.SetStateAction<Set<number>>>;

    // Advanced Handlers
    onLoadSlot: (path: string) => void;
    onUnloadSlot: (slotNum: number) => void;
    onSlotScaleChange: (slotNum: number, scale: number) => void;
    onSlotGroupScaleChange: (slotNum: number, group: string, scale: number) => void;
    onSlotLayerScaleChange?: (slotNum: number, layer: number, scale: number) => void;
    temporalScheduleActive?: boolean;
    onTemporalSchedulePreset?: (preset: 'switch' | 'verse-chorus' | null) => void;

    // Global Scale Overrides
    globalScaleOverrideEnabled?: boolean;
    onGlobalOverrideToggle?: (enabled: boolean) => void;
    globalOverallScale?: number;
    onGlobalOverallScaleChange?: (scale: number) => void;
    globalGroupScales?: { self_attn: number; cross_attn: number; mlp: number };
    onGlobalGroupScaleChange?: (group: string, value: number) => void;
    /** When true, skip the accordion header — used inside DrawerContainers */
    embedded?: boolean;
}

export const AdaptersAccordion: React.FC<AdaptersAccordionProps> = ({
    customMode,
    isOpen,
    onToggle,
    advancedAdapters,
    onAdvancedAdaptersChange,
    loraPath,
    onLoraPathChange,
    loraLoaded,
    isLoraLoading,
    onLoraToggle,
    loraError,
    loraScale,
    onLoraScaleChange,
    adapterFolder,
    onAdapterFolderChange,
    onScanFolder,
    adapterFiles,
    adapterSlots,
    loadingAdapterPath,
    adapterLoadingMessage,
    expandedSlots,
    setExpandedSlots,
    onLoadSlot,
    onUnloadSlot,
    onSlotScaleChange,
    onSlotGroupScaleChange,
    onSlotLayerScaleChange,
    temporalScheduleActive,
    onTemporalSchedulePreset,
    globalScaleOverrideEnabled,
    onGlobalOverrideToggle,
    globalOverallScale,
    onGlobalOverallScaleChange,
    globalGroupScales,
    onGlobalGroupScaleChange,
    embedded = false,
}) => {
    const { t } = useI18n();
    const { token } = useAuth();
    const [browsedFiles, setBrowsedFiles] = useState<AdapterFile[]>([]);
    const [showBrowse, setShowBrowse] = useState(false);
    const [expandedLayers, setExpandedLayers] = useState<Set<number>>(new Set());

    // Pending drag state for raw range inputs (Role Blend + individual layers)
    // Key format: `${slotNum}-${roleKey}` or `${slotNum}-layer-${layerIdx}`
    const [pendingRoleBlend, setPendingRoleBlend] = useState<Record<string, number>>({});
    const [pendingLayerScale, setPendingLayerScale] = useState<Record<string, number>>({});

    // In-browser file browser modal state
    const [fileBrowserOpen, setFileBrowserOpen] = useState(false);
    const [fileBrowserMode, setFileBrowserMode] = useState<'file' | 'folder'>('file');

    // Detected adapter type from file metadata
    const [detectedType, setDetectedType] = useState<'lokr' | 'lora' | 'unknown' | null>(null);

    // Detect adapter type when path changes
    useEffect(() => {
        if (!loraPath.trim() || !token) {
            setDetectedType(null);
            return;
        }
        let cancelled = false;
        generateApi.detectAdapterType(loraPath.trim(), token)
            .then(result => { if (!cancelled) setDetectedType(result.type); })
            .catch(() => { if (!cancelled) setDetectedType('unknown'); });
        return () => { cancelled = true; };
    }, [loraPath, token]);

    const Toggle: React.FC<{ on: boolean; onClick: () => void; disabled?: boolean }> = ({ on, onClick, disabled }) => (
        <button
            type="button"
            onClick={onClick}
            disabled={disabled}
            className={`w-10 h-5 rounded-full flex items-center transition-colors duration-200 px-0.5 border border-zinc-200 dark:border-white/5 ${on ? 'bg-pink-600' : 'bg-zinc-300 dark:bg-black/40'} ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
        >
            <div className={`w-4 h-4 rounded-full bg-white transform transition-transform duration-200 shadow-sm ${on ? 'translate-x-5' : 'translate-x-0'}`} />
        </button>
    );

    if (!customMode) return null;

    const content = (
                <div className={embedded ? "space-y-4" : "bg-white dark:bg-suno-card rounded-b-xl rounded-t-none border border-t-0 border-zinc-200 dark:border-white/5 p-4 space-y-4"}>
                    {/* Advanced Toggle */}
                    <div className="flex items-center justify-between py-1">
                        <div>
                            <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">Advanced (Multi-Adapter)</span>
                            <p className="text-[10px] text-zinc-500">Multi-slot adapter loading with per-slot and per-layer scale control</p>
                        </div>
                        <Toggle on={advancedAdapters} onClick={() => onAdvancedAdaptersChange(!advancedAdapters)} />
                    </div>

                    {!advancedAdapters ? (
                        /* BASIC MODE */
                        <>
                            {/* Adapter file selection */}
                            <div className="space-y-2">
                                <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">Adapter Path</label>
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={loraPath}
                                        onChange={(e) => onLoraPathChange(e.target.value)}
                                        placeholder="Paste path to .safetensors file or adapter folder"
                                        className="flex-1 bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg px-3 py-2 text-xs text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-pink-500"
                                    />
                                    <button
                                        onClick={() => { setFileBrowserMode('file'); setFileBrowserOpen(true); }}
                                        title="Browse for adapter files"
                                        className="px-3 py-2 rounded-lg text-xs font-semibold bg-pink-50 dark:bg-pink-900/20 text-pink-600 dark:text-pink-400 hover:bg-pink-100 dark:hover:bg-pink-900/30 transition-colors flex items-center gap-1.5 flex-shrink-0"
                                    >
                                        <FolderSearch size={14} />
                                        Browse
                                    </button>
                                    {loraPath && (
                                        <button
                                            onClick={() => onLoraPathChange('')}
                                            className="px-2 py-2 rounded-lg text-xs text-zinc-400 hover:text-red-500 dark:hover:text-red-400 transition-colors flex-shrink-0"
                                            title="Clear selection"
                                        >
                                            ✕
                                        </button>
                                    )}
                                </div>
                                {loraPath && (() => {
                                    const fileName = loraPath.split(/[\\/]/).pop() || '';
                                    const tag = detectedType === 'lokr' ? 'LOKR' : detectedType === 'lora' ? 'LORA' : '...';
                                    const tagColor = detectedType === 'lokr'
                                        ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'
                                        : 'bg-pink-100 dark:bg-pink-900/30 text-pink-600 dark:text-pink-400';
                                    return (
                                        <div className="flex items-center gap-2">
                                            <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${tagColor}`}>
                                                {tag}
                                            </span>
                                            <span className="text-[10px] text-zinc-500 truncate" title={loraPath}>
                                                {fileName}
                                            </span>
                                        </div>
                                    );
                                })()}
                            </div>

                            {/* LoRA Load/Unload Toggle */}
                            <div className="space-y-2">
                                <div className="flex items-center justify-between py-2 border-t border-zinc-100 dark:border-white/5">
                                    <div className="flex items-center gap-2">
                                        <div className={`w-2 h-2 rounded-full ${loraLoaded ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
                                        <span className={`text-xs font-medium ${loraLoaded ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                                            {loraLoaded ? t('loraLoaded') : t('loraUnloaded')}
                                        </span>
                                    </div>
                                    <button
                                        onClick={onLoraToggle}
                                        disabled={!loraPath.trim() || isLoraLoading}
                                        className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed ${loraLoaded
                                            ? 'bg-gradient-to-r from-green-500 to-emerald-600 text-white shadow-lg shadow-green-500/20 hover:from-green-600 hover:to-emerald-700'
                                            : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-700'
                                            }`}
                                    >
                                        {isLoraLoading ? '...' : (loraLoaded ? t('loraUnload') : t('loraLoad'))}
                                    </button>
                                </div>
                                {loraError && (
                                    <div className="text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 px-2 py-1 rounded">
                                        {loraError}
                                    </div>
                                )}
                            </div>

                            {/* LoRA Scale Slider */}
                            <div className={!loraLoaded ? 'opacity-40 pointer-events-none' : ''}>
                                <EditableSlider
                                    label={t('loraScale')}
                                    value={loraScale}
                                    min={0}
                                    max={2}
                                    step={0.05}
                                    onChange={onLoraScaleChange}
                                    formatDisplay={(val) => val.toFixed(2)}
                                    helpText={t('loraScaleDescription')}
                                />
                            </div>
                        </>
                    ) : (
                        /* ADVANCED MODE */
                        <>
                            <div className="space-y-2">
                                <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">Adapter Folder</label>
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={adapterFolder}
                                        onChange={(e) => onAdapterFolderChange(e.target.value)}
                                        placeholder="./lokr_output"
                                        className="flex-1 bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg px-3 py-2 text-xs text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-pink-500"
                                    />
                                    <button
                                        onClick={async () => {
                                            const folder = adapterFolder.trim() || './lokr_output';
                                            onAdapterFolderChange(folder);
                                            onScanFolder();
                                        }}
                                        disabled={!adapterFolder.trim() && false}
                                        className="px-3 py-2 rounded-lg text-xs font-semibold bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-700 disabled:opacity-40 transition-colors"
                                    >
                                        Scan
                                    </button>
                                    <button
                                        onClick={() => { setFileBrowserMode('folder'); setFileBrowserOpen(true); }}
                                        className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold bg-pink-50 dark:bg-pink-900/20 text-pink-600 dark:text-pink-400 hover:bg-pink-100 dark:hover:bg-pink-900/30 transition-colors"
                                        title="Browse for adapter folder"
                                    >
                                        <FolderSearch size={14} />
                                        Browse
                                    </button>
                                </div>
                                {showBrowse && browsedFiles.length > 0 && (
                                    <div className="space-y-1">
                                        <div className="flex items-center justify-between">
                                            <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
                                                Found ({browsedFiles.length})
                                            </label>
                                            <button
                                                onClick={() => setShowBrowse(false)}
                                                className="text-[10px] text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300"
                                            >
                                                Hide
                                            </button>
                                        </div>
                                        <div className="max-h-28 overflow-y-auto space-y-1">
                                            {browsedFiles.map((file) => {
                                                const isAlreadyLoaded = adapterSlots.some(s => s.path === file.path);
                                                return (
                                                    <button
                                                        key={file.path}
                                                        onClick={() => {
                                                            if (!isAlreadyLoaded) onLoadSlot(file.path);
                                                            setShowBrowse(false);
                                                        }}
                                                        disabled={isAlreadyLoaded}
                                                        className={`w-full flex items-center justify-between bg-zinc-50 dark:bg-black/20 rounded-lg px-3 py-2 text-left transition-colors ${isAlreadyLoaded ? 'opacity-50 cursor-not-allowed' : 'hover:bg-pink-50 dark:hover:bg-pink-900/10 cursor-pointer'}`}
                                                    >
                                                        <div className="flex items-center gap-2 min-w-0">
                                                            <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${file.type === 'lora' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400' : 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'}`}>
                                                                {file.type.toUpperCase()}
                                                            </span>
                                                            <span className="text-xs text-zinc-700 dark:text-zinc-300 truncate">{file.name}</span>
                                                        </div>
                                                        <span className="text-[10px] text-zinc-400 flex-shrink-0 ml-2">
                                                            {isAlreadyLoaded ? '✓ Loaded' : `${(file.size / 1024 / 1024).toFixed(1)}MB`}
                                                        </span>
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* File list */}
                            {adapterFiles.length > 0 && (
                                <div className="space-y-1">
                                    <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">Available Adapters ({adapterFiles.length})</label>
                                    <div className="max-h-32 overflow-y-auto space-y-1">
                                        {adapterFiles.map((file) => {
                                            const isAlreadyLoaded = adapterSlots.some(s => s.path === file.path);
                                            return (
                                                <div key={file.path} className="flex items-center justify-between bg-zinc-50 dark:bg-black/20 rounded-lg px-3 py-2">
                                                    <div className="flex items-center gap-2 min-w-0">
                                                        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${file.type === 'lora' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400' : 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'}`}>
                                                            {file.type.toUpperCase()}
                                                        </span>
                                                        <span className="text-xs text-zinc-700 dark:text-zinc-300 truncate">{file.name}</span>
                                                        <span className="text-[10px] text-zinc-400">{(file.size / 1024 / 1024).toFixed(1)}MB</span>
                                                    </div>
                                                    {isAlreadyLoaded ? (
                                                        <span className="px-2 py-1 rounded text-[10px] font-semibold text-emerald-500 dark:text-emerald-400 bg-emerald-500/10">
                                                            ✓ Loaded
                                                        </span>
                                                    ) : (
                                                        <button
                                                            onClick={() => onLoadSlot(file.path)}
                                                            disabled={isLoraLoading || adapterSlots.length >= 4}
                                                            className="px-2 py-1 rounded text-[10px] font-semibold bg-pink-500/10 text-pink-600 dark:text-pink-400 hover:bg-pink-500/20 disabled:opacity-40 transition-colors flex items-center gap-1"
                                                        >
                                                            {loadingAdapterPath === file.path ? (
                                                                <><span className="inline-block w-3 h-3 border-2 border-pink-400 border-t-transparent rounded-full animate-spin" /> Loading…</>
                                                            ) : isLoraLoading ? 'Wait…' : 'Load'}
                                                        </button>
                                                    )}
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}

                            {/* Adapter loading status */}
                            {adapterLoadingMessage && (
                                <div className="flex items-center gap-2 text-xs text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 px-2 py-1.5 rounded">
                                    <span className="inline-block w-3 h-3 border-2 border-amber-400 border-t-transparent rounded-full animate-spin" />
                                    {adapterLoadingMessage}
                                </div>
                            )}

                            {/* Error display */}
                            {loraError && (
                                <div className="text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 px-2 py-1 rounded">
                                    {loraError}
                                </div>
                            )}

                            {/* Loaded adapter slots */}
                            {adapterSlots.length > 0 && (
                                <div className="space-y-2">
                                    <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">Loaded Adapters ({adapterSlots.length}/4)</label>
                                    {(() => {
                                        const totalScale = globalScaleOverrideEnabled && globalOverallScale !== undefined
                                            ? adapterSlots.length * globalOverallScale
                                            : adapterSlots.reduce((sum, s) => sum + s.scale, 0);
                                        return totalScale > 1.0 ? (
                                            <div className="flex items-start gap-1.5 text-[11px] text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 px-2 py-1.5 rounded">
                                                <span className="mt-0.5">⚠️</span>
                                                <span>Combined adapter strength is <strong>{totalScale.toFixed(2)}</strong>{globalScaleOverrideEnabled ? ' (global override)' : ''} — values above 1.0 may produce unexpected or distorted output.</span>
                                            </div>
                                        ) : null;
                                    })()}
                                    {adapterSlots.map((slot) => (
                                        <div key={slot.slot} className="bg-zinc-50 dark:bg-black/20 rounded-lg p-3 space-y-2 border border-zinc-200 dark:border-white/5">
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center gap-2">
                                                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                                                    <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">{slot.name}</span>
                                                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${slot.type === 'peft_lora' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400' : 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'}`}>
                                                        {slot.type === 'peft_lora' ? 'LoRA' : 'LoKr'}
                                                    </span>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <button
                                                        onClick={() => setExpandedSlots(prev => {
                                                            const next = new Set(prev);
                                                            if (next.has(slot.slot)) next.delete(slot.slot); else next.add(slot.slot);
                                                            return next;
                                                        })}
                                                        className="text-[10px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
                                                    >
                                                        {expandedSlots.has(slot.slot) ? '▼' : '▶'} Groups
                                                    </button>
                                                    <button
                                                        onClick={() => setExpandedLayers(prev => {
                                                            const next = new Set(prev);
                                                            if (next.has(slot.slot)) next.delete(slot.slot); else next.add(slot.slot);
                                                            return next;
                                                        })}
                                                        className="text-[10px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
                                                    >
                                                        {expandedLayers.has(slot.slot) ? '▼' : '▶'} Layers
                                                    </button>
                                                    <button
                                                        onClick={() => onUnloadSlot(slot.slot)}
                                                        disabled={isLoraLoading}
                                                        className="px-2 py-1 rounded text-[10px] font-semibold text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 disabled:opacity-40 transition-colors"
                                                    >
                                                        Unload
                                                    </button>
                                                </div>
                                            </div>

                                            {/* Overall scale slider */}
                                            <div className={globalScaleOverrideEnabled ? 'opacity-40 pointer-events-none' : ''}>
                                            <EditableSlider
                                                label={`Scale`}
                                                value={slot.scale}
                                                min={0}
                                                max={2}
                                                step={0.05}
                                                onChange={(v) => onSlotScaleChange(slot.slot, v)}
                                                onChangeCommitted={(v) => onSlotScaleChange(slot.slot, v)}
                                                formatDisplay={(v) => v.toFixed(2)}
                                            />
                                            </div>

                                            {/* Per-group sliders (expandable) */}
                                            {expandedSlots.has(slot.slot) && (
                                                <div className={`space-y-1 pl-2 border-l-2 border-pink-500/20 ${globalScaleOverrideEnabled ? 'opacity-40 pointer-events-none' : ''}`}>
                                                    {(['self_attn', 'cross_attn', 'mlp'] as const).map((group) => {
                                                        const groupInfo = {
                                                            self_attn: {
                                                                label: 'Self-Attn',
                                                                helpText: 'Controls how audio frames relate to each other over time.',
                                                                tooltip: 'Self-Attention: each audio frame attends to all other frames in the sequence. Controls internal temporal coherence — how rhythmic patterns, melodic phrases, and structural transitions hold together over time. Reducing this loosens the temporal "glue" within the adapter\'s influence.',
                                                            },
                                                            cross_attn: {
                                                                label: 'Cross-Attn',
                                                                helpText: 'How strongly your text prompt shapes the output vs. the adapter\'s baked-in character.',
                                                                tooltip: 'Cross-Attention: audio frames attend to the text/style conditioning — the bridge between your prompt and the output. Lowering this lets the adapter\'s baked-in character dominate over explicit prompt instructions like genre tags, mood, and lyrics.',
                                                            },
                                                            mlp: {
                                                                label: 'MLP',
                                                                helpText: 'Controls the adapter\'s stored timbre, tonal texture, and sonic character.',
                                                                tooltip: 'Feed-Forward Network (MLP): per-frame feature transformation — the "knowledge store" of learned audio patterns. Vocal timbre, tonal texture, and specific sonic character are thought to live primarily here. Reducing MLP strips specific character while preserving rhythmic and structural patterns.',
                                                            },
                                                        };
                                                        const info = groupInfo[group];
                                                        return (
                                                            <EditableSlider
                                                                key={group}
                                                                label={info.label}
                                                                value={slot.group_scales[group]}
                                                                min={0}
                                                                max={2}
                                                                step={0.05}
                                                                onChange={(v) => onSlotGroupScaleChange(slot.slot, group, v)}
                                                                onChangeCommitted={(v) => onSlotGroupScaleChange(slot.slot, group, v)}
                                                                formatDisplay={(v) => v.toFixed(2)}
                                                                helpText={info.helpText}
                                                                tooltip={info.tooltip}
                                                            />
                                                        );
                                                    })}
                                                </div>
                                            )}

                                            {/* ─── Role Sliders ───────────────────────────────────────── */}
                                            {onSlotLayerScaleChange && adapterSlots.length > 1 && (() => {
                                                const ROLES = [
                                                    {
                                                        key: 'voice',
                                                        label: '🎤 Voice',
                                                        layers: [0, 1, 2, 3, 4, 5, 6, 7],
                                                        color: 'blue',
                                                        tooltip: 'Controls layers 0–7. Affects vocal timbre and how much this adapter\'s singer character comes through. Reducing this makes the output sound more like the base model\'s default voice. Approximately ~60% of vocal identity lives here.',
                                                    },
                                                    {
                                                        key: 'style',
                                                        label: '🎸 Style',
                                                        layers: [8, 9, 10, 11, 12, 13, 14, 15],
                                                        color: 'pink',
                                                        tooltip: 'Controls layers 8–15. Affects musical style, energy, tempo feel and genre character. Reducing this makes the output less aggressive/energetic while preserving the voice. These layers also help maintain long-range song structure.',
                                                    },
                                                    {
                                                        key: 'coherence',
                                                        label: '🔗 Coherence',
                                                        layers: [16, 17, 18, 19, 20, 21, 22, 23],
                                                        color: 'emerald',
                                                        tooltip: 'Controls layers 16–23. Acts as the integration glue that binds voice and style into a harmonically coherent output. Without these, voice and style layers can produce discordant results. Reducing coherence too far causes audio artifacts.',
                                                    },
                                                ] as const;

                                                return (
                                                    <div className="space-y-1.5 pt-1">
                                                        <div className="flex items-center gap-1">
                                                            <span className="text-[10px] font-semibold text-zinc-500 dark:text-zinc-400 uppercase tracking-wide">Role Blend</span>
                                                            <span className="text-[9px] text-zinc-400 dark:text-zinc-500">(approx.)</span>
                                                        </div>
                                                        {ROLES.map(({ key, label, layers, color, tooltip }) => {
                                                            const avg = layers.reduce((sum, i) => sum + (slot.layer_scales?.[i] ?? 1.0), 0) / layers.length;
                                                            const isModified = Math.abs(avg - 1.0) > 0.02;
                                                            return (
                                                                <div key={key} className="flex items-center gap-2 group/role">
                                                                    <div className="relative flex items-center gap-1 w-24 flex-shrink-0">
                                                                        <span className={`text-[10px] font-semibold w-full ${isModified ? `text-${color}-500 dark:text-${color}-400` : 'text-zinc-600 dark:text-zinc-400'}`}>
                                                                            {label}
                                                                        </span>
                                                                        {/* Tooltip */}
                                                                        <div className="absolute left-0 bottom-full mb-1.5 hidden group-hover/role:block z-50 w-56 bg-zinc-900 dark:bg-zinc-800 text-zinc-100 text-[10px] leading-relaxed rounded-lg px-2.5 py-2 shadow-xl border border-white/10 pointer-events-none">
                                                                            {tooltip}
                                                                        </div>
                                                                    </div>
                                                                    <input
                                                                        type="range"
                                                                        min={0}
                                                                        max={2}
                                                                        step={0.05}
                                                                        value={pendingRoleBlend[`${slot.slot}-${key}`] ?? avg}
                                                                        onChange={(e) => {
                                                                            setPendingRoleBlend(prev => ({ ...prev, [`${slot.slot}-${key}`]: parseFloat(e.target.value) }));
                                                                        }}
                                                                        onPointerUp={() => {
                                                                            const val = pendingRoleBlend[`${slot.slot}-${key}`];
                                                                            if (val !== undefined) {
                                                                                for (const i of layers) {
                                                                                    onSlotLayerScaleChange(slot.slot, i, val);
                                                                                }
                                                                                setPendingRoleBlend(prev => { const next = { ...prev }; delete next[`${slot.slot}-${key}`]; return next; });
                                                                            }
                                                                        }}
                                                                        onTouchEnd={() => {
                                                                            const val = pendingRoleBlend[`${slot.slot}-${key}`];
                                                                            if (val !== undefined) {
                                                                                for (const i of layers) {
                                                                                    onSlotLayerScaleChange(slot.slot, i, val);
                                                                                }
                                                                                setPendingRoleBlend(prev => { const next = { ...prev }; delete next[`${slot.slot}-${key}`]; return next; });
                                                                            }
                                                                        }}
                                                                        className={`flex-1 accent-${color}-500 h-1`}
                                                                        style={{ height: '4px' }}
                                                                    />
                                                                    <span className={`text-[10px] font-mono w-7 text-right ${isModified ? `text-${color}-500 dark:text-${color}-400 font-semibold` : 'text-zinc-500'}`}>
                                                                        {(pendingRoleBlend[`${slot.slot}-${key}`] ?? avg).toFixed(2)}
                                                                    </span>
                                                                </div>
                                                            );
                                                        })}
                                                    </div>
                                                );
                                            })()}

                                            {/* Per-layer sliders (expandable) */}
                                            {expandedLayers.has(slot.slot) && (
                                                <div className="space-y-2 pl-2 border-l-2 border-purple-500/20">
                                                    <div className="flex items-center justify-between">
                                                        <span className="text-[10px] font-semibold text-zinc-500 dark:text-zinc-400 uppercase tracking-wide">Layer Scales (0–23)</span>
                                                        <button
                                                            onClick={() => {
                                                                if (onSlotLayerScaleChange) {
                                                                    for (let i = 0; i < 24; i++) {
                                                                        if ((slot.layer_scales?.[i] ?? 1.0) !== 1.0) {
                                                                            onSlotLayerScaleChange(slot.slot, i, 1.0);
                                                                        }
                                                                    }
                                                                }
                                                            }}
                                                            className="text-[10px] text-zinc-400 hover:text-pink-500 transition-colors"
                                                        >
                                                            Reset All
                                                        </button>
                                                    </div>
                                                    <div className="grid grid-cols-6 gap-x-2 gap-y-1">
                                                        {Array.from({ length: 24 }, (_, i) => {
                                                            const val = slot.layer_scales?.[i] ?? 1.0;
                                                            const isModified = Math.abs(val - 1.0) > 0.01;
                                                            return (
                                                                <div key={i} className="flex flex-col items-center">
                                                                    <span className={`text-[9px] font-mono ${isModified ? 'text-purple-500 font-bold' : 'text-zinc-400'}`}>{i}</span>
                                                                    <input
                                                                        type="range"
                                                                        min={0}
                                                                        max={2}
                                                                        step={0.05}
                                                                        value={pendingLayerScale[`${slot.slot}-layer-${i}`] ?? val}
                                                                        onChange={(e) => setPendingLayerScale(prev => ({ ...prev, [`${slot.slot}-layer-${i}`]: parseFloat(e.target.value) }))}
                                                                        onPointerUp={() => {
                                                                            const v = pendingLayerScale[`${slot.slot}-layer-${i}`];
                                                                            if (v !== undefined) {
                                                                                onSlotLayerScaleChange?.(slot.slot, i, v);
                                                                                setPendingLayerScale(prev => { const next = { ...prev }; delete next[`${slot.slot}-layer-${i}`]; return next; });
                                                                            }
                                                                        }}
                                                                        onTouchEnd={() => {
                                                                            const v = pendingLayerScale[`${slot.slot}-layer-${i}`];
                                                                            if (v !== undefined) {
                                                                                onSlotLayerScaleChange?.(slot.slot, i, v);
                                                                                setPendingLayerScale(prev => { const next = { ...prev }; delete next[`${slot.slot}-layer-${i}`]; return next; });
                                                                            }
                                                                        }}
                                                                        className="w-full h-1 accent-purple-500"
                                                                        style={{ WebkitAppearance: 'none', height: '4px' }}
                                                                    />
                                                                    <span className={`text-[8px] ${isModified ? 'text-purple-400 font-semibold' : 'text-zinc-500'}`}>{(pendingLayerScale[`${slot.slot}-layer-${i}`] ?? val).toFixed(1)}</span>
                                                                </div>
                                                            );
                                                        })}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* ── Global Scale Overrides ─────────────────────────────── */}
                            {adapterSlots.length > 0 && (
                                <div className="space-y-3 pt-3 border-t border-zinc-200 dark:border-white/5">
                                    <div className="flex items-center justify-between py-1">
                                        <div>
                                            <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400 flex items-center gap-1.5">
                                                🌐 Global Scale Overrides
                                                {globalScaleOverrideEnabled && (
                                                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400 animate-pulse">
                                                        ACTIVE
                                                    </span>
                                                )}
                                            </span>
                                            <p className="text-[10px] text-zinc-500">Override all per-adapter scales with a single set of global values</p>
                                        </div>
                                        <Toggle on={globalScaleOverrideEnabled || false} onClick={() => onGlobalOverrideToggle?.(!globalScaleOverrideEnabled)} />
                                    </div>
                                    {globalScaleOverrideEnabled && (
                                        <div className="space-y-2 pl-2 border-l-2 border-amber-500/30 bg-amber-50/50 dark:bg-amber-900/10 rounded-r-lg p-3">
                                            <p className="text-[10px] text-amber-600 dark:text-amber-400/70 leading-tight mb-2">
                                                These values override all per-adapter scales. Individual adapter settings are preserved and restored when this is disabled.
                                            </p>
                                            <EditableSlider
                                                label="Overall Scale"
                                                value={globalOverallScale ?? 1.0}
                                                min={0}
                                                max={2}
                                                step={0.05}
                                                onChange={(v) => onGlobalOverallScaleChange?.(v)}
                                                onChangeCommitted={(v) => onGlobalOverallScaleChange?.(v)}
                                                formatDisplay={(v) => v.toFixed(2)}
                                            />
                                            <div className="space-y-1">
                                                {(['self_attn', 'cross_attn', 'mlp'] as const).map((group) => {
                                                    const groupInfo = {
                                                        self_attn: {
                                                            label: 'Self-Attn',
                                                            helpText: 'Controls how audio frames relate to each other over time.',
                                                        },
                                                        cross_attn: {
                                                            label: 'Cross-Attn',
                                                            helpText: 'How strongly your text prompt shapes the output vs. the adapter\'s baked-in character.',
                                                        },
                                                        mlp: {
                                                            label: 'MLP',
                                                            helpText: 'Controls the adapter\'s stored timbre, tonal texture, and sonic character.',
                                                        },
                                                    };
                                                    const info = groupInfo[group];
                                                    return (
                                                        <EditableSlider
                                                            key={group}
                                                            label={info.label}
                                                            value={globalGroupScales?.[group] ?? 1.0}
                                                            min={0}
                                                            max={2}
                                                            step={0.05}
                                                            onChange={(v) => onGlobalGroupScaleChange?.(group, v)}
                                                            onChangeCommitted={(v) => onGlobalGroupScaleChange?.(group, v)}
                                                            formatDisplay={(v) => v.toFixed(2)}
                                                            helpText={info.helpText}
                                                        />
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Temporal Schedule Section — only when 2+ adapters loaded */}
                            {adapterSlots.length >= 2 && (
                                <div className="space-y-2 pt-2 border-t border-zinc-200 dark:border-white/5">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">Temporal Schedule</span>
                                            {temporalScheduleActive && (
                                                <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400 animate-pulse">
                                                    ACTIVE
                                                </span>
                                            )}
                                        </div>
                                        {temporalScheduleActive && (
                                            <button
                                                onClick={() => onTemporalSchedulePreset?.(null)}
                                                className="text-[10px] text-red-500 hover:text-red-400 transition-colors"
                                            >
                                                Clear
                                            </button>
                                        )}
                                    </div>
                                    <p className="text-[10px] text-zinc-400 dark:text-zinc-500 leading-tight">
                                        Switch between adapters during generation. The first two loaded adapters will be used as A and B.
                                    </p>
                                    <div className="flex gap-1.5">
                                        <button
                                            onClick={() => onTemporalSchedulePreset?.('switch')}
                                            className={`flex-1 px-2 py-1.5 rounded-lg text-[10px] font-semibold border transition-colors ${temporalScheduleActive
                                                ? 'bg-zinc-100 dark:bg-black/30 border-zinc-200 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-white/10'
                                                : 'bg-zinc-100 dark:bg-black/30 border-zinc-200 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-white/10'
                                                }`}
                                        >
                                            🔀 A→B Switch
                                        </button>
                                        <button
                                            onClick={() => onTemporalSchedulePreset?.('verse-chorus')}
                                            className="flex-1 px-2 py-1.5 rounded-lg text-[10px] font-semibold border transition-colors bg-zinc-100 dark:bg-black/30 border-zinc-200 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-white/10"
                                        >
                                            🎵 Verse/Chorus
                                        </button>
                                    </div>
                                </div>
                            )}

                            {adapterSlots.length === 0 && adapterFiles.length === 0 && (
                                <div className="text-xs text-zinc-400 dark:text-zinc-600 text-center py-2">
                                    Enter an adapter folder path and click Scan to browse available adapters
                                </div>
                            )}
                        </>
                    )}
                </div>
    );

    if (embedded) return (
        <>
            {content}
            <FileBrowserModal
                open={fileBrowserOpen}
                onClose={() => setFileBrowserOpen(false)}
                onSelect={(path) => {
                    if (fileBrowserMode === 'file') {
                        onLoraPathChange(path);
                    } else {
                        onAdapterFolderChange(path);
                    }
                    setFileBrowserOpen(false);
                }}
                mode={fileBrowserMode}
            />
        </>
    );

    return (
        <div>
            <button
                onClick={onToggle}
                className={`w-full flex items-center justify-between px-4 py-3 bg-white dark:bg-suno-card border border-zinc-200 dark:border-white/5 text-sm font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors ${isOpen ? 'rounded-t-xl rounded-b-none border-b-0' : 'rounded-xl'}`}
            >
                <div className="flex items-center gap-2">
                    <Sliders size={16} className="text-zinc-500" />
                    <span>Adapters (LoRA / LoKR)</span>
                </div>
                <ChevronDown size={18} className={`text-pink-500 chevron-icon ${isOpen ? 'rotated' : ''}`} />
            </button>

            {isOpen && content}
            <FileBrowserModal
                open={fileBrowserOpen}
                onClose={() => setFileBrowserOpen(false)}
                onSelect={(path) => {
                    if (fileBrowserMode === 'file') {
                        onLoraPathChange(path);
                    } else {
                        onAdapterFolderChange(path);
                    }
                    setFileBrowserOpen(false);
                }}
                mode={fileBrowserMode}
            />
        </div>
    );
};
export default AdaptersAccordion;
