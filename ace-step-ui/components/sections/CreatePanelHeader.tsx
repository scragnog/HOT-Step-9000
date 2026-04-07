import React from 'react';
import { ChevronDown, RefreshCw, Loader2, Download, Upload } from 'lucide-react';
import { useI18n } from '../../context/I18nContext';
import { TaskTypeSelector } from './TaskTypeSelector';

interface ModelInfo {
    id: string;
    name: string;
    is_preloaded?: boolean;
    is_active?: boolean;
}

interface CreatePanelHeaderProps {
    modelMenuRef: React.RefObject<HTMLDivElement>;
    showModelMenu: boolean;
    setShowModelMenu: (val: boolean) => void;
    availableModels: { id: string, name: string }[];
    selectedModel: string;
    setSelectedModel: (val: string) => void;
    backendUnavailable: boolean;
    fetchedModels: ModelInfo[];
    setInferenceSteps: (val: number) => void;
    setUseAdg: (val: boolean) => void;
    getModelDisplayName: (id: string) => string;
    isTurboModel: (id: string) => boolean;
    activeBackendModel: string | null;
    isSwitching: boolean;
    isGenerating: boolean;
    handleSwitchModel: (id: string) => void;
    // Task type
    taskType: string;
    setTaskType: (val: string) => void;
    useReferenceAudio: boolean;
    // JSON import/export
    fileInputRef: React.RefObject<HTMLInputElement>;
    onImportJson: (e: React.ChangeEvent<HTMLInputElement>) => void;
    onExportJson: () => void;
}

export const CreatePanelHeader: React.FC<CreatePanelHeaderProps> = ({
    modelMenuRef,
    showModelMenu,
    setShowModelMenu,
    availableModels,
    selectedModel,
    setSelectedModel,
    backendUnavailable,
    fetchedModels,
    setInferenceSteps,
    setUseAdg,
    getModelDisplayName,
    isTurboModel,
    activeBackendModel,
    isSwitching,
    isGenerating,
    handleSwitchModel,
    taskType,
    setTaskType,
    useReferenceAudio,
    fileInputRef,
    onImportJson,
    onExportJson,
}) => {
    const { t } = useI18n();

    return (
        <>
            {/* Two-column header: Logo left, controls right */}
            <div className="flex gap-4 items-start">
                {/* Left column — Logo */}
                <div className="flex-shrink-0 flex items-start pt-1">
                    <img
                        src="/hotstep-logo-small.webp"
                        alt="HOT-Step 9000 Logo"
                        style={{ width: '200px', height: 'auto' }}
                        className="rounded opacity-90 object-contain hover:opacity-100 transition-opacity"
                    />
                </div>

                {/* Right column — Model, Task Type, JSON buttons */}
                <div className="flex-1 flex flex-col gap-2 min-w-0">
                    {/* Model Selector */}
                    <div className="relative" ref={modelMenuRef}>
                        <label className="block text-[10px] font-semibold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider mb-1">
                            Model
                        </label>
                        <button
                            onClick={() => setShowModelMenu(!showModelMenu)}
                            className="w-full bg-zinc-200 dark:bg-black/40 border border-zinc-300 dark:border-white/5 rounded-md px-2.5 py-1.5 text-[11px] font-medium text-zinc-900 dark:text-white hover:bg-zinc-300 dark:hover:bg-black/50 transition-colors flex items-center justify-between gap-1"
                            disabled={availableModels.length === 0}
                        >
                            <span className="truncate">
                                {availableModels.length === 0 ? '...' : getModelDisplayName(selectedModel)}
                            </span>
                            <ChevronDown size={10} className="text-zinc-600 dark:text-zinc-400 flex-shrink-0" />
                        </button>

                        {/* Floating Model Menu */}
                        {showModelMenu && availableModels.length > 0 && (
                            <div className="absolute top-full right-0 mt-1 w-72 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-xl shadow-2xl z-50 overflow-hidden">
                                {/* Backend unavailable hint */}
                                {backendUnavailable && fetchedModels.length === 0 && (
                                    <div className="px-4 py-2 text-xs text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 border-b border-amber-200 dark:border-amber-800 flex items-center gap-2">
                                        <span className="inline-block w-2 h-2 rounded-full bg-amber-400 dark:bg-amber-500 animate-pulse flex-shrink-0" />
                                        {t('backendNotStarted') || 'ACE-Step 后端暂未启动，使用默认模型列表'}
                                    </div>
                                )}
                                <div className="max-h-96 overflow-y-auto custom-scrollbar">
                                    {availableModels.map(model => (
                                        <button
                                            key={model.id}
                                            onClick={() => {
                                                setSelectedModel(model.id);
                                                if (!isTurboModel(model.id)) {
                                                    setInferenceSteps(20);
                                                    setUseAdg(true);
                                                }
                                                setShowModelMenu(false);
                                            }}
                                            className={`w-full px-4 py-3 text-left hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors border-b border-zinc-100 dark:border-zinc-800 last:border-b-0 ${selectedModel === model.id ? 'bg-zinc-50 dark:bg-zinc-800/50' : ''
                                                }`}
                                        >
                                            <div className="flex items-center justify-between mb-1">
                                                <div className="flex items-center gap-2">
                                                    <span className="text-sm font-semibold text-zinc-900 dark:text-white">
                                                        {getModelDisplayName(model.id)}
                                                    </span>
                                                    {fetchedModels.find(m => m.name === model.id)?.is_preloaded && (
                                                        <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400">
                                                            {fetchedModels.find(m => m.name === model.id)?.is_active ? t('modelActive') : t('modelReady')}
                                                        </span>
                                                    )}
                                                </div>
                                                {selectedModel === model.id && (
                                                    <div className="w-4 h-4 rounded-full bg-pink-500 flex items-center justify-center">
                                                        <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                                                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                                        </svg>
                                                    </div>
                                                )}
                                            </div>
                                            <p className="text-xs text-zinc-500 dark:text-zinc-400">{model.id}</p>
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Task Type Selector (compact — no card wrapper) */}
                    <TaskTypeSelector
                        taskType={taskType}
                        setTaskType={setTaskType}
                        useReferenceAudio={useReferenceAudio}
                        selectedModel={selectedModel}
                    />

                    {/* JSON Import / Export — compact row */}
                    <div className="flex items-center gap-1.5">
                        <input
                            type="file"
                            accept=".json"
                            className="hidden"
                            ref={fileInputRef}
                            onChange={onImportJson}
                        />
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            className="flex-1 flex items-center justify-center gap-1 py-1 px-2 bg-zinc-100 dark:bg-white/5 border border-zinc-200 dark:border-white/10 hover:border-indigo-300 dark:hover:border-indigo-500/50 rounded-md text-[10px] font-semibold text-zinc-500 dark:text-zinc-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-all"
                            title="Import settings from a JSON file"
                        >
                            <Download size={10} />
                            Import JSON
                        </button>
                        <button
                            onClick={onExportJson}
                            className="flex-1 flex items-center justify-center gap-1 py-1 px-2 bg-zinc-100 dark:bg-white/5 border border-zinc-200 dark:border-white/10 hover:border-emerald-300 dark:hover:border-emerald-500/50 rounded-md text-[10px] font-semibold text-zinc-500 dark:text-zinc-400 hover:text-emerald-600 dark:hover:text-emerald-400 transition-all"
                            title="Export current settings as a JSON file"
                        >
                            <Upload size={10} />
                            Export JSON
                        </button>
                    </div>
                </div>
            </div>

            {/* Model mismatch warning */}
            {activeBackendModel && selectedModel !== activeBackendModel && (
                <div className="mt-2 px-3 py-2 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700/50 flex items-center justify-between gap-2">
                    <p className="text-xs text-amber-700 dark:text-amber-300">
                        <span className="font-semibold">{getModelDisplayName(selectedModel)}</span> selected but <span className="font-semibold">{getModelDisplayName(activeBackendModel)}</span> is loaded
                    </p>
                    <button
                        onClick={() => handleSwitchModel(selectedModel)}
                        disabled={isSwitching || isGenerating}
                        className="shrink-0 px-2.5 py-1 rounded-md text-[11px] font-semibold bg-amber-500 hover:bg-amber-600 text-white transition-colors disabled:opacity-50 flex items-center gap-1"
                    >
                        {isSwitching ? (
                            <><Loader2 size={10} className="animate-spin" /> Switching…</>
                        ) : (
                            <><RefreshCw size={10} /> Switch</>
                        )}
                    </button>
                </div>
            )}

            {/* XL (4B) model warnings */}
            {selectedModel.includes('-xl-') && (
                <>
                    <div className="mt-2 px-3 py-2 rounded-lg bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-700/50">
                        <p className="text-xs text-orange-700 dark:text-orange-300 font-medium">
                            ⚠️ <span className="font-bold">XL Model (4B params)</span> — Requires ≥16 GB VRAM. xl_turbo: ~12 GB, xl_base/sft: ~16 GB.
                        </p>
                    </div>
                    <div className="mt-1 px-3 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800/50 border border-zinc-200 dark:border-zinc-700/50">
                        <p className="text-[10px] text-zinc-500 dark:text-zinc-400">
                            ⚠️ XL models require adapters trained specifically on the XL architecture — standard 1.5B adapters are not compatible
                        </p>
                    </div>
                </>
            )}
        </>
    );
};

export default CreatePanelHeader;
