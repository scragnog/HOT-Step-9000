import React from 'react';
import { ChevronDown, RefreshCw, Loader2 } from 'lucide-react';
import { useI18n } from '../../context/I18nContext';

interface ModelInfo {
    id: string;
    name: string;
    is_preloaded?: boolean;
    is_active?: boolean;
}

interface CreatePanelHeaderProps {
    customMode: boolean;
    setCustomMode: (val: boolean) => void;
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
}

export const CreatePanelHeader: React.FC<CreatePanelHeaderProps> = ({
    customMode,
    setCustomMode,
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
    handleSwitchModel
}) => {
    const { t } = useI18n();

    return (
        <>
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <img 
            src="/hotstep-logo-small.webp" 
            alt="HOT-Step 9000 Logo" 
            style={{ width: '200px', height: 'auto' }}
            className="rounded opacity-90 object-contain hover:opacity-100 transition-opacity"
          />
                </div>

                <div className="flex items-center gap-2">
                    {/* Mode Toggle */}
                    <div className="flex items-center bg-zinc-200 dark:bg-black/40 rounded-lg p-1 border border-zinc-300 dark:border-white/5">
                        <button
                            onClick={() => setCustomMode(false)}
                            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${!customMode ? 'bg-white dark:bg-zinc-800 text-black dark:text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-300'}`}
                        >
                            {t('simple')}
                        </button>
                        <button
                            onClick={() => setCustomMode(true)}
                            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${customMode ? 'bg-white dark:bg-zinc-800 text-black dark:text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-300'}`}
                        >
                            {t('custom')}
                        </button>
                    </div>

                    {/* Model Selection */}
                    <div className="relative" ref={modelMenuRef}>
                        <button
                            onClick={() => setShowModelMenu(!showModelMenu)}
                            className="bg-zinc-200 dark:bg-black/40 border border-zinc-300 dark:border-white/5 rounded-md px-2 py-1 text-[11px] font-medium text-zinc-900 dark:text-white hover:bg-zinc-300 dark:hover:bg-black/50 transition-colors flex items-center gap-1"
                            disabled={availableModels.length === 0}
                        >
                            {availableModels.length === 0 ? '...' : getModelDisplayName(selectedModel)}
                            <ChevronDown size={10} className="text-zinc-600 dark:text-zinc-400" />
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
                                                // Auto-adjust parameters for non-turbo models
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
                </div>
            </div>

            {activeBackendModel && selectedModel !== activeBackendModel && (
                <div className="mx-4 mt-2 px-3 py-2 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700/50 flex items-center justify-between gap-2">
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
        </>
    );
};

export default CreatePanelHeader;
