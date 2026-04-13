import React from 'react';
import { Brain, Music2, ChevronDown } from 'lucide-react';
import { useI18n } from '../../context/I18nContext';
import { EditableSlider } from '../EditableSlider';

interface LmCotAccordionProps {
    isOpen: boolean;
    onToggle: () => void;
    // Thinking toggle
    thinking: boolean;
    onThinkingToggle: () => void;
    loraLoaded: boolean;
    // LM Backend
    lmBackend: 'pt' | 'vllm' | 'custom-vllm';
    onLmBackendChange: (val: 'pt' | 'vllm' | 'custom-vllm') => void;
    // LM Model
    lmModel: string;
    onLmModelChange: (val: string) => void;
    isLmSwitching?: boolean;
    // LM Parameters sub-accordion
    showLmParams: boolean;
    onToggleLmParams: () => void;
    lmTemperature: number;
    onLmTemperatureChange: (val: number) => void;
    lmCfgScale: number;
    onLmCfgScaleChange: (val: number) => void;
    lmTopK: number;
    onLmTopKChange: (val: number) => void;
    lmTopP: number;
    onLmTopPChange: (val: number) => void;
    lmRepetitionPenalty: number;
    onLmRepetitionPenaltyChange: (val: number) => void;
    lmNegativePrompt: string;
    onLmNegativePromptChange: (val: string) => void;
    // Allow LM Batch
    allowLmBatch: boolean;
    onAllowLmBatchToggle: () => void;
    // CoT options
    useCotMetas: boolean;
    onUseCotMetasToggle: () => void;
    useCotCaption: boolean;
    onUseCotCaptionToggle: () => void;
    useCotLanguage: boolean;
    onUseCotLanguageToggle: () => void;
    // LM Batch Chunk Size
    lmBatchChunkSize: number;
    onLmBatchChunkSizeChange: (val: number) => void;
    // Constrained Decoding Debug
    constrainedDecodingDebug: boolean;
    onConstrainedDecodingDebugToggle: () => void;
    // Format Caption
    isFormatCaption: boolean;
    onIsFormatCaptionToggle: () => void;
    // LM Codes Scale (blending strength of LM audio codes)
    lmCodesScale?: number;
    onLmCodesScaleChange?: (val: number) => void;
    /** When true, skip the accordion header — used inside DrawerContainers */
    embedded?: boolean;
}

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

export const LmCotAccordion: React.FC<LmCotAccordionProps> = ({
    isOpen, onToggle,
    thinking, onThinkingToggle, loraLoaded,
    lmBackend, onLmBackendChange,
    lmModel, onLmModelChange, isLmSwitching,
    showLmParams, onToggleLmParams,
    lmTemperature, onLmTemperatureChange,
    lmCfgScale, onLmCfgScaleChange,
    lmTopK, onLmTopKChange,
    lmTopP, onLmTopPChange,
    lmRepetitionPenalty, onLmRepetitionPenaltyChange,
    lmNegativePrompt, onLmNegativePromptChange,
    allowLmBatch, onAllowLmBatchToggle,
    useCotMetas, onUseCotMetasToggle,
    useCotCaption, onUseCotCaptionToggle,
    useCotLanguage, onUseCotLanguageToggle,
    lmBatchChunkSize, onLmBatchChunkSizeChange,
    constrainedDecodingDebug, onConstrainedDecodingDebugToggle,
    isFormatCaption, onIsFormatCaptionToggle,
    lmCodesScale, onLmCodesScaleChange,
    embedded,
}) => {
    const { t } = useI18n();


    const selectClass = "w-full bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-xl px-2 py-1.5 text-xs text-zinc-900 dark:text-white focus:outline-none focus:border-pink-500 dark:focus:border-pink-500 transition-colors cursor-pointer [&>option]:bg-white [&>option]:dark:bg-zinc-800 [&>option]:text-zinc-900 [&>option]:dark:text-white";

    const content = (
                <div className={embedded ? "space-y-4" : "bg-white dark:bg-suno-card rounded-b-xl rounded-t-none border border-t-0 border-zinc-200 dark:border-white/5 p-4 space-y-4"}>
                    {/* Thinking Toggle */}
                    <div className="flex items-center justify-between py-1">
                        <div>
                            <span className={`text-xs font-medium text-zinc-600 dark:text-zinc-400`} title={t('thinkingTooltip')}>{t('thinkingCot')}</span>
                            <p className="text-[10px] text-zinc-500">Let the AI reason about your prompt before generating music</p>
                        </div>
                        <div className="flex items-center gap-2">
                            {loraLoaded && thinking && <span className="text-[10px] text-amber-500 dark:text-amber-400">⚠ experimental</span>}
                            <Toggle on={thinking} onClick={onThinkingToggle} />
                        </div>
                    </div>

                    {/* LM Backend */}
                    <div className="space-y-1.5">
                        <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{t('lmBackendLabel')}</label>
                        <select value={lmBackend} onChange={(e) => onLmBackendChange(e.target.value as 'pt' | 'vllm' | 'custom-vllm')} className={selectClass}>
                            <option value="pt">{t('lmBackendPt')}</option>
                            <option value="vllm">{t('lmBackendVllm')}</option>
                            <option value="custom-vllm">Custom VLLM</option>
                        </select>
                        <p className="text-[10px] text-zinc-500">{t('lmBackendHint')}</p>
                    </div>

                    {/* LM Model */}
                    <div className="space-y-1.5">
                        <div className="flex items-center gap-2">
                            <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{t('lmModelLabel')}</label>
                            {isLmSwitching && (
                                <span className="text-[10px] text-pink-500 animate-pulse">switching…</span>
                            )}
                        </div>
                        <select value={lmModel} onChange={(e) => onLmModelChange(e.target.value)} disabled={isLmSwitching} className={selectClass + (isLmSwitching ? ' opacity-50' : '')}>
                            <option value="acestep-5Hz-lm-0.6B">{t('lmModel06B')}</option>
                            <option value="acestep-5Hz-lm-1.7B">{t('lmModel17B')}</option>
                            <option value="acestep-5Hz-lm-4B">{t('lmModel4B')}</option>
                        </select>
                        <p className="text-[10px] text-zinc-500">{t('lmModelHint')}</p>
                    </div>



                    {/* LM Parameters sub-accordion */}
                    <button
                        type="button"
                        onClick={onToggleLmParams}
                        className={`w-full flex items-center justify-between px-4 py-3 bg-white/60 dark:bg-black/20 border border-zinc-200/70 dark:border-white/10 text-sm font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors ${showLmParams ? 'rounded-t-xl rounded-b-none border-b-0' : 'rounded-xl'}`}
                    >
                        <div className="flex items-center gap-2">
                            <Music2 size={16} className="text-zinc-500" />
                            <div className="flex flex-col items-start">
                                <span title={t('lmParametersTooltip')}>{t('lmParameters')}</span>
                                <span className="text-[11px] text-zinc-400 dark:text-zinc-500 font-normal">{t('controlLyricGeneration')}</span>
                            </div>
                        </div>
                        <ChevronDown size={18} className={`text-pink-500 chevron-icon ${showLmParams ? 'rotated' : ''}`} />
                    </button>

                    {showLmParams && (
                        <div className="bg-zinc-50 dark:bg-black/10 rounded-b-xl rounded-t-none border border-t-0 border-zinc-200 dark:border-white/5 p-4 space-y-4">
                            <EditableSlider label={t('lmTemperature')} value={lmTemperature} min={0} max={2} step={0.05} onChange={onLmTemperatureChange} formatDisplay={(val) => val.toFixed(2)} helpText={t('higherMoreRandom')} title={t('lmTemperatureTooltip')} />
                            <EditableSlider label={t('lmCfgScale')} value={lmCfgScale} min={1} max={5} step={0.1} onChange={onLmCfgScaleChange} formatDisplay={(val) => val.toFixed(1)} helpText={t('noCfgScale')} title={t('lmGuidanceScaleTooltip')} />
                            <div className="grid grid-cols-2 gap-3">
                                <EditableSlider label={t('topK')} value={lmTopK} min={0} max={100} step={1} onChange={onLmTopKChange} title={t('lmTopKTooltip')} helpText="Limits token choices per step. 0 = unlimited, 10-30 = safe and predictable." />
                                <EditableSlider label={t('topP')} value={lmTopP} min={0} max={1} step={0.01} onChange={onLmTopPChange} formatDisplay={(val) => val.toFixed(2)} title={t('lmTopPTooltip')} helpText="Nucleus sampling threshold. 0.92 = ignore bottom 8% of unlikely tokens. Lower = safer." />
                            </div>
                            <EditableSlider label={t('lmRepetitionPenalty')} value={lmRepetitionPenalty ?? 1.0} min={1.0} max={1.5} step={0.01} onChange={onLmRepetitionPenaltyChange} formatDisplay={(val) => val.toFixed(2)} helpText={t('lmRepetitionPenaltyHelp')} title={t('lmRepetitionPenaltyTooltip')} />
                            <div className="space-y-1.5">
                                <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400" title={t('lmNegativePromptTooltip')}>{t('lmNegativePrompt')}</label>
                                <textarea value={lmNegativePrompt} onChange={(e) => onLmNegativePromptChange(e.target.value)} placeholder={t('thingsToAvoid')} className="w-full h-16 bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg p-2 text-xs text-zinc-900 dark:text-white focus:outline-none resize-none" />
                                <p className="text-[10px] text-zinc-500">{t('useWhenCfgScaleGreater')}</p>
                            </div>
                        </div>
                    )}

                    {/* Allow LM Batch */}
                    <div className="flex items-center justify-between py-1">
                        <div>
                            <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400" title={t('allowLmBatchTooltip')}>{t('allowLmBatch')}</span>
                            <p className="text-[10px] text-zinc-500">Process multiple prompts at once for faster batch generation</p>
                        </div>
                        <Toggle on={allowLmBatch} onClick={onAllowLmBatchToggle} />
                    </div>

                    {/* CoT Options (only visible when Thinking is ON) */}
                    {thinking && (
                        <div className="space-y-2 pt-2 border-t border-zinc-100 dark:border-white/5">
                            <p className="text-[10px] text-zinc-500 uppercase tracking-wide font-bold">{t('cotOptions')}</p>
                            {[
                                { label: t('useCotMetas'), tooltip: t('useCotMetadataTooltip'), value: useCotMetas, onToggle: onUseCotMetasToggle },
                                { label: t('useCotCaption'), tooltip: t('useCotCaptionTooltip'), value: useCotCaption, onToggle: onUseCotCaptionToggle },
                                { label: t('useCotLanguage'), tooltip: t('useCotLanguageTooltip'), value: useCotLanguage, onToggle: onUseCotLanguageToggle },
                            ].map(({ label, tooltip, value, onToggle: tog }) => (
                                <div key={label} className="flex items-center justify-between py-1">
                                    <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400" title={tooltip}>{label}</span>
                                    <Toggle on={value} onClick={tog} />
                                </div>
                            ))}
                            {/* LM Codes Scale — only when thinking */}
                            {onLmCodesScaleChange && (
                                <EditableSlider
                                    label="LM Codes Scale"
                                    value={lmCodesScale ?? 1.0}
                                    min={0}
                                    max={1}
                                    step={0.05}
                                    onChange={onLmCodesScaleChange}
                                    formatDisplay={(val) => val.toFixed(2)}
                                    helpText="How strongly the LM's audio codes influence diffusion. 1.0 = full, 0.5 = blended, 0 = off"
                                    title="Blends LM-generated latent hints with the original source latents before diffusion"
                                />
                            )}
                        </div>
                    )}

                    {/* LM Batch Chunk Size */}
                    <EditableSlider label={t('lmBatchChunkSize')} value={lmBatchChunkSize} min={1} max={32} step={1} onChange={onLmBatchChunkSizeChange} formatDisplay={(val) => `${val}`} helpText={t('lmBatchChunkSizeHelp')} title={t('lmBatchChunkSizeTooltip')} />

                    {/* Constrained Decoding Debug */}
                    <div className="flex items-center justify-between py-1">
                        <div>
                            <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{t('constrainedDecodingDebug')}</span>
                            <p className="text-[11px] text-zinc-400 dark:text-zinc-500">{t('constrainedDecodingDebugHelp')}</p>
                        </div>
                        <Toggle on={constrainedDecodingDebug} onClick={onConstrainedDecodingDebugToggle} />
                    </div>

                    {/* Format Caption */}
                    <div className="flex items-center justify-between py-1">
                        <div>
                            <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{t('formatCaption')}</span>
                            <p className="text-[11px] text-zinc-400 dark:text-zinc-500">{t('formatCaptionHelp')}</p>
                        </div>
                        <Toggle on={isFormatCaption} onClick={onIsFormatCaptionToggle} />
                    </div>
                </div>
    );

    if (embedded) return content;

    return (
        <div>
            <button
                type="button"
                onClick={onToggle}
                className={`w-full flex items-center justify-between px-4 py-3 bg-white dark:bg-suno-card border border-zinc-200 dark:border-white/5 text-sm font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors ${isOpen ? 'rounded-t-xl rounded-b-none border-b-0' : 'rounded-xl'}`}
            >
                <div className="flex items-center gap-2">
                    <Brain size={16} className="text-zinc-500" />
                    <span>{t('lmCotSettings')}</span>
                </div>
                <ChevronDown size={18} className={`text-pink-500 chevron-icon ${isOpen ? 'rotated' : ''}`} />
            </button>
            {isOpen && content}
        </div>
    );
};

export default LmCotAccordion;
