import React from 'react';
import { Crosshair, ChevronDown } from 'lucide-react';
import { useI18n } from '../../context/I18nContext';
import { EditableSlider } from '../EditableSlider';

type GuidanceMode = 'apg' | 'adg' | 'pag' | 'cfg_pp' | 'dynamic_cfg' | 'rescaled_cfg';

interface GuidanceSettingsAccordionProps {
    isOpen: boolean;
    onToggle: () => void;
    guidanceScale: number;
    onGuidanceScaleChange: (val: number) => void;
    guidanceMode: GuidanceMode;
    onGuidanceModeChange: (mode: GuidanceMode) => void;
    pagStart: number;
    onPagStartChange: (val: number) => void;
    pagEnd: number;
    onPagEndChange: (val: number) => void;
    pagScale: number;
    onPagScaleChange: (val: number) => void;
    cfgIntervalStart: number;
    onCfgIntervalStartChange: (val: number) => void;
    cfgIntervalEnd: number;
    onCfgIntervalEndChange: (val: number) => void;
    isTurbo?: boolean;

    // Advanced Guidance Parameters
    guidanceScaleText?: number;
    onGuidanceScaleTextChange?: (val: number) => void;
    guidanceScaleLyric?: number;
    onGuidanceScaleLyricChange?: (val: number) => void;
    apgMomentum?: number;
    onApgMomentumChange?: (val: number) => void;
    apgNormThreshold?: number;
    onApgNormThresholdChange?: (val: number) => void;
    omegaScale?: number;
    onOmegaScaleChange?: (val: number) => void;
    ergScale?: number;
    onErgScaleChange?: (val: number) => void;
}

export const GuidanceSettingsAccordion: React.FC<GuidanceSettingsAccordionProps> = ({
    isOpen,
    onToggle,
    guidanceScale,
    onGuidanceScaleChange,
    guidanceMode,
    onGuidanceModeChange,
    pagStart,
    onPagStartChange,
    pagEnd,
    onPagEndChange,
    pagScale,
    onPagScaleChange,
    cfgIntervalStart,
    onCfgIntervalStartChange,
    cfgIntervalEnd,
    onCfgIntervalEndChange,
    isTurbo = false,
    guidanceScaleText,
    onGuidanceScaleTextChange,
    guidanceScaleLyric,
    onGuidanceScaleLyricChange,
    apgMomentum,
    onApgMomentumChange,
    apgNormThreshold,
    onApgNormThresholdChange,
    omegaScale,
    onOmegaScaleChange,
    ergScale,
    onErgScaleChange,
}) => {
    const { t } = useI18n();

    return (
        <div>
            <button
                type="button"
                onClick={onToggle}
                className={`w-full flex items-center justify-between px-4 py-3 bg-white dark:bg-suno-card border border-zinc-200 dark:border-white/5 text-sm font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors ${isOpen ? 'rounded-t-xl rounded-b-none border-b-0' : 'rounded-xl'}`}
            >
                <span className="flex items-center gap-2">
                    <Crosshair size={16} className="text-pink-500" />
                    {t('guidanceSettings')}
                </span>
                <ChevronDown size={18} className={`text-pink-500 chevron-icon ${isOpen ? 'rotated' : ''}`} />
            </button>

            {isOpen && (
                <div className="bg-white dark:bg-suno-card rounded-b-xl rounded-t-none border border-t-0 border-zinc-200 dark:border-white/5 p-4 space-y-4">
                    {/* Turbo model notice */}
                    {isTurbo && (
                        <div className="flex items-start gap-2 p-2.5 bg-amber-50 dark:bg-amber-900/15 border border-amber-200 dark:border-amber-500/20 rounded-lg">
                            <svg className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                            <p className="text-[11px] text-amber-700 dark:text-amber-300">Turbo models don't use classifier-free guidance. These settings only apply to base/SFT models.</p>
                        </div>
                    )}
                    {/* Guidance Scale */}
                    <EditableSlider
                        label={t('guidanceScale')}
                        value={guidanceScale}
                        min={1}
                        max={15}
                        step={0.5}
                        onChange={onGuidanceScaleChange}
                        formatDisplay={(val) => val.toFixed(1)}
                        helpText={t('howCloselyFollowPrompt')}
                        title={t('guidanceScaleTooltip')}
                    />

                    {/* Guidance Mode */}
                    <div className="space-y-1.5">
                        <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400" title={t('guidanceModeTooltip')}>{t('guidanceMode')}</label>
                        <select
                            value={guidanceMode}
                            onChange={(e) => onGuidanceModeChange(e.target.value as GuidanceMode)}
                            className="w-full bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-xl px-2 py-1.5 text-xs text-zinc-900 dark:text-white focus:outline-none focus:border-pink-500 dark:focus:border-pink-500 transition-colors cursor-pointer [&>option]:bg-white [&>option]:dark:bg-zinc-800 [&>option]:text-zinc-900 [&>option]:dark:text-white"
                        >
                            <option value="apg">{t('guidanceApg')}</option>
                            <option value="adg">{t('guidanceAdg')}</option>
                            <option value="pag">{t('guidancePag')}</option>
                            <option value="cfg_pp">{t('guidanceCfgPp')}</option>
                            <option value="dynamic_cfg">{t('guidanceDynamic')}</option>
                            <option value="rescaled_cfg">{t('guidanceRescaled')}</option>
                        </select>
                        <p className="text-[11px] text-zinc-400 dark:text-zinc-500 mt-1">
                            {guidanceMode === 'apg' && t('guidanceApgDesc')}
                            {guidanceMode === 'adg' && t('guidanceAdgDesc')}
                            {guidanceMode === 'pag' && t('guidancePagDesc')}
                            {guidanceMode === 'cfg_pp' && t('guidanceCfgPpDesc')}
                            {guidanceMode === 'dynamic_cfg' && t('guidanceDynamicDesc')}
                            {guidanceMode === 'rescaled_cfg' && t('guidanceRescaledDesc')}
                        </p>
                    </div>

                    {/* PAG Sub-Controls */}
                    {guidanceMode === 'pag' && (
                        <div className="space-y-3 p-3 bg-amber-50/50 dark:bg-amber-900/10 border border-amber-200/50 dark:border-amber-500/20 rounded-xl">
                            {[
                                { label: 'PAG Start', value: pagStart, onChange: onPagStartChange },
                                { label: 'PAG End', value: pagEnd, onChange: onPagEndChange },
                                { label: 'PAG Scale', value: pagScale, onChange: onPagScaleChange },
                            ].map(({ label, value, onChange }) => (
                                <div key={label} className="space-y-1">
                                    <div className="flex justify-between">
                                        <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{label}</label>
                                        <span className="text-xs text-zinc-500">{value.toFixed(2)}</span>
                                    </div>
                                    <input type="range" min="0" max="1" step="0.05" value={value} onChange={(e) => onChange(parseFloat(e.target.value))} className="w-full accent-pink-500" />
                                </div>
                            ))}
                        </div>
                    )}

                    {/* CFG Interval */}
                    <div className="space-y-1.5">
                        <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{t('cfgInterval')}</label>
                        <p className="text-[11px] text-zinc-400 dark:text-zinc-500">{t('cfgIntervalHelp')}</p>
                        <div className="grid grid-cols-2 gap-3">
                            <EditableSlider
                                label={t('cfgIntervalStart')}
                                value={cfgIntervalStart}
                                min={0}
                                max={1}
                                step={0.05}
                                onChange={onCfgIntervalStartChange}
                                formatDisplay={(val) => val.toFixed(2)}
                                title={t('cfgIntervalStartTooltip')}
                            />
                            <EditableSlider
                                label={t('cfgIntervalEnd')}
                                value={cfgIntervalEnd}
                                min={0}
                                max={1}
                                step={0.05}
                                onChange={onCfgIntervalEndChange}
                                formatDisplay={(val) => val.toFixed(2)}
                                title={t('cfgIntervalEndTooltip')}
                            />
                        </div>
                    </div>

                    {/* Advanced Guidance Parameters */}
                    <div className="rounded-lg border border-indigo-500/20 bg-indigo-500/5 p-3 space-y-3">
                        <p className="text-[10px] font-semibold text-indigo-400 uppercase tracking-wider">Advanced Guidance</p>
                        <div className="grid grid-cols-2 gap-3">
                            <EditableSlider
                                label="Text Scale"
                                value={guidanceScaleText ?? 0}
                                min={0} max={15} step={0.5}
                                onChange={(v) => onGuidanceScaleTextChange?.(v)}
                                formatDisplay={(v) => v.toFixed(1)}
                                helpText="Independent text guidance (0 = use main)"
                            />
                            <EditableSlider
                                label="Lyric Scale"
                                value={guidanceScaleLyric ?? 0}
                                min={0} max={15} step={0.5}
                                onChange={(v) => onGuidanceScaleLyricChange?.(v)}
                                formatDisplay={(v) => v.toFixed(1)}
                                helpText="Independent lyric guidance (0 = use main)"
                            />
                        </div>
                        {guidanceMode === 'apg' && (
                            <div className="grid grid-cols-2 gap-3">
                                <EditableSlider
                                    label="APG Momentum"
                                    value={apgMomentum ?? 0}
                                    min={0} max={1} step={0.01}
                                    onChange={(v) => onApgMomentumChange?.(v)}
                                    formatDisplay={(v) => v.toFixed(2)}
                                    helpText="Momentum parameter (0 = default)"
                                />
                                <EditableSlider
                                    label="APG Norm Threshold"
                                    value={apgNormThreshold ?? 0}
                                    min={0} max={10} step={0.1}
                                    onChange={(v) => onApgNormThresholdChange?.(v)}
                                    formatDisplay={(v) => v.toFixed(1)}
                                    helpText="Norm threshold (0 = default)"
                                />
                            </div>
                        )}
                        <div className="grid grid-cols-2 gap-3">
                            <EditableSlider
                                label="Omega Scale"
                                value={omegaScale ?? 1}
                                min={0} max={3} step={0.1}
                                onChange={(v) => onOmegaScaleChange?.(v)}
                                formatDisplay={(v) => v.toFixed(1)}
                                helpText="Prompt reweighting (1.0 = normal)"
                            />
                            <EditableSlider
                                label="ERG Scale"
                                value={ergScale ?? 1}
                                min={0} max={3} step={0.1}
                                onChange={(v) => onErgScaleChange?.(v)}
                                formatDisplay={(v) => v.toFixed(1)}
                                helpText="ERG reweighting (1.0 = normal)"
                            />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default GuidanceSettingsAccordion;
