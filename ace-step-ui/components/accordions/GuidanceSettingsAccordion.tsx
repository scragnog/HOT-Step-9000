import React from 'react';
import { Crosshair, ChevronDown, RotateCcw } from 'lucide-react';
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

    // Guidance Envelope Parameters
    guidanceIntervalDecay?: number;
    onGuidanceIntervalDecayChange?: (val: number) => void;
    minGuidanceScale?: number;
    onMinGuidanceScaleChange?: (val: number) => void;

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
    guidanceIntervalDecay,
    onGuidanceIntervalDecayChange,
    minGuidanceScale,
    onMinGuidanceScaleChange,
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

                    {/* Guidance Envelope Settings */}
                    {(guidanceIntervalDecay !== undefined && minGuidanceScale !== undefined) && (
                        <div className="space-y-3 p-3 bg-zinc-50/50 dark:bg-zinc-900/10 border border-zinc-200/50 dark:border-white/5 rounded-xl">
                            <h4 className="text-xs font-semibold text-zinc-700 dark:text-zinc-300" title="Controls how guidance strength changes over the diffusion process. Combines a decay rate with a minimum floor to shape the guidance profile.">Guidance Envelope</h4>
                            <EditableSlider
                                label="Interval Decay"
                                value={guidanceIntervalDecay}
                                min={0}
                                max={1}
                                step={0.05}
                                onChange={onGuidanceIntervalDecayChange!}
                                formatDisplay={(val) => val.toFixed(2)}
                                helpText={
                                    guidanceIntervalDecay === 0
                                        ? 'Off — constant guidance strength throughout all steps'
                                        : guidanceIntervalDecay <= 0.3
                                            ? 'Gentle taper — guidance slowly decreases toward the end'
                                            : guidanceIntervalDecay <= 0.6
                                                ? 'Moderate — guidance fades noticeably in later steps'
                                                : 'Aggressive — guidance drops quickly to the minimum floor'
                                }
                                title="Rate at which guidance strength decreases over diffusion steps. Higher values mean guidance weakens faster, giving the model more creative freedom in later steps while maintaining prompt adherence early on."
                            />
                            <EditableSlider
                                label="Minimum Scale"
                                value={minGuidanceScale}
                                min={1}
                                max={15}
                                step={0.5}
                                onChange={onMinGuidanceScaleChange!}
                                formatDisplay={(val) => val.toFixed(1)}
                                helpText={
                                    guidanceIntervalDecay === 0
                                        ? 'No effect — decay is off, guidance stays constant'
                                        : minGuidanceScale >= guidanceScale
                                            ? 'At or above main scale — decay has no effect'
                                            : `Guidance will taper from ${guidanceScale.toFixed(1)} down to ${minGuidanceScale.toFixed(1)}`
                                }
                                title="The lowest value guidance can reach when decay is active. Acts as a floor — guidance will decrease from the main scale toward this value but never below it."
                            />
                        </div>
                    )}

                    {/* PAG Sub-Controls */}
                    {guidanceMode === 'pag' && (
                        <div className="space-y-3 p-3 bg-amber-50/50 dark:bg-amber-900/10 border border-amber-200/50 dark:border-amber-500/20 rounded-xl">
                            <EditableSlider
                                label="PAG Start"
                                value={pagStart}
                                min={0} max={1} step={0.05}
                                onChange={onPagStartChange}
                                formatDisplay={(v) => v.toFixed(2)}
                                helpText={
                                    pagStart === 0
                                        ? 'PAG active from the very first diffusion step'
                                        : `PAG begins at ${Math.round(pagStart * 100)}% through the diffusion process`
                                }
                                title="The diffusion timestep (0–1) where PAG starts taking effect. Earlier values (closer to 0) apply PAG sooner, influencing overall structure. Later values apply PAG only during refinement."
                            />
                            <EditableSlider
                                label="PAG End"
                                value={pagEnd}
                                min={0} max={1} step={0.05}
                                onChange={onPagEndChange}
                                formatDisplay={(v) => v.toFixed(2)}
                                helpText={
                                    pagEnd >= 1
                                        ? 'PAG active until the final diffusion step'
                                        : `PAG stops at ${Math.round(pagEnd * 100)}% through the process`
                                }
                                title="The diffusion timestep (0–1) where PAG stops. Setting this below 1.0 disables PAG for the final refinement steps, which can improve fine detail."
                            />
                            <EditableSlider
                                label="PAG Scale"
                                value={pagScale}
                                min={0} max={1} step={0.05}
                                onChange={onPagScaleChange}
                                formatDisplay={(v) => v.toFixed(2)}
                                helpText={
                                    pagScale === 0
                                        ? 'Disabled — PAG has no effect'
                                        : pagScale <= 0.3
                                            ? 'Subtle — light structural consistency boost'
                                            : pagScale <= 0.6
                                                ? 'Moderate — noticeable structural guidance'
                                                : 'Strong — heavy structural consistency enforcement'
                                }
                                title="Strength of Perturbed Attention Guidance. PAG improves structural consistency by comparing normal vs self-attention-perturbed outputs. Higher values enforce more consistent structure."
                            />
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
                        <div className="flex items-center justify-between">
                            <p className="text-[10px] font-semibold text-indigo-400 uppercase tracking-wider">Advanced Guidance</p>
                            <button
                                type="button"
                                onClick={() => {
                                    onGuidanceScaleTextChange?.(0);
                                    onGuidanceScaleLyricChange?.(0);
                                    onApgMomentumChange?.(0);
                                    onApgNormThresholdChange?.(2.5);
                                    onOmegaScaleChange?.(1.0);
                                    onErgScaleChange?.(1.0);
                                }}
                                className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] text-indigo-400 hover:text-indigo-300 hover:bg-indigo-500/20 transition-colors"
                                title="Reset all advanced guidance parameters to model defaults"
                            >
                                <RotateCcw className="w-3 h-3" />
                                Reset
                            </button>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                            <EditableSlider
                                label="Text Scale"
                                value={guidanceScaleText ?? 0}
                                min={0} max={15} step={0.5}
                                onChange={(v) => onGuidanceScaleTextChange?.(v)}
                                formatDisplay={(v) => v.toFixed(1)}
                                helpText={
                                    (guidanceScaleText ?? 0) === 0
                                        ? 'Using main guidance scale for text/style prompts'
                                        : (guidanceScaleText ?? 0) <= 3
                                            ? 'Subtle — loose interpretation, more creative freedom'
                                            : (guidanceScaleText ?? 0) <= 7
                                                ? 'Balanced — follows text prompts without being rigid'
                                                : 'Strong — tightly follows text/style, may reduce variety'
                                }
                                title="Controls how strongly the text/style description guides generation, independently of the main guidance scale. Set to 0 to defer to the main Guidance Scale slider."
                            />
                            <EditableSlider
                                label="Lyric Scale"
                                value={guidanceScaleLyric ?? 0}
                                min={0} max={15} step={0.5}
                                onChange={(v) => onGuidanceScaleLyricChange?.(v)}
                                formatDisplay={(v) => v.toFixed(1)}
                                helpText={
                                    (guidanceScaleLyric ?? 0) === 0
                                        ? 'Using main guidance scale for lyric content'
                                        : (guidanceScaleLyric ?? 0) <= 3
                                            ? 'Loose — lyrics lightly influence melody & phrasing'
                                            : (guidanceScaleLyric ?? 0) <= 7
                                                ? 'Balanced — lyrics shape phrasing and rhythm'
                                                : 'Strict — tightly follows lyric structure and syllable count'
                                }
                                title="Controls how strongly the lyric content guides generation, independently of the main guidance scale. Set to 0 to defer to the main Guidance Scale slider."
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
                                    helpText={
                                        (apgMomentum ?? 0) === 0
                                            ? 'Default — no momentum smoothing applied'
                                            : (apgMomentum ?? 0) <= 0.3
                                                ? 'Light smoothing — subtle guidance stabilisation'
                                                : (apgMomentum ?? 0) <= 0.7
                                                    ? 'Medium smoothing — reduces guidance oscillations'
                                                    : 'Heavy smoothing — very stable but may blur fine detail'
                                    }
                                    title="Applies momentum-based smoothing to APG guidance updates across diffusion steps. Higher values carry forward previous guidance direction, reducing jitter but potentially softening transitions."
                                />
                                <EditableSlider
                                    label="APG Norm Threshold"
                                    value={apgNormThreshold ?? 0}
                                    min={0} max={10} step={0.1}
                                    onChange={(v) => onApgNormThresholdChange?.(v)}
                                    formatDisplay={(v) => v.toFixed(1)}
                                    helpText={
                                        (apgNormThreshold ?? 2.5) <= 1
                                            ? 'Tight clamp — strongly limits guidance corrections'
                                            : (apgNormThreshold ?? 2.5) <= 3
                                                ? 'Default range — balanced gradient clamping'
                                                : (apgNormThreshold ?? 2.5) <= 5
                                                    ? 'Moderate — permits larger corrections'
                                                    : 'Permissive — only limits very large corrections'
                                    }
                                    title="Caps the magnitude of APG guidance corrections. Prevents individual steps from overshooting, which can cause artifacts. Lower values = more conservative guidance, higher values = more freedom."
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
                                helpText={
                                    (omegaScale ?? 1) === 0
                                        ? 'Disabled — prompt weighting has no effect'
                                        : (omegaScale ?? 1) < 0.8
                                            ? 'Reduced — de-emphasises prompt, more diffusion freedom'
                                            : (omegaScale ?? 1) <= 1.2
                                                ? 'Normal — standard prompt weighting balance'
                                                : 'Amplified — stronger prompt emphasis, tighter adherence'
                                }
                                title="Multiplier for prompt-based reweighting during the diffusion process. Values below 1.0 give the model more creative freedom; values above 1.0 force closer prompt adherence at the risk of reduced naturalness."
                            />
                            <EditableSlider
                                label="ERG Scale"
                                value={ergScale ?? 1}
                                min={0} max={3} step={0.1}
                                onChange={(v) => onErgScaleChange?.(v)}
                                formatDisplay={(v) => v.toFixed(1)}
                                helpText={
                                    (ergScale ?? 1) === 0
                                        ? 'Disabled — no entropy regularisation'
                                        : (ergScale ?? 1) < 0.8
                                            ? 'Low — sharper, more decisive predictions'
                                            : (ergScale ?? 1) <= 1.2
                                                ? 'Normal — balanced prediction diversity'
                                                : 'High — broader predictions, more sonic variety'
                                }
                                title="Entropy Regularisation Guidance. Controls how spread out the model's predictions are across possible outputs. Lower values produce more focused/peaked results; higher values encourage more diverse, exploratory generation."
                            />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default GuidanceSettingsAccordion;
