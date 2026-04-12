import React from 'react';
import { Settings2, ChevronDown, Cpu, Dices, Hash, RotateCcw } from 'lucide-react';
import { useI18n } from '../../context/I18nContext';
import { usePersistedState } from '../../hooks/usePersistedState';
import { EditableSlider } from '../EditableSlider';
import GuidanceSettingsAccordion from './GuidanceSettingsAccordion';
import LmCotAccordion from './LmCotAccordion';
import ExpertControlsAccordion from './ExpertControlsAccordion';

type GuidanceMode = 'apg' | 'adg' | 'pag' | 'cfg_pp' | 'dynamic_cfg' | 'rescaled_cfg';

interface GenerationSettingsAccordionProps {
    isOpen: boolean;
    onToggle: () => void;
    // Batch / Bulk
    batchSize: number;
    onBatchSizeChange: (val: number) => void;
    bulkCount: number;
    onBulkCountChange: (val: number) => void;
    // Seed
    seed: number;
    onSeedChange: (val: number) => void;
    randomSeed: boolean;
    onRandomSeedToggle: () => void;
    // Shift
    shift: number;
    onShiftChange: (val: number) => void;
    // Model type
    isTurbo: boolean;
    // Inference
    inferenceSteps: number;
    onInferenceStepsChange: (val: number) => void;
    inferMethod: 'ode' | 'euler' | 'heun' | 'dpm2m' | 'dpm3m' | 'rk4' | 'jkass_quality' | 'jkass_fast' | 'stork2' | 'stork4';
    onInferMethodChange: (val: 'ode' | 'euler' | 'heun' | 'dpm2m' | 'dpm3m' | 'rk4' | 'jkass_quality' | 'jkass_fast' | 'stork2' | 'stork4') => void;
    scheduler: string;
    onSchedulerChange: (val: string) => void;
    // Audio Format
    audioFormat: 'mp3' | 'flac' | 'wav' | 'opus';
    onAudioFormatChange: (val: 'mp3' | 'flac' | 'wav' | 'opus') => void;
    // Guidance
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
    guidanceIntervalDecay?: number;
    onGuidanceIntervalDecayChange?: (val: number) => void;
    minGuidanceScale?: number;
    onMinGuidanceScaleChange?: (val: number) => void;
    // LM / CoT
    thinking: boolean;
    onThinkingToggle: () => void;
    loraLoaded: boolean;
    lmBackend: 'pt' | 'vllm' | 'custom-vllm';
    onLmBackendChange: (val: 'pt' | 'vllm' | 'custom-vllm') => void;
    lmModel: string;
    onLmModelChange: (val: string) => void;
    isLmSwitching?: boolean;
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
    allowLmBatch: boolean;
    onAllowLmBatchToggle: () => void;
    useCotMetas: boolean;
    onUseCotMetasToggle: () => void;
    useCotCaption: boolean;
    onUseCotCaptionToggle: () => void;
    useCotLanguage: boolean;
    onUseCotLanguageToggle: () => void;
    lmBatchChunkSize: number;
    onLmBatchChunkSizeChange: (val: number) => void;
    constrainedDecodingDebug: boolean;
    onConstrainedDecodingDebugToggle: () => void;
    isFormatCaption: boolean;
    onIsFormatCaptionToggle: () => void;

    // Expert Controls
    uploadError: string | null;
    audioCodes: string;
    onAudioCodesChange: (val: string) => void;
    instruction: string;
    onInstructionChange: (val: string) => void;
    customTimesteps: string;
    onCustomTimestepsChange: (val: string) => void;
    trackName: string;
    onTrackNameChange: (val: string) => void;
    completeTrackClasses: string;
    onCompleteTrackClassesChange: (val: string) => void;
    autogen: boolean;
    onToggleAutogen: () => void;
    getLrc: boolean;
    onToggleGetLrc: () => void;
    /** When true, skip the accordion header — used inside DrawerContainers */
    embedded?: boolean;

    // Anti-Autotune
    antiAutotune?: number;
    onAntiAutotuneChange?: (val: number) => void;

    // JKASS Fast solver parameters
    beatStability?: number;
    onBeatStabilityChange?: (val: number) => void;
    frequencyDamping?: number;
    onFrequencyDampingChange?: (val: number) => void;
    temporalSmoothing?: number;
    onTemporalSmoothingChange?: (val: number) => void;

    // STORK solver parameters
    storkSubsteps?: number;
    onStorkSubstepsChange?: (val: number) => void;

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

export const GenerationSettingsAccordion: React.FC<GenerationSettingsAccordionProps> = (props) => {
    const { t } = useI18n();

    const [showInferenceSettings, setShowInferenceSettings] = usePersistedState('acestep-showInferenceSettings', false);
    const [showGuidancePanel, setShowGuidancePanel] = usePersistedState('acestep-showGuidancePanel', false);
    const [showLmCot, setShowLmCot] = usePersistedState('ace-showCotSection', false);
    const [showLmParams, setShowLmParams] = usePersistedState('acestep-showLmParams', false);
    const [showExpert, setShowExpert] = usePersistedState('acestep-showExpertPanel', false);

    const selectClass = "w-full bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-xl px-2 py-1.5 text-xs text-zinc-900 dark:text-white focus:outline-none focus:border-pink-500 dark:focus:border-pink-500 transition-colors cursor-pointer [&>option]:bg-white [&>option]:dark:bg-zinc-800 [&>option]:text-zinc-900 [&>option]:dark:text-white";

    const content = (
                <div className={props.embedded ? "space-y-5" : "bg-white dark:bg-suno-card rounded-b-xl rounded-t-none border border-t-0 border-zinc-200 dark:border-white/5 p-4 space-y-5"}>

                    {/* Batch Size */}
                    <EditableSlider
                        label={t('batchSize')}
                        value={props.batchSize}
                        min={1}
                        max={8}
                        step={1}
                        onChange={props.onBatchSizeChange}
                        helpText={t('numberOfVariations')}
                        title={t('batchSizeTooltip')}
                    />

                    {/* Bulk Generate */}
                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{t('bulkGenerate')}</label>
                            <span className="text-xs font-mono text-zinc-900 dark:text-white bg-zinc-100 dark:bg-black/20 px-2 py-0.5 rounded">
                                {props.bulkCount} {t(props.bulkCount === 1 ? 'job' : 'jobs')}
                            </span>
                        </div>
                        <div className="flex items-center gap-1">
                            {[1, 2, 3, 5, 10].map((count) => (
                                <button
                                    key={count}
                                    type="button"
                                    onClick={() => props.onBulkCountChange(count)}
                                    className={`flex-1 py-2 rounded-lg text-xs font-bold transition-all ${props.bulkCount === count
                                        ? 'bg-gradient-to-r from-orange-500 to-pink-600 text-white shadow-md'
                                        : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-700'
                                        }`}
                                >
                                    {count}
                                </button>
                            ))}
                        </div>
                        <p className="text-[10px] text-zinc-500">{t('queueMultipleJobs')}</p>
                    </div>

                    {/* Seed */}
                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Dices size={14} className="text-zinc-500" />
                                <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400" title={t('seedTooltip')}>{t('seed')}</span>
                            </div>
                            <button
                                type="button"
                                onClick={props.onRandomSeedToggle}
                                className={`w-10 h-5 rounded-full flex items-center transition-colors duration-200 px-0.5 border border-zinc-200 dark:border-white/5 ${props.randomSeed ? 'bg-pink-600' : 'bg-zinc-300 dark:bg-black/40'}`}
                            >
                                <div className={`w-4 h-4 rounded-full bg-white transform transition-transform duration-200 shadow-sm ${props.randomSeed ? 'translate-x-5' : 'translate-x-0'}`} />
                            </button>
                        </div>
                        <div className="flex items-center gap-2">
                            <Hash size={14} className="text-zinc-500" />
                            <input
                                type="number"
                                value={props.seed}
                                onChange={(e) => props.onSeedChange(Number(e.target.value))}
                                placeholder={t('enterFixedSeed')}
                                disabled={props.randomSeed}
                                onWheel={(e) => (e.currentTarget as HTMLInputElement).blur()}
                                className={`flex-1 bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg px-3 py-1.5 text-xs text-zinc-900 dark:text-white focus:outline-none appearance-none [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none ${props.randomSeed ? 'opacity-40 cursor-not-allowed' : ''}`}
                            />
                        </div>
                        <p className="text-[10px] text-zinc-500">{props.randomSeed ? t('randomSeedRecommended') : t('fixedSeedReproducible')}</p>
                    </div>

                    {/* Shift */}
                    <div className="space-y-1.5">
                        <div className="flex items-center justify-between">
                            <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400" title={'Timestep schedule warping factor. Turbo models are trained with shift=3.0; base/SFT models default to 1.0.'}>
                                {t('shift')}
                            </span>
                            {!props.isTurbo && (
                                <button
                                    type="button"
                                    onClick={() => {
                                        if (props.shift < 0) {
                                            // Switch from auto to manual — restore a sensible default
                                            props.onShiftChange(3.0);
                                        } else {
                                            // Switch to auto mode
                                            props.onShiftChange(-1);
                                        }
                                    }}
                                    className={`px-2 py-0.5 rounded-md text-[10px] font-semibold transition-all ${
                                        props.shift < 0
                                            ? 'bg-gradient-to-r from-emerald-500 to-teal-500 text-white shadow-sm'
                                            : 'bg-zinc-100 dark:bg-black/20 text-zinc-500 dark:text-zinc-400 hover:text-emerald-500 dark:hover:text-emerald-400'
                                    }`}
                                    title="Auto mode: dynamically adjusts shift based on song duration and step count"
                                >
                                    Auto
                                </button>
                            )}
                        </div>
                        {props.shift < 0 ? (
                            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                                <span className="text-[10px] text-emerald-400">🎯 Dynamic shift — auto-adjusts based on duration and step count</span>
                            </div>
                        ) : (
                            <EditableSlider
                                label=""
                                value={props.shift}
                                min={1}
                                max={5}
                                step={0.1}
                                onChange={props.onShiftChange}
                                formatDisplay={(val) => val.toFixed(1)}
                                helpText={props.isTurbo ? undefined : 'Controls how the model distributes denoising effort. Higher = more focus on structure, less on fine detail.'}
                                disabled={props.isTurbo}
                                disabledReason={props.isTurbo ? 'Locked to 3.0 — turbo models are trained with this value' : undefined}
                            />
                        )}
                    </div>

                    {/* Inference Settings (nested accordion) */}
                    <div>
                        <button
                            type="button"
                            onClick={() => setShowInferenceSettings(!showInferenceSettings)}
                            className={`w-full flex items-center justify-between px-4 py-3 bg-white dark:bg-suno-card border border-zinc-200 dark:border-white/5 text-sm font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors ${showInferenceSettings ? 'rounded-t-xl rounded-b-none border-b-0' : 'rounded-xl'}`}
                        >
                            <span className="flex items-center gap-2">
                                <Cpu size={16} className="text-pink-500" />
                                {t('inferenceSettings')}
                            </span>
                            <ChevronDown size={18} className={`text-pink-500 chevron-icon ${showInferenceSettings ? 'rotated' : ''}`} />
                        </button>

                        {showInferenceSettings && (
                            <div className="bg-white dark:bg-suno-card rounded-b-xl rounded-t-none border border-t-0 border-zinc-200 dark:border-white/5 p-4 space-y-4">
                                <EditableSlider
                                    label={t('inferenceSteps')}
                                    value={props.inferenceSteps}
                                    min={4}
                                    max={props.isTurbo ? 20 : 200}
                                    step={1}
                                    onChange={props.onInferenceStepsChange}
                                    helpText={props.isTurbo ? 'Turbo models are optimised for 8 steps (max 20)' : t('moreStepsBetterQuality')}
                                    title={t('inferenceStepsTooltip')}
                                />
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="space-y-1.5">
                                        <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{t('audioFormat')}</label>
                                        <select value={props.audioFormat} onChange={(e) => props.onAudioFormatChange(e.target.value as 'mp3' | 'flac' | 'wav' | 'opus')} className={selectClass}>
                                            <option value="mp3" title={t('audioFormatMp3Desc')}>{t('mp3Smaller')}</option>
                                            <option value="flac" title={t('audioFormatFlacDesc')}>{t('flacLossless')}</option>
                                            <option value="wav" title={t('audioFormatWavDesc')}>WAV</option>
                                            <option value="opus" title={t('audioFormatOpusDesc')}>Opus</option>
                                        </select>
                                        <p className="text-[10px] leading-tight text-zinc-500 dark:text-zinc-500">
                                            {t(({ mp3: 'audioFormatMp3Desc', flac: 'audioFormatFlacDesc', wav: 'audioFormatWavDesc', opus: 'audioFormatOpusDesc' } as const)[props.audioFormat])}
                                        </p>
                                    </div>
                                    <div className="space-y-1.5">
                                        <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{t('inferMethod')}</label>
                                        <select value={props.inferMethod} onChange={(e) => props.onInferMethodChange(e.target.value as any)} className={selectClass}>
                                            <option value="ode" title={t('solverEulerDesc')}>{t('solverEuler')}</option>
                                            <option value="heun" title={t('solverHeunDesc')}>{t('solverHeun')}</option>
                                            <option value="dpm2m" title={t('solverDpm2mDesc')}>{t('solverDpm2m')}</option>
                                            <option value="dpm3m" title={t('solverDpm3mDesc')}>{t('solverDpm3m')}</option>
                                            <option value="rk4" title={t('solverRk4Desc')}>{t('solverRk4')}</option>
                                            <option value="jkass_quality" title="JKASS Quality: Heun with derivative averaging (2 NFE)">JKASS Quality</option>
                                            <option value="jkass_fast" title="JKASS Fast: Euler with momentum, frequency damping, and temporal smoothing (1 NFE)">JKASS Fast</option>
                                            <option value="stork2" title="STORK 2: 2nd-order stabilized Runge-Kutta-Gegenbauer with Taylor velocity (1 NFE)">STORK 2</option>
                                            <option value="stork4" title="STORK 4: 4th-order ROCK4 with precomputed Chebyshev coefficients (1 NFE)">STORK 4</option>
                                        </select>
                                        <p className="text-[10px] leading-tight text-zinc-500 dark:text-zinc-500">
                                            {({ ode: t('solverEulerDesc'), euler: t('solverEulerDesc'), heun: t('solverHeunDesc'), dpm2m: t('solverDpm2mDesc'), dpm3m: t('solverDpm3mDesc'), rk4: t('solverRk4Desc'), jkass_quality: 'Heun with derivative averaging — smooth, high-accuracy results (2× cost)', jkass_fast: 'Euler with beat stability, frequency damping & temporal smoothing', stork2: 'Stabilized 2nd-order ODE solver for stiff flow matching (1 NFE)', stork4: 'Stabilized 4th-order ODE solver with ROCK4 sub-stepping (1 NFE)' } as Record<string, string>)[props.inferMethod] || ''}
                                        </p>
                                    </div>
                                </div>

                                {/* JKASS Fast Sub-Controls */}
                                {props.inferMethod === 'jkass_fast' && (
                                    <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 space-y-3">
                                        <div className="flex items-center justify-between">
                                            <p className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider">JKASS Fast Controls</p>
                                            <button
                                                type="button"
                                                onClick={() => {
                                                    props.onBeatStabilityChange?.(0.25);
                                                    props.onFrequencyDampingChange?.(0.4);
                                                    props.onTemporalSmoothingChange?.(0.13);
                                                }}
                                                className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] text-amber-400 hover:text-amber-300 hover:bg-amber-500/20 transition-colors"
                                                title="Reset JKASS Fast parameters to recommended defaults"
                                            >
                                                <RotateCcw className="w-3 h-3" />
                                                Reset
                                            </button>
                                        </div>
                                        <EditableSlider
                                            label="Beat Stability"
                                            value={props.beatStability ?? 0}
                                            min={0} max={1} step={0.01}
                                            onChange={(v) => props.onBeatStabilityChange?.(v)}
                                            formatDisplay={(v) => v.toFixed(2)}
                                            helpText="Momentum blending with previous step — smooths rhythmic elements"
                                        />
                                        <EditableSlider
                                            label="Frequency Damping"
                                            value={props.frequencyDamping ?? 0}
                                            min={0} max={5} step={0.1}
                                            onChange={(v) => props.onFrequencyDampingChange?.(v)}
                                            formatDisplay={(v) => v.toFixed(1)}
                                            helpText="Exponential decay on high-frequency bins — tames harsh overtones"
                                        />
                                        <EditableSlider
                                            label="Temporal Smoothing"
                                            value={props.temporalSmoothing ?? 0}
                                            min={0} max={1} step={0.01}
                                            onChange={(v) => props.onTemporalSmoothingChange?.(v)}
                                            formatDisplay={(v) => v.toFixed(2)}
                                            helpText="Smoothing kernel across time axis — reduces temporal jitter"
                                        />
                                    </div>
                                )}

                                {/* STORK Sub-Steps Control */}
                                {(props.inferMethod === 'stork2' || props.inferMethod === 'stork4') && (
                                    <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/5 p-3 space-y-3">
                                        <div className="flex items-center justify-between">
                                            <p className="text-[10px] font-semibold text-cyan-400 uppercase tracking-wider">STORK Controls</p>
                                            <button
                                                type="button"
                                                onClick={() => props.onStorkSubstepsChange?.(50)}
                                                className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/20 transition-colors"
                                                title="Reset STORK sub-steps to paper default (50)"
                                            >
                                                <RotateCcw className="w-3 h-3" />
                                                Reset
                                            </button>
                                        </div>
                                        <EditableSlider
                                            label="Sub-Steps"
                                            value={props.storkSubsteps ?? 10}
                                            min={2} max={50} step={1}
                                            onChange={(v) => props.onStorkSubstepsChange?.(v)}
                                            formatDisplay={(v) => String(v)}
                                            helpText="Chebyshev sub-iterations per step. Auto-adapts downward if unstable. Higher = more stability work but may not help (default: 10)"
                                        />
                                    </div>
                                )}

                                {/* Anti-Autotune */}
                                <EditableSlider
                                    label="Anti-Autotune"
                                    value={props.antiAutotune ?? 0}
                                    min={0} max={1} step={0.01}
                                    onChange={(v) => props.onAntiAutotuneChange?.(v)}
                                    formatDisplay={(v) => v.toFixed(2)}
                                    helpText="Spectral smoothing to reduce robotic/autotuned vocal artifacts (0=off, 1=full)"
                                />
                                <div className="col-span-2 grid grid-cols-2 gap-3">
                                    <div className="space-y-1.5">
                                        <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{t('scheduler')}</label>
                                        <select value={
                                            props.scheduler.startsWith('composite') ? 'composite'
                                            : props.scheduler.startsWith('beta:') ? 'beta'
                                            : props.scheduler.startsWith('power:') ? 'power'
                                            : props.scheduler
                                        } onChange={(e) => {
                                            if (e.target.value === 'composite') {
                                                props.onSchedulerChange('composite:bong_tangent+linear:0.5:0.5');
                                            } else if (e.target.value === 'beta') {
                                                props.onSchedulerChange('beta:0.50:0.70');
                                            } else if (e.target.value === 'power') {
                                                props.onSchedulerChange('power:2.00');
                                            } else {
                                                props.onSchedulerChange(e.target.value);
                                            }
                                        }} className={selectClass}>
                                            <option value="linear" title={t('schedulerLinearDesc')}>Linear</option>
                                            <option value="beta57" title={t('schedulerBeta57Desc')}>Beta 57</option>
                                            <option value="beta" title={t('schedulerBetaDesc')}>Beta (Custom)</option>
                                            <option value="cosine" title={t('schedulerCosineDesc')}>Cosine</option>
                                            <option value="power" title={t('schedulerPowerDesc')}>Power</option>
                                            <option value="ddim_uniform" title={t('schedulerDdimDesc')}>DDIM Uniform</option>
                                            <option value="sgm_uniform" title={t('schedulerSgmDesc')}>SGM-Uniform (Karras)</option>
                                            <option value="bong_tangent" title={t('schedulerBongDesc')}>Bong Tangent</option>
                                            <option value="linear_quadratic" title={t('schedulerLinQuadDesc')}>Linear-Quadratic</option>
                                            <option value="composite" title={t('schedulerCompositeDesc')}>Composite (2-Stage)</option>
                                        </select>
                                        <p className="text-[10px] leading-tight text-zinc-500 dark:text-zinc-500">
                                            {props.scheduler.startsWith('composite') ? t('schedulerCompositeDesc')
                                            : props.scheduler.startsWith('beta:') ? t('schedulerBetaDesc')
                                            : props.scheduler.startsWith('power:') ? t('schedulerPowerDesc')
                                            : t(({ linear: 'schedulerLinearDesc', beta57: 'schedulerBeta57Desc', cosine: 'schedulerCosineDesc', ddim_uniform: 'schedulerDdimDesc', sgm_uniform: 'schedulerSgmDesc', bong_tangent: 'schedulerBongDesc', linear_quadratic: 'schedulerLinQuadDesc' } as Record<string, string>)[props.scheduler] || 'schedulerLinearDesc')}
                                        </p>
                                    </div>
                                </div>
                                {/* Beta (Custom) Sub-Controls */}
                                {props.scheduler.startsWith('beta:') && (() => {
                                    const parts = props.scheduler.split(':');
                                    const alpha = parseFloat(parts[1] || '0.5');
                                    const betaParam = parseFloat(parts[2] || '0.7');
                                    const updateBeta = (a: number, b: number) => {
                                        props.onSchedulerChange(`beta:${a.toFixed(2)}:${b.toFixed(2)}`);
                                    };
                                    return (
                                        <div className="col-span-2 rounded-lg border border-teal-500/20 bg-teal-500/5 p-3 space-y-3">
                                            <div className="flex items-center justify-between">
                                                <p className="text-[10px] font-semibold text-teal-400 uppercase tracking-wider">Beta Distribution</p>
                                                <button
                                                    type="button"
                                                    onClick={() => updateBeta(0.5, 0.7)}
                                                    className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] text-teal-400 hover:text-teal-300 hover:bg-teal-500/20 transition-colors"
                                                    title="Reset to Beta 57 defaults (α=0.5, β=0.7)"
                                                >
                                                    <RotateCcw className="w-3 h-3" />
                                                    Reset
                                                </button>
                                            </div>
                                            <EditableSlider
                                                label="Alpha (α)"
                                                value={alpha}
                                                min={0.1} max={2.0} step={0.05}
                                                onChange={(v) => updateBeta(v, betaParam)}
                                                formatDisplay={(v) => v.toFixed(2)}
                                                helpText="Shape parameter. Lower = more density at edges, higher = more density in middle."
                                            />
                                            <EditableSlider
                                                label="Beta (β)"
                                                value={betaParam}
                                                min={0.1} max={2.0} step={0.05}
                                                onChange={(v) => updateBeta(alpha, v)}
                                                formatDisplay={(v) => v.toFixed(2)}
                                                helpText="Shape parameter. Lower = front-loaded (structure), higher = back-loaded (detail)."
                                            />
                                        </div>
                                    );
                                })()}
                                {/* Power Sub-Controls */}
                                {props.scheduler.startsWith('power:') && (() => {
                                    const parts = props.scheduler.split(':');
                                    const exponent = parseFloat(parts[1] || '2.0');
                                    return (
                                        <div className="col-span-2 rounded-lg border border-orange-500/20 bg-orange-500/5 p-3 space-y-3">
                                            <div className="flex items-center justify-between">
                                                <p className="text-[10px] font-semibold text-orange-400 uppercase tracking-wider">Power Law</p>
                                                <button
                                                    type="button"
                                                    onClick={() => props.onSchedulerChange('power:2.00')}
                                                    className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] text-orange-400 hover:text-orange-300 hover:bg-orange-500/20 transition-colors"
                                                    title="Reset exponent to default (2.0)"
                                                >
                                                    <RotateCcw className="w-3 h-3" />
                                                    Reset
                                                </button>
                                            </div>
                                            <EditableSlider
                                                label="Exponent"
                                                value={exponent}
                                                min={0.25} max={4.0} step={0.05}
                                                onChange={(v) => props.onSchedulerChange(`power:${v.toFixed(2)}`)}
                                                formatDisplay={(v) => v.toFixed(2)}
                                                helpText="p>1 = front-loaded (structural focus), p=1 = linear, p<1 = back-loaded (detail focus)."
                                            />
                                        </div>
                                    );
                                })()}
                                {/* Composite Scheduler Sub-Controls */}
                                {props.scheduler.startsWith('composite') && (() => {
                                    const parts = props.scheduler.split(':');
                                    const schedulerPair = (parts[1] || 'bong_tangent+linear').split('+');
                                    const stageA = schedulerPair[0] || 'bong_tangent';
                                    const stageB = schedulerPair[1] || 'linear';
                                    const crossover = parseFloat(parts[2] || '0.5');
                                    const split = parseFloat(parts[3] || '0.5');
                                    const updateComposite = (a: string, b: string, c: number, s: number) => {
                                        props.onSchedulerChange(`composite:${a}+${b}:${c.toFixed(2)}:${s.toFixed(2)}`);
                                    };
                                    const stageSelectClass = selectClass;
                                    return (
                                        <div className="col-span-2 rounded-lg border border-purple-500/20 bg-purple-500/5 p-3 space-y-3">
                                            <div className="grid grid-cols-2 gap-3">
                                                <div className="space-y-1">
                                                    <label className="text-[10px] font-medium text-purple-400">{t('compositeStageA')}</label>
                                                    <select value={stageA} onChange={(e) => updateComposite(e.target.value, stageB, crossover, split)} className={stageSelectClass}>
                                                        <option value="linear">Linear</option>
                                                        <option value="beta57">Beta 57</option>
                                                        <option value="cosine">Cosine</option>
                                                        <option value="ddim_uniform">DDIM Uniform</option>
                                                        <option value="sgm_uniform">SGM Uniform</option>
                                                        <option value="bong_tangent">Bong Tangent</option>
                                                        <option value="linear_quadratic">Linear-Quadratic</option>
                                                    </select>
                                                </div>
                                                <div className="space-y-1">
                                                    <label className="text-[10px] font-medium text-purple-400">{t('compositeStageB')}</label>
                                                    <select value={stageB} onChange={(e) => updateComposite(stageA, e.target.value, crossover, split)} className={stageSelectClass}>
                                                        <option value="linear">Linear</option>
                                                        <option value="beta57">Beta 57</option>
                                                        <option value="cosine">Cosine</option>
                                                        <option value="ddim_uniform">DDIM Uniform</option>
                                                        <option value="sgm_uniform">SGM Uniform</option>
                                                        <option value="bong_tangent">Bong Tangent</option>
                                                        <option value="linear_quadratic">Linear-Quadratic</option>
                                                    </select>
                                                </div>
                                            </div>
                                            <EditableSlider label={t('compositeCrossover')} value={crossover} onChange={(v) => updateComposite(stageA, stageB, v, split)} min={0.1} max={0.9} step={0.05} tooltip={t('compositeCrossoverDesc')} />
                                            <EditableSlider label={t('compositeSplit')} value={split} onChange={(v) => updateComposite(stageA, stageB, crossover, v)} min={0.1} max={0.9} step={0.05} tooltip={t('compositeSplitDesc')} />
                                        </div>
                                    );
                                })()}
                            </div>
                        )}
                    </div>

                    {/* Guidance Settings */}
                    <GuidanceSettingsAccordion
                        isOpen={showGuidancePanel}
                        onToggle={() => setShowGuidancePanel(!showGuidancePanel)}
                        guidanceScale={props.guidanceScale}
                        onGuidanceScaleChange={props.onGuidanceScaleChange}
                        guidanceMode={props.guidanceMode}
                        onGuidanceModeChange={props.onGuidanceModeChange}
                        pagStart={props.pagStart}
                        onPagStartChange={props.onPagStartChange}
                        pagEnd={props.pagEnd}
                        onPagEndChange={props.onPagEndChange}
                        pagScale={props.pagScale}
                        onPagScaleChange={props.onPagScaleChange}
                        cfgIntervalStart={props.cfgIntervalStart}
                        onCfgIntervalStartChange={props.onCfgIntervalStartChange}
                        cfgIntervalEnd={props.cfgIntervalEnd}
                        onCfgIntervalEndChange={props.onCfgIntervalEndChange}
                        guidanceIntervalDecay={props.guidanceIntervalDecay}
                        onGuidanceIntervalDecayChange={props.onGuidanceIntervalDecayChange}
                        minGuidanceScale={props.minGuidanceScale}
                        onMinGuidanceScaleChange={props.onMinGuidanceScaleChange}
                        isTurbo={props.isTurbo}
                        isThinking={props.thinking}
                        guidanceScaleText={props.guidanceScaleText}
                        onGuidanceScaleTextChange={props.onGuidanceScaleTextChange}
                        guidanceScaleLyric={props.guidanceScaleLyric}
                        onGuidanceScaleLyricChange={props.onGuidanceScaleLyricChange}
                        apgMomentum={props.apgMomentum}
                        onApgMomentumChange={props.onApgMomentumChange}
                        apgNormThreshold={props.apgNormThreshold}
                        onApgNormThresholdChange={props.onApgNormThresholdChange}
                        omegaScale={props.omegaScale}
                        onOmegaScaleChange={props.onOmegaScaleChange}
                        ergScale={props.ergScale}
                        onErgScaleChange={props.onErgScaleChange}
                    />

                    {/* LM / CoT Settings */}
                    <LmCotAccordion
                        isOpen={showLmCot}
                        onToggle={() => setShowLmCot(!showLmCot)}
                        thinking={props.thinking}
                        onThinkingToggle={props.onThinkingToggle}
                        loraLoaded={props.loraLoaded}
                        lmBackend={props.lmBackend}
                        onLmBackendChange={props.onLmBackendChange}
                        lmModel={props.lmModel}
                        onLmModelChange={props.onLmModelChange}
                        isLmSwitching={props.isLmSwitching}
                        showLmParams={showLmParams}
                        onToggleLmParams={() => setShowLmParams(!showLmParams)}
                        lmTemperature={props.lmTemperature}
                        onLmTemperatureChange={props.onLmTemperatureChange}
                        lmCfgScale={props.lmCfgScale}
                        onLmCfgScaleChange={props.onLmCfgScaleChange}
                        lmTopK={props.lmTopK}
                        onLmTopKChange={props.onLmTopKChange}
                        lmTopP={props.lmTopP}
                        onLmTopPChange={props.onLmTopPChange}
                        lmRepetitionPenalty={props.lmRepetitionPenalty}
                        onLmRepetitionPenaltyChange={props.onLmRepetitionPenaltyChange}
                        lmNegativePrompt={props.lmNegativePrompt}
                        onLmNegativePromptChange={props.onLmNegativePromptChange}
                        allowLmBatch={props.allowLmBatch}
                        onAllowLmBatchToggle={props.onAllowLmBatchToggle}
                        useCotMetas={props.useCotMetas}
                        onUseCotMetasToggle={props.onUseCotMetasToggle}
                        useCotCaption={props.useCotCaption}
                        onUseCotCaptionToggle={props.onUseCotCaptionToggle}
                        useCotLanguage={props.useCotLanguage}
                        onUseCotLanguageToggle={props.onUseCotLanguageToggle}
                        lmBatchChunkSize={props.lmBatchChunkSize}
                        onLmBatchChunkSizeChange={props.onLmBatchChunkSizeChange}
                        constrainedDecodingDebug={props.constrainedDecodingDebug}
                        onConstrainedDecodingDebugToggle={props.onConstrainedDecodingDebugToggle}
                        isFormatCaption={props.isFormatCaption}
                        onIsFormatCaptionToggle={props.onIsFormatCaptionToggle}

                    />

                    {/* Expert Controls */}
                    <ExpertControlsAccordion
                        isOpen={showExpert}
                        onToggle={() => setShowExpert(!showExpert)}
                        uploadError={props.uploadError}
                        audioCodes={props.audioCodes}
                        onAudioCodesChange={props.onAudioCodesChange}
                        instruction={props.instruction}
                        onInstructionChange={props.onInstructionChange}
                        customTimesteps={props.customTimesteps}
                        onCustomTimestepsChange={props.onCustomTimestepsChange}
                        trackName={props.trackName}
                        onTrackNameChange={props.onTrackNameChange}
                        completeTrackClasses={props.completeTrackClasses}
                        onCompleteTrackClassesChange={props.onCompleteTrackClassesChange}
                        autogen={props.autogen}
                        onToggleAutogen={props.onToggleAutogen}
                        getLrc={props.getLrc}
                        onToggleGetLrc={props.onToggleGetLrc}
                    />
                </div>
    );

    if (props.embedded) return content;

    return (
        <div>
            <button
                type="button"
                onClick={props.onToggle}
                className={`w-full flex items-center justify-between px-4 py-3 bg-white dark:bg-suno-card border border-zinc-200 dark:border-white/5 text-sm font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors ${props.isOpen ? 'rounded-t-xl rounded-b-none border-b-0' : 'rounded-xl'}`}
            >
                <span className="flex items-center gap-2">
                    <Settings2 size={16} className="text-pink-500" />
                    {t('generationSettings')}
                </span>
                <ChevronDown size={18} className={`text-pink-500 chevron-icon ${props.isOpen ? 'rotated' : ''}`} />
            </button>

            {props.isOpen && content}
        </div>
    );
};

export default GenerationSettingsAccordion;
