import React from 'react';
import { ChevronDown, Music, SlidersHorizontal } from 'lucide-react';
import { useI18n } from '../../context/I18nContext';
import { EditableSlider } from '../EditableSlider';

interface CoverRepaintSettingsProps {
    taskType: string;
    audioCoverStrength: number;
    setAudioCoverStrength: (val: number) => void;
    coverNoiseStrength: number;
    setCoverNoiseStrength: (val: number) => void;
    tempoScale: number;
    setTempoScale: (val: number) => void;
    pitchShift: number;
    setPitchShift: (val: number) => void;
    bpm: number | null;
    keyScale: string | null;
    detectedBpm: number | null;
    detectedKey: string | null;
    autoMaster: boolean;
    setAutoMaster: (val: boolean) => void;
    onOpenMasteringConsole?: () => void;
    enableNormalization: boolean;
    setEnableNormalization: (val: boolean) => void;
    normalizationDb: number;
    setNormalizationDb: (val: number) => void;
    latentShift: number;
    setLatentShift: (val: number) => void;
    latentRescale: number;
    setLatentRescale: (val: number) => void;
    repaintingStart: number;
    setRepaintingStart: (val: number) => void;
    repaintingEnd: number;
    setRepaintingEnd: (val: number) => void;
    showCoverSettings: boolean;
    setShowCoverSettings: (val: boolean) => void;
    showOutputProcessing: boolean;
    setShowOutputProcessing: (val: boolean) => void;
    masteringParams?: any;
    setMasteringParams?: (params: any) => void;
    onUploadMatcheringRef?: (file: File) => Promise<string>;
    isUploadingMatchering?: boolean;
}

export const CoverRepaintSettings: React.FC<CoverRepaintSettingsProps> = ({
    taskType,
    audioCoverStrength,
    setAudioCoverStrength,
    coverNoiseStrength,
    setCoverNoiseStrength,
    tempoScale,
    setTempoScale,
    pitchShift,
    setPitchShift,
    bpm,
    keyScale,
    detectedBpm,
    detectedKey,
    autoMaster,
    setAutoMaster,
    onOpenMasteringConsole,
    enableNormalization,
    setEnableNormalization,
    normalizationDb,
    setNormalizationDb,
    latentShift,
    setLatentShift,
    latentRescale,
    setLatentRescale,
    repaintingStart,
    setRepaintingStart,
    repaintingEnd,
    setRepaintingEnd,
    showCoverSettings,
    setShowCoverSettings,
    showOutputProcessing,
    setShowOutputProcessing,
    masteringParams,
    setMasteringParams,
    onUploadMatcheringRef,
    isUploadingMatchering
}) => {
    const { t } = useI18n();

    // Show cover/repaint section for tasks that use source audio controls
    // (not text2music and not complete — complete uses source audio but no cover controls)
    const showCoverSection = !['text2music', 'complete'].includes(taskType);
    const showCoverStrength = !['repaint', 'extract', 'lego'].includes(taskType);
    const showCoverNoise = taskType === 'cover';
    const showTempoAndPitch = ['cover', 'repaint', 'audio2audio'].includes(taskType);
    const showRepaintingRange = ['repaint', 'lego'].includes(taskType);

    // Chromatic key transposition (handles both sharps AND flats)
    const CHROMATIC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const FLAT_TO_SHARP: Record<string, string> = {
        'Cb': 'B', 'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    };
    const transposeKey = (keyStr: string, semitones: number): string => {
        if (!keyStr || semitones === 0) return keyStr;
        const parts = keyStr.split(' ');
        const note = parts[0];
        const scale = parts.slice(1).join(' ');
        const normalizedNote = FLAT_TO_SHARP[note] || note;
        const idx = CHROMATIC.indexOf(normalizedNote);
        if (idx === -1) return keyStr;
        const newIdx = ((idx + semitones) % 12 + 12) % 12;
        return `${CHROMATIC[newIdx]}${scale ? ' ' + scale : ''}`;
    };

    const coverTitle = taskType === 'repaint' ? t('repaintSettings') : taskType === 'lego' ? t('legoTask') : t('coverSettings');

    return (
        <>
            {/* COVER / REPAINT CONTROLS — accordion */}
            {showCoverSection && (
                <div>
                    <button
                        type="button"
                        onClick={() => setShowCoverSettings(!showCoverSettings)}
                        className={`w-full flex items-center justify-between px-4 py-3 bg-white dark:bg-suno-card border border-zinc-200 dark:border-white/5 text-sm font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors ${showCoverSettings ? 'rounded-t-xl rounded-b-none border-b-0' : 'rounded-xl'}`}
                    >
                        <span className="flex items-center gap-2"><Music size={16} className="text-pink-500" />{coverTitle}</span>
                        <ChevronDown size={18} className={`text-pink-500 chevron-icon ${showCoverSettings ? 'rotated' : ''}`} />
                    </button>
                    {showCoverSettings && (
                        <div className="bg-white dark:bg-suno-card rounded-b-xl rounded-t-none border border-t-0 border-zinc-200 dark:border-white/5 p-4 space-y-4">
                            {/* Audio Cover Strength & Cover Noise Strength side-by-side */}
                            {showCoverStrength && (
                                <div className={`grid ${showCoverNoise ? 'grid-cols-2' : 'grid-cols-1'} gap-3`}>
                                    <EditableSlider
                                        label={t('audioCoverStrength')}
                                        value={audioCoverStrength}
                                        min={0}
                                        max={1}
                                        step={0.05}
                                        onChange={setAudioCoverStrength}
                                        formatDisplay={(val) => val.toFixed(2)}
                                        helpText={t('audioCoverStrengthHelp')}
                                        title={t('audioCoverStrengthTooltip')}
                                    />
                                    {showCoverNoise && (
                                        <EditableSlider
                                            label={t('coverNoiseStrength')}
                                            value={coverNoiseStrength}
                                            min={0}
                                            max={1}
                                            step={0.01}
                                            onChange={setCoverNoiseStrength}
                                            formatDisplay={(val) => val.toFixed(2)}
                                            helpText={t('coverNoiseStrengthHelp')}
                                            title={t('coverNoiseStrengthTooltip')}
                                        />
                                    )}
                                </div>
                            )}

                            {/* Tempo Scale & Pitch Shift — only for cover/repaint/a2a */}
                            {showTempoAndPitch && (
                                <div className="grid grid-cols-2 gap-3">
                                    <EditableSlider
                                        label={t('tempoScale')}
                                        value={tempoScale}
                                        min={0.5}
                                        max={2.0}
                                        step={0.05}
                                        onChange={setTempoScale}
                                        formatDisplay={(val) => `${val.toFixed(2)}x`}
                                        helpText={t('tempoScaleHelp')}
                                        title={t('tempoScaleTooltip')}
                                    />
                                    <EditableSlider
                                        label={t('pitchShift')}
                                        value={pitchShift}
                                        min={-12}
                                        max={12}
                                        step={1}
                                        onChange={setPitchShift}
                                        formatDisplay={(val) => val === 0 ? '0' : val > 0 ? `+${val} ♯` : `${val} ♭`}
                                        helpText={t('pitchShiftHelp')}
                                        title={t('pitchShiftTooltip')}
                                    />
                                </div>
                            )}

                            {/* Source Analysis Info — show user inputs + computed output */}
                            {showTempoAndPitch && bpm !== 0 && keyScale && (
                                <div className="rounded-lg bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800/30 px-3 py-2 text-[11px] space-y-1">
                                    <div className="flex items-center gap-1.5 text-emerald-700 dark:text-emerald-400 font-medium">
                                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2z" />
                                        </svg>
                                        Base: {bpm} BPM, {keyScale}
                                    </div>
                                    {(tempoScale !== 1.0 || pitchShift !== 0) && (
                                        <div className="text-emerald-600 dark:text-emerald-500">
                                            → Output: {bpm ? Math.round(bpm * tempoScale) : 0} BPM
                                            {pitchShift !== 0 && (
                                                <span className="ml-2">| Key: {transposeKey(keyScale, pitchShift)}</span>
                                            )}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Repainting Start/End — repaint and lego modes */}
                            {showRepaintingRange && (
                                <>
                                    <EditableSlider
                                        label={t('repaintingStart')}
                                        value={repaintingStart}
                                        min={0}
                                        max={600}
                                        step={1}
                                        onChange={setRepaintingStart}
                                        formatDisplay={(val) => val === 0 ? t('beginning') : `${val}s`}
                                        helpText={t('repaintingStartHelp')}
                                        title={t('repaintingStartTooltip')}
                                    />
                                    <EditableSlider
                                        label={t('repaintingEnd')}
                                        value={repaintingEnd}
                                        min={-1}
                                        max={600}
                                        step={1}
                                        onChange={setRepaintingEnd}
                                        formatDisplay={(val) => val === -1 ? t('endOfTrack') : `${val}s`}
                                        helpText={t('repaintingEndHelp')}
                                        title={t('repaintingEndTooltip')}
                                    />
                                </>
                            )}
                        </div>
                    )}
                </div>
            )}

            {/* OUTPUT PROCESSING — accordion, visible for ALL task types */}
            <div>
                <button
                    type="button"
                    onClick={() => setShowOutputProcessing(!showOutputProcessing)}
                    className={`w-full flex items-center justify-between px-4 py-3 bg-white dark:bg-suno-card border border-zinc-200 dark:border-white/5 text-sm font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors ${showOutputProcessing ? 'rounded-t-xl rounded-b-none border-b-0' : 'rounded-xl'}`}
                >
                    <span className="flex items-center gap-2"><SlidersHorizontal size={16} className="text-pink-500" />{t('outputProcessing')}</span>
                    <ChevronDown size={18} className={`text-pink-500 chevron-icon ${showOutputProcessing ? 'rotated' : ''}`} />
                </button>
                {showOutputProcessing && (
                    <div className="bg-white dark:bg-suno-card rounded-b-xl rounded-t-none border border-t-0 border-zinc-200 dark:border-white/5 p-4 space-y-4">
                        {/* Auto-Master toggle */}
                        <div className="flex items-center justify-between py-1">
                            <div className="flex-1">
                                <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400" title={t('autoMasterTooltip')}>{t('autoMaster')}</span>
                                <p className="text-[10px] text-zinc-500">{t('autoMasterHelp')}</p>
                            </div>
                            <div className="flex items-center gap-2">
                                {autoMaster && onOpenMasteringConsole && (
                                    <button
                                        onClick={onOpenMasteringConsole}
                                        className="px-2 py-1 text-[10px] font-bold text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-500/10 hover:bg-amber-100 dark:hover:bg-amber-500/20 rounded-md transition-colors"
                                        title="Open mastering console"
                                    >
                                        🎛️ Settings
                                    </button>
                                )}
                                <button
                                    onClick={() => setAutoMaster(!autoMaster)}
                                    className={`w-10 h-5 rounded-full flex items-center transition-colors duration-200 px-0.5 border border-zinc-200 dark:border-white/5 ${autoMaster ? 'bg-pink-600' : 'bg-zinc-300 dark:bg-black/40'}`}
                                >
                                    <div className={`w-4 h-4 rounded-full bg-white transform transition-transform duration-200 shadow-sm ${autoMaster ? 'translate-x-5' : 'translate-x-0'}`} />
                                </button>
                            </div>
                        </div>

                        {/* Mastering Method Selection */}
                        {autoMaster && (
                            <div className="space-y-3 p-3 bg-zinc-50 dark:bg-black/20 rounded-lg border border-zinc-200 dark:border-white/5">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">Mastering Method</span>
                                    <div className="flex bg-zinc-200 dark:bg-zinc-800 rounded-lg p-0.5">
                                        <button
                                            onClick={() => setMasteringParams?.({ ...(masteringParams || {}), mode: 'builtin' })}
                                            className={`px-3 py-1 text-[10px] font-bold rounded-md transition-colors ${(!masteringParams?.mode || masteringParams.mode === 'builtin') ? 'bg-white dark:bg-zinc-600 text-zinc-900 dark:text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'}`}
                                        >
                                            Built-in
                                        </button>
                                        <button
                                            onClick={() => setMasteringParams?.({ ...(masteringParams || {}), mode: 'matchering' })}
                                            className={`px-3 py-1 text-[10px] font-bold rounded-md transition-colors ${masteringParams?.mode === 'matchering' ? 'bg-white dark:bg-zinc-600 text-zinc-900 dark:text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'}`}
                                        >
                                            Matchering
                                        </button>
                                    </div>
                                </div>
                                
                                {masteringParams?.mode === 'matchering' && (
                                    <div className="space-y-2">
                                        <div className="text-[10px] text-zinc-500">
                                            Matchering requires a reference audio file to analyze and match the EQ/loudness characteristics.
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <input
                                                type="file"
                                                accept="audio/*"
                                                id="matchering-upload"
                                                className="hidden"
                                                onChange={async (e) => {
                                                    const file = e.target.files?.[0];
                                                    if (file && onUploadMatcheringRef) {
                                                        try {
                                                            const url = await onUploadMatcheringRef(file);
                                                            setMasteringParams?.({ ...(masteringParams || {}), reference_file: url, reference_name: file.name });
                                                        } catch (err) {
                                                            console.error('Failed to upload matchering ref:', err);
                                                        }
                                                    }
                                                    e.target.value = '';
                                                }}
                                            />
                                            <label
                                                htmlFor="matchering-upload"
                                                className={`flex items-center gap-2 px-3 py-1.5 text-xs font-semibold rounded-lg border cursor-pointer transition-colors ${isUploadingMatchering ? 'bg-zinc-100 dark:bg-zinc-800 text-zinc-400 border-zinc-200 dark:border-white/10 cursor-wait' : 'bg-white dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 border-zinc-200 dark:border-white/10 hover:border-pink-300 dark:hover:border-pink-500/50 hover:text-pink-600 dark:hover:text-pink-400'}`}
                                            >
                                                {isUploadingMatchering ? (
                                                    <><span className="w-3 h-3 border-2 border-zinc-400 border-t-transparent rounded-full animate-spin" /> Uploading...</>
                                                ) : (
                                                    <>Upload Reference</>
                                                )}
                                            </label>
                                            {masteringParams?.reference_file && (
                                                <div className="flex-1 truncate text-xs text-pink-600 dark:text-pink-400 font-medium">
                                                    Selected: {masteringParams?.reference_name || 'Audio File'}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Normalization toggle */}
                        <div className="flex items-center justify-between py-1">
                            <div>
                                <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400" title={t('enableNormalizationTooltip')}>{t('enableNormalization')}</span>
                                <p className="text-[10px] text-zinc-500">{t('enableNormalizationHelp')}</p>
                            </div>
                            <button
                                onClick={() => setEnableNormalization(!enableNormalization)}
                                className={`w-10 h-5 rounded-full flex items-center transition-colors duration-200 px-0.5 border border-zinc-200 dark:border-white/5 ${enableNormalization ? 'bg-pink-600' : 'bg-zinc-300 dark:bg-black/40'}`}
                            >
                                <div className={`w-4 h-4 rounded-full bg-white transform transition-transform duration-200 shadow-sm ${enableNormalization ? 'translate-x-5' : 'translate-x-0'}`} />
                            </button>
                        </div>

                        {/* Normalization dB - only shown when normalization is enabled */}
                        {enableNormalization && (
                            <EditableSlider
                                label={t('normalizationDb')}
                                value={normalizationDb}
                                min={-10}
                                max={0}
                                step={0.1}
                                onChange={setNormalizationDb}
                                formatDisplay={(val) => `${val.toFixed(1)} dB`}
                                helpText={t('normalizationDbHelp')}
                                title={t('normalizationDbTooltip')}
                            />
                        )}

                        {/* Latent Shift & Latent Rescale side-by-side */}
                        <div className="grid grid-cols-2 gap-3">
                            <EditableSlider
                                label={t('latentShift')}
                                value={latentShift}
                                min={-0.2}
                                max={0.2}
                                step={0.01}
                                onChange={setLatentShift}
                                formatDisplay={(val) => val.toFixed(2)}
                                helpText={t('latentShiftHelp')}
                                title={t('latentShiftTooltip')}
                            />
                            <EditableSlider
                                label={t('latentRescale')}
                                value={latentRescale}
                                min={0.5}
                                max={1.5}
                                step={0.01}
                                onChange={setLatentRescale}
                                formatDisplay={(val) => val.toFixed(2)}
                                helpText={t('latentRescaleHelp')}
                                title={t('latentRescaleTooltip')}
                            />
                        </div>
                    </div>
                )}
            </div>
        </>
    );
};

export default CoverRepaintSettings;
