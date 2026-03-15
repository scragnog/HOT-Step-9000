import React from 'react';
import { KEY_SIGNATURES, TIME_SIGNATURES } from '../CreatePanel';
import { useI18n } from '../../context/I18nContext';
import { EditableSlider } from '../EditableSlider';

interface MusicParametersSectionProps {
    bpm: number;
    setBpm: (val: number) => void;
    keyScale: string;
    setKeyScale: (val: string) => void;
    timeSignature: string;
    setTimeSignature: (val: string) => void;
    duration: number;
    setDuration: (val: number) => void;
    detectedBpm?: number | null;
    detectedKey?: string | null;
    taskType?: string;
    sourceDuration?: number;
    tempoScale?: number;
}

export const MusicParametersSection: React.FC<MusicParametersSectionProps> = ({
    bpm,
    setBpm,
    keyScale,
    setKeyScale,
    timeSignature,
    setTimeSignature,
    duration,
    setDuration,
    detectedBpm,
    detectedKey,
    taskType,
    sourceDuration,
    tempoScale = 1.0,
}) => {
    const { t } = useI18n();

    return (
        <div className="space-y-4 pt-2 border-t border-zinc-200 dark:border-white/5">
            <h4 className="text-xs font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wide">{t('musicParameters')}</h4>

            {/* BPM */}
            <EditableSlider
                label={t('bpm')}
                value={bpm}
                min={0}
                max={300}
                step={5}
                onChange={setBpm}
                formatDisplay={(val) => val === 0 ? t('auto') : val.toString()}
                title={t('bpmTooltip')}
                autoLabel={t('auto')}
            />
            {detectedBpm !== null && detectedBpm !== undefined && (
                <p className="text-[10px] text-amber-600 dark:text-amber-400 -mt-2 flex items-center gap-1">
                    <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                    </svg>
                    {t('detectedOverrideWarning')}
                </p>
            )}

            {/* Key & Time Signature */}
            <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                    <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400" title={t('keyTooltip')}>{t('key')}</label>
                    <select
                        value={keyScale}
                        onChange={(e) => setKeyScale(e.target.value)}
                        className="w-full bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg px-2 py-1.5 text-xs text-zinc-900 dark:text-white focus:outline-none focus:border-pink-500 dark:focus:border-pink-500 transition-colors cursor-pointer [&>option]:bg-white [&>option]:dark:bg-zinc-800 [&>option]:text-zinc-900 [&>option]:dark:text-white"
                    >
                        <option value="">{t('autoOption')}</option>
                        {KEY_SIGNATURES.filter(k => k).map(key => (
                            <option key={key} value={key}>{key}</option>
                        ))}
                    </select>
                    <p className="text-[10px] text-zinc-500">Musical key (e.g. C major). Auto = AI chooses</p>
                </div>
                <div className="space-y-1.5">
                    <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400" title={t('timeTooltip')}>{t('time')}</label>
                    <select
                        value={timeSignature}
                        onChange={(e) => setTimeSignature(e.target.value)}
                        className="w-full bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg px-2 py-1.5 text-xs text-zinc-900 dark:text-white focus:outline-none focus:border-pink-500 dark:focus:border-pink-500 transition-colors cursor-pointer [&>option]:bg-white [&>option]:dark:bg-zinc-800 [&>option]:text-zinc-900 [&>option]:dark:text-white"
                    >
                        <option value="">{t('autoOption')}</option>
                        {TIME_SIGNATURES.filter(t => t).map(time => (
                            <option key={time} value={time}>{time}</option>
                        ))}
                    </select>
                    <p className="text-[10px] text-zinc-500">4/4 = pop/rock, 3/4 = waltz, 6/8 = folk</p>
                </div>
            </div>

            {/* Duration */}
            {(taskType === 'cover' || taskType === 'repaint') ? (
                <div className="space-y-1.5">
                    <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{t('duration')}</label>
                    <div className="w-full bg-zinc-100 dark:bg-black/30 border border-zinc-200 dark:border-white/10 rounded-lg px-3 py-2 text-sm text-zinc-500 dark:text-zinc-400 cursor-not-allowed select-none">
                        {sourceDuration && sourceDuration > 0
                            ? `${Math.round(sourceDuration / tempoScale)}s (${Math.round(sourceDuration)}s × ${tempoScale}x tempo)`
                            : 'Set cover audio first'
                        }
                    </div>
                    <p className="text-[10px] text-zinc-500">Duration is determined by the source audio and tempo scale</p>
                </div>
            ) : (
                <EditableSlider
                    label={t('duration')}
                    value={duration}
                    min={-1}
                    max={600}
                    step={5}
                    onChange={setDuration}
                    formatDisplay={(val) => val === -1 ? t('auto') : `${val}${t('seconds')}`}
                    title={t('durationTooltip')}
                    autoLabel={t('auto')}
                />
            )}
        </div>
    );
};

export default MusicParametersSection;
