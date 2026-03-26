import React from 'react';
import { useI18n } from '../../context/I18nContext';
import { VOCAL_LANGUAGE_KEYS } from '../../utils/constants';
import { MusicParametersSection } from '../sections/MusicParametersSection';

interface TrackDetailsAccordionProps {
    instrumental: boolean;
    setInstrumental: (val: boolean) => void;
    vocalLanguage: string;
    setVocalLanguage: (val: string) => void;
    vocalGender: string;
    setVocalGender: (val: string) => void;

    // Music Parameters Props
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
    triggerWord?: string;
    taskType?: string;
    sourceDuration?: number;
    tempoScale?: number;
    effectiveBpm?: number;
    effectiveKeyScale?: string;
}

/**
 * Track setup content panel — title, instrumental, vocal settings, music params.
 * Designed to live inside a DrawerContainer (no accordion toggle of its own).
 * Lyrics and Style sections are rendered separately as always-visible inputs.
 */
export const TrackDetailsAccordion: React.FC<TrackDetailsAccordionProps> = ({
    instrumental, setInstrumental,
    vocalLanguage, setVocalLanguage,
    vocalGender, setVocalGender,
    bpm, setBpm, keyScale, setKeyScale, timeSignature, setTimeSignature, duration, setDuration,
    detectedBpm, detectedKey, triggerWord,
    taskType, sourceDuration, tempoScale,
    effectiveBpm, effectiveKeyScale,
}) => {
    const { t } = useI18n();

    return (
        <div className="space-y-4">
            {/* Instrumental Toggle */}
            <div className="flex items-center justify-between py-1 border-b border-zinc-200 dark:border-white/5 pb-3">
                <div>
                    <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{t('instrumental')}</span>
                    <p className="text-[11px] text-zinc-400 dark:text-zinc-500">{t('instrumentalTooltip')}</p>
                </div>
                <button
                    onClick={() => setInstrumental(!instrumental)}
                    className={`w-10 h-5 rounded-full flex items-center transition-colors duration-200 px-0.5 border border-zinc-200 dark:border-white/5 ${instrumental ? 'bg-pink-600' : 'bg-zinc-300 dark:bg-black/40'} cursor-pointer`}
                >
                    <div className={`w-4 h-4 rounded-full bg-white transform transition-transform duration-200 shadow-sm ${instrumental ? 'translate-x-5' : 'translate-x-0'}`} />
                </button>
            </div>

            {/* Vocal Language & Gender */}
            {!instrumental && (
                <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1.5">
                        <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400" title={t('vocalLanguageTooltip')}>{t('vocalLanguage')}</label>
                        <select
                            value={vocalLanguage}
                            onChange={(e) => setVocalLanguage(e.target.value)}
                            className="w-full bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg px-2 py-2 text-xs text-zinc-900 dark:text-white focus:outline-none focus:border-pink-500 dark:focus:border-pink-500 transition-colors cursor-pointer [&>option]:bg-white [&>option]:dark:bg-zinc-800 [&>option]:text-zinc-900 [&>option]:dark:text-white"
                        >
                            {VOCAL_LANGUAGE_KEYS.map(lang => (
                                <option key={lang.value} value={lang.value}>{t(lang.key)}</option>
                            ))}
                        </select>
                    </div>
                    <div className="space-y-1.5">
                        <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400" title={t('vocalGenderTooltip')}>{t('vocalGender')}</label>
                        <div className="flex items-center gap-2">
                            <button
                                type="button"
                                onClick={() => setVocalGender(vocalGender === 'male' ? '' : 'male')}
                                className={`flex-1 px-3 py-2 rounded-lg text-xs font-semibold border transition-colors ${vocalGender === 'male' ? 'bg-pink-600 text-white border-pink-600' : 'border-zinc-200 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:border-zinc-300 dark:hover:border-white/20'}`}
                            >
                                {t('male')}
                            </button>
                            <button
                                type="button"
                                onClick={() => setVocalGender(vocalGender === 'female' ? '' : 'female')}
                                className={`flex-1 px-3 py-2 rounded-lg text-xs font-semibold border transition-colors ${vocalGender === 'female' ? 'bg-pink-600 text-white border-pink-600' : 'border-zinc-200 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:border-zinc-300 dark:hover:border-white/20'}`}
                            >
                                {t('female')}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Music Parameters */}
            <MusicParametersSection
                bpm={bpm}
                setBpm={setBpm}
                keyScale={keyScale}
                setKeyScale={setKeyScale}
                timeSignature={timeSignature}
                setTimeSignature={setTimeSignature}
                duration={duration}
                setDuration={setDuration}
                detectedBpm={detectedBpm}
                detectedKey={detectedKey}
                taskType={taskType}
                sourceDuration={sourceDuration}
                tempoScale={tempoScale}
            />
        </div>
    );
};

export default TrackDetailsAccordion;
