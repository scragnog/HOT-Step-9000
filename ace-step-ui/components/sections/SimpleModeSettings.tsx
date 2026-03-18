import React from 'react';
import { Sliders } from 'lucide-react';
import { useI18n } from '../../context/I18nContext';
import { EditableSlider } from '../EditableSlider';
import { VOCAL_LANGUAGE_KEYS, KEY_SIGNATURES, TIME_SIGNATURES } from '../../utils/constants';

interface SimpleModeSettingsProps {
    songDescription: string;
    setSongDescription: (val: string) => void;
    vocalLanguage: string;
    setVocalLanguage: (val: string) => void;
    vocalGender: string;
    setVocalGender: (val: string) => void;
    duration: number;
    setDuration: (val: number) => void;
    bpm: number;
    setBpm: (val: number) => void;
    keyScale: string;
    setKeyScale: (val: string) => void;
    timeSignature: string;
    setTimeSignature: (val: string) => void;
    batchSize: number;
    setBatchSize: (val: number) => void;
}

export const SimpleModeSettings: React.FC<SimpleModeSettingsProps> = ({
    songDescription,
    setSongDescription,
    vocalLanguage,
    setVocalLanguage,
    vocalGender,
    setVocalGender,
    duration,
    setDuration,
    bpm,
    setBpm,
    keyScale,
    setKeyScale,
    timeSignature,
    setTimeSignature,
    batchSize,
    setBatchSize
}) => {
    const { t } = useI18n();

    return (
        <div className="space-y-5">
            {/* Song Description */}
            <div className="bg-white dark:bg-suno-card rounded-xl border border-zinc-200 dark:border-white/5 overflow-hidden">
                <div className="px-3 py-2.5 text-xs font-bold uppercase tracking-wide text-zinc-500 dark:text-zinc-400 border-b border-zinc-100 dark:border-white/5 bg-zinc-50 dark:bg-white/5">
                    {t('describeYourSong')}
                </div>
                <textarea
                    value={songDescription}
                    onChange={(e) => setSongDescription(e.target.value)}
                    placeholder={t('songDescriptionPlaceholder')}
                    className="w-full h-32 bg-transparent p-3 text-sm text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none resize-none"
                />
            </div>

            {/* Vocal Language (Simple) */}
            <div className="bg-white dark:bg-suno-card rounded-xl border border-zinc-200 dark:border-white/5 overflow-hidden">
                <div className="px-3 py-2.5 text-xs font-bold uppercase tracking-wide text-zinc-500 dark:text-zinc-400 border-b border-zinc-100 dark:border-white/5 bg-zinc-50 dark:bg-white/5">
                    {t('vocalLanguage')}
                </div>
                <div className="flex flex-wrap items-center gap-2 p-3">
                    <select
                        value={vocalLanguage}
                        onChange={(e) => setVocalLanguage(e.target.value)}
                        className="flex-1 min-w-[180px] bg-transparent text-sm text-zinc-900 dark:text-white focus:outline-none"
                    >
                        {VOCAL_LANGUAGE_KEYS.map(lang => (
                            <option key={lang.value} value={lang.value}>{t(lang.key)}</option>
                        ))}
                    </select>
                    <div className="flex items-center gap-2">
                        <button
                            type="button"
                            onClick={() => setVocalGender(vocalGender === 'male' ? '' : 'male')}
                            className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors ${vocalGender === 'male' ? 'bg-pink-600 text-white border-pink-600' : 'border-zinc-200 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:border-zinc-300 dark:hover:border-white/20'}`}
                        >
                            {t('male')}
                        </button>
                        <button
                            type="button"
                            onClick={() => setVocalGender(vocalGender === 'female' ? '' : 'female')}
                            className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors ${vocalGender === 'female' ? 'bg-pink-600 text-white border-pink-600' : 'border-zinc-200 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:border-zinc-300 dark:hover:border-white/20'}`}
                        >
                            {t('female')}
                        </button>
                    </div>
                </div>
            </div>

            {/* Quick Settings (Simple Mode) */}
            <div className="bg-white dark:bg-suno-card rounded-xl border border-zinc-200 dark:border-white/5 p-4 space-y-4">
                <h3 className="text-xs font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wide flex items-center gap-2">
                    <Sliders size={14} />
                    {t('quickSettings')}
                </h3>

                {/* Duration */}
                <EditableSlider
                    label={t('duration')}
                    value={duration}
                    min={-1}
                    max={600}
                    step={5}
                    onChange={setDuration}
                    formatDisplay={(val) => val === -1 ? t('auto') : `${val}${t('seconds')}`}
                    title={''}
                    autoLabel={t('auto')}
                />

                {/* BPM */}
                <EditableSlider
                    label="BPM"
                    value={bpm}
                    min={0}
                    max={300}
                    step={5}
                    onChange={setBpm}
                    formatDisplay={(val) => val === 0 ? t('auto') : val.toString()}
                    autoLabel={t('auto')}
                />

                {/* Key & Time Signature */}
                <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1.5">
                        <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{t('key')}</label>
                        <select
                            value={keyScale}
                            onChange={(e) => setKeyScale(e.target.value)}
                            className="w-full bg-white dark:bg-zinc-800/50 border border-zinc-200 dark:border-white/10 rounded-xl px-3 py-1.5 text-xs text-zinc-700 dark:text-zinc-200 focus:outline-none focus:ring-2 focus:ring-pink-500/50 focus:border-pink-500 dark:focus:border-pink-500 transition-all cursor-pointer hover:border-pink-300 dark:hover:border-pink-500/50 appearance-none bg-[url('data:image/svg+xml;charset=UTF-8,%3csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 24 24%27 fill=%27none%27 stroke=%27%236b7280%27 stroke-width=%272%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27%3e%3cpolyline points=%276 9 12 15 18 9%27%3e%3c/polyline%3e%3c/svg%3e')] bg-no-repeat bg-[length:1rem] bg-[right_0.5rem_center] pr-8 shadow-sm"
                        >
                            <option value="">{t('autoOption')}</option>
                            {KEY_SIGNATURES.filter(k => k).map(key => (
                                <option key={key} value={key}>{key}</option>
                            ))}
                        </select>
                    </div>
                    <div className="space-y-1.5">
                        <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{t('time')}</label>
                        <select
                            value={timeSignature}
                            onChange={(e) => setTimeSignature(e.target.value)}
                            className="w-full bg-white dark:bg-zinc-800/50 border border-zinc-200 dark:border-white/10 rounded-xl px-3 py-1.5 text-xs text-zinc-700 dark:text-zinc-200 focus:outline-none focus:ring-2 focus:ring-pink-500/50 focus:border-pink-500 dark:focus:border-pink-500 transition-all cursor-pointer hover:border-pink-300 dark:hover:border-pink-500/50 appearance-none bg-[url('data:image/svg+xml;charset=UTF-8,%3csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 24 24%27 fill=%27none%27 stroke=%27%236b7280%27 stroke-width=%272%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27%3e%3cpolyline points=%276 9 12 15 18 9%27%3e%3c/polyline%3e%3c/svg%3e')] bg-no-repeat bg-[length:1rem] bg-[right_0.5rem_center] pr-8 shadow-sm"
                        >
                            <option value="">{t('autoOption')}</option>
                            {TIME_SIGNATURES.filter(t => t).map(time => (
                                <option key={time} value={time}>{time}</option>
                            ))}
                        </select>
                    </div>
                </div>

                {/* Variations */}
                <EditableSlider
                    label={t('variations')}
                    value={batchSize}
                    min={1}
                    max={8}
                    step={1}
                    onChange={setBatchSize}
                />
                <div style={{ display: 'none' }}>
                    <input
                        type="range"
                        min="1"
                        max="8"
                        step="1"
                        value={batchSize}
                        onChange={(e) => setBatchSize(Number(e.target.value))}
                        className="w-full h-2 bg-zinc-200 dark:bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
                    />
                    <p className="text-[10px] text-zinc-500">{t('numberOfVariations')}</p>
                </div>
            </div>
        </div>
    );
};

export default SimpleModeSettings;
