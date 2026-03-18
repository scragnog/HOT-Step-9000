import React from 'react';
import { Sparkles } from 'lucide-react';
import { useI18n } from '../../context/I18nContext';
import { VOCAL_LANGUAGE_KEYS } from '../../utils/constants';
import { EditableSlider } from '../EditableSlider';

interface AutoWriteSectionProps {
  songDescription: string;
  setSongDescription: (val: string) => void;
  vocalLanguage: string;
  setVocalLanguage: (val: string) => void;
  vocalGender: 'male' | 'female' | '';
  setVocalGender: (val: 'male' | 'female' | '') => void;
  duration: number;
  setDuration: (val: number) => void;
  bpm: number;
  setBpm: (val: number) => void;
}

/**
 * The Auto-Write task type section. Replaces Simple Mode.
 * Shows a song description textarea and quick music parameters.
 * The LM will generate lyrics, style, title, and metadata from the description.
 */
export const AutoWriteSection: React.FC<AutoWriteSectionProps> = ({
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
}) => {
  const { t } = useI18n();

  return (
    <div className="space-y-4">
      {/* Description textarea */}
      <div className="bg-white dark:bg-suno-card rounded-xl border border-zinc-200 dark:border-white/5 overflow-hidden">
        <div className="px-3 py-2.5 border-b border-zinc-100 dark:border-white/5 flex items-center gap-2">
          <Sparkles size={14} className="text-pink-500" />
          <span className="text-xs font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wide">
            Describe Your Song
          </span>
        </div>
        <div className="p-3">
          <textarea
            value={songDescription}
            onChange={(e) => setSongDescription(e.target.value)}
            placeholder="e.g. An upbeat electronic pop song about summer nights with catchy synth hooks and dreamy vocals..."
            rows={4}
            className="w-full resize-y min-h-[80px] bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg p-3 text-sm text-zinc-900 dark:text-white placeholder:text-zinc-400 dark:placeholder:text-zinc-600 focus:outline-none focus:border-pink-400 dark:focus:border-pink-500 transition-colors"
          />
          <p className="text-[10px] text-zinc-400 dark:text-zinc-600 mt-1.5 leading-relaxed">
            The AI will generate lyrics, style, title, and all metadata from your description
          </p>
        </div>
      </div>

      {/* Quick settings row */}
      <div className="bg-white dark:bg-suno-card rounded-xl border border-zinc-200 dark:border-white/5 p-3 space-y-3">
        {/* Vocal language */}
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">{t('vocalLanguage')}</span>
          <select
            value={vocalLanguage}
            onChange={(e) => setVocalLanguage(e.target.value)}
            className="bg-zinc-100 dark:bg-black/30 border border-zinc-200 dark:border-white/10 rounded-lg px-2.5 py-1.5 text-xs font-medium text-zinc-900 dark:text-white focus:outline-none focus:border-pink-500 transition-colors cursor-pointer"
          >
            {VOCAL_LANGUAGE_KEYS.map(lang => (
              <option key={lang.value} value={lang.value}>
                {t(lang.key)}
              </option>
            ))}
          </select>
        </div>

        {/* Vocal gender */}
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">{t('vocalGender')}</span>
          <div className="flex gap-1">
            {(['', 'male', 'female'] as const).map(g => (
              <button
                key={g}
                onClick={() => setVocalGender(g)}
                className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                  vocalGender === g
                    ? 'bg-pink-500/10 text-pink-600 dark:text-pink-400 border border-pink-500/30'
                    : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500 dark:text-zinc-400 border border-zinc-200 dark:border-zinc-700 hover:border-zinc-300 dark:hover:border-zinc-600'
                }`}
              >
                {g === '' ? t('any') : t(g)}
              </button>
            ))}
          </div>
        </div>

        {/* Duration */}
        <EditableSlider
          label={t('duration')}
          value={duration}
          onChange={setDuration}
          min={-1}
          max={240}
          step={1}
          formatValue={(v) => v <= 0 ? t('auto') : `${v}s`}
        />

        {/* BPM */}
        <EditableSlider
          label="BPM"
          value={bpm}
          onChange={setBpm}
          min={0}
          max={240}
          step={1}
          formatValue={(v) => v === 0 ? t('auto') : String(v)}
        />
      </div>
    </div>
  );
};
