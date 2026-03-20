import React from 'react';
import { Sparkles, Music2, Mic } from 'lucide-react';
import { useI18n } from '../../context/I18nContext';
import { VOCAL_LANGUAGE_KEYS } from '../../utils/constants';

interface AutoWriteSectionProps {
  songDescription: string;
  setSongDescription: (val: string) => void;
  lyrics: string;
  setLyrics: (val: string) => void;
  vocalLanguage: string;
  setVocalLanguage: (val: string) => void;
  instrumental: boolean;
  setInstrumental: (val: boolean) => void;
}

/**
 * The Simple Mode task type section.
 * Shows a song description textarea plus — when not instrumental — a lyrics textarea.
 * The LM generates caption, BPM, key, time signature, and duration from the description.
 * Lyrics are provided by the user (CoT lyrics generation is not yet implemented).
 */
export const AutoWriteSection: React.FC<AutoWriteSectionProps> = ({
  songDescription,
  setSongDescription,
  lyrics,
  setLyrics,
  vocalLanguage,
  setVocalLanguage,
  instrumental,
  setInstrumental,
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
            placeholder={instrumental
              ? "e.g. A chill lo-fi instrumental with jazzy piano chords and vinyl crackle..."
              : "e.g. An upbeat electronic pop song about summer nights with catchy synth hooks and dreamy female vocals..."}
            rows={3}
            className="w-full resize-y min-h-[60px] bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg p-3 text-sm text-zinc-900 dark:text-white placeholder:text-zinc-400 dark:placeholder:text-zinc-600 focus:outline-none focus:border-pink-400 dark:focus:border-pink-500 transition-colors"
          />
          <p className="text-[10px] text-zinc-400 dark:text-zinc-600 mt-1.5 leading-relaxed">
            {instrumental
              ? 'The AI will generate style, BPM, key, and all metadata from your description.'
              : 'Describe the style, mood, and vocal character. The AI will generate the musical arrangement. Include vocal hints (e.g. "female vocals", "raspy male singer").'}
          </p>
        </div>
      </div>

      {/* Lyrics textarea — only when not instrumental */}
      {!instrumental && (
        <div className="bg-white dark:bg-suno-card rounded-xl border border-zinc-200 dark:border-white/5 overflow-hidden">
          <div className="px-3 py-2.5 border-b border-zinc-100 dark:border-white/5 flex items-center gap-2">
            <Mic size={14} className="text-purple-500" />
            <span className="text-xs font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wide">
              {t('lyrics') || 'Lyrics'}
            </span>
          </div>
          <div className="p-3">
            <textarea
              value={lyrics}
              onChange={(e) => setLyrics(e.target.value)}
              placeholder={"[Verse]\nYour lyrics here...\n\n[Chorus]\n..."}
              rows={6}
              className="w-full resize-y min-h-[100px] bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg p-3 text-sm text-zinc-900 dark:text-white placeholder:text-zinc-400 dark:placeholder:text-zinc-600 focus:outline-none focus:border-purple-400 dark:focus:border-purple-500 transition-colors font-mono"
            />
            <p className="text-[10px] text-zinc-400 dark:text-zinc-600 mt-1.5 leading-relaxed">
              Write your lyrics using structure tags like [Verse], [Chorus], [Bridge]. Leave empty for vocalise/humming.
            </p>
          </div>
        </div>
      )}

      {/* Quick settings */}
      <div className="bg-white dark:bg-suno-card rounded-xl border border-zinc-200 dark:border-white/5 p-3 space-y-3">
        {/* Instrumental toggle */}
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Music2 size={14} className="text-zinc-400" />
              <span className="text-xs font-medium text-zinc-600 dark:text-zinc-300">{t('instrumental')}</span>
            </div>
            <p className="text-[10px] text-zinc-400 dark:text-zinc-500 mt-0.5 ml-[22px]">No vocals or lyrics</p>
          </div>
          <button
            type="button"
            onClick={() => setInstrumental(!instrumental)}
            className={`w-10 h-5 rounded-full flex items-center transition-colors duration-200 px-0.5 border border-zinc-200 dark:border-white/5 cursor-pointer ${instrumental ? 'bg-pink-600' : 'bg-zinc-300 dark:bg-black/40'}`}
          >
            <div className={`w-4 h-4 rounded-full bg-white transform transition-transform duration-200 shadow-sm ${instrumental ? 'translate-x-5' : 'translate-x-0'}`} />
          </button>
        </div>

        {/* Vocal language — only when not instrumental */}
        {!instrumental && (
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
        )}
      </div>
    </div>
  );
};
