import React, { useState, useEffect } from 'react';
import { ChevronDown, Zap, Flame, Brain } from 'lucide-react';
import { useI18n } from '../context/I18nContext';
import { usePersistedState } from '../hooks/usePersistedState';

interface GenerateFooterProps {
  /** Called with optional step/thinking/mastering/coverArt overrides. Undefined = use current panel values. */
  onGenerate: (overrides?: { inferenceSteps?: number; thinking?: boolean; autoMaster?: boolean; generateCoverArt?: boolean }) => void;
  isGenerating: boolean;
  isAuthenticated: boolean;
  activeJobCount: number;
  /** Whether the currently selected model is a turbo model (hides presets) */
  isTurboModel: boolean;
  /** Whether auto-write task type is selected (hides presets — thinking is mandatory) */
  isAutoWrite?: boolean;
}

interface PresetButton {
  id: string;
  icon: React.ReactNode;
  label: string;
  steps: number;
  thinking: boolean;
}

/**
 * Sticky footer with the primary "Summon Bangers" button and a collapsible
 * "Advanced Summoning Modes" accordion containing preset generation buttons.
 */
export const GenerateFooter: React.FC<GenerateFooterProps> = ({
  onGenerate,
  isGenerating,
  isAuthenticated,
  activeJobCount,
  isTurboModel: isTurbo,
  isAutoWrite = false,
}) => {
  const { t } = useI18n();
  const [elapsedSecs, setElapsedSecs] = useState(0);
  const [showPresets, setShowPresets] = usePersistedState('ace-showPresets', false);
  const [advancedMastering, setAdvancedMastering] = usePersistedState('ace-advancedMastering', false);
  const [advancedCoverArt, setAdvancedCoverArt] = usePersistedState('ace-advancedCoverArt', true);

  // Read configurable step counts from localStorage (set in SettingsModal)
  const quickSteps = (() => {
    const saved = localStorage.getItem('quick_preset_steps');
    return saved !== null ? parseInt(saved, 10) : 12;
  })();

  const hqSteps = (() => {
    const saved = localStorage.getItem('hq_preset_steps');
    return saved !== null ? parseInt(saved, 10) : 60;
  })();

  // Build preset buttons
  const presets: PresetButton[] = [
    { id: 'quick', icon: <Zap size={14} />, label: `Quick (${quickSteps})`, steps: quickSteps, thinking: false },
    { id: 'hq', icon: <Flame size={14} />, label: `HQ (${hqSteps})`, steps: hqSteps, thinking: false },
    { id: 'quick-think', icon: <><Zap size={14} /><Brain size={12} /></>, label: `Quick+Think (${quickSteps})`, steps: quickSteps, thinking: true },
    { id: 'hq-think', icon: <><Flame size={14} /><Brain size={12} /></>, label: `HQ+Think (${hqSteps})`, steps: hqSteps, thinking: true },
  ];

  // Elapsed timer
  useEffect(() => {
    if (!isGenerating) {
      setElapsedSecs(0);
      return;
    }
    setElapsedSecs(0);
    const start = Date.now();
    const id = setInterval(() => {
      setElapsedSecs(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [isGenerating]);

  const formatElapsed = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return m > 0 ? `${m}m ${String(s).padStart(2, '0')}s` : `${s}s`;
  };

  return (
    <div className="p-4 mt-auto sticky bottom-0 bg-zinc-50/95 dark:bg-suno-panel/95 backdrop-blur-sm z-10 border-t border-zinc-200 dark:border-white/5 space-y-3">

      {/* Advanced Summoning Modes accordion — hidden for turbo models and auto-write */}
      {!isTurbo && !isAutoWrite && (
        <div className="rounded-xl border border-zinc-200 dark:border-white/5 overflow-hidden">
          <button
            type="button"
            onClick={() => setShowPresets(!showPresets)}
            className="w-full flex items-center justify-between px-3 py-2 text-xs font-semibold text-zinc-500 dark:text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800/50 transition-colors"
          >
            <span>Advanced Summoning Modes</span>
            <ChevronDown
              size={14}
              className={`transition-transform duration-200 ${showPresets ? 'rotate-180' : ''}`}
            />
          </button>

          {showPresets && (
            <div className="px-3 pb-3 pt-1 space-y-3">
              <div className="grid grid-cols-2 gap-2">
                {presets.map(preset => (
                  <button
                    key={preset.id}
                    type="button"
                    onClick={() => onGenerate({
                      inferenceSteps: preset.steps,
                      thinking: preset.thinking,
                      autoMaster: advancedMastering,
                      generateCoverArt: advancedCoverArt,
                    })}
                    disabled={!isAuthenticated || isGenerating}
                    className="flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold transition-all
                      bg-gradient-to-r from-orange-500 to-pink-600 text-white
                      border border-orange-400/30 dark:border-orange-500/20
                      hover:brightness-110 hover:shadow-md
                      disabled:opacity-40 disabled:cursor-not-allowed
                      active:scale-[0.97]"
                  >
                    <span className="flex items-center gap-0.5">{preset.icon}</span>
                    <span>{preset.label}</span>
                  </button>
                ))}
              </div>

              {/* Auto-Master toggle — only applies to advanced preset buttons */}
              <div className="flex items-center justify-between py-1 border-t border-zinc-200 dark:border-white/5 pt-2">
                <div>
                  <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">Auto-Master</span>
                  <p className="text-[10px] text-zinc-400 dark:text-zinc-500">Apply mastering to preset generations</p>
                </div>
                <button
                  type="button"
                  onClick={() => setAdvancedMastering(!advancedMastering)}
                  className={`w-10 h-5 rounded-full flex items-center transition-colors duration-200 px-0.5 border border-zinc-200 dark:border-white/5 cursor-pointer ${advancedMastering ? 'bg-pink-600' : 'bg-zinc-300 dark:bg-black/40'}`}
                >
                  <div className={`w-4 h-4 rounded-full bg-white transform transition-transform duration-200 shadow-sm ${advancedMastering ? 'translate-x-5' : 'translate-x-0'}`} />
                </button>
              </div>

              {/* AI Cover Art toggle — only applies to advanced preset buttons */}
              <div className="flex items-center justify-between py-1 border-t border-zinc-200 dark:border-white/5 pt-2">
                <div>
                  <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">AI Cover Art</span>
                  <p className="text-[10px] text-zinc-400 dark:text-zinc-500">Generate cover art for preset generations</p>
                </div>
                <button
                  type="button"
                  onClick={() => setAdvancedCoverArt(!advancedCoverArt)}
                  className={`w-10 h-5 rounded-full flex items-center transition-colors duration-200 px-0.5 border border-zinc-200 dark:border-white/5 cursor-pointer ${advancedCoverArt ? 'bg-pink-600' : 'bg-zinc-300 dark:bg-black/40'}`}
                >
                  <div className={`w-4 h-4 rounded-full bg-white transform transition-transform duration-200 shadow-sm ${advancedCoverArt ? 'translate-x-5' : 'translate-x-0'}`} />
                </button>
              </div>

              <p className="text-[10px] text-zinc-400 dark:text-zinc-600 text-center">
                These override your current step count and thinking settings for one generation only
              </p>
            </div>
          )}
        </div>
      )}

      {/* Primary generate button */}
      <button
        onClick={() => onGenerate()}
        className={`w-full h-12 rounded-xl font-bold text-base flex items-center justify-center gap-2 transition-all transform active:scale-[0.98] shadow-lg ${isGenerating
          ? 'bg-gradient-to-r from-orange-400/80 to-pink-500/80 text-white hover:brightness-110'
          : 'bg-gradient-to-r from-orange-500 to-pink-600 text-white hover:brightness-110'
          }`}
        disabled={!isAuthenticated}
      >
        {isGenerating ? (
          <>
            <span className="inline-block w-4 h-4 border-2 border-white/60 border-t-transparent rounded-full animate-spin" />
            <span>
              {activeJobCount > 0
                ? `${t('queueNext')} (${activeJobCount} active)`
                : 'Generating…'
              }
            </span>
            {elapsedSecs > 0 && (
              <span className="text-xs font-normal text-white/70 tabular-nums ml-1">
                {formatElapsed(elapsedSecs)}
              </span>
            )}
          </>
        ) : (
          <span>{t('createButton')}</span>
        )}
      </button>

      {!isAuthenticated && (
        <p className="text-center text-xs text-rose-500 font-medium">{t('loginRequired')}</p>
      )}
    </div>
  );
};

export default GenerateFooter;
