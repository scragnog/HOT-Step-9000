import React, { useState, useEffect } from 'react';
import { ChevronDown, Zap, Flame, Brain } from 'lucide-react';
import { useI18n } from '../context/I18nContext';
import { usePersistedState } from '../hooks/usePersistedState';

interface GenerateFooterProps {
  /** Called with optional step/thinking overrides. Undefined = use current panel values. */
  onGenerate: (overrides?: { inferenceSteps?: number; thinking?: boolean }) => void;
  isGenerating: boolean;
  isAuthenticated: boolean;
  activeJobCount: number;
  /** Whether the currently selected model is a turbo model (hides presets) */
  isTurboModel: boolean;
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
 * Replaces CreateButtonFooter.
 */
export const GenerateFooter: React.FC<GenerateFooterProps> = ({
  onGenerate,
  isGenerating,
  isAuthenticated,
  activeJobCount,
  isTurboModel: isTurbo,
}) => {
  const { t } = useI18n();
  const [elapsedSecs, setElapsedSecs] = useState(0);
  const [showPresets, setShowPresets] = usePersistedState('ace-showPresets', false);

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

      {/* Advanced Summoning Modes accordion — hidden for turbo models */}
      {!isTurbo && (
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
            <div className="px-3 pb-3 pt-1">
              <div className="grid grid-cols-2 gap-2">
                {presets.map(preset => (
                  <button
                    key={preset.id}
                    type="button"
                    onClick={() => onGenerate({ inferenceSteps: preset.steps, thinking: preset.thinking })}
                    disabled={!isAuthenticated || isGenerating}
                    className="flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold transition-all
                      bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300
                      border border-zinc-200 dark:border-zinc-700
                      hover:bg-zinc-200 dark:hover:bg-zinc-700 hover:border-zinc-300 dark:hover:border-zinc-600
                      disabled:opacity-40 disabled:cursor-not-allowed
                      active:scale-[0.97]"
                  >
                    <span className="flex items-center gap-0.5">{preset.icon}</span>
                    <span>{preset.label}</span>
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-zinc-400 dark:text-zinc-600 mt-2 text-center">
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
          <span>{t('createSong')}</span>
        )}
      </button>

      {/* Keyboard shortcut hint */}
      {!isGenerating && (
        <p className="text-center text-[10px] text-zinc-400 dark:text-zinc-600">
          <kbd className="font-mono">Ctrl</kbd>+<kbd className="font-mono">Enter</kbd> to generate
        </p>
      )}

      {!isAuthenticated && (
        <p className="text-center text-xs text-rose-500 font-medium">{t('loginRequired')}</p>
      )}
    </div>
  );
};

export default GenerateFooter;
