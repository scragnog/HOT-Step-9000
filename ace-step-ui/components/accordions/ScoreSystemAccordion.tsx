import React from 'react';
import { BarChart3, ChevronDown } from 'lucide-react';
import { useI18n } from '../../context/I18nContext';
import { EditableSlider } from '../EditableSlider';

interface ScoreSystemAccordionProps {
    isOpen: boolean;
    onToggle: () => void;
    getScores: boolean;
    onToggleGetScores: () => void;
    scoreScale: number;
    onScoreScaleChange: (val: number) => void;
    /** When true, skip the accordion header — used inside DrawerContainers */
    embedded?: boolean;
}

export const ScoreSystemAccordion: React.FC<ScoreSystemAccordionProps> = ({
    isOpen,
    onToggle,
    getScores,
    onToggleGetScores,
    scoreScale,
    onScoreScaleChange,
    embedded = false,
}) => {
    const { t } = useI18n();

    const content = (
        <div className={embedded ? "space-y-4" : "bg-white dark:bg-suno-card rounded-b-xl rounded-t-none border border-t-0 border-zinc-200 dark:border-white/5 p-4 space-y-4"}>
            {/* Auto Quality Scoring */}
            <div className="flex items-center justify-between py-1">
                <div>
                    <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{t('autoQualityScoring')}</span>
                    <p className="text-[11px] text-zinc-400 dark:text-zinc-500">{t('autoQualityScoringHelp')}</p>
                </div>
                <button
                    type="button"
                    onClick={onToggleGetScores}
                    className={`w-10 h-5 rounded-full flex items-center transition-colors duration-200 px-0.5 border border-zinc-200 dark:border-white/5 ${getScores ? 'bg-pink-600' : 'bg-zinc-300 dark:bg-black/40'} cursor-pointer`}
                >
                    <div className={`w-4 h-4 rounded-full bg-white transform transition-transform duration-200 shadow-sm ${getScores ? 'translate-x-5' : 'translate-x-0'}`} />
                </button>
            </div>

            {/* Score Sensitivity */}
            <EditableSlider
                label={t('scoreSensitivity')}
                value={scoreScale}
                min={0.01}
                max={1}
                step={0.01}
                onChange={(e) => onScoreScaleChange(Number(e.target.value))}
                formatDisplay={(val) => val.toFixed(2)}
                helpText={t('scoreSensitivityHelp')}
                title={t('scoreScaleTooltip')}
            />
        </div>
    );

    if (embedded) return content;

    return (
        <div>
            <button
                type="button"
                onClick={onToggle}
                className={`w-full flex items-center justify-between px-4 py-3 bg-white dark:bg-suno-card border border-zinc-200 dark:border-white/5 text-sm font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors ${isOpen ? 'rounded-t-xl rounded-b-none border-b-0' : 'rounded-xl'}`}
            >
                <span className="flex items-center gap-2">
                    <BarChart3 size={16} className="text-pink-500" />
                    {t('scoreSystem')}
                </span>
                <ChevronDown size={18} className={`text-pink-500 chevron-icon ${isOpen ? 'rotated' : ''}`} />
            </button>

            {isOpen && content}
        </div>
    );

};

export default ScoreSystemAccordion;
