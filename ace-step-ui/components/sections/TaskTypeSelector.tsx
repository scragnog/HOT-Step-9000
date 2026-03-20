import React from 'react';
import { useI18n } from '../../context/I18nContext';
import { isBaseModel } from '../../utils/modelUtils';

interface TaskTypeSelectorProps {
    taskType: string;
    setTaskType: (val: string) => void;
    useReferenceAudio: boolean;
    selectedModel: string;
}



// Task type descriptions
const TASK_DESCRIPTIONS: Record<string, string> = {
    'auto-write': 'taskDescAutoWrite',
    text2music: 'taskDescText2music',
    cover: 'taskDescCover',
    repaint: 'taskDescRepaint',
    extract: 'taskDescExtract',
    lego: 'taskDescLego',
    complete: 'taskDescComplete',
    audio2audio: 'taskDescAudio2audio',
};

export const TaskTypeSelector: React.FC<TaskTypeSelectorProps> = ({
    taskType,
    setTaskType,
    useReferenceAudio,
    selectedModel
}) => {
    const { t } = useI18n();

    const descKey = TASK_DESCRIPTIONS[taskType] || TASK_DESCRIPTIONS.text2music;

    return (
        <div className="bg-white dark:bg-suno-card rounded-xl border border-zinc-200 dark:border-white/5 overflow-hidden">
            <div className="px-3 py-2.5 flex items-center justify-between gap-2">
                <span className="text-xs font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wide flex-shrink-0 whitespace-nowrap">{t('taskType')}</span>
                <select
                    value={taskType}
                    onChange={(e) => {
                        setTaskType(e.target.value);
                    }}
                    className="min-w-0 max-w-[60%] bg-zinc-100 dark:bg-black/30 border border-zinc-200 dark:border-white/10 rounded-lg px-2.5 py-1.5 text-xs font-medium text-zinc-900 dark:text-white focus:outline-none focus:border-pink-500 dark:focus:border-pink-500 transition-colors cursor-pointer [&>option]:bg-white [&>option]:dark:bg-zinc-800 [&>option]:text-zinc-900 [&>option]:dark:text-white"
                >
                    <option value="auto-write">Simple Mode ✨</option>
                    <option value="text2music">{t('textToMusic')}</option>
                    <option value="cover">{t('coverTask')}</option>
                    <option value="repaint">{t('repaintTask')}</option>
                    <option value="extract" disabled={!isBaseModel(selectedModel)}>{t('extractTask')}{!isBaseModel(selectedModel) ? ` (${t('requiresBaseModel')})` : ''}</option>
                    <option value="lego" disabled={!isBaseModel(selectedModel)}>{t('legoTask')}{!isBaseModel(selectedModel) ? ` (${t('requiresBaseModel')})` : ''}</option>
                    <option value="complete" disabled={!isBaseModel(selectedModel)}>{t('completeTask')}{!isBaseModel(selectedModel) ? ` (${t('requiresBaseModel')})` : ''}</option>
                    <option value="audio2audio">{t('audio2audio')}</option>
                </select>
            </div>
            <div className="px-3 pb-2.5 -mt-0.5">
                <p className="text-[11px] text-zinc-400 dark:text-zinc-500 leading-snug">{t(descKey)}</p>
            </div>
        </div>
    );
};

export default TaskTypeSelector;
