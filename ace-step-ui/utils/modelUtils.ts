// Model utility functions extracted from CreatePanel.tsx

/** Map model ID to short display name */
export const getModelDisplayName = (modelId: string): string => {
  const mapping: Record<string, string> = {
    'acestep-v15-base': '1.5B',
    'acestep-v15-sft': '1.5S',
    'acestep-v15-turbo-shift1': '1.5TS1',
    'acestep-v15-turbo-shift3': '1.5TS3',
    'acestep-v15-turbo-continuous': '1.5TC',
    'acestep-v15-turbo': '1.5T',
    'acestep-v15-merge-sft-turbo-0.5': 'ST.5',
    'acestep-v15-merge-sft-turbo-0.4': 'ST.4',
    'acestep-v15-merge-sft-turbo-0.3': 'ST.3',
    'acestep-v15-merge-base-turbo-0.5': 'BT.5',
    'acestep-v15-merge-base-sft-0.5': 'BS.5',
    // XL (4B DiT) models
    'acestep-v15-xl-base': 'XL-B',
    'acestep-v15-xl-sft': 'XL-S',
    'acestep-v15-xl-turbo': 'XL-T',
    // XL merge models
    'acestep-v15-merge-sft-turbo-xl-ta-0.5': 'XL-ST',
    'acestep-v15-merge-base-turbo-xl-ta-0.5': 'XL-BT',
    'acestep-v15-merge-base-sft-xl-ta-0.5': 'XL-BS',
  };
  return mapping[modelId] || modelId;
};

/**
 * Check if model is a turbo variant.
 * Excludes merged models — SFT+Turbo merges are weight averages,
 * NOT pure distillation checkpoints.
 */
export const isTurboModel = (modelId: string): boolean => {
  return modelId.includes('turbo') && !modelId.includes('merge');
};

/** Check if model is a pure base model (only base supports extract/lego/complete) */
export const isBaseModel = (modelId: string): boolean => {
  return modelId.includes('base');
};

/** Check if a task type requires a base model */
export const isBaseOnlyTask = (task: string): boolean => {
  return ['extract', 'lego', 'complete'].includes(task);
};

/** Check if model is an XL (4B DiT) variant — needs ≥16GB VRAM, requires XL-trained adapters */
export const isXlModel = (modelId: string): boolean => {
  return modelId.includes('-xl-');
};
