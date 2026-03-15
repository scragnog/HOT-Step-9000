import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { usePersistedState } from '../hooks/usePersistedState';
import { Sparkles, ChevronDown, Settings2, Trash2, Music2, Sliders, Dices, Hash, RefreshCw, Plus, Upload, Play, Pause, Loader2, Brain, Crosshair, BarChart3, FileText, Download } from 'lucide-react';
import { GenerationParams, Song } from '../types';
import { useAuth } from '../context/AuthContext';
import { useI18n } from '../context/I18nContext';
import { generateApi } from '../services/api';
import { MAIN_STYLES, SUB_STYLES, ALL_STYLES } from '../data/genres';
import { EditableSlider } from './EditableSlider';
import GenerationSettingsAccordion from './accordions/GenerationSettingsAccordion';
import { AudioSelectionSection } from './sections/AudioSelectionSection';
import { LyricsSection } from './sections/LyricsSection';
import { StyleSection } from './sections/StyleSection';
import { MusicParametersSection } from './sections/MusicParametersSection';
import { CoverRepaintSettings } from './sections/CoverRepaintSettings';
import { CreatePanelHeader } from './sections/CreatePanelHeader';
import { TaskTypeSelector } from './sections/TaskTypeSelector';
import { ExtractTrackSelector } from './sections/ExtractTrackSelector';
import { SimpleModeSettings } from './sections/SimpleModeSettings';
import { TrackDetailsAccordion } from './accordions/TrackDetailsAccordion';
import { AudioLibraryModal } from './sections/AudioLibraryModal';
import { CreateButtonFooter } from './sections/CreateButtonFooter';
import { LyricsLibrary } from './LyricsLibrary';
import { MasteringConsoleModal, MasteringParams as MasteringParamsType } from './MasteringConsoleModal';

import AdaptersAccordion from './accordions/AdaptersAccordion';
import { LayerAblationPanel } from './accordions/LayerAblationPanel';
import ScoreSystemAccordion from './accordions/ScoreSystemAccordion';
import { ActivationSteeringSection } from './sections/ActivationSteeringSection';

interface ReferenceTrack {
  id: string;
  filename: string;
  storage_key: string;
  duration: number | null;
  file_size_bytes: number | null;
  tags: string[] | null;
  created_at: string;
  audio_url: string;
}

interface CreatePanelProps {
  onGenerate: (params: GenerationParams) => void;
  isGenerating: boolean;
  activeJobCount?: number;
  initialData?: { song: Song, timestamp: number } | null;
  createdSongs?: Song[];
  pendingAudioSelection?: { target: 'reference' | 'source'; url: string; title?: string } | null;
  onAudioSelectionApplied?: () => void;
  // Ablation diff pins (lifted from App.tsx)
  diffPinnedA?: Song | null;
  diffPinnedB?: Song | null;
  onClearDiffA?: () => void;
  onClearDiffB?: () => void;
}

export const KEY_SIGNATURES = [
  '',
  'C major', 'C minor',
  'C# major', 'C# minor',
  'Db major', 'Db minor',
  'D major', 'D minor',
  'D# major', 'D# minor',
  'Eb major', 'Eb minor',
  'E major', 'E minor',
  'F major', 'F minor',
  'F# major', 'F# minor',
  'Gb major', 'Gb minor',
  'G major', 'G minor',
  'G# major', 'G# minor',
  'Ab major', 'Ab minor',
  'A major', 'A minor',
  'A# major', 'A# minor',
  'Bb major', 'Bb minor',
  'B major', 'B minor'
];

export const TIME_SIGNATURES = ['', '2/4', '3/4', '4/4', '6/8'];

export const VOCAL_LANGUAGE_KEYS = [
  { value: 'unknown', key: 'autoInstrumental' as const },
  { value: 'ar', key: 'vocalArabic' as const },
  { value: 'az', key: 'vocalAzerbaijani' as const },
  { value: 'bg', key: 'vocalBulgarian' as const },
  { value: 'bn', key: 'vocalBengali' as const },
  { value: 'ca', key: 'vocalCatalan' as const },
  { value: 'cs', key: 'vocalCzech' as const },
  { value: 'da', key: 'vocalDanish' as const },
  { value: 'de', key: 'vocalGerman' as const },
  { value: 'el', key: 'vocalGreek' as const },
  { value: 'en', key: 'vocalEnglish' as const },
  { value: 'es', key: 'vocalSpanish' as const },
  { value: 'fa', key: 'vocalPersian' as const },
  { value: 'fi', key: 'vocalFinnish' as const },
  { value: 'fr', key: 'vocalFrench' as const },
  { value: 'he', key: 'vocalHebrew' as const },
  { value: 'hi', key: 'vocalHindi' as const },
  { value: 'hr', key: 'vocalCroatian' as const },
  { value: 'ht', key: 'vocalHaitianCreole' as const },
  { value: 'hu', key: 'vocalHungarian' as const },
  { value: 'id', key: 'vocalIndonesian' as const },
  { value: 'is', key: 'vocalIcelandic' as const },
  { value: 'it', key: 'vocalItalian' as const },
  { value: 'ja', key: 'vocalJapanese' as const },
  { value: 'ko', key: 'vocalKorean' as const },
  { value: 'la', key: 'vocalLatin' as const },
  { value: 'lt', key: 'vocalLithuanian' as const },
  { value: 'ms', key: 'vocalMalay' as const },
  { value: 'ne', key: 'vocalNepali' as const },
  { value: 'nl', key: 'vocalDutch' as const },
  { value: 'no', key: 'vocalNorwegian' as const },
  { value: 'pa', key: 'vocalPunjabi' as const },
  { value: 'pl', key: 'vocalPolish' as const },
  { value: 'pt', key: 'vocalPortuguese' as const },
  { value: 'ro', key: 'vocalRomanian' as const },
  { value: 'ru', key: 'vocalRussian' as const },
  { value: 'sa', key: 'vocalSanskrit' as const },
  { value: 'sk', key: 'vocalSlovak' as const },
  { value: 'sr', key: 'vocalSerbian' as const },
  { value: 'sv', key: 'vocalSwedish' as const },
  { value: 'sw', key: 'vocalSwahili' as const },
  { value: 'ta', key: 'vocalTamil' as const },
  { value: 'te', key: 'vocalTelugu' as const },
  { value: 'th', key: 'vocalThai' as const },
  { value: 'tl', key: 'vocalTagalog' as const },
  { value: 'tr', key: 'vocalTurkish' as const },
  { value: 'uk', key: 'vocalUkrainian' as const },
  { value: 'ur', key: 'vocalUrdu' as const },
  { value: 'vi', key: 'vocalVietnamese' as const },
  { value: 'yue', key: 'vocalCantonese' as const },
  { value: 'zh', key: 'vocalChineseMandarin' as const },
];

export const CreatePanel: React.FC<CreatePanelProps> = ({
  onGenerate,
  isGenerating,
  activeJobCount = 0,
  initialData,
  createdSongs = [],
  pendingAudioSelection,
  onAudioSelectionApplied,
  diffPinnedA,
  diffPinnedB,
  onClearDiffA,
  onClearDiffB,
}) => {
  const { isAuthenticated, token, user } = useAuth();
  const { t } = useI18n();

  // Randomly select 6 music tags from MAIN_STYLES
  const [musicTags, setMusicTags] = useState<string[]>(() => {
    const shuffled = [...MAIN_STYLES].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, 6);
  });

  // Function to refresh music tags
  const refreshMusicTags = useCallback(() => {
    const shuffled = [...MAIN_STYLES].sort(() => Math.random() - 0.5);
    setMusicTags(shuffled.slice(0, 6));
  }, []);

  // Mode
  const [customMode, setCustomMode] = usePersistedState('ace-customMode', true);

  // Simple Mode
  const [songDescription, setSongDescription] = usePersistedState('ace-songDescription', '');

  // Custom Mode
  const [lyrics, setLyrics] = usePersistedState('ace-lyrics', '');
  const [style, setStyle] = usePersistedState('ace-style', '');
  const [title, setTitle] = usePersistedState('ace-title', '');

  // Common
  const [instrumental, setInstrumental] = usePersistedState('ace-instrumental', false);
  const [vocalLanguage, setVocalLanguage] = usePersistedState('ace-vocalLanguage', 'en');
  const [vocalGender, setVocalGender] = usePersistedState<'male' | 'female' | ''>('ace-vocalGender', '');

  // Music Parameters
  const [bpm, setBpm] = usePersistedState('ace-bpm', 0);
  const [keyScale, setKeyScale] = usePersistedState('ace-keyScale', '');
  const [timeSignature, setTimeSignature] = usePersistedState('ace-timeSignature', '');

  // Accordion open/close state
  const [showGenerationSettings, setShowGenerationSettings] = usePersistedState('acestep-showGenerationSettings', false);
  const [showScorePanel, setShowScorePanel] = usePersistedState('ace-showScorePanel', false);
  const [showTrackDetails, setShowTrackDetails] = usePersistedState('ace-showTrackDetails', true);
  const [showLyricsSub, setShowLyricsSub] = usePersistedState('ace-showLyricsSub', true);
  const [showStyleSub, setShowStyleSub] = usePersistedState('ace-showStyleSub', true);
  const [showMusicParamsSub, setShowMusicParamsSub] = usePersistedState('ace-showMusicParamsSub', false);
  const [showCoverSettings, setShowCoverSettings] = usePersistedState('ace-showCoverSettings', true);
  const [showOutputProcessing, setShowOutputProcessing] = usePersistedState('ace-showOutputProcessing', false);
  const [duration, setDuration] = usePersistedState('ace-duration', -1);
  const [batchSize, setBatchSize] = usePersistedState('ace-batchSize', 1);
  const [bulkCount, setBulkCount] = usePersistedState('ace-bulkCount', 1);
  const [guidanceScale, setGuidanceScale] = usePersistedState('ace-guidanceScale', 9.0);
  const [randomSeed, setRandomSeed] = usePersistedState('ace-randomSeed', true);
  const [seed, setSeed] = usePersistedState('ace-seed', -1);
  const [thinking, setThinking] = usePersistedState('ace-thinking', false); // Default false for GPU compatibility
  const [audioFormat, setAudioFormat] = usePersistedState<'mp3' | 'flac' | 'wav' | 'opus'>('ace-audioFormat', 'mp3');
  const [inferenceSteps, setInferenceSteps] = usePersistedState('ace-inferenceSteps', 12);
  const [inferMethod, setInferMethod] = usePersistedState<'ode' | 'euler' | 'heun' | 'dpm2m' | 'rk4'>('ace-inferMethod', 'ode');
  const [scheduler, setScheduler] = usePersistedState<string>('ace-scheduler', 'linear');
  const [lmBackend, setLmBackend] = usePersistedState<'pt' | 'vllm'>('ace-lmBackend', 'pt');
  const [lmModel, setLmModel] = usePersistedState('ace-lmModel', 'acestep-5Hz-lm-0.6B');
  const [shift, setShift] = usePersistedState('ace-shift', 3.0);

  // LM Parameters (under Expert)
  const [showLmParams, setShowLmParams] = usePersistedState('ace-showLmParams', false);
  const [lmTemperature, setLmTemperature] = usePersistedState('ace-lmTemperature', 0.8);
  const [lmCfgScale, setLmCfgScale] = usePersistedState('ace-lmCfgScale', 2.2);
  const [lmTopK, setLmTopK] = usePersistedState('ace-lmTopK', 0);
  const [lmTopP, setLmTopP] = usePersistedState('ace-lmTopP', 0.92);
  const [lmRepetitionPenalty, setLmRepetitionPenalty] = usePersistedState('ace-lmRepetitionPenalty', 1.0);
  const [lmNegativePrompt, setLmNegativePrompt] = usePersistedState('ace-lmNegativePrompt', 'NO USER INPUT');

  // Expert Parameters — session-specific audio state (NOT persisted)
  const [referenceAudioUrl, setReferenceAudioUrl] = useState('');
  const [sourceAudioUrl, setSourceAudioUrl] = useState('');
  const [referenceAudioTitle, setReferenceAudioTitle] = useState('');
  const [sourceAudioTitle, setSourceAudioTitle] = useState('');

  // Source audio analysis state (Essentia BPM/key detection)
  const [detectedBpm, setDetectedBpm] = useState<number | null>(null);
  const [detectedKey, setDetectedKey] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [audioCodes, setAudioCodes] = useState('');
  const [repaintingStart, setRepaintingStart] = useState(0);
  const [repaintingEnd, setRepaintingEnd] = useState(-1);
  const [instruction, setInstruction] = useState(t('instructionDefault'));
  const [audioCoverStrength, setAudioCoverStrength] = usePersistedState('ace-audioCoverStrength', 1.0);
  const [coverNoiseStrength, setCoverNoiseStrength] = usePersistedState('ace-coverNoiseStrength', 0.15);
  const [tempoScale, setTempoScale] = usePersistedState('ace-tempoScale', 1.0);
  const [pitchShift, setPitchShift] = usePersistedState('ace-pitchShift', 0);
  const [autoMaster, setAutoMaster] = usePersistedState('ace-autoMaster', true);
  const [masteringParams, setMasteringParams] = useState<MasteringParamsType | null>(null);
  const [showMasteringConsole, setShowMasteringConsole] = useState(false);
  const [enableNormalization, setEnableNormalization] = usePersistedState('ace-enableNormalization', true);
  const [normalizationDb, setNormalizationDb] = usePersistedState('ace-normalizationDb', -1.0);
  const [latentShift, setLatentShift] = usePersistedState('ace-latentShift', 0.0);
  const [latentRescale, setLatentRescale] = usePersistedState('ace-latentRescale', 1.0);
  const [taskType, setTaskType] = usePersistedState('ace-taskType', 'text2music');
  const [useAdg, setUseAdg] = usePersistedState('ace-useAdg', false);
  // Guidance Mode: 'apg' (default), 'adg', or 'pag'
  const [guidanceMode, setGuidanceMode] = usePersistedState<'apg' | 'adg' | 'pag' | 'cfg_pp' | 'dynamic_cfg' | 'rescaled_cfg'>('ace-guidanceMode', 'apg');
  // PAG (Perturbed-Attention Guidance) Parameters
  const [usePag, setUsePag] = usePersistedState('ace-usePag', false);
  const [pagStart, setPagStart] = usePersistedState('ace-pagStart', 0.30);
  const [pagEnd, setPagEnd] = usePersistedState('ace-pagEnd', 0.80);
  const [pagScale, setPagScale] = usePersistedState('ace-pagScale', 0.2);
  const [cfgIntervalStart, setCfgIntervalStart] = usePersistedState('ace-cfgIntervalStart', 0.0);
  const [cfgIntervalEnd, setCfgIntervalEnd] = usePersistedState('ace-cfgIntervalEnd', 1.0);
  const [customTimesteps, setCustomTimesteps] = useState('');
  const [useCotMetas, setUseCotMetas] = usePersistedState('ace-useCotMetas', true);
  const [useCotCaption, setUseCotCaption] = usePersistedState('ace-useCotCaption', true);
  const [useCotLanguage, setUseCotLanguage] = usePersistedState('ace-useCotLanguage', true);
  const [autogen, setAutogen] = useState(false);
  const [constrainedDecodingDebug, setConstrainedDecodingDebug] = useState(false);
  const [allowLmBatch, setAllowLmBatch] = usePersistedState('ace-allowLmBatch', true);
  const [getScores, setGetScores] = usePersistedState('ace-getScores', false);
  const [getLrc, setGetLrc] = useState(true);
  const [scoreScale, setScoreScale] = usePersistedState('ace-scoreScale', 0.5);
  const [lmBatchChunkSize, setLmBatchChunkSize] = usePersistedState('ace-lmBatchChunkSize', 8);
  const [trackName, setTrackName] = useState('');
  const [extractTracks, setExtractTracks] = usePersistedState<string[]>('ace-extractTracks', ['vocals']);
  const [completeTrackClasses, setCompleteTrackClasses] = useState('');
  const [isFormatCaption, setIsFormatCaption] = usePersistedState('ace-isFormatCaption', false);
  const [maxDurationWithLm, setMaxDurationWithLm] = usePersistedState('ace-maxDurationWithLm', 240);
  const [maxDurationWithoutLm, setMaxDurationWithoutLm] = usePersistedState('ace-maxDurationWithoutLm', 240);

  // LoRA Parameters
  const [showLoraPanel, setShowLoraPanel] = usePersistedState('ace-showLoraPanel', false);
  const [loraPath, setLoraPath] = usePersistedState('ace-loraPath', './lokr_output/final/lokr_weights.safetensors');
  const [loraLoaded, setLoraLoaded] = useState(false);
  const [loraScale, setLoraScale] = usePersistedState('ace-loraScale', 1.0);
  const [loraError, setLoraError] = useState<string | null>(null);
  const [adapterTriggerWord, setAdapterTriggerWord] = useState('');
  const [isLoraLoading, setIsLoraLoading] = useState(false);

  // LM LoRA Parameters (PEFT adapter on the 5Hz language model)
  const [lmLoraPath, setLmLoraPath] = usePersistedState('ace-lmLoraPath', '');
  const [lmLoraLoaded, setLmLoraLoaded] = useState(false);
  const [lmLoraScale, setLmLoraScale] = usePersistedState('ace-lmLoraScale', 1.0);
  const [lmLoraError, setLmLoraError] = useState<string | null>(null);
  const [isLmLoraLoading, setIsLmLoraLoading] = useState(false);


  // Advanced adapter state
  const [advancedAdapters, setAdvancedAdapters] = usePersistedState('ace-advancedAdapters', false);
  const [adapterFolder, setAdapterFolder] = usePersistedState('ace-adapterFolder', './lokr_output');
  const [adapterFiles, setAdapterFiles] = useState<Array<{ name: string; path: string; size: number; type: string }>>([]);
  const [loadingAdapterPath, setLoadingAdapterPath] = useState<string | null>(null);;
  const [adapterSlots, setAdapterSlots] = useState<Array<{
    slot: number; name: string; path: string; type: string; scale: number;
    delta_keys: number; group_scales: { self_attn: number; cross_attn: number; mlp: number };
  }>>([]);
  const [expandedSlots, setExpandedSlots] = useState<Set<number>>(new Set());
  // Per-adapter persisted scales (keyed by adapter filename)
  const [savedGroupScales, setSavedGroupScales] = usePersistedState<Record<string, { self_attn: number; cross_attn: number; mlp: number }>>('ace-adapterGroupScales', {});
  const [savedOverallScales, setSavedOverallScales] = usePersistedState<Record<string, number>>('ace-adapterOverallScales', {});
  const [adapterLoadingMessage, setAdapterLoadingMessage] = useState<string | null>(null);

  // Activation Steering State
  const [showSteeringPanel, setShowSteeringPanel] = usePersistedState('ace-showSteeringPanel', false);
  const [steeringEnabled, setSteeringEnabled] = useState(false);
  const [steeringLoaded, setSteeringLoaded] = useState<string[]>([]);
  const [steeringAlphas, setSteeringAlphas] = useState<Record<string, number>>({});

  // Model selection
  const [selectedModel, setSelectedModel] = usePersistedState('ace-model', 'acestep-v15-turbo-shift3');
  const [showModelMenu, setShowModelMenu] = useState(false);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const previousModelRef = useRef<string>(selectedModel);
  const isMountedRef = useRef(true);
  const modelsRetryTimeoutRef = useRef<number | null>(null);

  // Available models fetched from backend
  const [fetchedModels, setFetchedModels] = useState<{ name: string; is_active: boolean; is_preloaded: boolean }[]>([]);
  const [activeBackendModel, setActiveBackendModel] = useState<string | null>(null);
  const [isSwitching, setIsSwitching] = useState(false);
  const [backendUnavailable, setBackendUnavailable] = useState(false);

  // Fallback model list when backend is unavailable
  const availableModels = useMemo(() => {
    if (fetchedModels.length > 0) {
      return fetchedModels.map(m => ({ id: m.name, name: m.name }));
    }
    // Fallback — only default ACE-Step models.
    // Replaced by dynamic list from /v1/models once Python API starts.
    return [
      { id: 'acestep-v15-base', name: 'acestep-v15-base' },
      { id: 'acestep-v15-sft', name: 'acestep-v15-sft' },
      { id: 'acestep-v15-turbo', name: 'acestep-v15-turbo' },
      { id: 'acestep-v15-turbo-shift1', name: 'acestep-v15-turbo-shift1' },
      { id: 'acestep-v15-turbo-shift3', name: 'acestep-v15-turbo-shift3' },
      { id: 'acestep-v15-turbo-continuous', name: 'acestep-v15-turbo-continuous' },
    ];
  }, [fetchedModels]);

  // Map model ID to short display name
  const getModelDisplayName = (modelId: string): string => {
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
    };
    return mapping[modelId] || modelId;
  };

  // Check if model is a turbo variant (excluding merged models —
  // SFT+Turbo merges are weight averages, NOT pure distillation checkpoints)
  const isTurboModel = (modelId: string): boolean => {
    return modelId.includes('turbo') && !modelId.includes('merge');
  };

  // Check if model is a pure base model (only base supports extract/lego/complete)
  const isBaseModel = (modelId: string): boolean => {
    return modelId.includes('base');
  };

  const isBaseOnlyTask = (task: string): boolean => {
    return ['extract', 'lego', 'complete'].includes(task);
  };

  // Genre selection state (cascading)
  // Two-level genre cascade states
  const [showGenreDropdown, setShowGenreDropdown] = useState(false);
  const [showSubGenreDropdown, setShowSubGenreDropdown] = useState(false);
  const [genreSearch, setGenreSearch] = useState<string>('');
  const [selectedMainGenre, setSelectedMainGenre] = useState<string>('');
  const [selectedSubGenre, setSelectedSubGenre] = useState<string>('');
  const genreDropdownRef = useRef<HTMLDivElement>(null);
  const subGenreDropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (genreDropdownRef.current && !genreDropdownRef.current.contains(event.target as Node)) {
        setShowGenreDropdown(false);
      }
      if (subGenreDropdownRef.current && !subGenreDropdownRef.current.contains(event.target as Node)) {
        setShowSubGenreDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Get sub-genres for a main genre (styles that contain the main genre name)
  const getSubGenres = (mainGenre: string) => {
    return ALL_STYLES.filter(style =>
      style.toLowerCase().includes(mainGenre.toLowerCase()) &&
      style.toLowerCase() !== mainGenre.toLowerCase()
    );
  };

  // Get sub-genres count for each main genre
  const getSubGenreCount = (mainGenre: string) => {
    return getSubGenres(mainGenre).length;
  };

  // Other genres: ALL_STYLES 中既不是 MAIN_STYLES，也不是任何 MAIN_STYLE 的 sub-genre
  const OTHER_GENRES = useMemo(() => {
    const mainStylesLower = new Set(MAIN_STYLES.map(s => s.toLowerCase()));

    // 检查一个风格是否是某个 main genre 的 sub-genre
    const isSubGenreOfAnyMain = (style: string): boolean => {
      const styleLower = style.toLowerCase();
      return MAIN_STYLES.some(mainGenre => {
        const mainLower = mainGenre.toLowerCase();
        // 风格包含主流派关键词，且不是主流派本身
        return styleLower !== mainLower && styleLower.includes(mainLower);
      });
    };

    return ALL_STYLES.filter(style => {
      const styleLower = style.toLowerCase();
      // 不是 main style 本身
      if (mainStylesLower.has(styleLower)) return false;
      // 不是任何 main genre 的 sub-genre
      if (isSubGenreOfAnyMain(style)) return false;
      return true;
    });
  }, []);

  // Filter sub-genres based on selected main genre
  const filteredSubGenres = useMemo(() => {
    if (!selectedMainGenre) return [];
    return getSubGenres(selectedMainGenre);
  }, [selectedMainGenre]);

  // Combined and sorted genres for first level dropdown
  const combinedGenres = useMemo(() => {
    const mainSet = new Set(MAIN_STYLES.map(s => s.toLowerCase()));
    // Combine both lists with type indicator
    const combined = [
      ...MAIN_STYLES.map(g => ({ name: g, type: 'main' as const })),
      ...OTHER_GENRES.map(g => ({ name: g, type: 'other' as const }))
    ];
    // Sort alphabetically by name
    return combined.sort((a, b) => a.name.localeCompare(b.name));
  }, []);

  // Filter combined genres based on search query
  const filteredCombinedGenres = useMemo(() => {
    if (!genreSearch) return combinedGenres;
    return combinedGenres.filter(g => g.name.toLowerCase().includes(genreSearch.toLowerCase()));
  }, [genreSearch, combinedGenres]);

  const [isUploadingReference, setIsUploadingReference] = useState(false);
  const [isUploadingSource, setIsUploadingSource] = useState(false);
  const [isTranscribingReference, setIsTranscribingReference] = useState(false);
  const transcribeAbortRef = useRef<AbortController | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isFormattingStyle, setIsFormattingStyle] = useState(false);
  const [isFormattingLyrics, setIsFormattingLyrics] = useState(false);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [dragKind, setDragKind] = useState<'file' | 'audio' | null>(null);
  const referenceInputRef = useRef<HTMLInputElement>(null);
  const sourceInputRef = useRef<HTMLInputElement>(null);
  const dragDepthRef = useRef(0);
  const [showAudioModal, setShowAudioModal] = useState(false);
  const [audioModalTarget, setAudioModalTarget] = useState<'reference' | 'source'>('reference');
  const [tempAudioUrl, setTempAudioUrl] = useState('');
  const [useReferenceAudio, setUseReferenceAudio] = usePersistedState('acestep-useReferenceAudio', false);

  const referenceAudioRef = useRef<HTMLAudioElement>(null);
  const sourceAudioRef = useRef<HTMLAudioElement>(null);
  const [referencePlaying, setReferencePlaying] = useState(false);
  const [sourcePlaying, setSourcePlaying] = useState(false);
  const [referenceTime, setReferenceTime] = useState(0);
  const [sourceTime, setSourceTime] = useState(0);
  const [referenceDuration, setReferenceDuration] = useState(0);
  const [sourceDuration, setSourceDuration] = useState(0);

  // When 'Use Reference Audio' is toggled OFF, clear the loaded reference audio
  useEffect(() => {
    if (!useReferenceAudio) {
      setReferenceAudioUrl('');
      setReferenceAudioTitle('');
    }
  }, [useReferenceAudio]);

  // Reference tracks modal state
  const [referenceTracks, setReferenceTracks] = useState<ReferenceTrack[]>([]);
  const [isLoadingTracks, setIsLoadingTracks] = useState(false);
  const [playingTrackId, setPlayingTrackId] = useState<string | null>(null);
  const [playingTrackSource, setPlayingTrackSource] = useState<'uploads' | 'created' | null>(null);
  const modalAudioRef = useRef<HTMLAudioElement>(null);
  const [modalTrackTime, setModalTrackTime] = useState(0);
  const [modalTrackDuration, setModalTrackDuration] = useState(0);
  const [libraryTab, setLibraryTab] = useState<'uploads' | 'created'>('uploads');

  const createdTrackOptions = useMemo(() => {
    return createdSongs
      .filter(song => !song.isGenerating)
      .filter(song => (user ? song.userId === user.id : true))
      .filter(song => Boolean(song.audioUrl))
      .map(song => ({
        id: song.id,
        title: song.title || t('untitled'),
        audio_url: song.audioUrl!,
        duration: song.duration,
      }));
  }, [createdSongs, user]);

  const getAudioLabel = (url: string) => {
    try {
      const parsed = new URL(url);
      const name = decodeURIComponent(parsed.pathname.split('/').pop() || parsed.hostname);
      return name.replace(/\.[^/.]+$/, '') || name;
    } catch {
      const parts = url.split('/');
      const name = decodeURIComponent(parts[parts.length - 1] || url);
      return name.replace(/\.[^/.]+$/, '') || name;
    }
  };

  // Resize Logic
  const [lyricsHeight, setLyricsHeight] = useState(() => {
    const saved = localStorage.getItem('acestep_lyrics_height');
    return saved ? parseInt(saved, 10) : 144; // Default h-36 is 144px (9rem * 16)
  });
  const [isResizing, setIsResizing] = useState(false);
  const lyricsRef = useRef<HTMLDivElement>(null);

  const [styleHeight, setStyleHeight] = useState(() => {
    const saved = localStorage.getItem('acestep_style_height');
    return saved ? parseInt(saved, 10) : 80; // default initial height
  });
  const [isResizingStyle, setIsResizingStyle] = useState(false);
  const styleRef = useRef<HTMLDivElement>(null);


  // Close model menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(event.target as Node)) {
        setShowModelMenu(false);
      }
    };

    if (showModelMenu) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showModelMenu]);

  // Track model changes (fall back to text2music for base-only tasks on non-base models)
  useEffect(() => {
    // Note: LoRA is NOT unloaded here — handleSwitchModel handles re-application
    if (previousModelRef.current !== selectedModel && !isBaseModel(selectedModel) && isBaseOnlyTask(taskType)) {
      setTaskType('text2music');
    }
    previousModelRef.current = selectedModel;
  }, [selectedModel]);

  // Auto-disable advanced guidance modes when basic LoRA is loaded (PEFT hook limitation).
  // Advanced multi-adapter mode uses weight-space merging — no hook restriction — so we skip this there.
  useEffect(() => {
    if (loraLoaded && !advancedAdapters) {
      if (guidanceMode !== 'apg') {
        setGuidanceMode('apg');
        setUseAdg(false);
        setUsePag(false);
      }
    }
  }, [loraLoaded, advancedAdapters]);



  // LoRA API handlers
  const handleLoraToggle = async () => {
    if (!token) {
      setLoraError(t('pleaseSignInToUseLoRA'));
      return;
    }
    if (!loraPath.trim()) {
      setLoraError(t('pleaseEnterLoRAPath'));
      return;
    }

    setIsLoraLoading(true);
    setLoraError(null);

    try {
      if (loraLoaded) {
        await handleLoraUnload();
      } else {
        const result = await generateApi.loadLora({ lora_path: loraPath }, token);
        setLoraLoaded(true);
        console.log('LoRA loaded:', result?.message);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'LoRA operation failed';
      setLoraError(message);
      console.error('LoRA error:', err);
    } finally {
      setIsLoraLoading(false);
    }
  };

  const handleLoraUnload = async () => {
    if (!token) return;

    setIsLoraLoading(true);
    setLoraError(null);

    try {
      const result = await generateApi.unloadLora(token);
      setLoraLoaded(false);
      console.log('LoRA unloaded:', result?.message);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to unload LoRA';
      setLoraError(message);
      console.error('Unload error:', err);
    } finally {
      setIsLoraLoading(false);
    }
  };

  const handleLoraScaleChange = async (newScale: number) => {
    setLoraScale(newScale);

    if (!token || !loraLoaded) return;

    try {
      await generateApi.setLoraScale({ scale: newScale }, token);
    } catch (err) {
      console.error('Failed to set LoRA scale:', err);
    }
  };


  // LM LoRA API handlers
  const handleLmLoraToggle = async () => {
    if (!token) {
      setLmLoraError('Please sign in to use LM LoRA');
      return;
    }
    if (!lmLoraLoaded && !lmLoraPath.trim()) {
      setLmLoraError('Please enter an LM adapter path');
      return;
    }

    setIsLmLoraLoading(true);
    setLmLoraError(null);

    try {
      if (lmLoraLoaded) {
        const result = await generateApi.unloadLmLora(token);
        setLmLoraLoaded(false);
        console.log('LM LoRA unloaded:', result?.message);
      } else {
        const result = await generateApi.loadLmLora({ lm_lora_path: lmLoraPath, scale: lmLoraScale }, token);
        setLmLoraLoaded(true);
        console.log('LM LoRA loaded:', result?.message);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'LM LoRA operation failed';
      setLmLoraError(message);
      console.error('LM LoRA error:', err);
    } finally {
      setIsLmLoraLoading(false);
    }
  };

  const handleLmLoraScaleChange = async (newScale: number) => {
    setLmLoraScale(newScale);
    // Scale is baked at merge time — changing it while loaded requires re-merge
    // We only update local state here; user must unload+reload to apply new scale
  };



  // Advanced adapter handlers
  const handleScanFolder = async () => {
    if (!token || !adapterFolder.trim()) return;
    setLoraError(null);
    try {
      const result = await generateApi.listLoraFiles(adapterFolder, token);
      setAdapterFiles(result.files || []);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to scan folder';
      setLoraError(msg);
    }
  };

  const handleLoadSlot = async (filePath: string) => {
    if (!token) return;
    setIsLoraLoading(true);
    setLoadingAdapterPath(filePath);
    setAdapterLoadingMessage('Loading adapter weights...');
    setLoraError(null);
    try {
      const nextSlot = adapterSlots.length > 0
        ? Math.max(...adapterSlots.map(s => s.slot)) + 1
        : 0;
      await generateApi.loadLora({
        lora_path: filePath,
        slot: nextSlot,
        // Atomically apply saved scales in the same request
        ...(savedOverallScales[filePath.split('/').pop()?.replace(/\.[^/.]+$/, '') || ''] != null &&
          savedOverallScales[filePath.split('/').pop()?.replace(/\.[^/.]+$/, '') || ''] !== 1.0
          ? { scale: savedOverallScales[filePath.split('/').pop()?.replace(/\.[^/.]+$/, '') || ''] }
          : {}),
        ...(savedGroupScales[filePath.split('/').pop()?.replace(/\.[^/.]+$/, '') || ''] != null
          ? { group_scales: savedGroupScales[filePath.split('/').pop()?.replace(/\.[^/.]+$/, '') || ''] }
          : {}),
      }, token);
      // Refresh status to get actual slot info
      const status = await generateApi.getLoraStatus(token);
      if (status?.advanced?.slots) {
        setAdapterSlots(status.advanced.slots);
        setLoraLoaded(true);
        setAdapterTriggerWord((status as any).trigger_word || '');
      }
    } catch (err) {
      setLoraError(err instanceof Error ? err.message : 'Failed to load adapter');
    } finally {
      setIsLoraLoading(false);
      setLoadingAdapterPath(null);
      setAdapterLoadingMessage(null);
    }
  };

  const handleUnloadSlot = async (slot: number) => {
    if (!token) return;
    setIsLoraLoading(true);
    setLoraError(null);
    try {
      await generateApi.unloadLora(token, slot);
      const status = await generateApi.getLoraStatus(token);
      if (status?.advanced) {
        setAdapterSlots(status.advanced.slots || []);
        setLoraLoaded(status.advanced.loaded);
        setAdapterTriggerWord((status as any).trigger_word || '');
      } else {
        setAdapterSlots([]);
        setLoraLoaded(false);
        setAdapterTriggerWord('');
      }
    } catch (err) {
      setLoraError(err instanceof Error ? err.message : 'Failed to unload');
    } finally {
      setIsLoraLoading(false);
    }
  };

  const handleSlotScaleChange = async (slot: number, scale: number) => {
    if (!token) return;
    setAdapterSlots(prev => prev.map(s => s.slot === slot ? { ...s, scale } : s));
    // Persist per-adapter overall scale
    const slotData = adapterSlots.find(s => s.slot === slot);
    if (slotData) {
      setSavedOverallScales(prev => ({ ...prev, [slotData.name]: scale }));
    }
    try {
      await generateApi.setLoraScale({ scale, slot }, token);
    } catch (err) {
      console.error('Failed to set slot scale:', err);
    }
  };

  const handleSlotGroupScaleChange = async (slot: number, group: 'self_attn' | 'cross_attn' | 'mlp', value: number) => {
    if (!token) return;
    const slotData = adapterSlots.find(s => s.slot === slot);
    if (!slotData) return;
    const newScales = { ...slotData.group_scales, [group]: value };
    setAdapterSlots(prev => prev.map(s => s.slot === slot ? { ...s, group_scales: newScales } : s));
    // Save per-adapter
    const key = slotData.name;
    setSavedGroupScales(prev => ({ ...prev, [key]: newScales }));
    try {
      await generateApi.setSlotGroupScales({ slot, ...newScales }, token);
    } catch (err) {
      console.error('Failed to set slot group scales:', err);
    }
  };

  // Debounce layer scale API calls — state updates immediately (visual),
  // but the expensive backend merge only fires after 500ms of no changes.
  const layerScaleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingSlotRef = useRef<number | null>(null);

  const handleSlotLayerScaleChange = (slot: number, layer: number, scale: number) => {
    if (!token) return;
    // Immediate state update for visual feedback
    setAdapterSlots(prev => prev.map(s => {
      if (s.slot !== slot) return s;
      const newLayerScales = { ...(s.layer_scales || {}), [layer]: scale };
      if (Math.abs(scale - 1.0) < 0.01) delete newLayerScales[layer];
      return { ...s, layer_scales: newLayerScales };
    }));
    // Track which slot needs flushing
    pendingSlotRef.current = slot;
    if (layerScaleTimerRef.current) clearTimeout(layerScaleTimerRef.current);
    layerScaleTimerRef.current = setTimeout(() => {
      const slotId = pendingSlotRef.current;
      pendingSlotRef.current = null;
      if (slotId === null) return;
      // Read ALL current layer scales from state (not just the diff)
      // so the backend gets the complete picture
      setAdapterSlots(current => {
        const slotData = current.find(s => s.slot === slotId);
        const allScales = slotData?.layer_scales || {};
        generateApi.setSlotLayerScales({ slot: slotId, layer_scales: allScales }, token)
          .catch(err => console.error('Failed to set slot layer scales:', err));
        return current; // no state change, just reading
      });
    }, 500);
  };

  const handleBulkLayerScalesChange = async (slot: number, layerScales: Record<number, number>) => {
    if (!token) return;
    // Single state update for all layers
    setAdapterSlots(prev => prev.map(s => {
      if (s.slot !== slot) return s;
      const newLayerScales = { ...(s.layer_scales || {}) };
      for (const [layer, scale] of Object.entries(layerScales)) {
        if (Math.abs(scale - 1.0) < 0.01) {
          delete newLayerScales[Number(layer)];
        } else {
          newLayerScales[Number(layer)] = scale;
        }
      }
      return { ...s, layer_scales: newLayerScales };
    }));
    // Single API call
    try {
      await generateApi.setSlotLayerScales({ slot, layer_scales: layerScales }, token);
    } catch (err) {
      console.error('Failed to set slot layer scales:', err);
    }
  };

  const [temporalScheduleActive, setTemporalScheduleActive] = useState(false);

  // ────────────────────────────────────────────────────────────────────────────
  // Ablation Sweep state
  // ────────────────────────────────────────────────────────────────────────────
  const [sweepRunning, setSweepRunning] = useState(false);
  const [sweepProgress, setSweepProgress] = useState<{ current: number; total: number } | null>(null);
  const sweepCancelledRef = useRef(false);
  // Resolver that the sweep loop waits on; resolved when isGenerating → false
  const generationDoneResolverRef = useRef<(() => void) | null>(null);
  const sweepIsGeneratingRef = useRef(isGenerating);

  // Keep the ref in sync with the prop
  useEffect(() => {
    sweepIsGeneratingRef.current = isGenerating;
  }, [isGenerating]);

  // When sweep is active and isGenerating transitions true→false, resolve the waiter
  const prevSweepGeneratingRef = useRef(false);
  useEffect(() => {
    if (!sweepRunning) return;
    if (prevSweepGeneratingRef.current && !isGenerating) {
      // Generation just completed
      generationDoneResolverRef.current?.();
      generationDoneResolverRef.current = null;
    }
    prevSweepGeneratingRef.current = isGenerating;
  }, [isGenerating, sweepRunning]);

  const waitForGenerationDone = (): Promise<void> => {
    return new Promise((resolve) => {
      // If not currently generating, resolve immediately
      if (!sweepIsGeneratingRef.current) {
        resolve();
        return;
      }
      generationDoneResolverRef.current = resolve;
    });
  };

  const handleStartSweep = async () => {
    if (!adapterSlots.length) return;
    const slot = adapterSlots[0].slot;
    const totalLayers = 24;
    sweepCancelledRef.current = false;
    setSweepRunning(true);
    setSweepProgress({ current: 0, total: totalLayers });

    for (let layer = 0; layer < totalLayers; layer++) {
      if (sweepCancelledRef.current) break;

      // Build layer scales: all 1.0 except target layer = 0.0
      const layerScales: Record<number, number> = {};
      for (let i = 0; i < totalLayers; i++) {
        layerScales[i] = i === layer ? 0.0 : 1.0;
      }

      try {
        await handleBulkLayerScalesChange(slot, layerScales);
      } catch (err) {
        console.error(`[Sweep] Failed to set layer scales for layer ${layer}:`, err);
      }

      if (sweepCancelledRef.current) break;

      // Build title: pad layer index to 2 digits
      const layerSuffix = ` - layer${String(layer).padStart(2, '0')}`;
      const sweepTitle = (title || 'Ablation').trim() + layerSuffix;

      // Trigger generation (uses current form state, but overrides title)
      const styleWithGender = (() => {
        if (!vocalGender) return style;
        const genderHint = vocalGender === 'male' ? t('maleVocals') : t('femaleVocals');
        const trimmed = style.trim();
        return trimmed ? `${trimmed}\n${genderHint}` : genderHint;
      })();

      onGenerate({
        customMode,
        prompt: lyrics,
        lyrics,
        style: styleWithGender,
        title: sweepTitle,
        ditModel: selectedModel,
        instrumental,
        vocalLanguage,
        bpm,
        keyScale,
        timeSignature,
        duration,
        inferenceSteps,
        guidanceScale,
        batchSize: 1,
        randomSeed: false, // Fixed seed so all 24 are comparable
        seed,
        thinking,
        audioFormat,
        inferMethod,
        scheduler,
        lmBackend,
        lmModel,
        shift,
        lmTemperature,
        lmCfgScale,
        lmTopK,
        lmTopP,
        lmNegativePrompt,
        steeringEnabled,
        steeringLoaded,
        steeringAlphas,
        referenceAudioUrl: (useReferenceAudio && referenceAudioUrl.trim()) || undefined,
        sourceAudioUrl: sourceAudioUrl.trim() || undefined,
        taskType: 'text2music',
        loraLoaded,
        advancedAdapters,
        adapterSlots: adapterSlots.map(s => ({ ...s })),
      });

      // Wait for this generation to complete before proceeding
      await waitForGenerationDone();

      setSweepProgress({ current: layer + 1, total: totalLayers });
    }

    // Restore all layers to 1.0 after sweep
    const resetScales: Record<number, number> = {};
    for (let i = 0; i < 24; i++) resetScales[i] = 1.0;
    await handleBulkLayerScalesChange(slot, resetScales).catch(() => undefined);

    setSweepRunning(false);
    setSweepProgress(null);
  };

  const handleCancelSweep = () => {
    sweepCancelledRef.current = true;
  };

  const handleTemporalSchedulePreset = async (preset: 'switch' | 'verse-chorus' | null) => {
    if (!token) return;
    try {
      if (preset === null) {
        await generateApi.setTemporalSchedule({ clear: true }, token);
        setTemporalScheduleActive(false);
        return;
      }
      // Use first two loaded adapter slots
      const slotA = adapterSlots[0]?.slot;
      const slotB = adapterSlots[1]?.slot;
      if (slotA === undefined || slotB === undefined) return;

      let slot_segments: Record<number, Array<{ start: number; end: number; scale?: number; fade_in?: number; fade_out?: number }>>;

      if (preset === 'switch') {
        // A plays 0-55%, B plays 45-100%, crossfade 45-55%
        slot_segments = {
          [slotA]: [{ start: 0.0, end: 0.55, scale: 1.0, fade_out: 0.1 }],
          [slotB]: [{ start: 0.45, end: 1.0, scale: 1.0, fade_in: 0.1 }],
        };
      } else {
        // verse-chorus: A=0-30%, B=25-70%, A=65-100% with crossfades
        slot_segments = {
          [slotA]: [
            { start: 0.0, end: 0.30, scale: 1.0, fade_out: 0.05 },
            { start: 0.65, end: 1.0, scale: 1.0, fade_in: 0.05 },
          ],
          [slotB]: [
            { start: 0.25, end: 0.70, scale: 1.0, fade_in: 0.05, fade_out: 0.05 },
          ],
        };
      }
      await generateApi.setTemporalSchedule({ slot_segments }, token);
      setTemporalScheduleActive(true);
    } catch (err) {
      console.error('Failed to set temporal schedule:', err);
    }
  };

  // Reuse Effect - must be after all state declarations
  useEffect(() => {
    if (initialData) {
      setCustomMode(true);
      setStyle(initialData.song.style);
      setTitle(initialData.song.title);

      // Check if song is instrumental (empty lyrics, [Instrumental], or Instrumental)
      const trimmedLyrics = initialData.song.lyrics?.trim() || '';
      const isInstrumentalSong = trimmedLyrics.length === 0 ||
        trimmedLyrics === '[Instrumental]' ||
        trimmedLyrics === 'Instrumental';

      setInstrumental(isInstrumentalSong);
      // Only set lyrics if not instrumental
      setLyrics(isInstrumentalSong ? '' : initialData.song.lyrics);

      // Restore full generation parameters if available
      const gp = initialData.song.generationParams;
      if (gp) {
        if (gp.bpm != null) setBpm(gp.bpm);
        if (gp.keyScale != null) setKeyScale(gp.keyScale);
        if (gp.timeSignature != null) setTimeSignature(gp.timeSignature);
        if (gp.duration != null) setDuration(gp.duration);
        if (gp.inferenceSteps != null) setInferenceSteps(gp.inferenceSteps);
        if (gp.guidanceScale != null) setGuidanceScale(gp.guidanceScale);
        if (gp.seed != null && gp.seed >= 0) {
          setSeed(gp.seed);
          setRandomSeed(false);
        }
        if (gp.inferMethod) setInferMethod(gp.inferMethod);
        if (gp.scheduler) setScheduler(gp.scheduler);
        if (gp.shift != null) setShift(gp.shift);
        if (gp.audioFormat) setAudioFormat(gp.audioFormat);
        if (gp.thinking != null) setThinking(gp.thinking);
        if (gp.ditModel) setSelectedModel(gp.ditModel);
        if (gp.vocalLanguage) setVocalLanguage(gp.vocalLanguage);
        if (gp.referenceAudioUrl) {
          setReferenceAudioUrl(gp.referenceAudioUrl);
          if (gp.referenceAudioTitle) setReferenceAudioTitle(gp.referenceAudioTitle);
        }
        if (gp.sourceAudioUrl) {
          setSourceAudioUrl(gp.sourceAudioUrl);
          if (gp.sourceAudioTitle) setSourceAudioTitle(gp.sourceAudioTitle);
        }
        if (gp.taskType) setTaskType(gp.taskType);
        if (gp.lmBackend) setLmBackend(gp.lmBackend);
        if (gp.lmModel) setLmModel(gp.lmModel);
      } else if (initialData.song.ditModel) {
        setSelectedModel(initialData.song.ditModel);
      }
    }
  }, [initialData]);

  useEffect(() => {
    if (!pendingAudioSelection) return;
    applyAudioTargetUrl(
      pendingAudioSelection.target,
      pendingAudioSelection.url,
      pendingAudioSelection.title
    );
    onAudioSelectionApplied?.();
  }, [pendingAudioSelection, onAudioSelectionApplied]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;

      // Calculate new height based on mouse position relative to the lyrics container top
      // We can't easily get the container top here without a ref to it,
      // but we can use dy (delta y) from the previous position if we tracked it,
      // OR simpler: just update based on movement if we track the start.
      //
      // Better approach for absolute sizing:
      // 1. Get the bounding rect of the textarea wrapper on mount/resize start?
      //    We can just rely on the fact that we are dragging the bottom.
      //    So new height = currentMouseY - topOfElement.

      if (lyricsRef.current) {
        const rect = lyricsRef.current.getBoundingClientRect();
        const newHeight = e.clientY - rect.top;
        // detailed limits: min 96px (h-24), max 600px
        if (newHeight > 96 && newHeight < 600) {
          setLyricsHeight(newHeight);
        }
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.body.style.cursor = 'default';
      document.body.style.userSelect = 'auto';
      // Save height to localStorage
      localStorage.setItem('acestep_lyrics_height', String(lyricsHeight));
    };

    if (isResizing) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'ns-resize';
      document.body.style.userSelect = 'none'; // Prevent text selection while dragging
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'default';
      document.body.style.userSelect = 'auto';
    };
  }, [isResizing]);

  useEffect(() => {
    const handleMouseMoveStyle = (e: MouseEvent) => {
      if (!isResizingStyle) return;
      if (styleRef.current) {
        const rect = styleRef.current.getBoundingClientRect();
        const newHeight = e.clientY - rect.top;
        if (newHeight > 60 && newHeight < 600) {
          setStyleHeight(newHeight);
        }
      }
    };

    const handleMouseUpStyle = () => {
      setIsResizingStyle(false);
      document.body.style.cursor = 'default';
      document.body.style.userSelect = 'auto';
      localStorage.setItem('acestep_style_height', String(styleHeight));
    };

    if (isResizingStyle) {
      window.addEventListener('mousemove', handleMouseMoveStyle);
      window.addEventListener('mouseup', handleMouseUpStyle);
      document.body.style.cursor = 'ns-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMoveStyle);
      window.removeEventListener('mouseup', handleMouseUpStyle);
      document.body.style.cursor = 'default';
      document.body.style.userSelect = 'auto';
    };
  }, [isResizingStyle]);

  const refreshModels = useCallback(async (isInitial = false): Promise<boolean> => {
    try {
      const modelsRes = await fetch('/api/generate/models');
      if (modelsRes.ok) {
        const data = await modelsRes.json();
        const models = data.models || [];
        if (models.length > 0) {
          if (!isMountedRef.current) return false;
          setFetchedModels(models);
          setBackendUnavailable(false);
          if (data.active_model) {
            setActiveBackendModel(data.active_model);
          }
          // Only sync to backend's active model on initial load
          // After that, respect user's selection
          if (isInitial) {
            const active = models.find((m: any) => m.is_active);
            if (active) {
              setSelectedModel(active.name);
              localStorage.setItem('ace-model', active.name);
            }
          }

          // Also fetch LM model status to sync the LM dropdown on initial load
          if (isInitial) {
            try {
              const statusRes = await fetch('/api/models/status');
              if (statusRes.ok) {
                const statusData = await statusRes.json();
                const loadedLm = statusData?.lm_model;
                if (loadedLm && isMountedRef.current) {
                  setLmModel(loadedLm);
                  localStorage.setItem('ace-lmModel', JSON.stringify(loadedLm));
                }
              }
            } catch {
              // Non-critical — just use persisted/default value
            }
          }

          return true;
        }
      } else if (modelsRes.status === 503) {
        if (isMountedRef.current) setBackendUnavailable(true);
      }
    } catch {
      if (isMountedRef.current) setBackendUnavailable(true);
    }
    return false;
  }, []);

  const refreshLoraStatus = useCallback(async () => {
    if (!token) return;
    try {
      const status = await generateApi.getLoraStatus(token);
      if (!isMountedRef.current) return;
      setLoraLoaded(Boolean(status?.lora_loaded));
      if (typeof status?.lora_scale === 'number' && Number.isFinite(status.lora_scale)) {
        setLoraScale(status.lora_scale);
      }
      // Sync advanced adapter slots
      if (status?.advanced?.slots && status.advanced.slots.length > 0) {
        setAdapterSlots(status.advanced.slots);
      }
    } catch {
      // ignore - backend may be starting
    }
  }, [token]);

  const handleSwitchModel = useCallback(async (targetModel: string) => {
    if (!token || isSwitching) return;
    setIsSwitching(true);

    // Save current adapter state before switch (backend will unload them)
    const prevSlots = [...adapterSlots];
    const hadAdapters = prevSlots.length > 0 && loraLoaded;

    try {
      const result = await generateApi.switchModel(targetModel, token);
      if (result.switched) {
        setActiveBackendModel(result.active_model);
        void refreshModels(false);

        // Re-apply adapters to the new model if any were loaded
        if (hadAdapters) {
          setIsLoraLoading(true);
          setAdapterLoadingMessage('Re-applying adapters to new model...');
          setAdapterSlots([]);
          setLoraLoaded(false);

          for (const slot of prevSlots) {
            try {
              setAdapterLoadingMessage(`Loading ${slot.name}...`);
              const savedScale = savedOverallScales[slot.name];
              const savedGroups = savedGroupScales[slot.name];
              await generateApi.loadLora({
                lora_path: slot.path,
                slot: slot.slot,
                ...(savedScale != null && savedScale !== 1.0 ? { scale: savedScale } : {}),
                ...(savedGroups != null ? { group_scales: savedGroups } : {}),
              }, token);
            } catch (err) {
              console.error(`Failed to re-apply adapter ${slot.name}:`, err);
            }
          }

          // Refresh status after all adapters loaded
          try {
            const status = await generateApi.getLoraStatus(token);
            if (status?.advanced?.slots && status.advanced.slots.length > 0) {
              setAdapterSlots(status.advanced.slots);
              setLoraLoaded(true);
              setAdapterTriggerWord((status as any).trigger_word || '');
            }
          } catch {
            // ignore status refresh failure
          }
          setIsLoraLoading(false);
          setAdapterLoadingMessage('');
        }
      }
    } catch (err: any) {
      console.error('Model switch failed:', err.message);
    } finally {
      setIsSwitching(false);
    }
  }, [token, isSwitching, refreshModels, adapterSlots, loraLoaded, savedOverallScales, savedGroupScales]);

  useEffect(() => {
    isMountedRef.current = true;
    let cancelled = false;
    const loadModelsAndLimits = async () => {
      // Backend can start slowly; retry silently until models are available.
      const attemptRefresh = async (attempt: number) => {
        if (cancelled) return;
        // Pass true for isInitial only on first attempt
        const ok = await refreshModels(attempt === 0);
        if (!ok) {
          const delayMs = Math.min(15000, 800 * Math.pow(1.6, attempt));
          modelsRetryTimeoutRef.current = window.setTimeout(() => { void attemptRefresh(attempt + 1); }, delayMs);
        }
      };
      void attemptRefresh(0);

      void refreshLoraStatus();

      // Fetch limits
      try {
        const response = await fetch('/api/generate/limits');
        if (!response.ok) return;
        const data = await response.json();
        if (typeof data.max_duration_with_lm === 'number') {
          setMaxDurationWithLm(data.max_duration_with_lm);
        }
        if (typeof data.max_duration_without_lm === 'number') {
          setMaxDurationWithoutLm(data.max_duration_without_lm);
        }
      } catch {
        // ignore limits fetch failures
      }
    };

    loadModelsAndLimits();
    return () => {
      // best-effort cancel retry loop
      cancelled = true;
      isMountedRef.current = false;
      if (modelsRetryTimeoutRef.current != null) {
        window.clearTimeout(modelsRetryTimeoutRef.current);
        modelsRetryTimeoutRef.current = null;
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    };
  }, []);

  useEffect(() => {
    void refreshLoraStatus();
  }, [refreshLoraStatus]);

  // Re-fetch models after generation completes to update active model status
  // Don't change selection, just refresh the is_active status
  const prevIsGeneratingRef = useRef(isGenerating);
  useEffect(() => {
    if (prevIsGeneratingRef.current && !isGenerating) {
      void refreshModels(false);
    }
    prevIsGeneratingRef.current = isGenerating;
  }, [isGenerating, refreshModels]);

  // Ctrl+Enter / Cmd+Enter keyboard shortcut to trigger Generate from anywhere in the UI
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (!isGenerating && isAuthenticated) {
          handleGenerate();
        }
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isGenerating, isAuthenticated]);

  const activeMaxDuration = thinking ? maxDurationWithLm : maxDurationWithoutLm;

  useEffect(() => {
    if (duration > activeMaxDuration) {
      setDuration(activeMaxDuration);
    }
  }, [duration, activeMaxDuration]);

  useEffect(() => {
    const getDragKind = (e: DragEvent): 'file' | 'audio' | null => {
      if (!e.dataTransfer) return null;
      const types = Array.from(e.dataTransfer.types);
      if (types.includes('Files')) return 'file';
      if (types.includes('application/x-ace-audio')) return 'audio';
      return null;
    };

    const handleDragEnter = (e: DragEvent) => {
      const kind = getDragKind(e);
      if (!kind) return;
      dragDepthRef.current += 1;
      setIsDraggingFile(true);
      setDragKind(kind);
      e.preventDefault();
    };

    const handleDragOver = (e: DragEvent) => {
      const kind = getDragKind(e);
      if (!kind) return;
      setDragKind(kind);
      e.preventDefault();
    };

    const handleDragLeave = (e: DragEvent) => {
      const kind = getDragKind(e);
      if (!kind) return;
      dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
      if (dragDepthRef.current === 0) {
        setIsDraggingFile(false);
        setDragKind(null);
      }
    };

    const handleDrop = (e: DragEvent) => {
      const kind = getDragKind(e);
      if (!kind) return;
      e.preventDefault();
      dragDepthRef.current = 0;
      setIsDraggingFile(false);
      setDragKind(null);
    };

    window.addEventListener('dragenter', handleDragEnter);
    window.addEventListener('dragover', handleDragOver);
    window.addEventListener('dragleave', handleDragLeave);
    window.addEventListener('drop', handleDrop);

    return () => {
      window.removeEventListener('dragenter', handleDragEnter);
      window.removeEventListener('dragover', handleDragOver);
      window.removeEventListener('dragleave', handleDragLeave);
      window.removeEventListener('drop', handleDrop);
    };
  }, []);

  const startResizing = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  const startResizingStyle = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizingStyle(true);
  };

  const uploadAudio = async (file: File, target: 'reference' | 'source') => {
    if (!token) {
      setUploadError(t('pleaseSignInToUploadAudio'));
      return;
    }
    setUploadError(null);
    const setUploading = target === 'reference' ? setIsUploadingReference : setIsUploadingSource;
    const setUrl = target === 'reference' ? setReferenceAudioUrl : setSourceAudioUrl;
    setUploading(true);
    try {
      const result = await generateApi.uploadAudio(file, token);
      setUrl(result.url);
      setShowAudioModal(false);
      setTempAudioUrl('');
    } catch (err) {
      const message = err instanceof Error ? err.message : t('uploadFailed');
      setUploadError(message);
    } finally {
      setUploading(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>, target: 'reference' | 'source') => {
    const file = e.target.files?.[0];
    if (file) {
      void uploadReferenceTrack(file, target);
    }
    e.target.value = '';
  };

  // Format handler - uses LLM to enhance style/lyrics and auto-fill parameters
  const handleFormat = async (target: 'style' | 'lyrics') => {
    if (!token || !style.trim()) return;
    if (target === 'style') {
      setIsFormattingStyle(true);
    } else {
      setIsFormattingLyrics(true);
    }
    try {
      const result = await generateApi.formatInput({
        caption: style,
        lyrics: lyrics,
        bpm: bpm > 0 ? bpm : undefined,
        duration: duration > 0 ? duration : undefined,
        keyScale: keyScale || undefined,
        timeSignature: timeSignature || undefined,
        temperature: lmTemperature,
        topK: lmTopK > 0 ? lmTopK : undefined,
        topP: lmTopP,
        lmModel: lmModel || 'acestep-5Hz-lm-0.6B',
        lmBackend: lmBackend || 'pt',
      }, token);

      if (result.caption || result.lyrics || result.bpm || result.duration) {
        // Update fields with LLM-generated values
        if (target === 'style' && result.caption) setStyle(result.caption);
        if (target === 'lyrics' && result.lyrics) setLyrics(result.lyrics);
        if (result.bpm && result.bpm > 0) setBpm(result.bpm);
        if (result.duration && result.duration > 0) setDuration(result.duration);
        if (result.key_scale) setKeyScale(result.key_scale);
        if (result.time_signature) {
          const ts = String(result.time_signature);
          setTimeSignature(ts.includes('/') ? ts : `${ts}/4`);
        }
        if (result.vocal_language) setVocalLanguage(result.vocal_language);
        if (target === 'style') setIsFormatCaption(true);
      } else {
        console.error(t('formatFailed') + ':', result.error || result.status_message);
        alert(result.error || result.status_message || t('formatFailedLLMNotInitialized'));
      }
    } catch (err) {
      console.error('Format error:', err);
      alert(t('formatFailedLLMNotAvailable'));
    } finally {
      if (target === 'style') {
        setIsFormattingStyle(false);
      } else {
        setIsFormattingLyrics(false);
      }
    }
  };

  const openAudioModal = (target: 'reference' | 'source', tab: 'uploads' | 'created' = 'uploads') => {
    setAudioModalTarget(target);
    setTempAudioUrl('');
    setLibraryTab(tab);
    setShowAudioModal(true);
    void fetchReferenceTracks();
  };

  const fetchReferenceTracks = useCallback(async () => {
    if (!token) return;
    setIsLoadingTracks(true);
    try {
      const response = await fetch('/api/reference-tracks', {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setReferenceTracks(data.tracks || []);
      }
    } catch (err) {
      console.error(t('failedToFetchReferenceTracks'), err);
    } finally {
      setIsLoadingTracks(false);
    }
  }, [token]);

  const uploadReferenceTrack = async (file: File, target?: 'reference' | 'source') => {
    if (!token) {
      setUploadError(t('pleaseSignInToUploadAudio'));
      return;
    }
    setUploadError(null);
    setIsUploadingReference(true);
    try {
      const formData = new FormData();
      formData.append('audio', file);

      const response = await fetch('/api/reference-tracks', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || t('uploadFailed'));
      }

      const data = await response.json();
      setReferenceTracks(prev => [data.track, ...prev]);

      // Also set as current reference/source
      const selectedTarget = target ?? audioModalTarget;
      applyAudioTargetUrl(selectedTarget, data.track.audio_url, data.track.filename);
      if (data.whisper_available && data.track?.id) {
        void transcribeReferenceTrack(data.track.id).then(() => undefined);
      } else {
        setShowAudioModal(false);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : t('uploadFailed');
      setUploadError(message);
    } finally {
      setIsUploadingReference(false);
    }
  };

  const transcribeReferenceTrack = async (trackId: string) => {
    if (!token) return;
    setIsTranscribingReference(true);
    const controller = new AbortController();
    transcribeAbortRef.current = controller;
    try {
      const response = await fetch(`/api/reference-tracks/${trackId}/transcribe`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(t('failedToTranscribe'));
      }
      const data = await response.json();
      if (data.lyrics) {
        setLyrics(prev => prev || data.lyrics);
      }
    } catch (err) {
      if (controller.signal.aborted) return;
      console.error(t('transcriptionFailed'), err);
    } finally {
      if (transcribeAbortRef.current === controller) {
        transcribeAbortRef.current = null;
      }
      setIsTranscribingReference(false);
    }
  };

  const cancelTranscription = () => {
    if (transcribeAbortRef.current) {
      transcribeAbortRef.current.abort();
      transcribeAbortRef.current = null;
    }
    setIsTranscribingReference(false);
  };

  const deleteReferenceTrack = async (trackId: string) => {
    if (!token) return;
    try {
      const response = await fetch(`/api/reference-tracks/${trackId}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      });
      if (response.ok) {
        setReferenceTracks(prev => prev.filter(t => t.id !== trackId));
        if (playingTrackId === trackId && playingTrackSource === 'uploads') {
          setPlayingTrackId(null);
          setPlayingTrackSource(null);
          if (modalAudioRef.current) {
            modalAudioRef.current.pause();
          }
        }
      }
    } catch (err) {
      console.error(t('failedToDeleteTrack'), err);
    }
  };

  const useReferenceTrack = (track: { audio_url: string; title?: string }) => {
    applyAudioTargetUrl(audioModalTarget, track.audio_url, track.title);
    setShowAudioModal(false);
    setPlayingTrackId(null);
    setPlayingTrackSource(null);
  };

  const toggleModalTrack = (track: { id: string; audio_url: string; source: 'uploads' | 'created' }) => {
    if (playingTrackId === track.id) {
      if (modalAudioRef.current) {
        modalAudioRef.current.pause();
      }
      setPlayingTrackId(null);
      setPlayingTrackSource(null);
    } else {
      setPlayingTrackId(track.id);
      setPlayingTrackSource(track.source);
      if (modalAudioRef.current) {
        modalAudioRef.current.src = track.audio_url;
        modalAudioRef.current.play().catch(() => undefined);
      }
    }
  };

  const applyAudioUrl = () => {
    if (!tempAudioUrl.trim()) return;
    applyAudioTargetUrl(audioModalTarget, tempAudioUrl.trim());
    setShowAudioModal(false);
    setTempAudioUrl('');
  };

  const applyAudioTargetUrl = (target: 'reference' | 'source', url: string, title?: string) => {
    const derivedTitle = title ? title.replace(/\.[^/.]+$/, '') : getAudioLabel(url);
    if (target === 'reference') {
      setReferenceAudioUrl(url);
      setReferenceAudioTitle(derivedTitle);
      setReferenceTime(0);
      setReferenceDuration(0);
    } else {
      setSourceAudioUrl(url);
      setSourceAudioTitle(derivedTitle);
      setSourceTime(0);
      setSourceDuration(0);
      if (taskType === 'text2music') {
        setTaskType('cover');
      }
      // Don't override task type when in extract mode
    }
  };

  const formatTime = (time: number) => {
    if (!Number.isFinite(time) || time <= 0) return '0:00';
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${String(seconds).padStart(2, '0')}`;
  };

  const toggleAudio = (target: 'reference' | 'source') => {
    const audio = target === 'reference' ? referenceAudioRef.current : sourceAudioRef.current;
    if (!audio) return;
    if (audio.paused) {
      audio.play().catch(() => undefined);
    } else {
      audio.pause();
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>, target: 'reference' | 'source') => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      void uploadReferenceTrack(file, target);
      return;
    }
    const payload = e.dataTransfer.getData('application/x-ace-audio');
    if (payload) {
      try {
        const data = JSON.parse(payload);
        if (data?.url) {
          applyAudioTargetUrl(target, data.url, data.title);
        }
      } catch {
        // ignore
      }
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleWorkspaceDrop = (e: React.DragEvent<HTMLDivElement>) => {
    if (e.dataTransfer.files?.length || e.dataTransfer.types.includes('application/x-ace-audio')) {
      handleDrop(e, 'source');
    }
  };

  const handleWorkspaceDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    if (e.dataTransfer.types.includes('Files') || e.dataTransfer.types.includes('application/x-ace-audio')) {
      e.preventDefault();
    }
  };

  const handleSteeringChange = useCallback(
    (enabled: boolean, loaded: string[], alphas: Record<string, number>) => {
      setSteeringEnabled(enabled);
      setSteeringLoaded(loaded);
      setSteeringAlphas(alphas);
    },
    []
  );

  // Clear detected BPM/key when source audio changes
  React.useEffect(() => {
    setDetectedBpm(null);
    setDetectedKey(null);
  }, [sourceAudioUrl]);

  // Analyze source audio with Essentia (BPM & key detection)
  const handleAnalyzeSource = async () => {
    if (!sourceAudioUrl || isAnalyzing) return;
    setIsAnalyzing(true);
    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audioUrl: sourceAudioUrl }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        console.error('[Analyze] Failed:', err.error || res.statusText);
        return;
      }
      const data = await res.json();
      if (data.bpm && data.bpm > 0) {
        setDetectedBpm(data.bpm);
        setBpm(data.bpm);
      }
      if (data.key) {
        const keyStr = `${data.key} ${data.scale || ''}`.trim();
        setDetectedKey(keyStr);
        setKeyScale(keyStr);
      }
      console.log(`[Analyze] Detected BPM: ${data.bpm}, Key: ${data.key} ${data.scale}`);
    } catch (err) {
      console.error('[Analyze] Error:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleGenerate = () => {
    const styleWithGender = (() => {
      if (!vocalGender) return style;
      const genderHint = vocalGender === 'male' ? t('maleVocals') : t('femaleVocals');
      const trimmed = style.trim();
      return trimmed ? `${trimmed}\n${genderHint}` : genderHint;
    })();

    // For extract mode: one job per selected track; for others: single iteration
    const tracksToExtract = taskType === 'extract' ? (extractTracks.length > 0 ? extractTracks : ['vocals']) : [null as string | null];

    // Bulk generation: loop bulkCount times per track
    for (const currentTrack of tracksToExtract) {
      for (let i = 0; i < bulkCount; i++) {
        // Seed handling: first job uses user's seed, rest get random seeds
        let jobSeed = -1;
        if (!randomSeed && i === 0 && currentTrack === tracksToExtract[0]) {
          jobSeed = seed;
        } else if (!randomSeed) {
          // Subsequent jobs get random seeds for variety
          jobSeed = Math.floor(Math.random() * 4294967295);
        }

        const isVocalTrack = currentTrack === 'vocals' || currentTrack === 'backing_vocals';

        onGenerate({
          customMode,
          songDescription: taskType === 'extract' ? undefined : (customMode ? undefined : songDescription),
          prompt: taskType === 'extract'
            ? (isVocalTrack ? lyrics : '')
            : lyrics,
          lyrics: taskType === 'extract'
            ? (isVocalTrack ? lyrics : '')
            : lyrics,
          style: taskType === 'extract' ? style : styleWithGender,
          title: taskType === 'extract'
            ? `${(currentTrack || 'extract').charAt(0).toUpperCase() + (currentTrack || 'extract').slice(1).replace('_', ' ')}${sourceAudioTitle ? ` - ${sourceAudioTitle}` : ''}`
            : (bulkCount > 1 ? `${title} (${i + 1})` : title),
          ditModel: selectedModel,
          instrumental: taskType === 'extract'
            ? !isVocalTrack
            : instrumental,
          vocalLanguage,
          bpm: taskType === 'extract' ? 0 : (() => {
            // When source audio was analyzed and tempo is scaled, send effective BPM
            // so the model's conditioning matches the transformed audio waveform
            if (detectedBpm && tempoScale !== 1.0) return Math.round(detectedBpm * tempoScale);
            return bpm;
          })(),
          keyScale: taskType === 'extract' ? '' : (() => {
            // When source audio was analyzed and pitch is shifted, send transposed key
            if (detectedKey && pitchShift !== 0) {
              const CHROMATIC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
              const parts = detectedKey.split(' ');
              const idx = CHROMATIC.indexOf(parts[0]);
              if (idx !== -1) {
                const newIdx = ((idx + pitchShift) % 12 + 12) % 12;
                return `${CHROMATIC[newIdx]}${parts.length > 1 ? ' ' + parts.slice(1).join(' ') : ''}`;
              }
            }
            return keyScale;
          })(),
          timeSignature: taskType === 'extract' ? '' : timeSignature,
          duration,
          inferenceSteps,
          guidanceScale,
          batchSize,
          randomSeed: randomSeed || i > 0 || currentTrack !== tracksToExtract[0],
          seed: jobSeed,
          thinking,
          audioFormat,
          inferMethod,
          scheduler,
          lmBackend,
          lmModel,
          shift,
          lmTemperature,
          lmCfgScale,
          lmTopK,
          lmTopP,
          lmRepetitionPenalty,
          lmNegativePrompt,
          steeringEnabled,
          steeringLoaded,
          steeringAlphas,
          referenceAudioUrl: (useReferenceAudio && referenceAudioUrl.trim()) || undefined,
          sourceAudioUrl: sourceAudioUrl.trim() || undefined,
          referenceAudioTitle: (useReferenceAudio && referenceAudioTitle.trim()) || undefined,
          sourceAudioTitle: sourceAudioTitle.trim() || undefined,
          audioCodes: audioCodes.trim() || undefined,
          repaintingStart,
          repaintingEnd,
          instruction: taskType === 'extract' && currentTrack
            ? `Extract the ${currentTrack.toUpperCase()} track from the audio:`
            : instruction,
          audioCoverStrength,
          coverNoiseStrength,
          tempoScale,
          pitchShift,
          autoMaster,
          masteringParams: autoMaster && masteringParams ? masteringParams : undefined,
          enableNormalization,
          normalizationDb,
          latentShift,
          latentRescale,
          taskType,
          useAdg: guidanceMode === 'adg',
          guidanceMode,
          usePag: guidanceMode === 'pag',
          pagStart: guidanceMode === 'pag' ? pagStart : undefined,
          pagEnd: guidanceMode === 'pag' ? pagEnd : undefined,
          pagScale: guidanceMode === 'pag' ? pagScale : undefined,
          cfgIntervalStart,
          cfgIntervalEnd,
          customTimesteps: customTimesteps.trim() || undefined,
          useCotMetas,
          useCotCaption,
          useCotLanguage,
          autogen,
          constrainedDecodingDebug,
          allowLmBatch,
          getScores,
          getLrc,
          scoreScale,
          lmBatchChunkSize,
          loraPath: loraPath.trim() || undefined,
          loraScale,
          advancedAdapters,
          adapterSlots: advancedAdapters ? adapterSlots : undefined,
          trackName: taskType === 'extract' ? (currentTrack || undefined) : (trackName.trim() || undefined),
          completeTrackClasses: (() => {
            const parsed = completeTrackClasses
              .split(',')
              .map((item) => item.trim())
              .filter(Boolean);
            return parsed.length ? parsed : undefined;
          })(),
          isFormatCaption,
          loraLoaded,
        });
      }
    }
  };

  const handleExportJson = () => {
    const data = {
      model: selectedModel,
      customMode,
      songDescription: customMode ? undefined : songDescription,
      prompt: lyrics,
      style,
      title,
      instrumental,
      vocalLanguage,
      vocalGender,
      bpm,
      keyScale,
      timeSignature,
      duration,
      inferenceSteps,
      guidanceScale,
      batchSize,
      randomSeed,
      seed,
      thinking,
      audioFormat,
      inferMethod,
      scheduler,
      shift,
      lmBackend,
      lmModel,
      lmTemperature,
      lmCfgScale,
      lmTopK,
      lmTopP,
      lmNegativePrompt,
      referenceAudioUrl,
      sourceAudioUrl,
      referenceAudioTitle,
      sourceAudioTitle,
      audioCodes,
      repaintingStart,
      repaintingEnd,
      instruction,
      audioCoverStrength,
      taskType,
      guidanceMode,
      usePag,
      pagStart,
      pagEnd,
      pagScale,
      cfgIntervalStart,
      cfgIntervalEnd,
      customTimesteps,
      useCotMetas,
      useCotCaption,
      useCotLanguage,
      autogen,
      constrainedDecodingDebug,
      allowLmBatch,
      getScores,
      getLrc,
      scoreScale,
      lmBatchChunkSize,
      trackName,
      extractTracks,
      completeTrackClasses,
      isFormatCaption,
      loraPath,
      loraScale,
      loraLoaded,
      advancedAdapters,
      adapterSlots,
      adapterFolder,
      steeringEnabled,
      steeringLoaded,
      steeringAlphas
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `acestep_params_${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImportJson = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const json = JSON.parse(event.target?.result as string);
        if (json.model !== undefined) setSelectedModel(json.model);
        if (json.customMode !== undefined) setCustomMode(json.customMode);
        if (json.songDescription !== undefined) setSongDescription(json.songDescription);
        if (json.prompt !== undefined) setLyrics(json.prompt);
        if (json.style !== undefined) setStyle(json.style);
        if (json.title !== undefined) setTitle(json.title);
        if (json.instrumental !== undefined) setInstrumental(json.instrumental);
        if (json.vocalLanguage !== undefined) setVocalLanguage(json.vocalLanguage);
        if (json.vocalGender !== undefined) setVocalGender(json.vocalGender);
        if (json.bpm !== undefined) setBpm(json.bpm);
        if (json.keyScale !== undefined) setKeyScale(json.keyScale);
        if (json.timeSignature !== undefined) setTimeSignature(json.timeSignature);
        if (json.duration !== undefined) setDuration(json.duration);
        if (json.inferenceSteps !== undefined) setInferenceSteps(json.inferenceSteps);
        if (json.guidanceScale !== undefined) setGuidanceScale(json.guidanceScale);
        if (json.batchSize !== undefined) setBatchSize(json.batchSize);
        if (json.randomSeed !== undefined) setRandomSeed(json.randomSeed);
        if (json.seed !== undefined) setSeed(json.seed);
        if (json.thinking !== undefined) setThinking(json.thinking);
        if (json.audioFormat !== undefined) setAudioFormat(json.audioFormat);
        if (json.inferMethod !== undefined) setInferMethod(json.inferMethod);
        if (json.scheduler !== undefined) setScheduler(json.scheduler);
        if (json.shift !== undefined) setShift(json.shift);
        if (json.lmBackend !== undefined) setLmBackend(json.lmBackend);
        if (json.lmModel !== undefined) setLmModel(json.lmModel);
        if (json.lmTemperature !== undefined) setLmTemperature(json.lmTemperature);
        if (json.lmCfgScale !== undefined) setLmCfgScale(json.lmCfgScale);
        if (json.lmTopK !== undefined) setLmTopK(json.lmTopK);
        if (json.lmTopP !== undefined) setLmTopP(json.lmTopP);
        if (json.lmNegativePrompt !== undefined) setLmNegativePrompt(json.lmNegativePrompt);
        if (json.referenceAudioUrl !== undefined) setReferenceAudioUrl(json.referenceAudioUrl);
        if (json.sourceAudioUrl !== undefined) setSourceAudioUrl(json.sourceAudioUrl);
        if (json.referenceAudioTitle !== undefined) setReferenceAudioTitle(json.referenceAudioTitle);
        if (json.sourceAudioTitle !== undefined) setSourceAudioTitle(json.sourceAudioTitle);
        if (json.audioCodes !== undefined) setAudioCodes(json.audioCodes);
        if (json.repaintingStart !== undefined) setRepaintingStart(json.repaintingStart);
        if (json.repaintingEnd !== undefined) setRepaintingEnd(json.repaintingEnd);
        if (json.instruction !== undefined) setInstruction(json.instruction);
        if (json.audioCoverStrength !== undefined) setAudioCoverStrength(json.audioCoverStrength);
        if (json.coverNoiseStrength !== undefined) setCoverNoiseStrength(json.coverNoiseStrength);
        if (json.enableNormalization !== undefined) setEnableNormalization(json.enableNormalization);
        if (json.normalizationDb !== undefined) setNormalizationDb(json.normalizationDb);
        if (json.latentShift !== undefined) setLatentShift(json.latentShift);
        if (json.latentRescale !== undefined) setLatentRescale(json.latentRescale);
        if (json.taskType !== undefined) setTaskType(json.taskType);
        if (json.guidanceMode !== undefined) setGuidanceMode(json.guidanceMode);
        if (json.usePag !== undefined) setUsePag(json.usePag);
        if (json.pagStart !== undefined) setPagStart(json.pagStart);
        if (json.pagEnd !== undefined) setPagEnd(json.pagEnd);
        if (json.pagScale !== undefined) setPagScale(json.pagScale);
        if (json.cfgIntervalStart !== undefined) setCfgIntervalStart(json.cfgIntervalStart);
        if (json.cfgIntervalEnd !== undefined) setCfgIntervalEnd(json.cfgIntervalEnd);
        if (json.customTimesteps !== undefined) setCustomTimesteps(json.customTimesteps);
        if (json.useCotMetas !== undefined) setUseCotMetas(json.useCotMetas);
        if (json.useCotCaption !== undefined) setUseCotCaption(json.useCotCaption);
        if (json.useCotLanguage !== undefined) setUseCotLanguage(json.useCotLanguage);
        if (json.autogen !== undefined) setAutogen(json.autogen);
        if (json.constrainedDecodingDebug !== undefined) setConstrainedDecodingDebug(json.constrainedDecodingDebug);
        if (json.allowLmBatch !== undefined) setAllowLmBatch(json.allowLmBatch);
        if (json.getScores !== undefined) setGetScores(json.getScores);
        if (json.getLrc !== undefined) setGetLrc(json.getLrc);
        if (json.scoreScale !== undefined) setScoreScale(json.scoreScale);
        if (json.lmBatchChunkSize !== undefined) setLmBatchChunkSize(json.lmBatchChunkSize);
        if (json.trackName !== undefined) setTrackName(json.trackName);
        if (json.extractTracks !== undefined) setExtractTracks(json.extractTracks);
        // Backwards compat: old single-value format
        if (json.extractTrack !== undefined && !json.extractTracks) setExtractTracks([json.extractTrack]);
        if (json.completeTrackClasses !== undefined) setCompleteTrackClasses(json.completeTrackClasses);
        if (json.isFormatCaption !== undefined) setIsFormatCaption(json.isFormatCaption);
        if (json.loraPath !== undefined) setLoraPath(json.loraPath);
        if (json.loraScale !== undefined) setLoraScale(json.loraScale);
        if (json.loraLoaded !== undefined) setLoraLoaded(json.loraLoaded);
        if (json.advancedAdapters !== undefined) setAdvancedAdapters(json.advancedAdapters);
        if (json.adapterSlots !== undefined) setAdapterSlots(json.adapterSlots);
        if (json.adapterFolder !== undefined) setAdapterFolder(json.adapterFolder);
        if (json.steeringEnabled !== undefined) setSteeringEnabled(json.steeringEnabled);
        if (json.steeringLoaded !== undefined) setSteeringLoaded(json.steeringLoaded);
        if (json.steeringAlphas !== undefined) setSteeringAlphas(json.steeringAlphas);

        // Auto-load steering and lora if they were enabled
        if (json.steeringEnabled && json.steeringLoaded?.length > 0) {
          setShowSteeringPanel(true);
        }
        if (json.loraLoaded && (json.customMode || json.advancedAdapters || json.loraPath)) {
          setShowLoraPanel(true);
        }
      } catch (e) {
        console.error("Failed to parse imported JSON", e);
        alert(t('errorImportingJson'));
      }
    };
    reader.readAsText(file);
    // Reset input so importing the same file again triggers onChange
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div
      className="relative flex flex-col h-full bg-zinc-50 dark:bg-suno-panel w-full overflow-y-auto custom-scrollbar transition-colors duration-300"
      onDrop={handleWorkspaceDrop}
      onDragOver={handleWorkspaceDragOver}
    >
      {isDraggingFile && (
        <div className="absolute inset-0 z-[90] pointer-events-none">
          <div className="absolute inset-0 bg-white/70 dark:bg-black/50 backdrop-blur-sm" />
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="flex flex-col items-center gap-2 rounded-2xl border border-zinc-200 dark:border-white/10 bg-white/90 dark:bg-zinc-900/90 px-6 py-5 shadow-xl">
              {dragKind !== 'audio' && (
                <div className="w-12 h-12 rounded-full bg-gradient-to-br from-pink-500 to-purple-600 text-white flex items-center justify-center shadow-lg">
                  <Upload size={22} />
                </div>
              )}
              <div className="text-sm font-semibold text-zinc-900 dark:text-white">
                {dragKind === 'audio' ? t('dropToUseAudio') : t('dropToUpload')}
              </div>
              <div className="text-[11px] text-zinc-500 dark:text-zinc-400">
                {dragKind === 'audio'
                  ? t('usingAsCover')
                  : t('uploadingAsCover')}
              </div>
            </div>
          </div>
        </div>
      )}
      <div className="p-4 pt-14 md:pt-4 pb-24 lg:pb-32 space-y-5">
        <input
          ref={referenceInputRef}
          type="file"
          accept="audio/*"
          onChange={(e) => handleFileSelect(e, 'reference')}
          className="hidden"
        />
        <input
          ref={sourceInputRef}
          type="file"
          accept="audio/*"
          onChange={(e) => handleFileSelect(e, 'source')}
          className="hidden"
        />
        <audio
          ref={referenceAudioRef}
          src={referenceAudioUrl || undefined}
          onPlay={() => setReferencePlaying(true)}
          onPause={() => setReferencePlaying(false)}
          onEnded={() => setReferencePlaying(false)}
          onTimeUpdate={(e) => setReferenceTime(e.currentTarget.currentTime)}
          onLoadedMetadata={(e) => setReferenceDuration(e.currentTarget.duration || 0)}
        />
        <audio
          ref={sourceAudioRef}
          src={sourceAudioUrl || undefined}
          onPlay={() => setSourcePlaying(true)}
          onPause={() => setSourcePlaying(false)}
          onEnded={() => setSourcePlaying(false)}
          onTimeUpdate={(e) => setSourceTime(e.currentTarget.currentTime)}
          onLoadedMetadata={(e) => setSourceDuration(e.currentTarget.duration || 0)}
        />

        {/* Header - Mode Toggle & Model Selection */}
        <CreatePanelHeader
          customMode={customMode}
          setCustomMode={setCustomMode}
          modelMenuRef={modelMenuRef}
          showModelMenu={showModelMenu}
          setShowModelMenu={setShowModelMenu}
          availableModels={availableModels}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          backendUnavailable={backendUnavailable}
          fetchedModels={fetchedModels}
          setInferenceSteps={setInferenceSteps}
          setUseAdg={setUseAdg}
          getModelDisplayName={getModelDisplayName}
          isTurboModel={isTurboModel}
          activeBackendModel={activeBackendModel}
          isSwitching={isSwitching}
          isGenerating={isGenerating}
          handleSwitchModel={handleSwitchModel}
        />

        {/* TASK TYPE SELECTOR */}
        <TaskTypeSelector
          taskType={taskType}
          setTaskType={setTaskType}

          useReferenceAudio={useReferenceAudio}
          selectedModel={selectedModel}
        />

        {/* JSON Import/Export Actions */}
        <div className="flex items-center gap-2 px-1">
          <input
            type="file"
            accept=".json"
            className="hidden"
            ref={fileInputRef}
            onChange={handleImportJson}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex-1 flex items-center justify-center gap-2 py-2 px-3 bg-white dark:bg-white/5 border border-zinc-200 dark:border-white/10 hover:border-indigo-300 dark:hover:border-indigo-500/50 rounded-xl text-xs font-semibold text-zinc-700 dark:text-zinc-300 hover:text-indigo-600 dark:hover:text-indigo-400 transition-all shadow-sm group"
            title="Import Settings from JSON"
          >
            <Download size={14} className="group-hover:-translate-y-0.5 transition-transform" />
            {t('Import')}
          </button>
          <button
            onClick={handleExportJson}
            className="flex-1 flex items-center justify-center gap-2 py-2 px-3 bg-white dark:bg-white/5 border border-zinc-200 dark:border-white/10 hover:border-emerald-300 dark:hover:border-emerald-500/50 rounded-xl text-xs font-semibold text-zinc-700 dark:text-zinc-300 hover:text-emerald-600 dark:hover:text-emerald-400 transition-all shadow-sm group"
            title="Export Settings as JSON"
          >
            <Upload size={14} className="group-hover:-translate-y-0.5 transition-transform" />
            {t('Export')}
          </button>
        </div>

        {/* SIMPLE MODE — hidden in extract mode */}
        {!customMode && taskType !== 'extract' && (
          <SimpleModeSettings
            songDescription={songDescription}
            setSongDescription={setSongDescription}
            vocalLanguage={vocalLanguage}
            setVocalLanguage={setVocalLanguage}
            vocalGender={vocalGender}
            setVocalGender={setVocalGender}
            duration={duration}
            setDuration={setDuration}
            bpm={bpm}
            setBpm={setBpm}
            keyScale={keyScale}
            setKeyScale={setKeyScale}
            timeSignature={timeSignature}
            setTimeSignature={setTimeSignature}
            batchSize={batchSize}
            setBatchSize={setBatchSize}
          />
        )}

        {/* EXTRACT TRACK SELECTOR — only in extract mode */}
        {taskType === 'extract' && (
          <>
            <ExtractTrackSelector
              extractTracks={extractTracks}
              setExtractTracks={setExtractTracks}
              isTurboModel={isTurboModel(selectedModel)}
            />

            {/* Extract Quality Presets */}
            <div className="bg-white dark:bg-suno-card rounded-xl border border-zinc-200 dark:border-white/5 overflow-hidden">
              <div className="px-3 py-2.5 space-y-2">
                <span className="text-xs font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wide">
                  {t('extractQuality')}
                </span>
                <div className="flex gap-1.5">
                  {([
                    { key: 'extractLow', steps: 20, method: 'euler' as const, icon: '⚡' },
                    { key: 'extractMedium', steps: 50, method: 'heun' as const, icon: '⚖️' },
                    { key: 'extractHigh', steps: 200, method: 'rk4' as const, icon: '💎' },
                  ] as const).map(({ key, steps, method, icon }) => {
                    const isActive = inferenceSteps === steps && inferMethod === method && guidanceMode === 'dynamic_cfg';
                    return (
                      <button
                        key={key}
                        type="button"
                        onClick={() => {
                          setInferenceSteps(steps);
                          setInferMethod(method);
                          setGuidanceMode('dynamic_cfg');
                        }}
                        className={`flex-1 px-2 py-1.5 rounded-lg text-[10px] font-semibold border transition-colors ${isActive
                          ? 'bg-pink-600 text-white border-pink-500'
                          : 'bg-zinc-100 dark:bg-black/30 border-zinc-200 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-white/10'
                          }`}
                      >
                        {icon} {t(key)}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Optional style hint for extract */}
            <div>
              <div className="w-full flex items-center justify-between py-2 text-xs font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wide">
                <span>{t('extractStyleHint')}</span>
              </div>
              <div className="bg-zinc-50 dark:bg-black/20 rounded-lg border border-zinc-200 dark:border-white/10 overflow-hidden relative flex flex-col transition-colors focus-within:border-pink-500 dark:focus-within:border-pink-500">
                <textarea
                  value={style}
                  onChange={(e) => setStyle(e.target.value)}
                  placeholder={t('extractStyleHintPlaceholder')}
                  className="flex-1 bg-transparent p-3 text-sm text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none resize-none overflow-y-auto"
                  style={{ minHeight: '60px', maxHeight: '120px' }}
                />
              </div>
            </div>

            {/* Optional lyrics guidance for vocal extraction */}
            {(extractTracks.includes('vocals') || extractTracks.includes('backing_vocals')) && (
              <div>
                <div className="w-full flex items-center justify-between py-2 text-xs font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wide">
                  <span>{t('lyricsGuidanceOptional')}</span>
                </div>
                <div className="bg-zinc-50 dark:bg-black/20 rounded-lg border border-zinc-200 dark:border-white/10 overflow-hidden relative flex flex-col transition-colors focus-within:border-pink-500 dark:focus-within:border-pink-500">
                  <textarea
                    value={lyrics}
                    onChange={(e) => setLyrics(e.target.value)}
                    placeholder={t('lyricsGuidancePlaceholder')}
                    className="flex-1 bg-transparent p-3 text-sm text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none resize-none overflow-y-auto"
                    style={{ minHeight: '100px', maxHeight: '200px' }}
                  />
                </div>
              </div>
            )}
            <AudioSelectionSection
              useReferenceAudio={false}
              setUseReferenceAudio={setUseReferenceAudio}
              taskType={taskType}

              referenceAudioUrl={referenceAudioUrl}
              referenceAudioTitle={referenceAudioTitle}
              referencePlaying={referencePlaying}
              toggleAudio={toggleAudio}
              referenceDuration={referenceDuration}
              referenceTime={referenceTime}
              referenceAudioRef={referenceAudioRef}
              setReferenceAudioUrl={setReferenceAudioUrl}
              setReferenceAudioTitle={setReferenceAudioTitle}
              setReferencePlaying={setReferencePlaying}
              setReferenceTime={setReferenceTime}
              setReferenceDuration={setReferenceDuration}
              sourceAudioUrl={sourceAudioUrl}
              sourceAudioTitle={sourceAudioTitle}
              sourcePlaying={sourcePlaying}
              sourceDuration={sourceDuration}
              sourceTime={sourceTime}
              sourceAudioRef={sourceAudioRef}
              setSourceAudioUrl={setSourceAudioUrl}
              setSourceAudioTitle={setSourceAudioTitle}
              setSourcePlaying={setSourcePlaying}
              setSourceTime={setSourceTime}
              setSourceDuration={setSourceDuration}
              openAudioModal={openAudioModal}
              referenceInputRef={referenceInputRef}
              sourceInputRef={sourceInputRef}
              handleDrop={handleDrop}
              handleDragOver={handleDragOver}
              formatTime={formatTime}
              getAudioLabel={getAudioLabel}
              onAnalyzeSource={handleAnalyzeSource}
              isAnalyzing={isAnalyzing}
            />
          </>
        )}

        {/* CUSTOM MODE */}
        {customMode && taskType !== 'extract' && (
          <div className="space-y-5">
            {/* Lyrics Library — folder scanner for imported lyrics */}
            <LyricsLibrary
              setStyle={setStyle}
              setLyrics={setLyrics}
              setBpm={setBpm}
              setKeyScale={setKeyScale}
              setTitle={setTitle}
              setDuration={setDuration}
            />

            {/* Audio Section - Conditionally rendered */}

            <AudioSelectionSection
              useReferenceAudio={useReferenceAudio}
              setUseReferenceAudio={setUseReferenceAudio}
              taskType={taskType}

              referenceAudioUrl={referenceAudioUrl}
              referenceAudioTitle={referenceAudioTitle}
              referencePlaying={referencePlaying}
              toggleAudio={toggleAudio}
              referenceDuration={referenceDuration}
              referenceTime={referenceTime}
              referenceAudioRef={referenceAudioRef}
              setReferenceAudioUrl={setReferenceAudioUrl}
              setReferenceAudioTitle={setReferenceAudioTitle}
              setReferencePlaying={setReferencePlaying}
              setReferenceTime={setReferenceTime}
              setReferenceDuration={setReferenceDuration}
              sourceAudioUrl={sourceAudioUrl}
              sourceAudioTitle={sourceAudioTitle}
              sourcePlaying={sourcePlaying}
              sourceDuration={sourceDuration}
              sourceTime={sourceTime}
              sourceAudioRef={sourceAudioRef}
              setSourceAudioUrl={setSourceAudioUrl}
              setSourceAudioTitle={setSourceAudioTitle}
              setSourcePlaying={setSourcePlaying}
              setSourceTime={setSourceTime}
              setSourceDuration={setSourceDuration}
              openAudioModal={openAudioModal}
              referenceInputRef={referenceInputRef}
              sourceInputRef={sourceInputRef}
              handleDrop={handleDrop}
              handleDragOver={handleDragOver}
              formatTime={formatTime}
              getAudioLabel={getAudioLabel}
              onAnalyzeSource={handleAnalyzeSource}
              isAnalyzing={isAnalyzing}
            />
          </div>
        )}

        {/* TRACK DETAILS ACCORDION (Custom mode only, hidden in extract mode) */}
        {customMode && taskType !== 'extract' && (
          <TrackDetailsAccordion
            showTrackDetails={showTrackDetails}
            setShowTrackDetails={setShowTrackDetails}
            instrumental={instrumental}
            setInstrumental={setInstrumental}
            vocalLanguage={vocalLanguage}
            setVocalLanguage={setVocalLanguage}
            vocalGender={vocalGender}
            setVocalGender={setVocalGender}
            title={title}
            setTitle={setTitle}
            showLyricsSub={showLyricsSub}
            setShowLyricsSub={setShowLyricsSub}
            lyrics={lyrics}
            setLyrics={setLyrics}
            lyricsRef={lyricsRef}
            lyricsHeight={lyricsHeight}
            startResizing={startResizing}
            isFormattingLyrics={isFormattingLyrics}
            showStyleSub={showStyleSub}
            setShowStyleSub={setShowStyleSub}
            style={style}
            setStyle={setStyle}
            refreshMusicTags={refreshMusicTags}
            isFormattingStyle={isFormattingStyle}
            handleFormat={handleFormat}
            styleRef={styleRef}
            styleHeight={styleHeight}
            startResizingStyle={startResizingStyle}
            genreDropdownRef={genreDropdownRef}
            showGenreDropdown={showGenreDropdown}
            setShowGenreDropdown={setShowGenreDropdown}
            selectedMainGenre={selectedMainGenre}
            setSelectedMainGenre={setSelectedMainGenre}
            selectedSubGenre={selectedSubGenre}
            setSelectedSubGenre={setSelectedSubGenre}
            getSubGenreCount={getSubGenreCount}
            genreSearch={genreSearch}
            setGenreSearch={setGenreSearch}
            filteredCombinedGenres={filteredCombinedGenres}
            subGenreDropdownRef={subGenreDropdownRef}
            showSubGenreDropdown={showSubGenreDropdown}
            setShowSubGenreDropdown={setShowSubGenreDropdown}
            filteredSubGenres={filteredSubGenres}
            musicTags={musicTags}
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
            triggerWord={adapterTriggerWord}
          />
        )}

        {/* COMMON SETTINGS (Simple mode only, hidden in extract mode) */}
        <div className="space-y-4">
          {/* Instrumental Toggle (Simple Mode) */}
          {!customMode && taskType !== 'extract' && (
            <div className="flex items-center justify-between px-1 py-2">
              <div className="flex items-center gap-2">
                <Music2 size={14} className="text-zinc-500" />
                <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">{t('instrumental')}</span>
              </div>
              <button
                onClick={() => setInstrumental(!instrumental)}
                className={`w-11 h-6 rounded-full flex items-center transition-colors duration-200 px-1 border border-zinc-200 dark:border-white/5 ${instrumental ? 'bg-pink-600' : 'bg-zinc-300 dark:bg-black/40'}`}
              >
                <div className={`w-4 h-4 rounded-full bg-white transform transition-transform duration-200 shadow-sm ${instrumental ? 'translate-x-5' : 'translate-x-0'}`} />
              </button>
            </div>
          )}
        </div>

        {/* COVER / REPAINT SETTINGS (conditional on task type) */}
        <CoverRepaintSettings
          taskType={taskType}
          audioCoverStrength={audioCoverStrength}
          setAudioCoverStrength={setAudioCoverStrength}
          coverNoiseStrength={coverNoiseStrength}
          setCoverNoiseStrength={setCoverNoiseStrength}
          tempoScale={tempoScale}
          setTempoScale={setTempoScale}
          pitchShift={pitchShift}
          setPitchShift={setPitchShift}
          detectedBpm={detectedBpm}
          detectedKey={detectedKey}
          autoMaster={autoMaster}
          setAutoMaster={setAutoMaster}
          enableNormalization={enableNormalization}
          setEnableNormalization={setEnableNormalization}
          normalizationDb={normalizationDb}
          setNormalizationDb={setNormalizationDb}
          latentShift={latentShift}
          setLatentShift={setLatentShift}
          latentRescale={latentRescale}
          setLatentRescale={setLatentRescale}
          repaintingStart={repaintingStart}
          setRepaintingStart={setRepaintingStart}
          repaintingEnd={repaintingEnd}
          setRepaintingEnd={setRepaintingEnd}
          showCoverSettings={showCoverSettings}
          setShowCoverSettings={setShowCoverSettings}
          showOutputProcessing={showOutputProcessing}
          setShowOutputProcessing={setShowOutputProcessing}
          onOpenMasteringConsole={() => setShowMasteringConsole(true)}
        />

        {/* Mastering Console Modal */}
        <MasteringConsoleModal
          isOpen={showMasteringConsole}
          onClose={() => setShowMasteringConsole(false)}
          onParamsChange={setMasteringParams}
          currentParams={masteringParams}
        />

        {/* GENERATION SETTINGS */}
        <GenerationSettingsAccordion
          isOpen={showGenerationSettings}
          onToggle={() => setShowGenerationSettings(!showGenerationSettings)}
          isTurbo={isTurboModel(selectedModel)}
          batchSize={batchSize}
          onBatchSizeChange={setBatchSize}
          bulkCount={bulkCount}
          onBulkCountChange={setBulkCount}
          seed={seed}
          onSeedChange={setSeed}
          randomSeed={randomSeed}
          onRandomSeedToggle={() => setRandomSeed(!randomSeed)}
          shift={shift}
          onShiftChange={setShift}
          inferenceSteps={inferenceSteps}
          onInferenceStepsChange={setInferenceSteps}
          inferMethod={inferMethod}
          onInferMethodChange={setInferMethod}
          scheduler={scheduler}
          onSchedulerChange={setScheduler}
          audioFormat={audioFormat}
          onAudioFormatChange={setAudioFormat}
          guidanceScale={guidanceScale}
          onGuidanceScaleChange={setGuidanceScale}
          guidanceMode={guidanceMode}
          onGuidanceModeChange={(mode) => { setGuidanceMode(mode); setUseAdg(mode === 'adg'); setUsePag(mode === 'pag'); }}
          pagStart={pagStart}
          onPagStartChange={setPagStart}
          pagEnd={pagEnd}
          onPagEndChange={setPagEnd}
          pagScale={pagScale}
          onPagScaleChange={setPagScale}
          cfgIntervalStart={cfgIntervalStart}
          onCfgIntervalStartChange={setCfgIntervalStart}
          cfgIntervalEnd={cfgIntervalEnd}
          onCfgIntervalEndChange={setCfgIntervalEnd}
          thinking={thinking}
          onThinkingToggle={() => setThinking(!thinking)}
          loraLoaded={loraLoaded}
          lmBackend={lmBackend}
          onLmBackendChange={setLmBackend}
          lmModel={lmModel}
          onLmModelChange={setLmModel}
          lmTemperature={lmTemperature}
          onLmTemperatureChange={setLmTemperature}
          lmCfgScale={lmCfgScale}
          onLmCfgScaleChange={setLmCfgScale}
          lmTopK={lmTopK}
          onLmTopKChange={setLmTopK}
          lmTopP={lmTopP}
          onLmTopPChange={setLmTopP}
          lmRepetitionPenalty={lmRepetitionPenalty}
          onLmRepetitionPenaltyChange={setLmRepetitionPenalty}
          lmNegativePrompt={lmNegativePrompt}
          onLmNegativePromptChange={setLmNegativePrompt}
          allowLmBatch={allowLmBatch}
          onAllowLmBatchToggle={() => setAllowLmBatch(!allowLmBatch)}
          useCotMetas={useCotMetas}
          onUseCotMetasToggle={() => setUseCotMetas(!useCotMetas)}
          useCotCaption={useCotCaption}
          onUseCotCaptionToggle={() => setUseCotCaption(!useCotCaption)}
          useCotLanguage={useCotLanguage}
          onUseCotLanguageToggle={() => setUseCotLanguage(!useCotLanguage)}
          lmBatchChunkSize={lmBatchChunkSize}
          onLmBatchChunkSizeChange={setLmBatchChunkSize}
          constrainedDecodingDebug={constrainedDecodingDebug}
          onConstrainedDecodingDebugToggle={() => setConstrainedDecodingDebug(!constrainedDecodingDebug)}
          isFormatCaption={isFormatCaption}
          onIsFormatCaptionToggle={() => setIsFormatCaption(!isFormatCaption)}

          uploadError={uploadError}
          audioCodes={audioCodes}
          onAudioCodesChange={setAudioCodes}
          instruction={instruction}
          onInstructionChange={setInstruction}
          customTimesteps={customTimesteps}
          onCustomTimestepsChange={setCustomTimesteps}
          trackName={trackName}
          onTrackNameChange={setTrackName}
          completeTrackClasses={completeTrackClasses}
          onCompleteTrackClassesChange={setCompleteTrackClasses}
          autogen={autogen}
          onToggleAutogen={() => setAutogen(!autogen)}
          getLrc={getLrc}
          onToggleGetLrc={() => setGetLrc(!getLrc)}
        />


        {/* ADAPTERS */}
        <AdaptersAccordion
          customMode={customMode}
          isOpen={showLoraPanel}
          onToggle={() => setShowLoraPanel(!showLoraPanel)}
          advancedAdapters={advancedAdapters}
          onAdvancedAdaptersChange={setAdvancedAdapters}
          loraPath={loraPath}
          onLoraPathChange={setLoraPath}
          loraLoaded={loraLoaded}
          isLoraLoading={isLoraLoading}
          onLoraToggle={handleLoraToggle}
          loraError={loraError}
          loraScale={loraScale}
          onLoraScaleChange={handleLoraScaleChange}
          adapterFolder={adapterFolder}
          onAdapterFolderChange={setAdapterFolder}
          onScanFolder={handleScanFolder}
          adapterFiles={adapterFiles}
          adapterSlots={adapterSlots}
          loadingAdapterPath={loadingAdapterPath}
          adapterLoadingMessage={adapterLoadingMessage}
          expandedSlots={expandedSlots}
          setExpandedSlots={setExpandedSlots}
          onLoadSlot={handleLoadSlot}
          onUnloadSlot={handleUnloadSlot}
          onSlotScaleChange={handleSlotScaleChange}
          onSlotGroupScaleChange={handleSlotGroupScaleChange}
          onSlotLayerScaleChange={handleSlotLayerScaleChange}
          temporalScheduleActive={temporalScheduleActive}
          onTemporalSchedulePreset={handleTemporalSchedulePreset}
        />

        {/* LM ADAPTER (visible only when CoT/Thinking is enabled) */}
        {thinking && (
          <div className="accordion-section lm-lora-section">
            <button
              className="accordion-header"
              onClick={() => {}}
              style={{ cursor: 'default' }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Brain size={16} />
                <span>LM Adapter</span>
                {lmLoraLoaded && (
                  <span style={{
                    fontSize: '10px', padding: '1px 6px', borderRadius: '4px',
                    background: 'var(--accent-primary)', color: 'var(--bg-primary)',
                    fontWeight: 600,
                  }}>MERGED</span>
                )}
              </div>
            </button>
            <div className="accordion-content" style={{ padding: '8px 12px' }}>
              {/* Adapter path input + load button */}
              <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
                <input
                  type="text"
                  value={lmLoraPath}
                  onChange={(e) => setLmLoraPath(e.target.value)}
                  placeholder="Path to PEFT adapter directory..."
                  className="text-input"
                  style={{ flex: 1, fontSize: '12px' }}
                  disabled={isLmLoraLoading || lmLoraLoaded}
                />
                <button
                  className={`btn btn-sm ${lmLoraLoaded ? 'btn-danger' : 'btn-primary'}`}
                  onClick={handleLmLoraToggle}
                  disabled={isLmLoraLoading}
                  style={{ fontSize: '11px', padding: '4px 10px', whiteSpace: 'nowrap' }}
                >
                  {isLmLoraLoading ? (
                    <Loader2 size={12} className="spin" />
                  ) : lmLoraLoaded ? 'Unload' : 'Merge & Load'}
                </button>
              </div>

              {/* Scale slider (always visible — set before loading, locked after) */}
              <div style={{ marginBottom: '6px' }}>
                <EditableSlider
                  label={`Scale${lmLoraLoaded ? ' (locked — unload to change)' : ''}`}
                  value={lmLoraScale}
                  onChange={handleLmLoraScaleChange}
                  min={0}
                  max={10}
                  step={0.1}
                  disabled={lmLoraLoaded}
                />
              </div>

              {isLmLoraLoading && (
                <div style={{
                  fontSize: '11px', color: 'var(--text-secondary)',
                  padding: '4px 8px', fontStyle: 'italic',
                }}>Merging adapter into model... this takes ~2 min</div>
              )}

              {/* Error message */}
              {lmLoraError && (
                <div style={{
                  fontSize: '11px', color: 'var(--error, #ef4444)',
                  padding: '4px 8px', borderRadius: '4px',
                  background: 'rgba(239,68,68,0.1)',
                }}>{lmLoraError}</div>
              )}
            </div>
          </div>
        )}

        {/* LAYER ABLATION LAB (Developer Mode) */}
        <LayerAblationPanel
          customMode={customMode}
          hasLoadedAdapters={advancedAdapters && adapterSlots.length > 0}
          onLayerScaleChange={handleSlotLayerScaleChange}
          onBulkLayerScalesChange={handleBulkLayerScalesChange}
          onRunSweep={handleStartSweep}
          isSweepRunning={sweepRunning}
          sweepProgress={sweepProgress}
          onCancelSweep={handleCancelSweep}
          isGenerating={isGenerating}
          diffPinnedA={diffPinnedA}
          diffPinnedB={diffPinnedB}
          onClearDiffA={onClearDiffA}
          onClearDiffB={onClearDiffB}
        />

        {/* ACTIVATION STEERING */}
        <ActivationSteeringSection
          customMode={customMode}
          isOpen={showSteeringPanel}
          onToggle={() => setShowSteeringPanel(!showSteeringPanel)}
          onSteeringChange={handleSteeringChange}
        />

        {/* SCORE SYSTEM */}
        <ScoreSystemAccordion
          isOpen={showScorePanel}
          onToggle={() => setShowScorePanel(!showScorePanel)}
          getScores={getScores}
          onToggleGetScores={() => setGetScores(!getScores)}
          scoreScale={scoreScale}
          onScoreScaleChange={setScoreScale}
        />

      </div>

      <AudioLibraryModal
        showAudioModal={showAudioModal}
        setShowAudioModal={setShowAudioModal}
        audioModalTarget={audioModalTarget}
        setPlayingTrackId={setPlayingTrackId}
        setPlayingTrackSource={setPlayingTrackSource}
        uploadReferenceTrack={uploadReferenceTrack}
        isUploadingReference={isUploadingReference}
        isTranscribingReference={isTranscribingReference}
        uploadError={uploadError}
        cancelTranscription={cancelTranscription}
        libraryTab={libraryTab}
        setLibraryTab={setLibraryTab}
        isLoadingTracks={isLoadingTracks}
        referenceTracks={referenceTracks}
        setReferenceTracks={setReferenceTracks}
        toggleModalTrack={toggleModalTrack}
        playingTrackId={playingTrackId}
        playingTrackSource={playingTrackSource}
        modalTrackTime={modalTrackTime}
        setModalTrackTime={setModalTrackTime}
        modalTrackDuration={modalTrackDuration}
        setModalTrackDuration={setModalTrackDuration}
        modalAudioRef={modalAudioRef}
        formatTime={formatTime}
        useReferenceTrack={useReferenceTrack}
        deleteReferenceTrack={deleteReferenceTrack}
        createdTrackOptions={createdTrackOptions}
        token={token}
      />

      {/* Footer Create Button */}
      <div className="p-4 mt-auto sticky bottom-0 bg-zinc-50/95 dark:bg-suno-panel/95 backdrop-blur-sm z-10 border-t border-zinc-200 dark:border-white/5 space-y-3">
        <button
          onClick={handleGenerate}
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
                  : 'Sending…'
                }
              </span>
            </>
          ) : (
            <>
              <Sparkles size={18} />
              <span>
                {bulkCount > 1
                  ? `${t('createButton')} ${bulkCount} ${t('jobs')} (${bulkCount * batchSize} ${t('variations')})`
                  : `${t('createButton')}${batchSize > 1 ? ` (${batchSize} ${t('variations')})` : ''}`
                }
              </span>
            </>
          )}
        </button>
      </div>
    </div >
  );
};

export default CreatePanel;
