export interface Song {
  id: string;
  title: string;
  lyrics: string;
  style: string;
  coverUrl: string;
  duration: string;
  createdAt: Date;
  isGenerating?: boolean;
  queuePosition?: number; // Position in queue (undefined = actively generating, number = waiting in queue)
  progress?: number;
  stage?: string;
  generationParams?: any;
  tags: string[];
  audioUrl?: string;
  isPublic?: boolean;
  likeCount?: number;
  viewCount?: number;
  userId?: string;
  creator?: string;
  creator_avatar?: string;
  ditModel?: string; // DiT model used for generation (e.g., 'acestep-v15-sft')
}

export interface Playlist {
  id: string;
  name: string;
  description?: string;
  coverUrl?: string;
  cover_url?: string;
  songIds?: string[];
  isPublic?: boolean;
  is_public?: boolean;
  user_id?: string;
  creator?: string;
  created_at?: string;
  song_count?: number;
  songs?: any[];
}

export interface Comment {
  id: string;
  songId: string;
  userId: string;
  username: string;
  content: string;
  createdAt: Date;
}

export interface GenerationParams {
  // Mode
  customMode: boolean;

  // Simple Mode
  songDescription?: string;

  // Custom Mode
  prompt: string;
  lyrics: string;
  style: string;
  title: string;
  ditModel?: string;

  // Common
  instrumental: boolean;
  vocalLanguage: string;

  // Music Parameters
  bpm: number;
  keyScale: string;
  timeSignature: string;
  duration: number;

  // Generation Settings
  inferenceSteps: number;
  guidanceScale: number;
  batchSize: number;
  randomSeed: boolean;
  seed: number;
  thinking: boolean;
  audioFormat: 'mp3' | 'flac';
  inferMethod: 'ode' | 'euler' | 'heun' | 'dpm2m' | 'dpm3m' | 'rk4' | 'jkass_quality' | 'jkass_fast' | 'stork2' | 'stork4';
  scheduler: 'linear' | 'ddim_uniform' | 'sgm_uniform' | 'bong_tangent' | 'linear_quadratic';
  shift: number;

  // LM Parameters
  lmTemperature: number;
  lmCfgScale: number;
  lmTopK: number;
  lmTopP: number;
  lmNegativePrompt: string;
  lmRepetitionPenalty?: number;
  lmBackend?: 'pt' | 'vllm' | 'custom-vllm';
  lmModel?: string;
  lmCodesScale?: number;  // Scale influence of LM-generated audio codes (0.0-1.0, default 1.0)

  // Expert Parameters
  referenceAudioUrl?: string;
  sourceAudioUrl?: string;
  referenceAudioTitle?: string;
  sourceAudioTitle?: string;
  audioCodes?: string;
  repaintingStart?: number;
  repaintingEnd?: number;
  instruction?: string;
  audioCoverStrength?: number;
  coverNoiseStrength?: number;
  enableNormalization?: boolean;
  normalizationDb?: number;
  vocoderModel?: string;
  latentShift?: number;
  latentRescale?: number;
  taskType?: string;
  useAdg?: boolean;
  guidanceMode?: string;
  cfgIntervalStart?: number;
  cfgIntervalEnd?: number;
  customTimesteps?: string;
  useCotMetas?: boolean;
  useCotCaption?: boolean;
  useCotLanguage?: boolean;
  autogen?: boolean;
  constrainedDecoding?: boolean;
  constrainedDecodingDebug?: boolean;
  allowLmBatch?: boolean;
  getScores?: boolean;
  getLrc?: boolean;
  scoreScale?: number;
  lmBatchChunkSize?: number;
  trackName?: string;
  completeTrackClasses?: string[];
  isFormatCaption?: boolean;
  autoMaster?: boolean;
  masteringParams?: Record<string, any>;
  loraLoaded?: boolean;

  // PAG (Perturbed-Attention Guidance)
  usePag?: boolean;
  pagStart?: number;
  pagEnd?: number;
  pagScale?: number;

  // Anti-Autotune spectral smoothing
  antiAutotune?: number;

  // JKASS Fast solver parameters
  beatStability?: number;
  frequencyDamping?: number;
  temporalSmoothing?: number;

  // STORK solver parameters
  storkSubsteps?: number;

  // Advanced Guidance Parameters
  guidanceIntervalDecay?: number;
  minGuidanceScale?: number;
  referenceAsCover?: boolean;
  guidanceScaleText?: number;
  guidanceScaleLyric?: number;
  apgMomentum?: number;
  apgNormThreshold?: number;
  omegaScale?: number;
  ergScale?: number;

  // Activation Steering (TADA)
  steeringEnabled?: boolean;
  steeringLoaded?: string[];
  steeringAlphas?: Record<string, number>;

  // Adapters
  loraPath?: string;
  loraScale?: number;
  advancedAdapters?: boolean;
  adapterSlots?: Array<{
    slot: number;
    name: string;
    path: string;
    type: string;
    scale: number;
    delta_keys: number;
    group_scales: { self_attn: number; cross_attn: number; mlp: number };
  }>;

  // AI Cover Art
  coverArtSubject?: string;
  generateCoverArt?: boolean;
}

export interface PlayerState {
  currentSong: Song | null;
  isPlaying: boolean;
  progress: number;
  volume: number;
}

export interface User {
  id: string;
  username: string;
  createdAt: Date;
  followerCount?: number;
  followingCount?: number;
  isFollowing?: boolean;
  isAdmin?: boolean;
  avatar_url?: string;
  banner_url?: string;
}

export interface UserProfile {
  user: User;
  publicSongs: Song[];
  publicPlaylists: Playlist[];
  stats: {
    totalSongs: number;
    totalLikes: number;
  };
}

// Simplified views for ACE-Step UI
export type View = 'create' | 'library' | 'lyric-studio' | 'profile' | 'song' | 'playlist' | 'search';
