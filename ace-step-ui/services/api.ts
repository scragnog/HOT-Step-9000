// Use relative URLs so Vite proxy handles them (enables LAN access)
const API_BASE = '';

// Resolve audio URL based on storage type
export function getAudioUrl(audioUrl: string | undefined | null, songId?: string): string | undefined {
  if (!audioUrl) return undefined;

  // Local storage: already relative, works with proxy
  if (audioUrl.startsWith('/audio/')) {
    return audioUrl;
  }

  // Already a full URL
  return audioUrl;
}

interface ApiOptions {
  method?: string;
  body?: unknown;
  token?: string | null;
}

async function api<T>(endpoint: string, options: ApiOptions = {}): Promise<T> {
  const { method = 'GET', body, token } = options;

  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
    credentials: 'include',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Request failed' }));
    const errorMessage = error.error || error.message || 'Request failed';
    // Include status code in error for proper handling
    throw new Error(`${response.status}: ${errorMessage}`);
  }

  return response.json();
}

// Auth API (simplified - username only)
export interface User {
  id: string;
  username: string;
  isAdmin?: boolean;
  bio?: string;
  avatar_url?: string;
  banner_url?: string;
  createdAt?: string;
}

export interface AuthResponse {
  user: User;
  token: string;
}

export const authApi = {
  // Auto-login: Get existing user from database (for local single-user app)
  auto: (): Promise<AuthResponse> =>
    api('/api/auth/auto'),

  setup: (username: string): Promise<AuthResponse> =>
    api('/api/auth/setup', { method: 'POST', body: { username } }),

  me: (token: string): Promise<{ user: User }> =>
    api('/api/auth/me', { token }),

  logout: (): Promise<{ success: boolean }> =>
    api('/api/auth/logout', { method: 'POST' }),

  refresh: (token: string): Promise<AuthResponse> =>
    api('/api/auth/refresh', { method: 'POST', token }),

  updateUsername: (username: string, token: string): Promise<AuthResponse> =>
    api('/api/auth/username', { method: 'PATCH', body: { username }, token }),
};

// Songs API
export interface Song {
  id: string;
  title: string;
  lyrics: string;
  style: string;
  caption?: string;
  cover_url?: string;
  audio_url?: string;
  audioUrl?: string;
  duration?: number;
  bpm?: number;
  key_scale?: string;
  time_signature?: string;
  tags: string[];
  is_public?: boolean;
  like_count?: number;
  view_count?: number;
  user_id?: string;
  created_at: string;
  creator?: string;
  creator_avatar?: string;
  ditModel?: string;
  generation_params?: any;
}

// Transform songs to have proper audio URLs
function transformSongs(songs: Song[]): Song[] {
  return songs.map(song => {
    const rawUrl = song.audio_url || song.audioUrl;
    const resolvedUrl = getAudioUrl(rawUrl, song.id);

    // Parse generation_params JSON from DB into camelCase generationParams
    let generationParams: any = undefined;
    if (song.generation_params) {
      try {
        generationParams = typeof song.generation_params === 'string'
          ? JSON.parse(song.generation_params)
          : song.generation_params;
      } catch { /* ignore parse errors */ }
    }

    return {
      ...song,
      audio_url: resolvedUrl,
      audioUrl: resolvedUrl,
      generationParams,
    };
  });
}

export const songsApi = {
  getMySongs: async (token: string): Promise<{ songs: Song[] }> => {
    const result = await api('/api/songs', { token }) as { songs: Song[] };
    return { songs: transformSongs(result.songs) };
  },

  getPublicSongs: async (limit = 20, offset = 0): Promise<{ songs: Song[] }> => {
    const result = await api(`/api/songs/public?limit=${limit}&offset=${offset}`) as { songs: Song[] };
    return { songs: transformSongs(result.songs) };
  },

  getFeaturedSongs: async (): Promise<{ songs: Song[] }> => {
    const result = await api('/api/songs/public/featured') as { songs: Song[] };
    return { songs: transformSongs(result.songs) };
  },

  getSong: async (id: string, token?: string | null): Promise<{ song: Song }> => {
    const result = await api(`/api/songs/${id}`, { token: token || undefined }) as { song: Song };
    const rawUrl = result.song.audio_url || result.song.audioUrl;
    const resolvedUrl = getAudioUrl(rawUrl, result.song.id);
    return { song: { ...result.song, audio_url: resolvedUrl, audioUrl: resolvedUrl } };
  },

  getFullSong: async (id: string, token?: string | null): Promise<{ song: Song, comments: any[] }> => {
    const result = await api(`/api/songs/${id}/full`, { token: token || undefined }) as { song: Song, comments: any[] };
    const rawUrl = result.song.audio_url || result.song.audioUrl;
    const resolvedUrl = getAudioUrl(rawUrl, result.song.id);
    return { ...result, song: { ...result.song, audio_url: resolvedUrl, audioUrl: resolvedUrl } };
  },

  createSong: (song: Partial<Song>, token: string): Promise<{ song: Song }> =>
    api('/api/songs', { method: 'POST', body: song, token }),

  updateSong: async (id: string, updates: Partial<Song>, token: string): Promise<{ song: any }> => {
    const result = await api(`/api/songs/${id}`, { method: 'PATCH', body: updates, token }) as { song: any };
    const s = result.song;
    const rawUrl = s.audio_url || s.audioUrl;
    const resolvedUrl = getAudioUrl(rawUrl, s.id);

    const generationParams = (() => {
      try {
        if (!s.generation_params) return undefined;
        return typeof s.generation_params === 'string' ? JSON.parse(s.generation_params) : s.generation_params;
      } catch {
        return undefined;
      }
    })();

    return {
      song: {
        id: s.id,
        title: s.title,
        lyrics: s.lyrics,
        style: s.style,
        caption: s.caption,
        cover_url: s.cover_url,
        coverUrl: s.cover_url || s.coverUrl || `https://picsum.photos/seed/${s.id}/400/400`,
        duration: s.duration && s.duration > 0 ? `${Math.floor(s.duration / 60)}:${String(Math.floor(s.duration % 60)).padStart(2, '0')}` : '0:00',
        createdAt: new Date(s.created_at || s.createdAt),
        created_at: s.created_at,
        tags: s.tags || [],
        audioUrl: resolvedUrl,
        audio_url: resolvedUrl,
        isPublic: s.is_public ?? s.isPublic,
        is_public: s.is_public ?? s.isPublic,
        likeCount: s.like_count || s.likeCount || 0,
        like_count: s.like_count || s.likeCount || 0,
        viewCount: s.view_count || s.viewCount || 0,
        view_count: s.view_count || s.viewCount || 0,
        userId: s.user_id || s.userId,
        user_id: s.user_id || s.userId,
        creator: s.creator,
        creator_avatar: s.creator_avatar,
        ditModel: s.dit_model || s.ditModel,
        isGenerating: s.isGenerating,
        queuePosition: s.queuePosition,
        bpm: s.bpm,
        key_scale: s.key_scale,
        time_signature: s.time_signature,
        generation_params: s.generation_params,
        generationParams,
      }
    };
  },

  deleteSong: (id: string, token: string): Promise<{ success: boolean }> =>
    api(`/api/songs/${id}`, { method: 'DELETE', token }),

  deleteAllSongs: (token: string): Promise<{ success: boolean; deletedCount: number }> =>
    api('/api/songs/all', { method: 'DELETE', token }),

  deleteAllReferenceTracks: (token: string): Promise<{ success: boolean; deletedCount: number }> =>
    api('/api/reference-tracks/all', { method: 'DELETE', token }),

  toggleLike: (id: string, token: string): Promise<{ liked: boolean }> =>
    api(`/api/songs/${id}/like`, { method: 'POST', token }),

  getLikedSongs: async (token: string): Promise<{ songs: Song[] }> => {
    const result = await api('/api/songs/liked/list', { token }) as { songs: Song[] };
    return { songs: transformSongs(result.songs) };
  },

  togglePrivacy: (id: string, token: string): Promise<{ isPublic: boolean }> =>
    api(`/api/songs/${id}/privacy`, { method: 'PATCH', token }),

  trackPlay: (id: string, token?: string | null): Promise<{ viewCount: number }> =>
    api(`/api/songs/${id}/play`, { method: 'POST', token: token || undefined }),

  getComments: (id: string, token?: string | null): Promise<{ comments: Comment[] }> =>
    api(`/api/songs/${id}/comments`, { token: token || undefined }),

  addComment: (id: string, content: string, token: string): Promise<{ comment: Comment }> =>
    api(`/api/songs/${id}/comments`, { method: 'POST', body: { content }, token }),

  deleteComment: (commentId: string, token: string): Promise<{ success: boolean }> =>
    api(`/api/songs/comments/${commentId}`, { method: 'DELETE', token }),
};

interface Comment {
  id: string;
  song_id: string;
  user_id: string;
  username: string;
  content: string;
  created_at: string;
}

// Generation API
export interface GenerationParams {
  // Mode
  customMode: boolean;
  songDescription?: string;

  // Custom Mode
  prompt?: string;
  lyrics: string;
  style: string;
  title: string;

  // Model Selection
  ditModel?: string;

  // Common
  instrumental: boolean;
  vocalLanguage?: string;

  // Music Parameters
  duration?: number;
  bpm?: number;
  keyScale?: string;
  timeSignature?: string;

  // Generation Settings
  inferenceSteps?: number;
  guidanceScale?: number;
  batchSize?: number;
  randomSeed?: boolean;
  seed?: number;
  thinking?: boolean;
  audioFormat?: 'mp3' | 'flac';
  inferMethod?: 'ode' | 'euler' | 'heun' | 'dpm2m' | 'rk4';
  scheduler?: 'linear' | 'ddim_uniform' | 'sgm_uniform' | 'bong_tangent' | 'linear_quadratic';
  shift?: number;

  // LM Parameters
  lmTemperature?: number;
  lmCfgScale?: number;
  lmTopK?: number;
  lmTopP?: number;
  lmNegativePrompt?: string;
  lmBackend?: 'pt' | 'vllm';
  lmModel?: string;

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
  autoMaster?: boolean;
  masteringParams?: Record<string, any>;
  enableNormalization?: boolean;
  normalizationDb?: number;
  latentShift?: number;
  latentRescale?: number;
  taskType?: string;
  useAdg?: boolean;
  cfgIntervalStart?: number;
  cfgIntervalEnd?: number;
  customTimesteps?: string;
  useCotMetas?: boolean;
  useCotCaption?: boolean;
  useCotLanguage?: boolean;
  autogen?: boolean;
  constrainedDecodingDebug?: boolean;
  allowLmBatch?: boolean;
  getScores?: boolean;
  getLrc?: boolean;
  scoreScale?: number;
  lmBatchChunkSize?: number;
  trackName?: string;
  completeTrackClasses?: string[];
  isFormatCaption?: boolean;
  loraLoaded?: boolean;

  // PAG (Perturbed-Attention Guidance)
  usePag?: boolean;
  pagStart?: number;
  pagEnd?: number;
  pagScale?: number;

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
}

export interface GenerationJob {
  id?: string;
  params?: unknown;
  created_at?: string;
  jobId: string;
  status: 'pending' | 'queued' | 'running' | 'succeeded' | 'failed';
  queuePosition?: number;
  etaSeconds?: number;
  progress?: number;
  stage?: string;
  result?: {
    audioUrls: string[];
    bpm?: number;
    duration?: number;
    keyScale?: string;
    timeSignature?: string;
  };
  error?: string;
}

export const generateApi = {
  startGeneration: (params: GenerationParams, token: string): Promise<GenerationJob> =>
    api('/api/generate', { method: 'POST', body: params, token }),

  getStatus: (jobId: string, token: string): Promise<GenerationJob> =>
    api(`/api/generate/status/${jobId}`, { token }),

  getHistory: (token: string): Promise<{ jobs: GenerationJob[] }> =>
    api('/api/generate/history', { token }),

  deleteJob: (jobId: string, token: string): Promise<{ success: boolean }> =>
    api(`/api/generate/job/${jobId}`, { method: 'DELETE', token }),

  cancelJob: (jobId: string, token: string): Promise<{ success: boolean; jobId: string }> =>
    api(`/api/generate/cancel/${jobId}`, { method: 'POST', token }),

  cancelAllJobs: (token: string): Promise<{ success: boolean; cancelled: number }> =>
    api('/api/generate/cancel-all', { method: 'POST', token }),

  uploadAudio: async (file: File, token: string): Promise<{ url: string; key: string }> => {
    const formData = new FormData();
    formData.append('audio', file);
    const response = await fetch(`${API_BASE}/api/generate/upload-audio`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` },
      body: formData,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Upload failed' }));
      throw new Error(error.details || error.error || 'Upload failed');
    }
    return response.json();
  },

  formatInput: (params: {
    caption: string;
    lyrics?: string;
    bpm?: number;
    duration?: number;
    keyScale?: string;
    timeSignature?: string;
    temperature?: number;
    topK?: number;
    topP?: number;
    lmModel?: string;
    lmBackend?: string;
  }, token: string): Promise<{
    caption?: string;
    lyrics?: string;
    bpm?: number;
    duration?: number;
    key_scale?: string;
    vocal_language?: string;
    time_signature?: string;
    status_message?: string;
    error?: string;
  }> => api('/api/generate/format', { method: 'POST', body: params, token }),

  // LoRA Inference
  loadLora: (params: {
    lora_path: string;
    slot?: number;
    scale?: number;
    group_scales?: { self_attn: number; cross_attn: number; mlp: number };
  }, token: string): Promise<{
    message: string;
    lora_path: string;
    slot?: number;
  }> => api('/api/lora/load', { method: 'POST', body: params, token }),

  unloadLora: (token: string, slot?: number): Promise<{
    message: string;
  }> => api('/api/lora/unload', { method: 'POST', body: slot != null ? { slot } : {}, token }),

  setLoraScale: (params: {
    scale: number;
    slot?: number;
  }, token: string): Promise<{
    message: string;
    scale: number;
  }> => api('/api/lora/scale', { method: 'POST', body: params, token }),

  getLoraStatus: (token: string): Promise<{
    lora_loaded: boolean;
    use_lora: boolean;
    lora_scale: number;
    adapter_type: string | null;
    advanced?: {
      loaded: boolean;
      active: boolean;
      slots: Array<{
        slot: number;
        name: string;
        path: string;
        type: string;
        scale: number;
        delta_keys: number;
        group_scales: { self_attn: number; cross_attn: number; mlp: number };
      }>;
      group_scales: { self_attn: number; cross_attn: number; mlp: number };
    };
  }> => api('/api/lora/status', { token }),

  // LM LoRA (PEFT adapter on the 5Hz language model)
  loadLmLora: (params: {
    lm_lora_path: string;
    scale?: number;
  }, token: string): Promise<{
    message: string;
    lm_lora_path: string;
    scale: number;
  }> => api('/api/lora/lm-load', { method: 'POST', body: params, token }),

  unloadLmLora: (token: string): Promise<{
    message: string;
  }> => api('/api/lora/lm-unload', { method: 'POST', body: {}, token }),

  setLmLoraScale: (scale: number, token: string): Promise<{
    message: string;
    scale: number;
  }> => api('/api/lora/lm-scale', { method: 'POST', body: { scale }, token }),

  getLmLoraStatus: (token: string): Promise<{
    loaded: boolean;
    lm_lora_path: string;
    scale: number;
    backend: string;
    auto_switched: boolean;
    message: string;
  }> => api('/api/lora/lm-status', { token }),

  // Advanced adapter: file browser
  listLoraFiles: (folder: string, token: string): Promise<{
    files: Array<{
      name: string;
      path: string;
      size: number;
      type: string;
    }>;
    folder: string;
  }> => api(`/api/lora/list-files?folder=${encodeURIComponent(folder)}`, { token }),

  // Open native folder picker dialog
  browseLoraFolder: (token: string): Promise<{
    folder: string;
  }> => api('/api/lora/browse-folder', { token }),

  // Open native file picker dialog (.safetensors)
  browseLoraFile: (token: string): Promise<{
    file: string;
  }> => api('/api/lora/browse-file', { token }),

  // Advanced adapter: group scales
  setGroupScales: (params: {
    self_attn: number;
    cross_attn: number;
    mlp: number;
  }, token: string): Promise<{
    message: string;
    group_scales: { self_attn: number; cross_attn: number; mlp: number };
  }> => api('/api/lora/group-scales', { method: 'POST', body: params, token }),

  // Advanced adapter: per-slot group scales
  setSlotGroupScales: (params: {
    slot: number;
    self_attn: number;
    cross_attn: number;
    mlp: number;
  }, token: string): Promise<{
    message: string;
    slot: number;
    group_scales: { self_attn: number; cross_attn: number; mlp: number };
  }> => api('/api/lora/slot-group-scales', { method: 'POST', body: params, token }),

  // Advanced adapter: per-slot layer scales (batch)
  setSlotLayerScales: (params: {
    slot: number;
    layer_scales: Record<number, number>;
  }, token: string): Promise<{
    message: string;
    slot: number;
    layer_scales: Record<number, number>;
  }> => api('/api/lora/slot-layer-scales', { method: 'POST', body: params, token }),

  // Advanced adapter: single layer scale
  setSlotLayerScale: (params: {
    slot: number;
    layer: number;
    scale: number;
  }, token: string): Promise<{
    message: string;
    slot: number;
    layer: number;
    scale: number;
  }> => api('/api/lora/slot-layer-scale', { method: 'POST', body: params, token }),

  // Advanced adapter: temporal schedule for multi-singer switching
  setTemporalSchedule: (params: {
    clear?: boolean;
    slot_segments?: Record<number, Array<{
      start: number;
      end: number;
      scale?: number;
      fade_in?: number;
      fade_out?: number;
    }>>;
  }, token: string): Promise<{
    message: string;
  }> => api('/api/lora/temporal-schedule', { method: 'POST', body: params, token }),

  // Audio diff for layer ablation experiments
  computeAudioDiff: (params: {
    reference_path: string;
    ablated_path: string;
    amplify?: number;
  }, token: string): Promise<{
    output_path: string;
    rms_energy: number;
    peak: number;
    duration_match: boolean;
    sample_rate: number;
    duration_seconds: number;
  }> => api('/api/lora/audio-diff', { method: 'POST', body: params, token }),

  // Model Management
  getModels: (token: string): Promise<{
    models: Array<{
      name: string;
      is_active: boolean;
      is_preloaded: boolean;
      is_default: boolean;
    }>;
    active_model: string | null;
  }> => api('/api/models', { token }),

  switchModel: (model: string, token: string): Promise<{
    message: string;
    active_model: string;
    switched: boolean;
  }> => api('/api/models/switch', { method: 'POST', body: { model }, token }),

  switchLmModel: (model: string, token: string): Promise<{
    message: string;
    lm_model: string;
    switched: boolean;
  }> => api('/api/model/lm/switch', { method: 'POST', body: { model }, token }),
};

// Activation Steering (TADA) API
export interface SteeringStatus {
  enabled: boolean;
  loaded_concepts: string[];
  available_concepts: string[];
  builtin_concepts: string[];
  config: Record<string, { alpha: number; layers: string; mode: string }>;
}

export const steeringApi = {
  getConcepts: (token: string): Promise<SteeringStatus> =>
    api('/api/steering/concepts', { token }),

  compute: (params: {
    concept: string;
    num_steps?: number;
    num_samples?: number;
    seed?: number;
    positive_template?: string;
    negative_template?: string;
    custom_base_prompts?: string[];
  }, token: string): Promise<SteeringStatus & { message: string, filepath: string, elapsed_seconds: number }> =>
    api('/api/steering/compute', { method: 'POST', body: params, token }),

  load: (concept: string, token: string): Promise<SteeringStatus & { message: string }> =>
    api('/api/steering/load', { method: 'POST', body: { concept }, token }),

  unload: (concept: string, token: string): Promise<SteeringStatus & { message: string }> =>
    api('/api/steering/unload', { method: 'POST', body: { concept }, token }),

  config: (params: {
    concept: string;
    alpha?: number;
    layers?: string;
    timesteps?: string;
  }, token: string): Promise<SteeringStatus & { message: string }> =>
    api('/api/steering/config', { method: 'POST', body: params, token }),

  enable: (enabled: boolean, token: string): Promise<SteeringStatus & { message: string }> =>
    api('/api/steering/enable', { method: 'POST', body: { enabled }, token }),

  delete: (concept: string, token: string): Promise<SteeringStatus & { message: string }> =>
    api(`/api/steering/concepts/${encodeURIComponent(concept)}`, { method: 'DELETE', token }),
};

// Users API
export interface UserProfile extends User {
  bio?: string;
  avatar_url?: string;
  banner_url?: string;
  created_at: string;
}

export const usersApi = {
  getProfile: (username: string, token?: string | null): Promise<{ user: UserProfile }> =>
    api(`/api/users/${username}`, { token: token || undefined }),

  getPublicSongs: (username: string): Promise<{ songs: Song[] }> =>
    api(`/api/users/${username}/songs`),

  getPublicPlaylists: (username: string): Promise<{ playlists: any[] }> =>
    api(`/api/users/${username}/playlists`),

  getFeaturedCreators: (): Promise<{ creators: Array<UserProfile & { follower_count?: number }> }> =>
    api('/api/users/public/featured'),

  updateProfile: (updates: Partial<User>, token: string): Promise<{ user: User }> =>
    api('/api/users/me', { method: 'PATCH', body: updates, token }),

  uploadAvatar: async (file: File, token: string): Promise<{ user: UserProfile; url: string }> => {
    const formData = new FormData();
    formData.append('avatar', file);
    const response = await fetch(`${API_BASE}/api/users/me/avatar`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` },
      body: formData,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Upload failed' }));
      throw new Error(error.details || error.error || 'Upload failed');
    }
    return response.json();
  },

  uploadBanner: async (file: File, token: string): Promise<{ user: UserProfile; url: string }> => {
    const formData = new FormData();
    formData.append('banner', file);
    const response = await fetch(`${API_BASE}/api/users/me/banner`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` },
      body: formData,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Upload failed' }));
      throw new Error(error.error || 'Upload failed');
    }
    return response.json();
  },

  toggleFollow: (username: string, token: string): Promise<{ following: boolean, followerCount: number }> =>
    api(`/api/users/${username}/follow`, { method: 'POST', token }),

  getFollowers: (username: string): Promise<{ followers: User[] }> =>
    api(`/api/users/${username}/followers`),

  getFollowing: (username: string): Promise<{ following: User[] }> =>
    api(`/api/users/${username}/following`),

  getStats: (username: string, token?: string | null): Promise<{ followerCount: number, followingCount: number, isFollowing: boolean }> =>
    api(`/api/users/${username}/stats`, { token: token || undefined }),
};

// Playlists API
export interface Playlist {
  id: string;
  name: string;
  description?: string;
  cover_url?: string;
  is_public?: boolean;
  user_id?: string;
  created_at?: string;
  song_count?: number;
}

export const playlistsApi = {
  create: (name: string, description: string, isPublic: boolean, token: string): Promise<{ playlist: Playlist }> =>
    api('/api/playlists', { method: 'POST', body: { name, description, isPublic }, token }),

  getMyPlaylists: (token: string): Promise<{ playlists: Playlist[] }> =>
    api('/api/playlists', { token }),

  getPlaylist: (id: string, token?: string | null): Promise<{ playlist: Playlist, songs: any[] }> =>
    api(`/api/playlists/${id}`, { token: token || undefined }),

  getFeaturedPlaylists: (): Promise<{ playlists: Array<Playlist & { creator?: string; creator_avatar?: string }> }> =>
    api('/api/playlists/public/featured'),

  addSong: (playlistId: string, songId: string, token: string): Promise<{ success: boolean }> =>
    api(`/api/playlists/${playlistId}/songs`, { method: 'POST', body: { songId }, token }),

  removeSong: (playlistId: string, songId: string, token: string): Promise<{ success: boolean }> =>
    api(`/api/playlists/${playlistId}/songs/${songId}`, { method: 'DELETE', token }),

  update: (id: string, updates: Partial<Playlist>, token: string): Promise<{ playlist: Playlist }> =>
    api(`/api/playlists/${id}`, { method: 'PATCH', body: updates, token }),

  delete: (id: string, token: string): Promise<{ success: boolean }> =>
    api(`/api/playlists/${id}`, { method: 'DELETE', token }),
};

// Search API
export interface SearchResult {
  songs: Song[];
  creators: Array<UserProfile & { follower_count?: number }>;
  playlists: Array<Playlist & { creator?: string; creator_avatar?: string }>;
}

export const searchApi = {
  search: async (query: string, type?: 'songs' | 'creators' | 'playlists' | 'all'): Promise<SearchResult> => {
    const params = new URLSearchParams({ q: query });
    if (type && type !== 'all') params.append('type', type);
    const result = await api(`/api/search?${params}`) as SearchResult;
    return {
      ...result,
      songs: transformSongs(result.songs || []),
    };
  },
};

// Contact Form API
export interface ContactFormData {
  name: string;
  email: string;
  subject: string;
  message: string;
  category: 'general' | 'support' | 'business' | 'press' | 'legal';
}

export const contactApi = {
  submit: (data: ContactFormData): Promise<{ success: boolean; message: string; id: string }> =>
    api('/api/contact', { method: 'POST', body: data }),
};

// Model Management API
export interface ModelInfo {
  name: string;
  is_default: boolean;
  is_loaded: boolean;
}

export interface LmModelInfo {
  name: string;
  is_loaded: boolean;
}

export interface ModelInventory {
  models: ModelInfo[];
  default_model: string | null;
  lm_models: LmModelInfo[];
  loaded_lm_model: string | null;
  llm_initialized: boolean;
}

export interface HealthStatus {
  status: string;
  service: string;
  version: string;
  models_initialized: boolean;
  llm_initialized: boolean;
  loaded_model: string | null;
  loaded_lm_model: string | null;
}

export interface ServerStats {
  jobs: Record<string, number>;
  queue_size: number;
  queue_maxsize: number;
  avg_job_seconds: number;
}

export interface LoraAdapterInfo {
  lora_loaded: boolean;
  use_lora: boolean;
  lora_scale: number;
  adapter_type: string | null;
  scales: Record<string, number>;
  active_adapter: string | null;
  adapters: string[];
  synthetic_default_mode: boolean;
}

export const modelApi = {
  getHealth: (token?: string | null): Promise<HealthStatus> =>
    api('/api/model/health', { token: token || undefined }),

  getModels: (token: string): Promise<ModelInventory> =>
    api('/api/model/models', { token }),

  initModel: (params: {
    model?: string;
    init_llm?: boolean;
    lm_model_path?: string;
  }, token: string): Promise<{
    message: string;
    loaded_model: string | null;
    loaded_lm_model: string | null;
    models: ModelInfo[];
    lm_models: LmModelInfo[];
    llm_initialized: boolean;
  }> => api('/api/model/init', { method: 'POST', body: params, token }),

  reinitialize: (token: string): Promise<{
    message: string;
    reloaded: string[];
  }> => api('/api/model/reinitialize', { method: 'POST', token }),

  getStats: (token: string): Promise<ServerStats> =>
    api('/api/model/stats', { token }),
};

// Enhanced LoRA API (multi-adapter)
export const loraApi = {
  load: (params: {
    lora_path: string;
    adapter_name?: string;
  }, token: string): Promise<{
    message: string;
    lora_path: string;
    adapter_name?: string;
  }> => api('/api/lora/load', { method: 'POST', body: params, token }),

  unload: (token: string): Promise<{
    message: string;
  }> => api('/api/lora/unload', { method: 'POST', token }),

  toggle: (params: {
    use_lora: boolean;
  }, token: string): Promise<{
    message: string;
    use_lora: boolean;
  }> => api('/api/lora/toggle', { method: 'POST', body: params, token }),

  setScale: (params: {
    scale: number;
    adapter_name?: string;
  }, token: string): Promise<{
    message: string;
    scale: number;
    adapter_name?: string;
  }> => api('/api/lora/scale', { method: 'POST', body: params, token }),

  getStatus: (token: string): Promise<LoraAdapterInfo> =>
    api('/api/lora/status', { token }),
};

