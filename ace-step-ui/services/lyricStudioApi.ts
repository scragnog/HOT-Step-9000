/**
 * Lyric Studio API client — typed wrappers for /api/lireek/ endpoints.
 */

const API_BASE = '';

async function api<T>(endpoint: string, options: {
  method?: string;
  body?: unknown;
} = {}): Promise<T> {
  const { method = 'GET', body } = options;
  const headers: HeadersInit = { 'Content-Type': 'application/json' };

  const response = await fetch(`${API_BASE}${endpoint}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
    credentials: 'include',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Request failed' }));
    throw new Error(error.detail || error.error || error.message || `Request failed (${response.status})`);
  }

  return response.json();
}

// ── Types ────────────────────────────────────────────────────────────────────

export interface Artist {
  id: number;
  name: string;
  image_url?: string;
  genius_id?: number;
  lyrics_set_count?: number;
  created_at: string;
}

export interface LyricsSet {
  id: number;
  artist_id: number;
  artist_name: string;
  album: string | null;
  image_url?: string;
  songs: SongLyric[] | string;
  max_songs: number;
  created_at: string;
}

export interface SongLyric {
  title: string;
  lyrics: string;
  url?: string;
}

export interface Profile {
  id: number;
  lyrics_set_id: number;
  provider: string;
  model: string;
  profile_data: Record<string, any>;
  created_at: string;
}

export interface Generation {
  id: number;
  profile_id: number;
  provider: string;
  model: string;
  lyrics: string;
  title?: string;
  subject?: string;
  caption?: string;
  bpm?: number;
  key?: string;
  duration?: number;
  parent_generation_id?: number;
  created_at: string;
  // Context fields (from /generations/all)
  artist_name?: string;
  album?: string;
}

export interface AlbumPreset {
  id: number;
  lyrics_set_id: number;
  adapter_path?: string;
  adapter_scale?: number;
  adapter_group_scales?: { self_attn: number; cross_attn: number; mlp: number };
  matchering_reference_path?: string;
  created_at: string;
}

export interface AudioGeneration {
  id: number;
  generation_id: number;
  job_id: string;
  created_at: string;
}

// ── API ─────────────────────────────────────────────────────────────────────

export const lireekApi = {
  // ── Artists ──────────────────────────────────────────────────────────────
  listArtists: (): Promise<{ artists: Artist[] }> =>
    api('/api/lireek/artists'),

  deleteArtist: (id: number): Promise<{ deleted: boolean }> =>
    api(`/api/lireek/artists/${id}`, { method: 'DELETE' }),

  refreshArtistImage: (id: number): Promise<{ image_url: string }> =>
    api(`/api/lireek/artists/${id}/refresh-image`, { method: 'POST' }),

  setArtistImage: (id: number, imageUrl: string): Promise<{ image_url: string }> =>
    api(`/api/lireek/artists/${id}/set-image`, { method: 'POST', body: { image_url: imageUrl } }),

  listLyricsSets: (artistId?: number, includeFull?: boolean): Promise<{ lyrics_sets: LyricsSet[] }> => {
    const params = new URLSearchParams();
    if (artistId != null) params.set('artist_id', String(artistId));
    if (includeFull) params.set('include_full', 'true');
    const qs = params.toString();
    return api(`/api/lireek/lyrics-sets${qs ? `?${qs}` : ''}`);
  },

  getLyricsSet: (id: number): Promise<LyricsSet> =>
    api(`/api/lireek/lyrics-sets/${id}`),

  deleteLyricsSet: (id: number): Promise<{ deleted: boolean }> =>
    api(`/api/lireek/lyrics-sets/${id}`, { method: 'DELETE' }),

  removeSong: (lyricsSetId: number, songIndex: number): Promise<any> =>
    api(`/api/lireek/lyrics-sets/${lyricsSetId}/songs/${songIndex}`, { method: 'DELETE' }),

  editSong: (lyricsSetId: number, songIndex: number, lyrics: string): Promise<LyricsSet> =>
    api(`/api/lireek/lyrics-sets/${lyricsSetId}/songs/${songIndex}`, { method: 'PUT', body: { lyrics } }),

  refreshAlbumImage: (id: number): Promise<{ image_url: string }> =>
    api(`/api/lireek/lyrics-sets/${id}/refresh-image`, { method: 'POST' }),

  // ── Genius Fetch ────────────────────────────────────────────────────────
  fetchLyrics: (params: {
    artist: string;
    album?: string;
    max_songs?: number;
  }): Promise<{ artist: Artist; lyrics_set: LyricsSet; songs_fetched: number }> =>
    api('/api/lireek/fetch-lyrics', { method: 'POST', body: params }),

  listProfiles: (lyricsSetId?: number, includeFull?: boolean): Promise<{ profiles: Profile[] }> => {
    const params = new URLSearchParams();
    if (lyricsSetId != null) params.set('lyrics_set_id', String(lyricsSetId));
    if (includeFull) params.set('include_full', 'true');
    const qs = params.toString();
    return api(`/api/lireek/profiles${qs ? `?${qs}` : ''}`);
  },

  getProfile: (id: number): Promise<Profile> =>
    api(`/api/lireek/profiles/${id}`),

  deleteProfile: (id: number): Promise<{ deleted: boolean }> =>
    api(`/api/lireek/profiles/${id}`, { method: 'DELETE' }),

  buildProfile: (lyricsSetId: number, params: {
    provider: string;
    model?: string;
  }): Promise<Profile> =>
    api(`/api/lireek/lyrics-sets/${lyricsSetId}/build-profile`, { method: 'POST', body: params }),

  listGenerations: (profileId?: number, includeFull?: boolean): Promise<{ generations: Generation[] }> => {
    const params = new URLSearchParams();
    if (profileId != null) params.set('profile_id', String(profileId));
    if (includeFull) params.set('include_full', 'true');
    const qs = params.toString();
    return api(`/api/lireek/generations${qs ? `?${qs}` : ''}`);
  },

  listAllGenerations: (): Promise<{ generations: Generation[] }> =>
    api('/api/lireek/generations/all'),

  getGeneration: (id: number): Promise<Generation> =>
    api(`/api/lireek/generations/${id}`),

  generateLyrics: (profileId: number, params: {
    profile_id: number;
    provider: string;
    model?: string;
    extra_instructions?: string;
  }): Promise<Generation> =>
    api(`/api/lireek/profiles/${profileId}/generate`, { method: 'POST', body: params }),

  refineLyrics: (generationId: number, params: {
    provider: string;
    model?: string;
  }): Promise<Generation> =>
    api(`/api/lireek/generations/${generationId}/refine`, { method: 'POST', body: params }),

  updateMetadata: (generationId: number, updates: {
    title?: string;
    caption?: string;
    bpm?: number;
    key?: string;
    duration?: number;
    subject?: string;
  }): Promise<any> =>
    api(`/api/lireek/generations/${generationId}/metadata`, { method: 'PATCH', body: updates }),

  deleteGeneration: (id: number): Promise<{ deleted: boolean }> =>
    api(`/api/lireek/generations/${id}`, { method: 'DELETE' }),

  // ── Export ──────────────────────────────────────────────────────────────
  exportGeneration: (generationId: number): Promise<{ exported: boolean; path: string }> =>
    api(`/api/lireek/generations/${generationId}/export`, { method: 'POST' }),

  // ── Album Presets ───────────────────────────────────────────────────────
  getPreset: (lyricsSetId: number): Promise<{ preset: AlbumPreset | null }> =>
    api(`/api/lireek/lyrics-sets/${lyricsSetId}/preset`),

  upsertPreset: (lyricsSetId: number, params: {
    adapter_path?: string;
    adapter_scale?: number;
    adapter_group_scales?: { self_attn: number; cross_attn: number; mlp: number };
    matchering_reference_path?: string;
  }): Promise<{ preset: AlbumPreset }> =>
    api(`/api/lireek/lyrics-sets/${lyricsSetId}/preset`, { method: 'PUT', body: params }),

  deletePreset: (lyricsSetId: number): Promise<{ deleted: boolean }> =>
    api(`/api/lireek/lyrics-sets/${lyricsSetId}/preset`, { method: 'DELETE' }),

  // ── Slop Detection ──────────────────────────────────────────────────────
  slopScan: (text: string): Promise<any> =>
    api('/api/lireek/slop-scan', { method: 'POST', body: { text } }),

  // ── Bulk Operations ─────────────────────────────────────────────────────
  purgeAll: (): Promise<any> =>
    api('/api/lireek/purge', { method: 'POST' }),

  // ── Audio Generations ───────────────────────────────────────────────────
  linkAudio: (generationId: number, jobId: string): Promise<any> =>
    api(`/api/lireek/generations/${generationId}/audio`, { method: 'POST', body: { job_id: jobId } }),

  getAudioGenerations: (generationId: number): Promise<{ audio_generations: AudioGeneration[] }> =>
    api(`/api/lireek/generations/${generationId}/audio`),

  // ── Direct Audio Generation ─────────────────────────────────────────
  submitAudioGeneration: (params: {
    lyrics: string;
    prompt: string;
    bpm?: number;
    key_scale?: string;
    audio_duration?: number;
    lireek_adapter_path?: string;
    lireek_adapter_scale?: number;
    lireek_group_scales?: { self_attn: number; cross_attn: number; mlp: number };
    mastering_params?: { mode: string; reference_file?: string };
  }): Promise<{ job_id: string }> =>
    api('/api/generate', { method: 'POST', body: params }),



  // ── Prompts ───────────────────────────────────────────────────────────
  listPrompts: (): Promise<{ prompts: { name: string; source: string; content: string; has_default: boolean }[] }> =>
    api('/api/lireek/prompts'),

  savePrompt: (name: string, content: string): Promise<{ name: string; path: string; source: string }> =>
    api(`/api/lireek/prompts/${name}`, { method: 'PUT', body: { content } }),

  resetPrompt: (name: string): Promise<{ name: string; status: string }> =>
    api(`/api/lireek/prompts/${name}`, { method: 'DELETE' }),
};

// ── SSE Streaming ─────────────────────────────────────────────────────────

export interface StreamCallbacks {
  onChunk?: (text: string) => void;
  onPhase?: (phase: string) => void;
  onResult?: (data: any) => void;
  onError?: (message: string) => void;
}

async function consumeSSE(url: string, body: any, callbacks: StreamCallbacks): Promise<void> {
  const resp = await fetch(`${API_BASE}${url}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    credentials: 'include',
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `HTTP ${resp.status}`);
  }

  const reader = resp.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data: ')) continue;
      const jsonStr = trimmed.slice(6).trim();
      if (!jsonStr) continue;

      try {
        const event = JSON.parse(jsonStr);
        switch (event.type) {
          case 'chunk': callbacks.onChunk?.(event.text); break;
          case 'phase': callbacks.onPhase?.(event.text); break;
          case 'result': callbacks.onResult?.(event.data); break;
          case 'error': callbacks.onError?.(event.message); break;
        }
      } catch { /* skip malformed lines */ }
    }
  }
}

export const streamBuildProfile = (
  lyricsSetId: number,
  req: { provider: string; model?: string },
  callbacks: StreamCallbacks,
): Promise<void> =>
  consumeSSE(`/api/lireek/lyrics-sets/${lyricsSetId}/build-profile-stream`, req, callbacks);

export const streamGenerate = (
  profileId: number,
  req: { profile_id: number; provider: string; model?: string; extra_instructions?: string },
  callbacks: StreamCallbacks,
): Promise<void> =>
  consumeSSE(`/api/lireek/profiles/${profileId}/generate-stream`, req, callbacks);

export const streamRefine = (
  generationId: number,
  req: { provider: string; model?: string },
  callbacks: StreamCallbacks,
): Promise<void> =>
  consumeSSE(`/api/lireek/generations/${generationId}/refine-stream`, req, callbacks);

export const skipThinking = (): Promise<void> =>
  api('/api/lireek/skip-thinking', { method: 'POST' });

