/**
 * useAudioGeneration.ts — Encapsulates the full audio generation flow
 * for Lyric Studio V2.
 *
 * Extracted from V1's handleGenerateAudio + handleSendToCreate in LyricStudio.tsx.
 * Handles: preset loading → param merging → adapter loading → trigger word →
 * matchering → generation → audio linking.
 */

import { useCallback } from 'react';
import { lireekApi, Generation, Profile, AlbumPreset } from '../../../services/lyricStudioApi';
import { generateApi } from '../../../services/api';
import { useAuth } from '../../../context/AuthContext';

// ── Helpers ──────────────────────────────────────────────────────────────────

function readPersisted(key: string): any {
  try {
    const raw = localStorage.getItem(key);
    return raw !== null ? JSON.parse(raw) : undefined;
  } catch { return undefined; }
}

function mergeCreatePanelSettings(params: Record<string, any>): void {
  // Generation engine settings — stored by CreatePanel with 'ace-' prefix
  const map: [string, string][] = [
    ['ace-inferenceSteps', 'inferenceSteps'],
    ['ace-inferMethod', 'inferMethod'],
    ['ace-scheduler', 'scheduler'],
    ['ace-guidanceScale', 'guidanceScale'],
    ['ace-shift', 'shift'],
    ['ace-guidanceMode', 'guidanceMode'],
    ['ace-audioFormat', 'audioFormat'],
    ['ace-randomSeed', 'randomSeed'],
    ['ace-enableNormalization', 'enableNormalization'],
    ['ace-normalizationDb', 'normalizationDb'],
    ['ace-vocalLanguage', 'vocalLanguage'],
    ['ace-vocalGender', 'vocalGender'],
    // LM settings
    ['ace-lmTemperature', 'lmTemperature'],
    ['ace-lmCfgScale', 'lmCfgScale'],
    ['ace-lmTopK', 'lmTopK'],
    ['ace-lmTopP', 'lmTopP'],
    ['ace-lmModel', 'lmModel'],
    // Advanced guidance
    ['ace-useCotMetas', 'useCotMetas'],
    ['ace-useCotCaption', 'useCotCaption'],
    ['ace-useCotLanguage', 'useCotLanguage'],
    // Vocoder
    ['ace-vocoderModel', 'vocoderModel'],
    // Thinking
    ['ace-thinking', 'thinking'],
  ];
  for (const [storageKey, paramKey] of map) {
    const val = readPersisted(storageKey);
    if (val !== undefined && val !== null && val !== '') {
      params[paramKey] = val;
    }
  }
  // Always enable lyric sync
  params.getLrc = true;
  // Cover art: global setting
  params.generateCoverArt = localStorage.getItem('generate_cover_art') === 'true';
}

/** Read global adapter scale override settings (shared with CreatePanel). */
function getGlobalScaleOverride(): {
  enabled: boolean;
  overallScale: number;
  groupScales: { self_attn: number; cross_attn: number; mlp: number };
} {
  try {
    const enabled = JSON.parse(localStorage.getItem('ace-globalScaleOverride') || 'false');
    const overallScale = JSON.parse(localStorage.getItem('ace-globalOverallScale') || '1.0');
    const groupScales = JSON.parse(localStorage.getItem('ace-globalGroupScales') || 'null') || { self_attn: 1.0, cross_attn: 1.0, mlp: 1.0 };
    return { enabled: !!enabled, overallScale, groupScales };
  } catch {
    return { enabled: false, overallScale: 1.0, groupScales: { self_attn: 1.0, cross_attn: 1.0, mlp: 1.0 } };
  }
}

function applyTriggerWord(params: Record<string, any>, adapterPath: string): void {
  const useFilename = localStorage.getItem('ace-globalTriggerUseFilename') === 'true';
  const placement = (localStorage.getItem('ace-globalTriggerPlacement') as 'prepend' | 'append' | 'replace') || 'prepend';

  if (!useFilename) return;

  const fileName = adapterPath.replace(/\\/g, '/').split('/').pop() || '';
  const triggerWord = fileName.replace(/\.safetensors$/i, '');
  if (!triggerWord) return;

  const current = ((params.style as string) || '').trim();
  if (current.toLowerCase().includes(triggerWord.toLowerCase())) return;

  if (placement === 'replace') {
    params.style = triggerWord;
  } else if (placement === 'append') {
    params.style = current ? `${current}, ${triggerWord}` : triggerWord;
  } else {
    params.style = current ? `${triggerWord}, ${current}` : triggerWord;
  }
  console.log(`[LyricStudioV2] Trigger word '${triggerWord}' ${placement}ed → '${params.style}'`);
}

// ── Hook ─────────────────────────────────────────────────────────────────────

interface UseAudioGenerationOptions {
  profiles: Profile[];
  showToast: (msg: string) => void;
  onJobLinked?: (generationId: number, jobId: string) => void;
}

export function useAudioGeneration({ profiles, showToast, onJobLinked }: UseAudioGenerationOptions) {
  const { token } = useAuth();

  /**
   * Full audio generation flow: load preset → build params → load adapter →
   * trigger word → matchering → start gen → link audio.
   */
  const generateAudio = useCallback(async (gen: Generation): Promise<string | null> => {
    if (!token) { showToast('Not authenticated'); return null; }

    try {
      // 1) Find album preset for this generation
      const profile = profiles.find(p => p.id === gen.profile_id);
      let preset: AlbumPreset | null = null;
      if (profile) {
        const res = await lireekApi.getPreset(profile.lyrics_set_id);
        preset = res.preset;
      }

      // 2) Build base params
      const params: Record<string, any> = {
        customMode: true,
        lyrics: gen.lyrics || '',
        style: gen.caption || '',
        title: gen.title || '',
        instrumental: false,
        duration: gen.duration || 180,
      };
      if (gen.bpm) params.bpm = gen.bpm;
      if (gen.subject) params.coverArtSubject = gen.subject;
      if (gen.key) params.keyScale = gen.key;

      // 3) Merge persisted CreatePanel settings
      mergeCreatePanelSettings(params);

      // 4) Load adapter from album preset
      if (preset?.adapter_path) {
        // Check for global scale override (shared with CreatePanel)
        const scaleOverride = getGlobalScaleOverride();
        const effectiveScale = scaleOverride.enabled ? scaleOverride.overallScale : (preset.adapter_scale ?? 1.0);
        const effectiveGroupScales = scaleOverride.enabled ? scaleOverride.groupScales : preset.adapter_group_scales;

        if (scaleOverride.enabled) {
          console.log('[LyricStudioV2] Global scale override active — overallScale:', effectiveScale, 'groupScales:', JSON.stringify(effectiveGroupScales));
        }

        try {
          const loraStatus = await generateApi.getLoraStatus(token);
          const existingSlot = loraStatus?.advanced?.slots?.find(
            (s: any) => s.path === preset!.adapter_path
          );

          if (existingSlot) {
            console.log('[LyricStudioV2] Adapter already loaded, skipping reload');
            params.loraLoaded = true;
            params.loraPath = preset.adapter_path;
            params.loraScale = effectiveScale;
            // Apply group scales if they differ
            if (effectiveGroupScales) {
              console.log('[LyricStudioV2] Applying group scales to already-loaded adapter:', JSON.stringify(effectiveGroupScales), 'slot:', existingSlot.slot);
              try {
                await generateApi.setSlotGroupScales({
                  slot: existingSlot.slot,
                  ...effectiveGroupScales,
                }, token);
                console.log('[LyricStudioV2] Group scales applied successfully');
              } catch (gsErr) {
                console.warn('[LyricStudioV2] Failed to apply group scales:', gsErr);
              }
            } else {
              console.log('[LyricStudioV2] No adapter_group_scales in preset, skipping group scale application');
            }
          } else {
            // Unload existing adapters first
            if (loraStatus?.advanced?.slots && loraStatus.advanced.slots.length > 0) {
              showToast('Switching adapter...');
              await generateApi.unloadLora(token);
            } else {
              showToast('Loading adapter...');
            }
            const loadPayload: any = {
              lora_path: preset.adapter_path,
              slot: 0,  // Use advanced adapter slot so group_scales are applied
              scale: effectiveScale,
              ...(effectiveGroupScales ? { group_scales: effectiveGroupScales } : {}),
            };
            console.log('[LyricStudioV2] Loading adapter with payload:', JSON.stringify(loadPayload));
            await generateApi.loadLora(loadPayload, token);
            params.loraLoaded = true;
            params.loraPath = preset.adapter_path;
            params.loraScale = effectiveScale;
          }
        } catch (loadErr) {
          console.warn('[LyricStudioV2] Failed to load adapter, continuing without:', loadErr);
          showToast('Warning: adapter failed to load, generating without');
        }

        // 5) Trigger word
        applyTriggerWord(params, preset.adapter_path);
      }

      // 6) Reference Track — dual purpose: timbre conditioning + matchering
      if (preset?.reference_track_path) {
        // a) ACE-Step timbre conditioning (guides DiT toward target acoustics)
        params.referenceAudioUrl = preset.reference_track_path;
        params.audioCoverStrength = preset.audio_cover_strength ?? 0.5;
        // b) Post-processing matchering (polishes EQ/loudness)
        params.autoMaster = true;
        params.masteringParams = { mode: 'matchering', reference_file: preset.reference_track_path };
      }

      // 7) Mark as Lyric Studio generation
      params.source = 'lyric-studio';

      // 8) Start generation
      console.log(`[LyricStudioV2] Starting generation with coverArtSubject: "${params.coverArtSubject || '(empty)'}", title: "${params.title}"`);
      const res = await generateApi.startGeneration(params as any, token);
      const jobId = res.jobId || (res as any).job_id;
      showToast(`Audio job queued: ${jobId}`);

      // 8) Link audio to lyric generation
      if (jobId) {
        await lireekApi.linkAudio(gen.id, jobId);
        onJobLinked?.(gen.id, jobId);
      }

      return jobId;
    } catch (err) {
      showToast(`Audio generation failed: ${(err as Error).message}`);
      return null;
    }
  }, [token, profiles, showToast, onJobLinked]);

  /**
   * Prepare generation data and send to CreatePanel for manual tweaking.
   */
  const sendToCreate = useCallback(async (gen: Generation): Promise<void> => {
    const profile = profiles.find(p => p.id === gen.profile_id);
    let preset: AlbumPreset | null = null;
    if (profile) {
      try {
        const res = await lireekApi.getPreset(profile.lyrics_set_id);
        preset = res.preset;
      } catch { /* ignore */ }
    }

    const importData: Record<string, any> = {
      title: gen.title || '',
      prompt: gen.lyrics || '',
      style: gen.caption || '',
      instrumental: false,
    };
    if (gen.bpm) importData.bpm = gen.bpm;
    if (gen.key) importData.keyScale = gen.key;
    if (gen.duration) importData.duration = gen.duration;
    if (preset?.adapter_path) {
      importData.loraPath = preset.adapter_path;
      importData.loraScale = preset.adapter_scale ?? 1.0;
    }
    if (preset?.reference_track_path) {
      importData.autoMaster = true;
      importData.masteringParams = { mode: 'matchering', reference_file: preset.reference_track_path };
    }

    localStorage.setItem('hotstep_lireek_import', JSON.stringify(importData));
    window.history.pushState({}, '', '/create');
    window.dispatchEvent(new PopStateEvent('popstate'));
  }, [profiles]);

  return { generateAudio, sendToCreate };
}
