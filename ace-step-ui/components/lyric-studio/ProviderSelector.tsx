import React, { useState, useEffect, useRef } from 'react';
import { llmApi, LlmProviderInfo } from '../../services/api';

// ── Global provider cache ─────────────────────────────────────────────────
// Shared across all ProviderSelector instances — only fetched once per session
let _providerCache: LlmProviderInfo[] | null = null;
let _providerCacheTime = 0;
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

async function getCachedProviders(forceRefresh = false): Promise<LlmProviderInfo[]> {
  const now = Date.now();
  if (!forceRefresh && _providerCache && (now - _providerCacheTime) < CACHE_TTL_MS) {
    return _providerCache;
  }
  const res = await llmApi.getProviders();
  _providerCache = res.providers.filter(p => p.available);
  _providerCacheTime = now;
  return _providerCache;
}

// ── localStorage persistence for model selections ─────────────────────────
const STORAGE_KEY = 'lireek-model-selections';

export interface ModelSelections {
  profiling: { provider: string; model: string };
  generation: { provider: string; model: string };
  refinement: { provider: string; model: string };
}

function loadSelections(): ModelSelections {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return {
    profiling: { provider: 'gemini', model: '' },
    generation: { provider: 'gemini', model: '' },
    refinement: { provider: 'gemini', model: '' },
  };
}

function saveSelections(sel: ModelSelections) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sel));
}

// ── Single row selector ──────────────────────────────────────────────────
const RowSelector: React.FC<{
  label: string;
  color: string;
  providers: LlmProviderInfo[];
  selectedProvider: string;
  selectedModel: string;
  onProviderChange: (p: string) => void;
  onModelChange: (m: string) => void;
}> = ({ label, color, providers, selectedProvider, selectedModel, onProviderChange, onModelChange }) => {
  const currentProvider = providers.find(p => p.id === selectedProvider);
  const models = currentProvider?.models || [];

  return (
    <div className="flex items-center gap-1.5">
      <span className={`text-[10px] font-semibold uppercase tracking-wider w-[52px] flex-shrink-0 ${color}`}>{label}</span>
      <select
        value={selectedProvider}
        onChange={e => {
          const pid = e.target.value;
          onProviderChange(pid);
          const prov = providers.find(p => p.id === pid);
          if (prov?.default_model) onModelChange(prov.default_model);
        }}
        className="flex-[0.8] min-w-0 px-1.5 py-1 rounded bg-zinc-800 border border-white/10 text-[11px] text-white focus:outline-none focus:border-pink-500/50 appearance-none cursor-pointer"
        title={`${label} Provider`}
      >
        {providers.map(p => (
          <option key={p.id} value={p.id}>{p.name}</option>
        ))}
      </select>
      <select
        value={selectedModel}
        onChange={e => onModelChange(e.target.value)}
        className="flex-1 min-w-0 px-1.5 py-1 rounded bg-zinc-800 border border-white/10 text-[11px] text-white focus:outline-none focus:border-pink-500/50 appearance-none cursor-pointer"
        title={`${label} Model`}
      >
        {models.map(m => (
          <option key={m} value={m}>{m}</option>
        ))}
        {models.length === 0 && <option value="">No models</option>}
      </select>
    </div>
  );
};

// ── Triple Provider Selector ──────────────────────────────────────────────
interface TripleProviderSelectorProps {
  selections: ModelSelections;
  onSelectionsChange: (sel: ModelSelections) => void;
}

export const TripleProviderSelector: React.FC<TripleProviderSelectorProps> = ({
  selections,
  onSelectionsChange,
}) => {
  const [providers, setProviders] = useState<LlmProviderInfo[]>(_providerCache || []);
  const [loading, setLoading] = useState(!_providerCache);

  useEffect(() => {
    getCachedProviders()
      .then(p => {
        setProviders(p);
        // Auto-select first available provider for any unset slots
        if (p.length > 0) {
          const first = p[0];
          const updated = { ...selections };
          let changed = false;
          for (const role of ['profiling', 'generation', 'refinement'] as const) {
            if (!updated[role].provider || !p.find(pp => pp.id === updated[role].provider)) {
              updated[role] = { provider: first.id, model: first.default_model || '' };
              changed = true;
            }
          }
          if (changed) onSelectionsChange(updated);
        }
      })
      .catch(err => console.error('Failed to load LLM providers:', err))
      .finally(() => setLoading(false));
  }, []);

  const update = (role: keyof ModelSelections, field: 'provider' | 'model', value: string) => {
    const updated = {
      ...selections,
      [role]: { ...selections[role], [field]: value },
    };
    onSelectionsChange(updated);
    saveSelections(updated);
  };

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-xs text-zinc-500 py-1">
        <div className="w-3 h-3 border-2 border-zinc-500 border-t-transparent rounded-full animate-spin" />
        Loading providers…
      </div>
    );
  }

  if (providers.length === 0) {
    return (
      <div className="text-xs text-amber-400 py-1">
        ⚠ No LLM providers configured.
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      <RowSelector
        label="Profile"
        color="text-amber-400"
        providers={providers}
        selectedProvider={selections.profiling.provider}
        selectedModel={selections.profiling.model}
        onProviderChange={v => update('profiling', 'provider', v)}
        onModelChange={v => update('profiling', 'model', v)}
      />
      <RowSelector
        label="Generate"
        color="text-green-400"
        providers={providers}
        selectedProvider={selections.generation.provider}
        selectedModel={selections.generation.model}
        onProviderChange={v => update('generation', 'provider', v)}
        onModelChange={v => update('generation', 'model', v)}
      />
      <RowSelector
        label="Refine"
        color="text-purple-400"
        providers={providers}
        selectedProvider={selections.refinement.provider}
        selectedModel={selections.refinement.model}
        onProviderChange={v => update('refinement', 'provider', v)}
        onModelChange={v => update('refinement', 'model', v)}
      />
    </div>
  );
};

// ── Legacy single selector (kept for backward compat) ─────────────────────
interface ProviderSelectorProps {
  selectedProvider: string;
  selectedModel: string;
  onProviderChange: (provider: string) => void;
  onModelChange: (model: string) => void;
  label?: string;
  compact?: boolean;
}

export const ProviderSelector: React.FC<ProviderSelectorProps> = ({
  selectedProvider,
  selectedModel,
  onProviderChange,
  onModelChange,
  label = 'LLM Provider',
  compact = false,
}) => {
  const [providers, setProviders] = useState<LlmProviderInfo[]>(_providerCache || []);
  const [loading, setLoading] = useState(!_providerCache);

  useEffect(() => {
    getCachedProviders()
      .then(p => {
        setProviders(p);
        if (!selectedProvider && p.length > 0) {
          const first = p[0];
          onProviderChange(first.id);
          if (first.default_model) onModelChange(first.default_model);
        }
      })
      .catch(err => console.error('Failed to load LLM providers:', err))
      .finally(() => setLoading(false));
  }, []);

  const currentProvider = providers.find(p => p.id === selectedProvider);
  const models = currentProvider?.models || [];

  if (loading) {
    return (
      <div className={`flex items-center gap-2 text-xs text-zinc-500 ${compact ? '' : 'mb-3'}`}>
        <div className="w-3 h-3 border-2 border-zinc-500 border-t-transparent rounded-full animate-spin" />
        Loading providers…
      </div>
    );
  }

  if (providers.length === 0) {
    return (
      <div className={`text-xs text-amber-400 ${compact ? '' : 'mb-3'}`}>
        ⚠ No LLM providers configured. Open Settings → LLM to add one.
      </div>
    );
  }

  return (
    <div className={`flex ${compact ? 'flex-row items-center gap-2' : 'flex-col gap-2'}`}>
      {!compact && <label className="text-xs font-medium text-zinc-400 uppercase tracking-wider">{label}</label>}
      <div className={`flex ${compact ? 'flex-row' : 'flex-row'} gap-2 flex-1`}>
        <select
          value={selectedProvider}
          onChange={e => {
            const pid = e.target.value;
            onProviderChange(pid);
            const prov = providers.find(p => p.id === pid);
            if (prov?.default_model) onModelChange(prov.default_model);
          }}
          className="flex-1 px-2.5 py-1.5 rounded-lg bg-zinc-800 border border-white/10 text-sm text-white focus:outline-none focus:border-pink-500/50 appearance-none cursor-pointer"
          title="LLM Provider"
        >
          {providers.map(p => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>
        <select
          value={selectedModel}
          onChange={e => onModelChange(e.target.value)}
          className="flex-1 px-2.5 py-1.5 rounded-lg bg-zinc-800 border border-white/10 text-sm text-white focus:outline-none focus:border-pink-500/50 appearance-none cursor-pointer"
          title="Model"
        >
          {models.map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
          {models.length === 0 && <option value="">No models</option>}
        </select>
      </div>
    </div>
  );
};

export { loadSelections, saveSelections };
