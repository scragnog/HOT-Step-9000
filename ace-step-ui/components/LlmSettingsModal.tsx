import React, { useState, useEffect, useCallback } from 'react';
import { X, Brain, RefreshCw, Check, AlertCircle, Eye, EyeOff, ChevronDown } from 'lucide-react';
import { llmApi, type LlmProviderInfo } from '../services/api';

interface LlmSettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
}

// Setting key → display label mapping
const SETTING_LABELS: Record<string, { label: string; type: 'key' | 'url' | 'model' | 'text'; placeholder: string }> = {
    gemini_api_key: { label: 'Gemini API Key', type: 'key', placeholder: 'AIza...' },
    openai_api_key: { label: 'OpenAI API Key', type: 'key', placeholder: 'sk-...' },
    anthropic_api_key: { label: 'Anthropic API Key', type: 'key', placeholder: 'sk-ant-...' },
    ollama_base_url: { label: 'Ollama URL', type: 'url', placeholder: 'http://localhost:11434' },
    lmstudio_base_url: { label: 'LM Studio URL', type: 'url', placeholder: 'http://localhost:1234/v1' },
    unsloth_base_url: { label: 'Unsloth Studio URL', type: 'url', placeholder: 'http://127.0.0.1:8888' },
    unsloth_username: { label: 'Unsloth Username', type: 'text', placeholder: 'username' },
    unsloth_password: { label: 'Unsloth Password', type: 'key', placeholder: 'password' },
    genius_access_token: { label: 'Genius API Token', type: 'key', placeholder: 'Bearer token from genius.com' },
};

// Provider groups for the UI
const PROVIDER_GROUPS = [
    {
        title: 'Cloud Providers',
        providers: [
            { id: 'gemini', settingKeys: ['gemini_api_key'] },
            { id: 'openai', settingKeys: ['openai_api_key'] },
            { id: 'anthropic', settingKeys: ['anthropic_api_key'] },
        ],
    },
    {
        title: 'Local Providers',
        providers: [
            { id: 'ollama', settingKeys: ['ollama_base_url'] },
            { id: 'lmstudio', settingKeys: ['lmstudio_base_url'] },
            { id: 'unsloth', settingKeys: ['unsloth_base_url', 'unsloth_username', 'unsloth_password'] },
        ],
    },
    {
        title: 'External Services',
        providers: [
            { id: '_genius', settingKeys: ['genius_access_token'] },
        ],
    },
];

export const LlmSettingsModal: React.FC<LlmSettingsModalProps> = ({ isOpen, onClose }) => {
    const [providers, setProviders] = useState<LlmProviderInfo[]>([]);
    const [settings, setSettings] = useState<Record<string, string>>({});
    const [editedSettings, setEditedSettings] = useState<Record<string, string>>({});
    const [defaultProvider, setDefaultProvider] = useState('gemini');
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
    const [revealedKeys, setRevealedKeys] = useState<Set<string>>(new Set());

    const fetchData = useCallback(async () => {
        setLoading(true);
        try {
            const [provResult, settResult] = await Promise.all([
                llmApi.getProviders(),
                llmApi.getSettings(),
            ]);
            setProviders(provResult.providers);
            setSettings(settResult.settings);
            setEditedSettings({});
            setDefaultProvider(settResult.settings.default_llm_provider || 'gemini');
        } catch (err) {
            console.error('Failed to fetch LLM settings:', err);
        }
        setLoading(false);
    }, []);

    useEffect(() => {
        if (isOpen) {
            fetchData();
            setRevealedKeys(new Set());
        }
    }, [isOpen, fetchData]);

    const handleSettingChange = (key: string, value: string) => {
        setEditedSettings(prev => ({ ...prev, [key]: value }));
    };

    const handleSave = async () => {
        setSaving(true);
        setSaveMessage(null);
        try {
            const toSave = { ...editedSettings };
            if (defaultProvider !== (settings.default_llm_provider || 'gemini')) {
                toSave.default_llm_provider = defaultProvider;
            }
            if (Object.keys(toSave).length === 0) {
                setSaveMessage({ type: 'success', text: 'No changes to save' });
                setSaving(false);
                return;
            }
            const result = await llmApi.updateSettings(toSave);
            setSaveMessage({ type: 'success', text: `Updated ${result.count} setting${result.count !== 1 ? 's' : ''}` });
            // Refresh to get masked values
            await fetchData();
        } catch (err) {
            setSaveMessage({ type: 'error', text: 'Failed to save settings' });
        }
        setSaving(false);
        setTimeout(() => setSaveMessage(null), 3000);
    };

    const toggleReveal = (key: string) => {
        setRevealedKeys(prev => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key);
            else next.add(key);
            return next;
        });
    };

    const getProviderInfo = (id: string) => providers.find(p => p.id === id);

    const hasChanges = Object.keys(editedSettings).length > 0 ||
        defaultProvider !== (settings.default_llm_provider || 'gemini');

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/50 z-[60] flex items-center justify-center p-4" onClick={onClose}>
            <div
                className="bg-white dark:bg-zinc-900 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-zinc-200 dark:border-white/5">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-600 flex items-center justify-center">
                            <Brain size={20} className="text-white" />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-zinc-900 dark:text-white">LLM Providers</h2>
                            <p className="text-xs text-zinc-500 dark:text-zinc-400">Configure AI providers for lyrics generation</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={fetchData}
                            disabled={loading}
                            className="p-2 hover:bg-zinc-100 dark:hover:bg-white/5 rounded-lg transition-colors"
                            title="Refresh"
                        >
                            <RefreshCw size={18} className={`text-zinc-500 ${loading ? 'animate-spin' : ''}`} />
                        </button>
                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-zinc-100 dark:hover:bg-white/5 rounded-full transition-colors"
                        >
                            <X size={20} className="text-zinc-500" />
                        </button>
                    </div>
                </div>

                <div className="p-6 space-y-6">
                    {/* Default Provider Selector */}
                    <div className="space-y-2">
                        <label className="text-sm font-semibold text-zinc-900 dark:text-white">Default Provider</label>
                        <div className="relative">
                            <select
                                value={defaultProvider}
                                onChange={(e) => setDefaultProvider(e.target.value)}
                                className="w-full appearance-none py-3 px-4 pr-10 rounded-lg border-2 border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 text-zinc-900 dark:text-white font-medium transition-colors hover:border-zinc-400 dark:hover:border-zinc-600 focus:outline-none focus:border-violet-500"
                            >
                                {providers.map(p => (
                                    <option key={p.id} value={p.id}>
                                        {p.name} {p.available ? '✓' : '✗'}
                                    </option>
                                ))}
                            </select>
                            <ChevronDown size={20} className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 pointer-events-none" />
                        </div>
                    </div>

                    {/* Provider Groups */}
                    {PROVIDER_GROUPS.map(group => (
                        <div key={group.title} className="space-y-3">
                            <h3 className="text-sm font-semibold text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">
                                {group.title}
                            </h3>
                            <div className="space-y-3">
                                {group.providers.map(({ id, settingKeys }) => {
                                    const info = getProviderInfo(id);
                                    return (
                                        <div
                                            key={id}
                                            className="rounded-xl border border-zinc-200 dark:border-zinc-700/50 bg-zinc-50 dark:bg-zinc-800/50 p-4 space-y-3"
                                        >
                                            {/* Provider header */}
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center gap-2">
                                                    <span className="text-sm font-semibold text-zinc-900 dark:text-white">
                                                        {info?.name || id}
                                                    </span>
                                                    {info && (
                                                        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold ${
                                                            info.available
                                                                ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400'
                                                                : 'bg-zinc-200 dark:bg-zinc-700 text-zinc-500 dark:text-zinc-400'
                                                        }`}>
                                                            {info.available ? (
                                                                <><Check size={10} /> Online</>
                                                            ) : (
                                                                <><AlertCircle size={10} /> Offline</>
                                                            )}
                                                        </span>
                                                    )}
                                                </div>
                                                {info?.available && info.models.length > 0 && (
                                                    <span className="text-[10px] text-zinc-400 dark:text-zinc-500">
                                                        {info.models.length} model{info.models.length !== 1 ? 's' : ''}
                                                    </span>
                                                )}
                                            </div>
                                            {/* Settings fields */}
                                            {settingKeys.map(key => {
                                                const meta = SETTING_LABELS[key];
                                                if (!meta) return null;
                                                const currentValue = editedSettings[key] ?? settings[key] ?? '';
                                                const isSecret = meta.type === 'key';
                                                const isRevealed = revealedKeys.has(key);
                                                return (
                                                    <div key={key} className="space-y-1">
                                                        <label className="text-xs text-zinc-500 dark:text-zinc-400 font-medium">
                                                            {meta.label}
                                                        </label>
                                                        <div className="relative">
                                                            <input
                                                                type={isSecret && !isRevealed ? 'password' : 'text'}
                                                                value={currentValue}
                                                                onChange={(e) => handleSettingChange(key, e.target.value)}
                                                                placeholder={meta.placeholder}
                                                                className="w-full py-2 px-3 pr-10 rounded-lg border border-zinc-300 dark:border-zinc-600 bg-white dark:bg-zinc-800 text-zinc-900 dark:text-white text-sm font-mono focus:outline-none focus:border-violet-500 transition-colors placeholder:text-zinc-400 dark:placeholder:text-zinc-600"
                                                            />
                                                            {isSecret && (
                                                                <button
                                                                    type="button"
                                                                    onClick={() => toggleReveal(key)}
                                                                    className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300 transition-colors"
                                                                >
                                                                    {isRevealed ? <EyeOff size={14} /> : <Eye size={14} />}
                                                                </button>
                                                            )}
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Footer */}
                <div className="border-t border-zinc-200 dark:border-white/5 p-6 flex items-center justify-between">
                    <div className="min-h-[24px]">
                        {saveMessage && (
                            <p className={`text-sm font-medium ${
                                saveMessage.type === 'success' ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'
                            }`}>
                                {saveMessage.text}
                            </p>
                        )}
                    </div>
                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            className="px-5 py-2 text-sm font-medium text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleSave}
                            disabled={saving || !hasChanges}
                            className={`px-6 py-2 rounded-lg text-sm font-semibold transition-all ${
                                hasChanges
                                    ? 'bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white hover:from-violet-700 hover:to-fuchsia-700 shadow-lg shadow-violet-500/20'
                                    : 'bg-zinc-200 dark:bg-zinc-700 text-zinc-400 dark:text-zinc-500 cursor-not-allowed'
                            }`}
                        >
                            {saving ? 'Saving...' : 'Save Changes'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};
