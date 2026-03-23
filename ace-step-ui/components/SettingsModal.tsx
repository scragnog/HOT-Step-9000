import React, { useState } from 'react';
import { X, User as UserIcon, Palette, Info, Edit3, ExternalLink, Globe, ChevronDown, Github, Save, Activity, Sliders, Download } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useI18n } from '../context/I18nContext';
import { EditProfileModal } from './EditProfileModal';
import { isPersistenceEnabled, setPersistenceEnabled, clearPersistedSettings } from '../hooks/usePersistedState';

interface SettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
    theme: 'light' | 'dark';
    onToggleTheme: () => void;
    onNavigateToProfile?: (username: string) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, theme, onToggleTheme, onNavigateToProfile }) => {
    const { user } = useAuth();
    const { t, language, setLanguage } = useI18n();
    const [isEditProfileOpen, setIsEditProfileOpen] = useState(false);
    const [persistEnabled, setPersistEnabled] = useState(isPersistenceEnabled);
    const [hqUpscaleSteps, setHqUpscaleSteps] = useState(() => {
        const saved = localStorage.getItem('hq_upscale_steps');
        return saved !== null ? parseInt(saved, 10) : 160;
    });
    const [quickPresetSteps, setQuickPresetSteps] = useState(() => {
        const saved = localStorage.getItem('quick_preset_steps');
        return saved !== null ? parseInt(saved, 10) : 12;
    });
    const [hqPresetSteps, setHqPresetSteps] = useState(() => {
        const saved = localStorage.getItem('hq_preset_steps');
        return saved !== null ? parseInt(saved, 10) : 60;
    });
    const [visualizerBg, setVisualizerBg] = useState(() => localStorage.getItem('visualizer_songlist_bg') === 'true');
    const [bounceIntensity, setBounceIntensity] = useState(() => {
        const saved = localStorage.getItem('waveform-bounce-intensity');
        return saved !== null ? parseFloat(saved) : 0.5;
    });
    const [enabledPresets, setEnabledPresets] = useState<string[]>(() => {
        try {
            const saved = localStorage.getItem('visualizer_enabled_presets');
            if (saved) return JSON.parse(saved);
        } catch { }
        return ['NCS Circle', 'Linear Bars', 'Dual Mirror', 'Oscilloscope'];
    });
    const [mp3Bitrate, setMp3Bitrate] = useState(() => localStorage.getItem('mp3_export_bitrate') || 'V0');
    const [opusBitrate, setOpusBitrate] = useState(() => localStorage.getItem('opus_export_bitrate') || '128');
    const [generateCoverArt, setGenerateCoverArt] = useState(() => localStorage.getItem('generate_cover_art') === 'true');

    if (!isOpen || !user) {
        if (isEditProfileOpen && user) {
            return (
                <EditProfileModal
                    isOpen={isEditProfileOpen}
                    onClose={() => setIsEditProfileOpen(false)}
                    onSaved={() => setIsEditProfileOpen(false)}
                />
            );
        }
        return null;
    }

    return (
        <div className="fixed inset-0 bg-black/50 z-[60] flex items-center justify-center p-4" onClick={onClose}>
            <div
                className="bg-white dark:bg-zinc-900 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-zinc-200 dark:border-white/5">
                    <h2 className="text-2xl font-bold text-zinc-900 dark:text-white">{t('settings')}</h2>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-zinc-100 dark:hover:bg-white/5 rounded-full transition-colors"
                    >
                        <X size={20} className="text-zinc-500" />
                    </button>
                </div>

                <div className="p-6 space-y-8">
                    {/* User Profile Section */}
                    <div className="bg-zinc-50 dark:bg-zinc-800/50 rounded-xl p-6">
                        <div className="flex items-center gap-4">
                            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-2xl font-bold text-white shadow-lg overflow-hidden">
                                {user.avatar_url ? (
                                    <img src={user.avatar_url} alt={user.username} className="w-full h-full object-cover" />
                                ) : (
                                    user.username[0].toUpperCase()
                                )}
                            </div>
                            <div className="flex-1">
                                <h3 className="text-xl font-bold text-zinc-900 dark:text-white">@{user.username}</h3>
                                <p className="text-xs text-zinc-400 dark:text-zinc-500 mt-1">
                                    {t('memberSince')} {new Date(user.createdAt).toLocaleDateString(language === 'zh' ? 'zh-CN' : 'en-US', { month: 'long', year: 'numeric' })}
                                </p>
                            </div>
                            <div className="flex gap-2">
                                <button
                                    onClick={() => {
                                        onClose();
                                        setIsEditProfileOpen(true);
                                    }}
                                    className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 transition-colors"
                                >
                                    <Edit3 size={16} />
                                    {t('editProfile')}
                                </button>
                                <button
                                    onClick={() => {
                                        onClose();
                                        onNavigateToProfile?.(user.username);
                                    }}
                                    className="flex items-center gap-2 px-4 py-2 bg-zinc-200 dark:bg-zinc-700 text-zinc-900 dark:text-white rounded-lg text-sm font-medium hover:bg-zinc-300 dark:hover:bg-zinc-600 transition-colors"
                                >
                                    <ExternalLink size={16} />
                                    {t('viewProfile')}
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Account Section */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 text-zinc-900 dark:text-white">
                            <UserIcon size={20} />
                            <h3 className="font-semibold">{t('account')}</h3>
                        </div>
                        <div className="pl-7 space-y-3">
                            <div>
                                <label className="text-sm text-zinc-500 dark:text-zinc-400">{t('username')}</label>
                                <p className="text-zinc-900 dark:text-white font-medium">@{user.username}</p>
                            </div>
                        </div>
                    </div>

                    {/* Language Section */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 text-zinc-900 dark:text-white">
                            <Globe size={20} />
                            <h3 className="font-semibold">{t('language')}</h3>
                        </div>
                        <div className="pl-7 space-y-3">
                            <div className="relative">
                                <select
                                    value={language}
                                    onChange={(e) => setLanguage(e.target.value as 'en' | 'zh' | 'ja' | 'ko')}
                                    className="w-full appearance-none py-3 px-4 pr-10 rounded-lg border-2 border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 text-zinc-900 dark:text-white font-medium transition-colors hover:border-zinc-400 dark:hover:border-zinc-600 focus:outline-none focus:border-indigo-500 dark:focus:border-indigo-500 cursor-pointer"
                                >
                                    <option value="en">{t('english')}</option>
                                    <option value="zh">{t('chinese')}</option>
                                    <option value="ja">{t('japaneseLanguage')}</option>
                                    <option value="ko">{t('koreanLanguage')}</option>
                                </select>
                                <ChevronDown
                                    size={20}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 pointer-events-none"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Theme Section */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 text-zinc-900 dark:text-white">
                            <Palette size={20} />
                            <h3 className="font-semibold">{t('appearance')}</h3>
                        </div>
                        <div className="pl-7 space-y-3">
                            <div className="flex gap-3">
                                <button
                                    onClick={theme === 'dark' ? onToggleTheme : undefined}
                                    className={`flex-1 py-3 px-4 rounded-lg border-2 font-medium transition-colors ${theme === 'light'
                                        ? 'border-indigo-500 bg-indigo-50 text-indigo-700'
                                        : 'border-zinc-300 dark:border-zinc-700 hover:border-zinc-400 dark:hover:border-zinc-600'
                                        }`}
                                >
                                    {t('light')}
                                </button>
                                <button
                                    onClick={theme === 'light' ? onToggleTheme : undefined}
                                    className={`flex-1 py-3 px-4 rounded-lg border-2 font-medium transition-colors ${theme === 'dark'
                                        ? 'border-indigo-500 bg-indigo-950 text-indigo-300'
                                        : 'border-zinc-300 dark:border-zinc-700 hover:border-zinc-400 dark:hover:border-zinc-600'
                                        }`}
                                >
                                    {t('dark')}
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Visualizer Section */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 text-zinc-900 dark:text-white">
                            <Activity size={20} />
                            <h3 className="font-semibold">Visualizer</h3>
                        </div>
                        <div className="pl-7 space-y-3">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-zinc-900 dark:text-white font-medium">Song list background</p>
                                    <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">Show a dimmed visualizer behind tracks while music plays</p>
                                </div>
                                <button
                                    onClick={() => {
                                        const next = !visualizerBg;
                                        setVisualizerBg(next);
                                        localStorage.setItem('visualizer_songlist_bg', String(next));
                                    }}
                                    className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${visualizerBg ? 'bg-pink-600' : 'bg-zinc-300 dark:bg-zinc-600'}`}
                                >
                                    <span className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform duration-200 ${visualizerBg ? 'translate-x-5' : 'translate-x-0'}`} />
                                </button>
                            </div>
                            {/* Waveform Bounce Intensity */}
                            <div className="space-y-2">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm text-zinc-900 dark:text-white font-medium">Waveform bass bounce</p>
                                        <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">Bars pulse with the bass — crank it up!</p>
                                    </div>
                                    <span className="text-xs font-mono text-zinc-500 dark:text-zinc-400 min-w-[60px] text-right">
                                        {bounceIntensity === 0 ? 'Off' : bounceIntensity <= 0.3 ? 'Subtle' : bounceIntensity <= 0.6 ? 'Medium' : bounceIntensity <= 0.8 ? 'Strong' : 'Maximum'}
                                    </span>
                                </div>
                                <input
                                    type="range"
                                    min="0"
                                    max="1"
                                    step="0.05"
                                    value={bounceIntensity}
                                    onChange={(e) => {
                                        const val = parseFloat(e.target.value);
                                        setBounceIntensity(val);
                                        const key = 'waveform-bounce-intensity';
                                        localStorage.setItem(key, String(val));
                                        // Dispatch synthetic event so Player picks it up in same tab
                                        window.dispatchEvent(new StorageEvent('storage', { key, newValue: String(val) }));
                                    }}
                                    className="w-full h-2 bg-zinc-200 dark:bg-zinc-700 rounded-full appearance-none cursor-pointer accent-pink-600"
                                />
                                <div className="flex justify-between text-[10px] text-zinc-400 dark:text-zinc-500">
                                    <span>Off</span>
                                    <span>Maximum</span>
                                </div>
                            </div>
                            {/* Visualizer Preset Rotation Pool */}
                            <div className="space-y-2 pt-2 border-t border-zinc-200 dark:border-zinc-700/50">
                                <div>
                                    <p className="text-sm text-zinc-900 dark:text-white font-medium">Visualizer presets</p>
                                    <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">Choose which styles cycle in random mode</p>
                                </div>
                                <div className="grid grid-cols-2 gap-1.5">
                                    {(['NCS Circle', 'Linear Bars', 'Dual Mirror', 'Center Wave', 'Orbital', 'Hexagon', 'Oscilloscope', 'Digital Rain', 'Shockwave', 'Minimal'] as const).map(preset => {
                                        const labels: Record<string, string> = {
                                            'NCS Circle': '🔵 Classic NCS',
                                            'Linear Bars': '📊 Spectrum',
                                            'Dual Mirror': '🪞 Mirror',
                                            'Center Wave': '🌊 Shockwave',
                                            'Orbital': '🪐 Orbital',
                                            'Hexagon': '⬡ Hex Core',
                                            'Oscilloscope': '📈 Analog',
                                            'Digital Rain': '🟢 Matrix',
                                            'Shockwave': '💥 Pulse',
                                            'Minimal': '✨ Clean',
                                        };
                                        const enabled = enabledPresets.includes(preset);
                                        return (
                                            <button
                                                key={preset}
                                                onClick={() => {
                                                    let next: string[];
                                                    if (enabled) {
                                                        next = enabledPresets.filter((p: string) => p !== preset);
                                                        if (next.length === 0) return; // must keep at least one
                                                    } else {
                                                        next = [...enabledPresets, preset];
                                                    }
                                                    setEnabledPresets(next);
                                                    localStorage.setItem('visualizer_enabled_presets', JSON.stringify(next));
                                                    window.dispatchEvent(new StorageEvent('storage', { key: 'visualizer_enabled_presets', newValue: JSON.stringify(next) }));
                                                }}
                                                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-all border ${enabled
                                                    ? 'bg-pink-500/10 text-pink-600 dark:text-pink-400 border-pink-500/30'
                                                    : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500 dark:text-zinc-400 border-zinc-200 dark:border-zinc-700 hover:border-zinc-300 dark:hover:border-zinc-600'
                                                    }`}
                                            >
                                                <span className={`w-3.5 h-3.5 rounded border flex items-center justify-center flex-shrink-0 ${enabled ? 'bg-pink-600 border-pink-600' : 'border-zinc-300 dark:border-zinc-600'
                                                    }`}>
                                                    {enabled && <span className="text-white text-[8px] font-black">✓</span>}
                                                </span>
                                                <span className="truncate">{labels[preset] || preset}</span>
                                            </button>
                                        );
                                    })}
                                </div>
                                <div className="flex gap-2 pt-1">
                                    <button
                                        onClick={() => {
                                            const all = ['NCS Circle', 'Linear Bars', 'Dual Mirror', 'Center Wave', 'Orbital', 'Hexagon', 'Oscilloscope', 'Digital Rain', 'Shockwave', 'Minimal'];
                                            setEnabledPresets(all);
                                            localStorage.setItem('visualizer_enabled_presets', JSON.stringify(all));
                                            window.dispatchEvent(new StorageEvent('storage', { key: 'visualizer_enabled_presets', newValue: JSON.stringify(all) }));
                                        }}
                                        className="text-[10px] font-semibold text-zinc-500 dark:text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 transition-colors"
                                    >
                                        Select All
                                    </button>
                                    <span className="text-zinc-300 dark:text-zinc-600 text-[10px]">|</span>
                                    <button
                                        onClick={() => {
                                            // Keep only one — the first enabled one
                                            const first = enabledPresets[0] || 'NCS Circle';
                                            setEnabledPresets([first]);
                                            localStorage.setItem('visualizer_enabled_presets', JSON.stringify([first]));
                                            window.dispatchEvent(new StorageEvent('storage', { key: 'visualizer_enabled_presets', newValue: JSON.stringify([first]) }));
                                        }}
                                        className="text-[10px] font-semibold text-zinc-500 dark:text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 transition-colors"
                                    >
                                        Clear
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Persistent Settings Section */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 text-zinc-900 dark:text-white">
                            <Save size={20} />
                            <h3 className="font-semibold">Persistent Settings</h3>
                        </div>
                        <div className="pl-7 space-y-3">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-zinc-900 dark:text-white font-medium">Remember my settings</p>
                                    <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">Save generation parameters across sessions</p>
                                </div>
                                <button
                                    onClick={() => {
                                        const next = !persistEnabled;
                                        setPersistenceEnabled(next);
                                        setPersistEnabled(next);
                                        if (!next) {
                                            // Reload so all components re-init with defaults
                                            window.location.reload();
                                        }
                                    }}
                                    className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${persistEnabled ? 'bg-indigo-600' : 'bg-zinc-300 dark:bg-zinc-600'
                                        }`}
                                >
                                    <span className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform duration-200 ${persistEnabled ? 'translate-x-5' : 'translate-x-0'
                                        }`} />
                                </button>
                            </div>
                            {persistEnabled && (
                                <button
                                    onClick={() => {
                                        clearPersistedSettings();
                                        window.location.reload();
                                    }}
                                    className="w-full py-2 px-4 text-sm font-medium text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800/50 rounded-lg hover:bg-red-100 dark:hover:bg-red-950/50 transition-colors"
                                >
                                    Reset to Defaults
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Generation Settings Section */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 text-zinc-900 dark:text-white">
                            <Sliders size={20} />
                            <h3 className="font-semibold">Generation</h3>
                        </div>
                        <div className="pl-7 space-y-3">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-zinc-900 dark:text-white font-medium">HQ Upscale Steps</p>
                                    <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">Inference steps when upscaling a preview to high quality</p>
                                </div>
                                <input
                                    type="number"
                                    min={8}
                                    max={500}
                                    step={8}
                                    value={hqUpscaleSteps}
                                    onChange={(e) => {
                                        const val = Math.max(8, Math.min(500, parseInt(e.target.value, 10) || 160));
                                        setHqUpscaleSteps(val);
                                        localStorage.setItem('hq_upscale_steps', String(val));
                                    }}
                                    className="w-20 px-2 py-1.5 text-sm font-mono text-center rounded-lg border border-zinc-300 dark:border-zinc-600 bg-white dark:bg-zinc-800 text-zinc-900 dark:text-white focus:outline-none focus:border-indigo-500"
                                />
                            </div>
                        </div>
                            {/* Quick Preset Steps */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-zinc-900 dark:text-white font-medium">Quick Preset Steps</p>
                                    <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">Inference steps for the ⚡ Quick preset button</p>
                                </div>
                                <input
                                    type="number"
                                    min={4}
                                    max={200}
                                    step={4}
                                    value={quickPresetSteps}
                                    onChange={(e) => {
                                        const val = Math.max(4, Math.min(200, parseInt(e.target.value, 10) || 12));
                                        setQuickPresetSteps(val);
                                        localStorage.setItem('quick_preset_steps', String(val));
                                    }}
                                    className="w-20 px-2 py-1.5 text-sm font-mono text-center rounded-lg border border-zinc-300 dark:border-zinc-600 bg-white dark:bg-zinc-800 text-zinc-900 dark:text-white focus:outline-none focus:border-indigo-500"
                                />
                            </div>
                            {/* HQ Preset Steps */}
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-zinc-900 dark:text-white font-medium">HQ Preset Steps</p>
                                    <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">Inference steps for the 🔥 HQ preset button</p>
                                </div>
                                <input
                                    type="number"
                                    min={20}
                                    max={500}
                                    step={10}
                                    value={hqPresetSteps}
                                    onChange={(e) => {
                                        const val = Math.max(20, Math.min(500, parseInt(e.target.value, 10) || 60));
                                        setHqPresetSteps(val);
                                        localStorage.setItem('hq_preset_steps', String(val));
                                    }}
                                    className="w-20 px-2 py-1.5 text-sm font-mono text-center rounded-lg border border-zinc-300 dark:border-zinc-600 bg-white dark:bg-zinc-800 text-zinc-900 dark:text-white focus:outline-none focus:border-indigo-500"
                                />
                            </div>
                            {/* AI Cover Art */}
                            <div className="flex items-center justify-between pt-2 border-t border-zinc-200 dark:border-zinc-700/50">
                                <div>
                                    <p className="text-sm text-zinc-900 dark:text-white font-medium">AI Cover Art</p>
                                    <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">Generate relevant album art with AI after each song (SDXL Turbo)</p>
                                </div>
                                <button
                                    onClick={() => {
                                        const next = !generateCoverArt;
                                        setGenerateCoverArt(next);
                                        localStorage.setItem('generate_cover_art', String(next));
                                    }}
                                    className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${generateCoverArt ? 'bg-indigo-600' : 'bg-zinc-300 dark:bg-zinc-600'}`}
                                >
                                    <span className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform duration-200 ${generateCoverArt ? 'translate-x-5' : 'translate-x-0'}`} />
                                </button>
                            </div>
                    </div>

                    {/* Audio Export Section */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 text-zinc-900 dark:text-white">
                            <Download size={20} />
                            <h3 className="font-semibold">Audio Export</h3>
                        </div>
                        <div className="pl-7 space-y-3">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-zinc-900 dark:text-white font-medium">MP3 Bitrate</p>
                                    <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">Quality for MP3 exports</p>
                                </div>
                                <div className="relative">
                                    <select
                                        value={mp3Bitrate}
                                        onChange={(e) => {
                                            const val = e.target.value;
                                            setMp3Bitrate(val);
                                            localStorage.setItem('mp3_export_bitrate', val);
                                        }}
                                        className="appearance-none py-1.5 px-3 pr-8 rounded-lg border border-zinc-300 dark:border-zinc-600 bg-white dark:bg-zinc-800 text-zinc-900 dark:text-white text-sm font-medium transition-colors hover:border-zinc-400 dark:hover:border-zinc-500 focus:outline-none focus:border-indigo-500 cursor-pointer"
                                    >
                                        <option value="V0">VBR V0 (~245 kbps)</option>
                                        <option value="128">128 kbps</option>
                                        <option value="192">192 kbps</option>
                                        <option value="256">256 kbps</option>
                                        <option value="320">320 kbps</option>
                                    </select>
                                    <ChevronDown size={14} className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 pointer-events-none" />
                                </div>
                            </div>
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-zinc-900 dark:text-white font-medium">Opus Bitrate</p>
                                    <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">Quality for Opus/OGG exports</p>
                                </div>
                                <div className="relative">
                                    <select
                                        value={opusBitrate}
                                        onChange={(e) => {
                                            const val = e.target.value;
                                            setOpusBitrate(val);
                                            localStorage.setItem('opus_export_bitrate', val);
                                        }}
                                        className="appearance-none py-1.5 px-3 pr-8 rounded-lg border border-zinc-300 dark:border-zinc-600 bg-white dark:bg-zinc-800 text-zinc-900 dark:text-white text-sm font-medium transition-colors hover:border-zinc-400 dark:hover:border-zinc-500 focus:outline-none focus:border-indigo-500 cursor-pointer"
                                    >
                                        <option value="64">64 kbps</option>
                                        <option value="96">96 kbps</option>
                                        <option value="128">128 kbps</option>
                                        <option value="192">192 kbps</option>
                                        <option value="256">256 kbps</option>
                                    </select>
                                    <ChevronDown size={14} className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 pointer-events-none" />
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* About Section */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 text-zinc-900 dark:text-white">
                            <Info size={20} />
                            <h3 className="font-semibold">{t('about')}</h3>
                        </div>
                        <div className="pl-7 space-y-3 text-sm text-zinc-600 dark:text-zinc-400">
                            <p>{t('version')} 2.5.0</p>
                            <p>ACE-Step UI - {t('localAIMusicGenerator')}</p>
                            <p className="text-xs text-zinc-400 dark:text-zinc-500 mt-2">
                                {t('poweredBy')}
                            </p>
                            <div className="pt-3 border-t border-zinc-200 dark:border-zinc-700/50 mt-4 space-y-4">
                                <div>
                                    <p className="text-zinc-900 dark:text-white font-medium mb-2">{t('createdBy')}</p>
                                    <div className="flex flex-wrap gap-2">
                                        <a
                                            href="https://x.com/AmbsdOP"
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="inline-flex items-center gap-2 px-4 py-2 bg-black dark:bg-white text-white dark:text-black rounded-lg text-sm font-medium hover:bg-zinc-800 dark:hover:bg-zinc-200 transition-colors"
                                        >
                                            <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                                                <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                                            </svg>
                                            {t('follow')} @AmbsdOP
                                        </a>
                                        <a
                                            href="https://github.com/fspecii/ace-step-ui"
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="inline-flex items-center gap-2 px-4 py-2 bg-zinc-800 dark:bg-zinc-700 text-white rounded-lg text-sm font-medium hover:bg-zinc-700 dark:hover:bg-zinc-600 transition-colors"
                                        >
                                            <Github size={16} />
                                            GitHub Repo
                                        </a>
                                    </div>
                                    <p className="text-xs text-zinc-400 dark:text-zinc-500 mt-2">
                                        Report issues or request features on GitHub
                                    </p>
                                </div>
                                <div>
                                    <p className="text-zinc-900 dark:text-white font-medium mb-2">{t('localizedBy')}</p>
                                    <div className="flex flex-wrap gap-2">
                                        <a
                                            href="https://x.com/bdsqlsz"
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="inline-flex items-center gap-2 px-4 py-2 bg-black dark:bg-white text-white dark:text-black rounded-lg text-sm font-medium hover:bg-zinc-800 dark:hover:bg-zinc-200 transition-colors"
                                        >
                                            <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                                                <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                                            </svg>
                                            {t('follow')} @bdsqlsz
                                        </a>
                                        <a
                                            href="https://space.bilibili.com/219296"
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="inline-flex items-center gap-2 px-4 py-2 bg-[#00A1D6] text-white rounded-lg text-sm font-medium hover:bg-[#0090C0] transition-colors"
                                        >
                                            <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                                                <path d="M17.813 4.653h.854c1.51.054 2.769.578 3.773 1.574 1.004.995 1.524 2.249 1.56 3.76v7.36c-.036 1.51-.556 2.769-1.56 3.773s-2.262 1.524-3.773 1.56H5.333c-1.51-.036-2.769-.556-3.773-1.56S.036 18.858 0 17.347v-7.36c.036-1.511.556-2.765 1.56-3.76 1.004-.996 2.262-1.52 3.773-1.574h.774l-1.174-1.12a1.234 1.234 0 0 1-.373-.906c0-.356.124-.658.373-.907l.027-.027c.267-.249.573-.373.92-.373.347 0 .653.124.92.373L9.653 4.44c.071.071.134.142.187.213h4.267a.836.836 0 0 1 .16-.213l2.853-2.747c.267-.249.573-.373.92-.373.347 0 .662.151.929.4.267.249.391.551.391.907 0 .355-.124.657-.373.906zM5.333 7.24c-.746.018-1.373.276-1.88.773-.506.498-.769 1.13-.786 1.894v7.52c.017.764.28 1.395.786 1.893.507.498 1.134.756 1.88.773h13.334c.746-.017 1.373-.275 1.88-.773.506-.498.769-1.129.786-1.893v-7.52c-.017-.765-.28-1.396-.786-1.894-.507-.497-1.134-.755-1.88-.773zM8 11.107c.373 0 .684.124.933.373.25.249.383.569.4.96v1.173c-.017.391-.15.711-.4.96-.249.25-.56.374-.933.374s-.684-.125-.933-.374c-.25-.249-.383-.569-.4-.96V12.44c0-.373.129-.689.386-.947.258-.257.574-.386.947-.386zm8 0c.373 0 .684.124.933.373.25.249.383.569.4.96v1.173c-.017.391-.15.711-.4.96-.249.25-.56.374-.933.374s-.684-.125-.933-.374c-.25-.249-.383-.569-.4-.96V12.44c.017-.391.15-.711.4-.96.249-.249.56-.373.933-.373Z" />
                                            </svg>
                                            {t('follow')} 青龙圣者
                                        </a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="border-t border-zinc-200 dark:border-white/5 p-6 flex justify-end">
                    <button
                        onClick={onClose}
                        className="px-6 py-2 bg-zinc-900 dark:bg-white text-white dark:text-black font-semibold rounded-lg hover:bg-zinc-800 dark:hover:bg-zinc-200 transition-colors"
                    >
                        {t('done')}
                    </button>
                </div>
            </div>

            <EditProfileModal
                isOpen={isEditProfileOpen}
                onClose={() => setIsEditProfileOpen(false)}
                onSaved={() => setIsEditProfileOpen(false)}
            />
        </div>
    );
};
