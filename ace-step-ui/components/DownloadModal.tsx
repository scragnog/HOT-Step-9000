import React from 'react';
import { useI18n } from '../context/I18nContext';
import { Download, X } from 'lucide-react';

export type DownloadFormat = 'mp3' | 'flac' | 'wav' | 'opus';
export type DownloadVersion = 'mastered' | 'original' | 'both';

interface DownloadModalProps {
    isOpen: boolean;
    onClose: () => void;
    onDownload: (format: DownloadFormat, version: DownloadVersion) => void;
    songTitle?: string;
    songCount?: number;
    hasOriginal?: boolean;
}

export const DownloadModal: React.FC<DownloadModalProps> = ({
    isOpen,
    onClose,
    onDownload,
    songTitle,
    songCount,
    hasOriginal
}) => {
    const { t } = useI18n();
    const [selectedFormat, setSelectedFormat] = React.useState<DownloadFormat>('mp3');
    const [selectedVersion, setSelectedVersion] = React.useState<DownloadVersion>('mastered');

    if (!isOpen) return null;

    const handleDownload = () => {
        onDownload(selectedFormat, selectedVersion);
        onClose();
    };

    const formats: { value: DownloadFormat; label: string; desc: string }[] = [
        { value: 'mp3', label: 'MP3', desc: t('mp3Smaller') },
        { value: 'flac', label: 'FLAC', desc: t('flacLossless') },
        { value: 'wav', label: 'WAV', desc: 'Uncompressed Audio' },
        { value: 'opus', label: 'Opus', desc: 'High Quality Web' }
    ];

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
            <div className="bg-zinc-900 border border-white/10 rounded-2xl w-full max-w-sm shadow-2xl relative animate-in fade-in zoom-in-95 duration-200">

                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-white/5">
                    <div className="flex items-center gap-2 text-white">
                        <Download size={18} className="text-pink-500" />
                        <h3 className="font-semibold text-sm">{songCount && songCount > 1 ? `Download ${songCount} Tracks` : 'Download Track'}</h3>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-zinc-400 hover:text-white transition-colors p-1"
                    >
                        <X size={16} />
                    </button>
                </div>

                {/* Body */}
                <div className="p-4 space-y-4">
                    <p className="text-xs text-zinc-400">
                        {songCount && songCount > 1
                            ? <>Select audio format for <span className="text-white font-medium">{songCount} selected tracks</span>:</>
                            : <>Select audio format for <span className="text-white font-medium">"{songTitle || 'Untitled'}"</span>:</>
                        }
                    </p>

                    <div className="grid grid-cols-2 gap-2">
                        {formats.map((fmt) => (
                            <button
                                key={fmt.value}
                                onClick={() => setSelectedFormat(fmt.value)}
                                className={`flex flex-col items-center justify-center p-3 rounded-xl border transition-all ${selectedFormat === fmt.value
                                        ? 'bg-pink-500/10 border-pink-500 text-pink-400'
                                        : 'bg-black/20 border-white/5 text-zinc-400 hover:border-white/20'
                                    }`}
                            >
                                <span className="font-bold">{fmt.label}</span>
                                <span className="text-[10px] opacity-70">{fmt.desc}</span>
                            </button>
                        ))}
                    </div>

                    {hasOriginal && (
                        <div className="pt-2">
                            <p className="text-xs text-zinc-400 mb-2 font-semibold tracking-wider uppercase">Version</p>
                            <div className="flex bg-black/20 p-1 rounded-xl border border-white/5">
                                {['mastered', 'original', 'both'].map((v) => (
                                    <button
                                        key={v}
                                        onClick={() => setSelectedVersion(v as DownloadVersion)}
                                        className={`flex-1 py-1.5 text-xs font-semibold rounded-lg transition-all capitalize ${selectedVersion === v
                                            ? 'bg-zinc-800 text-white shadow'
                                            : 'text-zinc-500 hover:text-zinc-300'
                                        }`}
                                    >
                                        {v}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="p-4 pt-0">
                    <button
                        onClick={handleDownload}
                        className="w-full flex items-center justify-center gap-2 py-2.5 bg-gradient-to-r from-orange-500 to-pink-600 hover:from-orange-400 hover:to-pink-500 text-white rounded-xl font-medium text-sm transition-all shadow-lg shadow-pink-500/20"
                    >
                        <Download size={16} />
                        Download {selectedFormat.toUpperCase()}{songCount && songCount > 1 ? ` (${songCount})` : ''}
                    </button>
                </div>
            </div>
        </div>
    );
};
