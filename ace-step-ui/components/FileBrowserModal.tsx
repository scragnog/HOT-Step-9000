import React, { useState, useEffect, useCallback } from 'react';
import { Folder, FileText, ChevronUp, X, Loader2 } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { generateApi } from '../services/api';

interface BrowseEntry {
    name: string;
    path: string;
    type: 'dir' | 'file';
    size?: number;
}

interface FileBrowserModalProps {
    /** Controls visibility */
    open: boolean;
    /** Called when modal is dismissed without selection */
    onClose: () => void;
    /** Called with the selected path (file or folder) */
    onSelect: (path: string) => void;
    /** 'file' = only .safetensors selectable, 'folder' = only directories */
    mode: 'file' | 'folder';
    /** Optional starting directory */
    startPath?: string;
}

function formatSize(bytes?: number): string {
    if (bytes == null) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export const FileBrowserModal: React.FC<FileBrowserModalProps> = ({
    open, onClose, onSelect, mode, startPath,
}) => {
    const { token } = useAuth();
    const [currentDir, setCurrentDir] = useState(startPath || '');
    const [entries, setEntries] = useState<BrowseEntry[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [pathInput, setPathInput] = useState('');

    const loadDir = useCallback(async (dir: string) => {
        if (!token) return;
        setLoading(true);
        setError(null);
        try {
            const result = await generateApi.browseDir(dir, token);
            setCurrentDir(result.current);
            setPathInput(result.current);
            setEntries(result.entries);
        } catch (err: any) {
            setError(err?.message || 'Failed to list directory');
        } finally {
            setLoading(false);
        }
    }, [token]);

    // Load initial directory when modal opens
    useEffect(() => {
        if (open) {
            loadDir(startPath || '');
        }
    }, [open, startPath, loadDir]);

    if (!open) return null;

    const handleEntryClick = (entry: BrowseEntry) => {
        if (entry.type === 'dir') {
            loadDir(entry.path);
        } else if (mode === 'file') {
            onSelect(entry.path);
        }
    };

    const handleSelectFolder = () => {
        if (mode === 'folder' && currentDir) {
            onSelect(currentDir);
        }
    };

    const handlePathSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (pathInput.trim()) {
            loadDir(pathInput.trim());
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
            onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
        >
            <div className="w-[540px] max-h-[70vh] bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-white/10 shadow-2xl flex flex-col overflow-hidden">
                {/* Header */}
                <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-100 dark:border-white/5">
                    <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
                        {mode === 'file' ? 'Select Adapter File' : 'Select Adapter Folder'}
                    </h3>
                    <button onClick={onClose} className="text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300 transition-colors">
                        <X size={16} />
                    </button>
                </div>

                {/* Path bar */}
                <form onSubmit={handlePathSubmit} className="px-4 py-2 border-b border-zinc-100 dark:border-white/5">
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={pathInput}
                            onChange={(e) => setPathInput(e.target.value)}
                            placeholder="Enter path..."
                            className="flex-1 bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg px-3 py-1.5 text-xs text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-pink-500 font-mono"
                        />
                        <button
                            type="submit"
                            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
                        >
                            Go
                        </button>
                    </div>
                </form>

                {/* Error */}
                {error && (
                    <div className="px-4 py-2 text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20">
                        {error}
                    </div>
                )}

                {/* File list */}
                <div className="flex-1 overflow-y-auto min-h-[200px] max-h-[400px]">
                    {loading ? (
                        <div className="flex items-center justify-center py-12 text-zinc-400">
                            <Loader2 size={20} className="animate-spin mr-2" />
                            <span className="text-xs">Loading...</span>
                        </div>
                    ) : (
                        <div className="divide-y divide-zinc-50 dark:divide-white/5">
                            {entries.map((entry) => (
                                <button
                                    key={entry.path}
                                    onClick={() => handleEntryClick(entry)}
                                    className="w-full flex items-center gap-3 px-4 py-2 text-left hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors group"
                                >
                                    {entry.type === 'dir' ? (
                                        entry.name === '..' ? (
                                            <ChevronUp size={16} className="text-zinc-400 flex-shrink-0" />
                                        ) : (
                                            <Folder size={16} className="text-amber-500 flex-shrink-0" />
                                        )
                                    ) : (
                                        <FileText size={16} className="text-pink-500 flex-shrink-0" />
                                    )}
                                    <span className={`text-xs truncate flex-1 ${
                                        entry.type === 'dir'
                                            ? 'text-zinc-700 dark:text-zinc-300 font-medium'
                                            : 'text-zinc-600 dark:text-zinc-400'
                                    }`}>
                                        {entry.name}
                                    </span>
                                    {entry.type === 'file' && entry.size != null && (
                                        <span className="text-[10px] text-zinc-400 flex-shrink-0">{formatSize(entry.size)}</span>
                                    )}
                                    {entry.type === 'file' && mode === 'file' && (
                                        <span className="text-[10px] font-semibold text-pink-500 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0">
                                            Select
                                        </span>
                                    )}
                                </button>
                            ))}
                            {entries.length === 0 && !loading && (
                                <div className="py-8 text-center text-xs text-zinc-400">
                                    No folders or .safetensors files found
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between px-4 py-3 border-t border-zinc-100 dark:border-white/5 bg-zinc-50 dark:bg-black/20">
                    <span className="text-[10px] text-zinc-400 truncate max-w-[60%] font-mono" title={currentDir}>
                        {currentDir}
                    </span>
                    <div className="flex gap-2">
                        {mode === 'folder' && (
                            <button
                                onClick={handleSelectFolder}
                                className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-pink-600 text-white hover:bg-pink-700 transition-colors"
                            >
                                Select This Folder
                            </button>
                        )}
                        <button
                            onClick={onClose}
                            className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-zinc-200 dark:bg-zinc-700 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-300 dark:hover:bg-zinc-600 transition-colors"
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};
