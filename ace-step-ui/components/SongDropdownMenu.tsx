import React, { useEffect, useRef } from 'react';
import { Song } from '../types';
import { useI18n } from '../context/I18nContext';
import {
    Video,
    Edit3,
    Layers,
    Repeat,
    ListPlus,
    Download,
    Trash2,
    Share2,
    Sparkles,
    ArrowUpCircle,
    Wand2
} from 'lucide-react';
import { openStemSplitter } from './StemSplitterModal';
import { openAudioEnhancer } from './AudioEnhancerModal';

interface SongDropdownMenuProps {
    song: Song;
    isOpen: boolean;
    onClose: () => void;
    isOwner?: boolean;
    position?: 'left' | 'right';
    direction?: 'up' | 'down';
    onCreateVideo?: () => void;
    onEditAudio?: () => void;
    onExtractStems?: () => void;
    onReusePrompt?: () => void;
    onAddToPlaylist?: () => void;
    onDownloadFormat?: () => void;
    onShare?: () => void;
    onDelete?: () => void;
    onUseAsReference?: () => void;
    onCoverSong?: () => void;
    onUpscaleToHQ?: () => void;
    onRemaster?: () => void;
}

interface MenuItemProps {
    icon: React.ReactNode;
    label: string;
    onClick?: () => void;
    danger?: boolean;
    disabled?: boolean;
}

const MenuItem: React.FC<MenuItemProps> = ({ icon, label, onClick, danger, disabled }) => (
    <button
        onClick={onClick}
        disabled={disabled}
        className={`w-full px-3 py-2 text-left text-sm flex items-center gap-3 transition-colors
            ${danger
                ? 'text-red-400 hover:bg-red-500/10 hover:text-red-300'
                : 'text-zinc-300 hover:bg-white/5 hover:text-white'}
            ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        `}
    >
        <span className="w-4 h-4 flex items-center justify-center opacity-70">{icon}</span>
        <span>{label}</span>
    </button>
);

const MenuDivider: React.FC = () => (
    <div className="h-px bg-white/10 my-1 mx-2" />
);

export const SongDropdownMenu: React.FC<SongDropdownMenuProps> = ({
    song,
    isOpen,
    onClose,
    isOwner = false,
    position = 'right',
    direction = 'down',
    onCreateVideo,
    onEditAudio,
    onExtractStems,
    onReusePrompt,
    onAddToPlaylist,
    onDownloadFormat,
    onShare,
    onDelete,
    onUseAsReference,
    onCoverSong,
    onUpscaleToHQ,
    onRemaster
}) => {
    const { t } = useI18n();
    const menuRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
                onClose();
            }
        };

        const handleEscape = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                onClose();
            }
        };

        if (isOpen) {
            document.addEventListener('mousedown', handleClickOutside);
            document.addEventListener('keydown', handleEscape);
        }

        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
            document.removeEventListener('keydown', handleEscape);
        };
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    const handleAction = (action?: () => void) => {
        if (action) {
            action();
        }
        onClose();
    };

    const handleEditAudio = () => {
        if (!song.audioUrl) return;
        const audioUrl = song.audioUrl.startsWith('http')
            ? song.audioUrl
            : `${window.location.origin}${song.audioUrl}`;
        window.open(`/editor?audioUrl=${encodeURIComponent(audioUrl)}`, '_blank');
        onClose();
    };

    const handleExtractStems = () => {
        if (!song.audioUrl) return;
        openStemSplitter(song.audioUrl, song.title);
        onClose();
    };

    const handleDownload = () => {
        if (onDownloadFormat) {
            onDownloadFormat();
        }
        onClose();
    };

    const positionClasses = position === 'left' ? 'left-0' : 'right-0';
    const directionClasses = direction === 'up'
        ? 'bottom-full mb-2'
        : 'top-full mt-2';
    const animationClasses = direction === 'up'
        ? 'animate-in fade-in slide-in-from-bottom-2'
        : 'animate-in fade-in slide-in-from-top-2';

    return (
        <div
            ref={menuRef}
            className={`absolute ${positionClasses} ${directionClasses} w-52
                bg-zinc-900 rounded-xl shadow-2xl border border-white/10 py-1.5 z-50
                ${animationClasses} duration-150`}
            onClick={(e) => e.stopPropagation()}
        >
            {/* Creative Actions */}
            <MenuItem
                icon={<Video size={14} />}
                label={t('createVideo')}
                onClick={() => handleAction(onCreateVideo)}
            />
            {isOwner && (
                <MenuItem
                    icon={<Edit3 size={14} />}
                    label={t('editAudio')}
                    onClick={onEditAudio ? () => handleAction(onEditAudio) : handleEditAudio}
                />
            )}
            <MenuItem
                icon={<Layers size={14} />}
                label={t('extractStems')}
                onClick={onExtractStems ? () => handleAction(onExtractStems) : handleExtractStems}
            />
            <MenuItem
                icon={<Sparkles size={14} />}
                label={t('enhanceAudio')}
                onClick={() => {
                    if (song.audioUrl) {
                        openAudioEnhancer(song.audioUrl, song.title, song.id);
                    }
                    onClose();
                }}
                disabled={!song.audioUrl}
            />
            {onUpscaleToHQ && (
                <MenuItem
                    icon={<ArrowUpCircle size={14} />}
                    label={t('upscaleToHQ') || 'Upscale to HQ'}
                    onClick={() => handleAction(onUpscaleToHQ)}
                />
            )}
            {onRemaster && (
                <MenuItem
                    icon={<Wand2 size={14} />}
                    label="Remaster"
                    onClick={() => handleAction(onRemaster)}
                    disabled={!song.audioUrl}
                />
            )}
            {onReusePrompt && (
                <MenuItem
                    icon={<Repeat size={14} />}
                    label={t('reusePrompt')}
                    onClick={() => handleAction(onReusePrompt)}
                />
            )}
            {onUseAsReference && (
                <MenuItem
                    icon={<Layers size={14} />}
                    label={t('useAsReference')}
                    onClick={() => handleAction(onUseAsReference)}
                    disabled={!song.audioUrl}
                />
            )}
            {onCoverSong && (
                <MenuItem
                    icon={<Layers size={14} />}
                    label={t('coverSong')}
                    onClick={() => handleAction(onCoverSong)}
                    disabled={!song.audioUrl}
                />
            )}

            <MenuDivider />

            {/* Library Actions */}
            <MenuItem
                icon={<ListPlus size={14} />}
                label={t('addToPlaylist')}
                onClick={() => handleAction(onAddToPlaylist)}
            />
            <MenuItem
                icon={<Download size={14} />}
                label={t('download')}
                onClick={handleDownload}
            />
            <MenuItem
                icon={<Share2 size={14} />}
                label={t('share')}
                onClick={() => handleAction(onShare)}
            />

            {/* Owner-only Actions */}
            {isOwner && (
                <>
                    <MenuDivider />
                    <MenuItem
                        icon={<Trash2 size={14} />}
                        label={t('deleteSong')}
                        onClick={() => handleAction(onDelete)}
                        danger
                    />
                </>
            )}
        </div>
    );
};
