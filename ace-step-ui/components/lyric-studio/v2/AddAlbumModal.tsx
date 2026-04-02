import React, { useState } from 'react';
import { X, Disc3, Link2 } from 'lucide-react';

interface AddAlbumModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (albumName: string | undefined, imageUrl?: string) => Promise<void>;
  artistName: string;
}

export const AddAlbumModal: React.FC<AddAlbumModalProps> = ({ isOpen, onClose, onSubmit, artistName }) => {
  const [albumName, setAlbumName] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [submitting, setSubmitting] = useState(false);

  React.useEffect(() => {
    if (isOpen) {
      setAlbumName('');
      setImageUrl('');
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      await onSubmit(albumName.trim() || undefined, imageUrl.trim() || undefined);
      onClose();
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div
        className="bg-zinc-900 border border-white/10 rounded-2xl shadow-2xl w-full max-w-md overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center">
              <Disc3 className="w-4 h-4 text-indigo-400" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Add Album</h2>
              <p className="text-xs text-zinc-500">{artistName}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-white/5 text-zinc-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-5">
          <div>
            <label className="block text-sm font-medium text-zinc-300 mb-1.5">
              <span className="flex items-center gap-1.5">
                <Disc3 className="w-3.5 h-3.5 text-zinc-400" />
                Album Name
                <span className="text-zinc-500 font-normal">(optional)</span>
              </span>
            </label>
            <input
              type="text"
              value={albumName}
              onChange={(e) => setAlbumName(e.target.value)}
              placeholder="Leave blank for loose lyrics collection"
              className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white placeholder:text-zinc-500 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/20 transition-all"
              autoFocus
            />
            <p className="text-xs text-zinc-500 mt-1">
              If left blank, songs will be stored as a loose lyrics collection
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-zinc-300 mb-1.5">
              <span className="flex items-center gap-1.5">
                <Link2 className="w-3.5 h-3.5 text-zinc-400" />
                Cover Art URL
                <span className="text-zinc-500 font-normal">(optional)</span>
              </span>
            </label>
            <input
              type="text"
              value={imageUrl}
              onChange={(e) => setImageUrl(e.target.value)}
              placeholder="https://example.com/cover.jpg"
              className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-white placeholder:text-zinc-500 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/20 transition-all"
            />
          </div>

          <button
            type="submit"
            disabled={submitting}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white font-semibold transition-all"
          >
            {submitting ? 'Creating…' : 'Create Album'}
          </button>
        </form>
      </div>
    </div>
  );
};
