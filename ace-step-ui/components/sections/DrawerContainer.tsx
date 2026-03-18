import React, { useEffect, useRef } from 'react';
import { ArrowLeft } from 'lucide-react';

interface DrawerContainerProps {
  /** Drawer title shown in the back-button header */
  title: string;
  /** Whether this drawer is currently visible */
  isOpen: boolean;
  /** Called when the user clicks the header to close */
  onClose: () => void;
  /** The drawer's content */
  children: React.ReactNode;
}

/**
 * An inline expandable drawer that renders in-place when open.
 * The entire header bar is clickable to close the drawer.
 */
export const DrawerContainer: React.FC<DrawerContainerProps> = ({
  title,
  isOpen,
  onClose,
  children,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Scroll the drawer into view when it opens
  useEffect(() => {
    if (isOpen && containerRef.current) {
      containerRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div
      ref={containerRef}
      className="rounded-xl border border-indigo-200 dark:border-indigo-500/20 bg-white dark:bg-zinc-900/80 overflow-hidden animate-in fade-in duration-150"
    >
      {/* Header — entire bar is clickable to close */}
      <button
        type="button"
        onClick={onClose}
        className="w-full flex items-center gap-2 px-3 py-2 bg-indigo-50/50 dark:bg-indigo-500/5 border-b border-indigo-200 dark:border-indigo-500/10 hover:bg-indigo-100/50 dark:hover:bg-indigo-500/10 transition-colors group cursor-pointer"
      >
        <ArrowLeft
          size={14}
          className="flex-shrink-0 text-zinc-500 dark:text-zinc-400 group-hover:text-zinc-900 dark:group-hover:text-white group-hover:-translate-x-0.5 transition-all"
        />
        <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400 group-hover:text-zinc-900 dark:group-hover:text-white transition-colors">
          Back
        </span>
        <span className="text-zinc-300 dark:text-zinc-600">|</span>
        <h3 className="text-xs font-semibold text-zinc-900 dark:text-white truncate">
          {title}
        </h3>
      </button>

      {/* Drawer content */}
      <div className="p-3 space-y-3">
        {children}
      </div>
    </div>
  );
};
