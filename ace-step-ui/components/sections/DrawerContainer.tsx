import React, { useEffect, useRef } from 'react';
import { ArrowLeft } from 'lucide-react';

interface DrawerContainerProps {
  /** Drawer title shown in the back-button header */
  title: string;
  /** Whether this drawer is currently visible */
  isOpen: boolean;
  /** Called when the user clicks the back button */
  onBack: () => void;
  /** The drawer's content (typically an existing accordion's contents) */
  children: React.ReactNode;
}

/**
 * A full-height sliding drawer that replaces the main panel content.
 * When open, it slides in from the right with a back-button header.
 * The parent should conditionally render either the main panel or the drawer.
 */
export const DrawerContainer: React.FC<DrawerContainerProps> = ({
  title,
  isOpen,
  onBack,
  children,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Scroll to top when the drawer opens
  useEffect(() => {
    if (isOpen && containerRef.current) {
      containerRef.current.scrollTop = 0;
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div
      ref={containerRef}
      className="flex flex-col h-full animate-in slide-in-from-right-4 duration-200"
    >
      {/* Header with back button */}
      <div className="sticky top-0 z-10 flex items-center gap-2 px-4 py-3 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-md border-b border-zinc-200 dark:border-white/5">
        <button
          type="button"
          onClick={onBack}
          className="flex items-center gap-1.5 text-sm font-medium text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors group"
        >
          <ArrowLeft
            size={16}
            className="group-hover:-translate-x-0.5 transition-transform"
          />
          Back
        </button>
        <span className="text-zinc-300 dark:text-zinc-600">|</span>
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-white truncate">
          {title}
        </h3>
      </div>

      {/* Drawer content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {children}
      </div>
    </div>
  );
};
