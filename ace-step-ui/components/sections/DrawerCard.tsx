import React from 'react';
import { ChevronRight } from 'lucide-react';

interface DrawerCardProps {
  /** Lucide icon or any React node */
  icon: React.ReactNode;
  /** Card title */
  title: string;
  /** Brief description shown below the title */
  description: string;
  /** Summary badge text showing current values, e.g. "Euler · 12 steps · CFG 9.0" */
  summary: string;
  /** Called when the card is clicked */
  onClick: () => void;
  /** When true, the card is completely hidden */
  hidden?: boolean;
}

/**
 * A compact Tier-2 card that shows an icon, title, description, and a summary
 * badge of current settings. Clicking it opens the corresponding drawer.
 */
export const DrawerCard: React.FC<DrawerCardProps> = ({
  icon,
  title,
  description,
  summary,
  onClick,
  hidden = false,
}) => {
  if (hidden) return null;

  return (
    <button
      type="button"
      onClick={onClick}
      className="w-full text-left group bg-zinc-50 dark:bg-zinc-800/50 hover:bg-zinc-100 dark:hover:bg-zinc-800 border border-zinc-200 dark:border-white/5 rounded-xl p-3.5 transition-all duration-150 hover:shadow-sm"
    >
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-white dark:bg-zinc-700/50 border border-zinc-200 dark:border-white/5 flex items-center justify-center text-zinc-500 dark:text-zinc-400 group-hover:text-indigo-500 dark:group-hover:text-indigo-400 transition-colors">
          {icon}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <h4 className="text-sm font-semibold text-zinc-900 dark:text-white truncate">
              {title}
            </h4>
            <ChevronRight
              size={14}
              className="flex-shrink-0 text-zinc-400 dark:text-zinc-500 group-hover:text-indigo-500 dark:group-hover:text-indigo-400 group-hover:translate-x-0.5 transition-all"
            />
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5 leading-relaxed">
            {description}
          </p>
          {summary && (
            <div className="mt-1.5">
              <span className="inline-block text-[10px] font-mono font-medium px-2 py-0.5 rounded-md bg-zinc-200/60 dark:bg-zinc-700/60 text-zinc-600 dark:text-zinc-300 truncate max-w-full">
                {summary}
              </span>
            </div>
          )}
        </div>
      </div>
    </button>
  );
};
