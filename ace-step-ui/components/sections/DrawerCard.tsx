import React from "react";
import { ChevronRight } from "lucide-react";

interface DrawerCardProps {
  /** Emoji or icon */
  icon: React.ReactNode;
  /** Card title */
  title: string;
  /** Brief description — shown as tooltip on hover */
  description: string;
  /** Summary badge text */
  summary: string;
  /** Called when the card is clicked */
  onClick: () => void;
  /** When true, the card is completely hidden */
  hidden?: boolean;
  /** Optional element rendered between summary and chevron (e.g. toggle switch) */
  rightElement?: React.ReactNode;
}

/**
 * A compact clickable card for opening a drawer panel.
 * Compact single-line design with icon, title, summary badge, and chevron.
 * Description appears as a native tooltip on hover.
 */
export const DrawerCard: React.FC<DrawerCardProps> = ({
  icon,
  title,
  description,
  summary,
  onClick,
  hidden = false,
  rightElement,
}) => {
  if (hidden) return null;

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick?.();
        }
      }}
      title={description}
      className="w-full text-left group bg-zinc-50 dark:bg-zinc-800/50 hover:bg-zinc-100 dark:hover:bg-zinc-800 border border-zinc-200 dark:border-white/5 rounded-lg px-3 py-2 transition-all duration-150 hover:shadow-sm cursor-pointer"
    >
      <div className="flex items-center gap-2.5">
        {/* Icon */}
        <span className="flex-shrink-0 text-sm">{icon}</span>

        {/* Title */}
        <span className="text-xs font-semibold text-zinc-700 dark:text-zinc-200 truncate">
          {title}
        </span>

        {/* Summary badge */}
        {summary && (
          <span className="ml-auto flex-shrink-0 text-[10px] font-mono font-medium px-1.5 py-0.5 rounded bg-zinc-200/60 dark:bg-zinc-700/60 text-zinc-500 dark:text-zinc-400 truncate max-w-[45%]">
            {summary}
          </span>
        )}

        {/* Optional right element (e.g. toggle) */}
        {rightElement && (
          <span className="flex-shrink-0" onClick={(e) => e.stopPropagation()}>
            {rightElement}
          </span>
        )}

        {/* Chevron */}
        <ChevronRight
          size={12}
          className="flex-shrink-0 text-zinc-400 dark:text-zinc-500 group-hover:text-indigo-500 dark:group-hover:text-indigo-400 group-hover:translate-x-0.5 transition-all"
        />
      </div>
    </div>
  );
};
