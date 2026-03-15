import React from 'react';
import { Library, Disc, Search, User, LogIn, LogOut, Sun, Moon, Power } from 'lucide-react';
import { View } from '../types';
import { useI18n } from '../context/I18nContext';

interface SidebarProps {
  currentView: View;
  onNavigate: (view: View) => void;
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
  user?: { username: string; isAdmin?: boolean; avatar_url?: string } | null;
  onLogin?: () => void;
  onLogout?: () => void;
  onOpenSettings?: () => void;
  onQuit?: () => void;
  isOpen?: boolean;
  onToggle?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentView,
  onNavigate,
  theme,
  onToggleTheme,
  user,
  onLogin,
  onLogout,
  onOpenSettings,
  onQuit,
  isOpen = true,
  onToggle,
}) => {
  const { t } = useI18n();

  return (
    <>
      {/* Backdrop for mobile - only when expanded */}
      {isOpen && onToggle && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 md:hidden"
          onClick={onToggle}
        />
      )}

      {/* Sidebar */}
      <div className={`
        flex flex-col h-full bg-white dark:bg-suno-sidebar border-r border-zinc-200 dark:border-white/5 flex-shrink-0 py-4 overflow-y-auto scrollbar-hide transition-all duration-300
        fixed left-0 top-0 z-50 md:relative
        ${isOpen ? 'w-[200px]' : 'w-[72px]'}
      `}>
        {/* Logo & Brand */}
        <div className={`mb-8 flex items-center ${isOpen ? 'px-3' : 'justify-center'}`}>
          <div className="flex items-center gap-3">
            <button
              className="w-10 h-10 rounded-full bg-gradient-to-br from-pink-500 to-purple-600 flex items-center justify-center shadow-lg hover:scale-105 transition-transform flex-shrink-0"
              onClick={onToggle}
              title={isOpen ? t('collapse') : t('aceStepUI')}
            >
              <svg className={`w-5 h-5 text-white transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
            {isOpen && (
              <div className="flex flex-col items-center leading-tight">
                <span className="text-lg font-bold text-zinc-900 dark:text-white whitespace-nowrap">HOT-Step</span>
                <span className="text-xs font-semibold tracking-[0.25em] text-zinc-500 dark:text-zinc-400">9000</span>
              </div>
            )}
          </div>
        </div>

        <nav className="flex-1 flex flex-col gap-2 w-full px-3">
          <NavItem
            icon={<Disc size={20} />}
            label={t('create')}
            active={currentView === 'create'}
            onClick={() => onNavigate('create')}
            isExpanded={isOpen}
          />
          <NavItem
            icon={<Library size={20} />}
            label={t('library')}
            active={currentView === 'library'}
            onClick={() => onNavigate('library')}
            isExpanded={isOpen}
          />
          <NavItem
            icon={<Search size={20} />}
            label={t('search')}
            active={currentView === 'search'}
            onClick={() => onNavigate('search')}
            isExpanded={isOpen}
          />

          <div className="mt-auto flex flex-col gap-2">
            {/* Theme Toggle */}
            <button
              onClick={onToggleTheme}
              className={`
              w-full rounded-xl flex items-center gap-3 transition-all duration-200 text-zinc-500 dark:text-zinc-400 hover:text-black dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-white/5
              ${isOpen ? 'px-3 py-2.5 justify-start' : 'aspect-square justify-center'}
            `}
              title={theme === 'dark' ? t('lightMode') : t('darkMode')}
            >
              <div className="flex-shrink-0">{theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}</div>
              {isOpen && (
                <span className="text-sm font-medium whitespace-nowrap">
                  {theme === 'dark' ? t('lightMode') : t('darkMode')}
                </span>
              )}
            </button>

            {user ? (
              <>
                {/* User Settings */}
                <button
                  onClick={onOpenSettings}
                  className={`
                  w-full rounded-xl flex items-center gap-3 transition-all duration-200 text-zinc-500 dark:text-zinc-400 hover:text-black dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-white/5
                  ${isOpen ? 'px-3 py-2.5 justify-start' : 'aspect-square justify-center'}
                `}
                  title={`${user.username} - ${t('settings')}`}
                >
                  <div className="w-6 h-6 rounded-full bg-gradient-to-br from-pink-500 to-purple-600 flex items-center justify-center text-white text-xs font-bold border border-white/20 overflow-hidden flex-shrink-0">
                    {user.avatar_url ? (
                      <img src={user.avatar_url} alt={user.username} className="w-full h-full object-cover" />
                    ) : (
                      user.username.charAt(0).toUpperCase()
                    )}
                  </div>
                  {isOpen && (
                    <span className="text-sm font-medium whitespace-nowrap truncate flex-1 text-left">
                      {user.username}
                    </span>
                  )}
                </button>
                {/* Logout */}
                <button
                  onClick={onLogout}
                  className={`
                  w-full rounded-xl flex items-center gap-3 transition-all duration-200 text-zinc-500 hover:text-red-500 hover:bg-red-500/10
                  ${isOpen ? 'px-3 py-2.5 justify-start' : 'aspect-square justify-center'}
                `}
                  title={t('signOut')}
                >
                  <div className="flex-shrink-0"><LogOut size={20} /></div>
                  {isOpen && (
                    <span className="text-sm font-medium whitespace-nowrap">{t('signOut')}</span>
                  )}
                </button>
              </>
            ) : (
              <button
                onClick={onLogin}
                className={`
                w-full rounded-xl flex items-center gap-3 transition-all duration-200 text-zinc-500 dark:text-zinc-400 hover:text-pink-500 hover:bg-zinc-100 dark:hover:bg-white/5
                ${isOpen ? 'px-3 py-2.5 justify-start' : 'aspect-square justify-center'}
              `}
                title={t('signIn')}
              >
                <div className="flex-shrink-0"><LogIn size={20} /></div>
                {isOpen && (
                  <span className="text-sm font-medium whitespace-nowrap">{t('signIn')}</span>
                )}
              </button>
            )}

            {/* Quit / Shutdown */}
            {onQuit && (
              <button
                onClick={onQuit}
                className={`
                w-full rounded-xl flex items-center gap-3 transition-all duration-200 text-red-400 hover:text-red-300 hover:bg-red-500/10
                ${isOpen ? 'px-3 py-2.5 justify-start' : 'aspect-square justify-center'}
              `}
                title="Quit HOT-Step 9000"
              >
                <div className="flex-shrink-0"><Power size={20} /></div>
                {isOpen && (
                  <span className="text-sm font-medium whitespace-nowrap">Quit</span>
                )}
              </button>
            )}
          </div>
        </nav>
      </div>
    </>
  );
};

interface NavItemProps {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  onClick: () => void;
  isExpanded?: boolean;
}

const NavItem: React.FC<NavItemProps> = ({ icon, label, active, onClick, isExpanded }) => (
  <button
    onClick={onClick}
    className={`
      w-full rounded-xl flex items-center gap-3 transition-all duration-200 group relative overflow-hidden
      ${isExpanded ? 'px-3 py-2.5 justify-start' : 'aspect-square justify-center'}
      ${active ? 'bg-zinc-100 dark:bg-white/10 text-black dark:text-white' : 'text-zinc-500 hover:text-black dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-white/5'}
    `}
    title={label}
  >
    {active && <div className="absolute left-0 top-1/2 -translate-y-1/2 h-8 w-1 bg-pink-500 rounded-r-full"></div>}
    <div className="flex-shrink-0">{icon}</div>
    {isExpanded && (
      <span className="text-sm font-medium whitespace-nowrap">{label}</span>
    )}
  </button>
);
