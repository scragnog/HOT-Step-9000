import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Bug, ChevronLeft, ChevronRight, ChevronUp, ChevronDown, Cpu, MemoryStick, Monitor, Search, Trash2, X } from 'lucide-react';

interface SystemMetrics {
    gpu: {
        name?: string;
        allocated_gb?: number;
        reserved_gb?: number;
        free_gb?: number;
        total_gb?: number;
        error?: string;
    };
    ram: {
        used_gb?: number;
        total_gb?: number;
        percent?: number;
        error?: string;
    };
    cpu: {
        percent?: number;
        count?: number;
        error?: string;
    };
}

function MetricBar({
    label,
    value,
    max,
    unit = 'GB',
    percent,
    icon: Icon,
    color,
}: {
    label: string;
    value: number;
    max: number;
    unit?: string;
    percent?: number;
    icon: React.ElementType;
    color: string;
}) {
    const pct = percent ?? (max > 0 ? (value / max) * 100 : 0);
    const barColor =
        pct > 90
            ? 'bg-red-500'
            : pct > 70
                ? 'bg-amber-500'
                : color;

    return (
        <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-1.5 text-zinc-400">
                    <Icon size={12} />
                    {label}
                </span>
                <span className="text-zinc-300 font-mono text-[11px]">
                    {value.toFixed(1)} / {max.toFixed(1)} {unit}
                </span>
            </div>
            <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                <div
                    className={`h-full rounded-full transition-all duration-500 ${barColor}`}
                    style={{ width: `${Math.min(100, pct)}%` }}
                />
            </div>
        </div>
    );
}

function getLogLevelColor(line: string): string {
    const upper = line.toUpperCase();
    if (upper.includes('| ERROR |') || upper.includes('ERROR:') || upper.includes('TRACEBACK'))
        return 'text-red-400';
    if (upper.includes('| WARNING |') || upper.includes('WARNING:'))
        return 'text-amber-400';
    if (upper.includes('| DEBUG |') || upper.includes('DEBUG:'))
        return 'text-green-800';
    if (upper.includes('| SUCCESS |'))
        return 'text-green-300';
    return 'text-green-400';
}

/** Render a log line with search matches highlighted */
function HighlightedLine({ line, query, isCurrentMatch }: { line: string; query: string; isCurrentMatch: boolean }) {
    if (!query) return <>{line}</>;

    const lowerLine = line.toLowerCase();
    const lowerQuery = query.toLowerCase();
    const parts: React.ReactNode[] = [];
    let lastIdx = 0;
    let idx = lowerLine.indexOf(lowerQuery);

    while (idx !== -1) {
        if (idx > lastIdx) parts.push(line.slice(lastIdx, idx));
        parts.push(
            <mark
                key={idx}
                className={`${isCurrentMatch ? 'bg-amber-400 text-black' : 'bg-yellow-600/60 text-yellow-100'} rounded-sm px-px`}
            >
                {line.slice(idx, idx + query.length)}
            </mark>
        );
        lastIdx = idx + query.length;
        idx = lowerLine.indexOf(lowerQuery, lastIdx);
    }
    if (lastIdx < line.length) parts.push(line.slice(lastIdx));

    return <>{parts}</>;
}

export default function DebugPanel({ isOpen, onToggle }: { isOpen: boolean; onToggle: () => void }) {
    const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
    const [logLines, setLogLines] = useState<string[]>([]);
    const [apiReachable, setApiReachable] = useState(true);
    const logEndRef = useRef<HTMLDivElement>(null);
    const logContainerRef = useRef<HTMLDivElement>(null);
    const [autoScroll, setAutoScroll] = useState(true);
    const initialScrollDone = useRef(false);

    // Search state
    const [searchQuery, setSearchQuery] = useState('');
    const [showSearch, setShowSearch] = useState(false);
    const [currentMatchIdx, setCurrentMatchIdx] = useState(0);
    const searchInputRef = useRef<HTMLInputElement>(null);
    const lineRefs = useRef<Map<number, HTMLDivElement>>(new Map());

    // Compute matching line indices
    const matchingLineIndices = useMemo(() => {
        if (!searchQuery.trim()) return [];
        const q = searchQuery.toLowerCase();
        const indices: number[] = [];
        logLines.forEach((line, i) => {
            if (line.toLowerCase().includes(q)) indices.push(i);
        });
        return indices;
    }, [searchQuery, logLines]);

    // Clamp currentMatchIdx when matches change
    useEffect(() => {
        if (matchingLineIndices.length === 0) {
            setCurrentMatchIdx(0);
        } else if (currentMatchIdx >= matchingLineIndices.length) {
            setCurrentMatchIdx(matchingLineIndices.length - 1);
        }
    }, [matchingLineIndices.length, currentMatchIdx]);

    // Scroll to current match
    useEffect(() => {
        if (matchingLineIndices.length === 0) return;
        const lineIdx = matchingLineIndices[currentMatchIdx];
        const el = lineRefs.current.get(lineIdx);
        if (el) {
            setAutoScroll(false);
            el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }, [currentMatchIdx, matchingLineIndices]);

    // Focus search input when opened
    useEffect(() => {
        if (showSearch) {
            requestAnimationFrame(() => searchInputRef.current?.focus());
        }
    }, [showSearch]);

    const navigateMatch = useCallback((dir: 'next' | 'prev') => {
        if (matchingLineIndices.length === 0) return;
        setCurrentMatchIdx(prev => {
            if (dir === 'next') return (prev + 1) % matchingLineIndices.length;
            return (prev - 1 + matchingLineIndices.length) % matchingLineIndices.length;
        });
    }, [matchingLineIndices.length]);

    // Keyboard shortcut: Ctrl+F to open search, Escape to close, Enter/Shift+Enter to navigate
    useEffect(() => {
        if (!isOpen) return;
        const handler = (e: KeyboardEvent) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
                e.preventDefault();
                setShowSearch(true);
            }
            if (e.key === 'Escape' && showSearch) {
                setShowSearch(false);
                setSearchQuery('');
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [isOpen, showSearch]);

    // Scroll the log container to the bottom
    const scrollToBottom = useCallback((instant = false) => {
        const el = logContainerRef.current;
        if (!el) return;
        if (instant) {
            el.scrollTop = el.scrollHeight;
        } else {
            el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
        }
    }, []);

    // Auto-scroll log when new lines arrive
    useEffect(() => {
        if (!autoScroll) return;
        if (!initialScrollDone.current && logLines.length > 0) {
            initialScrollDone.current = true;
            requestAnimationFrame(() => scrollToBottom(true));
            setTimeout(() => scrollToBottom(true), 350);
        } else if (initialScrollDone.current) {
            requestAnimationFrame(() => scrollToBottom(false));
        }
    }, [logLines, autoScroll, scrollToBottom]);

    // Reset initial scroll flag when panel closes
    useEffect(() => {
        if (!isOpen) {
            initialScrollDone.current = false;
        }
    }, [isOpen]);

    // Detect manual scroll to pause auto-scroll
    const handleLogScroll = useCallback(() => {
        const el = logContainerRef.current;
        if (!el) return;
        const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
        setAutoScroll(isAtBottom);
    }, []);

    // Poll system metrics every 2s when panel is open
    useEffect(() => {
        if (!isOpen) return;
        let alive = true;

        const poll = async () => {
            while (alive) {
                try {
                    const res = await fetch('/api/system/metrics');
                    if (res.ok) {
                        const data = await res.json();
                        setMetrics(data);
                        setApiReachable(true);
                    } else {
                        setApiReachable(false);
                    }
                } catch {
                    setApiReachable(false);
                }
                await new Promise((r) => setTimeout(r, 2000));
            }
        };

        poll();
        return () => { alive = false; };
    }, [isOpen]);

    // SSE log stream when panel is open
    useEffect(() => {
        if (!isOpen) return;
        let alive = true;
        let eventSource: EventSource | null = null;

        const connect = () => {
            eventSource = new EventSource('/api/system/logs');

            eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.lines && data.lines.length > 0) {
                        setLogLines((prev) => {
                            const next = [...prev, ...data.lines];
                            // Keep last 1000 lines in memory
                            return next.length > 1000 ? next.slice(-1000) : next;
                        });
                    }
                } catch {
                    // ignore parse errors
                }
            };

            eventSource.onerror = () => {
                if (alive) {
                    eventSource?.close();
                    // Retry after 3s
                    setTimeout(() => {
                        if (alive) connect();
                    }, 3000);
                }
            };
        };

        connect();

        return () => {
            alive = false;
            eventSource?.close();
        };
    }, [isOpen]);

    // Determine which line is the "current" match for highlighting
    const currentMatchLineIdx = matchingLineIndices.length > 0 ? matchingLineIndices[currentMatchIdx] : -1;

    return (
        <>
            {/* Toggle tab — always visible on right edge */}
            <button
                onClick={onToggle}
                className={`
          fixed right-0 top-1/2 -translate-y-1/2 z-[90]
          flex items-center justify-center
          w-6 h-16 rounded-l-lg
          bg-zinc-800/90 hover:bg-zinc-700/90
          border border-r-0 border-zinc-600/50
          text-zinc-400 hover:text-zinc-200
          backdrop-blur-sm
          transition-all duration-200
          ${isOpen ? 'right-[400px]' : 'right-0'}
        `}
                title={isOpen ? 'Close debug panel' : 'Open debug panel'}
            >
                {isOpen ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
            </button>

            {/* Panel */}
            <div
                className={`
          fixed top-0 right-0 z-[80]
          h-full w-[400px]
          bg-zinc-900/95 backdrop-blur-md
          border-l border-zinc-700/50
          flex flex-col
          transition-transform duration-300 ease-in-out
          ${isOpen ? 'translate-x-0' : 'translate-x-full'}
        `}
            >
                {/* Header */}
                <div className="flex items-center gap-2 px-4 py-3 border-b border-zinc-700/50 flex-shrink-0">
                    <Bug size={16} className="text-emerald-400" />
                    <h2 className="text-sm font-semibold text-zinc-200">Debug Panel</h2>
                    {!apiReachable && (
                        <span className="ml-auto text-[10px] px-2 py-0.5 rounded-full bg-red-900/50 text-red-400 border border-red-700/30">
                            API Offline
                        </span>
                    )}
                </div>

                {/* Metrics section */}
                <div className="px-4 py-3 border-b border-zinc-700/50 space-y-3 flex-shrink-0">
                    <h3 className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
                        System Metrics
                    </h3>

                    {metrics ? (
                        <div className="space-y-3">
                            {/* GPU */}
                            {metrics.gpu.error ? (
                                <div className="text-xs text-zinc-500 italic">
                                    GPU: {metrics.gpu.error}
                                </div>
                            ) : (
                                <>
                                    <div className="text-[10px] text-zinc-500 font-mono">
                                        {metrics.gpu.name || 'GPU'}
                                    </div>
                                    <MetricBar
                                        label="VRAM"
                                        value={metrics.gpu.allocated_gb ?? 0}
                                        max={metrics.gpu.total_gb ?? 0}
                                        icon={Monitor}
                                        color="bg-emerald-500"
                                    />
                                    <MetricBar
                                        label="VRAM Reserved"
                                        value={metrics.gpu.reserved_gb ?? 0}
                                        max={metrics.gpu.total_gb ?? 0}
                                        icon={Monitor}
                                        color="bg-teal-500/60"
                                    />
                                </>
                            )}

                            {/* RAM */}
                            {metrics.ram.error ? (
                                <div className="text-xs text-zinc-500 italic">
                                    RAM: {metrics.ram.error}
                                </div>
                            ) : (
                                <MetricBar
                                    label="RAM"
                                    value={metrics.ram.used_gb ?? 0}
                                    max={metrics.ram.total_gb ?? 0}
                                    percent={metrics.ram.percent}
                                    icon={MemoryStick}
                                    color="bg-blue-500"
                                />
                            )}

                            {/* CPU */}
                            {metrics.cpu.error ? (
                                <div className="text-xs text-zinc-500 italic">
                                    CPU: {metrics.cpu.error}
                                </div>
                            ) : (
                                <MetricBar
                                    label={`CPU (${metrics.cpu.count ?? '?'} cores)`}
                                    value={metrics.cpu.percent ?? 0}
                                    max={100}
                                    unit="%"
                                    percent={metrics.cpu.percent}
                                    icon={Cpu}
                                    color="bg-violet-500"
                                />
                            )}
                        </div>
                    ) : (
                        <div className="text-xs text-zinc-500 italic animate-pulse">
                            Connecting to API...
                        </div>
                    )}
                </div>

                {/* Log section — fills remaining */}
                <div className="flex-1 flex flex-col min-h-0">
                    {/* Log header with controls */}
                    <div className="flex items-center justify-between px-4 py-2 flex-shrink-0">
                        <h3 className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
                            API Log
                        </h3>
                        <div className="flex items-center gap-2">
                            {!autoScroll && (
                                <button
                                    onClick={() => {
                                        setAutoScroll(true);
                                        logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
                                    }}
                                    className="text-[10px] text-zinc-500 hover:text-zinc-300 transition-colors"
                                >
                                    ↓ Bottom
                                </button>
                            )}
                            <button
                                onClick={() => { setShowSearch(s => !s); if (showSearch) setSearchQuery(''); }}
                                className={`p-1 rounded transition-colors ${showSearch ? 'text-amber-400 bg-amber-900/30' : 'text-zinc-500 hover:text-zinc-300'}`}
                                title="Search log (Ctrl+F)"
                            >
                                <Search size={12} />
                            </button>
                            <button
                                onClick={() => { setLogLines([]); setSearchQuery(''); setCurrentMatchIdx(0); }}
                                className="p-1 rounded text-zinc-500 hover:text-red-400 transition-colors"
                                title="Clear log"
                            >
                                <Trash2 size={12} />
                            </button>
                            <span className="text-[10px] text-zinc-600 font-mono">
                                {logLines.length}
                            </span>
                        </div>
                    </div>

                    {/* Search toolbar */}
                    {showSearch && (
                        <div className="flex items-center gap-1.5 px-3 pb-2 flex-shrink-0">
                            <div className="flex-1 flex items-center bg-black border border-zinc-700 rounded-lg overflow-hidden">
                                <Search size={12} className="text-zinc-500 ml-2 flex-shrink-0" />
                                <input
                                    ref={searchInputRef}
                                    type="text"
                                    value={searchQuery}
                                    onChange={(e) => { setSearchQuery(e.target.value); setCurrentMatchIdx(0); }}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter') {
                                            e.preventDefault();
                                            navigateMatch(e.shiftKey ? 'prev' : 'next');
                                        }
                                        if (e.key === 'Escape') {
                                            setShowSearch(false);
                                            setSearchQuery('');
                                        }
                                    }}
                                    placeholder="Search logs..."
                                    className="flex-1 bg-transparent text-xs text-zinc-200 placeholder-zinc-600 px-2 py-1.5 outline-none"
                                />
                                {searchQuery && (
                                    <span className="text-[10px] text-zinc-500 font-mono px-1 flex-shrink-0">
                                        {matchingLineIndices.length > 0
                                            ? `${currentMatchIdx + 1}/${matchingLineIndices.length}`
                                            : '0/0'}
                                    </span>
                                )}
                            </div>
                            <button
                                onClick={() => navigateMatch('prev')}
                                disabled={matchingLineIndices.length === 0}
                                className="p-1 rounded text-zinc-400 hover:text-zinc-200 disabled:text-zinc-700 disabled:cursor-not-allowed transition-colors"
                                title="Previous match (Shift+Enter)"
                            >
                                <ChevronUp size={14} />
                            </button>
                            <button
                                onClick={() => navigateMatch('next')}
                                disabled={matchingLineIndices.length === 0}
                                className="p-1 rounded text-zinc-400 hover:text-zinc-200 disabled:text-zinc-700 disabled:cursor-not-allowed transition-colors"
                                title="Next match (Enter)"
                            >
                                <ChevronDown size={14} />
                            </button>
                            <button
                                onClick={() => { setShowSearch(false); setSearchQuery(''); }}
                                className="p-1 rounded text-zinc-500 hover:text-zinc-300 transition-colors"
                                title="Close search (Esc)"
                            >
                                <X size={14} />
                            </button>
                        </div>
                    )}

                    <div
                        ref={logContainerRef}
                        onScroll={handleLogScroll}
                        className="flex-1 overflow-y-auto overflow-x-hidden mx-3 mb-3 p-3 rounded border border-green-800/60 bg-black font-mono text-[11px] leading-[1.6] scroll-smooth shadow-[inset_0_0_20px_rgba(0,255,0,0.03)]"
                    >
                        {logLines.length === 0 ? (
                            <div className="text-green-700 italic text-center py-8">
                                {'>'} Waiting for log output...
                                <span className="animate-pulse">_</span>
                            </div>
                        ) : (
                            logLines.map((line, i) => (
                                <div
                                    key={i}
                                    ref={(el) => { if (el) lineRefs.current.set(i, el); else lineRefs.current.delete(i); }}
                                    className={`whitespace-pre-wrap break-all py-px ${getLogLevelColor(line)} ${i === currentMatchLineIdx ? 'bg-amber-900/20 rounded' : ''}`}
                                >
                                    <HighlightedLine
                                        line={line}
                                        query={searchQuery}
                                        isCurrentMatch={i === currentMatchLineIdx}
                                    />
                                </div>
                            ))
                        )}
                        <div ref={logEndRef} />
                    </div>
                </div>
            </div>
        </>
    );
}
