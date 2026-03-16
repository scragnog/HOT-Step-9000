// AudioAnalysisContext.tsx — Shared audio analysis for live visualizers
// Provides an AnalyserNode connected to the main player's HTMLAudioElement

import React, { createContext, useContext, useRef, useCallback, useState } from 'react';

interface AudioAnalysisContextValue {
    /** Connect the analyser to an audio element. Call once on first play. */
    connect: (audioElement: HTMLAudioElement) => void;
    /** Resume the AudioContext if it was suspended by the browser. */
    resume: () => void;
    /** The AnalyserNode, null until connect() is called. */
    analyserNode: AnalyserNode | null;
    /** Whether the analyser is connected to an audio element. */
    isConnected: boolean;
}

const AudioAnalysisContext = createContext<AudioAnalysisContextValue>({
    connect: () => { },
    resume: () => { },
    analyserNode: null,
    isConnected: false,
});

export const useAudioAnalysis = () => useContext(AudioAnalysisContext);

interface AudioAnalysisProviderProps {
    children: React.ReactNode;
}

export const AudioAnalysisProvider: React.FC<AudioAnalysisProviderProps> = ({ children }) => {
    const audioContextRef = useRef<AudioContext | null>(null);
    const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const connectedElementRef = useRef<HTMLAudioElement | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);

    // Resume a suspended AudioContext — call this from visualizer animation loops
    // to recover from browser power-saving suspension during long idle periods.
    const resume = useCallback(() => {
        const ctx = audioContextRef.current;
        if (ctx && ctx.state === 'suspended') {
            ctx.resume().catch(console.error);
        }
    }, []);

    const connect = useCallback((audioElement: HTMLAudioElement) => {
        // Already connected to this element — skip (use ref to avoid stale closure)
        if (connectedElementRef.current === audioElement && analyserRef.current) {
            // Still resume in case the context was suspended
            if (audioContextRef.current?.state === 'suspended') {
                audioContextRef.current.resume().catch(console.error);
            }
            return;
        }

        // createMediaElementSource can only be called ONCE per element.
        // If we already connected a different element, we'd need a new AudioContext.
        // In practice, App.tsx creates one Audio element for the lifetime of the app.

        try {
            if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
                const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
                audioContextRef.current = new AudioContextClass();
            }

            const audioCtx = audioContextRef.current;

            // Resume if suspended (browser autoplay policy)
            if (audioCtx.state === 'suspended') {
                audioCtx.resume().catch(console.error);
            }

            // Create analyser
            const analyser = audioCtx.createAnalyser();
            analyser.fftSize = 2048;
            analyserRef.current = analyser;
            setAnalyserNode(analyser);

            // Connect: element → source → analyser → destination
            // createMediaElementSource can only be called once per element
            if (!sourceRef.current) {
                const source = audioCtx.createMediaElementSource(audioElement);
                sourceRef.current = source;
                source.connect(analyser);
                analyser.connect(audioCtx.destination);
            } else {
                // If source already exists (reconnecting), just rewire the analyser
                sourceRef.current.disconnect();
                sourceRef.current.connect(analyser);
                analyser.connect(audioCtx.destination);
            }

            connectedElementRef.current = audioElement;
            setIsConnected(true);
        } catch (err) {
            console.error('AudioAnalysis: Failed to connect', err);
        }
    }, []);

    return (
        <AudioAnalysisContext.Provider value={{
            connect,
            resume,
            analyserNode,
            isConnected,
        }}>
            {children}
        </AudioAnalysisContext.Provider>
    );
};
