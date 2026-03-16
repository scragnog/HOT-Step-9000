import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useAudioAnalysis } from '../context/AudioAnalysisContext';

interface WaveformVisualizerProps {
  audioUrl: string | null;
  currentTime: number;
  duration: number;
  progressBarRef?: React.RefObject<HTMLDivElement>;
  onSeek: (time: number) => void;
  /** AnalyserNode for bass-reactive bounce. Optional — no bounce when null. */
  analyserNode?: AnalyserNode | null;
  /** Whether audio is currently playing (enables rAF redraw loop). */
  isPlaying?: boolean;
  /** Bass bounce intensity 0-1. 0 = off, 1 = max bounce. */
  bounceIntensity?: number;
}

interface WaveformData {
  peaks: number[];
  length: number;
}

// Module-level cache: URL → peaks.  Keeps at most N entries to bound memory.
const WAVEFORM_CACHE_MAX = 30;
const waveformCache = new Map<string, WaveformData>();
function cacheSet(url: string, data: WaveformData) {
  if (waveformCache.size >= WAVEFORM_CACHE_MAX) {
    const firstKey = waveformCache.keys().next().value;
    if (firstKey) waveformCache.delete(firstKey);
  }
  waveformCache.set(url, data);
}

// Shared AudioContext — reused across all instances / re-renders
let sharedAudioCtx: AudioContext | null = null;
function getAudioContext(): AudioContext {
  if (!sharedAudioCtx || sharedAudioCtx.state === 'closed') {
    sharedAudioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
  }
  return sharedAudioCtx;
}

export const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({
  audioUrl,
  currentTime,
  duration,
  progressBarRef,
  onSeek,
  analyserNode,
  isPlaying = false,
  bounceIntensity = 0,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [waveformData, setWaveformData] = useState<WaveformData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const rafRef = useRef<number>(0);
  const smoothBassRef = useRef(0);
  const { resume } = useAudioAnalysis();

  // Generate waveform data from audio file
  useEffect(() => {
    if (!audioUrl) {
      setWaveformData(null);
      return;
    }

    const cached = waveformCache.get(audioUrl);
    if (cached) {
      setWaveformData(cached);
      setIsLoading(false);
      return;
    }

    if (abortRef.current) {
      abortRef.current.abort();
    }
    const controller = new AbortController();
    abortRef.current = controller;

    setIsLoading(true);

    const audioContext = getAudioContext();

    fetch(audioUrl, { signal: controller.signal })
      .then(response => {
        if (!response.ok) throw new Error('Failed to fetch audio');
        return response.arrayBuffer();
      })
      .then(arrayBuffer => {
        if (controller.signal.aborted) return;
        return audioContext.decodeAudioData(arrayBuffer);
      })
      .then(audioBuffer => {
        if (!audioBuffer || controller.signal.aborted) return;

        const channelData = audioBuffer.getChannelData(0);
        const samples = Math.min(1600, Math.floor(canvasRef.current?.clientWidth || 600) * 2);
        const blockSize = Math.floor(channelData.length / samples);
        const peaks: number[] = [];

        for (let i = 0; i < samples; i++) {
          let peak = 0;
          for (let j = 0; j < blockSize; j++) {
            const sample = Math.abs(channelData[i * blockSize + j] || 0);
            if (sample > peak) peak = sample;
          }
          peaks.push(peak);
        }

        const data: WaveformData = { peaks, length: audioBuffer.duration };
        cacheSet(audioUrl, data);
        if (!controller.signal.aborted) {
          setWaveformData(data);
          setIsLoading(false);
        }
      })
      .catch(error => {
        if (error?.name === 'AbortError') return;
        console.error('Failed to generate waveform:', error);
        if (!controller.signal.aborted) {
          setIsLoading(false);
        }
      });

    return () => {
      controller.abort();
    };
  }, [audioUrl]);

  // Draw waveform — extracted so it can be called from both static and rAF paths
  const drawWaveform = useCallback((bassMultiplier: number = 1) => {
    const canvas = canvasRef.current;
    if (!canvas || !waveformData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const centerY = height / 2;

    const totalBars = waveformData.peaks.length;
    const step = width / totalBars;
    const barWidth = Math.max(1, step * 0.65);
    const playedPercent = duration > 0 ? Math.max(0, Math.min(1, currentTime / duration)) : 0;
    const playedPixelWidth = playedPercent * width;

    const isDarkMode = document.documentElement.classList.contains('dark');

    ctx.clearRect(0, 0, width, height);

    waveformData.peaks.forEach((peak, index) => {
      const smoothedPeak = index > 0 && index < totalBars - 1
        ? (peak + waveformData.peaks[index - 1] + waveformData.peaks[index + 1]) / 3
        : peak;

      // Apply bass multiplier to bar height — grows symmetrically from center
      const baseBarHeight = Math.max(3, smoothedPeak * (height - 8));
      const barHeight = baseBarHeight * bassMultiplier;
      const x = index * step;

      const barCenterX = x + barWidth / 2;
      const isPlayed = barCenterX <= playedPixelWidth;

      if (isPlayed) {
        const gradient = ctx.createLinearGradient(0, centerY - barHeight / 2, 0, centerY + barHeight / 2);
        gradient.addColorStop(0, '#f43f5e');
        gradient.addColorStop(0.5, '#ec4899');
        gradient.addColorStop(1, '#f43f5e');
        ctx.fillStyle = gradient;
      } else {
        ctx.fillStyle = isDarkMode ? 'rgba(160, 160, 170, 0.4)' : 'rgba(100, 100, 110, 0.35)';
      }

      // Clamp height but allow it to fill the full canvas when boosted
      const barH = Math.min(barHeight, height);
      const y = centerY - barH / 2;

      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barH, 1);
      ctx.fill();
    });
  }, [waveformData, currentTime, duration]);

  // Static draw — when not playing or no analyser
  useEffect(() => {
    if (isPlaying && analyserNode && bounceIntensity > 0) return; // rAF loop handles it
    drawWaveform(1);
  }, [waveformData, currentTime, duration, isPlaying, analyserNode, bounceIntensity, drawWaveform]);

  // rAF loop — bass-reactive redraw when playing
  useEffect(() => {
    if (!isPlaying || !analyserNode || bounceIntensity <= 0 || !waveformData) {
      smoothBassRef.current = 0;
      return;
    }

    // Resume AudioContext if it was suspended by the browser
    resume();

    const freqData = new Uint8Array(analyserNode.frequencyBinCount);

    const tick = () => {
      analyserNode.getByteFrequencyData(freqData);

      // Average first 10 bins (~0-430Hz) for bass energy
      let bass = 0;
      for (let i = 0; i < 10; i++) bass += freqData[i];
      bass = (bass / 10) / 255; // normalize 0-1

      // Smooth with exponential decay — snappy attack, gentle release
      const prev = smoothBassRef.current;
      smoothBassRef.current = bass > prev
        ? prev + (bass - prev) * 0.5    // fast attack
        : prev + (bass - prev) * 0.06;  // slow release

      // Bass multiplier: intensity controls how much the bars grow
      // At intensity=1, bars can grow up to 2.0x their normal height
      const bassMultiplier = 1 + smoothBassRef.current * bounceIntensity * 1.0;
      drawWaveform(bassMultiplier);

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(rafRef.current);
    };
  }, [isPlaying, analyserNode, bounceIntensity, waveformData, drawWaveform, resume]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!duration) return;

    const container = progressBarRef?.current || canvasRef.current;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percent = Math.max(0, Math.min(1, x / rect.width));
    onSeek(percent * duration);
  };

  if (isLoading || !waveformData) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="w-full h-0.5 bg-zinc-300/20 dark:bg-zinc-600/20" />
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      className="w-full h-full cursor-pointer"
      style={{ width: '100%', height: '100%' }}
    />
  );
};
