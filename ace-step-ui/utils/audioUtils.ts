// Audio utility functions extracted from CreatePanel.tsx

/** Extract a clean label from an audio URL */
export const getAudioLabel = (url: string): string => {
  try {
    const parsed = new URL(url);
    const name = decodeURIComponent(parsed.pathname.split('/').pop() || parsed.hostname);
    return name.replace(/\.[^/.]+$/, '') || name;
  } catch {
    const parts = url.split('/');
    const name = decodeURIComponent(parts[parts.length - 1] || url);
    return name.replace(/\.[^/.]+$/, '') || name;
  }
};

/** Format seconds into m:ss display string */
export const formatTime = (time: number): string => {
  if (!Number.isFinite(time) || time <= 0) return '0:00';
  const minutes = Math.floor(time / 60);
  const seconds = Math.floor(time % 60);
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
};

const CHROMATIC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const FLAT_TO_SHARP: Record<string, string> = {
  'Cb': 'B', 'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
};

/**
 * Calculate effective BPM after cover tempo scaling.
 * Returns 0 for extract tasks, scaled BPM for cover/repaint, or original BPM.
 */
export const computeEffectiveBpm = (
  taskType: string,
  bpm: number,
  tempoScale: number,
): number => {
  if (taskType === 'extract') return 0;
  if (bpm && tempoScale !== 1.0) return Math.round(bpm * tempoScale);
  return bpm;
};

/**
 * Calculate effective key signature after pitch shifting.
 * Returns '' for extract tasks, transposed key for cover/repaint, or original key.
 */
export const computeEffectiveKeyScale = (
  taskType: string,
  keyScale: string,
  pitchShift: number,
): string => {
  if (taskType === 'extract') return '';
  if (keyScale && pitchShift !== 0) {
    const parts = keyScale.split(' ');
    const note = parts[0];
    const normalizedNote = FLAT_TO_SHARP[note] || note;
    const idx = CHROMATIC.indexOf(normalizedNote);
    if (idx !== -1) {
      const newIdx = ((idx + pitchShift) % 12 + 12) % 12;
      return `${CHROMATIC[newIdx]}${parts.length > 1 ? ' ' + parts.slice(1).join(' ') : ''}`;
    }
  }
  return keyScale;
};
