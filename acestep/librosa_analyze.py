import sys
import json
import numpy as np
import librosa

def get_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_vals = np.sum(chroma, axis=1)
    
    # Krumhansl-Schmuckler profiles
    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    maj_corrs = [np.corrcoef(chroma_vals, np.roll(maj_profile, i))[0,1] for i in range(12)]
    min_corrs = [np.corrcoef(chroma_vals, np.roll(min_profile, i))[0,1] for i in range(12)]
    
    max_maj = max(maj_corrs)
    max_min = max(min_corrs)
    
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    if max_maj > max_min:
        return keys[maj_corrs.index(max_maj)], "major"
    else:
        return keys[min_corrs.index(max_min)], "minor"

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing audio path"}))
        sys.exit(1)
        
    path = sys.argv[1]
    
    try:
        y, sr = librosa.load(path, sr=None)
        
        # In newer librosa (0.10+), librosa.beat.tempo is deprecated for librosa.feature.tempo
        if hasattr(librosa.feature, 'tempo'):
            tempo = librosa.feature.tempo(y=y, sr=sr)[0]
        else:
            tempo = librosa.beat.tempo(y=y, sr=sr)[0]
            
        key, scale = get_key(y, sr)
        
        print(json.dumps({
            "rhythm": {"bpm": float(tempo)},
            "tonal": {"key_edma": {"key": key, "scale": scale}}
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
