#!/usr/bin/env python
"""
Stem-by-Stem Matchering Prototype
===================================
Splits both a reference audio and a generated audio into 6 stems using
BS-Roformer-SW, merges guitar/piano/other into a single "other" group,
runs matchering on each stem pair (gen → ref), then recombines the
generated stems into a final mastered track.

This is a standalone prototype to evaluate whether per-stem matchering
produces better results than full-mix matchering.

Usage:
    python scripts/test_stem_matchering.py --reference "path/to/reference.wav" --generation "path/to/generation.wav"
    python scripts/test_stem_matchering.py --reference ref.wav --generation gen.wav --also-full-mix
    python scripts/test_stem_matchering.py --reference ref.wav --generation gen.wav --save-stems

The --also-full-mix flag also runs standard full-mix matchering for A/B comparison.
The --save-stems flag saves individual stem pairs for inspection.

Outputs saved alongside the generation file with '_stem_mastered' suffix.
"""

import argparse
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# BFloat16 workaround (same as stem_service.py)
# ---------------------------------------------------------------------------
@contextmanager
def _float32_default_dtype():
    try:
        import torch
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            yield
        finally:
            torch.set_default_dtype(prev)
    except ImportError:
        yield


# ---------------------------------------------------------------------------
# Stem separation using BS-Roformer-SW (6 stems)
# ---------------------------------------------------------------------------

ROFORMER_SW_MODEL = "BS-Roformer-SW.ckpt"
# The 6 raw stems from BS-Roformer-SW
RAW_STEM_TYPES = ("vocals", "drums", "bass", "guitar", "piano", "other")
# Guitar, piano, and other are merged into a single "other" group for matchering
# (these stems are often too noisy/sparse individually for matchering to handle well)
MERGE_INTO_OTHER = ("guitar", "piano", "other")
# The 4 stem groups we actually run matchering on
MATCHERING_GROUPS = ("vocals", "drums", "bass", "other")


def classify_stem_type(filename_lower: str) -> str:
    """Classify a stem file by its name."""
    for stem in RAW_STEM_TYPES:
        if stem in filename_lower:
            return stem
    return "other"


def separate_to_stems(
    audio_path: str,
    output_dir: str,
    label: str = "",
) -> Tuple[Dict[str, str], int]:
    """Separate audio into 6 stems using BS-Roformer-SW.

    Returns (stems_dict, sample_rate) where stems_dict maps
    stem_type -> file_path.
    """
    from audio_separator.separator import Separator

    prefix = f"[{label}] " if label else ""
    print(f"  {prefix}Loading audio_separator...")
    sep = Separator()
    sep.output_format = "wav"
    sep.output_dir = output_dir

    with _float32_default_dtype():
        print(f"  {prefix}Loading BS-Roformer-SW model...")
        sep.load_model(model_filename=ROFORMER_SW_MODEL)

        print(f"  {prefix}Separating into 6 stems...")
        files = sep.separate(audio_path)

    stems: Dict[str, str] = {}
    stem_sr = None

    for fp in files:
        fp_path = Path(fp) if Path(fp).is_absolute() else Path(output_dir) / fp
        fp_str = str(fp_path)
        stem_type = classify_stem_type(fp_path.stem.lower())

        # Read just to get sample rate and verify
        info = sf.info(fp_str)
        if stem_sr is None:
            stem_sr = info.samplerate

        stems[stem_type] = fp_str
        print(f"    {prefix}{stem_type}: {fp_path.name} ({info.duration:.1f}s, {info.samplerate}Hz)")

    # Merge guitar + piano + other into a single "other" stem
    stems, stem_sr_final = _merge_minor_stems(stems, stem_sr or 44100, output_dir, label)
    return stems, stem_sr_final


def _merge_minor_stems(
    stems: Dict[str, str],
    stem_sr: int,
    output_dir: str,
    label: str = "",
) -> Tuple[Dict[str, str], int]:
    """Merge guitar, piano, and other stems into a single 'other' stem.

    Returns updated stems dict with only vocals/drums/bass/other keys.
    """
    prefix = f"[{label}] " if label else ""
    to_merge = []
    for stem_type in MERGE_INTO_OTHER:
        if stem_type in stems:
            to_merge.append((stem_type, stems[stem_type]))

    if not to_merge:
        return stems, stem_sr

    # Load and sum
    combined = None
    sr_out = stem_sr
    for stem_type, path in to_merge:
        data, sr = sf.read(path, dtype="float32")
        if data.ndim == 1:
            data = np.column_stack((data, data))
        if sr != sr_out:
            import librosa
            data_t = data.T
            resampled = [librosa.resample(data_t[ch], orig_sr=sr, target_sr=sr_out) for ch in range(data_t.shape[0])]
            data = np.stack(resampled, axis=-1)
        if combined is None:
            combined = data.copy()
        else:
            # Pad/trim to match
            if data.shape[0] < combined.shape[0]:
                data = np.pad(data, ((0, combined.shape[0] - data.shape[0]), (0, 0)))
            elif data.shape[0] > combined.shape[0]:
                combined = np.pad(combined, ((0, data.shape[0] - combined.shape[0]), (0, 0)))
            combined += data

    merge_names = [s[0] for s in to_merge]
    print(f"    {prefix}Merged {'+'.join(merge_names)} → other")

    # Write combined "other" stem
    merged_path = os.path.join(output_dir, "merged_other.wav")
    sf.write(merged_path, combined, sr_out)

    # Build new stems dict with only the matchering groups
    new_stems: Dict[str, str] = {}
    for key in ("vocals", "drums", "bass"):
        if key in stems:
            new_stems[key] = stems[key]
    new_stems["other"] = merged_path

    return new_stems, sr_out


# ---------------------------------------------------------------------------
# Per-stem matchering
# ---------------------------------------------------------------------------

def matchering_single_stem(
    target_path: str,
    reference_path: str,
    output_path: str,
    stem_name: str,
) -> bool:
    """Run matchering on a single stem pair.

    Returns True on success, False on failure (e.g. silent stem).
    """
    import matchering as mg

    try:
        # Check if the target stem is near-silent (matchering will fail on silence)
        target_data, _ = sf.read(target_path, dtype="float32")
        rms = np.sqrt(np.mean(target_data ** 2))
        if rms < 1e-5:
            print(f"    [{stem_name}] Skipping — near-silent (RMS={rms:.2e})")
            # Copy the target as-is (silence stays silence)
            sf.write(output_path, target_data, sf.info(target_path).samplerate)
            return True

        # Also check reference
        ref_data, _ = sf.read(reference_path, dtype="float32")
        ref_rms = np.sqrt(np.mean(ref_data ** 2))
        if ref_rms < 1e-5:
            print(f"    [{stem_name}] Skipping — reference stem near-silent (RMS={ref_rms:.2e})")
            sf.write(output_path, target_data, sf.info(target_path).samplerate)
            return True

        mg.process(
            target=target_path,
            reference=reference_path,
            results=[mg.pcm16(output_path)],
        )
        return True

    except Exception as e:
        print(f"    [{stem_name}] Matchering FAILED: {e}")
        # Fallback: copy target stem unprocessed
        try:
            data, sr = sf.read(target_path, dtype="float32")
            sf.write(output_path, data, sr)
        except Exception:
            pass
        return False


def run_stem_matchering(
    generation_path: str,
    reference_path: str,
    output_path: str,
    save_stems: bool = False,
    stems_output_dir: str = None,
) -> bool:
    """Full stem-by-stem matchering pipeline.

    1. Separate reference into 6 stems
    2. Separate generation into 6 stems
    3. Run matchering on each stem pair
    4. Recombine the mastered generation stems

    Returns True on success.
    """
    t_start = time.time()

    with tempfile.TemporaryDirectory(prefix="stem_match_") as tmp_dir:
        ref_stem_dir = os.path.join(tmp_dir, "ref_stems")
        gen_stem_dir = os.path.join(tmp_dir, "gen_stems")
        matched_dir = os.path.join(tmp_dir, "matched_stems")
        os.makedirs(ref_stem_dir, exist_ok=True)
        os.makedirs(gen_stem_dir, exist_ok=True)
        os.makedirs(matched_dir, exist_ok=True)

        # --- Step 1: Separate reference ---
        print("\n--- Step 1/4: Separating REFERENCE into stems ---")
        ref_stems, ref_sr = separate_to_stems(reference_path, ref_stem_dir, label="REF")

        # --- Step 2: Separate generation ---
        print("\n--- Step 2/4: Separating GENERATION into stems ---")
        gen_stems, gen_sr = separate_to_stems(generation_path, gen_stem_dir, label="GEN")

        # --- Step 3: Match each stem pair ---
        print("\n--- Step 3/4: Running matchering per stem ---")
        matched_stems: Dict[str, str] = {}
        results: Dict[str, str] = {}

        for stem_type in MATCHERING_GROUPS:
            gen_stem_path = gen_stems.get(stem_type)
            ref_stem_path = ref_stems.get(stem_type)

            if not gen_stem_path:
                print(f"  [{stem_type}] No generation stem found — skipping")
                results[stem_type] = "MISSING_GEN"
                continue

            if not ref_stem_path:
                print(f"  [{stem_type}] No reference stem found — using unmatched generation")
                matched_stems[stem_type] = gen_stem_path
                results[stem_type] = "NO_REF"
                continue

            out_path = os.path.join(matched_dir, f"{stem_type}_matched.wav")
            print(f"  [{stem_type}] Matchering: gen → ref...")

            t_stem = time.time()
            success = matchering_single_stem(gen_stem_path, ref_stem_path, out_path, stem_type)
            elapsed = time.time() - t_stem

            if success and os.path.exists(out_path):
                matched_stems[stem_type] = out_path
                results[stem_type] = f"OK ({elapsed:.1f}s)"
                print(f"    [{stem_type}] ✓ Done in {elapsed:.1f}s")
            else:
                # Fallback to unmatched generation stem
                matched_stems[stem_type] = gen_stem_path
                results[stem_type] = f"FALLBACK ({elapsed:.1f}s)"
                print(f"    [{stem_type}] ⚠ Fallback to unmatched stem")

        # --- Print results summary ---
        print("\n  Matchering Results:")
        for stem_type, result in results.items():
            print(f"    {stem_type:>8}: {result}")

        # --- Step 4: Recombine matched stems ---
        print("\n--- Step 4/4: Recombining matched stems ---")

        # Load the generation to get target sample rate and length
        gen_data, gen_original_sr = sf.read(generation_path, dtype="float32")
        if gen_data.ndim == 1:
            gen_data = np.column_stack((gen_data, gen_data))
        target_samples = gen_data.shape[0]
        target_sr = gen_original_sr

        # Sum all matched stems
        combined = None
        for stem_type, stem_path in matched_stems.items():
            data, sr = sf.read(stem_path, dtype="float32")
            if data.ndim == 1:
                data = np.column_stack((data, data))

            # Resample if needed (matchering outputs at 44100 by default)
            if sr != target_sr:
                import librosa
                print(f"  Resampling {stem_type} from {sr}Hz to {target_sr}Hz...")
                # librosa expects [channels, samples]
                data_t = data.T
                resampled = []
                for ch in range(data_t.shape[0]):
                    resampled.append(librosa.resample(data_t[ch], orig_sr=sr, target_sr=target_sr))
                data = np.stack(resampled, axis=-1)  # back to [samples, channels]

            # Ensure same length (pad or trim)
            if data.shape[0] < target_samples:
                pad = np.zeros((target_samples - data.shape[0], data.shape[1]))
                data = np.concatenate([data, pad], axis=0)
            elif data.shape[0] > target_samples:
                data = data[:target_samples]

            if combined is None:
                combined = data.copy()
            else:
                combined += data

        if combined is None:
            print("  ERROR: No stems to combine!")
            return False

        # --- LUFS loudness matching against the reference track ---
        # Per-stem matchering makes each stem match its (quiet) reference
        # stem, so the recombined sum is much quieter than the full
        # reference mix.  Fix: measure reference LUFS and match it.
        import pyloudnorm as pyln
        from pedalboard import Pedalboard, Limiter, Gain

        meter = pyln.Meter(target_sr)

        # Measure reference loudness
        ref_data, ref_sr_file = sf.read(reference_path, dtype="float32")
        if ref_data.ndim == 1:
            ref_data = np.column_stack((ref_data, ref_data))
        if ref_sr_file != target_sr:
            import librosa
            ref_data = librosa.resample(ref_data.T, orig_sr=ref_sr_file, target_sr=target_sr).T
        ref_lufs = meter.integrated_loudness(ref_data)

        # Measure combined loudness
        combined_lufs = meter.integrated_loudness(combined)

        if not np.isinf(ref_lufs) and not np.isinf(combined_lufs):
            gain_db = ref_lufs - combined_lufs
            print(f"  LUFS matching: combined={combined_lufs:.1f}, reference={ref_lufs:.1f}, gain={gain_db:+.1f} dB")
        else:
            gain_db = 0.0
            print(f"  LUFS measurement failed (combined={combined_lufs}, ref={ref_lufs}), skipping normalization")

        # Apply gain + brickwall limiter in one pass via Pedalboard.
        # Unlike simple peak normalization (which scales the ENTIRE signal
        # down equally and kills loudness), a real limiter only catches
        # transient peaks while preserving the overall perceived loudness.
        board = Pedalboard([
            Gain(gain_db=gain_db),
            Limiter(threshold_db=-0.5, release_ms=100.0),
        ])
        # Pedalboard expects [channels, samples]
        combined_T = combined.T.copy()  # [channels, samples]
        combined_T = board(combined_T, target_sr)
        combined = combined_T.T  # back to [samples, channels]

        final_lufs = meter.integrated_loudness(combined)
        final_peak = np.max(np.abs(combined))
        print(f"  After LUFS+limiter: LUFS={final_lufs:.1f}, peak={final_peak:.4f}")

        # Save stem-only result
        sf.write(output_path, combined, target_sr, subtype="PCM_16")
        print(f"\n  Stem-mastered saved: {output_path}")

        # --- Step 5/5: Final full-mix matchering polish ---
        # The stem matchering gave us per-element tonal matching, but the
        # recombined mix is missing the inter-stem "glue" — how the bass
        # and kick interact, how vocals sit in the mix, overall spectral
        # balance.  A final light matchering pass corrects this.
        # Since the recombined mix is already close to the reference,
        # matchering will only apply small corrections (not overcook).
        print("\n--- Step 5/5: Final full-mix matchering polish ---")
        import matchering as mg

        stem_plus_full_path = output_path.replace("_stem_mastered", "_stem_plus_full")
        if stem_plus_full_path == output_path:
            stem_plus_full_path = output_path.replace(".wav", "_plus_full.wav")

        try:
            with tempfile.TemporaryDirectory(prefix="final_match_") as final_tmp:
                # matchering needs a wav input — we already saved one
                temp_out = os.path.join(final_tmp, "final_polished.wav")
                mg.process(
                    target=output_path,
                    reference=reference_path,
                    results=[mg.pcm16(temp_out)],
                )
                # Read back and measure
                polished, pol_sr = sf.read(temp_out, dtype="float32")
                if polished.ndim == 1:
                    polished = np.column_stack((polished, polished))
                # Resample if matchering changed sample rate
                if pol_sr != target_sr:
                    import librosa
                    polished = librosa.resample(polished.T, orig_sr=pol_sr, target_sr=target_sr).T

                pol_lufs = meter.integrated_loudness(polished)
                pol_peak = np.max(np.abs(polished))
                print(f"  Final polish: LUFS={pol_lufs:.1f}, peak={pol_peak:.4f}")

                sf.write(stem_plus_full_path, polished, target_sr, subtype="PCM_16")
                print(f"  Stem+full-mix saved: {stem_plus_full_path}")

        except Exception as e:
            print(f"  Final matchering polish FAILED: {e}")
            stem_plus_full_path = None

        # --- Optional: save individual stems for inspection ---
        if save_stems:
            if stems_output_dir is None:
                stems_output_dir = str(Path(output_path).parent / f"{Path(output_path).stem}_stems")
            os.makedirs(stems_output_dir, exist_ok=True)

            for stem_type in MATCHERING_GROUPS:
                # Save matched generation stem
                if stem_type in matched_stems:
                    src = matched_stems[stem_type]
                    dst = os.path.join(stems_output_dir, f"gen_{stem_type}_matched.wav")
                    data, sr = sf.read(src, dtype="float32")
                    sf.write(dst, data, sr)

                # Save original generation stem
                if stem_type in gen_stems:
                    src = gen_stems[stem_type]
                    dst = os.path.join(stems_output_dir, f"gen_{stem_type}_original.wav")
                    data, sr = sf.read(src, dtype="float32")
                    sf.write(dst, data, sr)

                # Save reference stem
                if stem_type in ref_stems:
                    src = ref_stems[stem_type]
                    dst = os.path.join(stems_output_dir, f"ref_{stem_type}.wav")
                    data, sr = sf.read(src, dtype="float32")
                    sf.write(dst, data, sr)

            print(f"  Stems saved to: {stems_output_dir}")

    total_time = time.time() - t_start
    print(f"\n  Total stem matchering time: {total_time:.1f}s")
    return True


# ---------------------------------------------------------------------------
# Full-mix matchering (for A/B comparison)
# ---------------------------------------------------------------------------

def run_full_mix_matchering(
    generation_path: str,
    reference_path: str,
    output_path: str,
) -> bool:
    """Standard full-mix matchering for comparison."""
    import matchering as mg

    print("\n--- Full-Mix Matchering (for A/B comparison) ---")
    t_start = time.time()

    try:
        mg.process(
            target=generation_path,
            reference=reference_path,
            results=[mg.pcm16(output_path)],
        )
        elapsed = time.time() - t_start
        print(f"  ✓ Full-mix matchering done in {elapsed:.1f}s")
        print(f"  Output: {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ Full-mix matchering FAILED: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stem-by-stem matchering prototype",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reference", "-r", required=True,
        help="Path to the reference/target audio (the sound you want to match)",
    )
    parser.add_argument(
        "--generation", "-g", required=True,
        help="Path to the generated audio (the sound to be mastered)",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Output directory (default: same as generation file)",
    )
    parser.add_argument(
        "--also-full-mix", action="store_true",
        help="Also run standard full-mix matchering for A/B comparison",
    )
    parser.add_argument(
        "--save-stems", action="store_true",
        help="Save individual stem pairs for inspection",
    )

    args = parser.parse_args()

    # Validate inputs
    ref_path = Path(args.reference)
    gen_path = Path(args.generation)

    if not ref_path.exists():
        print(f"ERROR: Reference file not found: {ref_path}")
        sys.exit(1)
    if not gen_path.exists():
        print(f"ERROR: Generation file not found: {gen_path}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = gen_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output filenames
    stem_output = out_dir / f"{gen_path.stem}_stem_mastered.wav"
    full_output = out_dir / f"{gen_path.stem}_full_mastered.wav"

    print("=" * 60)
    print("Stem-by-Stem Matchering Prototype")
    print("=" * 60)
    print(f"  Reference : {ref_path}")
    print(f"  Generation: {gen_path}")
    print(f"  Output dir: {out_dir}")
    print(f"  Save stems: {args.save_stems}")
    print(f"  Full-mix comparison: {args.also_full_mix}")

    # --- Run stem matchering ---
    success = run_stem_matchering(
        generation_path=str(gen_path),
        reference_path=str(ref_path),
        output_path=str(stem_output),
        save_stems=args.save_stems,
    )

    if not success:
        print("\nStem matchering failed!")
        sys.exit(1)

    # --- Optionally run full-mix matchering for comparison ---
    if args.also_full_mix:
        run_full_mix_matchering(
            generation_path=str(gen_path),
            reference_path=str(ref_path),
            output_path=str(full_output),
        )

    # --- Summary ---
    stem_plus_full = Path(str(stem_output).replace("_stem_mastered", "_stem_plus_full"))
    print("\n" + "=" * 60)
    print("Files for A/B comparison:")
    print(f"  Original generation : {gen_path}")
    print(f"  Stem-mastered       : {stem_output}")
    if stem_plus_full.exists():
        print(f"  Stem + full polish  : {stem_plus_full}")
    if args.also_full_mix and full_output.exists():
        print(f"  Full-mix mastered   : {full_output}")
    print("=" * 60)
    print("\nDone! Compare these files in your DAW or audio player.")


if __name__ == "__main__":
    main()
