import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, welch, find_peaks

# ============================
# CONFIG
# ============================

INPUT_DIR = "outputs/audio"
OUTPUT_DIR = "outputs/audio_notched"

PSD_NPERSEG = 4096
PEAK_DB_ABOVE_MED = 20
MAX_NOTCHES = 2

NOTCH_R = 0.93   # closer to 1 = narrower notch

FREQ_MIN = 20
FREQ_MAX = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# NOTCH DESIGN
# ============================

def design_notch(f0, fs, r=0.98):
    """
    H(z) = (1 - 2cos(w0)z^-1 + z^-2) / (1 - 2r cos(w0)z^-1 + r^2 z^-2)
    """
    w0 = 2 * np.pi * f0 / fs

    b = np.array([1.0, -2*np.cos(w0), 1.0], dtype=np.float64)
    a = np.array([1.0, -2*r*np.cos(w0), r**2], dtype=np.float64)

    zi = np.zeros(max(len(a), len(b)) - 1, dtype=np.float64)
    return b, a, zi


def detect_notch_freqs(signal, fs):
    f, Pxx = welch(signal, fs=fs, nperseg=PSD_NPERSEG)
    Pxx_db = 10*np.log10(Pxx + 1e-20)

    mask = (f >= FREQ_MIN) & (f <= FREQ_MAX)
    f2 = f[mask]
    P2 = Pxx_db[mask]

    #noise_floor = np.median(P2)
    noise_floor = np.median(Pxx_db)

    peaks, props = find_peaks(P2, height=noise_floor + PEAK_DB_ABOVE_MED)
    if len(peaks) == 0:
        return []

    peak_freqs = f2[peaks]
    peak_heights = props["peak_heights"]

    idx_sort = np.argsort(peak_heights)[::-1]
    peak_freqs = peak_freqs[idx_sort][:MAX_NOTCHES]

    return peak_freqs.tolist()


def apply_notches(signal, fs, freqs, r=NOTCH_R):
    y = signal.astype(np.float64)

    for f0 in freqs:
        b, a, zi = design_notch(f0, fs, r=r)
        y, _ = lfilter(b, a, y, zi=zi)

    return y


# ============================
# MAIN
# ============================

print(f"Scanning folder: {INPUT_DIR}")

wav_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".wav")]

if not wav_files:
    print("No WAV files found.")
    quit()

for filename in wav_files:
    path = os.path.join(INPUT_DIR, filename)

    fs, audio = wavfile.read(path)

    # Convert stereo to mono if needed
    if audio.ndim == 2:
        audio = audio[:, 0]

    # Convert to float in range [-1, 1]
    if audio.dtype == np.int16:
        x = audio.astype(np.float64) / 32768.0
    elif audio.dtype == np.int32:
        x = audio.astype(np.float64) / 2147483648.0
    else:
        x = audio.astype(np.float64)

    freqs = detect_notch_freqs(x, fs)

    print(f"\nFile: {filename}")
    if freqs:
        print("  Detected notches:")
        for f0 in freqs:
            print(f"    {f0:.1f} Hz")
    else:
        print("  No strong tonal peaks found.")
        freqs = []

    y = apply_notches(x, fs, freqs)

    # Clip to valid range
    y = np.clip(y, -1.0, 1.0)

    # Convert back to int16 WAV
    y_int16 = (y * 32767.0).astype(np.int16)

    out_path = os.path.join(OUTPUT_DIR, filename.replace(".wav", "_notched.wav"))
    wavfile.write(out_path, fs, y_int16)

    print(f"  Saved -> {out_path}")

print("\nDone.")