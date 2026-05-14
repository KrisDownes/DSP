import os
import queue
import time
import numpy as np
import sounddevice as sd
from scipy.signal import lfilter, butter, welch, find_peaks

# ============================
# CONFIG
# ============================

FS = 44100
CHANNELS = 1
BLOCKSIZE = 1024

HP_CUTOFF = 80
HP_ORDER = 2

PSD_UPDATE_SEC = 2.0
PSD_NPERSEG = 4096

PEAK_DB_ABOVE_MED = 20
MAX_NOTCHES = 6

NOTCH_R = 0.98   # controls notch width (closer to 1 = narrower)

FREQ_MIN = 40
FREQ_MAX = 4000

os.makedirs("outputs/audio", exist_ok=True)

audio_queue = queue.Queue(maxsize=50)

# ============================
# HIGH PASS FILTER
# ============================

b_hp, a_hp = butter(HP_ORDER, HP_CUTOFF, btype="highpass", fs=FS)
zi_hp = np.zeros(max(len(a_hp), len(b_hp)) - 1, dtype=np.float32)

# ============================
# NOTCH FILTER STATE
# ============================

notch_filters = []   # list of (b, a, zi)

def design_notch(f0, fs, r=0.98):
    """
    H(z) = (1 - 2cos(w0)z^-1 + z^-2) / (1 - 2r cos(w0)z^-1 + r^2 z^-2)
    """
    w0 = 2*np.pi*f0/fs

    b = np.array([1.0, -2*np.cos(w0), 1.0], dtype=np.float32)
    a = np.array([1.0, -2*r*np.cos(w0), r**2], dtype=np.float32)

    zi = np.zeros(max(len(a), len(b)) - 1, dtype=np.float32)
    return b, a, zi

def update_notches(signal_block):
    global notch_filters

    f, Pxx = welch(signal_block, fs=FS, nperseg=PSD_NPERSEG)
    Pxx_db = 10*np.log10(Pxx + 1e-20)

    # restrict frequency range
    mask = (f >= FREQ_MIN) & (f <= FREQ_MAX)
    f2 = f[mask]
    P2 = Pxx_db[mask]

    noise_floor = np.median(P2)

    peaks, props = find_peaks(P2, height=noise_floor + PEAK_DB_ABOVE_MED)

    if len(peaks) == 0:
        return

    peak_freqs = f2[peaks]
    peak_heights = props["peak_heights"]

    # Sort by strongest peak
    idx_sort = np.argsort(peak_heights)[::-1]
    peak_freqs = peak_freqs[idx_sort]

    # Keep only top N peaks
    peak_freqs = peak_freqs[:MAX_NOTCHES]

    notch_filters = []
    for f0 in peak_freqs:
        b, a, zi = design_notch(f0, FS, r=NOTCH_R)
        notch_filters.append([b, a, zi])

    print("\n[Notch update] Designed notches at:")
    for f0 in peak_freqs:
        print(f"  {f0:.1f} Hz")

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    block = indata[:, 0].copy()
    try:
        audio_queue.put_nowait(block)
    except queue.Full:
        pass

print("Starting notch suppressor. Ctrl+C to stop.")

try:
    with sd.InputStream(
        samplerate=FS,
        channels=CHANNELS,
        blocksize=BLOCKSIZE,
        callback=audio_callback
    ):
        psd_buffer = []
        last_update = time.time()

        while True:
            block = audio_queue.get()

            # highpass
            block_hp, zi_hp = lfilter(b_hp, a_hp, block, zi=zi_hp)

            # update PSD buffer
            psd_buffer.append(block_hp)

            # update notch filters periodically
            now = time.time()
            if (now - last_update) > PSD_UPDATE_SEC:
                chunk = np.concatenate(psd_buffer)
                psd_buffer = []
                update_notches(chunk)
                last_update = now

            # apply notch cascade
            y = block_hp
            for nf in notch_filters:
                b, a, zi = nf
                y, zi = lfilter(b, a, y, zi=zi)
                nf[2] = zi

            # (optional) output audio to speakers:
            # sd.play(y, FS)

except KeyboardInterrupt:
    print("\nStopped.")