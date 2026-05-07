import os
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.signal import find_peaks


# ----------------------------
# Output directories
# ----------------------------
os.makedirs("outputs/plots", exist_ok=True)


# Read Audio
fs, audio = wavfile.read("outputs/audio/my_recording.wav")
audio = audio.astype(np.float32) / 32767

# ----------------------------
# Normalize only for plotting
# ----------------------------
audio_plot = audio / (np.max(np.abs(audio)) + 1e-12)


# ----------------------------
# Time axis
# ----------------------------
t = np.arange(len(audio)) / fs


# ============================================================
# Clap Detection (Time-domain transient detection)
# ============================================================

# Step 1: Rectify (absolute value)
# This makes the signal always positive so averaging doesn't cancel it out.
rectified = np.abs(audio)

# Step 2: Smooth using moving average (~10 ms)
# This creates an amplitude envelope.
win_ms = 10
win_len = int(fs * win_ms / 1000)
win = np.ones(win_len) / win_len
envelope = np.convolve(rectified, win, mode="same")

# Step 3: Estimate noise floor robustly
# Median is better than mean because it ignores rare big spikes (claps).
noise_floor = np.median(envelope)

# Step 4: Threshold selection
# Multiplier controls sensitivity.
threshold = noise_floor * 8

# Step 5: Find peaks (claps)
# distance prevents multiple detections from the same clap tail.
min_distance = int(0.30 * fs)
peaks, props = find_peaks(envelope, height=threshold, distance=min_distance)

print(f"\nDetected {len(peaks)} clap(s):")
for i, p in enumerate(peaks):
    print(f"  Clap {i+1}: time = {p/fs:.3f} sec, envelope = {props['peak_heights'][i]:.6f}")


# ============================================================
# FFT (whole signal)
# ============================================================

N = len(audio)
window = np.hanning(N)
audio_win = audio_plot * window

X = np.fft.rfft(audio_win)
freqs = np.fft.rfftfreq(N, 1/fs)

mag = np.abs(X) / (np.sum(window) / 2)
mag_db = 20 * np.log10(mag + 1e-12)


# ============================================================
# Welch PSD
# ============================================================

f_welch, Pxx = signal.welch(audio, fs=fs, nperseg=4096)
Pxx_db = 10 * np.log10(Pxx + 1e-20)


# ============================================================
# Spectrogram / STFT
# ============================================================

f_stft, t_stft, Zxx = signal.stft(audio, fs=fs, nperseg=2048, noverlap=1536, window="hann")
Sxx = np.abs(Zxx)**2
Sxx_db = 10 * np.log10(Sxx + 1e-20)


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(14, 14))

# --- Time domain ---
plt.subplot(5, 1, 1)
plt.plot(t, audio_plot)
plt.title("Time Domain (normalized)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# --- Envelope + clap detection ---
plt.subplot(5, 1, 2)
plt.plot(t, envelope, label="Envelope (smoothed |audio|)")
plt.plot(peaks/fs, envelope[peaks], "rx", label="Detected claps")
plt.axhline(threshold, color="r", linestyle="--", label="Threshold")
plt.title("Clap Detection Envelope")
plt.xlabel("Time (s)")
plt.ylabel("Envelope amplitude")
plt.grid(True)
plt.legend()

# --- FFT ---
plt.subplot(5, 1, 3)
plt.plot(freqs, mag_db)
plt.title("Windowed FFT Magnitude Spectrum (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs/2)
plt.grid(True)

# --- Welch PSD ---
plt.subplot(5, 1, 4)
plt.plot(f_welch, Pxx_db)
plt.title("Welch PSD (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.xlim(0, 5000)
plt.grid(True)

# --- Spectrogram ---
plt.subplot(5, 1, 5)
plt.pcolormesh(t_stft, f_stft, Sxx_db, shading="gouraud")
plt.title("Spectrogram (STFT)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.ylim(0, 5000)
plt.colorbar(label="Power (dB)")

plt.tight_layout()
plt.savefig("outputs/plots/audio_analysis.png", dpi=200)
print("\nSaved plot to outputs/plots/audio_analysis.png")