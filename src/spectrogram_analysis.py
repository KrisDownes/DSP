import os
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

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

# ----------------------------
# Windowed FFT (whole signal)
# ----------------------------
N = len(audio)
window = np.hanning(N)
audio_win = audio_plot * window

X = np.fft.rfft(audio_win)
freqs = np.fft.rfftfreq(N, 1/fs)

mag = np.abs(X) / (np.sum(window) / 2)
mag_db = 20 * np.log10(mag + 1e-12)

# ----------------------------
# Welch PSD
# ----------------------------
f_welch, Pxx = signal.welch(audio, fs=fs, nperseg=4096)
Pxx_db = 10 * np.log10(Pxx + 1e-20)

# ----------------------------
# Spectrogram / STFT
# ----------------------------
f_stft, t_stft, Zxx = signal.stft(audio, fs=fs, nperseg=2048, noverlap=1536, window="hann")
Sxx = np.abs(Zxx)**2
Sxx_db = 10 * np.log10(Sxx + 1e-20)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(14, 12))

plt.subplot(4, 1, 1)
plt.plot(t, audio_plot)
plt.title("Time Domain (normalized)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(freqs, mag_db)
plt.title("Windowed FFT Magnitude Spectrum (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs/2)
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(f_welch, Pxx_db)
plt.title("Welch PSD (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.xlim(0, 5000)
plt.grid(True)

plt.subplot(4, 1, 4)
plt.pcolormesh(t_stft, f_stft, Sxx_db, shading="gouraud")
plt.title("Spectrogram (STFT)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.ylim(0, 5000)
plt.colorbar(label="Power (dB)")

plt.tight_layout()
plt.savefig("outputs/plots/audio_analysis.png", dpi=200)
print("Saved plot to outputs/plots/audio_analysis.png")