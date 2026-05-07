import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import os

fs = 44100
duration = 5

os.makedirs("outputs/audio", exist_ok=True)

print("Recording...")
recording = sd.rec(int(duration * fs),
                   samplerate=fs,
                   channels=1,
                   dtype='float32')

sd.wait()

audio = recording.flatten()

audio = audio - np.mean(audio)

wavfile.write(
    "outputs/audio/my_recording.wav",
    fs,
    (audio * 32767).astype(np.int16)
)

print("Saved recording.")