import os
import queue
import time
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import lfilter, butter

# ============================
# CONFIG
# ============================

FS = 44100
CHANNELS = 1
BLOCKSIZE = 1024
BUFFER_SECONDS = 10

ENVELOPE_MS = 10

# Detection sensitivity
THRESH_STD = 6.0

# Hysteresis (prevents repeated triggers on steady noise)
HYST_RATIO = 0.6

# Minimum time between saved events
REFRACTORY_SEC = 0.75

# Calibration time (learn baseline noise before detection starts)
CALIBRATION_SEC = 3.0

# Optional slope gating (reject stationary noise)
USE_SLOPE_GATE = True
SLOPE_STD = 4.0

CLIP_PRE_SEC = 0.5
CLIP_POST_SEC = 1.5

# Noise model adaptation speed
NOISE_TAU_SEC = 2.0

# Highpass filter
HP_CUTOFF = 150
HP_ORDER = 4

os.makedirs("outputs/audio/events", exist_ok=True)

# ============================
# RING BUFFER
# ============================

BUFFER_LEN = int(FS * BUFFER_SECONDS)
ring = np.zeros(BUFFER_LEN, dtype=np.float32)
write_idx = 0

# ============================
# ENVELOPE FILTER (MOVING AVERAGE FIR)
# ============================

M = int(FS * ENVELOPE_MS / 1000)
M = max(M, 1)

h = np.ones(M, dtype=np.float32) / M
zi_env = np.zeros(M - 1, dtype=np.float32)

# ============================
# HIGHPASS FILTER
# ============================

b_hp, a_hp = butter(HP_ORDER, HP_CUTOFF, btype="highpass", fs=FS)
zi_hp = np.zeros(max(len(a_hp), len(b_hp)) - 1, dtype=np.float32)

# ============================
# NOISE MODEL
# ============================

dt_block = BLOCKSIZE / FS
alpha = dt_block / NOISE_TAU_SEC
alpha = min(max(alpha, 0.001), 0.05)

noise_mu = 0.0
noise_sigma = 1e-9

# ============================
# DETECTOR STATE
# ============================

last_event_time = 0.0
event_counter = 0
block_counter = 0

armed = False
in_event = False

audio_queue = queue.Queue(maxsize=50)

# ============================
# CALLBACK
# ============================

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)

    block = indata[:, 0].copy()

    try:
        audio_queue.put_nowait(block)
    except queue.Full:
        pass


print("Starting real-time detector. Press Ctrl+C to stop.")
start_time = time.time()

try:
    with sd.InputStream(
        samplerate=FS,
        channels=CHANNELS,
        blocksize=BLOCKSIZE,
        callback=audio_callback
    ):
        while True:
            block = audio_queue.get()
            block_counter += 1

            t0 = time.perf_counter()
            now = time.time()

            # ----------------------------
            # Ring buffer write
            # ----------------------------
            n = len(block)
            end_idx = write_idx + n

            if end_idx < BUFFER_LEN:
                ring[write_idx:end_idx] = block
            else:
                part1 = BUFFER_LEN - write_idx
                ring[write_idx:] = block[:part1]
                ring[:end_idx % BUFFER_LEN] = block[part1:]

            write_idx = (write_idx + n) % BUFFER_LEN

            # ----------------------------
            # Highpass filter
            # ----------------------------
            block_hp, zi_hp[:] = lfilter(b_hp, a_hp, block, zi=zi_hp)

            # ----------------------------
            # Envelope energy (squared + moving average)
            # ----------------------------
            rectified = block_hp * block_hp
            envelope_block, zi_env[:] = lfilter(h, [1.0], rectified, zi=zi_env)

            peak = float(np.max(envelope_block))

            # Robust block statistics
            med = float(np.median(envelope_block))
            mad = float(np.median(np.abs(envelope_block - med)))
            sigma_est = 1.4826 * mad

            # ----------------------------
            # Calibration phase
            # ----------------------------
            if not armed:
                noise_mu = (1 - alpha) * noise_mu + alpha * med
                noise_sigma = (1 - alpha) * noise_sigma + alpha * sigma_est

                if (now - start_time) >= CALIBRATION_SEC:
                    armed = True
                    print(f"Detector armed after {CALIBRATION_SEC}s calibration.")
                    print(f"Initial baseline mu={noise_mu:.3e}, sigma={noise_sigma:.3e}")

                continue

            # ----------------------------
            # Compute thresholds
            # ----------------------------
            threshold_high = noise_mu + THRESH_STD * noise_sigma
            threshold_low = noise_mu + HYST_RATIO * THRESH_STD * noise_sigma

            # ----------------------------
            # Optional slope gating
            # Detect rapid rises (events) vs steady noise
            # ----------------------------
            slope_peak = float(np.max(np.diff(envelope_block)))
            slope_threshold = SLOPE_STD * noise_sigma

            slope_ok = True
            if USE_SLOPE_GATE:
                slope_ok = slope_peak > slope_threshold

            # ----------------------------
            # State machine detection
            # ----------------------------
            if not in_event:
                if peak > threshold_high and slope_ok and (now - last_event_time) > REFRACTORY_SEC:
                    in_event = True
                    last_event_time = now
                    event_counter += 1

                    print(f"[EVENT {event_counter}] peak={peak:.3e} thr={threshold_high:.3e} slope={slope_peak:.3e}")

                    # Save clip
                    pre = int(FS * CLIP_PRE_SEC)
                    post = int(FS * CLIP_POST_SEC)
                    clip_len = pre + post

                    end_pos = write_idx
                    start_pos = (end_pos - clip_len) % BUFFER_LEN

                    if start_pos < end_pos:
                        clip = ring[start_pos:end_pos].copy()
                    else:
                        clip = np.concatenate((ring[start_pos:], ring[:end_pos])).copy()

                    clip = clip - np.mean(clip)
                    clip = clip / (np.max(np.abs(clip)) + 1e-12)
                    clip_int16 = (clip * 32767).astype(np.int16)

                    filename = f"outputs/audio/events/event_{event_counter:04d}.wav"
                    wavfile.write(filename, FS, clip_int16)
                    print(f"Saved {filename}")

                else:
                    # Update baseline only when not triggered
                    noise_mu = (1 - alpha) * noise_mu + alpha * med
                    noise_sigma = (1 - alpha) * noise_sigma + alpha * sigma_est

            else:
                # We are in an event, wait until envelope drops below low threshold
                if peak < threshold_low:
                    in_event = False

            # ----------------------------
            # Performance metrics
            # ----------------------------
            t1 = time.perf_counter()
            dt_ms = (t1 - t0) * 1000
            budget_ms = (BLOCKSIZE / FS) * 1000
            util = dt_ms / budget_ms

            if block_counter % 50 == 0:
                print(
                    f"proc={dt_ms:.3f}ms util={util:.3f} "
                    f"mu={noise_mu:.3e} sig={noise_sigma:.3e} "
                    f"thrH={threshold_high:.3e} thrL={threshold_low:.3e}"
                )

except KeyboardInterrupt:
    print("\nStopped.")