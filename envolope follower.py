import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 192000          # sample rate in Hz
duration = 0.02      # 20 ms snippet to keep the plot manageable
t = np.arange(0, duration, 1/fs)
N = len(t)

bits = 18
max_val = 2**(bits-1) - 1  # 18-bit signed max

# Test signal: 1 kHz sine with amplitude ramp up then down
freq = 1000  # Hz
amplitude_env = np.concatenate([
    np.linspace(0.1, 1.0, N//2),
    np.linspace(1.0, 0.2, N - N//2)
])
x_float = amplitude_env * np.sin(2 * np.pi * freq * t)

# Quantize to 18-bit integer
x_int = np.clip(np.round(x_float * max_val), -max_val, max_val).astype(np.int32)

# Basic envelope follower (fast attack, exponential-ish decay)
def envelope_follower(x, attack_coeff=1.0, decay_shift=10):
    """
    x: integer signal
    attack_coeff: if 1.0 => instant attack (env jumps to |x| when larger)
    decay_shift: env = env - (env >> decay_shift) when |x| < env
                 (larger decay_shift => slower decay)
    """
    x = np.asarray(x, dtype=np.int64)
    env = np.zeros_like(x)
    for n in range(len(x)):
        a = abs(x[n])
        if n == 0:
            env[n] = a
        else:
            if a > env[n-1]:
                # fast attack (can blend with env[n-1] if attack_coeff < 1)
                env[n] = int(attack_coeff * a + (1 - attack_coeff) * env[n-1])
            else:
                # exponential-ish decay using shift
                env[n] = env[n-1] - (env[n-1] >> decay_shift)
    return env

env_int = envelope_follower(x_int, attack_coeff=1.0, decay_shift=10)

# Convert to float [-1, 1] for plotting
x_plot = x_int / max_val
env_plot = env_int / max_val

# Plot a short segment
plt.figure(figsize=(8, 4))
samples_to_show = min(2000, N)  # about 10.4 ms at 192 kHz
time_show = t[:samples_to_show] * 1000  # ms

plt.plot(time_show, x_plot[:samples_to_show], label="Signal (18-bit quantized)")
plt.plot(time_show, env_plot[:samples_to_show], label="Envelope", linewidth=2)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (normalized)")
plt.title("Basic Envelope Follower at 192 kHz, 18-bit")
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 192000          # sample rate in Hz
duration = 0.02      # 20 ms
t = np.arange(0, duration, 1/fs)
N = len(t)

bits = 18
max_val = 2**(bits-1) - 1  # 18-bit signed max

# Test signal: 1 kHz sine with amplitude ramp DOWN
freq = 1000  # Hz
amplitude_env = np.linspace(1.0, 0.1, N)
x_float = amplitude_env * np.sin(2 * np.pi * freq * t)

# Quantize to 18-bit integer
x_int = np.clip(np.round(x_float * max_val), -max_val, max_val).astype(np.int32)

# Basic envelope follower (fast attack, exponential-ish decay)
def envelope_follower(x, attack_coeff=1.0, decay_shift=10):
    x = np.asarray(x, dtype=np.int64)
    env = np.zeros_like(x)
    for n in range(len(x)):
        a = abs(x[n])
        if n == 0:
            env[n] = a
        else:
            if a > env[n-1]:
                # fast attack
                env[n] = int(attack_coeff * a + (1 - attack_coeff) * env[n-1])
            else:
                # exponential-ish decay
                env[n] = env[n-1] - (env[n-1] >> decay_shift)
    return env

env_int = envelope_follower(x_int, attack_coeff=1.0, decay_shift=10)

# Convert to float [-1, 1] for plotting
x_plot = x_int / max_val
env_plot = env_int / max_val

# Plot a short segment
plt.figure(figsize=(8, 4))
samples_to_show = min(2000, N)  # ~10.4 ms at 192 kHz
time_show = t[:samples_to_show] * 1000  # ms

plt.plot(time_show, x_plot[:samples_to_show], label="Signal (18-bit quantized)")
plt.plot(time_show, env_plot[:samples_to_show], label="Envelope", linewidth=2)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (normalized)")
plt.title("Envelope Follower at 192 kHz, 18-bit (Decreasing Gain)")
plt.legend()
plt.tight_layout()
plt.show()
