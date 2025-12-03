import numpy as np
import matplotlib.pyplot as plt

# ============================
# 0) Global settings
# ============================
fs = 192000.0        # sample rate in Hz
num_points = 32768   # FFT grid so narrow bands are resolved
adc_bits = 18        # ADC resolution (real analog path)
design_bits = 20     # target ENOB for plotted design noise floor (spec line)
coeff_bits = 24      # word length for coefficient quantization to .coe
num_taps = 2048      # taps per band (IP max)
eps = 1e-12          # small value for log/guard

# ------------------------------------------------------------------
# Band layout
#  - Bands 0..17: log-ish coverage from ~40 Hz to 20 kHz.
#  - Band18: low “voice” helper band (80–300 Hz).
#  - Band19: mid band inserted between 11/12 so it’s in order.
# ------------------------------------------------------------------
bands = [
    # Coverage bands (0..10)
    dict(name="Band0",  fpl=40,    fpu=48,    fsl=36,    fsu=52),
    dict(name="Band1",  fpl=48,    fpu=69,    fsl=42,    fsu=75),
    dict(name="Band2",  fpl=69,    fpu=100,   fsl=61,    fsu=108),
    dict(name="Band3",  fpl=100,   fpu=144,   fsl=88,    fsu=156),
    dict(name="Band4",  fpl=144,   fpu=207,   fsl=127,   fsu=225),
    dict(name="Band5",  fpl=207,   fpu=299,   fsl=182,   fsu=324),
    dict(name="Band6",  fpl=299,   fpu=431,   fsl=263,   fsu=466),
    dict(name="Band7",  fpl=431,   fpu=621,   fsl=379,   fsu=672),
    dict(name="Band8",  fpl=621,   fpu=894,   fsl=546,   fsu=969),
    dict(name="Band9",  fpl=894,   fpu=1289,  fsl=787,   fsu=1397),
    dict(name="Band10", fpl=1289,  fpu=1858,  fsl=1134,  fsu=2013),

    # Split mid region into three contiguous bands:
    dict(name="Band11", fpl=1858,  fpu=2400,  fsl=1635,  fsu=2600),
    dict(name="Band19", fpl=2400,  fpu=3100,  fsl=2200,  fsu=3300),
    dict(name="Band12", fpl=3100,  fpu=3860,  fsl=2800,  fsu=4182),

    # High coverage bands
    dict(name="Band13", fpl=3860,  fpu=5564,  fsl=3397,  fsu=6027),
    dict(name="Band14", fpl=5564,  fpu=8019,  fsl=4896,  fsu=8687),
    dict(name="Band15", fpl=8019,  fpu=11558, fsl=7056,  fsu=12521),
    dict(name="Band16", fpl=11558, fpu=16659, fsl=10170, fsu=18047),
    dict(name="Band17", fpl=16659, fpu=20000, fsl=14659, fsu=20000),

    # Low voice helper band (still treated like a normal band)
    dict(name="Band18", fpl=80,    fpu=300,   fsl=60,    fsu=360),
]

# ============================
# 1) Kaiser + Bessel helpers
# ============================

def bessel_i0(x, terms=50):
    x = np.array(x, dtype=np.float64)
    result = np.ones_like(x)
    term = np.ones_like(x)
    fact = 1.0
    for k in range(1, terms):
        fact *= k
        term *= (x * x) / 4.0
        result += term / (fact * fact)
    return result

def kaiser_beta_for_enob(bits):
    A = 6.02 * bits + 1.76
    if A <= 21:
        beta = 0.0
    elif A <= 50:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21)
    else:
        beta = 0.1102 * (A - 8.7)
    return A, beta

def kaiser_window(N, beta):
    n = np.arange(N, dtype=np.float64)
    alpha = (N - 1) / 2.0
    r = (n - alpha) / alpha
    t = np.sqrt(np.clip(1.0 - r**2, 0.0, 1.0))
    return bessel_i0(beta * t) / bessel_i0(beta)

# Stronger design than 20 bits so stopbands are below 20-bit noise floor
design_bits_window = design_bits + 2
A_target_db, kaiser_beta = kaiser_beta_for_enob(design_bits_window)
print(f"Kaiser design target from {design_bits_window}-bit window:")
print(f"  Target attenuation A_target ≈ {A_target_db:.2f} dB")
print(f"  Kaiser beta                 ≈ {kaiser_beta:.2f}")

extra_bits_band0 = 1
bits_band0 = design_bits_window + extra_bits_band0
A0_db, kaiser_beta_band0 = kaiser_beta_for_enob(bits_band0)
print(f"\nBand0 uses stronger Kaiser:")
print(f"  Effective ENOB target (window): {bits_band0} bits")
print(f"  A0 target                      : {A0_db:.2f} dB")
print(f"  Kaiser beta (Band0)            : {kaiser_beta_band0:.2f}")

# ============================
# 2) Ideal bandpass design
# ============================

def ideal_lowpass(num_taps, fc, fs):
    n = np.arange(num_taps, dtype=np.float64)
    M = (num_taps - 1) / 2.0
    t = n - M
    return 2.0 * (fc / fs) * np.sinc(2.0 * fc * t / fs)

def ideal_bandpass(num_taps, fpl, fpu, fs):
    return ideal_lowpass(num_taps, fpu, fs) - ideal_lowpass(num_taps, fpl, fs)

def compute_response(h, fs, num_points):
    H = np.fft.rfft(h, n=num_points)
    freqs = np.linspace(0.0, fs / 2.0, len(H))
    return freqs, H

def design_bandpass_fir(num_taps, fs, fpl, fpu, beta, norm_to_center=True):
    h_ideal = ideal_bandpass(num_taps, fpl, fpu, fs)
    w = kaiser_window(num_taps, beta)
    h = h_ideal * w
    if norm_to_center:
        f_center = 0.5 * (fpl + fpu)
        freqs_tmp, H_tmp = compute_response(h, fs, num_points=8192)
        idx = np.argmin(np.abs(freqs_tmp - f_center))
        gain_center = np.abs(H_tmp[idx])
        if gain_center > 0:
            h = h / gain_center
    return h

# ============================
# 3) Spec extraction & ENOB
# ============================

def extract_band_specs(freqs, H, fpl, fpu, fsl, fsu):
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H), eps))

    pass_mask = (freqs >= fpl) & (freqs <= fpu)
    if not np.any(pass_mask):
        center = 0.5 * (fpl + fpu)
        idx_center = np.argmin(np.abs(freqs - center))
        pass_mask = np.zeros_like(freqs, dtype=bool)
        pass_mask[idx_center] = True

    stop_mask = (freqs >= fsu)
    if not np.any(stop_mask):
        stop_mask = ~pass_mask

    pb_mag = mag_db[pass_mask]
    sb_mag = mag_db[stop_mask]

    pb_max = np.max(pb_mag)
    pb_min = np.min(pb_mag)
    ripple = pb_max - pb_min
    worst_stop = np.max(sb_mag)

    target = pb_max - 3.0
    center = 0.5 * (fpl + fpu)

    idx_low_candidates = np.where((freqs < center) & (mag_db <= target))[0]
    idx_high_candidates = np.where((freqs >= center) & (mag_db <= target))[0]

    f_3db_low = freqs[idx_low_candidates[-1]] if idx_low_candidates.size > 0 else np.nan
    f_3db_high = freqs[idx_high_candidates[0]] if idx_high_candidates.size > 0 else np.nan

    return {
        "pb_max_db": float(pb_max),
        "pb_min_db": float(pb_min),
        "ripple_db": float(ripple),
        "worst_stop_db": float(worst_stop),
        "f_3db_low_Hz": float(f_3db_low),
        "f_3db_high_Hz": float(f_3db_high),
    }

def enob_from_noise_floor(noise_floor_db):
    snr_db = -noise_floor_db
    return (snr_db - 1.76) / 6.02

snr_adc_db = 6.02 * adc_bits + 1.76
noise_floor_adc_db = -snr_adc_db
snr_design_db = 6.02 * design_bits + 1.76
noise_floor_design_db = -snr_design_db

# ============================
# 4) Design all bands
# ============================

coeffs_list = []
H_all = []
specs_per_band = []

for i, b in enumerate(bands):
    beta_use = kaiser_beta_band0 if i == 0 else kaiser_beta
    h = design_bandpass_fir(num_taps, fs, b["fpl"], b["fpu"], beta_use)
    coeffs_list.append(h)

freqs, _dummy = compute_response(coeffs_list[0], fs, num_points)
for h, b in zip(coeffs_list, bands):
    _, H = compute_response(h, fs, num_points)
    H_all.append(H)
    specs = extract_band_specs(freqs, H, b["fpl"], b["fpu"], b["fsl"], b["fsu"])
    specs_per_band.append((b["name"], specs))

H_all = np.array(H_all)

print("\n=== Auto Extracted Specs per Band (math Kaiser) ===")
enob_list = []
for (name, s), b in zip(specs_per_band, bands):
    worst_stop_db = s['worst_stop_db']
    stop_margin_adc_db = noise_floor_adc_db - worst_stop_db
    stop_margin_20_db = noise_floor_design_db - worst_stop_db
    enob_total = enob_from_noise_floor(worst_stop_db)
    enob_margin_vs_adc = enob_total - adc_bits
    enob_margin_vs_design = enob_total - design_bits
    enob_list.append((name, enob_total))

    print(f"\n{name}:")
    print(f"  Passband ripple                     : {s['ripple_db']:.3f} dB")
    print(f"  Passband max (gain)                 : {s['pb_max_db']:.3f} dB")
    print(f"  Passband min (gain)                 : {s['pb_min_db']:.3f} dB")
    print(f"  Worst stopband level (noise floor)  : {worst_stop_db:.3f} dBFS")
    print(f"  f_3dB low edge                      : {s['f_3db_low_Hz']:.1f} Hz")
    print(f"  f_3dB high edge                     : {s['f_3db_high_Hz']:.1f} Hz")
    print(f"  Margin vs {adc_bits}-bit noise floor   : {stop_margin_adc_db:.3f} dB")
    print(f"  Margin vs {design_bits}-bit noise floor: {stop_margin_20_db:.3f} dB")
    print(f"  ENOB from stopband noise floor (total) : {enob_total:.2f} bits")
    print(f"    ENOB margin vs {adc_bits}-bit ADC    : {enob_margin_vs_adc:.2f} bits")
    print(f"    ENOB margin vs {design_bits}-bit spec: {enob_margin_vs_design:.2f} bits")

limiting_band_name, limiting_enob = min(enob_list, key=lambda x: x[1])
print("\n=== Overall Filter Bank Limits ===")
print(f"  Limiting band (by stopband leakage): {limiting_band_name}")
print(f"  ENOB limited by worst stopband     : {limiting_enob:.2f} bits")

# ============================
# 6) Per-band taper
# ============================

coeffs_list = np.array(coeffs_list)
band_centers = [0.5 * (b["fpl"] + b["fpu"]) for b in bands]

# Taper ALL bands now (Band19 no longer special)
num_bands_to_taper = len(bands)
target_mag = 1.0
gains = np.ones(len(bands))

for k in range(num_bands_to_taper):
    f_center = band_centers[k]
    idx = np.argmin(np.abs(freqs - f_center))
    Hk_mag = np.abs(H_all[k, idx])
    existing_mag = np.abs(np.dot(gains[:k], H_all[:k, idx])) if k > 0 else 0.0
    if Hk_mag < 1e-9:
        gains[k] = 0.0
    else:
        gk = max(0.0, (target_mag - existing_mag) / Hk_mag)
        gains[k] = min(gk, 1.0)   # only cut

# Extra manual tweak: gently reduce lowest bands to pull left side toward 0 dB
low_adjust = [0.90, 0.93, 0.96, 0.98]  # Band0..Band3
for idx, fac in enumerate(low_adjust):
    gains[idx] *= fac

print("\nPer-band taper gains (linear):")
for i, g in enumerate(gains):
    print(f"  {bands[i]['name']}: {g:.6f}")

coeffs_tapered = coeffs_list * gains[:, None]

# ============================
# 7) Global gain so sum ~0 dB up to 20 kHz
# ============================

H_sum_tapered = np.sum(
    [compute_response(h, fs, num_points)[1] for h in coeffs_tapered],
    axis=0
)
mag_sum_tapered = np.abs(H_sum_tapered)

f_roi_min = bands[0]["fpl"]
f_roi_max = 20000.0
roi_mask = (freqs >= f_roi_min) & (freqs <= f_roi_max)

avg_mag = mag_sum_tapered[roi_mask].mean()
g_global2 = 1.0 / avg_mag
g_global2_db = 20.0 * np.log10(g_global2)
print(f"\nGlobal gain g_global2 = {g_global2:.6f} ({g_global2_db:.3f} dB)")

coeffs_final = coeffs_tapered * g_global2
design_eq_db = np.array([
    -5.2,  # Band0  (example: pull low end down 1.5 dB)
    -1.0,  # Band1
    -0.5,  # Band2
     -5,  # Band3
     -3,  # Band4
     -3,  # Band5
     -2.5,  # Band6
     -1.5,  # Band7
     -.8,  # Band8
     -.3,  # Band9
     0.0,  # Band10
     0.0,  # Band11
     0.0,  # Band19
     0.0,  # Band12
     0.0,  # Band13
     0.0,  # Band14
     0.0,  # Band15
     0.0,  # Band16
     0.0,  # Band17
     0.0,  # Band18
])

# Convert dB -> linear
design_eq_lin = 10.0**(design_eq_db / 20.0)

# Apply to the final coefficients: this changes BOTH the simulation
# responses and the .coe files you export.
coeffs_final = coeffs_final * design_eq_lin[:, None]

# ============================
# 8) Responses & "undo" gains
# ============================

H_final = []
mag_all_db = []
for h in coeffs_final:
    _, H = compute_response(h, fs, num_points)
    H_final.append(H)
    mag_all_db.append(20.0 * np.log10(np.maximum(np.abs(H), eps)))
H_final = np.array(H_final)
mag_all_db = np.array(mag_all_db)

H_sum_final = np.sum(H_final, axis=0)
mag_sum_final = np.abs(H_sum_final)
mag_sum_final_db = 20.0 * np.log10(np.maximum(mag_sum_final, eps))

scale_total = gains * g_global2
atten_db = 20.0 * np.log10(np.maximum(scale_total, eps))
undo_db = -atten_db

band_peaks_db = []
for k, b in enumerate(bands):
    mask = (freqs >= b["fpl"]) & (freqs <= b["fpu"])
    mag_band = np.abs(H_final[k, mask])
    band_peaks_db.append(20.0 * np.log10(np.maximum(mag_band.max(), eps)))

print("\n=== Band heights and EQ gains to get back to original 0 dB ===")
print("Band  centerHz  per-band-gain  net-atten-dB  gain-to-undo-dB  peak-dB")
for i, (c, g, att, ud, pk) in enumerate(
    zip(band_centers, gains, atten_db, undo_db, band_peaks_db)
):
    print(f"{i:2d}  {c:8.1f}    {g:8.5f}   {att:10.3f}   {ud:10.3f}   {pk:8.3f}")

# ============================
# 9) Noise lines
# ============================

noise_line_adc = np.full_like(freqs, noise_floor_adc_db)
noise_line_design = np.full_like(freqs, noise_floor_design_db)

# ============================
# 10) Quantization & .coe
# ============================

def quantize_coeffs(coeffs, bits):
    max_pos = 2**(bits - 1) - 1
    min_neg = -2**(bits - 1)
    scale = max_pos
    q = np.round(coeffs * scale)
    q = np.clip(q, min_neg, max_pos)
    return q.astype(int)

def write_coe(filename, coeffs_int, radix=10):
    with open(filename, "w") as f:
        f.write(f"radix = {radix};\n")
        f.write("coefdata = \n")
        for i, c in enumerate(coeffs_int):
            if i == len(coeffs_int) - 1:
                f.write(f"{int(c)};\n")
            else:
                f.write(f"{int(c)},\n")
    print(f"Wrote {filename}")

def write_all_coeffs_coe(filename, all_coeffs_int, radix=10):
    total = sum(len(arr) for arr in all_coeffs_int)
    idx = 0
    with open(filename, "w") as f:
        f.write(f"radix = {radix};\n")
        f.write("coefdata = \n")
        for arr in all_coeffs_int:
            for c in arr:
                if idx == total - 1:
                    f.write(f"{int(c)};\n")
                else:
                    f.write(f"{int(c)},\n")
                idx += 1
    print(f"Wrote {filename} (all bands combined)")

all_coeffs_int = []
for h, b in zip(coeffs_final, bands):
    q = quantize_coeffs(h, coeff_bits)
    all_coeffs_int.append(q)
    fname = f"{b['name']}_fir_{coeff_bits}b_kaiser_scaled_math.coe"
    write_coe(fname, q, radix=10)

write_all_coeffs_coe("all_bands_combined_kaiser_scaled_math.coe", all_coeffs_int, radix=10)

# ============================
# 11) Plotting
# ============================

plt.figure(figsize=(10, 6))
for i, b in enumerate(bands):
    plt.plot(freqs, mag_all_db[i], alpha=0.6, label=b["name"])
plt.plot(freqs, mag_sum_final_db, linewidth=2.5, label="Sum of bands (scaled)")
plt.plot(freqs, noise_line_adc, linestyle="--", label=f"ADC noise floor ({adc_bits}-bit)")
plt.plot(freqs, noise_line_design, linestyle=":", label=f"Design noise floor ({design_bits}-bit)")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Multi-band FIR (Kaiser, all-math, scaled): bands, sum, and noise floors")
plt.ylim([-160, 10])
plt.grid(True, which="both", ls=":")
plt.xlim(0, 25000)   # hard cut at 20 kHz
plt.legend(loc="upper right", ncol=2, fontsize=8)
plt.tight_layout()
plt.show()
 