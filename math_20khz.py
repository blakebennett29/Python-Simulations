import numpy as np
import matplotlib.pyplot as plt

# ============================
# 0) Global settings
# ============================
fs = 192000.0        # sample rate in Hz
num_points = 32768  # FFT grid so narrow bands are resolved
adc_bits = 18       # ADC resolution (real analog path)
design_bits = 20    # target ENOB for filter stopband (digital design goal)
coeff_bits = 24     # word length for coefficient quantization to .coe
num_taps = 2048     # taps per band (IP max)
eps = 1e-12         # small value for log/guard

# Band specs (log-ish spacing)
bands = [
    dict(name="Band0",  fpl=35,   fpu=45,    fsl=30,   fsu=52),
    dict(name="Band1",  fpl=45,   fpu=58,    fsl=38,   fsu=66),
    dict(name="Band2",  fpl=58,   fpu=73,    fsl=49,   fsu=84),
    dict(name="Band3",  fpl=73,   fpu=93,    fsl=62,   fsu=107),
    dict(name="Band4",  fpl=93,   fpu=119,   fsl=79,   fsu=137),
    dict(name="Band5",  fpl=119,  fpu=152,   fsl=101,  fsu=174),
    dict(name="Band6",  fpl=152,  fpu=193,   fsl=129,  fsu=222),
    dict(name="Band7",  fpl=193,  fpu=246,   fsl=164,  fsu=283),
    dict(name="Band8",  fpl=246,  fpu=314,   fsl=209,  fsu=361),
    dict(name="Band9",  fpl=314,  fpu=400,   fsl=267,  fsu=460),

    dict(name="Band10", fpl=400,  fpu=510,   fsl=340,  fsu=586),
    dict(name="Band11", fpl=510,  fpu=650,   fsl=433,  fsu=747),
    dict(name="Band12", fpl=650,  fpu=828,   fsl=552,  fsu=952),
    dict(name="Band13", fpl=828,  fpu=1055,  fsl=704,  fsu=1213),
    dict(name="Band14", fpl=1055, fpu=1344,  fsl=896,  fsu=1546),
    dict(name="Band15", fpl=1344, fpu=1713,  fsl=1142, fsu=1969),
    dict(name="Band16", fpl=1713, fpu=2182,  fsl=1456, fsu=2510),
    dict(name="Band17", fpl=2182, fpu=2781,  fsl=1855, fsu=3198),
    dict(name="Band18", fpl=2781, fpu=3543,  fsl=2364, fsu=4075),
    dict(name="Band19", fpl=3543, fpu=4515,  fsl=3012, fsu=5193),
]

# ============================
# 1) Kaiser + Bessel helpers (all math)
# ============================

def bessel_i0(x, terms=50):
    """
    Zeroth-order modified Bessel function I0(x) using its power series:
        I0(x) = sum_{k=0}^∞ ( (x^2/4)^k / (k!)^2 ).

    Oppenheim & Schafer style series approximation.
    """
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
    """
    Map desired ENOB -> approx attenuation A (dB) -> Kaiser beta.

    Ideal SNR(bits) ≈ 6.02 * bits + 1.76  (dB).

    Classic Kaiser approximations (per Oppenheim & Schafer):
      beta = 0                                  for A <= 21 dB
      beta = 0.5842*(A-21)^0.4 + 0.07886*(A-21) for 21 < A <= 50
      beta = 0.1102*(A-8.7)                     for A > 50
    """
    A = 6.02 * bits + 1.76  # target attenuation in dB
    if A <= 21:
        beta = 0.0
    elif A <= 50:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21)
    else:
        beta = 0.1102 * (A - 8.7)
    return A, beta


def kaiser_window(N, beta):
    """
    Kaiser window of length N:

        w[n] = I0( beta * sqrt(1 - ((n - alpha)/alpha)^2) ) / I0(beta)
        where alpha = (N-1)/2.
    """
    n = np.arange(N, dtype=np.float64)
    alpha = (N - 1) / 2.0
    r = (n - alpha) / alpha
    t = np.sqrt(np.clip(1.0 - r**2, 0.0, 1.0))  # guard small negatives
    return bessel_i0(beta * t) / bessel_i0(beta)


A_target_db, kaiser_beta = kaiser_beta_for_enob(design_bits)
print(f"Kaiser design target from {design_bits}-bit:")
print(f"  Target attenuation A_target ≈ {A_target_db:.2f} dB")
print(f"  Kaiser beta                 ≈ {kaiser_beta:.2f}")

# For Band 0: push stopband below 18-bit noise floor with a slightly stronger window
extra_bits_band0 = 2           # bump ENOB target for Band 0
bits_band0 = design_bits + extra_bits_band0
A0_db, kaiser_beta_band0 = kaiser_beta_for_enob(bits_band0)
print(f"\nBand0 uses stronger Kaiser:")
print(f"  Effective ENOB target      : {bits_band0} bits")
print(f"  A0 target                  : {A0_db:.2f} dB")
print(f"  Kaiser beta (Band0)        : {kaiser_beta_band0:.2f}")

# ============================
# 2) Ideal bandpass (windowed-sinc) design
# ============================

def ideal_lowpass(num_taps, fc, fs):
    """
    Ideal lowpass impulse response with cutoff fc (Hz),
    sampled at fs (Hz), using standard sinc formula:

        h_lp[n] = 2 * (fc/fs) * sinc( 2 * fc * (n - M) / fs )

    where M = (N-1)/2, numpy's sinc(x) = sin(pi x)/(pi x).
    """
    n = np.arange(num_taps, dtype=np.float64)
    M = (num_taps - 1) / 2.0
    t = n - M
    return 2.0 * (fc / fs) * np.sinc(2.0 * fc * t / fs)


def ideal_bandpass(num_taps, fpl, fpu, fs):
    """
    Ideal bandpass by difference of two lowpasses:

        H_bp = H_lp(fc = fpu) - H_lp(fc = fpl)
        h_bp[n] = h_lp[n; fpu] - h_lp[n; fpl]
    """
    return ideal_lowpass(num_taps, fpu, fs) - ideal_lowpass(num_taps, fpl, fs)


def compute_response(h, fs, num_points):
    """
    Compute discrete-time frequency response via FFT.

      H[k] = sum_{n=0}^{N-1} h[n] * exp(-j 2 pi k n / N_fft)

    Use real FFT (rfft) and map bins to 0..fs/2 Hz.
    """
    H = np.fft.rfft(h, n=num_points)
    freqs = np.linspace(0.0, fs / 2.0, len(H))
    return freqs, H


def design_bandpass_fir(num_taps, fs, fpl, fpu, beta, norm_to_center=True):
    """
    Design a bandpass FIR with passband [fpl, fpu] (Hz) using:

      1) ideal bandpass h_ideal[n] from sinc formulas
      2) Kaiser window w[n]
      3) optional normalization so that |H(f_center)| = 1 (0 dB),
         where f_center = (fpl+fpu)/2.
    """
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
# 3) Auto spec extraction (for ENOB & noise floor checks)
# ============================

def extract_band_specs(freqs, H, fpl, fpu, fsl, fsu):
    """
    Auto spec extraction:
    - passband ripple
    - worst-case stopband attenuation
    - approximate 3 dB edges on each side

    For stopband specs we ONLY consider the UPPER stopband (freqs >= fsu).
    """
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H), eps))

    # Passband mask
    pass_mask = (freqs >= fpl) & (freqs <= fpu)
    if not np.any(pass_mask):
        center = 0.5 * (fpl + fpu)
        idx_center = np.argmin(np.abs(freqs - center))
        pass_mask = np.zeros_like(freqs, dtype=bool)
        pass_mask[idx_center] = True

    # Stopband: only upper side
    stop_mask = (freqs >= fsu)
    if not np.any(stop_mask):
        stop_mask = ~pass_mask

    pb_mag = mag_db[pass_mask]
    sb_mag = mag_db[stop_mask]

    pb_max = np.max(pb_mag)
    pb_min = np.min(pb_mag)
    ripple = pb_max - pb_min

    # "Worst" stopband: point closest to 0 dB (max of sb_mag)
    worst_stop = np.max(sb_mag)

    # Approx 3 dB edges around passband peak
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
    """
    Given a noise floor in dBFS (negative, e.g. -134 dB),
    return the ENOB corresponding to that SNR.

    Ideal SNR ≈ 6.02 * N + 1.76  (dB)
    ⇒ N = (SNR - 1.76) / 6.02
    """
    snr_db = -noise_floor_db
    return (snr_db - 1.76) / 6.02

snr_adc_db = 6.02 * adc_bits + 1.76
noise_floor_adc_db = -snr_adc_db
snr_design_db = 6.02 * design_bits + 1.76
noise_floor_design_db = -snr_design_db

# ============================
# 4) Design all bands (Band0 stronger Kaiser)
# ============================

coeffs_list = []
H_all = []
specs_per_band = []

for i, b in enumerate(bands):
    if i == 0:
        # Band0: use stronger Kaiser to push stopband below 18-bit noise floor
        beta_use = kaiser_beta_band0
    else:
        beta_use = kaiser_beta

    h = design_bandpass_fir(num_taps, fs, b["fpl"], b["fpu"], beta_use)
    coeffs_list.append(h)

# Use common frequency grid
freqs, _dummy = compute_response(coeffs_list[0], fs, num_points)
for h, b in zip(coeffs_list, bands):
    _, H = compute_response(h, fs, num_points)
    H_all.append(H)
    specs = extract_band_specs(freqs, H, b["fpl"], b["fpu"], b["fsl"], b["fsu"])
    specs_per_band.append((b["name"], specs))

H_all = np.array(H_all)  # shape (B, N)

# ============================
# 5) Print per-band specs & ENOB
# ============================

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
    print(f"  Margin vs {adc_bits}-bit noise floor   : {stop_margin_adc_db:.3f} dB "
          f"(>0 => stopband below ADC noise)")
    print(f"  Margin vs {design_bits}-bit noise floor: {stop_margin_20_db:.3f} dB "
          f"(>0 => stopband below {design_bits}-bit noise)")
    print(f"  ENOB from stopband noise floor (total) : {enob_total:.2f} bits")
    print(f"    ENOB margin vs {adc_bits}-bit ADC    : {enob_margin_vs_adc:.2f} bits")
    print(f"    ENOB margin vs {design_bits}-bit spec: {enob_margin_vs_design:.2f} bits")

# Limiting band (by stopband attenuation / ENOB)
limiting_band_name, limiting_enob = min(enob_list, key=lambda x: x[1])

print("\n=== Overall Filter Bank Limits ===")
print(f"  Limiting band (by stopband leakage): {limiting_band_name}")
print(f"  ENOB limited by worst stopband     : {limiting_enob:.2f} bits")
print(f"  ADC ENOB (ideal, from {adc_bits} bits) : {adc_bits:.2f} bits")
print(f"  Design target ENOB (filter only)       : {design_bits:.2f} bits")

# Explicitly check Band0 vs ADC noise floor
for name, s in specs_per_band:
    if name == "Band0":
        print(f"\nBand0 worst stopband vs 18-bit floor:")
        print(f"  Band0 worst_stop_db : {s['worst_stop_db']:.3f} dBFS")
        print(f"  18-bit noise floor  : {noise_floor_adc_db:.3f} dBFS")
        print(f"  Margin (floor - worst) = {noise_floor_adc_db - s['worst_stop_db']:.3f} dB")

# ============================
# 6) Per-band taper (scaled bands) – same logic as your firwin version
# ============================

coeffs_list = np.array(coeffs_list)  # shape (B, taps)
# H_all already computed for each band

band_centers = [0.5 * (b["fpl"] + b["fpu"]) for b in bands]

num_bands_to_taper = 13   # 0..11 tapered, 12..19 left ~unity
target_mag = 1.0          # target combined magnitude at each center (before global scale)
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
        # Only cut (no boosting)
        gains[k] = min(gk, 1.0)
gains[0]  = 0.395776  # was ≈ 0.98281  (≈ -3 dB extra cut)
print("\nPer-band taper gains (linear):")
for i, g in enumerate(gains):
    print(f"  {bands[i]['name']}: {g:.6f}")

# Apply per-band taper
coeffs_tapered = coeffs_list * gains[:, None]

# ============================
# 7) Global gain so orange line ~0 dB on average
# ============================

# Sum response with only per-band taper
H_sum_tapered = np.sum(
    [compute_response(h, fs, num_points)[1] for h in coeffs_tapered],
    axis=0
)
mag_sum_tapered = np.abs(H_sum_tapered)

f_roi_min = bands[0]["fpl"]
f_roi_max = bands[-1]["fpu"]
roi_mask = (freqs >= f_roi_min) & (freqs <= f_roi_max)

avg_mag = mag_sum_tapered[roi_mask].mean()
g_global2 = 1.0 / avg_mag          # make average 0 dB across ROI
g_global2_db = 20.0 * np.log10(g_global2)

print(f"\nGlobal gain g_global2 = {g_global2:.6f} ({g_global2_db:.3f} dB)")

# Final coefficients used in hardware:
coeffs_final = coeffs_tapered * g_global2

# ============================
# 8) Frequency responses, band heights, and "undo" gains
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

# Net attenuation per band and gain to undo it
scale_total = gains * g_global2
atten_db = 20.0 * np.log10(np.maximum(scale_total, eps))
undo_db = -atten_db

# Actual peak height of each band (after all scaling)
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
# 9) Noise floors & ENOB helpers
# ============================

noise_line_adc = np.full_like(freqs, noise_floor_adc_db)
noise_line_design = np.full_like(freqs, noise_floor_design_db)

# ============================
# 10) Quantization & .coe writers (scaled coefficients)
# ============================

def quantize_coeffs(coeffs, bits):
    """
    Quantize floating-point coefficients to signed integers with given bit width.
    bits=24 -> range [-2^(bits-1), 2^(bits-1)-1].
    """
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
    plt.plot(freqs, mag_all_db[i], alpha=0.7, label=b["name"])
plt.plot(freqs, mag_sum_final_db, linewidth=2.5, label="Sum of bands (scaled)", color="orange")
plt.plot(freqs, noise_line_adc, linestyle="--", label=f"ADC noise floor ({adc_bits}-bit)")
plt.plot(freqs, noise_line_design, linestyle=":", label=f"Design noise floor ({design_bits}-bit)")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Multi-band FIR (Kaiser, all-math, scaled): bands, sum, and noise floors")
plt.ylim([-160, 10])
plt.grid(True, which="both", ls=":")
plt.xlim(0, 12000)
plt.legend()
plt.tight_layout()

plt.show()
