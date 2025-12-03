import numpy as np
import matplotlib.pyplot as plt

# ============================
# 0) Global settings
# ============================
fs = 48000.0        # sample rate in Hz
num_points = 32768  # FFT grid so narrow bands are resolved
adc_bits = 18       # ADC resolution (real analog path)
design_bits = 20    # target ENOB for filter stopband (digital design goal)
coeff_bits = 24     # word length for coefficient quantization to .coe
num_taps = 2048     # taps per band
eps = 1e-12         # small value for log/guard

# ============================
# 1) Kaiser + Bessel helpers
# ============================

def bessel_i0(x, terms=50):
    """
    Zeroth-order modified Bessel function I0(x) using its power series:
        I0(x) = sum_{k=0}^∞ ( (x^2/4)^k / (k!)^2 ).

    We truncate after `terms` terms.
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
    Map a desired ENOB -> approx attenuation A (dB) -> Kaiser beta.

    Ideal SNR(bits) ≈ 6.02 * bits + 1.76  (dB)
    Classic Kaiser approximations (Oppenheim & Schafer):

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
    # guard for tiny numerical negatives inside sqrt
    t = np.sqrt(np.clip(1.0 - r**2, 0.0, 1.0))
    return bessel_i0(beta * t) / bessel_i0(beta)


A_target_db, kaiser_beta = kaiser_beta_for_enob(design_bits)
print(f"Kaiser design target from {design_bits}-bit:")
print(f"  Target attenuation A_target ≈ {A_target_db:.2f} dB")
print(f"  Kaiser beta                 ≈ {kaiser_beta:.2f}")

# ============================
# 2) Band layout (even-width)
# ============================

def make_even_bands(f_min, f_max, num_bands, trans_frac=0.3):
    """
    Make evenly spaced bands between f_min and f_max (Hz).

    Each band k has:
      passband   [fpl, fpu]
      stopbands  (-inf, fsl] and [fsu, +inf)
    where fsl/fpu are expanded by trans_frac * band_width on each side.
    """
    bands = []
    width = (f_max - f_min) / num_bands
    for k in range(num_bands):
        name = f"Band{k}"
        fpl = f_min + k * width
        fpu = fpl + width
        trans = trans_frac * width
        fsl = max(0.0, fpl - trans)
        fsu = fpu + trans
        bands.append(dict(name=name, fpl=fpl, fpu=fpu, fsl=fsl, fsu=fsu))
    return bands

# Example: 20 equal-width bands between 35 Hz and 4515 Hz
bands = make_even_bands(35.0, 4515.0, 20, trans_frac=0.3)

print("\n=== Even-width band layout ===")
for b in bands:
    print(
        f"{b['name']}: fpl={b['fpl']:.1f} Hz, fpu={b['fpu']:.1f} Hz, "
        f"fsl={b['fsl']:.1f} Hz, fsu={b['fsu']:.1f} Hz"
    )

# ============================
# 3) Ideal bandpass (windowed-sinc) design
# ============================

def ideal_lowpass(num_taps, fc, fs):
    """
    Ideal lowpass impulse response with cutoff fc (Hz),
    sampled at fs (Hz), using the standard sinc formula:

        h_lp[n] = 2 * (fc/fs) * sinc( 2 * fc * (n - M) / fs )

    where M = (N-1)/2, and numpy's sinc(x) = sin(pi x)/(pi x).
    """
    n = np.arange(num_taps, dtype=np.float64)
    M = (num_taps - 1) / 2.0
    t = n - M
    return 2.0 * (fc / fs) * np.sinc(2.0 * fc * t / fs)


def ideal_bandpass(num_taps, fpl, fpu, fs):
    """
    Ideal bandpass by difference of two lowpasses:

        H_bp = H_lp(fc = fpu) - H_lp(fc = fpl)

    so in time domain:

        h_bp[n] = h_lp[n; fpu] - h_lp[n; fpl]
    """
    return ideal_lowpass(num_taps, fpu, fs) - ideal_lowpass(num_taps, fpl, fs)


def compute_response(h, fs, num_points):
    """
    Compute discrete-time frequency response via FFT.

      H[k] = sum_{n=0}^{N-1} h[n] * exp(-j 2 pi k n / N_fft)

    We use an N_fft = num_points real FFT (rfft) and map bins
    to physical frequency in Hz: 0 .. fs/2.
    """
    H = np.fft.rfft(h, n=num_points)
    # rfft length is num_points//2 + 1 bins spanning 0..fs/2
    freqs = np.linspace(0.0, fs / 2.0, len(H))
    return freqs, H


def design_bandpass_fir(num_taps, fs, fpl, fpu, beta, norm_to_center=True):
    """
    Design a bandpass FIR with passband [fpl, fpu] (Hz) using:

      1) ideal bandpass h_ideal[n] from sinc formulas
      2) Kaiser window w[n]
      3) optional normalization so that |H(f_center)| = 1 (0 dB),
         where f_center = (fpl+fpu)/2.

    This is mathematically equivalent to what scipy.signal.firwin()
    does for a Kaiser-windowed bandpass, but implemented by hand.
    """
    h_ideal = ideal_bandpass(num_taps, fpl, fpu, fs)
    w = kaiser_window(num_taps, beta)
    h = h_ideal * w  # windowed-sinc method

    if norm_to_center:
        # Normalize gain at band center to 0 dB
        f_center = 0.5 * (fpl + fpu)
        freqs_tmp, H_tmp = compute_response(h, fs, num_points=8192)
        idx = np.argmin(np.abs(freqs_tmp - f_center))
        gain_center = np.abs(H_tmp[idx])
        if gain_center > 0:
            h = h / gain_center

    return h

# ============================
# 4) Auto spec extraction
# ============================

def extract_band_specs(freqs, H, fpl, fpu, fsl, fsu):
    """
    Auto spec extraction:
    - passband ripple
    - worst-case stopband attenuation
    - approximate 3 dB edges on each side

    IMPORTANT: for stopband specs we ONLY consider the
    UPPER stopband (freqs >= fsu). The low-frequency side
    near 0 Hz is ignored when computing worst_stop_db/ENOB.
    """
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H), eps))

    # Passband mask
    pass_mask = (freqs >= fpl) & (freqs <= fpu)

    # If passband mask is empty (too narrow vs grid), pick closest bin to center
    if not np.any(pass_mask):
        center = 0.5 * (fpl + fpu)
        idx_center = np.argmin(np.abs(freqs - center))
        pass_mask = np.zeros_like(freqs, dtype=bool)
        pass_mask[idx_center] = True

    # Stopband: ONLY upper side (ignore everything <= fsl / near 0 Hz)
    stop_mask = (freqs >= fsu)

    # If stopband mask is empty, treat everything outside passband as stopband
    if not np.any(stop_mask):
        stop_mask = ~pass_mask

    pb_mag = mag_db[pass_mask]
    sb_mag = mag_db[stop_mask]

    pb_max = np.max(pb_mag)
    pb_min = np.min(pb_mag)
    ripple = pb_max - pb_min

    # "Worst" stopband is the point closest to 0 dB (max of sb_mag)
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
    }, mag_db

# ============================
# 5) Quantization & .coe writers
# ============================

def quantize_coeffs(coeffs, bits):
    """
    Quantize floating-point coefficients to signed integers with given bit width.
    E.g. bits=24 -> range [-2^(bits-1), 2^(bits-1)-1].

    We assume |h| <= 1 (after normalization) and map h -> round(h * (2^(bits-1) - 1)).
    """
    max_pos = 2**(bits - 1) - 1
    min_neg = -2**(bits - 1)
    scale = max_pos  # assume |h| <= 1

    q = np.round(coeffs * scale)
    q = np.clip(q, min_neg, max_pos)
    return q.astype(int)


def write_coe(filename, coeffs_int, radix=10):
    """
    Write a Vivado-style FIR coefficient .coe file like:

      radix = 10;
      coefdata = c0, c1, ..., cN;

    Negative values are written as signed integers in the given radix.
    """
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
    """
    Write ALL quantized coefficients for ALL bands into a single .coe file.

    Coefficients are concatenated in band order: Band0, Band1, ..., Band19.
    """
    total = sum(len(arr) for arr in all_coeffs_int)
    idx = 0

    with open(filename, "w") as f:
        f.write(f"radix = {radix};\n")
        f.write("coefdata = \n")
        for arr in all_coeffs_int:
            for c in arr:
                c_int = int(c)
                if idx == total - 1:
                    f.write(f"{c_int};\n")
                else:
                    f.write(f"{c_int},\n")
                idx += 1

    print(f"Wrote {filename} (all bands combined)")

# ============================
# 6) Design all bands & extract specs
# ============================

coeffs_list = []
H_all = []
mag_all_db = []
specs_per_band = []

for b in bands:
    h = design_bandpass_fir(num_taps, fs, b["fpl"], b["fpu"], kaiser_beta)
    coeffs_list.append(h)

# Use common frequency grid for all bands
freqs, _dummy_H = compute_response(coeffs_list[0], fs, num_points)

for h, b in zip(coeffs_list, bands):
    freqs, H = compute_response(h, fs, num_points)
    specs, mag_db = extract_band_specs(
        freqs, H,
        b["fpl"], b["fpu"],
        b["fsl"], b["fsu"]
    )
    H_all.append(H)
    mag_all_db.append(mag_db)
    specs_per_band.append((b["name"], specs))

H_all = np.array(H_all)          # shape (num_bands, num_freq_bins)
mag_all_db = np.array(mag_all_db)

# ============================
# 7) Combined response & overlap (UNSCALED)
# ============================

H_sum = np.sum(H_all, axis=0)
mag_sum_db = 20.0 * np.log10(np.maximum(np.abs(H_sum), eps))

# Overlap: "active" if within 6 dB of band max
threshold_db = -6.0
band_max_db = np.max(mag_all_db, axis=1, keepdims=True)
active_mask = mag_all_db >= (band_max_db + threshold_db)
active_count = np.sum(active_mask, axis=0)
max_active = int(np.max(active_count))

# ============================
# 8) Noise floors & ENOB helpers
# ============================

snr_adc_db = 6.02 * adc_bits + 1.76
noise_floor_adc_db = -snr_adc_db

snr_design_db = 6.02 * design_bits + 1.76
noise_floor_design_db = -snr_design_db

noise_line_adc = np.full_like(freqs, noise_floor_adc_db, dtype=np.float64)
noise_line_design = np.full_like(freqs, noise_floor_design_db, dtype=np.float64)

def enob_from_noise_floor(noise_floor_db):
    """
    Given a noise floor in dBFS (negative, e.g. -134 dB),
    return the *absolute* ENOB corresponding to that SNR.

    Ideal SNR ≈ 6.02 * N + 1.76  (dB)
    ⇒ N = (SNR - 1.76) / 6.02
    """
    snr_db = -noise_floor_db  # 0 dBFS - (negative noise) = positive SNR
    return (snr_db - 1.76) / 6.02

# ============================
# 9) Print auto-extracted specs + ENOB
# ============================

print("\n=== Auto Extracted Specs per Even-width Band (Kaiser, textbook) ===")
enob_list = []

for name, s in specs_per_band:
    worst_stop_db = s['worst_stop_db']    # noise floor (most-leaky point) in UPPER stopband

    # Margins vs ADC and vs 20-bit design target (in dB)
    stop_margin_adc_db = noise_floor_adc_db - worst_stop_db
    stop_margin_20_db = noise_floor_design_db - worst_stop_db

    # Total ENOB for this band from its worst stopband noise floor
    enob_total = enob_from_noise_floor(worst_stop_db)

    # ENOB margins vs ADC and vs design spec (in bits)
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

print("\n=== Overlap Check (within 6 dB of each band peak) ===")
print(f"  Max # of active bands at any frequency: {max_active}")
print("  (Goal: 2 or less; if >2, tweak band spacing or trans_frac.)")

# ============================
# 10) Generate .coe files
# ============================

all_coeffs_int = []

for h, b in zip(coeffs_list, bands):
    q = quantize_coeffs(h, coeff_bits)  # quantized h (normalized to ~1 max)
    all_coeffs_int.append(q)
    fname = f"{b['name']}_fir_{coeff_bits}b_kaiser_textbook.coe"
    write_coe(fname, q, radix=10)

write_all_coeffs_coe("all_bands_combined_kaiser_textbook.coe", all_coeffs_int, radix=10)

# ============================
# 11) Plotting
# ============================

plt.figure(figsize=(10, 6))
for i, b in enumerate(bands):
    plt.plot(freqs, mag_all_db[i], alpha=0.7, label=b["name"])
plt.plot(freqs, mag_sum_db, linewidth=2.0, label="Sum of bands (unscaled)")
plt.plot(freqs, noise_line_adc, linestyle="--", label=f"ADC noise floor ({adc_bits}-bit)")
plt.plot(freqs, noise_line_design, linestyle=":", label=f"Design noise floor ({design_bits}-bit)")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Even-width multi-band FIR (Kaiser, textbook): bands, sum, and noise floors")
plt.ylim([-160, 10])
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 3))
plt.plot(freqs, active_count)
plt.xlabel("Frequency (Hz)")
plt.ylabel("# of active bands (within 6 dB of peak)")
plt.title("How many bands are 'strong' at each frequency (even-width, Kaiser, textbook)")
plt.grid(True, which="both", ls=":")
plt.tight_layout()

# Uncomment when running locally to see plots
plt.show()
