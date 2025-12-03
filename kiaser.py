import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

# ============================
# 0) Global settings
# ============================
fs = 48000.0        # sample rate in Hz
num_points = 32768  # finer grid so narrow bands are resolved
adc_bits = 18       # ADC resolution (real analog path)
design_bits = 20    # target ENOB for filter stopband (digital design goal)
coeff_bits = 24     # word length for coefficient quantization to .coe
num_taps = 2048     # taps per band (IP max)
eps = 1e-12         # small value for log/guard

# Band specs
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
# 1) Helper functions
# ============================

def kaiser_beta_for_enob(bits):
    A = 6.02 * bits + 1.76  # target attenuation in dB
    if A <= 21:
        beta = 0.0
    elif A <= 50:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21)
    else:
        beta = 0.1102 * (A - 8.7)
    return A, beta

A_target_db, kaiser_beta = kaiser_beta_for_enob(design_bits)
print(f"Kaiser design target from {design_bits}-bit:")
print(f"  Target attenuation A_target ≈ {A_target_db:.2f} dB")
print(f"  Kaiser beta                 ≈ {kaiser_beta:.2f}")

def design_bandpass_fir(num_taps, fs, fpl, fpu, beta):
    """Design a bandpass FIR with passband [fpl, fpu] (Hz) using Kaiser."""
    w1 = fpl / (fs / 2.0)
    w2 = fpu / (fs / 2.0)
    return firwin(
        num_taps,
        [w1, w2],
        pass_zero=False,
        window=('kaiser', beta)
    )

def compute_response(h, fs, num_points):
    """Return (freqs_Hz, H_complex)."""
    w, H = freqz(h, worN=num_points)
    freqs = w * fs / (2.0 * np.pi)
    return freqs, H

def quantize_coeffs(coeffs, bits):
    """Quantize coefficients to signed integers with given bit width."""
    max_pos = 2**(bits - 1) - 1
    min_neg = -2**(bits - 1)
    scale = max_pos
    q = np.round(coeffs * scale)
    q = np.clip(q, min_neg, max_pos)
    return q.astype(int)

def write_coe(filename, coeffs_int, radix=10):
    with open(filename, "w") as f:
        f.write(f"radix={radix};\n")
        f.write("coefdata=\n")
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

# ============================
# 2) Design all bands (unscaled)
# ============================

coeffs_list = []
for b in bands:
    h = design_bandpass_fir(num_taps, fs, b["fpl"], b["fpu"], kaiser_beta)
    coeffs_list.append(h)

coeffs_list = np.array(coeffs_list)  # shape (B, taps)

# Common freq grid
freqs, _dummy = compute_response(coeffs_list[0], fs, num_points)

# Complex responses for each band (unscaled)
H_all = []
for h in coeffs_list:
    _, H = compute_response(h, fs, num_points)
    H_all.append(H)
H_all = np.array(H_all)  # shape (B, N)

# ============================
# 3) Automatic low-band taper
# ============================

num_bands_to_taper = 12   # 0..11 tapered, 12..19 left ~unity
target_mag = 1.0          # target combined magnitude at each center (before global scale)
gains = np.ones(len(bands))

band_centers = [0.5 * (b["fpl"] + b["fpu"]) for b in bands]

for k in range(num_bands_to_taper):
    f_center = band_centers[k]
    idx = np.argmin(np.abs(freqs - f_center))
    Hk_mag = np.abs(H_all[k, idx])
    existing_mag = np.abs(np.dot(gains[:k], H_all[:k, idx])) if k > 0 else 0.0
    if Hk_mag < 1e-9:
        gains[k] = 0.0
    else:
        # Only cut (no boosting): cap at 1.0
        gk = max(0.0, (target_mag - existing_mag) / Hk_mag)
        gains[k] = min(gk, 1.0)

print("\nPer-band taper gains (linear):")
for i, g in enumerate(gains):
    print(f"  {bands[i]['name']}: {g:.6f}")

# Apply per-band taper
coeffs_tapered = coeffs_list * gains[:, None]

# ============================
# 4) Global gain so orange line ~0 dB on average
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
g_global2 = 1.0 / avg_mag          # make average 0 dB
g_global2_db = 20.0 * np.log10(g_global2)

print(f"\nGlobal gain g_global2 = {g_global2:.6f} ({g_global2_db:.3f} dB)")

# Final coefficients used in hardware:
coeffs_final = coeffs_tapered * g_global2

# ============================
# 5) Frequency responses, band heights, and "undo" gains
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
# 6) Noise floors & ENOB helpers
# ============================

snr_adc_db = 6.02 * adc_bits + 1.76
noise_floor_adc_db = -snr_adc_db

snr_design_db = 6.02 * design_bits + 1.76
noise_floor_design_db = -snr_design_db

noise_line_adc = np.full_like(freqs, noise_floor_adc_db)
noise_line_design = np.full_like(freqs, noise_floor_design_db)

# ============================
# 7) Generate .coe files using scaled coefficients
# ============================

all_coeffs_int = []

for h, b in zip(coeffs_final, bands):
    q = quantize_coeffs(h, coeff_bits)
    all_coeffs_int.append(q)
    fname = f"{b['name']}_fir_{coeff_bits}b_kaiser_scaled.coe"
    write_coe(fname, q, radix=10)

write_all_coeffs_coe("all_bands_combined_kaiser_scaled.coe", all_coeffs_int, radix=10)

# ============================
# 8) Plotting
# ============================

plt.figure(figsize=(10, 6))
for i, b in enumerate(bands):
    plt.plot(freqs, mag_all_db[i], alpha=0.7, label=b["name"])

plt.plot(freqs, mag_sum_final_db, linewidth=2.5, label="Sum of bands (scaled)", color="orange")

plt.plot(freqs, noise_line_adc, linestyle="--", label=f"ADC noise floor ({adc_bits}-bit)")
plt.plot(freqs, noise_line_design, linestyle=":", label=f"Design noise floor ({design_bits}-bit)")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Multi-band FIR (Kaiser, scaled): bands, sum, and noise floors")
plt.ylim([-160, 10])
plt.grid(True, which="both", ls=":")
plt.xlim(0, 12000)
plt.legend()
plt.tight_layout()

plt.show()
