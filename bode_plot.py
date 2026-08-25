"""Frequency-response (Bode) analysis for fluorescent protein maturation models."""

import numpy as np
from scipy import signal


def transfer_function_1step(alpha, km, kb, kd):
    num = [alpha * km]
    den = np.polymul([1, km + kd], [1, kb + kd])
    return signal.TransferFunction(num, den)


def transfer_function_2step(alpha, k1, k2, kb, kd):
    a1 = k1 + kd
    a2 = k2 + kd
    a3 = kb + kd
    num = [alpha * k1 * k2]
    den = np.polymul(np.polymul([1, a1], [1, a2]), [1, a3])
    return signal.TransferFunction(num, den)


def bode_1step(alpha, km, kb, kd, w):
    sys = transfer_function_1step(alpha, km, kb, kd)
    return signal.bode(sys, w=w)


def bode_2step(alpha, k1, k2, kb, kd, w):
    sys = transfer_function_2step(alpha, k1, k2, kb, kd)
    return signal.bode(sys, w=w)


def analytical_cutoff_1step(km, kb, kd):
    """Exact -3 dB cutoff frequency for the 1-step model's two real poles."""
    p1 = km + kd
    p2 = kb + kd
    a = 1
    b = p1**2 + p2**2
    c = p1**2 * p2**2 * (1 - 2)
    discriminant = b**2 - 4 * a * c
    wc2 = (-b + np.sqrt(discriminant)) / (2 * a)
    return np.sqrt(wc2)


def analytical_cutoff_2step(k1, k2, kb, kd):
    """Exact -3 dB cutoff frequency for the 2-step model's three real poles.

    |H(jw)|^2 = 1/2 * |H(0)|^2 leads to a cubic in x = wc^2:
    x^3 + S1*x^2 + S2*x - P = 0, which has exactly one positive real root.
    """
    p1_sq = (k1 + kd) ** 2
    p2_sq = (k2 + kd) ** 2
    p3_sq = (kb + kd) ** 2

    S1 = p1_sq + p2_sq + p3_sq
    S2 = p1_sq * p2_sq + p1_sq * p3_sq + p2_sq * p3_sq
    P = p1_sq * p2_sq * p3_sq

    roots = np.roots([1, S1, S2, -P])
    real_positive = roots[np.isclose(roots.imag, 0) & (roots.real > 0)].real
    if len(real_positive) == 0:
        return None
    return np.sqrt(real_positive[0])


def numerical_cutoff(w, mag):
    """Interpolate the -3 dB (half power) cutoff frequency from a magnitude curve."""
    target = mag[0] - 3.0
    idx = np.where(mag <= target)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return w[0]
    w1, w2 = w[i - 1], w[i]
    m1, m2 = mag[i - 1], mag[i]
    frac = (target - m1) / (m2 - m1)
    return w1 + frac * (w2 - w1)
