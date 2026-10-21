"""
Streaming signal processing that reproduces the offline Stage-1 preprocessing.

Offline (01_preprocessing.py) every 30-min record was band-passed 0.5-45 Hz and
notched at 50 Hz with filtfilt (zero phase), then z-scored with the WHOLE record's
mean and SD. Live, filtfilt is replaced by its effective impulse response (the
autocorrelation of the IIR response, which is what forward + backward filtering
applies) truncated to +/- `lookahead` and run as a linear-phase FIR, so every output
sample is released `lookahead` after it arrives. With 3 s the result is within
~2 % (worst sample) of the offline filtfilt. The z-score uses a trailing window
instead of the whole record.

Python 3.6 / SciPy >= 0.19 compatible (JetPack 4.6 on the Jetson Nano).
"""
from __future__ import division

from collections import deque
from fractions import Fraction

import numpy as np
from scipy.signal import sosfilt


class SampleBuffer(object):
    """1-D signal addressed by absolute sample index; old samples can be dropped."""

    def __init__(self):
        self.data = np.zeros(0)
        self.start = 0

    @property
    def end(self):
        return self.start + len(self.data)

    def append(self, x):
        if len(x):
            self.data = np.concatenate([self.data, np.asarray(x, dtype=np.float64)])

    def get(self, a, b):
        if a < self.start or b > self.end:
            raise IndexError('samples [%d, %d) not in buffer [%d, %d)' % (a, b, self.start, self.end))
        return self.data[a - self.start:b - self.start]

    def trim_before(self, idx):
        k = idx - self.start
        if k > 0:
            self.data = self.data[k:].copy()
            self.start = idx


def kaiser_lowpass(num_taps, cutoff, beta=5.0):
    """Windowed-sinc low-pass with unit DC gain (= scipy.signal.firwin with a Kaiser
    window); `cutoff` is a fraction of the Nyquist rate."""
    n = np.arange(num_taps) - (num_taps - 1) / 2.0
    h = cutoff * np.sinc(cutoff * n) * np.kaiser(num_taps, beta)
    return h / h.sum()


class StreamResampler(object):
    """Rational L/M polyphase resampler (same filter design as scipy's resample_poly),
    processing an unbounded stream chunk by chunk. Output sample m corresponds to input
    time m / fs_out - delay_s (constant group delay of the anti-aliasing filter)."""

    def __init__(self, fs_in, fs_out, half_len=10):
        r = Fraction(float(fs_out) / float(fs_in)).limit_denominator(1000)
        self.L, self.M = r.numerator, r.denominator
        self.fs_in, self.fs_out = float(fs_in), float(fs_out)
        self.passthrough = (self.L == self.M == 1)
        max_rate = max(self.L, self.M)
        # ~half_len input samples each side, rounded so the delay is a whole output sample
        ntaps = 2 * self.M * int(np.ceil(half_len * max_rate / self.M)) + 1
        h = kaiser_lowpass(ntaps, 1.0 / max_rate) * self.L
        self.J = int(np.ceil(ntaps / self.L))
        hp = np.zeros(self.J * self.L)
        hp[:ntaps] = h
        self.H = hp.reshape(self.J, self.L).T                  # H[phase, j] = h[phase + j L]
        self.delay_out = 0 if self.passthrough else (ntaps - 1) // (2 * self.M)
        self.delay_s = self.delay_out / self.fs_out
        self.x = SampleBuffer()
        self.m = 0                                            # next output index

    def process(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.passthrough:
            return x
        self.x.append(x)
        L, M, J = self.L, self.M, self.J
        m_end = (self.x.end * L + M - 1) // M                 # outputs whose newest input exists
        if m_end <= self.m:
            return np.zeros(0)
        ms = np.arange(self.m, m_end, dtype=np.int64)
        t = ms * M
        nmax = t // L
        idx = nmax[:, None] - np.arange(J)[None, :]
        valid = idx >= self.x.start                           # before the stream start = zeros
        vals = np.zeros(idx.shape)
        vals[valid] = self.x.data[idx[valid] - self.x.start]
        y = np.sum(self.H[t - nmax * L] * vals, axis=1)
        self.m = m_end
        self.x.trim_before(max(0, (self.m * M) // L - J))
        return y


def zero_phase_kernel(sos, half_width, decay_s=20.0, fs=360.0):
    """Impulse response of filtfilt(sos) for lags -half_width..half_width: the
    autocorrelation of the forward impulse response. After truncation the DC gain is
    set back to exactly zero (a band-pass must not pass the electrode offset)."""
    n = max(int(decay_s * fs), 4 * half_width)
    d = np.zeros(n)
    d[0] = 1.0
    h = sosfilt(np.asarray(sos, dtype=np.float64), d)
    one_side = np.array([np.dot(h[:n - k], h[k:]) for k in range(half_width + 1)])
    g = np.concatenate([one_side[:0:-1], one_side])
    w = np.hanning(len(g) + 2)[1:-1]
    return g - w * g.sum() / w.sum()


class ZeroPhaseFIR(object):
    """Streaming zero-phase filter: output sample k is aligned with input sample k and
    released once input k + lookahead has arrived. The signal is extended backwards
    with its first value, which a band-pass turns into no transient."""

    def __init__(self, sos, lookahead, fs=360.0):
        self.K = int(lookahead)
        self.g = zero_phase_kernel(sos, self.K, fs=fs)
        self.buf = SampleBuffer()
        self.done = 0                                         # next output index
        self.started = False

    def process(self, x):
        x = np.asarray(x, dtype=np.float64)
        if len(x) and not self.started:
            self.buf.start = -self.K
            self.buf.append(np.full(self.K, x[0]))
            self.started = True
        self.buf.append(x)
        out_end = self.buf.end - self.K
        if out_end <= self.done:
            return np.zeros(0)
        seg = self.buf.get(self.done - self.K, self.buf.end)
        y = np.convolve(seg, self.g, mode='valid')
        self.done = out_end
        self.buf.trim_before(self.done - self.K)
        return y


class TrailingStats(object):
    """Mean / SD over the last `window_s` seconds, kept as per-second sums (so the cost
    does not grow with the window). Stands in for Stage 1's whole-record z-score."""

    def __init__(self, fs, window_s=300.0, min_s=10.0):
        self.chunk = int(round(fs))
        self.chunks = deque(maxlen=max(1, int(round(window_s))))
        self.min_n = int(round(min_s * fs))
        self.cur = np.zeros(3)                                # n, sum, sum of squares

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        i = 0
        while i < len(x):
            take = min(len(x) - i, self.chunk - int(self.cur[0]))
            seg = x[i:i + take]
            self.cur += (len(seg), seg.sum(), np.dot(seg, seg))
            i += take
            if self.cur[0] >= self.chunk:
                self.chunks.append(self.cur.copy())
                self.cur[:] = 0.0

    def _totals(self):
        tot = self.cur.copy()
        for c in self.chunks:
            tot += c
        return tot

    @property
    def ready(self):
        return self._totals()[0] >= self.min_n

    def mean_std(self):
        n, s, ss = self._totals()
        mean = s / n
        return mean, np.sqrt(max(ss / n - mean * mean, 1e-12))
