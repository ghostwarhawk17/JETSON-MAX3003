"""
Streaming R-peak detector.

The model was trained on beats centred on the MIT-BIH annotation positions, so live
it needs R-peaks from a detector. This uses the two-event-related-moving-averages
method (M. Elgendi, PLoS ONE 8(9):e73557, 2013; Se 99.78 % / +P 99.87 % on MIT-BIH):
    y  = (8-20 Hz band-pass of the ECG)^2
    blocks of interest where MA_97ms(y) > MA_611ms(y) + 0.08 * mean(y),
    blocks at least 97 ms wide, one beat per block.
Streaming: every call re-analyses the last `window_s` seconds; a beat is committed
only once `margin_s` of signal after it has been seen (centred moving averages and
the filter's edges), so the decision never changes afterwards. Two beats closer than
`refractory_s` keep the stronger one. The fiducial point is then placed on the
model's own (0.5-45 Hz) signal, see `refine`.
"""
from __future__ import division

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _centred_ma(x, w):
    """Centred moving average of width w (shrinks at the edges)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = len(x)
    i = np.arange(n)
    lo = np.clip(i - w // 2, 0, n)
    hi = np.clip(i - w // 2 + w, 0, n)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


class QRSDetector(object):
    def __init__(self, fs=360.0, window_s=8.0, margin_s=0.8, refractory_s=0.2,
                 w1_s=0.097, w2_s=0.611, beta=0.08, search_s=0.05, fiducial='abs'):
        self.fs = float(fs)
        nyq = self.fs / 2.0
        self.sos = butter(3, [8.0 / nyq, 20.0 / nyq], btype='band', output='sos')
        self.win = int(window_s * fs)
        self.margin = int(margin_s * fs)
        self.refr = int(refractory_s * fs)
        self.w1, self.w2 = int(round(w1_s * fs)), int(round(w2_s * fs))
        self.beta = beta
        self.search = int(round(search_s * fs))
        self.fiducial = fiducial
        self.last = None                    # last committed peak (absolute index)

    def reset(self):
        self.last = None

    def candidates(self, x):
        """Beats in a signal segment x (indices relative to x), refractory-resolved."""
        y = sosfiltfilt(self.sos, x)
        e = y * y
        ma1, ma2 = _centred_ma(e, self.w1), _centred_ma(e, self.w2)
        on = ma1 > ma2 + self.beta * e.mean()
        edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(np.int8), [0]])))
        peaks, strength = [], []
        for s, t in zip(edges[::2], edges[1::2]):
            if t - s < self.w1:
                continue
            p = s + int(np.argmax(np.abs(y[s:t])))
            peaks.append(self.refine(x, p))
            strength.append(np.abs(y[s:t]).max())
        if not peaks:
            return np.zeros(0, dtype=np.int64)
        peaks, strength = np.array(peaks), np.array(strength)
        keep = []
        for i in np.argsort(-strength):                       # strongest first
            if all(abs(peaks[i] - peaks[j]) >= self.refr for j in keep):
                keep.append(i)
        return np.sort(peaks[keep])

    def refine(self, x, p):
        """Fiducial point on the model's signal near the detector peak p: the largest
        deflection ('abs'), the largest positive value ('pos'), or p itself ('bp')."""
        if self.fiducial == 'bp':
            return p
        a, b = max(0, p - self.search), min(len(x), p + self.search + 1)
        seg = x[a:b]
        if self.fiducial == 'pos':
            return a + int(np.argmax(seg))
        base = np.median(x[max(0, p - int(0.3 * self.fs)):p + int(0.3 * self.fs)])
        return a + int(np.argmax(np.abs(seg - base)))

    def update(self, buf, end):
        """buf: SampleBuffer of the model's filtered signal, valid up to `end`.
        Returns newly committed R-peak indices (absolute, ascending)."""
        a = max(buf.start, end - self.win)
        if end - a < 2 * self.margin + self.w2:
            return []
        x = buf.get(a, end)
        cand = self.candidates(x) + a
        safe_lo = a + self.margin if a > buf.start else a
        new = []
        for p in cand:
            if p < safe_lo or p >= end - self.margin:
                continue
            if self.last is not None and p < self.last + self.refr:
                continue
            new.append(int(p))
            self.last = int(p)
        return new
