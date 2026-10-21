"""
Beat assembly: R-peaks + filtered signal -> the model's three inputs, exactly as
02_segmentation_features.py and new_student.py built them offline.

For beat k (R-peak r_k, all times in 360 Hz samples):
  window    s[r_k - 99 : r_k + 161], z-scored, then every 2nd sample keeping r_k
            -> 130 samples at 180 Hz with the R-peak at index 49
  pre_RR    (r_k - r_{k-1}) / 360 (beat 0: its post-RR, as Stage 2 does)
  post_RR   (r_{k+1} - r_k) / 360            -> the beat waits for the next R-peak
  local / global mean = mean pre_RR of the last 80 / 400 beats including beat k
  features  pre/local, post/local, pre/global, post/global
  context   median |change in pre/local| over the previous 11 changes (beats
            k-12..k-1; current beat excluded; 0 until two changes exist)
The 5 features are then standardised with the DS1-training mean / SD.
"""
from __future__ import division

from collections import deque

import numpy as np


class BeatFeaturizer(object):
    def __init__(self, fs=360, pre=99, post=160, decimate=2, local_beats=80,
                 global_beats=400, context_window=12):
        self.fs, self.pre, self.post, self.dec = fs, pre, post, decimate
        self.local, self.glob = local_beats, global_beats
        self.ctx_n = context_window - 1                       # changes in the context median
        self.reset()

    def reset(self):
        self.peaks = []                 # absolute R-peak indices
        self.ctx_flag = []              # beat takes part in the RR-context sequence
        self.pre_rr = []                # seconds, per beat (filled once known)
        self.p_local = []               # pre_rr / local mean, per beat
        self.ctx_prev = None            # p_local of the previous context beat
        self.ctx_changes = deque(maxlen=self.ctx_n)
        self.ctx_value = {}             # beat -> context feature
        self.next_beat = 0              # next beat to emit

    def add_peak(self, idx, in_context=True):
        self.peaks.append(int(idx))
        self.ctx_flag.append(bool(in_context))
        k = len(self.peaks) - 1
        if k == 1:                       # beat 0's pre-RR = first interval (Stage 2 rule)
            self._set_pre(0, self.peaks[1] - self.peaks[0])
        if k >= 1:
            self._set_pre(k, self.peaks[k] - self.peaks[k - 1])

    def _set_pre(self, k, samples):
        self.pre_rr.append(samples / float(self.fs))
        assert len(self.pre_rr) == k + 1
        lo = max(0, k - self.local + 1)
        local_mean = float(np.mean(self.pre_rr[lo:k + 1]))
        self.p_local.append(self.pre_rr[k] / (local_mean + 1e-8))
        if self.ctx_flag[k]:
            ch = [c for c in self.ctx_changes if c == c]      # drop the NaN start marker
            self.ctx_value[k] = float(np.median(ch)) if len(ch) >= 2 else 0.0
            self.ctx_changes.append(abs(self.p_local[k] - self.ctx_prev)
                                    if self.ctx_prev is not None else float('nan'))
            self.ctx_prev = self.p_local[k]

    def features(self, k, post_samples=None):
        pre = self.pre_rr[k]
        post = (self.peaks[k + 1] - self.peaks[k] if post_samples is None else post_samples) / float(self.fs)
        local = float(np.mean(self.pre_rr[max(0, k - self.local + 1):k + 1]))
        glob = float(np.mean(self.pre_rr[max(0, k - self.glob + 1):k + 1]))
        eps = 1e-8
        return np.array([pre / (local + eps), post / (local + eps), pre / (glob + eps),
                         post / (glob + eps), self.ctx_value.get(k, 0.0)])

    def window(self, sig, k, mean, std):
        """Normalised, decimated beat window of beat k from a SampleBuffer (or None if
        the window starts before the buffer)."""
        r = self.peaks[k]
        a, b = r - self.pre, r + self.post + 1
        if a < sig.start or b > sig.end:
            return None
        w = (sig.get(a, b) - mean) / std
        return w[self.pre % self.dec::self.dec]

    def ready(self, sig_end, final=False):
        """Beats whose next R-peak is known and whose window is complete; with
        final=True the last beat is released too (post-RR = its pre-RR, Stage 2 rule)."""
        out = []
        while self.next_beat < len(self.peaks) - 1 or (final and self.next_beat < len(self.peaks)):
            k = self.next_beat
            if self.peaks[k] + self.post + 1 > sig_end or k >= len(self.pre_rr):
                break
            out.append(k)
            self.next_beat += 1
        return out

    def post_samples(self, k):
        if k + 1 < len(self.peaks):
            return None
        return self.peaks[k] - self.peaks[k - 1] if k > 0 else None
