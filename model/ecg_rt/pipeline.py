"""
Real-time pipeline: ECG samples at the sensor's rate in, one classified beat out per
detected heartbeat.

    sensor samples (fs_in)
      -> StreamResampler          to 360 Hz (the MIT-BIH rate the model was built on)
      -> ZeroPhaseFIR             Stage-1 filter (0.5-45 Hz + 50 Hz notch), zero phase
      -> TrailingStats            z-score (trailing window instead of whole record)
      -> QRSDetector              R-peaks
      -> BeatFeaturizer           130-sample beat + 5 RR features
      -> MPFBModel                p(N), p(S), p(V)

Work is done in fixed hops of 90 samples (0.25 s at 360 Hz), so the output does
not depend on how the source chunks its samples. Latency per beat is the filter
lookahead (3 s) + the detector margin (0.8 s) + waiting for the next R-peak.
"""
from __future__ import division

import numpy as np

from .beats import BeatFeaturizer
from .dsp import SampleBuffer, StreamResampler, TrailingStats, ZeroPhaseFIR
from .model import CLASSES, MPFBModel
from .qrs import QRSDetector


class OraclePeaks(object):
    """Known R-peak positions (validation): each is released once `margin` samples
    after it have been processed, like the detector's commit rule."""

    def __init__(self, positions, in_context=None, margin=288):
        self.pos = np.asarray(positions, dtype=np.int64)
        self.ctx = np.ones(len(self.pos), bool) if in_context is None else np.asarray(in_context, bool)
        self.margin = margin
        self.i = 0

    def reset(self):
        pass

    def update(self, buf, end, final=False):
        out = []
        while self.i < len(self.pos) and (final or self.pos[self.i] < end - self.margin):
            out.append((int(self.pos[self.i]), bool(self.ctx[self.i])))
            self.i += 1
        return out


class _DetectorPeaks(object):
    def __init__(self, det):
        self.det = det

    def reset(self):
        self.det.reset()

    def update(self, buf, end, final=False):
        return [(p, True) for p in self.det.update(buf, end)]


class EcgPipeline(object):
    FS = 360

    def __init__(self, bundle, fs_in, lookahead_s=3.0, norm_window_s=300.0, warmup_s=10.0,
                 hop=90, peaks=None, prenormalized=False, detector_kwargs=None):
        """bundle: path to the exported .npz (or an MPFBModel). fs_in: sensor rate in Hz.
        prenormalized=True: input is already the Stage-1 filtered, z-scored 360 Hz
        signal (validation only). peaks: an OraclePeaks (validation) or None for the
        built-in detector."""
        self.model = bundle if isinstance(bundle, MPFBModel) else MPFBModel(bundle)
        cfg = self.model.config
        self.fs_in = float(fs_in)
        self.prenormalized = prenormalized
        self.lookahead = 0 if prenormalized else int(round(lookahead_s * self.FS))
        self.norm_window_s, self.warmup_s, self.hop = norm_window_s, warmup_s, hop
        self.sos = np.load(bundle)['filter_sos'] if isinstance(bundle, str) else None
        self._oracle = peaks
        self._det_kwargs = detector_kwargs or {}
        self._cfg = cfg
        self.t_offset = 0.0              # seconds of input before the last reset
        self.n_in = 0                    # input samples since the start
        self.n_beats = 0
        self.reset(keep_time=False)

    def reset(self, keep_time=True):
        """Start over (after a lead-off or data gap): filters, statistics and RR history."""
        if keep_time:
            self.t_offset = self.n_in / self.fs_in
        if self.prenormalized:
            self.resampler = StreamResampler(self.FS, self.FS)
        else:
            self.resampler = StreamResampler(self.fs_in, self.FS)
            if self.sos is None:
                raise ValueError('pass the bundle path so the filter can be loaded')
            self.filter = ZeroPhaseFIR(self.sos, self.lookahead, fs=self.FS)
        self.stats = TrailingStats(self.FS, self.norm_window_s, self.warmup_s)
        self.sig = SampleBuffer()
        self.pending = np.zeros(0)
        cfg = self._cfg
        self.feat = BeatFeaturizer(cfg['fs'], cfg['pre'], cfg['post'], cfg['decimate'],
                                   cfg['rr_local_beats'], cfg['rr_global_beats'],
                                   cfg['rr_context_window'])
        if self._oracle is not None:
            self.peaks = self._oracle
        else:
            self.peaks = _DetectorPeaks(QRSDetector(self.FS, **self._det_kwargs))
        self.peaks.reset()
        self.skipped_warmup = 0

    @property
    def latency_floor_s(self):
        return (self.lookahead + getattr(self.peaks, 'margin', 0)) / self.FS + self.resampler.delay_s

    def push(self, x):
        """Feed samples (any number, at fs_in). Returns a list of beat dicts."""
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n_in += len(x)
        y = self.resampler.process(x)
        self.pending = np.concatenate([self.pending, y]) if len(self.pending) else y
        out = []
        while len(self.pending) >= self.hop:
            chunk, self.pending = self.pending[:self.hop], self.pending[self.hop:]
            out.extend(self._step(chunk))
        return out

    def flush(self):
        """End of a finite recording: process the partial hop and release the last
        beats (validation)."""
        out = self._step(self.pending, final=True) if len(self.pending) else []
        self.pending = np.zeros(0)
        for idx, ctx in self.peaks.update(self.sig, self.sig.end, final=True):
            self.feat.add_peak(idx, ctx)
        return out + self._emit(self.feat.ready(self.sig.end, final=True))

    def _step(self, chunk, final=False):
        f = chunk if self.prenormalized else self.filter.process(chunk)
        self.sig.append(f)
        if not self.prenormalized:
            self.stats.update(f)
        for idx, ctx in self.peaks.update(self.sig, self.sig.end, final):
            self.feat.add_peak(idx, ctx)
        out = self._emit(self.feat.ready(self.sig.end))
        keep = self.sig.end - 12 * self.FS
        if self.feat.next_beat < len(self.feat.peaks):
            keep = min(keep, self.feat.peaks[self.feat.next_beat] - self.feat.pre - 1)
        self.sig.trim_before(max(self.sig.start, keep))
        return out

    def _emit(self, ks):
        if not ks:
            return []
        if self.prenormalized:
            mean, std = 0.0, 1.0
        elif not self.stats.ready:
            self.skipped_warmup += len(ks)
            return []
        else:
            mean, std = self.stats.mean_std()
        rows, wins, rrs = [], [], []
        for k in ks:
            w = self.feat.window(self.sig, k, mean, std)
            if w is None:
                continue
            rr = self.feat.features(k, self.feat.post_samples(k))
            rows.append((k, rr, float(w[self._cfg['r_index']])))
            wins.append(w)
            rrs.append(rr)
        if not rows:
            return []
        probs = self.model.predict(np.array(wins), self.model.standardize_rr(np.array(rrs)))
        # stream index -> seconds of sensor time since start (resampler delay removed)
        now = self.n_in / self.fs_in
        out = []
        for (k, rr, r_amp), p in zip(rows, probs):
            r = self.feat.peaks[k]
            t = self.t_offset + (r - self.resampler.delay_out) / float(self.FS)
            pre_s = self.feat.pre_rr[k]
            post_s = rr[1] / rr[0] * pre_s if rr[0] > 0 else float('nan')
            out.append({'beat': self.n_beats, 'r_index': r, 't': t,
                        'label': CLASSES[int(np.argmax(p))], 'probs': p,
                        'rr_pre_s': pre_s, 'rr_post_s': post_s,
                        'hr_bpm': 60.0 / pre_s if pre_s > 0 else float('nan'),
                        'rr_features': rr, 'r_amp': r_amp, 'latency_s': now - t})
            self.n_beats += 1
        return out
