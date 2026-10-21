"""
NumPy forward pass of the MPFB matched-filter CNN (new_student.py, 180 Hz model).

No TensorFlow needed: the weights are exported once on the PC by export_model.py into
an .npz bundle. BatchNorm (inference statistics) is folded into the Conv1D weights,
which is exact because tanh follows the BatchNorm.

    raw beat (130) --Conv1D(32 taps, 22 filters)+BN--> tanh --> max over time (22)
    derivative (129) --Conv1D(32 taps, 22 filters)+BN--> tanh --> max over time (22)
    RR features (5) --per feature Dense(32,16,8) ReLU--> flatten (40)
    concat (84) --> Dense(3) softmax -> p(N), p(S), p(V)
"""
from __future__ import division

import json

import numpy as np
from numpy.lib.stride_tricks import as_strided

CLASSES = ('N', 'S', 'V')


def _frames(x, width):
    """(B, T) -> (B, T - width + 1, width) sliding windows (a view, no copy)."""
    x = np.ascontiguousarray(x)
    b, t = x.shape
    s0, s1 = x.strides
    return as_strided(x, shape=(b, t - width + 1, width), strides=(s0, s1, s1))


def _relu(x):
    return np.maximum(x, 0.0)


class MPFBModel(object):
    def __init__(self, bundle_path):
        z = np.load(bundle_path, allow_pickle=False)
        self.config = json.loads(str(z['config_json']))
        self.w_raw, self.b_raw = z['conv_raw_w'], z['conv_raw_b']        # (taps, NF), (NF,)
        self.w_der, self.b_der = z['conv_der_w'], z['conv_der_b']
        self.rr_layers = [(z['rr_w%d' % i], z['rr_b%d' % i]) for i in range(3)]
        self.w_out, self.b_out = z['out_w'], z['out_b']                  # (84, 3), (3,)
        self.rr_mu, self.rr_sd = z['rr_mu'], z['rr_sd']                  # DS1-train statistics
        self.taps = self.w_raw.shape[0]
        self.beat_len = int(self.config['beat_len'])

    def standardize_rr(self, rr):
        return (np.asarray(rr, dtype=np.float64) - self.rr_mu) / self.rr_sd

    def _bank(self, x, w, b):
        h = np.tanh(np.matmul(_frames(x, self.taps), w) + b)              # (B, T', NF)
        return h.max(axis=1)

    def predict(self, beats, rr_std):
        """beats: (B, beat_len) normalised beats at 180 Hz (R-peak at config['r_index']);
        rr_std: (B, n_rr) standardised RR features. Returns (B, 3) probabilities."""
        beats = np.atleast_2d(np.asarray(beats, dtype=np.float64))
        rr = np.atleast_2d(np.asarray(rr_std, dtype=np.float64))
        if beats.shape[1] != self.beat_len:
            raise ValueError('expected %d-sample beats, got %d' % (self.beat_len, beats.shape[1]))
        f_raw = self._bank(beats, self.w_raw, self.b_raw)
        f_der = self._bank(np.diff(beats, axis=1), self.w_der, self.b_der)
        r = rr[:, :, None]                                                # per-feature Dense stack
        for w, b in self.rr_layers:
            r = _relu(np.matmul(r, w) + b)
        r = r.reshape(len(rr), -1)                                        # Keras Flatten order
        z = np.concatenate([f_raw, f_der, r], axis=1).dot(self.w_out) + self.b_out
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
