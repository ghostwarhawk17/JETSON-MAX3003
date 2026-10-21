"""
Export the 98.75 % MPFB model (results_default_180Hz/new_student_seed7.keras) into a
TensorFlow-free bundle for the Jetson Nano, and check the NumPy forward pass against
Keras on every DS2 beat. Run on the PC (needs TensorFlow + the project's features):

    python deploy/export_model.py

The bundle (deploy/model/mpfb_180hz_seed7.npz) holds:
  * Conv1D weights with BatchNorm folded in, the RR Dense stack and the output layer;
  * the DS1-training mean / SD of the 5 RR features (pre/post RR over the 80- and
    400-beat means + RR context), computed exactly as new_student.py does;
  * the Stage-1 filter (3rd-order Butterworth 0.5-45 Hz + 50 Hz notch, Q 30) as
    second-order sections, and the windowing constants.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from scipy.signal import butter, iirnotch, tf2sos
import tensorflow as tf

import new_student as ns
from ecg_rt.model import MPFBModel

MODEL = "results_default_180Hz/new_student_seed7.keras"
SAVED_RUN = "results_default_180Hz/new_student_results.npz"
OUT = os.path.join(HERE, "model", "mpfb_180hz_seed7.npz")
TRAIN = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122',
         '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
TEST = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210',
        '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']
FS, PRE, POST, DECIMATE = 360, 99, 160, 2


def fold_bn(conv, bn):
    w, b = conv.get_weights()                         # (taps, 1, NF), (NF,)
    gamma, beta, mean, var = bn.get_weights()
    scale = gamma / np.sqrt(var + bn.epsilon)
    return (w[:, 0, :] * scale).astype(np.float64), ((b - mean) * scale + beta).astype(np.float64)


def bn_after(model, conv_name):
    out_len = model.get_layer(conv_name).output.shape[1]
    bns = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)
           and l.output.shape[1] == out_len]
    assert len(bns) == 1, conv_name
    return bns[0]


def stage1_sos():
    nyq = FS / 2.0
    bp = butter(3, [0.5 / nyq, 45.0 / nyq], btype='band', output='sos')
    b, a = iirnotch(50.0 / nyq, 30.0)
    return np.vstack([bp, tf2sos(b, a)])


def main():
    model = tf.keras.models.load_model(MODEL)
    beat_len, n_rr = int(model.inputs[0].shape[1]), int(model.inputs[2].shape[-1])
    assert beat_len == 130 and n_rr == 5, (beat_len, n_rr)

    w_raw, b_raw = fold_bn(model.get_layer('matched_filter_raw'), bn_after(model, 'matched_filter_raw'))
    w_der, b_der = fold_bn(model.get_layer('matched_filter'), bn_after(model, 'matched_filter'))
    dense = {tuple(l.get_weights()[0].shape): l for l in model.layers
             if isinstance(l, tf.keras.layers.Dense)}
    rr_stack = [dense[(1, 32)], dense[(32, 16)], dense[(16, 8)]]
    out = dense[(84, 3)]

    # RR statistics exactly as new_student.py / eval_with_222.py build them.
    X360, labels, rr4, _, rid = ns.load_dataset("./features_perrecord_aami")
    rr_ctx = ns.rr_irregularity(rr4[:, 0].copy(), rid, 12)
    keep = np.isin(labels, list(ns.AAMI_MAP))
    y = np.array([ns.CLASSES.index(ns.AAMI_MAP[s]) for s in labels[keep]])
    X360, rr, rid = X360[keep], np.column_stack([rr4, rr_ctx])[keep], rid[keep]
    train, test = np.isin(rid, TRAIN), np.isin(rid, TEST)
    mu, sd = rr[train].mean(0), rr[train].std(0) + 1e-8

    config = {
        "name": "MPFB new_student 180 Hz seed 7 (DS2 98.75 % without record 222)",
        "source_model": MODEL, "classes": list(ns.CLASSES),
        "fs": FS, "pre": PRE, "post": POST, "decimate": DECIMATE,
        "beat_len": beat_len, "r_index": (PRE - PRE % DECIMATE) // DECIMATE,
        "rr_local_beats": 80, "rr_global_beats": 400, "rr_context_window": 12,
        "rr_features": ["pre_rr_local", "post_rr_local", "pre_rr_global", "post_rr_global",
                        "rr_context"],
        "filter": "Butterworth order 3 band-pass 0.5-45 Hz + iirnotch 50 Hz Q 30, zero phase",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    arrays = dict(conv_raw_w=w_raw, conv_raw_b=b_raw, conv_der_w=w_der, conv_der_b=b_der,
                  out_w=out.get_weights()[0].astype(np.float64),
                  out_b=out.get_weights()[1].astype(np.float64),
                  rr_mu=mu.astype(np.float64), rr_sd=sd.astype(np.float64),
                  filter_sos=stage1_sos(), config_json=np.array(json.dumps(config)))
    for i, l in enumerate(rr_stack):
        arrays['rr_w%d' % i] = l.get_weights()[0].astype(np.float64)
        arrays['rr_b%d' % i] = l.get_weights()[1].astype(np.float64)
    np.savez(OUT, **arrays)
    print("wrote %s (%.1f kB)" % (OUT, os.path.getsize(OUT) / 1024.0))

    # Check: NumPy vs Keras on all 22 DS2 records, and vs the saved 21-record run.
    X = X360[:, PRE % DECIMATE::DECIMATE]
    rr_std = ((rr - mu) / sd).astype(np.float32)
    p_keras = model.predict([X[test][..., None], np.diff(X[test], axis=1)[..., None], rr_std[test]],
                            batch_size=4096, verbose=0)
    npm = MPFBModel(OUT)
    p_np = npm.predict(X[test], npm.standardize_rr(rr[test]))
    in21 = ~np.isin(rid[test], ['222'])
    saved = np.load(SAVED_RUN)["y_pred_prob"]
    m21 = ns.summarize(y[test][in21], p_np[in21].argmax(1))
    print("NumPy vs Keras, %d DS2 beats: max |dp| = %.2e, same labels: %s"
          % (len(p_np), np.abs(p_np - p_keras).max(),
             np.array_equal(p_np.argmax(1), p_keras.argmax(1))))
    print("NumPy vs saved run (21 records): max |dp| = %.2e, same labels: %s"
          % (np.abs(p_np[in21] - saved).max(), np.array_equal(p_np[in21].argmax(1), saved.argmax(1))))
    print("NumPy model, DS2 without 222: " + ns.fmt(m21))


if __name__ == "__main__":
    main()
