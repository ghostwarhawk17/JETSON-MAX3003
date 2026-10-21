"""
Replay MIT-BIH records through the real-time pipeline and score it against the
cardiologist annotations. Needs only NumPy + SciPy and the Stage-1 files in
processed/ (signal_raw, ann_sym, ann_pos), so it also runs on the Jetson Nano.

Modes (each adds one live component, so the cost of each can be read off):
  reference  offline Stage-1 signal + annotated R-peaks -> must reproduce the paper's
             98.75 % beat for beat (checks the NumPy model and the RR features)
  oracle     raw signal through the streaming filter / z-score, annotated R-peaks
  detect     raw signal through the full live pipeline (R-peak detector included)
--device-fs 256 first resamples the 360 Hz record to the sensor's rate, so the
pipeline's resampler is exercised exactly as with the MAX30003.

    python deploy/validate_stream.py --mode reference
    python deploy/validate_stream.py --mode detect --device-fs 256
    python deploy/validate_stream.py --mode detect --records 100 233 --processed ./processed
"""
from __future__ import division, print_function

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ecg_rt import CLASSES, EcgPipeline, OraclePeaks   # noqa: E402

AAMI = {'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N', 'A': 'S', 'a': 'S', 'J': 'S',
        'S': 'S', 'V': 'V', 'E': 'V'}
BEATS = set('N L R B A a J S V r F e j n E / f Q ?'.split())
DS1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122',
       '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
DS2 = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210',
       '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']
FS, PRE, POST, TOL = 360, 99, 160, 54          # 54 samples = 150 ms matching window


def summarize(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    se = tp / np.maximum(cm.sum(1), 1)
    ppv = tp / np.maximum(cm.sum(0), 1)
    f1 = np.where(se + ppv > 0, 2 * se * ppv / np.maximum(se + ppv, 1e-12), 0.0)
    out = {'accuracy': float(tp.sum() / max(cm.sum(), 1)), 'macro_f1': float(f1.mean()),
           'cm': cm.astype(int).tolist()}
    for i, c in enumerate(CLASSES):
        out['se_' + c], out['ppv_' + c] = float(se[i]), float(ppv[i])
    return out


def fmt(m):
    return ('acc %6.2f | macro-F1 %6.2f | S Se/+P %5.1f/%5.1f | V Se/+P %5.1f/%5.1f'
            % (100 * m['accuracy'], 100 * m['macro_f1'], 100 * m['se_S'], 100 * m['ppv_S'],
               100 * m['se_V'], 100 * m['ppv_V']))


def match(det, ann):
    """Greedy one-to-one match of detections to annotations within TOL samples.
    Returns ann_to_det (index or -1)."""
    a2d = np.full(len(ann), -1)
    used = np.zeros(len(det), bool)
    j = np.searchsorted(det, ann)
    for k, a in enumerate(ann):
        best, bd = -1, TOL + 1
        for jj in (j[k] - 1, j[k], j[k] + 1):
            if 0 <= jj < len(det) and not used[jj] and abs(det[jj] - a) < bd:
                best, bd = jj, abs(det[jj] - a)
        if best >= 0:
            a2d[k] = best
            used[best] = True
    return a2d


def run_record(job):
    rid, args = job
    p = np.load(os.path.join(args.processed, '%s_processed.npz' % rid), allow_pickle=True)
    sym, pos = p['ann_sym'], p['ann_pos']
    is_beat = np.isin(sym, list(BEATS))
    sym, pos = sym[is_beat], pos[is_beat]
    if args.mode == 'reference':                 # Stage-2 rule: windows must fit
        n = len(p['signal_filtered'])
        ok = (pos - PRE >= 0) & (pos + POST + 1 <= n)
        sym, pos = sym[ok], pos[ok]
    in_ctx = np.isin(sym, list(AAMI))

    t0 = time.time()
    if args.mode == 'reference':
        x, fs_in = p['signal_filtered'], FS
    else:
        x, fs_in = p['signal_raw'], FS
        if args.device_fs and args.device_fs != FS:
            from fractions import Fraction
            from scipy.signal import resample_poly
            fr = Fraction(args.device_fs / FS).limit_denominator(1000)
            x, fs_in = resample_poly(x, fr.numerator, fr.denominator), args.device_fs
    pipe_kw = dict(lookahead_s=args.lookahead, norm_window_s=args.norm_window,
                   warmup_s=args.warmup, prenormalized=(args.mode == 'reference'),
                   detector_kwargs={'fiducial': args.fiducial})
    probe = EcgPipeline(args.bundle, fs_in, **pipe_kw)
    delay = int(probe.resampler.delay_out)
    if args.mode in ('reference', 'oracle'):
        pipe_kw['peaks'] = OraclePeaks(pos + delay, in_ctx)
    pipe = EcgPipeline(args.bundle, fs_in, **pipe_kw)
    beats = []
    for i in range(0, len(x), args.chunk):
        beats.extend(pipe.push(x[i:i + args.chunk]))
    beats.extend(pipe.flush())
    elapsed = time.time() - t0

    r_all = np.array(pipe.feat.peaks, dtype=np.int64) - delay       # every R-peak used
    r_out = np.array([b['r_index'] for b in beats], dtype=np.int64) - delay
    lab_out = np.array([CLASSES.index(b['label']) for b in beats], dtype=np.int64)
    prob_out = np.array([b['probs'] for b in beats]) if beats else np.zeros((0, 3))

    a2d = match(r_all, pos)                                          # detection quality
    det = {'tp': int((a2d >= 0).sum()), 'fn': int((a2d < 0).sum()),
           'fp': int(len(r_all) - (a2d >= 0).sum())}
    a2o = match(r_out, pos)                                          # classified beats
    aami = np.array([AAMI.get(s, '') for s in sym])
    scored = (aami != '') & (a2o >= 0)
    cm = np.zeros((3, 3), dtype=int)
    y_true = np.array([CLASSES.index(c) for c in aami[scored]], dtype=np.int64)
    y_pred = lab_out[a2o[scored]]
    np.add.at(cm, (y_true, y_pred), 1)
    lat = [b['latency_s'] for b in beats]
    return {'record': rid, 'cm': cm.tolist(), 'det': det,
            'aami_beats': int((aami != '').sum()), 'aami_scored': int(scored.sum()),
            'skipped_warmup': int(pipe.skipped_warmup), 'seconds_ecg': len(x) / float(fs_in),
            'seconds_cpu': elapsed, 'latency_median_s': float(np.median(lat)) if lat else None,
            'y_true': y_true.tolist(), 'prob': prob_out[a2o[scored]].tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['reference', 'oracle', 'detect'], default='detect')
    ap.add_argument('--records', nargs='+', default=None, help='default: DS2 without 222')
    ap.add_argument('--set', choices=['DS1', 'DS2'], default='DS2')
    ap.add_argument('--include-222', action='store_true')
    ap.add_argument('--processed', default=os.path.join(os.path.dirname(HERE), 'processed'))
    ap.add_argument('--bundle', default=os.path.join(HERE, 'model', 'mpfb_180hz_seed7.npz'))
    ap.add_argument('--device-fs', type=float, default=None, help='simulate the sensor rate, e.g. 256')
    ap.add_argument('--lookahead', type=float, default=3.0)
    ap.add_argument('--norm-window', type=float, default=300.0)
    ap.add_argument('--warmup', type=float, default=10.0)
    ap.add_argument('--fiducial', choices=['bp', 'abs', 'pos'], default='abs',
                    help="R-peak placement rule of the detector (default: %(default)s)")
    ap.add_argument('--chunk', type=int, default=97, help='samples per push (any value)')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--check-against', default=None,
                    help='results_full_ds2/180Hz_seed7.npz: compare reference probabilities')
    ap.add_argument('--out', default=None, help='write the summary JSON here')
    args = ap.parse_args()

    recs = args.records or [r for r in (DS2 if args.set == 'DS2' else DS1)
                            if args.include_222 or r != '222']
    jobs = [(r, args) for r in recs]
    if args.workers > 1:
        pool = Pool(args.workers)
        res = pool.map(run_record, jobs)
        pool.close()
    else:
        res = [run_record(j) for j in jobs]

    cm = np.sum([r['cm'] for r in res], axis=0)
    tp, fn, fp = (sum(r['det'][k] for r in res) for k in ('tp', 'fn', 'fp'))
    tot = summarize(cm)
    print('\nMode: %s%s | records: %d | lookahead %.1f s | z-score window %.0f s'
          % (args.mode, (' | sensor %g sps' % args.device_fs) if args.device_fs else '',
             len(res), args.lookahead, args.norm_window))
    print('%-6s %7s %7s %6s %6s %7s  %s' % ('record', 'beats', 'scored', 'FN', 'FP', 'acc%', 'S / V errors'))
    for r in res:
        m = summarize(r['cm'])
        c = np.array(r['cm'])
        print('%-6s %7d %7d %6d %6d %7.2f  S miss %d, N->S %d, V miss %d, N->V %d'
              % (r['record'], r['aami_beats'], r['aami_scored'], r['det']['fn'], r['det']['fp'],
                 100 * m['accuracy'], c[1, 0] + c[1, 2], c[0, 1], c[2, 0] + c[2, 1], c[0, 2]))
    beats = sum(r['aami_beats'] for r in res)
    scored = sum(r['aami_scored'] for r in res)
    print('\nR-peak detection (all beat annotations, 150 ms): Se %.3f %%  +P %.3f %%  (FN %d, FP %d)'
          % (100.0 * tp / max(tp + fn, 1), 100.0 * tp / max(tp + fp, 1), fn, fp))
    print('Classified N/S/V beats: %d of %d (%.2f %%; rest = missed or warm-up)'
          % (scored, beats, 100.0 * scored / max(beats, 1)))
    print('Classification: ' + fmt(tot))
    print('Confusion matrix (rows true N,S,V; cols predicted):\n  ' + '\n  '.join(str(r) for r in cm.tolist()))
    lat = [r['latency_median_s'] for r in res if r['latency_median_s'] is not None]
    secs, cpu = sum(r['seconds_ecg'] for r in res), sum(r['seconds_cpu'] for r in res)
    if lat:
        print('Median beat latency %.2f s | processing %.0fx faster than real time (this machine)'
              % (float(np.median(lat)), secs / max(cpu, 1e-9)))

    if args.check_against:
        ref = np.load(args.check_against, allow_pickle=True)
        same, dmax = True, 0.0
        for r in res:
            m = ref['record'] == r['record']
            pr, yr = ref['prob'][m], ref['y_true'][m]
            if len(pr) != len(r['prob']) or not np.array_equal(yr, r['y_true']):
                print('  !! %s: beat set differs (%d vs %d)' % (r['record'], len(pr), len(r['prob'])))
                same = False
                continue
            d = np.abs(np.array(r['prob']) - pr).max() if len(pr) else 0.0
            dmax = max(dmax, d)
            same = same and np.array_equal(np.argmax(r['prob'], 1), pr.argmax(1))
        print('Check vs %s: same beats and labels: %s (max |dp| %.1e)' % (args.check_against, same, dmax))

    if args.out:
        for r in res:
            r.pop('prob'), r.pop('y_true')
        with open(args.out, 'w') as f:
            json.dump({'args': vars(args), 'total': tot, 'detection': {'tp': tp, 'fn': fn, 'fp': fp},
                       'scored': scored, 'aami_beats': beats, 'records': res}, f, indent=1)


if __name__ == '__main__':
    main()
