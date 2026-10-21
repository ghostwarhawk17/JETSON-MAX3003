#!/usr/bin/env python3
"""
Real-time N / S / V heartbeat classification on the Jetson Nano.

    # MAX30003 wired to the Jetson's SPI header (see README):
    sudo python3 run_live.py --source spi --log beats.csv --save-raw raw.csv

    # MAX30003 on an Arduino / ESP32 running firmware/max30003_bridge over USB:
    python3 run_live.py --source serial --port /dev/ttyACM0

    # No hardware: replay an MIT-BIH record at the sensor's rate, in real time,
    # with the cardiologist's label shown next to each prediction:
    python3 run_live.py --source replay --file processed/233_processed.npz --sim-fs 256

Output, one line per beat (about 5 s after it happened):
    [  62.41 s] beat   71  HR  76  S  (N 0.08 S 0.90 V 0.02)  <-- supraventricular ectopic
The first ~10 s after start (or after an electrode comes off) are warm-up.
"""
from __future__ import division, print_function

import argparse
import collections
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ecg_rt import EcgPipeline                                   # noqa: E402

NAMES = {'S': 'supraventricular ectopic', 'V': 'ventricular ectopic'}


def make_source(a):
    if a.source == 'spi':
        from ecg_rt.sources import Max30003Source
        return Max30003Source(a.spi_bus, a.spi_dev, rate=a.rate, gain=a.gain, invert=a.invert,
                              dhpf=a.dhpf)
    if a.source == 'serial':
        from ecg_rt.sources import SerialBridgeSource
        return SerialBridgeSource(a.port, a.baud, fs=a.fs)
    from ecg_rt.sources import ReplaySource
    if not a.file:
        sys.exit('--source replay needs --file')
    return ReplaySource(a.file, speed=a.speed, sim_fs=a.sim_fs, start_s=a.start, duration_s=a.duration)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', choices=['spi', 'serial', 'replay'], default='spi')
    ap.add_argument('--bundle', default=os.path.join(HERE, 'model', 'mpfb_180hz_seed7.npz'))
    g = ap.add_argument_group('MAX30003 on the Jetson SPI header')
    g.add_argument('--spi-bus', type=int, default=0)
    g.add_argument('--spi-dev', type=int, default=0)
    g.add_argument('--rate', type=int, default=256, help='sps: 512/256/128 (default 256)')
    g.add_argument('--gain', type=int, default=20, help='V/V: 20/40/80/160')
    g.add_argument('--invert', action='store_true', help='flip ECG polarity (R-waves negative)')
    g.add_argument('--dhpf', action='store_true', help='enable the on-chip 0.5 Hz high-pass')
    g = ap.add_argument_group('MAX30003 via a USB serial bridge')
    g.add_argument('--port', default='/dev/ttyACM0')
    g.add_argument('--baud', type=int, default=115200)
    g.add_argument('--fs', type=float, default=None, help='override the bridge sample rate')
    g = ap.add_argument_group('replay')
    g.add_argument('--file', help='processed/<record>_processed.npz or a --save-raw CSV')
    g.add_argument('--speed', type=float, default=1.0, help='1 = real time, 0 = as fast as possible')
    g.add_argument('--sim-fs', type=float, default=None, help='resample to this sensor rate first')
    g.add_argument('--start', type=float, default=0.0, help='start offset in the recording (s)')
    g.add_argument('--duration', type=float, default=None)
    g = ap.add_argument_group('pipeline and output')
    g.add_argument('--lookahead', type=float, default=3.0, help='zero-phase filter lookahead (s)')
    g.add_argument('--warmup', type=float, default=10.0)
    g.add_argument('--log', help='write one CSV row per classified beat')
    g.add_argument('--save-raw', help='save the raw mV samples (replayable with --source replay)')
    g.add_argument('--quiet', action='store_true', help='print only S / V beats and summaries')
    g.add_argument('--summary-every', type=float, default=10.0, help='seconds between summaries')
    a = ap.parse_args()

    src = make_source(a)
    pipe = EcgPipeline(a.bundle, src.fs, lookahead_s=a.lookahead, warmup_s=a.warmup)
    print('MPFB beat classifier | source %s at %g sps -> 360 Hz | model %s'
          % (a.source, src.fs, os.path.basename(a.bundle)))
    print('warming up (%.0f s); each beat is reported about %.1f s + one beat interval after it occurs'
          % (a.warmup, pipe.latency_floor_s + 0.8))

    log_f = open(a.log, 'w') if a.log else None
    log = csv.writer(log_f) if log_f else None
    if log:
        log.writerow(['beat', 't_s', 'wall_clock', 'label', 'p_N', 'p_S', 'p_V', 'hr_bpm',
                      'rr_pre_s', 'rr_post_s', 'latency_s'] + (['reference'] if a.source == 'replay' else []))
    raw_f = open(a.save_raw, 'w') if a.save_raw else None
    if raw_f:
        raw_f.write('# fs=%g source=%s started=%s\nindex,mV\n' % (src.fs, a.source, time.strftime('%Y-%m-%d %H:%M:%S')))
    n_raw = 0

    recent = collections.deque()                 # (t, label, hr) of the last 60 s
    counts = collections.Counter()
    polarity, polarity_checked = [], False
    agree = collections.Counter()
    t_last_summary = time.time()
    try:
        for x, event in src.chunks():
            if event in ('gap', 'leads_off'):
                print('*** %s - restarting (warm-up %.0f s after signal returns)'
                      % ('LEADS OFF' if event == 'leads_off' else 'data gap / saturation', a.warmup))
                pipe.reset()
                if raw_f:
                    raw_f.write('# event %s at index %d\n' % (event, n_raw))
                continue
            if event == 'leads_on':
                print('*** leads on')
                continue
            if raw_f and len(x):
                raw_f.write(''.join('%d,%.5f\n' % (n_raw + i, v) for i, v in enumerate(x)))
                n_raw += len(x)
            for b in pipe.push(x):
                lab, p = b['label'], b['probs']
                counts[lab] += 1
                recent.append((b['t'], lab, b['hr_bpm']))
                ref = src.reference_label(b['t']) if a.source == 'replay' else None
                if ref is not None:
                    agree[(ref, lab)] += 1
                if not polarity_checked:
                    polarity.append(b['r_amp'])
                    if len(polarity) >= 20:
                        polarity_checked = True
                        if np.median(polarity) < 0:
                            print('!!! The R-waves are mostly NEGATIVE. The model was trained on lead II '
                                  '(MIT-BIH MLII) with upright R-waves: swap the electrodes or use --invert.')
                if not a.quiet or lab != 'N':
                    print('[%8.2f s] beat %5d  HR %3.0f  %s  (N %.2f S %.2f V %.2f)%s%s'
                          % (b['t'], b['beat'], b['hr_bpm'], lab, p[0], p[1], p[2],
                             ('  <-- ' + NAMES[lab]) if lab in NAMES else '',
                             ('   [reference %s]' % ref) if ref is not None else ''))
                if log:
                    log.writerow([b['beat'], '%.3f' % b['t'], time.strftime('%H:%M:%S'), lab,
                                  '%.4f' % p[0], '%.4f' % p[1], '%.4f' % p[2], '%.1f' % b['hr_bpm'],
                                  '%.3f' % b['rr_pre_s'], '%.3f' % b['rr_post_s'], '%.2f' % b['latency_s']]
                                 + ([ref or ''] if a.source == 'replay' else []))
            while recent and recent[-1][0] - recent[0][0] > 60.0:
                recent.popleft()
            if time.time() - t_last_summary >= a.summary_every and recent:
                t_last_summary = time.time()
                c = collections.Counter(l for _, l, _ in recent)
                print('--- last %.0f s: %d beats, median HR %.0f bpm | N %d  S %d  V %d'
                      % (recent[-1][0] - recent[0][0], len(recent),
                         np.median([h for _, _, h in recent]), c['N'], c['S'], c['V']))
                if log_f:
                    log_f.flush()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nTotal classified beats: %d  (N %d, S %d, V %d)'
              % (sum(counts.values()), counts['N'], counts['S'], counts['V']))
        if agree:
            aami = {'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N', 'A': 'S', 'a': 'S', 'J': 'S',
                    'S': 'S', 'V': 'V', 'E': 'V'}
            ok = sum(n for (r, l), n in agree.items() if aami.get(r) == l)
            tot = sum(n for (r, l), n in agree.items() if r in aami)
            print('Agreement with the MIT-BIH annotations (N/S/V beats): %d / %d = %.2f %%'
                  % (ok, tot, 100.0 * ok / max(tot, 1)))
        for f in (log_f, raw_f):
            if f:
                f.close()


if __name__ == '__main__':
    main()
