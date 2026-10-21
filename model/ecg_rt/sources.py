"""
ECG sample sources. Every source has `.fs` (Hz) and `.chunks()`, a generator of
(samples_mV, event) where event is None or one of
    'gap'        samples were lost or invalid before this chunk (FIFO overflow, dropped
                 serial lines, fast-recovery after saturation): restart the pipeline
    'leads_off'  an electrode came off (no samples until 'leads_on')
    'leads_on'   electrodes back on
"""
from __future__ import division

import queue
import threading
import time

import numpy as np

EMPTY = np.zeros(0)


class Max30003Source(object):
    """MAX30003 wired straight to the Jetson's SPI. A reader thread empties the
    32-sample FIFO every `poll_s` so it cannot overflow while the main thread works
    (the FIFO lasts 125 ms at 256 sps)."""

    def __init__(self, bus=0, device=0, rate=256, gain=20, invert=False, dhpf=False,
                 speed_hz=2000000, poll_s=0.01):
        from .max30003 import MAX30003
        self.dev = MAX30003(bus, device, speed_hz)
        self.info = self.dev.configure(rate=rate, gain=gain, invert=invert, dhpf=dhpf)
        self.fs = self.dev.fs
        self.poll_s = poll_s
        self.q = queue.Queue()
        self._stop = False

    def _run(self):
        leads_off = False
        try:
            while not self._stop:
                st = self.dev.status()
                if st['leads_off'] != leads_off:
                    leads_off = st['leads_off']
                    self.q.put((EMPTY, 'leads_off' if leads_off else 'leads_on'))
                vals, invalid, overflow = self.dev.read_fifo()
                if overflow or st['overflow']:
                    self.q.put((EMPTY, 'gap'))
                elif invalid:
                    self.q.put((EMPTY, 'gap'))
                elif vals and not leads_off:
                    self.q.put((self.dev.counts_to_mv(np.array(vals, dtype=np.float64)), None))
                time.sleep(self.poll_s)
        except Exception as e:                 # surface SPI errors in the main thread
            self.q.put((EMPTY, e))

    def chunks(self):
        t = threading.Thread(target=self._run)
        t.daemon = True
        t.start()
        try:
            while True:
                x, ev = self.q.get()
                if isinstance(ev, Exception):
                    raise ev
                yield x, ev
        finally:
            self._stop = True
            t.join(1.0)
            self.dev.close()


class SerialBridgeSource(object):
    """MAX30003 on an Arduino / ESP32 running firmware/max30003_bridge, sending
    'seq,counts,flags' lines (seq 0-255, flags bit0 fast-mode, bit1 leads off,
    bit2 FIFO overflow) and '#MAX30003 fs=256 gain=20' header lines."""

    def __init__(self, port, baud=115200, fs=None, gain=None, header_timeout_s=8.0):
        import serial                          # pip3 install pyserial
        self.ser = serial.Serial(port, baud, timeout=1.0)
        self.fs, self.gain = fs, gain
        t_end = time.time() + header_timeout_s
        while (self.fs is None or self.gain is None) and time.time() < t_end:
            line = self.ser.readline().decode('ascii', 'ignore').strip()
            if line.startswith('#'):
                self._parse_header(line)
        if self.fs is None:
            raise RuntimeError('no "#MAX30003 fs=..." header from %s; pass --fs' % port)
        self.gain = self.gain or 20

    def _parse_header(self, line):
        for tok in line.replace(',', ' ').split():
            if tok.startswith('fs=') and self.fs is None:
                self.fs = float(tok[3:])
            elif tok.startswith('gain=') and self.gain is None:
                self.gain = int(tok[5:])

    def chunks(self, batch=16):
        last, buf, leads_off = None, [], False
        scale = 1000.0 / (131072.0 * self.gain)
        while True:
            line = self.ser.readline().decode('ascii', 'ignore').strip()
            if not line or line.startswith('#'):
                continue
            try:
                seq, val, flags = [int(v) for v in line.split(',')]
            except ValueError:
                continue
            event = None
            if last is not None and seq != (last + 1) % 256:
                event = 'gap'
            last = seq
            if bool(flags & 2) != leads_off:
                leads_off = bool(flags & 2)
                event = 'leads_off' if leads_off else 'leads_on'
            elif flags & 5:
                event = 'gap'
            if event is not None:
                if buf:
                    yield np.array(buf) * scale, None
                    buf = []
                yield EMPTY, event
            if not leads_off and not (flags & 5):
                buf.append(val)
            if len(buf) >= batch:
                yield np.array(buf) * scale, None
                buf = []


class ReplaySource(object):
    """Replay a recording: a Stage-1 processed/*.npz (MIT-BIH, with annotations) or a
    CSV saved by run_live.py --save-raw. sim_fs resamples it to the sensor's rate
    first; speed 1 = real time, 0 = as fast as possible."""

    def __init__(self, path, speed=1.0, sim_fs=None, start_s=0.0, duration_s=None):
        self.annotations = None
        if path.endswith('.npz'):
            z = np.load(path, allow_pickle=True)
            x, fs = np.asarray(z['signal_raw'], dtype=np.float64), float(z['fs'])
            if 'ann_pos' in z.files:
                sym, pos = np.asarray(z['ann_sym']).astype(str), np.asarray(z['ann_pos'])
                beat = np.isin(sym, list('NLRBAaJSVrFejnE/fQ?'))
                self.annotations = (sym[beat], pos[beat], fs)
        else:
            fs, rows = None, []
            with open(path) as f:
                for line in f:
                    if line.startswith('#'):
                        for tok in line[1:].split():
                            if tok.startswith('fs='):
                                fs = float(tok[3:])
                    elif line.strip() and not line.startswith('index'):
                        rows.append(float(line.split(',')[1]))
            if fs is None:
                raise ValueError('%s has no "# fs=..." header' % path)
            x = np.array(rows)
        a = int(start_s * fs)
        b = len(x) if duration_s is None else min(len(x), a + int(duration_s * fs))
        x = x[a:b]
        self.offset_s = start_s
        if sim_fs and sim_fs != fs:
            from fractions import Fraction
            from scipy.signal import resample_poly
            fr = Fraction(sim_fs / fs).limit_denominator(1000)
            x, fs = resample_poly(x, fr.numerator, fr.denominator), float(sim_fs)
        self.x, self.fs, self.speed = x, fs, speed

    def chunks(self, block_s=0.04):
        n = max(1, int(round(block_s * self.fs)))
        t0 = time.time()
        for i in range(0, len(self.x), n):
            if self.speed > 0:
                wait = t0 + (i + n) / self.fs / self.speed - time.time()
                if wait > 0:
                    time.sleep(wait)
            yield self.x[i:i + n], None

    def reference_label(self, t, tol_s=0.15):
        """MIT-BIH annotation symbol of the beat at replay time t (s), if any."""
        if self.annotations is None:
            return None
        sym, pos, fs = self.annotations
        target = (t + self.offset_s) * fs
        k = int(np.searchsorted(pos, target))
        best = None
        for j in (k - 1, k):
            if 0 <= j < len(pos) and abs(pos[j] - target) <= tol_s * fs:
                if best is None or abs(pos[j] - target) < abs(pos[best] - target):
                    best = j
        return None if best is None else sym[best]
