"""
MAX30003 driver over Linux spidev (Jetson Nano 40-pin header, /dev/spidev0.0).

Register values follow the MAX30003 datasheet (Rev 1):
  SPI      32-bit frames: command byte (address << 1 | R/W, 1 = read) + 24 data bits,
           CPOL = 0 / CPHA = 0 (mode 0).
  INFO     0x0F, bits 23:20 read 0b0101 on a MAX30003 (used as a presence check).
  CNFG_GEN 0x10: FMSTR (21:20), EN_ECG (19), EN_DCLOFF (13:12), DCLOFF_IMAG (10:8),
           EN_RBIAS (5:4), RBIASV (3:2), RBIASP (1), RBIASN (0).
  CNFG_EMUX 0x14: POL (23), OPENP (21), OPENN (20) - the inputs are OPEN after reset
           and must be connected (written 0) or no ECG reaches the ADC.
  CNFG_ECG 0x15: RATE (23:22), GAIN (17:16), DHPF (14), DLPF (13:12).
  ECG_FIFO 0x21: 32-word FIFO, word = 18-bit two's complement ECG (23:6),
           ETAG (5:3): 000 valid, 001 fast-mode, 010 valid+EOF, 011 fast+EOF,
           110 FIFO empty, 111 FIFO overflow.
  STATUS   0x01: EOVF (22), FSTINT (21), DCLOFFINT (20), LDOFF_* (3:0).
ECG in mV = ADC * VREF / (2^17 * GAIN) with VREF = 1000 mV. The classifier z-scores
the signal, so the absolute scale only matters for display.
"""
from __future__ import division

import time

REG_STATUS, REG_MNGR_INT, REG_SW_RST, REG_SYNCH, REG_FIFO_RST = 0x01, 0x04, 0x08, 0x09, 0x0A
REG_INFO, REG_CNFG_GEN, REG_CNFG_EMUX, REG_CNFG_ECG, REG_ECG_FIFO = 0x0F, 0x10, 0x14, 0x15, 0x21

# sample rate -> (FMSTR, RATE) (datasheet Table 22)
RATES = {512: (0, 0), 256: (0, 1), 128: (0, 2), 500: (1, 0), 250: (1, 1), 125: (1, 2), 200: (2, 2)}
GAINS = {20: 0, 40: 1, 80: 2, 160: 3}

ETAG_VALID, ETAG_FAST, ETAG_VALID_EOF, ETAG_FAST_EOF = 0, 1, 2, 3
ETAG_EMPTY, ETAG_OVERFLOW = 6, 7


class MAX30003(object):
    def __init__(self, bus=0, device=0, speed_hz=2000000):
        import spidev                       # pip3 install spidev
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.mode = 0
        self.spi.max_speed_hz = speed_hz
        self.fs = None
        self.gain = 20

    def close(self):
        self.spi.close()

    def write(self, reg, value):
        self.spi.xfer2([(reg << 1) & 0xFE, (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF])

    def read(self, reg):
        r = self.spi.xfer2([((reg << 1) | 1) & 0xFF, 0, 0, 0])
        return (r[1] << 16) | (r[2] << 8) | r[3]

    def configure(self, rate=256, gain=20, dhpf=False, dlpf=None, invert=False, lead_off=True):
        """rate: 512/256/128 (or 500/250/125/200) sps; gain: 20/40/80/160 V/V.
        dhpf: on-chip 0.5 Hz high-pass (off: the pipeline applies the training filter).
        dlpf: on-chip low-pass code (default ~100 Hz where the rate allows, else 40 Hz).
        invert: swap the input polarity in hardware (if the R-waves come out negative)."""
        if rate not in RATES:
            raise ValueError('rate must be one of %s' % sorted(RATES))
        if gain not in GAINS:
            raise ValueError('gain must be one of %s' % sorted(GAINS))
        self.write(REG_SW_RST, 0)
        time.sleep(0.1)
        info = self.read(REG_INFO)
        if (info >> 20) & 0xF != 0x5:
            raise RuntimeError('MAX30003 not found on SPI (INFO register = 0x%06X). Check wiring, '
                               'the 3.3 V logic jumper and that SPI is enabled.' % info)
        fmstr, ratebits = RATES[rate]
        gen = (fmstr << 20) | (1 << 19)                      # master clock, EN_ECG
        if lead_off:
            gen |= (0b01 << 12) | (0b010 << 8)               # DC lead-off, 10 nA, +/-300 mV
        gen |= (0b01 << 4) | (0b01 << 2) | 0b11              # 100 MOhm resistive bias, both inputs
        self.write(REG_CNFG_GEN, gen)
        time.sleep(0.1)                                      # PLL lock
        self.write(REG_CNFG_EMUX, (1 << 23) if invert else 0)  # inputs connected, no calibration
        if dlpf is None:
            dlpf = 0b10 if rate in (512, 256, 500, 250) else 0b01
        self.write(REG_CNFG_ECG, (ratebits << 22) | (GAINS[gain] << 16) | (int(bool(dhpf)) << 14)
                   | (dlpf << 12))
        time.sleep(0.05)
        self.write(REG_SYNCH, 0)                             # start recording, clear FIFO
        self.fs, self.gain = float(rate), gain
        return {'info': info, 'cnfg_gen': gen, 'fs': self.fs}

    def status(self):
        s = self.read(REG_STATUS)
        return {'overflow': bool(s & (1 << 22)), 'fast': bool(s & (1 << 21)),
                'leads_off': bool(s & (1 << 20)), 'ldoff_bits': s & 0xF}

    def read_fifo(self, max_words=64):
        """Read every sample currently in the FIFO.
        Returns (values, invalid, overflow): values are raw 18-bit counts; invalid is
        True if any sample was taken in fast-recovery mode (voltage not valid)."""
        values, invalid, overflow = [], False, False
        for _ in range(max_words):
            w = self.read(REG_ECG_FIFO)
            etag = (w >> 3) & 0x7
            if etag == ETAG_EMPTY:
                break
            if etag == ETAG_OVERFLOW:
                overflow = True
                self.write(REG_FIFO_RST, 0)
                break
            v = w >> 6
            if v & 0x20000:
                v -= 0x40000
            if etag in (ETAG_FAST, ETAG_FAST_EOF):
                invalid = True
            values.append(v)
            if etag in (ETAG_VALID_EOF, ETAG_FAST_EOF):
                break
        return values, invalid, overflow

    def counts_to_mv(self, counts):
        return counts * 1000.0 / (131072.0 * self.gain)
