# Real-time MPFB beat classification on the Jetson Nano

Live N / S / V classification of single-lead ECG from a ProtoCentral MAX30003 breakout,
using the 98.75 % model (`results_default_180Hz/new_student_seed7.keras`). The model runs in plain NumPy on the Jetson. TensorFlow is not needed.

**Setting up a Jetson for the first time?** Follow [JETSON_NANO_SETUP.md](JETSON_NANO_SETUP.md):
a step-by-step guide from flashing the SD card to live ECG.

```
MAX30003 --SPI--> Jetson Nano: resample to 360 Hz -> Stage-1 filter (0.5-45 Hz + 50 Hz notch,
zero phase) -> z-score -> R-peak detector -> 130-sample beat + 5 RR features -> MPFB model -> N/S/V
```

## What is in this folder

| Path | Runs on | Purpose |
|---|---|---|
| `run_live.py` | Jetson | Main program. Takes samples from the sensor (SPI or USB bridge) or a replayed file and prints and logs one classified beat per heartbeat. |
| `ecg_rt/` | Jetson | The runtime package. `model.py` is the NumPy model, `dsp.py` the resampler, filter and z-score, `qrs.py` the R-peak detector, `beats.py` the RR features, `pipeline.py` ties them together, `max30003.py` is the SPI driver and `sources.py` holds the input sources. |
| `model/mpfb_180hz_seed7.npz` | Jetson | The exported model (25 kB): weights with BatchNorm folded in, DS1 RR statistics, and filter coefficients. |
| `validate_stream.py` | PC or Jetson | Replays MIT-BIH records through the live pipeline and scores the result against the annotations. |
| `export_model.py` | PC (TensorFlow) | Rebuilds the bundle from the `.keras` file and checks NumPy against Keras. |
| `firmware/max30003_bridge/` | Arduino / ESP32 | Only needed if the sensor sits on a microcontroller instead of the Jetson's header. |
| `validation/*.json` | | Saved results of the runs below. |

## Does the 98.75 % survive the move to real time?

Everything below is measured on the MIT-BIH DS2 records , replayed
through the live code. The live settings (filter lookahead, z-score window, R-peak
placement) were chosen on DS1 only.

| Run | Accuracy | Macro-F1 | S Se / +P | V Se / +P |
|---|---|---|---|---|
| Paper (offline, annotated R-peaks) | 98.75 | 94.39 | 82.4 / 94.7 | 95.4 / 95.9 |
| `reference`: NumPy model + live feature code | **98.75** (identical beats and labels) | 94.39 | 82.4 / 94.7 | 95.4 / 95.9 |
| `oracle`: streaming filter and z-score, sensor at 256 sps | 98.77 | 94.45 | 82.5 / 95.2 | 95.3 / 95.7 |
| **`detect`: full live pipeline, sensor at 256 sps** | **98.61** | **93.69** | 79.8 / 94.2 | 95.6 / 95.1 |
| `detect` at 128 / 512 sps | 98.60 / 98.61 | 93.69 / 93.70 | 80.1 / 79.8 (Se) | 95.6 / 95.6 (Se) |

* **R-peak detector on DS2:** Se 99.72 %, +P 99.81 %. 99.27 % of the N/S/V beats get a
  label; the rest are missed beats or fall in the 10 s warm-up.
* **Where the 0.14 points go:** mostly S sensitivity, and mostly record 232 (long pauses).
  The model itself does not care if the R-peak moves by ±6 samples, because global
  max-pooling absorbs the shift. What changes is the RR intervals, whenever a beat is
  missed or detected twice.
* **Latency:** the median is about 5 s from the heartbeat to its label (3 s filter
  lookahead + 0.8 s detector margin + waiting for the next beat). `--lookahead 2` gives
  the same accuracy on DS2 (98.61 %) with a 3.95 s median.
* **Speed:** on this PC the live pipeline processes ECG 110–150 times faster than real
  time. The Jetson's Cortex-A57 is slower, but the load is a fraction of one core. To
  measure it there, run `python3 validate_stream.py --mode detect --records 100 --workers 1`;
  it prints the real-time factor.

Reproduce:

```bash
python deploy/export_model.py                                    # PC, needs TensorFlow
python deploy/validate_stream.py --mode reference --check-against results_full_ds2/180Hz_seed7.npz
python deploy/validate_stream.py --mode detect --device-fs 256
```

## 1. Hardware

### Wiring: MAX30003 breakout to the Jetson Nano 40-pin header (SPI1, `/dev/spidev0.0`)

| Breakout pin | Jetson Nano pin |
|---|---|
| MOSI (SDI) | 19 (SPI1_MOSI) |
| MISO (SDO) | 21 (SPI1_MISO) |
| SCLK | 23 (SPI1_SCK) |
| CS | 24 (SPI1_CS0) |
| GND | 6 (or any GND) |
| VCC / 3V3 | 1 (3.3 V); see the note below |
| INT1 / INTB | not needed (the driver polls the FIFO) |

* **Logic level:** set the breakout's logic-level jumper to **3.3 V**. The Jetson's header
  pins are 3.3 V only, and 5 V on MISO can damage the Jetson.
* **Power input:** check your board revision's documentation before choosing the supply.
  ProtoCentral's guides describe on-board regulators, and I could not confirm whether the
  v3 board expects 3.3 V or 5 V on VCC. If its guide says 5 V, use pin 2 (5 V) for VCC
  and keep the logic jumper at 3.3 V.
* **Clock:** the v2/v3 boards carry their own 32.768 kHz crystal, so nothing needs to
  drive FCLK.
* **Wire length:** keep the wires short (under 15 cm). SPI runs at 2 MHz; lower
  `speed_hz` in `ecg_rt/sources.py` if reads fail.

### Electrodes: approximate lead II

The model was trained on MIT-BIH lead MLII, so place the electrodes to give an upright
lead-II-like signal:

* **negative input (ECGN, often labelled RA):** just below the right collarbone;
* **positive input (ECGP, often labelled LA):** left lower ribcage, at the "left leg" position;
* **reference (RL), if your cable has one:** lower right ribcage or abdomen.

Use fresh gel electrodes and sit still for the 10 s warm-up. After about 20 beats,
`run_live.py` checks the polarity. If the R-waves come out negative it tells you to
swap the two leads or pass `--invert`. Do not ignore that warning: inverted beats look
nothing like the training data.

### Alternative: sensor on an Arduino / ESP32 over USB

Flash `firmware/max30003_bridge/max30003_bridge.ino`. The wiring is listed in the
sketch: for an Uno, MISO D12, MOSI D11, SCLK D13 and CS D4. Plug the board into the
Jetson by USB and run with `--source serial`. Do not use ProtoCentral's
"01-ecg-openview" example for this: it reads one sample per fixed delay and ignores
the FIFO tags, so the timing of the samples is not reliable.

## 2. Jetson Nano setup (JetPack 4.6, Ubuntu 18.04, Python 3.6)

```bash
# 1. Enable SPI1 on the header, then reboot
sudo /opt/nvidia/jetson-io/jetson-io.py
#    Configure Jetson 40pin Header -> Configure header pins manually -> [*] spi1 (19,21,23,24,26)
#    -> Back -> Save pin changes -> Save and reboot to reconfigure pins

# 2. Load the spidev driver now and at every boot
sudo modprobe spidev
echo spidev | sudo tee /etc/modules-load.d/spidev.conf
ls /dev/spidev0.*            # expect /dev/spidev0.0 and /dev/spidev0.1

# 3. Optional: use SPI without sudo
sudo groupadd -f gpio && sudo usermod -aG gpio $USER
echo 'SUBSYSTEM=="spidev", GROUP="gpio", MODE="0660"' | sudo tee /etc/udev/rules.d/99-spidev.rules
sudo udevadm control --reload-rules && sudo udevadm trigger      # then log out and in

# 4. Python packages
sudo apt-get update && sudo apt-get install -y python3-pip python3-numpy python3-scipy
pip3 install --user spidev pyserial
```

If `jetson-io.py` closes immediately, that is a known JetPack 4.x problem: the tool
cannot find the board's device tree in `/boot/dtb`. NVIDIA's developer forum has the fix
(create `/boot/dtb` and copy the `tegra210-p3448-*.dtb` file for your board into it).

Copy this folder to the Jetson. Add a few MIT-BIH files if you want to replay without
the sensor:

```bash
scp -r deploy  user@jetson:~/mpfb
scp processed/233_processed.npz processed/232_processed.npz processed/100_processed.npz user@jetson:~/mpfb/
```

## 3. Running

```bash
cd ~/mpfb

# a) No hardware: replay an MIT-BIH record in real time at the sensor's rate.
#    Each line shows the cardiologist's label next to the prediction.
python3 run_live.py --source replay --file 233_processed.npz --sim-fs 256

# b) MAX30003 on the SPI header
python3 run_live.py --source spi --log beats.csv --save-raw session.csv
#    options: --rate 512|256|128   --gain 20|40|80|160   --invert   --quiet (only S/V beats)

# c) MAX30003 on an Arduino / ESP32
python3 run_live.py --source serial --port /dev/ttyACM0 --log beats.csv

# d) Re-analyse a recorded session
python3 run_live.py --source replay --file session.csv --speed 0
```

Example output:

```
[  62.41 s] beat    71  HR  76  N  (N 0.99 S 0.01 V 0.00)
[  63.02 s] beat    72  HR  98  V  (N 0.00 S 0.00 V 1.00)  <-- ventricular ectopic
--- last 60 s: 74 beats, median HR 77 bpm | N 72  S 0  V 2
```

`beats.csv` has one row per beat with these columns: beat, time, wall clock, label,
p(N), p(S), p(V), heart rate, pre-RR and post-RR intervals, and latency. A lead-off,
a FIFO overflow, or saturation (the chip's fast-recovery mode) restarts the pipeline
with a fresh 10 s warm-up.

To start at boot, use a systemd unit with
`ExecStart=/usr/bin/python3 /home/<user>/mpfb/run_live.py --source spi --log /home/<user>/beats.csv`
and `Restart=always`.

## 4. How the live pipeline matches the training preprocessing

| Training (offline) | Live (`ecg_rt`) | Measured difference |
|---|---|---|
| MIT-BIH at 360 Hz | MAX30003 at 256 sps, polyphase-resampled to 360 Hz | none in accuracy (128 / 256 / 512 sps all give 98.6 %) |
| `filtfilt` Butterworth 0.5–45 Hz + 50 Hz notch over the whole record | the same filter's zero-phase impulse response as an FIR, ±3 s | ≤ 2 % of the signal SD at the worst sample, 0.2 % RMS |
| z-score with the whole record's mean and SD | trailing 5-minute mean and SD (needs 10 s to start) | none (oracle run: 98.77 %) |
| R-peaks from the cardiologist annotations | Elgendi two-moving-average detector, peak = largest deflection | −0.14 points of accuracy |
| RR features: 80- and 400-beat means, RR context of 12 beats | identical code path | exact (reference run) |

## Limits to keep in mind

* **The 98.75 % is an MIT-BIH figure.** Your sensor, electrode placement and subjects
  differ from 1970s Holter tapes. Before trusting the numbers on real people, record
  sessions with `--save-raw`, have the S and V beats checked by a clinician, and score
  them.
* **Three classes only.** Fusion (F) and paced or unclassifiable (Q) beats are forced
  into N, S or V. Ventricular flutter or fibrillation has no class: in that rhythm the
  detector fires on flutter waves and the labels are meaningless (record 207 shows
  this).
* **S sensitivity is the weak point** (about 80 % live): many atrial premature beats are
  called N. V is reliable at about 95 %.
* **Mains notch:** the notch is fixed at 50 Hz, as in training. That suits Indian mains;
  do not change it, because the model was trained with it.
* **Not a medical device.** This is a research prototype and must not be used for
  diagnosis or alarms.
