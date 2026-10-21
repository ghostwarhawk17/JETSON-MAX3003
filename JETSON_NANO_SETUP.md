# Deploying the MPFB ECG classifier on a Jetson Nano: step by step

This guide takes a brand-new Jetson Nano to live N / S / V heartbeat classification.
It assumes a **Windows PC** and no earlier Jetson experience.

| Part | Steps | What you get | Sensor needed? |
|---|---|---|---|
| **A. Deploy and test the model** | 1–6 | The model running on the Jetson, checked against MIT-BIH recordings | No |
| **B. Live ECG** | 7–10 | Real-time classification from the MAX30003 | Yes |

Plan for about 1.5–2 hours for Part A (most of it is downloading and flashing) and
about 1 hour for Part B.

> **Already have a Jetson that boots into Ubuntu and that you can reach over SSH?**
> Skip to [Step 4](#step-4-install-the-python-packages-on-the-jetson-10-min).

---

## Step 0: What you need

**For Part A**

- [ ] **Jetson Nano Developer Kit** (4 GB, revision A02 or B01). The 2 GB kit needs its
      own SD-card image and is powered through USB-C, but otherwise works the same.
- [ ] **microSD card**, at least 32 GB and UHS-1 (64 GB is better), plus a card reader for your PC.
- [ ] **Power supply:** a **5 V ⎓ 4 A barrel-jack supply** (5.5 mm outer / 2.1 mm inner,
      centre positive) **and a jumper cap** for header J48. This is what lets you set up
      the Jetson without a monitor. A 5 V ⎓ 2 A micro-USB supply also works, but only for
      the monitor route (Option A in Step 2). **Do not power the Jetson from your laptop.**
      A laptop USB port (USB-C included) usually gives 0.5–1.5 A, and the Nano needs up
      to 2 A or more. On too little current it fails to boot or reboots at random. The
      cable to the laptop carries data only.
- [ ] **Micro-USB cable that carries data.** Many phone-charger cables are power-only
      and will not work.
- [ ] **Ethernet cable** from the Jetson to your router. The Nano has no Wi-Fi, and it
      needs internet once, in Step 4.
- [ ] Optional: an HDMI or DisplayPort monitor, USB keyboard and USB mouse.

**Software for your PC (all free)**

- [ ] [SD Memory Card Formatter](https://www.sdcard.org/downloads/formatter/) (SD Association)
- [ ] [balenaEtcher](https://etcher.balena.io/)
- [ ] [PuTTY](https://www.putty.org/), needed only for setup without a monitor
- [ ] `ssh` and `scp` come with Windows 10/11. Check by typing `ssh` in PowerShell.

**For Part B (later)**

- [ ] ProtoCentral MAX30003 breakout with its electrode cable, and gel ECG electrodes
- [ ] 6 female-to-female jumper wires, about 10–15 cm long

---

## Step 1: Write Ubuntu (JetPack) onto the microSD card *(on your PC, ~30 min)*

The Jetson has no operating system of its own; it boots from this card.

1. Download **"Jetson Nano Developer Kit SD Card Image"** (JetPack 4.6.x) from NVIDIA's
   [Getting Started page](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit).
   The file is a large `.zip` (the current one is named `jetson-nano-jp461-sd-card-image.zip`).
   Do not unzip it.
2. Put the card in the PC. Open **SD Card Formatter**, select the card, choose
   **Quick format**, leave the volume label empty, and click **Format**.
3. Open **balenaEtcher** and choose **Flash from file** (the `.zip`), then
   **Select target** (your SD card; check the drive letter and size) and **Flash!**.
   Writing and verifying takes about 10–15 minutes.
4. When it finishes, Windows may say the drive needs formatting or show several new
   drives. **Click Cancel** on every such prompt; the card is fine. Eject the card.

**Check:** Etcher says *Flash Complete!*

---

## Step 2: First boot and Ubuntu setup *(~20 min)*

Insert the microSD card into the slot on the **underside of the Jetson module** (the
small board under the big heatsink), contacts facing up toward the module. It only fits
one way; do not force it.

Pick **one** of the two options.

### Option A: with a monitor, keyboard and mouse (simplest)

1. Connect the monitor (HDMI or DP), keyboard, mouse and the **Ethernet cable**.
2. Plug in power: the barrel jack with the **J48 jumper fitted**, or a 5 V 2 A micro-USB supply.
   The green LED next to the micro-USB port lights up and Ubuntu's setup appears.
3. Go through the setup screens (see the list below), then log in at the desktop.
4. Open a terminal (Ctrl+Alt+T) and run `hostname -I`. Note the first IP address
   (for example `192.168.1.42`); you will use it in Step 3.
5. If you powered it from the **barrel jack**, also connect the **micro-USB cable** from
   the Jetson to your PC. It gives you a second way in (Step 3). If you powered it through
   micro-USB, that port is taken, so use the Ethernet IP.

### Option B: headless, with the PC as the screen (no monitor)

1. Put the **jumper cap on J48**, the two pins between the barrel jack and the camera
   connector. This selects barrel-jack power.
2. Connect the **Ethernet cable** to your router.
3. Connect the **micro-USB cable** from the Jetson to your PC.
4. Plug the **5 V 4 A supply into the barrel jack** and wait about 1 minute.
5. On the PC, open **Device Manager**, then **Ports (COM & LPT)**. Note the
   **USB Serial Device (COMx)** entry, for example COM5.
6. Open **PuTTY**: Connection type **Serial**, Serial line **COM5** (yours), Speed
   **115200**, then **Open**. If the window stays blank, press **Space**.
7. Go through the setup screens (list below) using the arrow keys, Tab and Enter.
   The Jetson reboots at the end.

### The setup screens (both options)

- Accept the NVIDIA license.
- Language, keyboard layout and time zone.
- **Your name, computer name, username and password.** Write the username and
  password down; you need them for every SSH login. In this guide, whenever you see
  `yourname`, replace it with this username.
- **APP partition size:** accept the suggested maximum.
- **Power mode (Nvpmodel)**, if asked: choose **MAXN** with the barrel-jack supply, or
  **5W** with a micro-USB supply.

**Check:** after the reboot the Jetson is at a login screen (Option A), or PuTTY shows a
`login:` prompt (Option B; press Enter if it is blank).

---

## Step 3: Connect to the Jetson from your PC (SSH) *(~5 min)*

SSH lets you type commands on your PC that actually run on the Jetson, as if you were
sitting in front of it. From here on, everything is typed in **PowerShell on your PC**;
no monitor, keyboard or mouse needs to be plugged into the Jetson.

### How the connection works

Both your PC and the Jetson are on the same network through your router:

```
  ┌──────────┐                    ┌──────────┐                   ┌──────────┐
  │  Your PC │ ◄── Wi-Fi/cable ──►│  Router  │ ◄── Eth cable ──►│  Jetson  │
  └──────────┘                    └──────────┘                   └──────────┘
```

Your router gave the Jetson an IP address (like `192.168.0.113`). You need to know
this address to connect. Here is how to find it:

- **If you used Option A (monitor):** you already noted it from `hostname -I` in Step 2.
- **If you used Option B (headless):** in the PuTTY serial session, type `hostname -I`.
  The first address it prints is the one you need.

### Connect

Open **PowerShell** on your PC and type:

```powershell
ssh yourname@192.168.0.113
```

Replace **both** parts:
- `yourname` → the username you chose during the Ubuntu setup in Step 2
- `192.168.0.113` → the IP address you found above

For example, if your username is `ravi` and the IP is `192.168.0.113`:
`ssh ravi@192.168.0.113`

### What happens when you connect

The first time, you will see something like:

```
The authenticity of host '192.168.0.113' can't be established.
Are you sure you want to continue connecting (yes/no)?
```

Type **yes** and press Enter. Then type your password. **Nothing appears on screen while
you type the password** — this is normal; just type it and press Enter.

If it worked, the prompt changes to something like:

```
yourname@jetson-nano:~$
```

This means you are now **on the Jetson**. Every command you type runs on the Jetson, not
on your PC.

### Verify the Jetson has internet

The Jetson gets internet through the same Ethernet cable. Check:

```bash
ping -c 3 google.com
```

**Check:** three replies from `google.com`.

> **If it does not connect:** wait one more minute after powering on (the Jetson is slow
> to boot). If the IP stops working after a router restart, find the new one with
> `hostname -I` from the PuTTY serial session or by checking your router's admin page.

---

## Step 4: Install the Python packages *(on the Jetson, ~10 min)*

The model runs in plain NumPy, so TensorFlow is not needed. In your SSH session:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-numpy python3-scipy
python3 --version
python3 -c "import numpy, scipy; print('numpy', numpy.__version__, '| scipy', scipy.__version__)"
```

**Check:** you see Python 3.6.9, then something like `numpy 1.13.3 | scipy 0.19.1`.
Any version at least that new is fine.

---

## Step 5: Copy the model and code to the Jetson *(~5 min)*

1. **On the Jetson** (SSH session), make the folders:
   ```bash
   mkdir -p ~/mpfb/processed
   ```
2. **On your PC**, open a **second** PowerShell window (not the SSH one) and run the
   commands below. Replace `yourname@192.168.0.113` with your Jetson username and IP
   (the same ones you used for SSH in Step 3).
   ```powershell
   cd C:\MTP_revised-20260911T082811Z-1-001\MTP_revised
   scp -r deploy\run_live.py deploy\validate_stream.py deploy\ecg_rt deploy\model 
   
   yourname@192.168.0.113:mpfb/
   scp processed\100_processed.npz processed\232_processed.npz processed\233_processed.npz yourname@192.168.0.113:mpfb/processed/
   ```
   Replace `yourname@192.168.0.113` with your Jetson username and IP (the same ones you
   used for SSH in Step 3). `mpfb/` means the `mpfb` folder in your home directory on
   the Jetson.
3. **Back on the Jetson**, check what arrived:
   ```bash
   ls ~/mpfb ~/mpfb/ecg_rt ~/mpfb/model ~/mpfb/processed
   ```

**Check:** the listing shows
```
~/mpfb:           ecg_rt  model  processed  run_live.py  validate_stream.py
~/mpfb/ecg_rt:    __init__.py beats.py dsp.py max30003.py model.py pipeline.py qrs.py sources.py
~/mpfb/model:     mpfb_180hz_seed7.npz
~/mpfb/processed: 100_processed.npz  232_processed.npz  233_processed.npz
```

> Prefer a GUI? [WinSCP](https://winscp.net/) can drag and drop files over SSH
> (host = your Jetson's IP, your username and password).

---

## Step 6: Test the model on the Jetson (no sensor) *(~5 min)*

The program replays an MIT-BIH record exactly as if it were arriving from the sensor
at 256 samples/s, and compares each prediction with the cardiologist's label.

```bash
cd ~/mpfb
python3 run_live.py --source replay --file processed/233_processed.npz --sim-fs 256 --speed 0 --duration 300 --quiet
```

**Check:** the last lines read (the same numbers as on the development PC):
```
Total classified beats: 497  (N 359, S 0, V 138)
Agreement with the MIT-BIH annotations (N/S/V beats): 488 / 492 = 99.19 %
```
If you see this, **the model is deployed and working on the Jetson.** A difference of a
beat or two would still be fine (older NumPy and SciPy on ARM can round differently).

Two more things worth running:

```bash
# Watch it work in real time, beat by beat, as it would with the sensor (Ctrl+C to stop)
python3 run_live.py --source replay --file processed/233_processed.npz --sim-fs 256

# How fast is the Jetson? Prints "processing N x faster than real time"
python3 validate_stream.py --mode detect --records 100 --processed processed --workers 1
```
Anything above about 5× faster than real time is plenty.

### Optional: reproduce the paper's 98.75 % on the Jetson

Copy all 21 DS2 test records (about 220 MB). From PowerShell on the PC:
```powershell
cd C:\MTP_revised-20260911T082811Z-1-001\MTP_revised
$ds2 = '100','103','105','111','113','117','121','123','200','202','210','212','213','214','219','221','228','231','232','233','234'
scp ($ds2 | ForEach-Object { "processed\$($_)_processed.npz" }) yourname@192.168.0.113:mpfb/processed/
```
Then on the Jetson:
```bash
cd ~/mpfb
python3 validate_stream.py --mode reference --processed processed --workers 4
python3 validate_stream.py --mode detect --device-fs 256 --processed processed --workers 4
```
**Check:** the first run prints `acc  98.75 | macro-F1  94.39`, which is the paper's
result. The second prints `acc  98.61`, the fully live pipeline.

**Part A is done.** Everything below is needed only for live ECG.

---

## Step 7: Enable the SPI port *(on the Jetson, ~10 min)*

The MAX30003 talks to the Jetson over SPI, which is switched off by default.

1. Run NVIDIA's pin-configuration tool. It needs a terminal at least 80×24, which a
   normal PowerShell window is.
   ```bash
   sudo /opt/nvidia/jetson-io/jetson-io.py
   ```
   Choose **Configure Jetson 40pin Header**, then **Configure header pins manually**.
   Select **spi1 (19,21,23,24,26)** (press Space so it shows `[*]`), then **Back**, then
   **Save pin changes**, then **Save and reboot to reconfigure pins**.
2. After the reboot (about 1 minute), SSH in again and load the SPI driver, now and at every boot:
   ```bash
   sudo modprobe spidev
   echo spidev | sudo tee /etc/modules-load.d/spidev.conf
   ls /dev/spidev*
   ```
3. Install the Python SPI library. Install it system-wide, because the program runs
   with `sudo` to access SPI:
   ```bash
   sudo -H pip3 install spidev
   ```

**Check:** `ls /dev/spidev*` shows `/dev/spidev0.0` and `/dev/spidev0.1`.

> **If `jetson-io.py` closes immediately without a menu**, this is a known JetPack 4.x
> problem: it cannot find the board's device tree. Copy it to where the tool looks
> (B01 board shown; use `a02` instead of `b00` for an A02 board), then run the tool again:
> ```bash
> ls /boot/*.dtb
> sudo mkdir -p /boot/dtb
> sudo cp /boot/tegra210-p3448-0000-p3449-0000-b00.dtb /boot/dtb/
> ```

---

## Step 8: Wire the MAX30003 to the Jetson *(Jetson OFF, ~15 min)*

1. Shut the Jetson down and **unplug its power** before touching the header:
   ```bash
   sudo shutdown -h now
   ```
2. On the MAX30003 breakout, set the **logic-level jumper to 3.3 V**. The Jetson's pins
   are 3.3 V only, and 5 V signals can damage it.
3. Connect the wires to the Jetson's 40-pin header (J41). Find **pin 1**, marked with a
   small "1" or triangle on the board's silkscreen, and count from there. Pin 2 is
   directly beside pin 1 in the other row; odd pins (1, 3, 5…) run along pin 1's row and
   even pins (2, 4, 6…) along the other. A printed pinout diagram for the "Jetson Nano
   J41 header" helps.

   | MAX30003 breakout | Jetson header pin |
   |---|---|
   | VCC / 3V3 | **1** (3.3 V)\* |
   | GND | **6** (GND) |
   | MOSI (SDI) | **19** |
   | MISO (SDO) | **21** |
   | SCLK | **23** |
   | CS | **24** |
   | INT1 / INTB | leave unconnected |

   \* Check your board's documentation for its supply. If the v3 board's guide asks for
   5 V on VCC, use **pin 2** (5 V) instead, and keep the logic jumper at 3.3 V.
4. **Check every wire once more against the table**, then plug the power back in. SSH
   in again after about 1 minute.

---

## Step 9: First live recording *(~10 min)*

> **Safety:** ECG electrodes connect the person electrically to the electronics. Run the
> PC end from a **laptop on battery** (unplugged), do not touch other mains-powered
> equipment while wired up, and do not use this on anyone with a pacemaker or other
> implanted device. This is a research prototype, not a medical device.

1. **Place the electrodes** to imitate lead II, which the model was trained on (fresh
   gel electrodes, clean dry skin):
   - **negative input (often labelled RA):** just below the right collarbone
   - **positive input (often labelled LA):** left lower ribcage
   - **reference (RL), if your cable has one:** lower right ribcage
2. Sit still and start the program:
   ```bash
   cd ~/mpfb
   sudo python3 run_live.py --source spi --log beats.csv --save-raw session.csv
   ```
3. **What you should see:**
   - right away: `MPFB beat classifier | source spi at 256 sps -> 360 Hz ...`
   - after about 15 s: one line per heartbeat, about 5 s behind real time, for example
     `[  11.43 s] beat     2  HR  72  N  (N 1.00 S 0.00 V 0.00)`
   - every 10 s, a summary: `--- last 60 s: ... median HR 72 bpm | N 70  S 0  V 0`
4. **Watch for these warnings:**
   - `!!! The R-waves are mostly NEGATIVE`: swap the RA and LA electrodes, or restart with
     `--invert`. **Do not ignore this**; inverted beats look nothing like the training data.
   - `*** LEADS OFF`: an electrode is not making contact. Press it down or replace it.
   - repeated `*** data gap / saturation`: movement or a loose cable. Sit still and check the cable.
5. Press **Ctrl+C** to stop. It prints the totals. Then, on the PC, copy the results back:
   ```powershell
   scp yourname@192.168.0.113:mpfb/beats.csv yourname@192.168.0.113:mpfb/session.csv .
   ```
   `beats.csv` has one row per heartbeat (time, label, probabilities, heart rate, RR
   intervals). `session.csv` is the raw signal; it can be replayed with
   `python3 run_live.py --source replay --file session.csv --speed 0`.

**Check:** a resting person gives a heart rate that matches their pulse, and almost all
beats are labelled N.

---

## Step 10 (optional): Start automatically when the Jetson boots

```bash
sudo tee /etc/systemd/system/mpfb.service > /dev/null <<'EOF'
[Unit]
Description=MPFB live ECG beat classification
After=multi-user.target

[Service]
WorkingDirectory=/home/<user>/mpfb
Environment=PYTHONUNBUFFERED=1
ExecStart=/bin/sh -c 'exec /usr/bin/python3 /home/<user>/mpfb/run_live.py --source spi --quiet --log /home/<user>/mpfb/beats_$(date +%%Y%%m%%d_%%H%%M%%S).csv'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now mpfb
journalctl -u mpfb -f          # live output (Ctrl+C leaves it running)
```

Replace `<user>` with your username (three places). Each start writes a new
`beats_<date>.csv`. To stop it: `sudo systemctl disable --now mpfb`.

---

## Troubleshooting

| What you see | Likely cause and fix |
|---|---|
| Cannot find the Jetson's IP address | Connect a monitor and run `hostname -I`, or check your router's admin page (usually `192.168.0.1` or `192.168.1.1` in a browser) for attached devices. |
| The Jetson switches off or reboots by itself | The power supply is too weak. Use the 5 V 4 A barrel-jack supply with J48 fitted. |
| `ssh: connect to host ... timed out` | The Jetson is still booting (wait 1–2 minutes), or the router gave it a new IP after a restart. Find the new IP from your router's admin page or by connecting a monitor and running `hostname -I`. |
| `No module named 'scipy'` | Step 4 was skipped or failed. Check internet (`ping google.com`), then repeat Step 4. |
| `No module named 'spidev'` (with sudo) | It was installed for your user only. Run `sudo -H pip3 install spidev`. |
| `No such file or directory: '/dev/spidev0.0'` | SPI is not enabled or the driver is not loaded. Redo Step 7. |
| `MAX30003 not found on SPI (INFO register = 0x000000 / 0xFFFFFF)` | Wiring: check MISO/MOSI are not swapped, CS is on pin 24, and the board has power and GND. Check the logic jumper is at 3.3 V. |
| `The R-waves are mostly NEGATIVE` | Swap the RA and LA electrodes, or add `--invert`. |
| Started, but no beats after 30 s | Check the electrodes and the "LEADS OFF" messages. Record with `--save-raw` and look at `session.csv`: values stuck at 0 mean no signal reaches the chip. |
| Many `data gap / saturation` messages | Movement, dry electrodes or a loose cable. Use fresh electrodes and sit still. |

For background (how the pipeline works, validation results, limitations), see [README.md](README.md).
