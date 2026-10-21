// MAX30003 -> USB serial bridge for `run_live.py --source serial`.
//
// Use this only if the ProtoCentral MAX30003 breakout is NOT wired to the Jetson's
// SPI header but to an Arduino / ESP32 plugged into the Jetson over USB.
// (ProtoCentral's OpenView example is not suitable: it reads one sample per fixed
// delay and ignores the FIFO tags, so samples are lost or repeated.)
//
// Output, one line per ECG sample:   seq,counts,flags
//   seq    0..255, increments by one per line (the Jetson detects lost lines)
//   counts 18-bit signed ADC value (mV = counts * 1000 / (131072 * gain))
//   flags  bit0 fast-recovery (voltage invalid), bit1 leads off, bit2 FIFO overflow
// plus a header line "#MAX30003 fs=256 gain=20" at start and every 1024 samples.
//
// Wiring (Arduino Uno; set the breakout's logic jumper to 5 V):
//   MISO D12, MOSI D11, SCLK D13, CS D4, VCC 5V, GND GND
// ESP32 (logic jumper 3.3 V): MISO 19, MOSI 23, SCLK 18, CS 5 -> set PIN_CS to 5.
// Register values: see deploy/ecg_rt/max30003.py (same configuration).

#include <SPI.h>

const int PIN_CS = 4;
const uint32_t BAUD = 115200;
const uint8_t RATE_CODE = 1;     // CNFG_ECG RATE with FMSTR = 00: 0 = 512, 1 = 256, 2 = 128 sps
const uint16_t FS = 256;         // must match RATE_CODE
const uint8_t GAIN_CODE = 0;     // 0 = 20, 1 = 40, 2 = 80, 3 = 160 V/V
const uint16_t GAIN = 20;        // must match GAIN_CODE
const bool INVERT = false;       // true if the R-waves come out negative

SPISettings spiSettings(2000000, MSBFIRST, SPI_MODE0);
uint8_t seq = 0;
uint16_t sinceHeader = 0;

void regWrite(uint8_t reg, uint32_t v) {
  SPI.beginTransaction(spiSettings);
  digitalWrite(PIN_CS, LOW);
  SPI.transfer(reg << 1);
  SPI.transfer((v >> 16) & 0xFF);
  SPI.transfer((v >> 8) & 0xFF);
  SPI.transfer(v & 0xFF);
  digitalWrite(PIN_CS, HIGH);
  SPI.endTransaction();
}

uint32_t regRead(uint8_t reg) {
  SPI.beginTransaction(spiSettings);
  digitalWrite(PIN_CS, LOW);
  SPI.transfer((reg << 1) | 1);
  uint32_t v = (uint32_t)SPI.transfer(0) << 16;
  v |= (uint32_t)SPI.transfer(0) << 8;
  v |= SPI.transfer(0);
  digitalWrite(PIN_CS, HIGH);
  SPI.endTransaction();
  return v;
}

void printHeader() {
  Serial.print("#MAX30003 fs=");
  Serial.print(FS);
  Serial.print(" gain=");
  Serial.println(GAIN);
}

void emit(int32_t counts, uint8_t flags) {
  Serial.print(seq);
  Serial.print(',');
  Serial.print(counts);
  Serial.print(',');
  Serial.println(flags);
  seq++;
  sinceHeader++;
}

void setup() {
  Serial.begin(BAUD);
  pinMode(PIN_CS, OUTPUT);
  digitalWrite(PIN_CS, HIGH);
  SPI.begin();
  delay(100);
  regWrite(0x08, 0);                                       // SW_RST
  delay(100);
  while (((regRead(0x0F) >> 20) & 0xF) != 0x5) {            // INFO: MAX30003 present?
    Serial.println("#ERROR MAX30003 not found - check wiring and the logic-level jumper");
    delay(1000);
  }
  // CNFG_GEN: EN_ECG, DC lead-off 10 nA +/-300 mV, 100 MOhm resistive bias on both inputs
  regWrite(0x10, (1UL << 19) | (1UL << 12) | (2UL << 8) | (1UL << 4) | (1UL << 2) | 3UL);
  delay(100);
  regWrite(0x14, INVERT ? (1UL << 23) : 0UL);              // CNFG_EMUX: inputs connected
  // CNFG_ECG: rate, gain, DHPF off, DLPF ~100 Hz (40 Hz is used internally at 128 sps)
  regWrite(0x15, ((uint32_t)RATE_CODE << 22) | ((uint32_t)GAIN_CODE << 16) | (2UL << 12));
  delay(50);
  regWrite(0x09, 0);                                       // SYNCH: start, clear FIFO
  printHeader();
}

void loop() {
  uint32_t status = regRead(0x01);
  uint8_t base = (status & (1UL << 20)) ? 2 : 0;           // DCLOFFINT: leads off
  for (int i = 0; i < 40; i++) {
    uint32_t w = regRead(0x21);                            // ECG_FIFO
    uint8_t etag = (w >> 3) & 0x7;
    if (etag == 6) break;                                  // FIFO empty
    if (etag == 7) {                                       // overflow: reset FIFO, flag a gap
      regWrite(0x0A, 0);
      emit(0, base | 4);
      break;
    }
    int32_t v = (int32_t)(w >> 6);
    if (v & 0x20000) v -= 0x40000;                         // 18-bit two's complement
    emit(v, base | ((etag == 1 || etag == 3) ? 1 : 0));
    if (etag == 2 || etag == 3) break;                     // last sample currently available
  }
  if (sinceHeader >= 1024) {
    printHeader();
    sinceHeader = 0;
  }
}
