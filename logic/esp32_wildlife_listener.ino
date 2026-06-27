#include <SPI.h>
#include <LoRa.h>

// =============================================================================
// ESP32 Wildlife Deterrent Node
// =============================================================================
//
// Commands received from Pi over LoRa:
//
//   ACTIVATE_SIREN    → siren ON  (holds until DEACTIVATE_SIREN)
//   DEACTIVATE_SIREN  → siren OFF
//   ACTIVATE_LIGHTS   → lights ON (holds until DEACTIVATE_LIGHTS)
//   DEACTIVATE_LIGHTS → lights OFF
//   ACTIVATE_BOTH     → siren + lights ON together
//   DEACTIVATE_BOTH   → siren + lights OFF together
//   HEARTBEAT         → keep-alive, no relay action
//
// Siren holds ON until explicit DEACTIVATE or 90s watchdog fires.
// Lights hold ON until explicit DEACTIVATE or 90s watchdog fires.
// Pi sends DEACTIVATE when animal leaves (MISS_THRESHOLD reached).
// =============================================================================

// --- Hardware Pins ---
const int sck  = 18;
const int miso = 19;
const int mosi = 23;
const int ss   = 5;
const int rst  = 14;
const int dio0 = 2;

const int RELAY_SIREN  = 26;
const int RELAY_LIGHTS = 25;

// Active-LOW relay: LOW = ON, HIGH = OFF
const int RELAY_ON  = LOW;
const int RELAY_OFF = HIGH;

// Watchdog: force everything OFF if Pi silent for 90s
const unsigned long HEARTBEAT_TIMEOUT = 90000UL;

unsigned long lastPacketTime = 0;
bool          linkHealthy    = true;

// =============================================================================
// SETUP
// =============================================================================

void setup() {
  Serial.begin(115200);

  // BOOT FIX: pre-load latch HIGH before switching to output.
  // Prevents relays firing during ESP32 boot sequence (GPIO floats LOW).
  digitalWrite(RELAY_SIREN,  RELAY_OFF);
  digitalWrite(RELAY_LIGHTS, RELAY_OFF);
  pinMode(RELAY_SIREN,  OUTPUT);
  pinMode(RELAY_LIGHTS, OUTPUT);
  digitalWrite(RELAY_SIREN,  RELAY_OFF);   // belt-and-suspenders
  digitalWrite(RELAY_LIGHTS, RELAY_OFF);

  SPI.begin(sck, miso, mosi, ss);
  LoRa.setPins(ss, rst, dio0);

  if (!LoRa.begin(433E6)) {
    Serial.println("[ERROR] LoRa init failed! Check wiring.");
    while (1);
  }

  LoRa.setSyncWord(0x12);          // must match Pi register 0x39
  LoRa.setSpreadingFactor(7);      // must match Pi SF7
  LoRa.setSignalBandwidth(125E3);
  LoRa.enableCrc();

  lastPacketTime = millis();

  Serial.println("=========================================");
  Serial.println(" ESP32 Wildlife Deterrent Node — ONLINE");
  Serial.println(" Commands: ACTIVATE/DEACTIVATE _SIREN");
  Serial.println("           ACTIVATE/DEACTIVATE _LIGHTS");
  Serial.println("           ACTIVATE/DEACTIVATE _BOTH");
  Serial.println("=========================================");
  Serial.print("SIREN  pin 26: ");
  Serial.println(digitalRead(RELAY_SIREN)  == RELAY_OFF ? "OFF (OK)" : "ON (PROBLEM!)");
  Serial.print("LIGHTS pin 25: ");
  Serial.println(digitalRead(RELAY_LIGHTS) == RELAY_OFF ? "OFF (OK)" : "ON (PROBLEM!)");
  Serial.println("Waiting for commands...");
}

// =============================================================================
// LOOP
// =============================================================================

void loop() {
  unsigned long now = millis();

  // --- 1. Receive LoRa packet ---
  int packetSize = LoRa.parsePacket();

  if (packetSize) {
    String cmd = "";
    while (LoRa.available()) cmd += (char)LoRa.read();

    int rssi = LoRa.packetRssi();
    Serial.print("[RX] \""); Serial.print(cmd);
    Serial.print("\"  RSSI: "); Serial.print(rssi); Serial.println(" dBm");

    lastPacketTime = now;
    if (!linkHealthy) {
      linkHealthy = true;
      Serial.println("[LINK] Pi link restored.");
    }

    // ── Siren ──────────────────────────────────────────────────────────────
    if (cmd == "ACTIVATE_SIREN") {
      digitalWrite(RELAY_SIREN, RELAY_ON);
      Serial.println(">>> SIREN ON");
    }
    else if (cmd == "DEACTIVATE_SIREN") {
      digitalWrite(RELAY_SIREN, RELAY_OFF);
      Serial.println("<<< SIREN OFF");
    }

    // ── Lights ─────────────────────────────────────────────────────────────
    else if (cmd == "ACTIVATE_LIGHTS") {
      digitalWrite(RELAY_LIGHTS, RELAY_ON);
      Serial.println(">>> LIGHTS ON");
    }
    else if (cmd == "DEACTIVATE_LIGHTS") {
      digitalWrite(RELAY_LIGHTS, RELAY_OFF);
      Serial.println("<<< LIGHTS OFF");
    }

    // ── Both ───────────────────────────────────────────────────────────────
    else if (cmd == "ACTIVATE_BOTH") {
      digitalWrite(RELAY_SIREN,  RELAY_ON);
      digitalWrite(RELAY_LIGHTS, RELAY_ON);
      Serial.println(">>> SIREN + LIGHTS ON");
    }
    else if (cmd == "DEACTIVATE_BOTH") {
      digitalWrite(RELAY_SIREN,  RELAY_OFF);
      digitalWrite(RELAY_LIGHTS, RELAY_OFF);
      Serial.println("<<< SIREN + LIGHTS OFF");
    }

    // ── Heartbeat ──────────────────────────────────────────────────────────
    else if (cmd == "HEARTBEAT") {
      Serial.println("[HB] Pi alive.");
    }

    else {
      Serial.print("[WARN] Unknown command: "); Serial.println(cmd);
    }
  }

  // --- 2. Watchdog: force everything OFF if Pi silent for 90s ---
  if (linkHealthy && (now - lastPacketTime) > HEARTBEAT_TIMEOUT) {
    linkHealthy = false;
    bool sirenWasOn  = digitalRead(RELAY_SIREN)  == RELAY_ON;
    bool lightsWasOn = digitalRead(RELAY_LIGHTS) == RELAY_ON;
    digitalWrite(RELAY_SIREN,  RELAY_OFF);
    digitalWrite(RELAY_LIGHTS, RELAY_OFF);
    if (sirenWasOn)  Serial.println("[WATCHDOG] SIREN  forced OFF — Pi silent >90s");
    if (lightsWasOn) Serial.println("[WATCHDOG] LIGHTS forced OFF — Pi silent >90s");
    Serial.println("[WATCHDOG] Pi link lost — all deterrents cleared.");
  }
}
