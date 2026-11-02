#pragma once
// Copy to Config.h (or run firmware/build/configure_arduino.py) and edit locally.
static constexpr char WIFI_SSID[] = "YOUR_WIFI_2_4_GHZ";
static constexpr char WIFI_PASSWORD[] = "YOUR_WIFI_PASSWORD";
// LAN IPv4 of any k3s node (or the host running the receiver). Not Tailscale.
static constexpr char RECEIVER_IP[] = "192.168.1.50";
static constexpr uint16_t UDP_PORT = 12345;
// Same value as OPEN_LEXA_TOKEN in the receiver Secret/.env. Never put API keys here.
static constexpr char RECEIVER_TOKEN[] = "00000000000000000000000000000000";
static constexpr unsigned SPEECH_RMS = 500;
static constexpr unsigned SILENCE_MS = 1000;
static constexpr unsigned MIC_GAIN_DB = 6;  // 0 or 6; start low to avoid clipping
static constexpr unsigned BEEP_VOLUME = 60; // ES8311 DAC percentage; tone amplitude is also reduced
