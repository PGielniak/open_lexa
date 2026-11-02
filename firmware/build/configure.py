#!/usr/bin/env python3
"""Create local configuration without overwriting existing files."""
import argparse
import json
from pathlib import Path
import secrets

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--receiver-ip', default='192.168.1.50',
                    help='LAN IPv4 of the receiver/k3s node (not a Tailscale address)')
args = parser.parse_args()
paths = [root / '.env', root / 'firmware/secrets.yaml']
if any(p.exists() for p in paths):
    parser.error('Local configuration already exists; edit .env and firmware/secrets.yaml directly.')
token = secrets.token_hex(16)
paths[0].write_text('STT_BACKEND=openai\nSTT_LANGUAGE=pl\nOPENAI_API_KEY=\n'
                    'OPENAI_STT_MODEL=gpt-transcribe\nOPEN_LEXA_TOKEN=' + token + '\n')
paths[1].write_text('wifi_ssid: "YOUR_WIFI_2_4_GHZ"\nwifi_password: "YOUR_WIFI_PASSWORD"\n'
                    + 'receiver_ip: ' + json.dumps(args.receiver_ip) + '\nreceiver_token: "' + token + '"\n')
for p in paths:
    p.chmod(0o600)
print('Created .env and firmware/secrets.yaml with a matching random token. '
      'Set Wi-Fi, receiver IP and API key/backend.')
