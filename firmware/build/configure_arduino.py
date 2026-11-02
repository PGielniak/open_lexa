#!/usr/bin/env python3
"""Create Config.h from existing local configuration; never print credentials."""
import json
from pathlib import Path
import sys
from dotenv import dotenv_values
import yaml

root = Path(__file__).resolve().parents[2]
out = root / 'firmware/src/OpenLexaNano/Config.h'
if out.exists():
    sys.exit('Config.h already exists; edit it directly (not overwritten).')
values = yaml.safe_load((root / 'firmware/secrets.yaml').read_text()) if (root / 'firmware/secrets.yaml').exists() else {}
env = dotenv_values(root / '.env')
token = env.get('OPEN_LEXA_TOKEN') or values.get('receiver_token')
if not token or len(token) != 32:
    sys.exit('First run firmware/build/configure.py, or set OPEN_LEXA_TOKEN in .env.')
bytes.fromhex(token)
receiver_ip = values.get('receiver_ip') or '192.168.1.50'
source = (out.parent / 'Config.example.h').read_text()
for old, new in [('YOUR_WIFI_2_4_GHZ', values.get('wifi_ssid', 'YOUR_WIFI_2_4_GHZ')),
                 ('YOUR_WIFI_PASSWORD', values.get('wifi_password', 'YOUR_WIFI_PASSWORD')),
                 ('192.168.1.50', receiver_ip),
                 ('00000000000000000000000000000000', token)]:
    source = source.replace(json.dumps(old), json.dumps(str(new), ensure_ascii=False))
with out.open('x') as f:
    f.write(source)
out.chmod(0o600)
print('Created firmware/src/OpenLexaNano/Config.h. Check Wi-Fi and receiver IP before flashing.')
