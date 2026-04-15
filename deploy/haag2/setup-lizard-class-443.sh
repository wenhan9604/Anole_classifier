#!/usr/bin/env bash
# Run on haag2 as root: sudo bash setup-lizard-class-443.sh
# Builds TLS fullchain from GT-provided .cer files and reloads nginx.
set -euo pipefail

LEAF=/etc/ssl/certs/lizard-class.cc.gatech.edu.cer
CHAIN=/etc/ssl/certs/lizard-class.cc.gatech.edu.chain.cer
FULLCHAIN=/etc/ssl/certs/lizard-class.cc.gatech.edu.fullchain.pem
KEY=/etc/ssl/private/lizard-class.cc.gatech.edu.key
SITE=/etc/nginx/sites-available/lizard-class

if [[ "${EUID:-}" -ne 0 ]]; then
  echo "Run with: sudo bash $0" >&2
  exit 1
fi

for f in "$LEAF" "$CHAIN" "$KEY" "$SITE"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required file: $f" >&2
    exit 1
  fi
done

install -m 0644 /dev/null "$FULLCHAIN.tmp"
cat "$LEAF" "$CHAIN" >"$FULLCHAIN.tmp"
mv "$FULLCHAIN.tmp" "$FULLCHAIN"

if ! grep -q 'ssl_certificate /etc/ssl/certs/lizard-class.cc.gatech.edu.fullchain.pem;' "$SITE"; then
  cp -a "$SITE" "${SITE}.bak.$(date +%Y%m%d%H%M%S)"
  sed -E -i \
    's|ssl_certificate[[:space:]]+/etc/ssl/certs/lizard-class\.cc\.gatech\.edu[^;]*;|ssl_certificate /etc/ssl/certs/lizard-class.cc.gatech.edu.fullchain.pem;|' \
    "$SITE"
fi

nginx -t
systemctl reload nginx

echo "OK: nginx reloaded. Check: ss -tlnp | grep 443 && curl -skI https://127.0.0.1/ -H 'Host: lizard-class.cc.gatech.edu' | head"
