#!/usr/bin/env bash
# Run on haag2 as root: sudo bash deploy/haag2/install-lizard-class-mjs-mime.sh
# From repo root on the server, or pass SNIPPET path as first argument.
set -euo pipefail

if [[ "${EUID:-}" -ne 0 ]]; then
  echo "Run with: sudo bash $0" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNIPPET="${1:-$SCRIPT_DIR/../nginx/lizard-class-mjs-wasm-mime.conf}"
SITE="${NGINX_SITE:-/etc/nginx/sites-available/lizard-class}"
SNIPPET_DST="/etc/nginx/snippets/lizard-class-mjs-wasm-mime.conf"

if [[ ! -f "$SNIPPET" ]]; then
  echo "Snippet not found: $SNIPPET" >&2
  exit 1
fi
if [[ ! -f "$SITE" ]]; then
  echo "Nginx site not found: $SITE" >&2
  exit 1
fi

mkdir -p /etc/nginx/snippets
install -m 0644 "$SNIPPET" "$SNIPPET_DST"

if grep -qF 'lizard-class-mjs-wasm-mime.conf' "$SITE"; then
  echo "OK: include already present in $SITE"
else
  cp -a "$SITE" "${SITE}.bak.$(date +%Y%m%d%H%M%S)"
  awk '
    /^[[:space:]]*index index.html;/ && !done {
      print
      print "    include /etc/nginx/snippets/lizard-class-mjs-wasm-mime.conf;"
      done = 1
      next
    }
    { print }
  ' "$SITE" > "${SITE}.new.$$" && mv "${SITE}.new.$$" "$SITE"
  echo "OK: inserted include into $SITE"
fi

nginx -t
systemctl reload nginx
echo "OK: nginx reloaded. Check:"
echo "  curl -sI 'https://lizard-class.cc.gatech.edu/ort-wasm-simd-threaded.jsep.mjs' | grep -i content-type"
