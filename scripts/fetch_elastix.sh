#!/bin/sh
# Download elastix 5.2.0 for Linux x86-64 into core/bin and core/lib.
# https://github.com/SuperElastix/elastix/releases/tag/5.2.0
set -eu
cd "$(dirname "$0")/.."
url="https://github.com/SuperElastix/elastix/releases/download/5.2.0/elastix-5.2.0-linux.zip"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -L --fail -o "$tmp/elastix.zip" "$url"
unzip -q "$tmp/elastix.zip" -d "$tmp/src"
elastix="$(find "$tmp/src" -type f -name elastix | head -n 1)"
if [ -z "$elastix" ]; then
  echo "elastix executable not found in the archive" >&2
  exit 1
fi
root="$(dirname "$(dirname "$elastix")")"
mkdir -p core/bin core/lib
cp "$root/bin/elastix" "$root/bin/transformix" core/bin/
cp "$root"/lib/libANNlib-5.2.so* core/lib/
chmod +x core/bin/elastix core/bin/transformix
echo "installed $(core/bin/elastix --version 2>&1 | head -n 1)"
