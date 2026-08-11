#!/usr/bin/env bash
# Measure each host we depend on, plus authoritative alternatives, for throttling.
# Reports throughput, TTFB and the edge IP actually served, 2 runs each.
set -u

hdr() { printf '\n===== %s =====\n' "$1"; }

# Resolve a python-build-standalone asset (the redistributable CPython uv itself uses,
# hosted on GitHub release infrastructure rather than python.org).
PBS_URL=$(curl -sS "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest" \
  | grep -o '"browser_download_url": *"[^"]*cpython-3\.13\.[0-9]*+[0-9]*-x86_64-pc-windows-msvc-install_only\.tar\.gz"' \
  | head -1 | sed 's/.*": *"//; s/"$//')
[ -z "$PBS_URL" ] && PBS_URL="SKIP"

declare -a NAMES=(
  "python.org-Windows-installer(install.ps1:2220,NO-FALLBACK)"
  "python-build-standalone-GitHub(alt-for-python.org)"
  "aka.ms-vc_redist(setup.ps1:1500)"
  "releases.astral.sh-uv(install.ps1:2541-primary)"
  "github.com-uv(install.ps1:2542-fallback)"
  "github-release-UnslothDesktop.exe"
)
declare -a URLS=(
  "https://www.python.org/ftp/python/3.13.13/python-3.13.13-amd64.exe"
  "$PBS_URL"
  "https://aka.ms/vs/17/release/vc_redist.x64.exe"
  "https://releases.astral.sh/github/uv/releases/download/0.12.1/uv-x86_64-pc-windows-msvc.zip"
  "https://github.com/astral-sh/uv/releases/download/0.12.1/uv-x86_64-pc-windows-msvc.zip"
  "https://github.com/unslothai/unsloth/releases/download/v0.1.70-beta/Unsloth-Desktop-0_1_70_beta-Windows.exe"
)

hdr "runner: $(uname -s) $(uname -m)"

for i in "${!URLS[@]}"; do
  name="${NAMES[$i]}"; url="${URLS[$i]}"
  if [ "$url" = "SKIP" ]; then
    printf 'MIRROR\t%s\tUNRESOLVED\n' "$name"
    continue
  fi
  for run in 1 2; do
    out=$(curl -sS -L --max-time 600 -o /dev/null \
      -w '%{http_code} %{time_total} %{speed_download} %{time_connect} %{time_starttransfer} %{size_download} %{remote_ip}' \
      "$url" 2>&1) || { printf 'MIRROR\t%s\trun%s\tERROR\t%s\n' "$name" "$run" "$out"; continue; }
    set -- $out
    mbps=$(awk -v s="$3" 'BEGIN{printf "%.2f", s/1048576}')
    mb=$(awk -v s="$6" 'BEGIN{printf "%.1f", s/1048576}')
    printf 'MIRROR\t%s\trun%s\thttp=%s\t%ss\t%s MB/s\t%s MB\tttfb=%s\tip=%s\n' \
      "$name" "$run" "$1" "$2" "$mbps" "$mb" "$5" "$7"
  done
done
