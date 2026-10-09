#!/usr/bin/env bash
# Installer for Qwen3.8-Flash-Next (NVFP4, MTP, vision) on a single DGX Spark (GB10).
#
#   git clone https://github.com/Hitheshkaranth/Qwen3.8-Flash-Next_Model_DGX_Spark_Setup.git
#   cd Qwen3.8-Flash-Next_Model_DGX_Spark_Setup && ./install.sh
#
# This repo is a thin, tested profile on top of MiaAI-Lab's single-Spark recipe
# (AGPL-3.0). install.sh clones that recipe at the pinned commit into
# ./upstream, applies our patch and .env, downloads the ~99 GiB checkpoint and
# starts the server, then warms it up. Every step skips work already done, so
# it is safe to re-run.
#
# Env overrides: UPSTREAM_REF, API_KEY (vLLM upstream key; defaults to
# gateway/upstream.key if present), SKIP_START=1 (download only).
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
UPSTREAM_URL="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark.git"
UPSTREAM_REF="${UPSTREAM_REF:-b7c5da1a9e7448a14b480deee66eda3799b1eb43}"

echo "==> Checking prerequisites"
for c in git docker curl python3; do
  command -v "$c" >/dev/null || { echo "$c is required" >&2; exit 1; }
done
docker info >/dev/null 2>&1 || { echo "Docker daemon not reachable" >&2; exit 1; }
mem_gib=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
[ "$mem_gib" -ge 115 ] || echo "WARNING: ${mem_gib} GiB RAM; this profile is sized for a 121 GiB GB10." >&2
free_gib=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$free_gib" -ge 160 ] || echo "WARNING: ${free_gib} GiB free; weights + PLE cache + image need ~150 GiB." >&2

echo "==> Upstream recipe @ ${UPSTREAM_REF:0:7}"
if [ ! -d upstream/.git ]; then
  git clone -q "$UPSTREAM_URL" upstream
fi
git -C upstream fetch -q origin
git -C upstream checkout -q "$UPSTREAM_REF"
for p in patches/*.patch; do
  if git -C upstream apply --check "$HERE/$p" 2>/dev/null; then
    git -C upstream apply "$HERE/$p" && echo "    applied $p"
  else
    echo "    $p already applied (or not applicable), skipping"
  fi
done

echo "==> .env (config/flash-next.env)"
cp config/flash-next.env upstream/.env
key="${API_KEY:-}"
[ -z "$key" ] && [ -f gateway/upstream.key ] && key="$(cat gateway/upstream.key)"
if [ -n "$key" ]; then
  sed -i "s|^API_KEY=.*|API_KEY=\"$key\"|" upstream/.env
  echo "    API_KEY set (vLLM will reject requests without it; clients go through the gateway)"
else
  echo "    no API_KEY: vLLM will be open on 172.17.0.1:8006. Run gateway/setup.sh first to lock it."
fi

echo "==> Image + checkpoint (~20 GiB image, ~99 GiB weights; Xet disabled, it stalls on this box)"
(cd upstream && HF_HUB_DISABLE_XET=1 ./download.sh)

[ "${SKIP_START:-0}" = 1 ] && { echo "==> SKIP_START=1, done"; exit 0; }

echo "==> Starting (stop any other vLLM using >30 GiB first; this model needs most of the box)"
(cd upstream && ./start.sh)
API_KEY="$key" ./scripts/warmup.sh
echo "==> Serving qwen3.8-flash-next on http://172.17.0.1:8006 (behind the gateway: :8080)"
