#!/usr/bin/env bash
# Install the pinned OpenCourant (OpenRadioss fork) Linux build used by this oracle.
#
#   oracles/opencourant/install.sh [PREFIX]      (default PREFIX=/home/user/oracle-env)
#
# Downloads the exact release asset, verifies its sha256 and unpacks it to
# $PREFIX/opencourant/OpenCourant.  Idempotent: an existing install whose marker file
# records the same hash is left alone.  OpenCourant keeps only its three newest builds on
# the release page; if the tag has been pruned the download fails loudly (keep a copy of
# the zip in $PREFIX/opencourant-dl, which is reused when its hash matches).
set -euo pipefail

TAG="latest-20261006"
ASSET="OpenCourant_linux64.zip"
SHA256="9d67531de156dd9beba05fbfe710dcdc2bcecbdf3dc3a12642cf85dcece80081"
COMMIT="33e685176cccf0c539a3ce07aa2096985a284e2a"
URL="https://github.com/OpenCourant/OpenCourant/releases/download/${TAG}/${ASSET}"

PREFIX="${1:-/home/user/oracle-env}"
DL="${PREFIX}/opencourant-dl"
DEST="${PREFIX}/opencourant"
MARKER="${DEST}/.installed-${SHA256}"

if [[ -f "${MARKER}" && -x "${DEST}/OpenCourant/exec/engine_linux64_gf" ]]; then
  echo "OpenCourant ${TAG} already installed in ${DEST}/OpenCourant"
  exit 0
fi

mkdir -p "${DL}" "${DEST}"
ZIP="${DL}/${ASSET}"
if [[ -f "${ZIP}" ]] && echo "${SHA256}  ${ZIP}" | sha256sum -c --status; then
  echo "using cached ${ZIP}"
else
  rm -f "${ZIP}.part"
  curl -fL --retry 3 -o "${ZIP}.part" "${URL}"
  mv "${ZIP}.part" "${ZIP}"
fi
echo "${SHA256}  ${ZIP}" | sha256sum -c -

TMP="$(mktemp -d "${DEST}/unpack.XXXXXX")"
trap 'rm -rf "${TMP}"' EXIT
unzip -q "${ZIP}" -d "${TMP}"
if [[ ! -d "${TMP}/OpenCourant/exec" ]]; then
  echo "unexpected archive layout" >&2
  exit 1
fi
rm -rf "${DEST}/OpenCourant"
mv "${TMP}/OpenCourant" "${DEST}/OpenCourant"
chmod +x "${DEST}"/OpenCourant/exec/* || true
touch "${MARKER}"

# Sanity check: the starter reports the pinned commit.
export OPENCOURANT_PATH="${DEST}/OpenCourant"
export LD_LIBRARY_PATH="${OPENCOURANT_PATH}/extlib/hm_reader/linux64:${OPENCOURANT_PATH}/extlib/h3d/lib/linux64:${LD_LIBRARY_PATH:-}"
export RAD_CFG_PATH="${OPENCOURANT_PATH}/hm_cfg_files"
if "${OPENCOURANT_PATH}/exec/starter_linux64_gf" -version 2>/dev/null | grep -q "${COMMIT}"; then
  echo "OpenCourant ${TAG} (commit ${COMMIT}) installed in ${OPENCOURANT_PATH}"
else
  echo "warning: starter -version does not report commit ${COMMIT}" >&2
fi
