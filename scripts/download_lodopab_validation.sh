#!/usr/bin/env bash
# Fetch the LoDoPaB-CT validation ground-truth HDF5 chunks from Zenodo.
#
#   Leuschner J., Schmidt M. et al. 2021. "LoDoPaB-CT, a benchmark dataset for
#   low-dose computed tomography reconstruction." Scientific Data 8, 109.
#   https://doi.org/10.1038/s41597-021-00893-z
#   Hosted at https://zenodo.org/records/3384092
#
# Zenodo packages the validation ground truth as a single ~1.5 GB ZIP
# (`ground_truth_validation.zip`) containing multiple HDF5 chunks of 128
# slices each. Our environment maps a flat slice index to (chunk, offset).
#
# Training (~46 GB) and test (~4.5 GB) partitions are NOT fetched.
#
# Idempotent: skips download+extract when chunks already on disk.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/data/lodopab_ct"
ZIP_FILE="${TARGET_DIR}/ground_truth_validation.zip"
ZENODO_URL="https://zenodo.org/api/records/3384092/files/ground_truth_validation.zip/content"
SENTINEL_HDF5="${TARGET_DIR}/ground_truth_validation_000.hdf5"

mkdir -p "${TARGET_DIR}"

if [ -f "${SENTINEL_HDF5}" ]; then
    echo "LoDoPaB-CT validation chunks already extracted under ${TARGET_DIR}. Skipping."
    exit 0
fi

if [ ! -f "${ZIP_FILE}" ]; then
    echo "Downloading LoDoPaB-CT validation ZIP (~1.5 GB) from Zenodo..."
    echo "This can take a few minutes depending on your connection."
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --progress-bar -o "${ZIP_FILE}" "${ZENODO_URL}"
    elif command -v wget >/dev/null 2>&1; then
        wget --progress=bar -O "${ZIP_FILE}" "${ZENODO_URL}"
    else
        echo "ERROR: need either curl or wget on PATH." >&2
        exit 1
    fi
else
    echo "ZIP already present at ${ZIP_FILE}; skipping download."
fi

echo "Extracting HDF5 chunks into ${TARGET_DIR}..."
if command -v unzip >/dev/null 2>&1; then
    unzip -q -o "${ZIP_FILE}" -d "${TARGET_DIR}"
else
    echo "ERROR: unzip not found on PATH." >&2
    exit 1
fi

# If the ZIP extracted into a nested directory (e.g. `ground_truth_validation/`),
# hoist the HDF5 chunks up one level so the env loader finds them.
nested="$(find "${TARGET_DIR}" -mindepth 2 -name "ground_truth_validation_*.hdf5" -print -quit || true)"
if [ -n "${nested}" ]; then
    nest_dir="$(dirname "${nested}")"
    echo "Hoisting chunks out of ${nest_dir}..."
    mv "${nest_dir}"/ground_truth_validation_*.hdf5 "${TARGET_DIR}/" 2>/dev/null || true
    rmdir "${nest_dir}" 2>/dev/null || true
fi

echo "Removing ZIP to reclaim disk..."
rm -f "${ZIP_FILE}"

echo "Done. Chunk layout:"
ls -lh "${TARGET_DIR}" | head -12

echo
echo "Next: set use_real_data=True in LodopabCtEnv / load_environment()."
