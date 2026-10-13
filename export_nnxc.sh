#!/usr/bin/env bash
set -euo pipefail

# Wrapper around export_eqx_to_fortran.py
# Exports Fx/Fc Equinox models to SIESTA ASCII files and stages them in nn_params/.

usage() {
  cat <<'EOF'
Usage:
  ./export_nnxc.sh --fx-model <FX_PREFIX> --fc-model <FC_PREFIX> [options]

Required:
  --fx-model <FX_PREFIX>   Path prefix to Fx model (without .eqx/.json)
  --fc-model <FC_PREFIX>   Path prefix to Fc model (without .eqx/.json)

Options:
  --out-prefix <name>      Output prefix for exported files (default: xc_nn)
  --out-dir <dir>          Where to put nn_params (default: ./nn_params)
  --python <exe>           Python executable (default: python)
  --keep-temp              Keep exported xc_nn_*.dat files in current dir (default: move them)
  -h, --help               Show this help

Example:
  ./export_nnxc.sh \
    --fx-model models/unpolarized/pretrained_in_slim05/GGA_Fx_G_transf_lin_d3_n16_s42_v_20000 \
    --fc-model models/unpolarized/pretrained_in_slim05/GGA_Fc_G_transf_lin_d3_n16_s42_v_20000 \
    --out-prefix xc_nn

This will create:
  ./nn_params/xc_nn_fx_W*.dat, ./nn_params/xc_nn_fx_b*.dat
  ./nn_params/xc_nn_fc_W*.dat, ./nn_params/xc_nn_fc_b*.dat
EOF
}

FX_MODEL=""
FC_MODEL=""
OUT_PREFIX="xc_nn"
OUT_DIR="./nn_params"
PYTHON_BIN="python"
KEEP_TEMP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fx-model)    FX_MODEL="${2:-}"; shift 2 ;;
    --fc-model)    FC_MODEL="${2:-}"; shift 2 ;;
    --out-prefix)  OUT_PREFIX="${2:-}"; shift 2 ;;
    --out-dir)     OUT_DIR="${2:-}"; shift 2 ;;
    --python)      PYTHON_BIN="${2:-}"; shift 2 ;;
    --keep-temp)   KEEP_TEMP=1; shift 1 ;;
    -h|--help)     usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$FX_MODEL" || -z "$FC_MODEL" ]]; then
  echo "ERROR: --fx-model and --fc-model are required." >&2
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORTER="${SCRIPT_DIR}/export_eqx_to_fortran.py"

if [[ ! -f "$EXPORTER" ]]; then
  echo "ERROR: Cannot find exporter script: $EXPORTER" >&2
  exit 2
fi

# Run export in the current working directory (so output files land here)
echo "Exporting NNXC parameters..."
echo "  Fx: $FX_MODEL"
echo "  Fc: $FC_MODEL"
echo "  out-prefix: $OUT_PREFIX"
"$PYTHON_BIN" "$EXPORTER" \
  --fx-model "$FX_MODEL" \
  --fc-model "$FC_MODEL" \
  --out-prefix "$OUT_PREFIX"

# Collect outputs
mkdir -p "$OUT_DIR"

shopt -s nullglob
FILES=( "${OUT_PREFIX}"_fx_*.dat "${OUT_PREFIX}"_fc_*.dat )
shopt -u nullglob

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "ERROR: No exported .dat files found (expected ${OUT_PREFIX}_fx_*.dat and ${OUT_PREFIX}_fc_*.dat)." >&2
  exit 1
fi

echo "Staging ${#FILES[@]} files into: $OUT_DIR"
if [[ "$KEEP_TEMP" -eq 1 ]]; then
  cp -f "${FILES[@]}" "$OUT_DIR/"
else
  mv -f "${FILES[@]}" "$OUT_DIR/"
fi

echo "Done."
echo "SIESTA expects prefix: ${OUT_DIR}/${OUT_PREFIX}"
