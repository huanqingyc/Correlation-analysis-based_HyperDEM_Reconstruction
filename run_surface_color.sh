#!/bin/bash
# Reproduce surface code and color code experiments from test.ipynb
#
# Surface correlation analysis includes:
#   - d=5, r=5,  shots=[50k,500k,5M],   max_order=4, eps=(1.6e-4, 3e-5), mode=given+inference
#   - d=5, r=15, shots=[7M],            max_order=4, eps=(1.6e-4, 3e-5), mode=inference
#   - d=5, r=25, shots=[9M],            max_order=4, eps=(1.6e-4, 3e-5), mode=inference
#
# Surface decoding uses:
#   - d=5, r=5, shots=[50k,500k,5M]
#
# Color correlation analysis and decoding use:
#   - d=5, r=4, shots=[150k,1.5M,15M], max_order=8, eps=(6e-5, 3e-5)
#
# Usage:
#   ./run_surface_color.sh [options]
# Options:
#   --surface        Run surface code experiment only
#   --color          Run color code experiment only
#   --ca-only        Run correlation analysis (CA) only, skip decoding
#   --decode-only    Run decoding only (requires CA output to exist)
#   --device DEVICE  Compute device, default cuda:0
#   --base-dir DIR   Output root directory, default data

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Default parameters
RUN_SURFACE=0
RUN_COLOR=0
RUN_CA=1
RUN_DECODE=1
DEVICE="${DEVICE:-cuda:0}"
BASE_DIR="${BASE_DIR:-data}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --surface)
            RUN_SURFACE=1
            shift
            ;;
        --color)
            RUN_COLOR=1
            shift
            ;;
        --ca-only)
            RUN_CA=1
            RUN_DECODE=0
            shift
            ;;
        --decode-only)
            RUN_CA=0
            RUN_DECODE=1
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--surface] [--color] [--ca-only] [--decode-only] [--device DEVICE] [--base-dir DIR]"
            echo ""
            echo "  --surface      Run surface code experiment only"
            echo "  --color        Run color code experiment only"
            echo "                 If neither specified, run both surface and color"
            echo ""
            echo "  --ca-only      Run correlation analysis only"
            echo "  --decode-only  Run decoding only (requires existing CA output)"
            echo "                 If neither specified, run CA then decode"
            echo ""
            echo "  --device       Compute device, default cuda:0"
            echo "  --base-dir     Output root directory, default data"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If neither --surface nor --color specified, run both
if [[ $RUN_SURFACE -eq 0 && $RUN_COLOR -eq 0 ]]; then
    RUN_SURFACE=1
    RUN_COLOR=1
fi

run_surface_ca() {
    echo "========== Surface Code: Correlation Analysis =========="
    python -c "
from experiments import run_correlation_analysis

run_correlation_analysis(
    distance=5,
    rounds=5,
    p_circuit=0.001,
    shots_analysis_list=[50000, 500000, 5000000],
    max_order=4,
    inference_eps=(1.6e-4, 3e-5),
    code_task='surface_code:rotated_memory_x',
    device='$DEVICE',
    CA_mode=['given', 'inference'],
    batch_shots=25000,
    base_dir='$BASE_DIR',
)

run_correlation_analysis(
    distance=5,
    rounds=15,
    p_circuit=0.001,
    shots_analysis_list=[7000000],
    max_order=4,
    inference_eps=(1.6e-4, 3e-5),
    code_task='surface_code:rotated_memory_x',
    device='$DEVICE',
    CA_mode=['inference'],
    batch_shots=100000,
    base_dir='$BASE_DIR',
)

run_correlation_analysis(
    distance=5,
    rounds=25,
    p_circuit=0.001,
    shots_analysis_list=[9000000],
    max_order=4,
    inference_eps=(1.6e-4, 3e-5),
    code_task='surface_code:rotated_memory_x',
    device='$DEVICE',
    CA_mode=['inference'],
    batch_shots=50000,
    base_dir='$BASE_DIR',
)
"
}

run_surface_decode() {
    echo "========== Surface Code: Decoding =========="
    python -c "
from experiments import run_decode_from_files
from utils import get_ca_base_dir

ler_paths = run_decode_from_files(
    get_ca_base_dir(
        code_task='surface_code:rotated_memory_x',
        distance=5,
        rounds=5,
        base_dir='$BASE_DIR',
    ),
    shots_analysis_list=[50000, 500000, 5000000],
    inference_eps=[1.6e-4, 3e-5],
    target_logical_errors=200,
    decoder='bposd',
)
print('Surface results saved:', ler_paths)
"
}

run_color_ca() {
    echo "========== Color Code: Correlation Analysis =========="
    python -c "
from experiments import run_correlation_analysis

run_correlation_analysis(
    distance=5,
    rounds=4,
    p_circuit=0.001,
    shots_analysis_list=[150000, 1500000, 15000000],
    max_order=8,
    inference_eps=(6e-5, 3e-5),
    code_task='color_code:memory_xyz',
    device='$DEVICE',
    CA_mode=['given', 'inference'],
    batch_shots=50000,
    base_dir='$BASE_DIR',
)
"
}

run_color_decode() {
    echo "========== Color Code: Decoding =========="
    python -c "
from experiments import run_decode_from_files
from utils import get_ca_base_dir

ler_paths = run_decode_from_files(
    get_ca_base_dir(
        code_task='color_code:memory_xyz',
        distance=5,
        rounds=4,
        base_dir='$BASE_DIR',
    ),
    shots_analysis_list=[150000, 1500000, 15000000],
    inference_eps=[6e-5, 3e-5],
    target_logical_errors=400,
    decoder='bposd',
)
print('Color results saved:', ler_paths)
"
}

# Execute
if [[ $RUN_SURFACE -eq 1 ]]; then
    if [[ $RUN_CA -eq 1 ]]; then
        run_surface_ca
    fi
    if [[ $RUN_DECODE -eq 1 ]]; then
        run_surface_decode
    fi
fi

if [[ $RUN_COLOR -eq 1 ]]; then
    if [[ $RUN_CA -eq 1 ]]; then
        run_color_ca
    fi
    if [[ $RUN_DECODE -eq 1 ]]; then
        run_color_decode
    fi
fi

echo ""
echo "========== Experiments completed =========="
