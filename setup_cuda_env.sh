set -euo pipefail

_try_source_activate() {
    local venv_dir="$1"
    local candidates=(
        "$venv_dir/bin/activate"
        "$venv_dir/activate"
        "$venv_dir/Scripts/activate"
        "$venv_dir/Scripts/activate.sh"
    )
    for a in "${candidates[@]}"; do
        if [ -f "$a" ]; then
            source "$a"
            return 0
        fi
    done
    return 1
}

echo "=== setup_cuda_env.sh ==="

if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Already inside virtual environment: $VIRTUAL_ENV"
    ACTIVE_PYTHON="$(python -c 'import sys; print(sys.executable)')"
else
    if [ -d "venv" ]; then
        if _try_source_activate "venv"; then
            echo "Activated venv"
        else
            echo "Found ./venv but couldn't source activation; will use its python directly"
            if [ -f "./venv/bin/python" ]; then
                ACTIVE_PYTHON="$(pwd)/venv/bin/python"
            else
                ACTIVE_PYTHON="$(pwd)/venv/Scripts/python.exe"
            fi
        fi
    elif [ -d ".venv" ]; then
        if _try_source_activate ".venv"; then
            echo "Activated .venv"
        else
            echo "Found ./.venv but couldn't source activation; will use its python directly"
            if [ -f "./.venv/bin/python" ]; then
                ACTIVE_PYTHON="$(pwd)/.venv/bin/python"
            else
                ACTIVE_PYTHON="$(pwd)/.venv/Scripts/python.exe"
            fi
        fi
    else
        echo "No venv found: creating ./venv"
        python -m venv venv
        if _try_source_activate "venv"; then
            echo "Created and activated ./venv"
        else
            echo "Created venv but couldn't source; will use venv python directly"
            if [ -f "./venv/bin/python" ]; then
                ACTIVE_PYTHON="$(pwd)/venv/bin/python"
            else
                ACTIVE_PYTHON="$(pwd)/venv/Scripts/python.exe"
            fi
        fi
    fi
fi

if [ -z "${ACTIVE_PYTHON:-}" ]; then
    ACTIVE_PYTHON="$(python -c 'import sys; print(sys.executable)')"
fi

echo "Using Python: $ACTIVE_PYTHON"

echo "Upgrading pip and installing NVIDIA wheels..."
"$ACTIVE_PYTHON" -m pip install --upgrade pip setuptools wheel
"$ACTIVE_PYTHON" -m pip install \
    nvidia-cudnn-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu12

if [ -f "./set_cuda_paths.py" ]; then
    echo "Evaluating set_cuda_paths.py to set environment variables in this shell..."
    eval "$("$ACTIVE_PYTHON" ./set_cuda_paths.py 2>/dev/null)"
    echo "Environment variables updated (CUDA_PATH / CUDA_PATH_V12_4 / PATH)."
else
    echo "Error: set_cuda_paths.py not found in current directory. Please place it next to this script."
    exit 1
fi

echo
echo "Done. To verify, run:"
echo "  python -c 'import os; print(\"CUDA_PATH=\", os.environ.get(\"CUDA_PATH\"))'"
echo
echo "If you want to run your faster-whisper script now, run e.g.:"
echo "  python transcribe.py"
