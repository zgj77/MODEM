set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.11 .venv
fi
uv pip install --python .venv/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python -r requirements.txt
