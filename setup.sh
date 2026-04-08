#!/bin/bash
# Mall Car Park RL — Mac Setup Script
# Run once: bash setup.sh

set -e

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Mall Car Park RL — Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── 1. Check python3 ──────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "❌  python3 not found."
  echo "    Install it from https://www.python.org/downloads/"
  echo "    or via Homebrew:  brew install python"
  exit 1
fi

PYTHON=python3
PIP=pip3
echo "✅  Python: $($PYTHON --version)"

# ── 2. Create virtual environment ─────────────
if [ ! -d "venv" ]; then
  echo ""
  echo "→  Creating virtual environment (venv/)..."
  $PYTHON -m venv venv
fi

source venv/bin/activate
echo "✅  Virtual environment active"

# ── 3. Upgrade pip silently ───────────────────
pip install --upgrade pip --quiet

# ── 4. Install all dependencies ───────────────
echo ""
echo "→  Installing dependencies (this may take 2-3 minutes)..."
pip install \
  numpy \
  pandas \
  openpyxl \
  gymnasium \
  stable-baselines3 \
  torch \
  tensorboard \
  tqdm \
  rich \
  --quiet

echo ""
echo "✅  All packages installed"

# ── 5. Smoke test ─────────────────────────────
echo ""
echo "→  Running environment smoke test..."
python3 -c "
from environment import MallCarParkEnv
from baselines import RandomAgent
env = MallCarParkEnv(n_steps=10)
obs, _ = env.reset()
for _ in range(10):
    a = RandomAgent().predict(obs, env)
    obs, _, term, trunc, _ = env.step(a)
    if term or trunc: break
print('  Environment OK — obs shape:', obs.shape)
"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete!"
echo ""
echo "  To train:"
echo "    source venv/bin/activate"
echo "    python3 train.py --data mall_carpark_rl_dataset.xlsx --timesteps 300000"
echo ""
echo "  To watch training live:"
echo "    tensorboard --logdir logs/"
echo ""
echo "  To run inference after training:"
echo "    python3 infer.py --data mall_carpark_rl_dataset.xlsx --render"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
