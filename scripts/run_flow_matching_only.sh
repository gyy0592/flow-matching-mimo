#!/bin/bash
#
# run_flow_matching_only.sh
# Run the flow-matching sweeps with the pretrained score model.
#

set -e

# Runtime configuration
PYTHON="${PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-0}"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=========================================="
echo "Flow Matching SNR sweeps"
echo "=========================================="
echo "Python: $PYTHON"
echo "GPU: $GPU"
echo "Working directory: $REPO_ROOT"
echo ""

# ============================================
# Sweep 1: OT-Flow, 200 steps
# ============================================
echo "[1/2] Running OT-Flow (200 steps)..."
$PYTHON test_score_velocity_multi_test.py \
  --train_config configs/training_config_baseline.yaml \
  --test_config configs/testing_config_velocity.yaml \
  --num_steps 200 \
  --pilots_list 30 \
  --batch_size_x_list 10 20 50 100 \
  --num_realizations 5 \
  --sample_joint True \
  --seed 4321

echo "OT-Flow 200 steps complete"
echo ""

# ============================================
# Sweep 2: OT-Flow, 400 steps
# ============================================
echo "[2/2] Running OT-Flow (400 steps)..."
$PYTHON test_score_velocity_multi_test.py \
  --train_config configs/training_config_baseline.yaml \
  --test_config configs/testing_config_velocity.yaml \
  --num_steps 400 \
  --pilots_list 30 \
  --batch_size_x_list 10 20 50 100 \
  --num_realizations 5 \
  --sample_joint True \
  --seed 4321

echo "OT-Flow 400 steps complete"
echo ""

# ============================================
# Output summary
# ============================================
echo ""
echo "=========================================="
echo "All sweeps complete"
echo "=========================================="
echo ""
echo "Results:"
echo "  - OT-Flow 200 steps: results_velocity_seed4321/steps200/"
echo "  - OT-Flow 400 steps: results_velocity_seed4321/steps400/"
echo ""
echo "List result files:"
echo "  ls -lh results_velocity_seed4321/steps200/*.npz"
echo "  ls -lh results_velocity_seed4321/steps400/*.npz"
echo ""
