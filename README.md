# Flow Matching for Joint Channel Estimation and Symbol Detection in Massive MIMO

Code for **Yiyang Guo, Jing Qiao, and Hao Ye**, *Flow Matching for Joint Channel Estimation and Symbol Detection in Massive MIMO*, IEEE International Conference on Communications (ICC), 2026, pp. 1–6.

[Paper DOI](https://doi.org/10.1109/ICC59461.2026.11587708) · [IEEE Xplore](https://ieeexplore.ieee.org/document/11587708/) · [Repository](https://github.com/gyy0592/flow-matching-mimo)

This implementation uses an upstream pretrained diffusion score model and 3GPP channel data for joint channel estimation and symbol detection. Optimal transport Flow Matching updates the channel estimate, pilot and payload likelihoods guide the inference, and Langevin updates refine the detected symbols. Inference saves estimates and error metrics; the analysis tools export metric tables and plots. No model training is required.

## Quick Start

### 1. Requirements

Use Python 3.11, a CUDA GPU, `curl`, and GNU `patch`. The inference environment uses PyTorch 2.10.0 with CUDA 12.8. From the repository root, install PyTorch and the project dependencies:

```bash
python3 -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install -r requirements.txt
```

Obtain the three helper files from the fixed upstream version and apply the included compatibility patch before running inference:

```bash
curl -L https://raw.githubusercontent.com/nzilberstein/joint-score-based-channel/f8f7078b879bdee93d1d7fad78cd219a12286d0b/data/loaders.py -o data/loaders.py
curl -L https://raw.githubusercontent.com/nzilberstein/joint-score-based-channel/f8f7078b879bdee93d1d7fad78cd219a12286d0b/data/sample_generator.py -o data/sample_generator.py
curl -L https://raw.githubusercontent.com/nzilberstein/joint-score-based-channel/f8f7078b879bdee93d1d7fad78cd219a12286d0b/utils/util.py -o utils/util.py
patch -p1 < source_compatibility.patch
```

### 2. Data & Models

Download the channel bank and pretrained score model from the [upstream authors' data and model directory](https://drive.google.com/drive/folders/1uHxGHTyU4SXyHYXvPtjnekEKCVNfb24r?usp=sharing), linked by their [official repository](https://github.com/nzilberstein/joint-score-based-channel). Data and weights are obtained separately from this repository.

| Download | Save as | Purpose |
| --- | --- | --- |
| `H_bank_64.mat` | `data/H_bank_64_original.mat` | 3GPP channel bank. |
| `model_3gpp_64x32.pt` | `models/model_3gpp_64x32.pt` | Pretrained diffusion score model. |

Create the model directory before saving the checkpoint, then convert the channel bank:

```bash
mkdir -p models
python3 data/mat2npz.py
```

The converter produces `data/H_bank_64.npz`. Inference first loads this NPZ file; when `data/H_bank_64_original.mat` exists, its test slice replaces the NPZ test slice. The default test slice contains 50 channels, indices 9500 through 9549, each with 64 receive and 32 transmit dimensions. Keep both input paths consistent with the configuration.

### 3. Run Inference

Run 200 Flow Matching steps with 30 pilot symbols, 50 payload symbols, and five random realizations:

```bash
CUDA_VISIBLE_DEVICES=0 python3 test_score_velocity_multi_test.py \
  --gpu 0 \
  --train_config configs/training_config_baseline.yaml \
  --test_config configs/testing_config_velocity.yaml \
  --num_steps 200 --pilots_list 30 --batch_size_x_list 50 \
  --num_realizations 5 --sample_joint True --seed 4321
```

The command evaluates 13 signal-to-noise ratio (SNR) points and saves estimates and metrics to `results_velocity_seed4321/steps200/all_realizations_batch50.npz`. To run only the first realization, add `--realization_idx 0` while keeping `--num_realizations 5`; its file is `realization_0_batch50.npz` in the same directory. The total realization count determines the generated random arrays.

For 400 steps, change `--num_steps 200` to `--num_steps 400`; the output directory becomes `results_velocity_seed4321/steps400`. The output root has a fixed name, so changing `--seed` does not rename it. To run the supplied 200/400-step sweeps with 10, 20, 50, and 100 payload symbols, use:

```bash
PYTHON=python3 GPU=0 bash scripts/run_flow_matching_only.sh
```

### 4. Analyze Results

Read the saved NPZ results directly to export a metric table and plots:

```bash
python3 paper_style_analysis_shift.py \
  --velocity-dir results_velocity_seed4321/steps200 \
  --velocity-method-name velocity_steps200 \
  --baseline-dir '' --write-csv --plot --skip-comparison \
  --output-dir output
```

The analysis exports `output/paper_metrics_summary.csv` and metric plots under `output/horizontal/` and `output/vertical/`. It reads the inference NPZ format without a separate merge or conversion. For a 400-step run, set `--velocity-dir results_velocity_seed4321/steps400` and `--velocity-method-name velocity_steps400` together.

Channel normalized mean squared error (NMSE) is reported in decibels; symbol NMSE remains linear. Symbol error rate (SER) is the mean error rate over the saved realizations. For the 32-transmit-dimension configuration, CSV and plot SNR coordinates add 15.05149978319906 dB to the internal SNR values, which range from −10 to 20 dB. Do not apply this offset twice.

External Langevin and LMMSE curves require separately obtained results with matching pilot counts, payload lengths, channel slices, random settings, and SNR conventions. The command above analyzes Flow Matching results alone. If matching reference results are available, supply their directories through `--baseline-dir`, `--single-langevin-dir`, and `--lmmse-dir`. External baseline results are not included.

## Code Structure

| Path | Role |
| --- | --- |
| `test_score_velocity_multi_test.py` | Joint channel estimation and symbol detection entry. |
| `mimo_grad/` | Channel/symbol updates, inference initialization, and result recording. |
| `utils/` and `runtime_config.py` | Shared utilities, random seeds, and runtime configuration. |
| `data/` | Channel loading and MAT-to-NPZ conversion. |
| `ncsnv2/models/` | Network definitions required to load the pretrained score model. |
| `configs/training_config_baseline.yaml` | Pretrained model and data configuration used by inference. |
| `configs/testing_config_velocity.yaml` and `configs/runtime.yaml` | Inference settings and shared runtime defaults. |
| `scripts/run_flow_matching_only.sh` | Flow Matching parameter sweeps. |
| `analysis/` and `paper_style_analysis_shift.py` | Metric aggregation, CSV export, and plotting. |

## Acknowledgements and Citation

This implementation uses the pretrained diffusion score model and code components from **Nicolas Zilberstein, Ananthram Swami, and Santiago Segarra**, *Joint channel estimation and data detection in massive MIMO systems based on diffusion models* (2023): [paper](https://arxiv.org/abs/2311.10311) · [code](https://github.com/nzilberstein/joint-score-based-channel). Reused components include channel loading, symbol generation, and shared utilities. The setup commands above obtain a fixed upstream version and apply this project's compatibility changes.

The pretrained network definitions come from [ncsnv2](https://github.com/ermongroup/ncsnv2); its [MIT License](ncsnv2/LICENSE), including Copyright (c) 2020 Yang Song, is retained unchanged. The upstream project also acknowledges [score-based-channels](https://github.com/utcsilab/score-based-channels), whose [research license](https://github.com/utcsilab/score-based-channels/blob/main/license.txt) applies to its code. Data and pretrained weights remain available through the upstream authors' download directory.

If you use this implementation, cite **Yiyang Guo, Jing Qiao, and Hao Ye**, *Flow Matching for Joint Channel Estimation and Symbol Detection in Massive MIMO*, IEEE ICC 2026, pp. 1–6, [doi:10.1109/ICC59461.2026.11587708](https://doi.org/10.1109/ICC59461.2026.11587708). Machine-readable paper metadata is provided in [CITATION.cff](CITATION.cff). Please also cite the upstream work when using its model or code.
