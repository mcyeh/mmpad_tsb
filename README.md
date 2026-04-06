## Multidimensional Matrix Profile Submission

This folder contains the self-contained submission implementation of
Multidimensional Matrix Profile for Anomaly Detection (MMPAD) for the TSB-AD
benchmark datasets. It bundles the submission entrypoints, the local MMPAD
detector code, and the CPU/GPU MSTOMP implementations used in the experiments
reported for this project. The implementation covers both univariate and
multivariate datasets.

The purpose of this README is to document the submission workflow, the
available hyperparameter selections, the runtime options, and the expected
inputs and outputs for reproducing the submission runs.

### What Is In This Folder

- `Run_MMPAD_U.py`: univariate submission runner
- `Run_MMPAD_M.py`: multivariate submission runner
- `mmpad_submission/`: bundled MMPAD implementation used by both runners

### Requirements

- A Python environment with `numpy`, `pandas`, and `TSB_AD`
- GPU runs also require `torch` with CUDA support

### Inputs

- `--dataset_dir` points to the directory containing the dataset CSV files
- `--file_list` points to a CSV with a required `file_name` column
- Each dataset CSV must contain feature columns followed by a final `Label`
  column
- Rows with missing values are dropped before scoring

### Quick Start

Univariate:

```bash
python Run_MMPAD_U.py \
  --dataset_dir /path/to/TSB-AD-U \
  --file_list /path/to/TSB-AD-U-Eva.csv \
  --score_dir eval/score/uni \
  --save_dir eval/metrics/uni
```

Multivariate:

```bash
python Run_MMPAD_M.py \
  --dataset_dir /path/to/TSB-AD-M \
  --file_list /path/to/TSB-AD-M-Eva.csv \
  --score_dir eval/score/multi \
  --save_dir eval/metrics/multi
```

GPU example:

```bash
python Run_MMPAD_U.py \
  --dataset_dir /path/to/TSB-AD-U \
  --file_list /path/to/TSB-AD-U-Eva.csv \
  --score_dir eval/score_gpu/uni \
  --save_dir eval/metrics_gpu/uni \
  --backend gpu \
  --gpu_execution auto \
  --gpu_precision float64 \
  --gpu_device cuda:0
```

### Hyperparameter Versions

The runners expose `--hp_version {0,1}`. The selected HP table is applied on
top of the common defaults in `mmpad_submission/wrapper_mmp_ad.py`.

The two HP versions correspond to two different selection rules used in this
project:

- `hp=0` is the original submission configuration selected from the tuning
  results by averaging `VUS-PR`
- `hp=1` is the alternative configuration selected from the tuning results by
  averaging per-dataset `VUS-PR` rank

Univariate:

- `hp=0`: `n_neighbor=5`, `post_processing=2`
- `hp=1`: `n_neighbor=10`, `post_processing=2`

Multivariate:

- `hp=0`: `n_dim=0.7`, `n_neighbor=15`, `post_processing=2`
- `hp=1`: `n_dim=0.5`, `n_neighbor=15`, `post_processing=2`

Common defaults that still apply to both HP versions:

- `sorting_place='pre'`
- `mode='discord'`
- `flat_mode='eps'`
- `budget_mode='downsample'`

### CLI

Both runners accept:

- `--dataset_dir`
- `--file_list`
- `--score_dir`
- `--save_dir`
- `--ad_name`
- `--hp_version {0,1}`
- `--n_job`
- `--verbose`
- `--backend {cpu,gpu}`
- `--gpu_precision {float64,float32}`
- `--gpu_execution {auto,gpu_pipeline,cpu_reference}`
- `--gpu_device`
- `--gpu_reseed_period`
- `--gpu_allow_tf32`

### Runtime Notes

- `--backend {cpu,gpu}` selects the MSTOMP backend
- `--gpu_execution auto` lets the GPU wrapper choose between the GPU pipeline
  and the CPU reference path per workload
- `--gpu_execution gpu_pipeline` forces the GPU path
- `--gpu_execution cpu_reference` forces the CPU MSTOMP path through the GPU
  wrapper
- `--gpu_precision` only matters on the GPU backend
- `--gpu_allow_tf32` only matters with `--gpu_precision float32`
- `--gpu_reseed_period` only matters on the GPU backend
- GPU-specific flags do nothing when `--backend cpu` is used

`n_job` only affects the CPU MSTOMP path:

- On `--backend cpu`, it controls CPU MSTOMP worker count
- On `--backend gpu --gpu_execution gpu_pipeline`, it is ignored
- On `--backend gpu --gpu_execution auto`, it matters only when the workload is
  routed to the CPU reference path

### Outputs

- Score files are written under `<score_dir>/MMPAD/*.npy`
- Metrics are written to `<save_dir>/MMPAD.csv`
- Output directories are created automatically if needed
- The metrics CSV is updated after each completed dataset, so partial progress
  is preserved if a long run is interrupted
- If an evaluation metric is undefined and returns `NaN`, the writer stores
  `0.0` in the CSV so the table stays numeric

### Archived Results

The submission folder also includes local copies of the final archived metric
CSVs:

- `archived_results/cpu_hp_0/uni/MMPAD.csv`
- `archived_results/cpu_hp_0/multi/MMPAD.csv`
- `archived_results/gpu_hp_0/uni/MMPAD.csv`
- `archived_results/gpu_hp_0/multi/MMPAD.csv`
- `archived_results/gpu_hp_1/uni/MMPAD.csv`
- `archived_results/gpu_hp_1/multi/MMPAD.csv`

### Archived Result Summary

The tables below summarize the archived CPU and GPU runs. Values are simple
averages over the archived metric CSVs.

Univariate:

| Metric | cpu_hp=0 | gpu_hp=0 | gpu_hp=1 |
|---|---:|---:|---:|
| AUC-PR | 0.365681 | 0.365727 | 0.352024 |
| AUC-ROC | 0.762911 | 0.762868 | 0.758829 |
| VUS-PR | 0.409957 | 0.409982 | 0.396144 |
| VUS-ROC | 0.777084 | 0.777037 | 0.775489 |
| Standard-F1 | 0.420296 | 0.420354 | 0.406417 |
| PA-F1 | 0.592281 | 0.592278 | 0.589360 |
| Event-based-F1 | 0.478375 | 0.478374 | 0.466481 |
| R-based-F1 | 0.404420 | 0.404504 | 0.398265 |
| Affiliation-F1 | 0.839399 | 0.839399 | 0.831621 |

Multivariate:

| Metric | cpu_hp=0 | gpu_hp=0 | gpu_hp=1 |
|---|---:|---:|---:|
| AUC-PR | 0.300560 | 0.302720 | 0.308575 |
| AUC-ROC | 0.735506 | 0.735871 | 0.740945 |
| VUS-PR | 0.354016 | 0.354794 | 0.359182 |
| VUS-ROC | 0.753886 | 0.754800 | 0.756515 |
| Standard-F1 | 0.355472 | 0.355100 | 0.363848 |
| PA-F1 | 0.485304 | 0.484627 | 0.499663 |
| Event-based-F1 | 0.384658 | 0.384468 | 0.391516 |
| R-based-F1 | 0.289034 | 0.288445 | 0.292446 |
| Affiliation-F1 | 0.742410 | 0.742642 | 0.722859 |

Summary:

- The tables above are archived reference results for `cpu_hp_0`,
  `gpu_hp_0`, and `gpu_hp_1`
- `hp=0` and `hp=1` are two alternative full submission configurations
- The most direct CPU/GPU comparison in this archive is `cpu_hp_0` versus
  `gpu_hp_0`
- The tables are intended as a record of those archived runs, not as a
  recommendation to mix `hp=0` for one track and `hp=1` for the other
- Undefined affiliation scores are averaged as `0.0` in these archived
  summaries

### Files

- `README.md`: this document
- `Run_MMPAD_U.py`: univariate submission runner
- `Run_MMPAD_M.py`: multivariate submission runner
- `mmpad_submission/mmp_ad.py`: matrix-profile detector and score reduction
- `mmpad_submission/mstomp.py`: CPU MSTOMP implementation
- `mmpad_submission/mstomp_gpu.py`: GPU MSTOMP implementation
- `mmpad_submission/submission_runner.py`: batch loop, score writing, and CSV
  metric writing
- `mmpad_submission/util.py`: preprocessing, downsampling, score validation,
  and helper utilities
- `mmpad_submission/wrapper_mmp_ad.py`: defaults, HP tables, and
  `run_MMPAD(...)`

### Citation

If you find this repository interesting and/or useful, please consider citing the following papers:

```bibtex
@inproceedings{yeh2024matrix,
  title={Matrix profile for anomaly detection on multidimensional time series},
  author={Yeh, Chin-Chia Michael and Der, Audrey and Saini, Uday Singh and Lai, Vivian and Zheng, Yan and Wang, Junpeng and Dai, Xin and Zhuang, Zhongfang and Fan, Yujie and Chen, Huiyuan and others},
  booktitle={2024 IEEE International Conference on Data Mining (ICDM)},
  pages={911--916},
  year={2024},
  organization={IEEE}
}

@misc{yeh2026matrix,
  title={Matrix Profile for Time-Series Anomaly Detection: A Reproducible Open-Source Benchmark on TSB-AD}, 
  author={Yeh, Chin-Chia Michael},
  year={2026},
  eprint={2604.02445},
  archivePrefix={arXiv},
  url={[https://arxiv.org/abs/2604.02445](https://arxiv.org/abs/2604.02445)}
}

