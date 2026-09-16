# Federated ASR Gradient Inversion

Research code and recorded artifacts for reconstructing long-form speech features from gradients produced by CTC-based automatic speech recognition models.

This repository accompanies the paper [Long-Form Speech Reconstruction from Gradients in Federated ASR using CTC Loss](https://takakhoo.com/docs/federated-asr-gradient-paper.pdf). It extends an earlier DeepSpeech gradient-matching pipeline with log-space CTC, overlapping temporal grids, per-grid optimizer resets, checkpoints, and analysis tooling.

![Ten-second DeepSpeech reconstruction](reports/10second_with_grid_Nov23/sampleidx_0_grid_concat_firstorder.png)

## Why this project matters

Federated learning keeps raw audio on-device, but the gradients exchanged during training can still encode sensitive information. This project asks whether the non-aligned CTC objective used by speech recognizers is enough to prevent inversion attacks. The experiments show that it is not: an attacker with the model and a client's gradient update can optimize a synthetic MFCC sequence toward the private input.

The engineering challenge is numerical and temporal. Long utterances make CTC gradients unstable and full-sequence optimization difficult. The implementation combines:

- log-space CTC calculations;
- overlapping temporal grids;
- gradient masking so only the active grid is updated;
- per-grid optimizer and learning-rate resets;
- resumable checkpoints and diagnostic plots; and
- quantitative post-processing for MAE, SNR, and transcript checks.

## Repository status

| Surface | Status | Evidence |
|---|---|---|
| Python source | Syntax-checked | All tracked files under `src/` parse successfully |
| CLI and experiment runner | Implemented | `src/main.py` and `run.bash` |
| Recorded 10-second DS1 run | Included | `reports/10second_with_grid_Nov23/` |
| Metrics export | Included | `reports/2025-11-22-long10s/metrics.json` |
| Full GPU reproduction | Environment-bound | Requires CUDA, the LibriSpeech HDF5 subset, and optional model checkpoints |
| Later paper schedules | Paper-only snapshot | The public code currently contains the sequential grid implementation; the paper also reports alternating and error-driven schedules |

The last row is deliberate scope disclosure. This repository should not be read as a one-command reproduction of every table in the paper.

## Recorded baseline

The committed November 2025 artifact uses DeepSpeech1, a 10-second LibriSpeech sample, 300-frame windows with 150-frame overlap, and 1,000 first-order steps per primary window.

| Metric | Recorded value | Interpretation |
|---|---:|---|
| Global MFCC MAE | 8.78 | Feature-space reconstruction error |
| Waveform SNR | -0.33 dB | Inverse-MFCC audio remains poor |
| Decoder WER | 1.00 | The saved evaluation used a randomly initialized decoder and is not a valid intelligibility result |
| Runtime | 3,200 s | Single recorded long-form run |

These values come directly from [`metrics.json`](reports/2025-11-22-long10s/metrics.json). The negative SNR and invalid decoder setup are retained because they are important limitations, not hidden as failed experiments.

## Quick start

The code targets Linux with an NVIDIA GPU. CPU execution is supported for inspection and small checks but is not practical for the full optimization.

```bash
git clone https://github.com/takakhoo/federated-asr-gradient-inversion.git
cd federated-asr-gradient-inversion

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The tracked `environment.yml` is the original Linux/CUDA environment snapshot. `requirements.txt` is the portable dependency list; use it for a clean installation on a new machine.

Prepare the expected data layout:

```text
datasets/
└── librispeech_sampled_600_file_0s_4s/
    ├── dataset_item_0.h5
    └── ...
```

Then run a single DS1 reconstruction:

```bash
python src/main.py \
  --model_name ds1 \
  --dataset_path datasets/librispeech_sampled_600_file_0s_4s \
  --batch_start_idx 0 \
  --batch_end_idx 1 \
  --min_duration_ms 0 \
  --max_duration_ms 1000 \
  --learning_rate 0.5 \
  --max_iterations 2000 \
  --context_frames 6 \
  --dropout_prob 0.0
```

For the recorded grid configuration:

```bash
DATASET_PATH=/absolute/path/to/librispeech_long_10s \
MODEL_NAME=ds1 \
MAX_ITERATIONS=4000 \
./run.bash
```

`run.bash` launches one experiment and exposes its main parameters as environment variables. It no longer starts a machine-specific eight-GPU sweep.

## How the pipeline works

```text
private MFCC + transcript
          |
          v
 DeepSpeech + CTC loss
          |
          v
 observed parameter gradients
          |
          v
 initialize synthetic MFCC
          |
          v
 match gradients one temporal grid at a time
          |
          v
 checkpoints, plots, feature metrics, optional decoding
```

The attack matches gradients at the final DeepSpeech layer. In grid mode, each optimization step computes the full model objective but masks the synthetic input gradient outside the active temporal region. Overlap is used to reduce boundary artifacts.

## Repository map

```text
src/main.py                         CLI and experiment orchestration
src/optimize.py                     full-sequence and grid optimization loops
src/ctc/                            CTC implementation
src/models/                         DS1 and DS2 adapters
src/data/                           HDF5 LibriSpeech loading
src/analysis/generate_metrics.py    metrics and reconstruction export
modules/deepspeech/                 vendored DeepSpeech research dependency
reports/                            selected, reviewable experiment artifacts
notebooks/                          exploratory analysis
environment.yml                    original cluster/CUDA snapshot
requirements.txt                   portable Python dependency list
```

## Reproducibility notes

- The dataset and pretrained checkpoints are intentionally not committed.
- Model checkpoints must match the teacher gradients used in a run. Decoding with random weights produces meaningless WER.
- The original experiments were run on Linux/CUDA. Library behavior and runtime will differ on CPU-only hosts.
- The included DeepSpeech dependency is research code, not a maintained production ASR stack.
- Results in the paper and results committed here belong to different experiment snapshots; use the paper for the full comparison and this repository for the public sequential-grid implementation.

## Verification

The lightweight repository check used for this refresh is:

```bash
python -m compileall -q src
bash -n run.bash
```

End-to-end verification additionally requires the external dataset and GPU environment described above. No claim in this README depends on a run that is absent from either `reports/` or the linked paper.

## Responsible research

Gradient inversion can expose information from training participants. Use this code only with data and models you are authorized to test. The purpose of the project is to measure leakage and motivate stronger privacy controls in federated speech systems.

## Citation

```bibtex
@misc{khoo2026federatedasr,
  title  = {Long-Form Speech Reconstruction from Gradients in Federated ASR using CTC Loss},
  author = {Khoo, Taka and Bui, Minh N. and Chin, Peter},
  year   = {2026},
  note   = {Manuscript under revision}
}
```

## License

No repository-wide license has been granted. The code is available for review and research collaboration; third-party components retain their original terms.
