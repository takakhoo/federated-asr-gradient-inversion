import argparse
import csv
import json
import math
import os
import sys
from typing import List, Tuple

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_SRC = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_SRC not in sys.path:
    sys.path.append(PROJECT_SRC)



def compute_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def compute_snr(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    signal_power = float(np.sum(a ** 2))
    noise_power = float(np.sum((a - b) ** 2))
    if noise_power < eps:
        return float("inf")
    return 10.0 * math.log10((signal_power + eps) / (noise_power + eps))


def iter_segments(n_frames: int, grid_size: int, stride: int) -> List[Tuple[int, int, int]]:
    if n_frames < 1 or grid_size < 1 or not 0 < stride <= grid_size:
        raise ValueError("require positive frames/grid and 0 < stride <= grid size")
    segments = []
    grid_idx = 0
    start = 0
    while start < n_frames:
        end = min(start + grid_size, n_frames)
        segments.append((grid_idx, start, end))
        start += stride
        grid_idx += 1
    return segments


def plot_mfcc_segment(gt: np.ndarray, recon: np.ndarray, start: int, end: int, out_path: str, title: str) -> None:
    seg_gt = gt[start:end]
    seg_rec = recon[start:end]
    seg_res = np.abs(seg_gt - seg_rec)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    data = [
        ("Ground Truth", seg_gt),
        ("Reconstruction", seg_rec),
        ("|Residual|", seg_res),
    ]
    for ax, (label, values) in zip(axes, data):
        im = ax.imshow(
            values.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
        )
        ax.set_ylabel("MFCC Coef")
        ax.set_title(f"{title} {label}")
        fig.colorbar(im, ax=ax, orientation="vertical")
    axes[-1].set_xlabel("Frame")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def ctc_greedy_decode(logits: torch.Tensor, alphabet) -> str:
    best = torch.argmax(logits, dim=-1).cpu().numpy()
    tokens = []
    prev = 0
    for idx in best[:, 0]:
        if idx == 0:
            prev = idx
            continue
        if idx != prev:
            tokens.append(idx)
        prev = idx
    return "".join(alphabet.get_symbols(tokens)).strip()


def word_error_rate(ref: str, hyp: str) -> float:
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    m, n = len(ref_words), len(hyp_words)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[m][n] / m


def reconstruct_audio(mfcc: np.ndarray, sample_rate: int) -> np.ndarray:
    # librosa expects shape (n_mfcc, n_frames)
    return librosa.feature.inverse.mfcc_to_audio(mfcc.T, sr=sample_rate, random_state=7)


def main():
    parser = argparse.ArgumentParser(description="Generate metrics/plots/audio for DS1 reconstructions.")
    parser.add_argument("--exp_path", required=True, help="Path to experiment folder (contains sampleidx_* files).")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to analyze.")
    parser.add_argument("--grid_size", type=int, default=300, help="Grid size used during optimization.")
    parser.add_argument("--grid_overlap", type=int, default=150, help="Grid overlap used during optimization.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate for audio reconstruction.")
    parser.add_argument("--context_frames", type=int, default=6, help="Context frames for DS1 model.")
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="Dropout prob for DS1 decoding model.")
    parser.add_argument(
        "--decoder_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional DS1 checkpoint. Accepts either (a) files whose top-level key "
            "is 'network' (Minh's format) or (b) raw state_dicts where keys already "
            "include the 'network.' prefix. If omitted, WER is omitted rather than "
            "reported from an untrained random decoder."
        ),
    )
    args = parser.parse_args()

    payload_path = os.path.join(args.exp_path, f"sampleidx_{args.sample_idx}_x_param_last.pt")
    if not os.path.exists(payload_path):
        raise FileNotFoundError(f"Could not find payload at {payload_path}")

    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    recon = payload["x_param"].squeeze(1).cpu().numpy()
    gt = payload["inputs"].squeeze(1).cpu().numpy()
    transcript = payload.get("transcript", "").strip()
    runtime = float(payload.get("time", 0.0))

    analysis_dir = os.path.join(args.exp_path, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    global_mae = compute_mae(gt, recon)
    global_snr = compute_snr(gt, recon)

    stride = args.grid_size - args.grid_overlap
    segment_rows = []
    for grid_idx, start, end in iter_segments(recon.shape[0], args.grid_size, stride):
        segment_rows.append(
            {
                "grid_index": grid_idx,
                "start_frame": start,
                "end_frame": end,
                "mae": compute_mae(gt[start:end], recon[start:end]),
                "snr_db": compute_snr(gt[start:end], recon[start:end]),
            }
        )

    seg_csv = os.path.join(analysis_dir, "segment_metrics.csv")
    with open(seg_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["grid_index", "start_frame", "end_frame", "mae", "snr_db"])
        for row in segment_rows:
            writer.writerow([row["grid_index"], row["start_frame"], row["end_frame"], row["mae"], row["snr_db"]])

    plot_mfcc_segment(
        gt,
        recon,
        0,
        recon.shape[0],
        os.path.join(analysis_dir, "mfcc_full_comparison.png"),
        "Full Clip",
    )
    for row in segment_rows:
        plot_mfcc_segment(
            gt,
            recon,
            row["start_frame"],
            row["end_frame"],
            os.path.join(analysis_dir, f"mfcc_grid{row['grid_index']}_comparison.png"),
            f"Grid {row['grid_index']}",
        )

    recon_audio = reconstruct_audio(recon, args.sample_rate)
    recon_audio_path = os.path.join(analysis_dir, f"sampleidx_{args.sample_idx}_reconstruction.wav")
    sf.write(recon_audio_path, recon_audio, args.sample_rate)

    hypothesis = None
    wer_value = None
    if args.decoder_checkpoint is not None:
        from models.ds1 import DeepSpeech1WithContextFrames
        decoder_model = DeepSpeech1WithContextFrames(
            n_context=args.context_frames, drop_prob=args.dropout_prob, use_relu=False).cpu()
        state = torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("network"), dict):
            state = {f"network.{k}": v for k, v in state["network"].items()}
        decoder_model.load_state_dict(state, strict=True)
        decoder_model.eval()
        with torch.no_grad():
            logits = decoder_model(torch.from_numpy(recon).unsqueeze(1))
        hypothesis = ctc_greedy_decode(logits, decoder_model.ALPHABET)
        wer_value = word_error_rate(transcript, hypothesis)
    else:
        print("[decoder] WER omitted: no trained decoder checkpoint supplied.")

    metrics = {
        "global_mae": global_mae,
        "global_snr_db": global_snr,
        "runtime_seconds": runtime,
        "segments": segment_rows,
        "decoder_hypothesis": hypothesis,
        "reference_transcript": transcript,
        "wer": wer_value,
        "wer_status": "trained checkpoint supplied; caller must verify provenance" if wer_value is not None else "not evaluated: no trained decoder",
        "snr_domain": "MFCC feature space, not waveform audio",
        "reconstruction_audio": os.path.relpath(recon_audio_path, args.exp_path),
    }
    metrics_path = os.path.join(analysis_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote metrics to {metrics_path}")
    print(f"Decoded transcript: {hypothesis}")
    print(f"WER: {wer_value:.4f}" if wer_value is not None else "WER: not evaluated")


if __name__ == "__main__":
    main()
