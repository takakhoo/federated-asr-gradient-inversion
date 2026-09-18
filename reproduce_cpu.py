"""Privacy diagnostic on owned synthetic features; no real speech or clients."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import argparse
import json
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ctc.ctc_loss_imp import ctc_loss_imp


def reconstruct(seed, noise, updates=250):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Conv1d(3, 8, 3, padding=1), nn.Tanh(), nn.Conv1d(8, 4, 1)).double().eval()
    truth = torch.randn(1, 3, 12, dtype=torch.double)
    labels = torch.tensor([[1, 2, 1]])
    parameters = tuple(model[-1].parameters())  # Last-layer gradients only.
    def observed_gradient(x, create_graph=False):
        lp = model(x).permute(2, 0, 1).log_softmax(-1)
        loss = ctc_loss_imp(lp, labels, [12], [3])
        return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, parameters, create_graph=create_graph)])
    clean = observed_gradient(truth).detach()
    generator = torch.Generator().manual_seed(seed+1000)
    observed = clean + noise * clean.norm()/clean.numel()**.5 * torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    estimate = nn.Parameter(torch.randn_like(truth))
    initial = estimate.detach().clone()
    optimizer = torch.optim.Adam([estimate], lr=.04)
    history = []
    for step in range(updates):
        gradient = observed_gradient(estimate, create_graph=True)
        loss = (gradient-observed).square().sum()/observed.square().sum().clamp_min(1e-12)
        optimizer.zero_grad()
        estimate.grad, = torch.autograd.grad(loss, estimate)
        optimizer.step()
        if step % 10 == 0 or step == updates-1:
            history.append({"update": step+1, "relative_gradient_error": float(loss.detach()),
                            "feature_mae": float((estimate.detach()-truth).abs().mean())})
    final_gradient = observed_gradient(estimate).detach()
    metrics = {"seed": seed, "relative_noise_std": noise, "updates": updates,
               "initial_feature_mae": float((initial-truth).abs().mean()),
               "final_feature_mae": float((estimate.detach()-truth).abs().mean()),
               "final_relative_gradient_error": float((final_gradient-observed).square().sum()/observed.square().sum()),
               "history": history}
    return metrics, truth.numpy()[0], estimate.detach().numpy()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/cpu-diagnostic"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    records, examples = [], {}
    for seed in (7, 19, 41):
        for noise in (0, .1):
            result, truth, recon = reconstruct(seed, noise)
            records.append(result)
            if seed == 7:
                examples[noise] = (truth, recon)
    payload = {"scope": "12 frames × 3 synthetic features, random tiny Conv1d teacher, known transcript, last-layer gradients",
               "not_claimed": "real ASR accuracy, waveform intelligibility, full thesis reproduction, differential privacy guarantee",
               "runs": records}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), layout="constrained")
    axes[1].imshow(examples[0][0], aspect="auto", vmin=-3, vmax=3, cmap="coolwarm")
    axes[1].set(title="True synthetic features", xlabel="Frame", ylabel="Feature")
    for noise, color in ((0, "#2166ac"), (.1, "#b35806")):
        for i, record in enumerate(r for r in records if r["relative_noise_std"] == noise):
            axes[0].semilogy([h["update"] for h in record["history"]], [h["relative_gradient_error"] for h in record["history"]],
                             color=color, alpha=.6, label=f"noise {noise}" if i == 0 else None)
        axes[2 if noise == 0 else 3].imshow(examples[noise][1], aspect="auto", vmin=-3, vmax=3, cmap="coolwarm")
        axes[2 if noise == 0 else 3].set(title=f"Reconstruction (noise {noise})", xlabel="Frame", ylabel="Feature")
    axes[0].set(title="Gradient fit ≠ exact feature recovery", xlabel="Update", ylabel="Relative gradient error")
    axes[0].legend()
    fig.savefig(args.output / "diagnostic.png", dpi=170)
    np.savez(args.output / "synthetic-features.npz", truth=examples[0][0], clean_reconstruction=examples[0][1], noisy_reconstruction=examples[.1][1])
    print(json.dumps([{k:v for k,v in r.items() if k != "history"} for r in records], indent=2))


if __name__ == "__main__":
    main()
