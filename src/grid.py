"""Window coverage and hard update boundaries for grid reconstruction."""
import torch
import hashlib
import json


def experiment_signature(model, targets, observed_gradients, settings):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for tensor in [*model.state_dict().values(), targets, *observed_gradients]:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def grid_ranges(n_frames, size, overlap):
    if n_frames < 1 or size < 1 or not 0 <= overlap < size:
        raise ValueError("require positive frames/size and 0 <= overlap < size")
    starts = [0]
    while starts[-1] + size < n_frames:
        starts.append(starts[-1] + size - overlap)
    return [(start, min(start+size, n_frames)) for start in starts]


def masked_step(optimizer, parameter, start, end):
    """Gradient masking alone cannot stop Adam momentum or weight decay drift."""
    with torch.no_grad():
        before = parameter.detach().clone()
        parameter.grad[:start] = 0
        parameter.grad[end:] = 0
    optimizer.step()
    with torch.no_grad():
        parameter[:start].copy_(before[:start])
        parameter[end:].copy_(before[end:])
