import torch
from torch import Tensor

'''
contains different ctc loss implementation:
    - ctc_loss_imp:  implementation of ctc loss using for loop
    - batched_ctc:   implementation of ctc loss using batched calculation
    - batched_ctc_v2: implementation of ctc loss using batched calculation with different way to calculate alpha next
    - batched_ctc_logspace: implementation of ctc loss using batched calculation in log space
    - batched_ctc_logspace_scale: implementation of ctc loss using batched calculation in log space with scaling

    should use bathced_ctc_logspace as it is the fastest
    should use batched_ctc_logspace_scale if need stable
'''

def ctc_loss_imp(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean'):
    input_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.as_tensor(target_lengths, dtype=torch.long)
    dt = log_probs.dtype
    log_probs = log_probs.double()  # we need the accuracy as we are not in logspace
    targets = targets.long()
    cum_target_lengths = target_lengths.cumsum(0)
    losses = []
    for i in range(log_probs.size(1)):
        input_length = input_lengths[i].item()
        target_length = target_lengths[i].item()
        cum_target_length = cum_target_lengths[i].item()
        # ==========================================================================================================
        targets_prime = targets.new_full((2 * target_length + 1,), blank)
        if targets.dim() == 2:
            targets_prime[1::2] = targets[i, :target_length]
        else:
            targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]
        # ==========================================================================================================
        probs = log_probs[:input_length, i].exp()
        # ==========================================================================================================
        alpha = log_probs.new_zeros((target_length * 2 + 1,))
        alpha[0] = probs[0, blank]
        alpha[1] = probs[0, targets_prime[1]]
        mask_third = (targets_prime[:-2] != targets_prime[2:])
        for t in range(1, input_length):
            alpha_next = alpha.clone()
            alpha_next[1:] += alpha[:-1]
            alpha_next[2:] += torch.where(mask_third, alpha[:-2], alpha.new_zeros(1))
            alpha = probs[t, targets_prime] * alpha_next
        # ==========================================================================================================
        losses.append(-alpha[-2:].sum().log()[None])
    output = torch.cat(losses, 0)
    if reduction == 'mean':
        return (output / target_lengths.to(dtype=output.dtype, device=output.device)).mean()
    elif reduction == 'sum':
        return output.sum()
    output = output.to(dt)
    return output

def batched_ctc(log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, blank=0):
    # out = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)
    batch_size = targets.size(0)
    max_target_length = targets.size(1)
    max_input_length  = log_probs.size(0)
    targets_prime = targets.new_full((batch_size, 2 * max_target_length + 1,), blank)
    targets_prime[:, 1::2] = targets[:, :max_target_length]
    log_probs = log_probs.double()
    probs = log_probs.exp()
    # Initialization
    alpha = log_probs.new_zeros((batch_size, max_target_length * 2 + 1, ))
    alpha[:, 0] = probs[0, :, blank]
    alpha[:, 1] = probs[0].gather(1, targets_prime[:, 1].unsqueeze(-1)).squeeze(-1)
    mask_third = targets_prime[:, :-2] != targets_prime[:, 2:]
    mask_third = targets_prime[:, :-2] != targets_prime[:, 2:]
    zero_tensor = torch.zeros_like(alpha[:, :-2])
    for t in range(1, max_input_length):
        alpha_next = alpha.clone()
        alpha_next[:, 1:] += alpha[:, :-1]
        alpha_next[:, 2:] += torch.where(mask_third, alpha[:, :-2], zero_tensor)
        alpha = probs[t].gather(1, targets_prime) * alpha_next

    tg = target_lengths.unsqueeze(1)

    out = -(alpha.gather(1, tg*2-1) + alpha.gather(1, tg*2)).log().squeeze()
    # out = -alpha[:, -2:].sum(-1).log()
    out = (out / target_lengths).mean()
    #ctx.save_for_backward(log_probs, targets, input_lengths, target_lengths, alpha)
    
    return out

def batched_ctc_v2(log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, blank=0):
    """
    Batched CTC loss with per-time-step normalization to avoid underflow.
    Keeps Minh's DS1 formulation intact while remaining stable for long clips.
    """
    batch_size = targets.size(0)
    max_target_length = targets.size(1)
    max_input_length = log_probs.size(0)
    device = log_probs.device

    targets_prime = targets.new_full((batch_size, 2 * max_target_length + 1,), blank)
    targets_prime[:, 1::2] = targets[:, :max_target_length]

    log_probs = log_probs.double()
    probs = log_probs.exp()

    alpha = log_probs.new_zeros((batch_size, max_target_length * 2 + 1,))
    alpha[:, 0] = probs[0, :, blank]
    alpha[:, 1] = probs[0].gather(1, targets_prime[:, 1].unsqueeze(-1)).squeeze(-1)

    mask_third = targets_prime[:, :-2] != targets_prime[:, 2:]
    zero_tensor = torch.zeros_like(alpha[:, :-2])
    log_scales = torch.zeros(batch_size, dtype=torch.float64, device=device)

    def normalize(curr_alpha, curr_log_scales):
        scale = curr_alpha.sum(dim=1, keepdim=True).clamp_min(1e-300)
        curr_alpha = curr_alpha / scale
        curr_log_scales = curr_log_scales + torch.log(scale.squeeze(1))
        return curr_alpha, curr_log_scales

    alpha, log_scales = normalize(alpha, log_scales)

    for t in range(1, max_input_length):
        alpha_next = alpha.clone()
        alpha_next[:, 2:] = alpha[:, 2:] + alpha[:, 1:-1] + torch.where(mask_third, alpha[:, :-2], zero_tensor)
        alpha_next[:, 1] = alpha[:, 1] + alpha[:, 0]
        alpha_next[:, 0] = alpha[:, 0]

        alpha = probs[t].gather(1, targets_prime) * alpha_next
        alpha, log_scales = normalize(alpha, log_scales)

    tg = target_lengths.unsqueeze(1)
    final_prob = (alpha.gather(1, tg * 2 - 1) + alpha.gather(1, tg * 2)).clamp_min(1e-300).squeeze()
    out = -(torch.log(final_prob) + log_scales)
    out = (out / target_lengths.to(out.dtype)).mean()
    return out

def batched_ctc_logspace(log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, blank=0):
    # out = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)
    batch_size = targets.size(0)
    max_target_length = targets.size(1)
    max_input_length  = log_probs.size(0)
    targets_prime = targets.new_full((batch_size, 2 * max_target_length + 1,), blank)
    targets_prime[:, 1::2] = targets[:, :max_target_length]

    # Initialization
    alpha = torch.full(  (batch_size, max_target_length * 2 + 1, )     , float('-inf'),dtype=torch.float64)
    # alpha = torch.full(  (batch_size, max_target_length * 2 + 1, )     , float('-inf'))
    alpha[:, 0] = log_probs[0, :, blank]
    alpha[:, 1] = log_probs[0].gather(1, targets_prime[:, 1].unsqueeze(-1)).squeeze(-1)
    mask_third = targets_prime[:, :-2] != targets_prime[:, 2:]

    inf_tensor = torch.full_like(alpha[:, :-2], float('-inf'))
    for t in range(1, max_input_length):
        alpha_next = alpha.clone()
        alpha_next[:, 1:] = torch.log(torch.exp(alpha_next[:, 1:]) + torch.exp(alpha[:, :-1]))
        alpha_next[:, 2:] = torch.log( torch.exp(alpha_next[:, 2:]) + 
            torch.exp(torch.where(mask_third, alpha[:, :-2], inf_tensor)) 
        )
        # alpha = torch.log(torch.exp(log_probs[t].gather(1, targets_prime)) * torch.exp(alpha_next))
        alpha = log_probs[t].gather(1, targets_prime) + alpha_next
        # print(alpha_next.exp())
        # print(alpha.exp())
        # print('='*20)
     
    tg = target_lengths.unsqueeze(1)

    out = -(alpha.gather(1, tg*2-1).exp() + alpha.gather(1, tg*2).exp() ).log().squeeze()
    # out = -alpha[:, -2:].sum(-1).log()
    out = (out / target_lengths).mean()
    #ctx.save_for_backward(log_probs, targets, input_lengths, target_lengths, alpha)
    
    return out

def batched_ctc_logspace_scale(log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, blank=0):
    """
    Stable CTC loss that keeps everything in log-space and avoids
    multiplying probabilities directly. Matches the DS1 formulation
    but prevents underflow for long utterances.
    """
    device = log_probs.device
    batch_size = targets.size(0)
    max_target_length = targets.size(1)
    max_input_length = log_probs.size(0)

    log_probs = log_probs.double()

    targets_prime = targets.new_full((batch_size, 2 * max_target_length + 1,), blank)
    targets_prime[:, 1::2] = targets[:, :max_target_length]

    alpha = torch.full((batch_size, 2 * max_target_length + 1,), float('-inf'),
                       dtype=torch.float64, device=device)
    alpha[:, 0] = log_probs[0, :, blank]
    alpha[:, 1] = log_probs[0].gather(1, targets_prime[:, 1].unsqueeze(-1)).squeeze(-1)
    mask_third = targets_prime[:, :-2] != targets_prime[:, 2:]
    neg_inf = torch.full_like(alpha[:, :-2], float('-inf'))

    for t in range(1, max_input_length):
        stay = alpha[:, 2:]
        prev = alpha[:, 1:-1]
        skip = torch.where(mask_third, alpha[:, :-2], neg_inf)

        merged = torch.stack((stay, prev, skip), dim=0)
        next_rest = torch.logsumexp(merged, dim=0)
        next_state = torch.cat(
            (
                alpha[:, 0:1],
                torch.logaddexp(alpha[:, 1:2], alpha[:, 0:1]),
                next_rest
            ),
            dim=1
        )

        alpha = log_probs[t].gather(1, targets_prime) + next_state

    tg = target_lengths.unsqueeze(1)
    last = torch.stack(
        (
            alpha.gather(1, tg * 2 - 1),
            alpha.gather(1, tg * 2)
        ),
        dim=0
    )
    out = -torch.logsumexp(last, dim=0).squeeze()
    out = (out / target_lengths.to(out.dtype)).mean()
    return out

