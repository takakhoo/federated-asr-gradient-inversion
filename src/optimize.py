"""
Optimization loop functions for data reconstruction.

This module contains the optimization routines used to reconstruct input data
by matching gradients with respect to model parameters.
"""

from datetime import datetime
from copy import deepcopy
from grid import grid_ranges, masked_step, experiment_signature

import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

# Local imports
from ctc.ctc_loss_imp import *
from loss.loss import *
from utils.plot import *
from utils.util import *

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Ensure logs propagate to root logger (which has the file handler)
logger.propagate = True


def plot_loss_curves(loss_history, loss_gm_history, loss_reg_history, prefix, args):
    """
    Plot and save loss curves from optimization.
    
    Args:
        loss_history: List of total loss values
        loss_gm_history: List of gradient matching loss values
        loss_reg_history: List of regularization loss values
        prefix: Prefix for the saved plot filename
        args: Arguments containing exp_path for saving
    """
    logger.info('Saving loss plot...')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Total Loss
    axes[0, 0].plot(loss_history, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss over Iterations')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Gradient Matching Loss
    axes[0, 1].plot(loss_gm_history, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Gradient Matching Loss')
    axes[0, 1].set_title('Gradient Matching Loss over Iterations')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Regularization Loss
    axes[1, 0].plot(loss_reg_history, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Regularization Loss')
    axes[1, 0].set_title('Regularization Loss over Iterations')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: All losses together (log scale)
    axes[1, 1].semilogy(loss_history, 'b-', linewidth=2, label='Total Loss', alpha=0.7)
    axes[1, 1].semilogy(loss_gm_history, 'g-', linewidth=2, label='Gradient Matching', alpha=0.7)
    if loss_reg_history and max(loss_reg_history) > 0:
        axes[1, 1].semilogy(loss_reg_history, 'r-', linewidth=2, label='Regularization', alpha=0.7)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Loss (log scale)')
    axes[1, 1].set_title('All Losses (Log Scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to the experiment path with the prefix and timestamp
    timestamp = '0'
    loss_plot_path = os.path.join(args.exp_path, f'{prefix}_loss_curves_{timestamp}.png')
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Loss plot saved to: {loss_plot_path}')


def zero_order_optimization_loop(inputs, x_param, output_sizes, target_size,
                                 model,
                                    dldw_targets , params_to_match, targets, prefix, args):
    """
    Zero-order (derivative-free) optimization loop using random search.
    
    Args:
        inputs: Original input data
        x_param: Parameter to optimize (reconstructed input)
        output_sizes: Output sequence lengths
        target_size: Target sequence lengths
        model: Neural network model
        dldw_targets: Target gradients to match
        params_to_match: Model parameters to match gradients for
        targets: Target labels
        prefix: Prefix for saving plots
        args: Arguments containing optimization settings
        
    Returns:
        Optimized x_param
    """
    device = inputs.device
    model.eval()

    loss_func = lambda x,y : batched_ctc_v2(x, y, output_sizes, target_size)
    

    i = 0 
    stop_condition = False

    def get_meta_loss(x_param):
        out = model(x_param)
        out = out.log_softmax(-1)
        mloss, dldws = meta_loss(out, targets, None, None, dldw_targets, params_to_match, loss_func, args)
        return mloss, dldws
    
    tolerance = 10
    step_size = args.zero_order_lr

    loss_history = []
    loss_gm_history = []
    loss_reg_history = []

    while i < args.zero_max_iterations and not stop_condition:
        # random 16 directions in the space of x_param
        directions = torch.randn(8, *x_param.shape).to(device)
        # normalize direction so that they have unit length
        shape = directions.shape
        directions = directions.reshape(8,-1)
        directions = torch.functional.F.normalize(directions, dim=1)
        directions = directions.reshape(shape)
        # create a list to store the loss for each direction
        losses = []
        current_loss, _ = get_meta_loss(x_param)
        # logger.info('Current loss: {}'.format(current_loss.item()))

        for d in directions:
            x_param_new = x_param + step_size * d
            mloss, _ = get_meta_loss(x_param_new)
            losses.append(mloss.item())

        # find the best direction by averaging direction that reduce loss
        # best_direction = directions[torch.tensor(losses) < current_loss.item()]
        # find the best direction by the smallest loss, argmin
        best_direction = directions[np.argmin(losses)]

        if  np.min(losses) < current_loss.item():
            # best_direction = best_direction.mean(dim=0)
            x_param = x_param + step_size * best_direction
            tolerance = 10
            step_size = args.zero_order_lr
        else:  
            logger.info('No direction found, reducing step size, tolerance: {}, {}'.format(step_size,tolerance))
            tolerance -=1
            step_size *= 0.5
            if tolerance < 0:
                stop_condition = True

        mae = torch.mean(torch.abs(x_param - inputs))
        logger.info('iter {}  loss: {}, step size: {}, mae: {}'.format(i, np.min(losses), step_size, mae.item()))


        loss_history.append(current_loss.item())    
        loss_gm_history.append(0)
        loss_reg_history.append(0)
        if i % 20 == 0:
            plot_four_graphs(inputs.detach(), x_param.detach(), loss_history, loss_gm_history,loss_reg_history ,i, prefix=prefix, args=args)
            pass
 
        i += 1

    return x_param


def first_order_optimization_loop(inputs, x_param, output_sizes, target_sizes,
                                  optimizer, scheduler, model,
                                  dldw_targets , params_to_match, targets,prefix,  args):
    """
    First-order (gradient-based) optimization loop.
    
    Uses gradient descent to optimize x_param by matching gradients with respect
    to model parameters.
    
    Args:
        inputs: Original input data
        x_param: Parameter to optimize (reconstructed input)
        output_sizes: Output sequence lengths
        target_sizes: Target sequence lengths
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        model: Neural network model
        dldw_targets: Target gradients to match
        params_to_match: Model parameters to match gradients for
        targets: Target labels
        prefix: Prefix for saving plots
        args: Arguments containing optimization settings
        
    Returns:
        Optimized x_param
    """
    model.eval()
    loss_func = lambda x,y : batched_ctc_v2(x, y, output_sizes, target_sizes)

    i=0
    loss_history = []
    loss_gm_history = []
    loss_reg_history = []
    stop_condition = False
    while i < args.max_iterations and not stop_condition:
        # x_param_full= torch.concat([x_param, x_pad], dim=2)
        out = model(x_param) # 1 176 29
        out = out.log_softmax(-1)
        # mloss, dldw_f = meta_loss(output, targets, output_sizes, target_sizes, dldw_target,  weight_param)
        mloss, dldws = meta_loss(out, targets, None, None, dldw_targets,  params_to_match, loss_func, args)
        gm_weight_distance = grad_distance(dldws[0], dldw_targets[0], args)
        # DeepSpeech1 has bias, DeepSpeech2 doesn't
        if len(dldw_targets) > 1:
            gm_bias_distance = grad_distance(dldws[1], dldw_targets[1], args)
        else:
            gm_bias_distance = 0.0

        # regloss = tv_norm(x_param)
        if args.regularization == 'L2':
            regloss = torch.norm(x_param, p=2)
        elif args.regularization == 'L1':
            regloss = torch.norm(x_param, p=1)
        elif args.regularization == 'TV':
            # need to make x_param from [n_frame, batch size, n_features] to [batch size, 1, n_features, n_frame]
            regloss = tv_norm(x_param.permute(1,0,2).unsqueeze(1))
        else:
            regloss = x_param.new_zeros(())
       
        loss = (1-args.reg_weight)* mloss + args.reg_weight * regloss



        optimizer.zero_grad()
        loss.backward()
        grad = x_param.grad.data

        # torch.nn.utils.clip_grad_norm_(x_param, 1.0)
        optimizer.step()
        scheduler.step()


        mae = torch.mean(torch.abs(x_param - inputs))

        loss_history.append(loss.item())
        loss_gm_history.append(mloss.item() )
        loss_reg_history.append(regloss.item() )

        if i % 10 == 0:
            # Convert gm_bias_distance to float if it's a tensor, otherwise use as-is
            bias_dist_val = gm_bias_distance.item() if hasattr(gm_bias_distance, 'item') else gm_bias_distance
            logger.info('Iter, Loss (A-G-Gw-Gb-R), Gradient Norm, Learning Rate, MAE: {:4d}, {:.8f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'\
                        .format(i, loss.item(), mloss.item(),  gm_weight_distance.item(), bias_dist_val, regloss.item()
            , grad.norm().item(), optimizer.param_groups[0]["lr"], mae.item()))
            # scheduler.step(mloss.item())

        if i % 100 == 0:
            plot_four_graphs(inputs.detach(), x_param.detach(), loss_history, loss_gm_history,loss_reg_history ,i,prefix=prefix, args=args)
            pass
            
        
        i+=1
        # stet stop condition true if loss not decrease in last 100 iteration
        if i > args.patience and loss_history[-1] > min(loss_history[-args.patience:]):
            stop_condition = True
    
    # Plot loss curves at the end of optimization
    plot_loss_curves(loss_history, loss_gm_history, loss_reg_history, prefix, args)
    
    return x_param


def first_order_optimization_grid_loop(inputs, x_param, output_sizes, target_sizes,
                                  optimizer, scheduler, model,
                                  dldw_targets , params_to_match, targets, prefix, args,
                                  grid_size=100, overlap=50):
    """
    Segment the inputs into (overlapping) grids and perform first-order optimization on each grid.
    Then aggregate the optimized grids back into a full input.
    Basically, if we optimize whole inputs it may be hard, we still initialize x_param as whole inputs, 
    but we only update a segment of it each time.
    
    Args:
        inputs: Original input data [n_frames, batch_size, n_features]
        x_param: Parameter to optimize [n_frames, batch_size, n_features]
        output_sizes: Output sequence lengths
        target_sizes: Target sequence lengths
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        model: Neural network model
        dldw_targets: Target gradients to match
        params_to_match: Model parameters to match gradients for
        targets: Target labels
        prefix: Prefix for saving plots
        args: Arguments containing optimization settings
        grid_size: Size of each grid segment (in frames)
        overlap: Number of overlapping frames between adjacent grids
        
    Returns:
        Optimized x_param
    """
    model.eval()
    loss_func = lambda x,y: batched_ctc_v2(x, y, output_sizes, target_sizes)
    
    n_frames, batch_size, n_features = x_param.shape
    stride = grid_size - overlap
    
    # Calculate number of grids needed to cover all frames
    windows = grid_ranges(n_frames, grid_size, overlap)
    n_grids = len(windows)
    initial_optimizer = deepcopy(optimizer.state_dict())
    initial_scheduler = deepcopy(scheduler.state_dict())
    signature = experiment_signature(model, targets, dldw_targets, {
        "grid_size": grid_size, "overlap": overlap, "shape": list(x_param.shape),
        "updates": args.max_iterations, "optimizer": initial_optimizer["param_groups"],
        "regularization": args.regularization, "reg_weight": args.reg_weight,
        "distance": args.distance_metric, "top_fraction": args.top_grad_percentage,
        "distance_weight": getattr(args, "distance_metric_weight", None)})
    
    logger.info(f'Grid optimization: n_frames={n_frames}, grid_size={grid_size}, overlap={overlap}, stride={stride}, n_grids={n_grids}')
    
    # Create weighting masks for each grid (triangular/Hann window for smooth blending)
    def create_blend_mask(grid_idx, start_frame, end_frame):
        """Create a triangular blending mask for the current grid segment."""
        segment_length = end_frame - start_frame
        mask = torch.ones(n_frames, device=x_param.device)
        
        # Apply triangular windowing in overlap regions
        if grid_idx > 0:  # Not the first grid - ramp up at the start
            ramp_up_length = min(overlap, segment_length)
            mask[start_frame:start_frame + ramp_up_length] = torch.linspace(0, 1, ramp_up_length, device=x_param.device)
        
        if grid_idx < n_grids - 1:  # Not the last grid - ramp down at the end
            ramp_down_length = min(overlap, segment_length)
            mask[end_frame - ramp_down_length:end_frame] = torch.linspace(1, 0, ramp_down_length, device=x_param.device)
        
        return mask.view(n_frames, 1, 1)  # Shape: [n_frames, 1, 1] for broadcasting
    
    i = 0
    loss_history = []
    loss_gm_history = []
    loss_reg_history = []
    stop_condition = False
    
    # Store optimized segments with their masks for later blending
    optimized_segments = []
    segment_masks = []
    
    # Iterate through each grid
    for grid_idx in range(n_grids):
        optimizer.load_state_dict(deepcopy(initial_optimizer))
        scheduler.load_state_dict(deepcopy(initial_scheduler))
        stop_condition = False
        start_frame = grid_idx * stride
        end_frame = min(start_frame + grid_size, n_frames)
        
        logger.info(f'Optimizing grid {grid_idx + 1}/{n_grids}: frames [{start_frame}:{end_frame}]')
        
        # Don't clone - work directly on x_param but mask gradients
        # Create the blending mask for this grid
        blend_mask = create_blend_mask(grid_idx, start_frame, end_frame)
        
        checkpoint_path = os.path.join(args.exp_path, f'{prefix}_grid{grid_idx}_checkpoint.pt')
        if getattr(args, "resume_grids", False) and os.path.exists(checkpoint_path):
            logger.info(f'Found checkpoint for grid {grid_idx + 1}/{n_grids} at {checkpoint_path}; loading and skipping optimization.')
            checkpoint = torch.load(checkpoint_path, map_location=x_param.device, weights_only=True)
            if not isinstance(checkpoint, dict) or checkpoint.get("signature") != signature:
                raise ValueError("Grid checkpoint provenance mismatch; start a fresh run")
            checkpoint_tensor = checkpoint['x_param']
            with torch.no_grad():
                x_param.copy_(checkpoint_tensor.to(x_param.device))
            optimized_segments.append(x_param[start_frame:end_frame].detach().clone())
            segment_masks.append(blend_mask[start_frame:end_frame])
            continue

        # Optimize this grid segment for a subset of iterations
        grid_iterations = args.max_iterations // n_grids + int(grid_idx < args.max_iterations % n_grids)
        grid_i = 0
        
        while grid_i < grid_iterations and not stop_condition:
            out = model(x_param)
            out = out.log_softmax(-1)
            
            mloss, dldws = meta_loss(out, targets, None, None, dldw_targets, params_to_match, loss_func, args)
            gm_weight_distance = grad_distance(dldws[0], dldw_targets[0], args)
            
            # DeepSpeech1 has bias, DeepSpeech2 doesn't
            if len(dldw_targets) > 1:
                gm_bias_distance = grad_distance(dldws[1], dldw_targets[1], args)
            else:
                gm_bias_distance = 0.0
            
            # Regularization
            if args.regularization == 'L2':
                regloss = torch.norm(x_param[start_frame:end_frame], p=2)
            elif args.regularization == 'L1':
                regloss = torch.norm(x_param[start_frame:end_frame], p=1)
            elif args.regularization == 'TV':
                regloss = tv_norm(x_param[start_frame:end_frame].permute(1, 0, 2).unsqueeze(1))
            else:
                regloss = x_param.new_zeros(())
            
            loss = (1 - args.reg_weight) * mloss + args.reg_weight * regloss
            
            optimizer.zero_grad()
            loss.backward()
            
            # Mask gradients: only update the current grid segment
            if x_param.grad is not None:
                grad_mask = torch.zeros_like(x_param)
                grad_mask[start_frame:end_frame] = 1.0
                x_param.grad.data *= grad_mask
                grad = x_param.grad.data
            else:
                grad = torch.zeros_like(x_param)
            
            masked_step(optimizer, x_param, start_frame, end_frame)
            scheduler.step()
            
            mae = torch.mean(torch.abs(x_param[start_frame:end_frame] - inputs[start_frame:end_frame]))
            
            loss_history.append(loss.item())
            loss_gm_history.append(mloss.item())
            loss_reg_history.append(regloss.item())
            
            if grid_i % 10 == 0:
                bias_dist_val = gm_bias_distance.item() if hasattr(gm_bias_distance, 'item') else gm_bias_distance
                logger.info('Grid {:d}/{:d}, Iter {:4d}, Loss (A-G-Gw-Gb-R), Gradient Norm, LR, MAE: {:.8f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'\
                            .format(grid_idx + 1, n_grids, grid_i, loss.item(), mloss.item(), gm_weight_distance.item(), 
                                    bias_dist_val, regloss.item(), grad.norm().item(), optimizer.param_groups[0]["lr"], mae.item()))
            
            if grid_i % 100 == 0:
                plot_prefix = f'{prefix}_grid{grid_idx}_iter{grid_i}'
                plot_four_graphs(inputs.detach(), x_param.detach(), loss_history, loss_gm_history, 
                                loss_reg_history, i + grid_i, prefix=plot_prefix, args=args)
            
            grid_i += 1
            
            # Check stop condition
            if grid_i > args.patience and loss_history[-1] > min(loss_history[-args.patience:]):
                stop_condition = True
        
        torch.save({"x_param": x_param.detach().cpu(), "signature": signature}, checkpoint_path)
        logger.info(f'Saved checkpoint for grid {grid_idx + 1}/{n_grids} to {checkpoint_path}')

        # Store the optimized segment with its mask for final blending (if needed)
        optimized_segments.append(x_param[start_frame:end_frame].detach().clone())
        segment_masks.append(blend_mask[start_frame:end_frame])
        
        i += grid_i
    
    # Since we optimized in-place with gradient masking, x_param is already updated
    # No need for aggregation - the masking ensures only relevant parts were updated
    logger.info('Grid optimization complete.')
    
    # Plot final loss curves
    plot_loss_curves(loss_history, loss_gm_history, loss_reg_history, prefix, args)
    
    return x_param
    

    

def optimization_loop(inputs, x_param, output_sizes, target_sizes,
                       optimizer, scheduler, model, 
                       dldw_targets , params_to_match, targets,prefix='',  args=None):
    """
    Main optimization loop that orchestrates first-order and optionally zero-order optimization.
    
    This function manages checkpointing and calls first_order_optimization_loop
    and optionally zero_order_optimization_loop.
    
    Args:
        inputs: Original input data
        x_param: Parameter to optimize (reconstructed input)
        output_sizes: Output sequence lengths
        target_sizes: Target sequence lengths
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        model: Neural network model
        dldw_targets: Target gradients to match
        params_to_match: Model parameters to match gradients for
        targets: Target labels
        prefix: Prefix for saving checkpoints and plots
        args: Arguments containing optimization settings
        
    Returns:
        Optimized x_param
    """
    device = inputs.device

    # loss_func = lambda x,y :ctc_loss_imp(x, y, output_sizes, target_sizes,reduction='mean')

    if not os.path.exists(os.path.join(args.exp_path, prefix+'_x_param_first_order.pt')) or not args.resume_from_first_order:
        logger.info('Running first order optimization loop')
        x_param = first_order_optimization_loop(inputs, x_param, output_sizes, target_sizes, optimizer, scheduler, model, dldw_targets, params_to_match, targets,prefix+'_firstorder', args) 
        torch.save(x_param.detach().cpu(), os.path.join(args.exp_path, prefix+'_x_param_first_order.pt'))
        logger.info('x_param_first_order.pt saved')
    else:
        x_param = torch.load(os.path.join(args.exp_path, prefix+'_x_param_first_order.pt')).to(device)
        logger.info('x_param_first_order.pt loaded')

    if args.use_zero_order_optimization:
        logger.info('Running zero order optimization loop')
        x_param = zero_order_optimization_loop(inputs, x_param, output_sizes, target_sizes, model, dldw_targets, params_to_match, targets,prefix+'_zeroorder',args)


    return x_param
