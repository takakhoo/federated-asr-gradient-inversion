# This is the main script to reconstruct data points for DeepSpeech architectures
# Supports both DeepSpeech1 (DS1) and DeepSpeech2 (DS2) architectures
#
# Key differences:
#   - DS1: Uses MFCC features (26 coefficients) with context frames (default 6)
#          Input dim = n_mfcc * (2*context_frames + 1) = 26 * 13 = 338
#   - DS2: Uses Log-Magnitude STFT features (spectrogram)
#          Input dim = sample_rate * winlen / 2 + 1 = 16000 * 0.032 / 2 + 1 = 257
#
# Example usage:
#   For DeepSpeech2:
#     python src/main.py --model_name DeepSpeech2 --batch_start_idx 0 --batch_end_idx 1 --min_duration_ms 1000 --max_duration_ms 2000
#   
#   For DeepSpeech1:
#     python src/main.py --model_name DeepSpeech1 --batch_start_idx 0 --batch_end_idx 1 --min_duration_ms 1000 --max_duration_ms 2000 --context_frames 6 --dropout_prob 0.0
#
import argparse
from ast import arg
import logging
import numpy as np
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose

# Configure basic logging (will be updated with file handler in main)
# Only show logs from this file by using a named logger
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
# logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Resolve imports relative to this file so the CLI works from any directory.
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
sys.path.insert(0, str(REPO_ROOT / 'modules' / 'deepspeech' / 'src'))
sys.path.insert(0, str(SRC_DIR))

# Local imports
from ctc.ctc_loss_imp import *
from data.librisubset import *
from loss.loss import *
from models.ds1 import DeepSpeech1WithContextFrames
from models.ds2 import DeepSpeech2
from optimize import zero_order_optimization_loop, first_order_optimization_loop, first_order_optimization_grid_loop, optimization_loop
from utils.plot import *
from utils.util import *

# Implements the model to use for the reconstruction
def get_model(args, use_relu):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.model_name.lower() == 'deepspeech2' or args.model_name.lower() == 'ds2':
        model = DeepSpeech2(winlen=0.032, winstep=0.02).to(device)
    elif args.model_name.lower() == 'deepspeech1' or args.model_name.lower() == 'ds1':
        model = DeepSpeech1WithContextFrames(n_context=args.context_frames, drop_prob=args.dropout_prob, use_relu=use_relu).to(device)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}. Use 'DeepSpeech1'/'ds1' or 'DeepSpeech2'/'ds2'")
    return device, model

def reconstruct_dataset(network, device, dataloader, args):
    torch.manual_seed(0)

    # loop through item in the dataloader
    for (i, batch) in enumerate(dataloader):


        # batch = A tuple of `((batch_x, batch_out_lens), batch_y)` where:
        logger.info('#'*20)
        logger.info('Processing batch {}/{}'.format(i, len(dataloader)))

        inputs ,input_sizes = batch[0]
        logger.info('inputs shape: {} (time_steps, batch, features)'.format(inputs.shape))
        logger.info('inputs mean and std: {:.4f}, {:.4f}'.format(inputs.mean(), inputs.std()))
        # input_sizes is tensor list of inputs.shape[1] elements with value inputs.shape[0]

        targets = batch[1]
        text = ''.join(network.ALPHABET.get_symbols(targets[0].tolist()))
        logger.info('TEXT: {}'.format(text))
        target_sizes = torch.Tensor([len(t) for t in targets]).int()

        #target is list of tensor with different length, pad it to the same length in a tensor
        targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)


        # transfer the data to the GPU
        inputs = inputs.to(device)
        targets = targets.long().to(device)

        input_sizes = input_sizes.long().to(device)
        target_sizes = target_sizes.long().to(device)


        out = network(inputs)


        # Select params_to_match based on model architecture
        # DeepSpeech2 uses only weight in last FC (no bias)
        # DeepSpeech1 uses both weight and bias in output layer
        if args.model_name.lower() == 'deepspeech2' or args.model_name.lower() == 'ds2':
            params_to_match = [network.network.fc.module[0].weight] # deep speech 2 doesnt use bias in last FC
        elif args.model_name.lower() == 'deepspeech1' or args.model_name.lower() == 'ds1':
            params_to_match = [network.network.out.module[0].weight, network.network.out.module[0].bias]
        else:
            raise ValueError(f"Unknown model name: {args.model_name}")
            
        output_sizes = (torch.ones(out.shape[1]) * out.shape[0]).int()
        out =  out.log_softmax(-1)

        loss_func = lambda x,y : batched_ctc_v2(x, y, output_sizes, target_sizes)

        loss = loss_func(out, targets)
        # loss_func_lib   = torch.nn.CTCLoss()
        # loss_lib = loss_func_lib(out.cpu(), targets.cpu(), output_sizes.cpu(), target_sizes.cpu())

        logger.debug('loss: {}'.format(loss.item()))
        dldw_targets = torch.autograd.grad(loss, params_to_match)

        ## zero out small values keep 10% largest dldw_target
        # logger.info('zero out small values keep 10% largest dldw_target')
        # dldw_target = dldw_target * (dldw_target.abs() > dldw_target.abs().topk(int(0.1*dldw_target.numel()))[0][-1])
        for ip, p in enumerate(params_to_match):
            p.requires_grad = True
            logger.debug('matching {}. params with shape {} and norm {} first ten {}'.format(ip, p.shape, p.norm(), p.flatten()[:10]))
            logger.debug('                    gradient norm {}'.format(dldw_targets[ip].norm()))


        x_init =  init_a_point(inputs, args)
        x_param = torch.nn.Parameter(x_init.to(device),requires_grad=True)


        if args.optimizer.lower() == 'adam':
            optimizer = optim.Adam([x_param], lr=args.learning_rate)
        elif args.optimizer.lower() == 'sgd':
            optimizer = optim.SGD([x_param], lr=args.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")

        # reduce lr at epoch 250, 500, 750 half
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(250,2000,250)), gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=.5,patience=50)


        # suggest an experiment name base on datapoint index, optimizer name,  learning rate, regularizer, regularizer weight
        logger.info('Experiment Name: {}'.format(os.path.basename(args.exp_path)))

        # timing the optimization loop
        start_time = time.time()
        
        # Choose between grid-based or vanilla optimization
        if args.use_grid_optimization:
            logger.info(f'Using grid-based optimization with grid_size={args.grid_size}, overlap={args.grid_overlap}')
            x_param = first_order_optimization_grid_loop(
                inputs, x_param, output_sizes, target_sizes,
                optimizer, scheduler, network,
                dldw_targets=dldw_targets, params_to_match=params_to_match, 
                targets=targets, prefix=f'sampleidx_{i}_grid', args=args,
                grid_size=args.grid_size, overlap=args.grid_overlap)
        else:
            x_param = optimization_loop(inputs, x_param, output_sizes, target_sizes,
                            optimizer, scheduler, network,
                            dldw_targets = dldw_targets, params_to_match =  params_to_match, targets = targets, prefix=f'sampleidx_{i}', args=args)
        
        end_time = time.time()
                        
        save_path = os.path.join(args.exp_path, f'sampleidx_{i}_' + 'x_param_last.pt'.format(i))

        # Capture the DS1 weights so downstream decoders can replay the exact model.
        network_state = {}
        for key, value in network.state_dict().items():
            if not torch.is_tensor(value):
                continue
            cpu_value = value.detach().cpu()
            if key.startswith('network.'):
                trimmed = key[len('network.'):]
                network_state[trimmed] = cpu_value
            else:
                network_state[key] = cpu_value

        # save x_param, optimization time, inputs, targets, and model snapshot
        torch.save({
            'x_param': x_param.detach().cpu(),
            'time': end_time - start_time,
            'inputs': inputs.detach().cpu(),
            'targets': targets.detach().cpu(),
            'transcript': text,
            'network': network_state
        }, save_path)

def main(args):
    """
    Main function for reconstructing data points with specified hyperparameters.
   
    Parameters:
    - index: Index of the data point to reconstruct.
    - lr: Learning rate for optimization.
    - reg: Type of regularization ('L1', 'L2', 'TV').
    - reg_weight: Weight of the regularization term.
    - iterations: Number of iterations for the optimization.
    """
    
    #check if exp_path exists or create
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)
    if not os.path.exists(os.path.join(args.exp_path, 'figures')):
        os.makedirs(os.path.join(args.exp_path, 'figures'))
    
    # Set up file logging now that exp_path exists
    log_file = os.path.join(args.exp_path, 'experiment.log')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Get root logger and add file handler
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    # Also add file handler to our module logger
    logger.addHandler(file_handler)
    
    logger.info('='*80)
    logger.info('Starting new experiment run')
    logger.info('='*80)
    
    # Log all arguments
    logger.info('Experiment configuration:')
    for key, value in args.__dict__.items():
        logger.info('  {}: {}'.format(key, value))
    logger.info('-'*80)
    
    # Change all print statements to logging statements
    logger.info('Optimizer: {}'.format(args.optimizer))
    logger.info('Learning rate: {}'.format(args.learning_rate))
    logger.info('Regularization: {}'.format(args.regularization))
    logger.info('Regularization weight: {}'.format(args.reg_weight))
    logger.info('Number of iterations: {}'.format(args.max_iterations))
    logger.info('exp_path: {}'.format(args.exp_path))
    logger.info('Log file: {}'.format(log_file))

    # model & devices
    device, model = get_model(args, use_relu=False)
    logger.info('Device: {}'.format(device))
    logger.info('Network: {}'.format((model.__class__.__name__)))
    
    # Log preprocessing information based on model type
    if args.model_name.lower() in ['deepspeech1', 'ds1']:
        logger.info('Preprocessing: MFCC features (n_mfcc={}, context_frames={})'.format(
            model._n_mfcc, model._n_context))
        logger.info('Input feature dimension: {} (MFCC coefficients * (2*context+1))'.format(
            model._n_mfcc * (2 * model._n_context + 1)))
    elif args.model_name.lower() in ['deepspeech2', 'ds2']:
        n_features = int((model._sample_rate * model._winlen) // 2 + 1)
        logger.info('Preprocessing: Log-Magnitude STFT (winlen={}, winstep={})'.format(
            model._winlen, model._winstep))
        logger.info('Input feature dimension: {} (STFT frequency bins)'.format(n_features))

    if args.checkpoint_path is not None:
        state_dict = torch.load(args.checkpoint_path)['network']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = 'network.' + k  # Add 'network.' prefix
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
        logger.info('Checkpoint loaded from {}'.format(args.checkpoint_path))
    else:
        # loging random init weight
        logger.info('Random init weight')

    # dataset & dataloader
    dataset, loader = get_dataset_libri_sampled_folder_subset(model, args)

    # run reconstruct
    reconstruct_dataset(model, device, loader, args)
    
    # Log completion
    logger.info('='*80)
    logger.info('Experiment completed successfully')
    logger.info('='*80)


def pretty_print_config(args):
    for key, value in args.__dict__.items():
        print("{:30} {}".format(key, value))
    
def parse_args():
    '''
    Example runs:
    
    For DeepSpeech2 (default):
        python src/main.py --model_name DeepSpeech2 --batch_start_idx 0 --batch_end_idx 1 --min_duration_ms 1000 --max_duration_ms 2000
    
    For DeepSpeech1:
        python src/main.py --model_name DeepSpeech1 --batch_start_idx 0 --batch_end_idx 1 --min_duration_ms 1000 --max_duration_ms 2000 --context_frames 6 --dropout_prob 0.0
    
    With checkpoint:
        python src/main.py --model_name ds2 --batch_start_idx 0 --batch_end_idx 1 --min_duration_ms 1000 --max_duration_ms 2000 --checkpoint_path /path/to/checkpoint.pt
        
    '''
    
    parser = argparse.ArgumentParser(description="Reconstruct a data point with specified parameters.")

    # env params
    parser.add_argument("--dataset_path"         , type=str  , default='./datasets/librispeech_sampled_600_file_0s_4s', help="Dataset path, symlinked from /scratch/f006pq6/datasets")
    parser.add_argument("--checkpoint_path"      , type=str  , default=None      , help="Path to checkpoint to resume from")
 
    # data params
    parser.add_argument("--min_duration_ms"      , type=int  , default=0         , help="Minimum duration of batch in milliseconds")
    parser.add_argument("--max_duration_ms"      , type=int  , default=1000      , help="Maximum duration of batch in milliseconds")
    parser.add_argument("--batch_start_idx"      , type=int  , required=True , help="Starting index of the batch"                       , default=0)
    parser.add_argument("--batch_end_idx"        , type=int  , required=True , help="Ending index of the batch"                         , default=1)
    parser.add_argument("--batch_size"           , type=int  , default=1         , help="Batch size")
 
    # model params
    parser.add_argument("--model_name"            , type=str  , default='DeepSpeech2'    , choices=['DeepSpeech1', 'ds1', 'DeepSpeech2', 'ds2'], help="Model architecture to use: 'DeepSpeech1'/'ds1' or 'DeepSpeech2'/'ds2'")
    parser.add_argument("--dropout_prob"         , type=float, default=0.0       , help="Dropout probability (for DeepSpeech1)")
    parser.add_argument("--context_frames"       , type=int  , default=6         , help="Number of context frames (for DeepSpeech1)")

    # optimization params
    parser.add_argument("--resume_from_first_order", action='store_true', help="Whether to resume from first order optimization checkpoint", default=False)
    parser.add_argument("--num_seeds"            , type=int  , default=10        , help="Number of random seeds to try")
    parser.add_argument("--optimizer"            , type=str  , default='Adam'    , help="Optimizer to use for optimization")
    parser.add_argument("--learning_rate"        , type=float, default=0.01      , help="Learning rate for first-order optimization")
    parser.add_argument("--regularization"       , type=str  , default='None'    , choices=["L1", "L2", "TV", "None"], help="Type of regularization")
    parser.add_argument("--reg_weight"           , type=float, default=0.0       , help="Weight of the regularization term")
    parser.add_argument("--max_iterations"       , type=int  , default=2000      , help="Maximum iterations for first-order optimization")
    parser.add_argument("--top_grad_percentage"  , type=float, default=1.0       , help="Percentage of top gradients to keep")
    parser.add_argument("--distance_metric"      , type=str  , default='cosine'  , help="Distance metric for gradient matching")
    parser.add_argument("--distance_metric_weight", type=float, default=0.5      , help="Weight for combining distance metrics (used in cosine+l2)")
    parser.add_argument("--initialization_method", type=str  , default='uniform' , help="Method for initializing input reconstruction")
    parser.add_argument("--patience"             , type=int  , default=1000       , help="Patience for early stopping based on loss improvement")


    parser.add_argument("--use_zero_order_optimization", action='store_true', help="Whether to use zero-order optimization after first-order optimization", default=False)
    parser.add_argument("--zero_order_lr"        , type=float, default=100       , help="Learning rate for zero-order optimization")
    parser.add_argument("--zero_max_iterations"  , type=int  , default=200       , help="Maximum iterations for zero-order optimization")

    # Grid optimization params
    parser.add_argument("--use_grid_optimization", action='store_true', help="Whether to use grid-based optimization instead of vanilla", default=False)
    parser.add_argument("--grid_size"            , type=int  , default=100       , help="Size of each grid segment (in frames)")
    parser.add_argument("--grid_overlap"         , type=int  , default=50        , help="Number of overlapping frames between adjacent grids")

    args = parser.parse_args()

    assert args.batch_size == 1, "Batch size must be 1"
    return args

def _format_duration_label(ms: int) -> str:
    """
    Convert millisecond boundaries into readable tokens for experiment folders.
    Defaults to seconds when possible (e.g., 3000 -> '3s') and otherwise keeps
    millisecond resolution (e.g., 1500 -> '1500ms').
    """
    if ms % 1000 == 0:
        return f"{ms // 1000}s"
    return f"{ms}ms"


def set_up_dir(args):
    duration_dir = f"{_format_duration_label(args.min_duration_ms)}-{_format_duration_label(args.max_duration_ms)}"
    exp_path = os.path.join('logging', duration_dir)
    
    # Normalize model name for experiment path
    model_short = 'DS1' if args.model_name.lower() in ['deepspeech1', 'ds1'] else 'DS2'

    cpt_name = os.path.basename(args.checkpoint_path) if args.checkpoint_path is not None else 'None'
    exp_name = f"{model_short}_batchstart_{args.batch_start_idx}_batch_end_{args.batch_end_idx}_init_{args.initialization_method}_opt_{args.optimizer}_lr_{args.learning_rate}_reg_{args.regularization}_regw_{args.reg_weight}_top-grad-perc_{args.top_grad_percentage}_cpt_{cpt_name}"
    args.exp_path=os.path.join(exp_path, exp_name)
    return args
 
 
if __name__ == "__main__":
    # args & dir
    args = parse_args()
    args = set_up_dir(args)
    pretty_print_config(args)
    # main
    main(args)
