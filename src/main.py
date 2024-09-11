__author__ = "alvrogd"


import argparse
import math
import os
import random
import re
import statistics
import sys
import time

import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.utils.data
import tqdm

import datasets.datasets as m_datasets
import models.rocket as m_rocket
import models.templates as m_templates
import utils.constants as m_constants


def initialize_parser() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser()

    # Execution
    parser.add_argument("--device", type=str, action="store", default=f"cuda:{torch.cuda.current_device()}"
                                                                      if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",   type=int, action="store", default=42)
    
    # Data
    parser.add_argument("--batch_size",      type=int, action="store", default=32)
    parser.add_argument("--split_procedure", type=str, action="store", default="split_by_plot", choices=[*m_datasets.SPLIT_PROCEDURES])
    parser.add_argument("--study_var",       type=str, action="store", default="SpecRichness",  choices=m_constants.STUDY_VARS)
    
    # Model
    parser.add_argument("--beta1",          type=float, action="store", default=0.9)
    parser.add_argument("--beta2",          type=float, action="store", default=0.999)
    parser.add_argument("--ensemble_count", type=int,   action="store", default=1)
    parser.add_argument("--epochs",         type=int,   action="store", default=1500)
    parser.add_argument("--eps",            type=float, action="store", default=1e-6)
    parser.add_argument("--lr",             type=float, action="store", default=0.001)
    parser.add_argument("--model",          type=str,   action="store", default="ResNet18T", choices=[*m_constants.MODELS])
    parser.add_argument("--warmup_epochs",  type=int,   action="store", default=150)
    parser.add_argument("--weight_decay",   type=float, action="store", default=0.001)
    
    # InceptionTime-specific arguments
    parser.add_argument("--bottleneck_factor", type=int,  action="store", default=4)
    parser.add_argument("--num_filters",       type=int,  action="store", default=64)
    
    # ResNet18-specific arguments
    parser.add_argument("--num_blocks_per_stage", type=int, nargs=4, action="store", default=[1, 1, 1, 1])
    parser.add_argument("--num_channels",         type=int, nargs=4, action="store", default=[64, 64, 64, 64])
    parser.add_argument("--kernel_size",          type=int,          action="store", default=5)
    parser.add_argument("--shortcut_pooling",     type=bool,         action="store", default=True)
    parser.add_argument("--stem_channels",        type=int,          action="store", default=96)
    parser.add_argument("--strides",              type=int, nargs=4, action="store", default=[1, 1, 2, 1])
    
    # InceptionTime-specific & ResNet18-specific arguments
    parser.add_argument("--original_training", type=bool, action="store", default=False)
    
    # Rocket-specific arguments
    parser.add_argument("--num_kernels", type=int, action="store", default=15000)

    return parser


def initialize_cuda() -> None:
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    
def initialize_seeds(seed: int) -> None:
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    # 0. Initialization

    parser = initialize_parser()
    args   = parser.parse_args()
    print(f"[*] Arguments: {vars(args)}")

    if args.device != "cpu":
        initialize_cuda()

    # For the sake of reproducibility
    initialize_seeds(args.seed)
    
    if not os.path.exists("logs"):
        os.mkdir("logs")


    # 1. Load data

    dataset         = m_datasets.SeBASDataset(study_var=args.study_var)
    split_procedure = m_datasets.SPLIT_PROCEDURES[args.split_procedure]
    
    folds = split_procedure(dataset, args.seed)


    # 2. Train & test model
    #    (using k-fold cross-validation)
    
    model_class = m_constants.MODELS[args.model]
    dataset.set_time_series(model_class.takes_time_series())
    
    fold_results = []
    time_results = []

    for fold, (train_ids, val_ids, test_ids) in enumerate(folds):

        print(f"[*] Fold {fold + 1}/{len(folds)}")

        # Normalize predictors inside each subset
        dataset.normalize_predictors(train_ids, val_ids, test_ids)

        # Create data loaders
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler   = torch.utils.data.SubsetRandomSampler(val_ids)
        test_subsampler  = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler,
                                                   pin_memory=args.device != "cpu")
        val_loader   = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=val_subsampler,
                                                   pin_memory=args.device != "cpu")
        test_loader  = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=test_subsampler,
                                                   pin_memory=args.device != "cpu")
        
        # Create model
        if model_class == m_rocket.Rocket:
            model = m_rocket.Rocket(vars(args))
        else:
            model = m_templates.EnsembleDeepRegressionModel(vars(args))
        model_summary = str(model)

        # Workarounds to properly log model size with GuildAI
        total_params_line           = re.search(r'Total params:.*', model_summary).group()
        total_params_line_no_commas = total_params_line.replace(',', '')
        model_summary               = model_summary.replace(total_params_line, total_params_line_no_commas)

        print(model_summary)        

        # Train model
        last_val_loss                = math.inf
        best_val_loss                = math.inf
        val_steps_no_improvement     = 0
        max_val_steps_no_improvement = args.epochs // (10 * 10) # 10% total epochs, validation every 10 epochs
        
        t_train_start = time.perf_counter()
        
        for epoch in tqdm.tqdm(range(args.warmup_epochs), desc="[*] Warming up the model...", file=sys.stdout):

            epoch_loss = model.run_step(train_loader, "train")
                
            # Show progress every 10 epochs         
            if (epoch + 1) % 10 == 0 or epoch == 0:
            
                tqdm.tqdm.write(
                    f"[*] Warmup epoch: {epoch + 1}/{args.warmup_epochs} - Train loss: {epoch_loss:.4f}"
                )
                
            model.lr_scheduler_step(-1, True)       
                    
        for epoch in tqdm.tqdm(range(args.epochs), desc="[*] Training the model...", file=sys.stdout):

            epoch_loss = model.run_step(train_loader, "train")
            
            # Progress, validation and checkpointing every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                
                last_val_loss = model.run_step(val_loader, "val")
                
                if last_val_loss < best_val_loss:
                    model.save_model(f"logs/{model.id}_best-model_{args.study_var}_fold-{fold + 1}")
                    best_val_loss = last_val_loss
                    val_steps_no_improvement = 0
                    
                else:
                    val_steps_no_improvement += 1
                    
                    if val_steps_no_improvement >= max_val_steps_no_improvement:
                        tqdm.tqdm.write(f"[*] Early stopping at epoch {epoch + 1}")
                        break
            
                tqdm.tqdm.write(
                    f"[*] Epoch: {epoch + 1}/{args.epochs} - Train loss: {epoch_loss:.4f} - Val loss: {last_val_loss:.4f}"
                )
                
            model.lr_scheduler_step(last_val_loss, False)
                    
        t_train_end = time.perf_counter()
        print(f"[*] Training time: {t_train_end - t_train_start:.2f} s")

        # Test model
        print(f"[*] Testing the best model...")
        model.load_model(f"logs/{model.id}_best-model_{args.study_var}_fold-{fold + 1}")
        
        t_test_start = time.perf_counter()
        fold_results.append(model.run_step(test_loader, "test"))
        t_test_end = time.perf_counter()
        print(f"[*] Testing time: {t_test_end - t_test_start:.2f} s")
        
        time_results.append([t_train_end - t_train_start, t_test_end - t_test_start])


    # 3. Final metrics

    r2_hat = statistics.mean(x[0] for x in fold_results)
    r2_sd  = statistics.pstdev(x[0] for x in fold_results)

    rrmse_hat = statistics.mean(x[1] for x in fold_results)
    rrmse_sd  = statistics.pstdev(x[1] for x in fold_results)

    rmses_hat = statistics.mean(x[2] for x in fold_results)
    rmses_sd  = statistics.pstdev(x[2] for x in fold_results)

    rmseu_hat = statistics.mean(x[3] for x in fold_results)
    rmseu_sd  = statistics.pstdev(x[3] for x in fold_results)
    
    time_train_mean = statistics.mean(x[0] for x in time_results)
    time_train_sd   = statistics.pstdev(x[0] for x in time_results)
    
    time_test_mean = statistics.mean(x[1] for x in time_results)
    time_test_sd   = statistics.pstdev(x[1] for x in time_results)

    print(f"[*] Final metrics:")
    print(f"[*] R2 mean: {r2_hat:.4f}")
    print(f"[*] R2 std: {r2_sd:.4f}")
    print(f"[*] RRMSE mean: {rrmse_hat:.4f}")
    print(f"[*] RRMSE std: {rrmse_sd:.4f}")
    print(f"[*] RMSES mean: {rmses_hat:.4f}")
    print(f"[*] RMSES std: {rmses_sd:.4f}")
    print(f"[*] RMSEU mean: {rmseu_hat:.4f}")
    print(f"[*] RMSEU std: {rmseu_sd:.4f}")
    print(f"[*] Training time mean: {time_train_mean:.2f} s")
    print(f"[*] Training time std: {time_train_sd:.2f} s")
    print(f"[*] Testing time mean: {time_test_mean:.2f} s")
    print(f"[*] Testing time std: {time_test_sd:.2f} s")
