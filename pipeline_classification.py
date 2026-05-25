import argparse
import os
import sys
import tomllib

from optuna_classification import run_optuna_search
from train_classification import train_classification
from test_classification import test_classification

def pipeline_classification():
    parser = argparse.ArgumentParser(description="Full classification pipeline model training and testing")
    parser.add_argument("--config", type=str, default="configs/config.toml", help="Path to baseline TOML configuration")
    parser.add_argument("--mode", type=str, choices=["LEFT", "RIGHT", "BOTH"], help="Override the lateral standardization mode")
    parser.add_argument("--trials", type=int, help="Override the baseline Optuna search budget")
    parser.add_argument("--target", type=str, default="KL", choices=["KL", "OA"], help="Target of the model (KL grading or Osteoarthritis separation)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Configuration file not found at: {args.config}")
        sys.exit(1)

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    lateral_mode = args.mode.upper() if args.mode else cfg["experiment"]["lateral_mode"]
    search_trials = args.trials if args.trials else cfg["optuna"]["n_trials"]
    base_epochs = cfg["training_defaults"]["num_epochs"]
    target = args.target.upper() if args.target else cfg["experiment"]["target"]

    print(f"Target Configuration: {args.config}")
    print(f"Active Execution Mode: {lateral_mode} | Search Trials: {search_trials} | Target: {target}")

    print(f"-- Phase 1: Optuna Hyperparameters searching")
    best_params_path = run_optuna_search(mode=lateral_mode, trials=search_trials, epochs=base_epochs, target=target)

    print(f"-- Phase 2: 5-fold training ({target})")
    timestamp, trained_mode = train_classification(config_path=best_params_path, mode_override=lateral_mode, target=target)

    if not timestamp:
        print("Training failed to yield a valid model timestamp. Aborting.")
        sys.exit(1)

    print(f"-- Phase 3: Evaluation with test fold for each model ({target})")
    test_classification(timestamp_override=timestamp, mode_override=trained_mode, target=target)

    print(f"Pipeline successfully completed for {target} (Mode: {lateral_mode})!")

if __name__ == "__main__":
    pipeline_classification()