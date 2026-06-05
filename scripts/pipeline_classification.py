import argparse
import sys
from pathlib import Path

from core.classification.optuna_search import run_optuna_search
from core.classification.train import train_classification
from core.classification.test import test_classification
from utils.config import load_toml_config

def pipeline_classification():
    parser = argparse.ArgumentParser(description="Full classification pipeline model training and testing")
    parser.add_argument("--config", type=str, default="configs/config.toml", help="Path to baseline TOML configuration")
    parser.add_argument("--mode", type=str, choices=["LEFT", "RIGHT", "BOTH"], help="Override the lateral standardization mode")
    parser.add_argument("--trials", type=int, help="Override the baseline Optuna search budget")
    parser.add_argument("--target", type=str, choices=["KL", "OA"], help="Target of the model (KL grading or Osteoarthritis separation)")
    parser.add_argument("-l", "--loss", type=str, choices=["CE", "MSE"], help="Override loss function")
    parser.add_argument("-s", "--strategy", type=str, choices=["single_holdout", "kfold_blind", "kfold_nested"], help="Override Cross-validation architecture strategy")
    args = parser.parse_args()

    cfg = load_toml_config(Path(args.config))

    lateral_mode = args.mode.upper() if args.mode else cfg["experiment"]["lateral_mode"]
    search_trials = args.trials if args.trials else cfg["optuna"]["n_trials"]
    base_epochs = cfg["classification_defaults"]["num_epochs"]
    target = args.target.upper() if args.target else cfg["experiment"]["target"]
    loss_fn = args.loss.upper() if args.loss else cfg["experiment"]["loss_function"]
    strategy = args.strategy if args.strategy else cfg["experiment"]["strategy"]

    print(f"Target Configuration: {args.config}")
    print(f"Active Execution Strategy: {strategy} | Target: {target} | Mode: {lateral_mode} | Loss: {loss_fn} | Search Trials: {search_trials}")

    print(f"-- Phase 1: Optuna Hyperparameters searching")
    timestamp, best_params_path = run_optuna_search(mode=lateral_mode, trials=search_trials, epochs=base_epochs, target=target, loss_fn=loss_fn, strategy=strategy)

    print(f"-- Phase 2: Training classification ({target})")
    timestamp, trained_mode = train_classification(timestamp_override=timestamp, config_path=best_params_path, mode=lateral_mode, target=target, loss_fn=loss_fn, strategy=strategy)

    if not timestamp:
        print("Training failed to yield a valid model timestamp. Aborting.")
        sys.exit(1)

    print(f"-- Phase 3: Evaluation with test fold for each model ({target})")
    test_classification(timestamp_override=timestamp, mode=trained_mode, target=target, loss_fn=loss_fn, strategy=strategy)

    print(f"Pipeline successfully completed for {target} (Mode: {lateral_mode})!")

if __name__ == "__main__":
    pipeline_classification()