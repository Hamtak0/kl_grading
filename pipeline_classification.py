import argparse
import os
import sys
import subprocess
import tomllib

def run_stage(cmd, stage_name):
    print(f"Running: {stage_name}")
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print(f"\n:x: Pipeline aborted during: {stage_name}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Master Execution Orchestrator")
    parser.add_argument("--config", type=str, default="configs/classification.toml", help="Path to baseline TOML configuration")
    parser.add_argument("--mode", type=str, choices=["LEFT", "RIGHT", "BOTH"], help="Override the lateral standardization mode")
    parser.add_argument("--trials", type=int, help="Override the baseline Optuna search budget")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f":x: Configuration file not found at: {args.config}")
        sys.exit(1)

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    lateral_mode = args.mode if args.mode else cfg["experiment"]["lateral_mode"]
    search_trials = args.trials if args.trials else cfg["optuna"]["n_trials"]
    base_epochs = cfg["training_defaults"]["num_epochs"]

    print(f"Target Configuration: {args.config}")
    print(f"Active Execution Mode: {lateral_mode} | Search Trials: {search_trials}")

    optuna_cmd = [sys.executable, "optuna_classification.py", "--mode", lateral_mode, "--trials", str(search_trials), "--epochs", str(base_epochs)]
    run_stage(optuna_cmd, f"Optuna Discovery ({lateral_mode})")

    from train_classification import train_classification
    timestamp, trained_mode = train_classification(
        config_path=f"./configs/best_params_{lateral_mode}.json",
        mode_override=lateral_mode
    )

    if not timestamp:
        print("\n:x: Training failed to yield a valid model timestamp. Aborting.")
        sys.exit(1)

    from test_classification import test_classification
    test_classification(timestamp_override=timestamp, mode_override=trained_mode)

    print(f"\n:trophy: Master Pipeline successfully completed for mode: {lateral_mode}!")

if __name__ == "__main__":
    main()