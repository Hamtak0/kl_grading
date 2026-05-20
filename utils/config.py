import tomllib
import sys
import os
from pathlib import Path

def load_toml_config(config_path: Path = "configs/config.toml") -> dict:
    """
    Loads the TOML configuration file
    """
    if not os.path.exists(config_path):
        print(f"Critical Error: Configuration file not found at {config_path}")
        sys.exit(1)
    
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    
    return config