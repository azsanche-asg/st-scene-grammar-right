import argparse

from utils.config import load_config
from eval.runner import run_experiment as execute_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to config YAML")
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    execute_experiment(cfg, cfg['save']['out_dir'])


if __name__ == "__main__":
    main()
