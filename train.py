import argparse
import yaml
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Starter trening: {cfg['env']}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

if __name__ == '__main__':
    main()
